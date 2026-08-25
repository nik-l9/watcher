"""FalkorDB implementation of GraphStore — one named graph per tenant.

Isolation model: `ctx.graph_name` selects a distinct FalkorDB graph. A query issued
against tenant A's graph cannot observe tenant B's data, because the two are
separate graph keys rather than separate subsets of one graph. There is no
tenant_id predicate to forget.

Every query in this module is parameterized. Labels and relationship types cannot
be parameterized in Cypher, so they are interpolated — but only ever from the
closed NodeLabel/RelType enums, never from caller input. `_label()` and `_rel()`
enforce that.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from falkordb.asyncio import FalkorDB

from cortex.config.settings import get_settings
from cortex.memory.entities import Edge, Node, NodeLabel, RelType
from cortex.memory.graph_store import GraphStore
from cortex.tenancy.context import TenantContext

# Labels that get a range index on `key` at provision time. Every label does,
# since (label, key) is the upsert path for all of them.
_INDEXED_LABELS = tuple(NodeLabel)

_MAX_DEPTH = 6


def _label(label: NodeLabel) -> str:
    """Render a node label for interpolation, proving it came from the enum."""
    if not isinstance(label, NodeLabel):
        raise TypeError(f"expected NodeLabel, got {type(label).__name__}")
    return label.value


def _rel(rel: RelType) -> str:
    if not isinstance(rel, RelType):
        raise TypeError(f"expected RelType, got {type(rel).__name__}")
    return rel.value


def _node_to_dict(node: Any) -> dict:
    """Flatten a FalkorDB node into a plain dict of label + properties."""
    labels = getattr(node, "labels", None) or []
    props = dict(getattr(node, "properties", {}) or {})
    return {"_label": labels[0] if labels else None, **props}


class FalkorDBGraphStore(GraphStore):
    def __init__(self, client: FalkorDB | None = None) -> None:
        self._client = client
        self._lock = asyncio.Lock()

    async def _get_client(self) -> FalkorDB:
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    s = get_settings()
                    self._client = FalkorDB(
                        host=s.falkordb_host,
                        port=s.falkordb_port,
                        password=s.falkordb_password or None,
                    )
        return self._client

    async def _graph(self, ctx: TenantContext):
        """Select the tenant's graph.

        This is the only place a graph is selected, and it reads the name from the
        context rather than accepting one from a caller.
        """
        client = await self._get_client()
        return client.select_graph(ctx.graph_name)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def ping(self) -> bool:
        """Readiness check. Exposed so callers need no access to the raw client."""
        client = await self._get_client()
        await client.list_graphs()
        return True

    # ------------------------------------------------------------ lifecycle

    async def provision(self, ctx: TenantContext) -> None:
        g = await self._graph(ctx)
        # A graph is created lazily on first write, so touch it to materialize.
        await g.query("RETURN 1")
        for label in _INDEXED_LABELS:
            try:
                await g.create_node_range_index(_label(label), "key")
            except Exception as exc:  # noqa: BLE001
                # Index already exists — provision must be idempotent.
                if "already indexed" not in str(exc).lower():
                    raise

    async def drop(self, ctx: TenantContext) -> None:
        g = await self._graph(ctx)
        try:
            await g.delete()
        except Exception as exc:  # noqa: BLE001
            # Offboarding must be retryable. FalkorDB reports an already-absent
            # graph as "Invalid graph operation on empty key" rather than a
            # not-found error, so both phrasings count as success.
            message = str(exc).lower()
            if not any(s in message for s in ("not exist", "empty key")):
                raise

    # ------------------------------------------------------------ writes

    async def upsert_nodes(self, ctx: TenantContext, nodes: Sequence[Node]) -> int:
        if not nodes:
            return 0
        g = await self._graph(ctx)
        written = 0
        # Grouped by label because the label cannot be parameterized; one UNWIND
        # per label keeps this to a handful of round trips instead of one per node.
        by_label: dict[NodeLabel, list[dict]] = {}
        for n in nodes:
            by_label.setdefault(n.label, []).append({"key": n.key, "props": n.props})

        for label, rows in by_label.items():
            q = (
                f"UNWIND $rows AS row "
                f"MERGE (n:{_label(label)} {{key: row.key}}) "
                f"SET n += row.props "
                f"RETURN count(n) AS c"
            )
            res = await g.query(q, {"rows": rows})
            written += res.result_set[0][0] if res.result_set else 0
        return written

    async def upsert_edges(self, ctx: TenantContext, edges: Sequence[Edge]) -> int:
        if not edges:
            return 0
        g = await self._graph(ctx)
        written = 0
        # Grouped by the (from_label, rel, to_label) triple for the same reason.
        by_shape: dict[tuple[NodeLabel, RelType, NodeLabel], list[dict]] = {}
        for e in edges:
            shape = (e.from_label, e.rel, e.to_label)
            by_shape.setdefault(shape, []).append(
                {"from": e.from_key, "to": e.to_key, "props": e.props}
            )

        for (from_label, rel, to_label), rows in by_shape.items():
            q = (
                f"UNWIND $rows AS row "
                f"MATCH (a:{_label(from_label)} {{key: row.from}}) "
                f"MATCH (b:{_label(to_label)} {{key: row.to}}) "
                f"MERGE (a)-[r:{_rel(rel)}]->(b) "
                f"SET r += row.props "
                f"RETURN count(r) AS c"
            )
            res = await g.query(q, {"rows": rows})
            written += res.result_set[0][0] if res.result_set else 0
        return written

    # ------------------------------------------------------------ reads

    async def get_node(self, ctx: TenantContext, label: NodeLabel, key: str) -> dict | None:
        g = await self._graph(ctx)
        res = await g.ro_query(
            f"MATCH (n:{_label(label)} {{key: $key}}) RETURN n LIMIT 1", {"key": key}
        )
        if not res.result_set:
            return None
        return _node_to_dict(res.result_set[0][0])

    async def neighbors(
        self,
        ctx: TenantContext,
        label: NodeLabel,
        key: str,
        *,
        rel_types: Sequence[RelType] | None = None,
        depth: int = 1,
        limit: int = 100,
    ) -> list[dict]:
        depth = max(1, min(int(depth), _MAX_DEPTH))
        limit = max(1, min(int(limit), 1000))
        rel_filter = "|".join(_rel(r) for r in rel_types) if rel_types else ""
        rel_pattern = f"[r:{rel_filter}*1..{depth}]" if rel_filter else f"[r*1..{depth}]"

        g = await self._graph(ctx)
        res = await g.ro_query(
            f"MATCH (n:{_label(label)} {{key: $key}})-{rel_pattern}-(m) "
            f"RETURN DISTINCT m LIMIT {limit}",
            {"key": key},
        )
        return [_node_to_dict(row[0]) for row in res.result_set]

    async def find_paths(
        self,
        ctx: TenantContext,
        from_label: NodeLabel,
        from_key: str,
        to_label: NodeLabel,
        to_key: str,
        *,
        max_depth: int = 5,
        limit: int = 10,
    ) -> list[list[dict]]:
        max_depth = max(1, min(int(max_depth), _MAX_DEPTH))
        limit = max(1, min(int(limit), 100))

        g = await self._graph(ctx)
        res = await g.ro_query(
            f"MATCH p = (a:{_label(from_label)} {{key: $from_key}})"
            f"-[*1..{max_depth}]-"
            f"(b:{_label(to_label)} {{key: $to_key}}) "
            f"RETURN nodes(p) AS ns LIMIT {limit}",
            {"from_key": from_key, "to_key": to_key},
        )
        return [[_node_to_dict(n) for n in row[0]] for row in res.result_set]

    async def nodes_by_label(
        self,
        ctx: TenantContext,
        label: NodeLabel,
        *,
        where: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[dict]:
        limit = max(1, min(int(limit), 1000))
        params: dict[str, Any] = {}
        clauses = []
        for i, (prop, value) in enumerate(sorted((where or {}).items())):
            # Property names cannot be parameterized either, so validate rather
            # than trust: only identifier-safe names reach the query text.
            if not prop.isidentifier():
                raise ValueError(f"invalid property name: {prop!r}")
            pname = f"p{i}"
            clauses.append(f"n.{prop} = ${pname}")
            params[pname] = value

        where_sql = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        g = await self._graph(ctx)
        res = await g.ro_query(
            f"MATCH (n:{_label(label)}) {where_sql}RETURN n LIMIT {limit}", params
        )
        return [_node_to_dict(row[0]) for row in res.result_set]

    async def stats(self, ctx: TenantContext) -> dict[str, int]:
        g = await self._graph(ctx)
        nodes = await g.ro_query("MATCH (n) RETURN count(n) AS c")
        rels = await g.ro_query("MATCH ()-[r]->() RETURN count(r) AS c")
        return {
            "nodes": nodes.result_set[0][0] if nodes.result_set else 0,
            "relationships": rels.result_set[0][0] if rels.result_set else 0,
        }
