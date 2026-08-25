"""Cross-tenant isolation for the knowledge graph.

Per the plan, this suite must pass before any real credential is entered. Its
job is to demonstrate that isolation comes from opening a different graph, not
from a predicate someone remembered to write.
"""

from __future__ import annotations

import pytest

from cortex.memory.entities import Edge, Node, NodeLabel, RelType
from cortex.memory.falkordb_store import FalkorDBGraphStore
from cortex.memory.graph_store import GraphStore
from cortex.memory.naming import GRAPH_PREFIX, InvalidTenantSlug, graph_name_for_new_tenant
from cortex.tenancy.context import TenantContext


async def _seed(graph: GraphStore, ctx: TenantContext, marker: str) -> None:
    await graph.upsert_nodes(
        ctx,
        [
            Node(NodeLabel.DEPLOY, f"deploy-{marker}", {"sha": marker, "owner": marker}),
            Node(NodeLabel.METRIC, f"metric-{marker}", {"name": "signups", "owner": marker}),
        ],
    )
    await graph.upsert_edges(
        ctx,
        [
            Edge(
                RelType.AFFECTS,
                NodeLabel.DEPLOY,
                f"deploy-{marker}",
                NodeLabel.METRIC,
                f"metric-{marker}",
                {"owner": marker},
            )
        ],
    )


async def test_distinct_graph_names(tenant_a: TenantContext, tenant_b: TenantContext) -> None:
    assert tenant_a.graph_name != tenant_b.graph_name
    assert tenant_a.graph_name.startswith(GRAPH_PREFIX)


async def test_get_node_does_not_cross_tenants(
    graph: FalkorDBGraphStore, tenant_a: TenantContext, tenant_b: TenantContext
) -> None:
    await _seed(graph, tenant_a, "aaa")
    await _seed(graph, tenant_b, "bbb")

    assert await graph.get_node(tenant_a, NodeLabel.DEPLOY, "deploy-aaa") is not None
    # Tenant A's key must be invisible from tenant B's graph and vice versa.
    assert await graph.get_node(tenant_b, NodeLabel.DEPLOY, "deploy-aaa") is None
    assert await graph.get_node(tenant_a, NodeLabel.DEPLOY, "deploy-bbb") is None


async def test_label_scan_returns_only_own_rows(
    graph: FalkorDBGraphStore, tenant_a: TenantContext, tenant_b: TenantContext
) -> None:
    await _seed(graph, tenant_a, "aaa")
    await _seed(graph, tenant_b, "bbb")

    rows = await graph.nodes_by_label(tenant_a, NodeLabel.DEPLOY, limit=1000)
    assert rows, "tenant A should see its own deploy"
    assert all(r.get("owner") == "aaa" for r in rows), rows


async def test_traversal_returns_only_own_rows(
    graph: FalkorDBGraphStore, tenant_a: TenantContext, tenant_b: TenantContext
) -> None:
    await _seed(graph, tenant_a, "aaa")
    await _seed(graph, tenant_b, "bbb")

    hops = await graph.neighbors(tenant_a, NodeLabel.DEPLOY, "deploy-aaa", depth=2)
    assert hops
    assert all(h.get("owner") == "aaa" for h in hops), hops


async def test_paths_do_not_span_tenants(
    graph: FalkorDBGraphStore, tenant_a: TenantContext, tenant_b: TenantContext
) -> None:
    await _seed(graph, tenant_a, "aaa")
    await _seed(graph, tenant_b, "bbb")

    # A path from tenant A's deploy to tenant B's metric must not exist.
    crossing = await graph.find_paths(
        tenant_a, NodeLabel.DEPLOY, "deploy-aaa", NodeLabel.METRIC, "metric-bbb"
    )
    assert crossing == []

    own = await graph.find_paths(
        tenant_a, NodeLabel.DEPLOY, "deploy-aaa", NodeLabel.METRIC, "metric-aaa"
    )
    assert own, "tenant A should still find its own path"


async def test_stats_are_per_tenant(
    graph: FalkorDBGraphStore, tenant_a: TenantContext, tenant_b: TenantContext
) -> None:
    await _seed(graph, tenant_a, "aaa")
    assert (await graph.stats(tenant_a))["nodes"] == 2
    assert (await graph.stats(tenant_b))["nodes"] == 0


async def test_graphstore_interface_exposes_no_raw_graph_access() -> None:
    """The interface must offer no way to name a graph or run arbitrary Cypher.

    This is the structural half of the guarantee: even a caller that wanted to
    reach another tenant has no method to do it.
    """
    public = {m for m in dir(GraphStore) if not m.startswith("_")}
    for forbidden in ("query", "ro_query", "cypher", "execute", "raw", "select_graph"):
        assert forbidden not in public, f"GraphStore must not expose {forbidden}()"

    import inspect

    for name in public:
        member = getattr(GraphStore, name)
        if not callable(member):
            continue
        params = inspect.signature(member).parameters
        assert "ctx" in params, f"GraphStore.{name}() must take a TenantContext"
        assert "graph_name" not in params, f"GraphStore.{name}() must not accept a graph name"


async def test_upsert_rejects_labels_outside_the_enum(
    graph: FalkorDBGraphStore, tenant_a: TenantContext
) -> None:
    """A string label would be Cypher injection; the enum check must reject it."""
    with pytest.raises(TypeError):
        await graph.upsert_nodes(
            tenant_a,
            [Node("Deploy) MATCH (x) DETACH DELETE x //", "k")],  # type: ignore[arg-type]
        )


async def test_property_filter_rejects_non_identifier_names(
    graph: FalkorDBGraphStore, tenant_a: TenantContext
) -> None:
    with pytest.raises(ValueError, match="invalid property name"):
        await graph.nodes_by_label(tenant_a, NodeLabel.DEPLOY, where={"owner} RETURN n //": "x"})


async def test_slug_validation_blocks_namespace_escape() -> None:
    import uuid

    for bad in ("../etc", "Tenant", "a", "a" * 80, "-lead", "trail-", "a b", "a:b", "a*"):
        with pytest.raises(InvalidTenantSlug):
            graph_name_for_new_tenant(bad, uuid.uuid4())


async def test_recycled_slug_gets_a_fresh_graph_name() -> None:
    """A new tenant must never inherit a deleted tenant's graph."""
    import uuid

    slug = "acme-corp"
    first = graph_name_for_new_tenant(slug, uuid.uuid4())
    second = graph_name_for_new_tenant(slug, uuid.uuid4())
    assert first != second


class TestStoreEdgePaths:
    """Branches that only fire on an empty write, a property filter, or an
    unexpected engine error. Each is a real path — the first two on ordinary use."""

    async def test_empty_upserts_short_circuit(
        self, graph: FalkorDBGraphStore, tenant_a: TenantContext
    ) -> None:
        """A step that gathered nothing must not issue a query. Without the guard an
        empty UNWIND runs per label on every barren sync."""
        assert await graph.upsert_nodes(tenant_a, []) == 0
        assert await graph.upsert_edges(tenant_a, []) == 0
        assert (await graph.stats(tenant_a))["nodes"] == 0

    async def test_property_filter_is_parameterised(
        self, graph: FalkorDBGraphStore, tenant_a: TenantContext
    ) -> None:
        """The `where` clause path: property names are validated and interpolated,
        values are bound. Both halves run here."""
        await graph.upsert_nodes(
            tenant_a,
            [
                Node(NodeLabel.DEPLOY, "d-prod", {"environment": "production", "sha": "aaa"}),
                Node(NodeLabel.DEPLOY, "d-stage", {"environment": "staging", "sha": "bbb"}),
            ],
        )
        rows = await graph.nodes_by_label(
            tenant_a, NodeLabel.DEPLOY, where={"environment": "production"}
        )
        assert [r["sha"] for r in rows] == ["aaa"]

    async def test_multiple_filter_properties_are_all_applied(
        self, graph: FalkorDBGraphStore, tenant_a: TenantContext
    ) -> None:
        await graph.upsert_nodes(
            tenant_a,
            [
                Node(NodeLabel.DEPLOY, "d1", {"environment": "production", "state": "success"}),
                Node(NodeLabel.DEPLOY, "d2", {"environment": "production", "state": "failure"}),
            ],
        )
        rows = await graph.nodes_by_label(
            tenant_a,
            NodeLabel.DEPLOY,
            where={"environment": "production", "state": "failure"},
        )
        assert [r["key"] for r in rows] == ["d2"]

    async def test_a_hostile_value_stays_a_value(
        self, graph: FalkorDBGraphStore, tenant_a: TenantContext
    ) -> None:
        """Values are bound, so Cypher in a filter value cannot become syntax."""
        hostile = "production'}) DETACH DELETE n //"
        await graph.upsert_nodes(tenant_a, [Node(NodeLabel.DEPLOY, "d1", {"env": hostile})])
        rows = await graph.nodes_by_label(tenant_a, NodeLabel.DEPLOY, where={"env": hostile})
        assert [r["key"] for r in rows] == ["d1"]
        # Nothing was deleted, so the injection did not execute.
        assert (await graph.stats(tenant_a))["nodes"] == 1

    async def test_an_unexpected_provision_error_is_not_swallowed(
        self, graph: FalkorDBGraphStore, tenant_a: TenantContext
    ) -> None:
        """Only "already indexed" is tolerated. Swallowing everything would hide a
        real engine failure behind a silently unprovisioned tenant."""
        original = graph._graph

        class _Boom:
            async def query(self, *_a: object, **_k: object) -> None:
                return None

            async def create_node_range_index(self, *_a: object) -> None:
                # Not the tolerated "already indexed" message, so it must propagate.
                raise RuntimeError("engine exploded")

        async def _broken(_ctx: TenantContext) -> object:
            return _Boom()

        graph._graph = _broken  # type: ignore[method-assign]
        try:
            with pytest.raises(RuntimeError, match="engine exploded"):
                await graph.provision(tenant_a)
        finally:
            graph._graph = original  # type: ignore[method-assign]

    async def test_an_unexpected_drop_error_is_not_swallowed(
        self, graph: FalkorDBGraphStore, tenant_a: TenantContext
    ) -> None:
        original = graph._graph

        class _Boom:
            async def delete(self) -> None:
                raise RuntimeError("engine exploded")

        async def _broken(_ctx: TenantContext) -> object:
            return _Boom()

        graph._graph = _broken  # type: ignore[method-assign]
        try:
            with pytest.raises(RuntimeError, match="engine exploded"):
                await graph.drop(tenant_a)
        finally:
            graph._graph = original  # type: ignore[method-assign]
