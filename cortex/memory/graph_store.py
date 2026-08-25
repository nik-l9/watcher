"""GraphStore — the tenant-isolated knowledge graph interface.

Contract: **graph per tenant**. Every method takes a TenantContext and the
implementation resolves it to one physical graph. There is deliberately no method
that accepts a raw graph name, no method that queries across tenants, and no
method that accepts caller-authored Cypher. Those omissions are the isolation
guarantee.

FalkorDB is the V1 implementation because it supports many isolated graphs per
instance with no license gate. A Neo4j Enterprise implementation (one database per
tenant) is a drop-in replacement behind this same interface — openCypher is shared
between them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from cortex.memory.entities import Edge, Node, NodeLabel, RelType
from cortex.tenancy.context import TenantContext


class GraphStore(ABC):
    @abstractmethod
    async def provision(self, ctx: TenantContext) -> None:
        """Create the tenant's graph and its indexes. Idempotent."""

    @abstractmethod
    async def drop(self, ctx: TenantContext) -> None:
        """Delete the tenant's graph entirely. Used for offboarding and test teardown."""

    @abstractmethod
    async def upsert_nodes(self, ctx: TenantContext, nodes: Sequence[Node]) -> int:
        """Idempotent upsert keyed on (label, key). Returns nodes written."""

    @abstractmethod
    async def upsert_edges(self, ctx: TenantContext, edges: Sequence[Edge]) -> int:
        """Idempotent upsert. Endpoints must already exist. Returns edges written."""

    @abstractmethod
    async def get_node(self, ctx: TenantContext, label: NodeLabel, key: str) -> dict | None: ...

    @abstractmethod
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
        """Expand outward from a node. Depth is bounded to keep traversal costs sane."""

    @abstractmethod
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
        """Paths between two nodes — the Campaign→Feature→PR→Ticket→Customer→Revenue walk."""

    @abstractmethod
    async def nodes_by_label(
        self,
        ctx: TenantContext,
        label: NodeLabel,
        *,
        where: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Filtered scan. `where` is exact-match on properties, parameterized."""

    @abstractmethod
    async def stats(self, ctx: TenantContext) -> dict[str, int]:
        """Node and relationship counts by type. Used for connector health and tests."""
