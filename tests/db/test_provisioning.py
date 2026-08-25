"""Tenant onboarding and offboarding.

Provisioning is the only moment a physical graph name is assigned, so it is also
the only place these guarantees can be established: that the graph and the row
are created together, and that offboarding removes both.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import Tenant
from cortex.memory.falkordb_store import FalkorDBGraphStore
from cortex.memory.graph_store import GraphStore
from cortex.memory.naming import GRAPH_PREFIX, InvalidTenantSlug
from cortex.tenancy.context import TenantContext
from cortex.tenancy.provisioning import TenantAlreadyExists, create_tenant, delete_tenant


class _RecordingGraph(GraphStore):
    """Records calls so provisioning order can be asserted without a live graph."""

    def __init__(self, fail_on_provision: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail_on_provision = fail_on_provision

    async def provision(self, ctx: TenantContext) -> None:
        self.calls.append(("provision", ctx.graph_name))
        if self.fail_on_provision:
            raise RuntimeError("graph backend unavailable")

    async def drop(self, ctx: TenantContext) -> None:
        self.calls.append(("drop", ctx.graph_name))

    async def upsert_nodes(self, ctx, nodes):  # type: ignore[no-untyped-def]
        return 0

    async def upsert_edges(self, ctx, edges):  # type: ignore[no-untyped-def]
        return 0

    async def get_node(self, ctx, label, key):  # type: ignore[no-untyped-def]
        return None

    async def neighbors(self, ctx, label, key, **kw):  # type: ignore[no-untyped-def]
        return []

    async def find_paths(self, ctx, from_label, from_key, to_label, to_key, **kw):  # type: ignore[no-untyped-def]
        return []

    async def nodes_by_label(self, ctx, label, **kw):  # type: ignore[no-untyped-def]
        return []

    async def stats(self, ctx):  # type: ignore[no-untyped-def]
        return {"nodes": 0, "relationships": 0}


class TestCreateTenant:
    async def test_creates_row_and_graph(self, session: AsyncSession) -> None:
        graph = _RecordingGraph()
        ctx = await create_tenant(session, graph, slug="acme-corp", name="Acme Corp")

        assert ctx.tenant_slug == "acme-corp"
        assert ctx.graph_name.startswith(f"{GRAPH_PREFIX}acme-corp_")
        assert ctx.role == "admin"
        assert graph.calls == [("provision", ctx.graph_name)]

        row = (await session.execute(select(Tenant).where(Tenant.slug == "acme-corp"))).scalar_one()
        assert row.graph_name == ctx.graph_name

    async def test_row_exists_before_graph_is_provisioned(self, session: AsyncSession) -> None:
        """Provisioning after the flush means a failed provision cannot orphan a graph."""
        seen: list[bool] = []

        class _Checking(_RecordingGraph):
            async def provision(self, ctx: TenantContext) -> None:
                found = (
                    await session.execute(select(Tenant.id).where(Tenant.id == ctx.tenant_id))
                ).scalar_one_or_none()
                seen.append(found is not None)

        await create_tenant(session, _Checking(), slug="acme-corp", name="Acme")
        assert seen == [True]

    async def test_rejects_duplicate_slug(self, session: AsyncSession) -> None:
        graph = _RecordingGraph()
        await create_tenant(session, graph, slug="acme-corp", name="Acme")
        with pytest.raises(TenantAlreadyExists):
            await create_tenant(session, graph, slug="acme-corp", name="Acme Again")

    async def test_duplicate_does_not_touch_the_graph(self, session: AsyncSession) -> None:
        graph = _RecordingGraph()
        await create_tenant(session, graph, slug="acme-corp", name="Acme")
        graph.calls.clear()
        with pytest.raises(TenantAlreadyExists):
            await create_tenant(session, graph, slug="acme-corp", name="Acme Again")
        assert graph.calls == [], "a rejected creation must not provision anything"

    async def test_rejects_invalid_slug_before_writing(self, session: AsyncSession) -> None:
        graph = _RecordingGraph()
        with pytest.raises(InvalidTenantSlug):
            await create_tenant(session, graph, slug="../escape", name="Bad")
        assert graph.calls == []
        count = (await session.execute(select(Tenant))).scalars().all()
        assert count == []

    async def test_provision_failure_propagates(self, session: AsyncSession) -> None:
        """The caller's transaction must be able to roll the row back with it."""
        graph = _RecordingGraph(fail_on_provision=True)
        with pytest.raises(RuntimeError, match="graph backend unavailable"):
            await create_tenant(session, graph, slug="acme-corp", name="Acme")

    async def test_two_tenants_get_distinct_graphs(self, session: AsyncSession) -> None:
        graph = _RecordingGraph()
        a = await create_tenant(session, graph, slug="tenant-one", name="One")
        b = await create_tenant(session, graph, slug="tenant-two", name="Two")
        assert a.graph_name != b.graph_name
        assert a.tenant_id != b.tenant_id


class TestDeleteTenant:
    async def test_drops_graph_before_row(self, session: AsyncSession) -> None:
        """Graph first: if the row vanished first, its graph name would be
        unrecoverable and the data orphaned rather than deleted."""
        graph = _RecordingGraph()
        ctx = await create_tenant(session, graph, slug="acme-corp", name="Acme")
        graph.calls.clear()

        await delete_tenant(session, graph, ctx)
        await session.flush()

        assert graph.calls == [("drop", ctx.graph_name)]
        remaining = (
            await session.execute(select(Tenant).where(Tenant.id == ctx.tenant_id))
        ).scalar_one_or_none()
        assert remaining is None

    async def test_is_idempotent_for_a_missing_row(self, session: AsyncSession) -> None:
        graph = _RecordingGraph()
        ctx = await create_tenant(session, graph, slug="acme-corp", name="Acme")
        await delete_tenant(session, graph, ctx)
        await session.flush()
        # A retried offboarding must not raise.
        await delete_tenant(session, graph, ctx)


class TestAgainstLiveGraph:
    """The same lifecycle against real FalkorDB, since provision() creating
    indexes is exactly the kind of thing a fake cannot verify."""

    async def test_provision_is_idempotent(
        self, session: AsyncSession, graph: FalkorDBGraphStore
    ) -> None:
        ctx = await create_tenant(session, graph, slug="live-provision-test", name="Live")
        try:
            # Re-provisioning must not raise on already-existing indexes.
            await graph.provision(ctx)
            await graph.provision(ctx)
            assert (await graph.stats(ctx))["nodes"] == 0
        finally:
            await graph.drop(ctx)

    async def test_drop_is_idempotent(
        self, session: AsyncSession, graph: FalkorDBGraphStore
    ) -> None:
        ctx = await create_tenant(session, graph, slug="live-drop-test", name="Live")
        await graph.drop(ctx)
        # Dropping a graph that is already gone must not raise.
        await graph.drop(ctx)
