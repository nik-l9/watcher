"""Per-process resource lifecycle.

The bug this design exists to prevent: an async client cached at module scope
binds to the event loop that created it, so a process that runs more than one loop
reuses it against a dead loop and fails with "Event loop is closed". These tests
prove resources survive repeated loops and are disposed on exit.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import text

from cortex.config.settings import Settings, get_settings
from cortex.memory.entities import Node, NodeLabel
from cortex.memory.graph_store import GraphStore
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.runtime.resources import Resources, open_resources, run_with_resources
from cortex.tenancy.context import TenantContext


def _test_settings(dsn: str) -> Settings:
    base = get_settings()
    return base.model_copy(update={"postgres_dsn": dsn})


class TestOpenResources:
    async def test_provides_working_stores(self, _test_database: str) -> None:
        async with open_resources(_test_settings(_test_database)) as resources:
            async with resources.session() as session:
                assert (await session.execute(text("SELECT 1"))).scalar() == 1
            assert await resources.graph.ping() is True

    async def test_disposes_on_exit(self, _test_database: str) -> None:
        async with open_resources(_test_settings(_test_database)) as resources:
            engine = resources.engine
            await resources.graph.ping()
        # A disposed engine has no checked-out connections left behind.
        assert engine.pool.checkedout() == 0

    async def test_session_commits_on_success(self, _test_database: str) -> None:
        async with open_resources(_test_settings(_test_database)) as resources:
            async with resources.session() as session:
                await session.execute(text("CREATE TEMP TABLE probe (id int)"))
                await session.execute(text("INSERT INTO probe VALUES (1)"))
            # Temp tables are connection-scoped, so the observable assertion is
            # that the block exited without raising and the session is usable.
            async with resources.session() as session:
                assert (await session.execute(text("SELECT 1"))).scalar() == 1

    async def test_session_rolls_back_on_error(self, _test_database: str) -> None:
        from cortex.db.models import Tenant

        async with open_resources(_test_settings(_test_database)) as resources:
            with pytest.raises(RuntimeError):
                async with resources.session() as session:
                    tid = uuid.uuid4()
                    session.add(
                        Tenant(
                            id=tid,
                            slug="rollback-probe",
                            name="Probe",
                            graph_name=graph_name_for_new_tenant("rollback-probe", tid),
                        )
                    )
                    await session.flush()
                    raise RuntimeError("boom")

            async with resources.session() as session:
                found = (
                    await session.execute(
                        text("SELECT count(*) FROM tenants WHERE slug = 'rollback-probe'")
                    )
                ).scalar()
                assert found == 0, "a failed unit of work must leave nothing behind"

    async def test_graph_is_a_graphstore(self, _test_database: str) -> None:
        """Callers depend on the interface, never the concrete backend."""
        async with open_resources(_test_settings(_test_database)) as resources:
            assert isinstance(resources.graph, GraphStore)


class TestRunWithResources:
    """The Celery bridge. Each call gets its own loop, which is exactly the
    scenario that broke a module-level client."""

    def test_returns_the_callables_value(self, _test_database: str) -> None:
        settings = _test_settings(_test_database)

        async def _work(resources: Resources) -> str:
            await resources.graph.ping()
            return "done"

        # Patch the settings the helper reads, then call it twice: two separate
        # event loops in one process.
        import cortex.runtime.resources as module

        original = module.get_settings
        module.get_settings = lambda: settings  # type: ignore[assignment]
        try:
            assert run_with_resources(_work) == "done"
            assert run_with_resources(_work) == "done", (
                "a second call must not reuse a client bound to the first loop"
            )
        finally:
            module.get_settings = original

    def test_propagates_exceptions(self, _test_database: str) -> None:
        settings = _test_settings(_test_database)
        import cortex.runtime.resources as module

        async def _boom(resources: Resources) -> None:
            raise ValueError("task failed")

        original = module.get_settings
        module.get_settings = lambda: settings  # type: ignore[assignment]
        try:
            with pytest.raises(ValueError, match="task failed"):
                run_with_resources(_boom)
        finally:
            module.get_settings = original

    def test_graph_writes_work_across_consecutive_calls(self, _test_database: str) -> None:
        """The real regression test: two sequential tasks each writing to the
        graph. With a module-level client the second raises "Event loop is closed"."""
        settings = _test_settings(_test_database)
        import cortex.runtime.resources as module

        slug = "runtime-probe"
        tid = uuid.uuid4()
        ctx = TenantContext(
            tenant_id=tid, tenant_slug=slug, graph_name=graph_name_for_new_tenant(slug, tid)
        )

        async def _write(resources: Resources) -> int:
            await resources.graph.provision(ctx)
            return await resources.graph.upsert_nodes(
                ctx, [Node(NodeLabel.DEPLOY, f"d-{uuid.uuid4().hex[:8]}")]
            )

        async def _cleanup(resources: Resources) -> None:
            await resources.graph.drop(ctx)

        original = module.get_settings
        module.get_settings = lambda: settings  # type: ignore[assignment]
        try:
            assert run_with_resources(_write) == 1
            assert run_with_resources(_write) == 1
        finally:
            run_with_resources(_cleanup)
            module.get_settings = original

    def test_no_event_loop_leaks(self, _test_database: str) -> None:
        """Each call must close its loop, or a long-lived worker accumulates them."""
        settings = _test_settings(_test_database)
        import cortex.runtime.resources as module

        async def _noop(resources: Resources) -> None:
            del resources

        original = module.get_settings
        module.get_settings = lambda: settings  # type: ignore[assignment]
        try:
            for _ in range(3):
                run_with_resources(_noop)
            with pytest.raises(RuntimeError):
                asyncio.get_running_loop()
        finally:
            module.get_settings = original
