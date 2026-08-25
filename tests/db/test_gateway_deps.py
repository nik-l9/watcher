"""Gateway dependency wiring, with nothing overridden.

`test_gateway_tenant_dep.py` overrides `get_session` so it can focus on the tenancy
error mapping. That leaves the wiring itself — resource lookup off `app.state`, the
session generator's commit-and-close, the graph accessor — running only in
production, which is the wrong place to discover a mistake in it.

Here the real `lifespan` builds real resources against the test database, and no
dependency is overridden.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.config.settings import Settings, get_settings
from cortex.db.models import Tenant
from cortex.memory.graph_store import GraphStore
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.runtime.resources import open_resources
from cortex.tenancy.context import TenantContext
from services.gateway.deps import get_graph, get_resources, get_session, require_tenant


@pytest.fixture
def app(_test_database: str) -> Iterator[FastAPI]:
    """An app whose resources come from the real lifespan, not a fixture."""
    settings: Settings = get_settings().model_copy(
        update={"postgres_dsn": _test_database, "env": "test"}
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        async with open_resources(settings) as resources:
            application.state.resources = resources
            yield

    application = FastAPI(lifespan=lifespan)

    @application.get("/via-session")
    async def via_session(session: AsyncSession = Depends(get_session)) -> dict[str, int]:
        return {"answer": (await session.execute(text("SELECT 42"))).scalar_one()}

    @application.get("/via-graph")
    async def via_graph(graph: GraphStore = Depends(get_graph)) -> dict[str, bool]:
        return {"is_graph_store": isinstance(graph, GraphStore)}

    @application.get("/via-resources")
    async def via_resources(resources=Depends(get_resources)) -> dict[str, str]:  # type: ignore[no-untyped-def]
        return {"env": resources.settings.env}

    @application.post("/writes")
    async def writes(session: AsyncSession = Depends(get_session)) -> dict[str, str]:
        """Writes through the injected session so the commit-on-success path runs."""
        slug = f"dep-{uuid.uuid4().hex[:8]}"
        tenant_id = uuid.uuid4()
        session.add(
            Tenant(
                id=tenant_id,
                slug=slug,
                name=slug,
                graph_name=graph_name_for_new_tenant(slug, tenant_id),
            )
        )
        await session.flush()
        return {"slug": slug}

    @application.get("/scoped")
    async def scoped(ctx: TenantContext = Depends(require_tenant)) -> dict[str, str]:
        return {"tenant": ctx.tenant_slug}

    yield application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


class TestResourceInjection:
    def test_resources_come_from_app_state(self, client: TestClient) -> None:
        assert client.get("/via-resources").json() == {"env": "test"}

    def test_graph_is_injected_as_the_interface(self, client: TestClient) -> None:
        """Callers depend on GraphStore, never the concrete backend."""
        assert client.get("/via-graph").json() == {"is_graph_store": True}

    def test_session_is_usable(self, client: TestClient) -> None:
        assert client.get("/via-session").json() == {"answer": 42}


class TestSessionLifecycle:
    def test_a_write_through_the_injected_session_commits(
        self, client: TestClient, _test_database: str
    ) -> None:
        """The dependency commits on success. Without that, every write through the
        gateway would silently roll back at request end."""
        slug = client.post("/writes").json()["slug"]

        # Verified on a separate connection: an uncommitted row would be invisible.
        import asyncio

        from sqlalchemy.ext.asyncio import create_async_engine

        async def _lookup() -> str | None:
            engine = create_async_engine(_test_database)
            try:
                async with engine.connect() as conn:
                    return (
                        await conn.execute(select(Tenant.slug).where(Tenant.slug == slug))
                    ).scalar_one_or_none()
            finally:
                await engine.dispose()

        assert asyncio.run(_lookup()) == slug

    def test_repeated_requests_reuse_the_pool(self, client: TestClient) -> None:
        """Each request takes a session from the same engine. A leak here shows up as
        pool exhaustion under load rather than as a test failure, so the check is
        simply that many sequential requests keep working."""
        for _ in range(15):
            assert client.get("/via-session").status_code == 200


class TestTenancyThroughRealWiring:
    def test_a_known_tenant_resolves(self, client: TestClient, _test_database: str) -> None:
        """require_tenant, get_session and get_resources all in one path, unmocked."""
        import asyncio

        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        slug = f"real-{uuid.uuid4().hex[:8]}"

        async def _seed() -> None:
            engine = create_async_engine(_test_database)
            try:
                async with async_sessionmaker(engine)() as session:
                    tenant_id = uuid.uuid4()
                    session.add(
                        Tenant(
                            id=tenant_id,
                            slug=slug,
                            name=slug,
                            graph_name=graph_name_for_new_tenant(slug, tenant_id),
                        )
                    )
                    await session.commit()
            finally:
                await engine.dispose()

        asyncio.run(_seed())

        response = client.get("/scoped", headers={"X-Cortex-Tenant": slug})
        assert response.status_code == 200
        assert response.json() == {"tenant": slug}

    def test_an_unknown_tenant_is_404_through_real_wiring(self, client: TestClient) -> None:
        assert client.get("/scoped", headers={"X-Cortex-Tenant": "nope-nope"}).status_code == 404
