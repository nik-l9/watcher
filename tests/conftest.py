"""Shared test fixtures.

Tests run against the real docker-compose stack rather than mocks. Tenant
isolation is the property most worth proving, and a mock graph would happily
"prove" a guarantee the real engine does not provide.

Postgres tests use a separate `cortex_test` database so a test run can never
disturb local development data.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from cortex.config.settings import get_settings
from cortex.db.models import Base
from cortex.memory.embeddings import DeterministicEmbeddings
from cortex.memory.falkordb_store import FalkorDBGraphStore
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.memory.vector_store import QdrantVectorStore
from cortex.tenancy.context import TenantContext

TEST_DB = "cortex_test"


def _make_ctx(slug: str) -> TenantContext:
    tenant_id = uuid.uuid4()
    return TenantContext(
        tenant_id=tenant_id,
        tenant_slug=slug,
        graph_name=graph_name_for_new_tenant(slug, tenant_id),
        role="admin",
    )


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


# ------------------------------------------------------------------ graph


@pytest_asyncio.fixture
async def graph() -> AsyncIterator[FalkorDBGraphStore]:
    store = FalkorDBGraphStore()
    try:
        yield store
    finally:
        await store.close()


@pytest_asyncio.fixture
async def tenant_a(graph: FalkorDBGraphStore) -> AsyncIterator[TenantContext]:
    ctx = _make_ctx("test-tenant-a")
    await graph.provision(ctx)
    try:
        yield ctx
    finally:
        await graph.drop(ctx)


@pytest_asyncio.fixture
async def tenant_b(graph: FalkorDBGraphStore) -> AsyncIterator[TenantContext]:
    ctx = _make_ctx("test-tenant-b")
    await graph.provision(ctx)
    try:
        yield ctx
    finally:
        await graph.drop(ctx)


# ------------------------------------------------------------------ vectors


#: The local Qdrant from docker-compose, regardless of what `.env` points at.
#:
#: Pinned rather than inherited: a developer `.env` names a managed cluster, and pointing
#: the isolation gate at it made the suite take 110 seconds and fail on a different test
#: each run with an empty `ResponseHandlingException` — a round trip to another continent,
#: not a logic error. A gate that is slow and flaky stops being run, which is worse than
#: one that tests slightly less. Real engine, local instance: the same choice the rest of
#: this file makes for Postgres and FalkorDB.
LOCAL_QDRANT_URL = "http://localhost:6333"


@pytest_asyncio.fixture
async def vectors() -> AsyncIterator[QdrantVectorStore]:
    """A vector store backed by the real Qdrant and a deterministic embedder.

    Real Qdrant, because per-tenant collection isolation is a property of the engine and
    a fake would happily "prove" it. Fake embeddings, because the isolation, upsert and
    ranking behaviour under test does not depend on semantic quality — and a suite that
    calls a paid embedding API cannot be run offline or in CI for free.
    """
    store = QdrantVectorStore(DeterministicEmbeddings(), url=LOCAL_QDRANT_URL)
    try:
        yield store
    finally:
        await store.close()


@pytest_asyncio.fixture
async def vector_tenant_a(vectors: QdrantVectorStore) -> AsyncIterator[TenantContext]:
    ctx = _make_ctx("vec-tenant-a")
    await vectors.provision(ctx)
    try:
        yield ctx
    finally:
        await vectors.drop(ctx)


@pytest_asyncio.fixture
async def vector_tenant_b(vectors: QdrantVectorStore) -> AsyncIterator[TenantContext]:
    ctx = _make_ctx("vec-tenant-b")
    await vectors.provision(ctx)
    try:
        yield ctx
    finally:
        await vectors.drop(ctx)


# ------------------------------------------------------------------ postgres


def _test_dsn() -> str:
    dsn = get_settings().postgres_dsn
    base, _, _ = dsn.rpartition("/")
    return f"{base}/{TEST_DB}"


@pytest_asyncio.fixture(scope="session")
async def _test_database() -> AsyncIterator[str]:
    """Create the test database and its schema once per session."""
    admin = create_async_engine(get_settings().postgres_dsn, isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        exists = (
            await conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": TEST_DB})
        ).scalar()
        if not exists:
            await conn.execute(text(f'CREATE DATABASE "{TEST_DB}"'))
    await admin.dispose()

    dsn = _test_dsn()
    engine = create_async_engine(dsn)
    async with engine.begin() as conn:
        # create_all rather than alembic: the migration is separately verified by
        # `make migrate`, and tests want a fast, deterministic schema.
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()

    yield dsn


@pytest_asyncio.fixture(autouse=True)
async def _clean_tables(request: pytest.FixtureRequest) -> AsyncIterator[None]:
    """Truncate every table before each DB-backed test.

    Rollback alone is not enough: a test that legitimately commits — anything
    exercising a real request path — would otherwise leak rows into later tests
    and produce failures that only appear in a full run.
    """
    if "session" not in request.fixturenames and "_test_database" not in request.fixturenames:
        yield
        return

    dsn = request.getfixturevalue("_test_database")
    engine = create_async_engine(dsn)
    tables = ", ".join(f'"{t.name}"' for t in reversed(Base.metadata.sorted_tables))
    async with engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    await engine.dispose()
    yield


@pytest_asyncio.fixture
async def session(_test_database: str) -> AsyncIterator[AsyncSession]:
    """A session for DB-backed tests. Isolation comes from _clean_tables."""
    engine = create_async_engine(_test_database)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        try:
            yield s
        finally:
            await s.rollback()
    await engine.dispose()
