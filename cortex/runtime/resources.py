"""Per-process resource container.

Every service — gateway, investigation worker, ingest worker — needs the same
backing stores and must own their lifecycle explicitly. Module-level singletons
were the previous approach and they are wrong here for a concrete reason: an
async client caches a connection bound to the event loop that created it, so a
process that runs more than one loop (a Celery worker, or a test suite) reuses a
client against a dead loop and fails with "Event loop is closed".

Resources are therefore created and disposed per process lifecycle, and handed to
callers by injection rather than imported.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from cortex.config.settings import Settings, get_settings
from cortex.memory.embeddings import Embeddings, VoyageEmbeddings
from cortex.memory.falkordb_store import FalkorDBGraphStore
from cortex.memory.graph_store import GraphStore
from cortex.memory.vector_store import QdrantVectorStore, VectorStore
from cortex.tenancy.limits import NullRateLimiter, RateLimiter, RedisRateLimiter


@dataclass(slots=True)
class Resources:
    settings: Settings
    engine: AsyncEngine
    sessionmaker: async_sessionmaker[AsyncSession]
    graph: GraphStore
    vectors: VectorStore
    embeddings: Embeddings
    #: Per-tenant call ceilings. A `NullRateLimiter` when Redis is unreachable at startup,
    #: because an advisory limiter must not be the reason a process cannot start.
    limiter: RateLimiter

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """A unit of work that commits on success and rolls back on failure."""
        async with self.sessionmaker() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise


def run_with_resources[T](fn: Callable[[Resources], Awaitable[T]]) -> T:
    """Bridge a synchronous entrypoint (a Celery task) to async code.

    A fresh event loop and a fresh set of resources per call: async clients cache
    connections against the loop that created them, so reusing a loop-bound client
    across tasks is what produces "Event loop is closed" in long-lived workers.
    """

    async def _main() -> T:
        async with open_resources() as resources:
            return await fn(resources)

    return asyncio.run(_main())


@asynccontextmanager
async def open_resources(settings: Settings | None = None) -> AsyncIterator[Resources]:
    """Create the process's resources and dispose of them on exit."""
    settings = settings or get_settings()
    engine = create_async_engine(settings.postgres_dsn, pool_pre_ping=True)
    graph = FalkorDBGraphStore()
    # Constructed unconditionally, but nothing here talks to Qdrant or Voyage until a
    # caller actually recalls: a process that never touches memory must not fail to start
    # because an embedding key is absent, and a process that does must fail loudly rather
    # than return an empty recall that reads as "no history".
    embeddings = VoyageEmbeddings()
    vectors = QdrantVectorStore(embeddings)
    limiter, redis_client = _build_limiter(settings)
    resources = Resources(
        settings=settings,
        engine=engine,
        sessionmaker=async_sessionmaker(engine, expire_on_commit=False),
        graph=graph,
        vectors=vectors,
        embeddings=embeddings,
        limiter=limiter,
    )
    try:
        yield resources
    finally:
        await graph.close()
        await vectors.close()
        if redis_client is not None:
            await redis_client.aclose()
        await engine.dispose()


def _build_limiter(settings: Settings) -> tuple[RateLimiter, Any]:
    """The per-tenant limiter, and the Redis client to close with it.

    Falls back to permitting everything when Redis cannot be constructed. An advisory
    limiter must not be the reason a worker fails to start — and the fallback is honest
    rather than silent, because `NullRateLimiter` reports `limited=False` on every decision,
    so a caller can tell "nothing was rejected" from "nothing was checked".
    """
    windows = [
        (60, settings.tenant_calls_per_minute),
        (3600, settings.tenant_calls_per_hour),
    ]
    # A zero disables that window, which is how a local run opts out without special-casing
    # the environment.
    active = tuple((window, limit) for window, limit in windows if limit > 0)
    if not active:
        return NullRateLimiter(), None

    try:
        import redis.asyncio as redis

        client = redis.from_url(settings.redis_url, decode_responses=True)
    except Exception:  # noqa: BLE001 - see the docstring
        return NullRateLimiter(), None
    return RedisRateLimiter(client, limits=active), client
