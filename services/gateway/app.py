"""Gateway service.

The only externally-reachable Cortex service. It authenticates, resolves tenancy,
persists investigation requests, and enqueues work. It deliberately does not run
investigations: a long LLM loop inside the request path would couple user-facing
availability to the slowest tool call, and would scale the HTTP tier for the wrong
reason.

Surface: health, readiness, the investigation endpoints, and the HTML report view.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from sqlalchemy import text

from cortex.config.settings import get_settings
from cortex.runtime.resources import Resources, open_resources
from services.gateway.deps import get_resources
from services.gateway.investigations import router as investigations_router
from services.gateway.slack_events import router as slack_router
from services.gateway.views import router as views_router

SERVICE_NAME = "gateway"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with open_resources() as resources:
        app.state.resources = resources
        yield


def create_app() -> FastAPI:
    """Factory, not a module-level instance, so tests can build isolated apps."""
    app = FastAPI(title="Cortex Gateway", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness only. Performs no IO, so a database blip cannot trigger a
        restart storm across the fleet."""
        return {"status": "ok", "service": SERVICE_NAME, "env": get_settings().env}

    @app.get("/ready")
    async def ready(resources: Resources = Depends(get_resources)) -> dict[str, object]:
        """Readiness: confirm each backing store answers.

        Reports degraded rather than raising, and reports the exception type only —
        readiness is often exposed unauthenticated and an exception message can
        carry a DSN or password.
        """
        checks: dict[str, str] = {}

        try:
            async with resources.session() as session:
                await session.execute(text("SELECT 1"))
            checks["postgres"] = "ok"
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            checks["postgres"] = f"error: {type(exc).__name__}"

        try:
            await resources.graph.ping()
            checks["falkordb"] = "ok"
        except Exception as exc:  # noqa: BLE001
            checks["falkordb"] = f"error: {type(exc).__name__}"

        return {
            "ready": all(v == "ok" for v in checks.values()),
            "service": SERVICE_NAME,
            "checks": checks,
        }

    # Mounted last so the health endpoints stay defined above it and remain
    # dependency-free — liveness must not acquire a session.
    app.include_router(investigations_router)
    # The HTML report view. `include_in_schema=False` on its router keeps it out of the
    # OpenAPI document: it is a rendering of the endpoints above, not a second API, and
    # listing it as one would invite a client to scrape HTML for data the JSON already
    # carries.
    app.include_router(views_router)
    # The Slack webhook. Out of the OpenAPI document like the HTML views, and deliberately last:
    # it is the only route with no Clerk or tenant-header dependency, because its authentication
    # is the request signature itself.
    app.include_router(slack_router)

    return app


app = create_app()
