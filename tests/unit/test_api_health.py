"""Gateway health and readiness.

The distinction matters operationally: liveness must never depend on a backing
store, or an orchestrator restarts healthy pods during a database blip. Readiness
must depend on them, and must report degraded rather than raise.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from services.gateway import app as gateway_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A fresh app per test.

    create_app() plus lifespan-managed resources means each client gets its own
    graph client on its own event loop — the reason resources are not module
    globals.
    """
    with TestClient(gateway_app.create_app()) as c:
        yield c


class TestHealth:
    def test_ok(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "gateway"

    def test_does_not_touch_backing_stores(self, client: TestClient) -> None:
        """Liveness must stay green with every dependency down, otherwise a
        database blip triggers a restart storm across the fleet."""
        resources = client.app.state.resources  # type: ignore[attr-defined]

        def _explode(*_a: object, **_k: object) -> None:
            raise AssertionError("liveness must not perform IO")

        original_maker, original_ping = resources.sessionmaker, resources.graph.ping
        resources.sessionmaker = _explode
        resources.graph.ping = _explode
        try:
            assert client.get("/health").status_code == 200
        finally:
            resources.sessionmaker = original_maker
            resources.graph.ping = original_ping


class TestReady:
    def test_ready_when_stores_answer(self, client: TestClient) -> None:
        body = client.get("/ready").json()
        assert body["ready"] is True
        assert body["checks"] == {"postgres": "ok", "falkordb": "ok"}

    def test_degrades_when_graph_is_down(self, client: TestClient) -> None:
        resources = client.app.state.resources  # type: ignore[attr-defined]

        async def _fail() -> bool:
            raise ConnectionError("falkordb unreachable")

        original = resources.graph.ping
        resources.graph.ping = _fail
        try:
            r = client.get("/ready")
            # Degraded, not a 500: a 500 tells an operator nothing about which
            # dependency failed.
            assert r.status_code == 200
            body = r.json()
            assert body["ready"] is False
            assert body["checks"]["falkordb"].startswith("error: ConnectionError")
            assert body["checks"]["postgres"] == "ok"
        finally:
            resources.graph.ping = original

    def test_degrades_when_postgres_is_down(self, client: TestClient) -> None:
        resources = client.app.state.resources  # type: ignore[attr-defined]

        def _broken() -> None:
            raise TimeoutError("postgres unreachable")

        original = resources.sessionmaker
        resources.sessionmaker = _broken
        try:
            body = client.get("/ready").json()
            assert body["ready"] is False
            assert body["checks"]["postgres"].startswith("error: TimeoutError")
        finally:
            resources.sessionmaker = original

    def test_does_not_leak_connection_strings(self, client: TestClient) -> None:
        """Readiness is often exposed unauthenticated, so it reports the exception
        type only — never a message that could carry a DSN or password."""
        resources = client.app.state.resources  # type: ignore[attr-defined]

        async def _fail() -> bool:
            raise ConnectionError(
                "could not connect to postgresql://cortex:sup3rsecret@10.0.0.5:5432/cortex"
            )

        original = resources.graph.ping
        resources.graph.ping = _fail
        try:
            raw = client.get("/ready").text
            assert "sup3rsecret" not in raw
            assert "10.0.0.5" not in raw
            assert "postgresql://" not in raw
        finally:
            resources.graph.ping = original


class TestSurface:
    def test_exposes_only_health_and_readiness(self, client: TestClient) -> None:
        """Investigation endpoints land in M7. Asserting their absence keeps an
        ungrounded or unauthenticated endpoint from appearing by accident."""
        paths = set(client.app.openapi()["paths"])  # type: ignore[attr-defined]
        # An explicit inventory, not a lower bound: a route that appears here without
        # being added deliberately is an unreviewed piece of public surface.
        assert paths == {
            "/health",
            "/ready",
            "/investigations",
            "/investigations/{investigation_id}",
            # Cooperative: it marks the row, and the loop reads the row once per step. Added
            # deliberately, which is what this inventory is for -- it caught this endpoint on
            # the run that introduced it.
            "/investigations/{investigation_id}/cancel",
            # What the analyst actually did, derived from the tool_calls audit rows rather
            # than from a stored event stream. Params only, never a response body -- the
            # audit row already draws that line, and this must not reintroduce a second
            # surface on which customer data reaches a UI.
            "/investigations/{investigation_id}/trace",
            "/reports/{report_id}",
            # Scoped to the caller's own tenant. There is deliberately no route that takes a
            # tenant parameter, so no version of this can read across tenants.
            "/spend",
        }, paths
