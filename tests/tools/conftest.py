"""Connector test harness.

Connectors are tested against recorded upstream payloads served by a mock httpx
transport, not against live APIs. Live tests would need real credentials, would be
non-deterministic, and could not exercise the failure paths that matter most —
429s, 401s, malformed bodies.

Google token minting is stubbed out here rather than mocked per test: every GA4 and
BigQuery test would otherwise have to fake a signed JWT exchange, which tests
google-auth rather than Cortex.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from cortex.memory.naming import graph_name_for_new_tenant
from cortex.tenancy.context import TenantContext
from cortex.tools import bigquery as bigquery_module
from cortex.tools import ga4 as ga4_module
from cortex.tools.base import ToolContext

# A syntactically complete service-account key with no real private key material.
FAKE_SERVICE_ACCOUNT = json.dumps(
    {
        "type": "service_account",
        "project_id": "cortex-test-project",
        "private_key_id": "fake-key-id",
        "private_key": "-----BEGIN PRIVATE KEY-----\nnot-a-real-key\n-----END PRIVATE KEY-----\n",
        "client_email": "cortex@cortex-test-project.iam.gserviceaccount.com",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
)


@pytest.fixture
def tenant() -> TenantContext:
    tenant_id = uuid.uuid4()
    return TenantContext(
        tenant_id=tenant_id,
        tenant_slug="tool-test",
        graph_name=graph_name_for_new_tenant("tool-test", tenant_id),
    )


@pytest.fixture(autouse=True)
def stub_google_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass the JWT-bearer exchange in connector tests.

    Token minting itself is covered separately in test_google_auth.py.
    """

    async def _token(raw: str, scopes: tuple[str, ...], **_: object) -> str:
        del raw, scopes
        return "test-access-token"

    monkeypatch.setattr(ga4_module, "access_token", _token)
    monkeypatch.setattr(bigquery_module, "access_token", _token)


class RecordedTransport(httpx.MockTransport):
    """A mock transport that records the requests it served.

    Recording matters as much as responding: several guarantees — that a value is
    bound as a query parameter rather than interpolated, that a limit is actually
    forwarded — are only observable in the outgoing request.
    """

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []

        def _wrapped(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return handler(request)

        super().__init__(_wrapped)

    def request_bodies(self) -> list[dict[str, Any]]:
        bodies = []
        for request in self.requests:
            raw = request.content
            if not raw:
                continue
            try:
                bodies.append(json.loads(raw))
            except ValueError:
                continue
        return bodies


@pytest.fixture
def patch_client(monkeypatch: pytest.MonkeyPatch) -> Callable[..., RecordedTransport]:
    """Route a tool's HTTP calls through a recorded mock transport.

    Patches the tool's own `_client` factory so the connector's real headers, base
    URL and request construction are exercised — only the socket is replaced.
    """

    def _install(
        tool: object,
        responses: dict[str, Any] | Callable[[httpx.Request], httpx.Response],
        *,
        is_async_factory: bool = False,
    ) -> RecordedTransport:
        if callable(responses):
            handler = responses
        else:

            def handler(request: httpx.Request) -> httpx.Response:
                for pattern, payload in responses.items():
                    if pattern in str(request.url):
                        if isinstance(payload, httpx.Response):
                            return payload
                        return httpx.Response(200, json=payload)
                return httpx.Response(404, json={"error": f"no fixture for {request.url.path}"})

        transport = RecordedTransport(handler)
        original = tool._client  # type: ignore[attr-defined]

        def _sync_client(ctx: ToolContext) -> httpx.AsyncClient:
            client = original(ctx)
            return httpx.AsyncClient(
                transport=transport,
                base_url=client.base_url,
                headers=client.headers,
                timeout=client.timeout,
            )

        async def _async_client(ctx: ToolContext) -> httpx.AsyncClient:
            client = await original(ctx)
            return httpx.AsyncClient(
                transport=transport,
                base_url=client.base_url,
                headers=client.headers,
                timeout=client.timeout,
            )

        monkeypatch.setattr(tool, "_client", _async_client if is_async_factory else _sync_client)
        return transport

    return _install
