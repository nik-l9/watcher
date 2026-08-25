"""Shared HTTP layer.

Failure mapping is exercised through the connectors; this file covers the parts
that belong to the transport itself — chiefly the response size ceiling from
F-04, whose absence let a 13.5 MB body reach an Evidence row and an LLM context.
"""

from __future__ import annotations

import json

import httpx
import pytest

from cortex.tools.base import RateLimited, UpstreamError
from cortex.tools.http import (
    MAX_RESPONSE_BYTES,
    AuthRejected,
    NotFound,
    ResponseTooLarge,
    request_json,
)


async def _call(handler, **kwargs) -> dict:  # type: ignore[no-untyped-def]
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="https://api.example.com") as client:
        return await request_json(client, "GET", "/resource", tool="probe", **kwargs)


class TestResponseSizeLimit:
    """F-04. Truncating would be worse than failing: a report would cite a partial
    result as a complete one."""

    async def test_rejects_a_body_over_the_ceiling(self) -> None:
        oversized = b"x" * (MAX_RESPONSE_BYTES + 1)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=oversized)

        with pytest.raises(ResponseTooLarge, match="limit"):
            await _call(handler)

    async def test_rejects_a_declared_oversize_without_reading_it(self) -> None:
        """A declared Content-Length is refused up front rather than streamed."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=b"{}",
                headers={"Content-Length": str(MAX_RESPONSE_BYTES + 1000)},
            )

        with pytest.raises(ResponseTooLarge, match="declared"):
            await _call(handler)

    async def test_enforces_the_limit_on_chunked_responses(self) -> None:
        """Chunked bodies declare no length, so the streaming check is what holds."""
        chunk = b"y" * (1024 * 1024)

        async def stream():  # type: ignore[no-untyped-def]
            for _ in range(20):
                yield chunk

        def handler(request: httpx.Request) -> httpx.Response:
            # No Content-Length, so only the streaming check can catch this.
            return httpx.Response(200, content=stream())

        with pytest.raises(ResponseTooLarge, match="exceeded"):
            await _call(handler)

    async def test_accepts_a_body_under_the_ceiling(self) -> None:
        payload = {"rows": [{"i": i} for i in range(1000)]}
        assert len(json.dumps(payload)) < MAX_RESPONSE_BYTES

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload)

        body = await _call(handler)
        assert len(body["rows"]) == 1000

    async def test_error_message_does_not_include_the_body(self) -> None:
        """The point of the limit is not moving the huge payload into a log line."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"S3CRET" + b"x" * MAX_RESPONSE_BYTES)

        with pytest.raises(ResponseTooLarge) as exc:
            await _call(handler)
        assert "S3CRET" not in str(exc.value)


class TestBodyHandling:
    async def test_wraps_a_top_level_array(self) -> None:
        """GitHub list endpoints return bare arrays; capabilities need a dict."""
        body = await _call(lambda r: httpx.Response(200, json=[{"a": 1}]))
        assert body == {"data": [{"a": 1}]}

    async def test_non_json_is_an_upstream_error(self) -> None:
        with pytest.raises(UpstreamError, match="non-JSON"):
            await _call(lambda r: httpx.Response(200, text="<html>proxy</html>"))

    async def test_invalid_utf8_does_not_crash(self) -> None:
        """A binary body must produce a diagnosable error, not a UnicodeDecodeError."""
        with pytest.raises(UpstreamError, match="non-JSON"):
            await _call(lambda r: httpx.Response(200, content=b"\xff\xfe\x00binary"))

    async def test_empty_body_is_an_upstream_error(self) -> None:
        with pytest.raises(UpstreamError, match="non-JSON"):
            await _call(lambda r: httpx.Response(200, content=b""))


class TestStatusMapping:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (429, RateLimited),
            (401, AuthRejected),
            (403, AuthRejected),
            (404, NotFound),
            (500, UpstreamError),
            (503, UpstreamError),
            (400, UpstreamError),
        ],
    )
    async def test_maps_status_to_exception(self, status: int, expected: type[Exception]) -> None:
        with pytest.raises(expected):
            await _call(lambda r: httpx.Response(status, json={"error": "x"}))

    async def test_retry_after_is_parsed(self) -> None:
        with pytest.raises(RateLimited) as exc:
            await _call(lambda r: httpx.Response(429, headers={"Retry-After": "17"}, json={}))
        assert exc.value.retry_after_seconds == 17.0

    async def test_http_date_retry_after_is_tolerated(self) -> None:
        """A date-form hint is survivable; the loop has its own backoff."""
        with pytest.raises(RateLimited) as exc:
            await _call(
                lambda r: httpx.Response(
                    429,
                    headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"},
                    json={},
                )
            )
        assert exc.value.retry_after_seconds is None


class TestNoCredentialLeakage:
    async def test_query_string_is_stripped_from_errors(self) -> None:
        """Query strings routinely carry access tokens and customer identifiers."""
        with pytest.raises(UpstreamError) as exc:
            transport = httpx.MockTransport(lambda r: httpx.Response(500, text="boom"))
            async with httpx.AsyncClient(
                transport=transport, base_url="https://api.example.com"
            ) as client:
                await request_json(
                    client,
                    "GET",
                    "/resource",
                    tool="probe",
                    params={"access_token": "SUPER-SECRET", "q": "customer@example.com"},
                )
        assert "SUPER-SECRET" not in str(exc.value)
        assert "customer@example.com" not in str(exc.value)

    async def test_error_bodies_are_truncated(self) -> None:
        with pytest.raises(UpstreamError) as exc:
            await _call(lambda r: httpx.Response(400, text="e" * 5000))
        assert len(str(exc.value)) < 1000
