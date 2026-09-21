"""Shared HTTP client for connectors.

Every connector faces the same failure modes — rate limits, auth rejection,
timeouts, 5xx — and each must map them to the same Cortex exceptions so the
investigation loop can react uniformly: retry a 429, give up on a 401, treat a 404
as an empty observation rather than an error.

Doing this once here also means an upstream error body cannot leak a credential
into a log: only status, method and a truncated body reach the exception, and the
Authorization header is never rendered.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from cortex.tools.base import RateLimited, ToolError, UpstreamError

#: Per-call timeouts.
#:
#: `connect` is 12 seconds, raised from 5 on a measurement rather than a preference. A
#: connector opens a fresh client per call, so the *first* request to each host in a process
#: pays DNS resolution plus a TLS handshake — and 5 seconds turned out not to cover it. In one
#: live investigation the two opening calls failed at 5,023ms and 5,013ms, and the identical
#: GitHub request succeeded in 1,864ms moments later once the name was resolved:
#:
#:     github.list_repositories   ok=False  5023ms  timeout calling GET /user/repos
#:     posthog.list_events        ok=False  5013ms  timeout calling GET /api/.../event_definitions/
#:     github.list_repositories   ok=True   1864ms
#:
#: Both casualties were the environment survey, which runs first by construction — so the
#: tightest connect budget in the system was being spent on the calls that establish what
#: exists, and their loss is the least visible kind. Nothing was wrong with either endpoint.
#:
#: `read` stays at 30 seconds. The two limits guard different failures: a slow *connection* is
#: almost always the network warming up, while a slow *response* is a query that will not
#: finish, and conflating them would make a hung read wait twice as long.
DEFAULT_TIMEOUT = httpx.Timeout(connect=12.0, read=30.0, write=10.0, pool=5.0)

#: For endpoints that search rather than fetch.
#:
#: A keyed read returns in well under 30 seconds; a full-text search over an entire Slack
#: workspace or HubSpot portal does not reliably. A real investigation lost two of three
#: Slack calls and two HubSpot deal searches to the 30-second read timeout, and reported
#: the human record as unavailable as a result — the analyst was left inferring from
#: metrics because the search never came back, which is the most valuable source failing
#: in the least visible way.
#:
#: Applied per call rather than raised globally, so a hung *keyed* read still fails fast.
SEARCH_TIMEOUT = httpx.Timeout(connect=12.0, read=90.0, write=10.0, pool=5.0)

# Upstream bodies can be enormous. Enough to diagnose, not enough to fill a log.
_MAX_ERROR_BODY = 500

# Hard ceiling on any single upstream response. An unbounded body costs worker
# memory, database size, and — because evidence becomes prompt input — tokens on a
# real invoice. Exceeding it is an error rather than a truncation: a silently
# truncated response would let a report cite a partial result as a complete one.
# See docs/security-findings.md F-04.
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class AuthRejected(ToolError):
    """Upstream refused the credential. Not retryable — the tenant must reconnect."""


class NotFound(UpstreamError):
    """The requested resource does not exist upstream."""


class ResponseTooLarge(UpstreamError):
    """Upstream returned more data than Cortex will accept for one observation."""


async def request_json(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    tool: str,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout: httpx.Timeout | None = None,
    headers: dict[str, str] | None = None,
    raw_on_non_json: bool = False,
) -> dict[str, Any] | str:
    """Perform a request and return parsed JSON, mapping failures to ToolErrors.

    `timeout` overrides the client's own for this one call — pass `SEARCH_TIMEOUT` for a
    full-text search, which is legitimately slower than a keyed read.

    `headers` adds to the client's own, for a per-request value such as a session id.

    `raw_on_non_json` returns the decoded body instead of raising when it will not parse.
    MCP's Streamable HTTP transport answers the same request with either a JSON object or
    an SSE stream, at the server's discretion, so for that caller a non-JSON body is a
    normal response rather than a broken one. Everything else keeps the strict behaviour:
    a connector that expects JSON and gets HTML has met an error, not a format.

    The size bounding above applies either way, which is why this is a parameter here
    rather than a second request path in the caller — an untrusted third-party server is
    exactly what those bounds are for.
    """
    try:
        # Streamed so an oversized body is refused while it arrives, rather than
        # after it has already been read into memory.
        extra: dict[str, Any] = {"timeout": timeout} if timeout is not None else {}
        if headers:
            extra["headers"] = headers
        request = client.build_request(method, url, params=params, json=json_body, **extra)
        response = await client.send(request, stream=True)
        try:
            _check_declared_size(response, tool, method, url)
            body_bytes = await _read_bounded(response, tool, method, url)
        finally:
            await response.aclose()
    except httpx.TimeoutException as exc:
        raise UpstreamError(f"{tool}: timeout calling {_safe_target(method, url)}") from exc
    except httpx.HTTPError as exc:
        raise UpstreamError(
            f"{tool}: transport error calling {_safe_target(method, url)}: {type(exc).__name__}"
        ) from exc

    if response.status_code == 429:
        raise RateLimited(
            f"{tool}: rate limited by {_safe_target(method, url)}",
            retry_after_seconds=_retry_after(response),
        )
    if response.status_code in (401, 403):
        raise AuthRejected(
            f"{tool}: credential rejected ({response.status_code}) by "
            f"{_safe_target(method, url)}; the tenant may need to reconnect"
        )
    if response.status_code == 404:
        raise NotFound(f"{tool}: not found at {_safe_target(method, url)}")
    if response.status_code >= 500:
        raise UpstreamError(
            f"{tool}: upstream error {response.status_code} from {_safe_target(method, url)}"
        )
    if response.status_code >= 400:
        raise UpstreamError(
            f"{tool}: {response.status_code} from {_safe_target(method, url)}: "
            f"{_truncate(_decode(body_bytes))}"
        )

    try:
        body = json.loads(body_bytes)
    except ValueError as exc:
        if raw_on_non_json:
            return _decode(body_bytes)
        raise UpstreamError(
            f"{tool}: {_safe_target(method, url)} returned non-JSON: "
            f"{_truncate(_decode(body_bytes))}"
        ) from exc

    if not isinstance(body, dict):
        # Wrapped rather than rejected: some APIs legitimately return a top-level
        # array, and capabilities need a dict to build a payload from.
        return {"data": body}
    return body


def _check_declared_size(response: httpx.Response, tool: str, method: str, url: str) -> None:
    """Refuse an oversized body before reading it, when the length is declared."""
    declared = response.headers.get("Content-Length")
    if declared and declared.isdigit() and int(declared) > MAX_RESPONSE_BYTES:
        raise ResponseTooLarge(
            f"{tool}: {_safe_target(method, url)} declared {int(declared)} bytes, "
            f"over the {MAX_RESPONSE_BYTES} byte limit; narrow the request"
        )


async def _read_bounded(response: httpx.Response, tool: str, method: str, url: str) -> bytes:
    """Read a response, aborting once it exceeds the ceiling.

    Chunked responses declare no length, so the limit is also enforced while
    reading — otherwise the check above would be trivially bypassed.
    """
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > MAX_RESPONSE_BYTES:
            raise ResponseTooLarge(
                f"{tool}: {_safe_target(method, url)} exceeded the "
                f"{MAX_RESPONSE_BYTES} byte limit; narrow the request"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _decode(body: bytes) -> str:
    return body.decode("utf-8", errors="replace")


def _safe_target(method: str, url: str) -> str:
    """A loggable description of the request.

    Query strings are dropped: they routinely carry access tokens, api keys and
    customer identifiers.
    """
    base, _, _ = url.partition("?")
    return f"{method.upper()} {base}"


def _truncate(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= _MAX_ERROR_BODY:
        return collapsed
    return collapsed[:_MAX_ERROR_BODY] + "…"


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        # Retry-After may be an HTTP date. The loop has its own backoff, so a
        # missing hint is survivable and not worth a date parser here.
        return None
