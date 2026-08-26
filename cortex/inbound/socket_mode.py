"""Slack Socket Mode: receiving events over an outbound WebSocket instead of a webhook.

## Why this exists

The HTTP transport works and has one requirement that turns out to be expensive: Slack must be
able to reach the gateway, which means a public HTTPS endpoint. During this project's trial that
was a tunnel, and a tunnel URL changes every time it restarts while Slack keeps posting to the old
one. The failure is silent — the gateway is healthy, the signing secret is present, the app is
installed, and no event ever arrives. It happened, and the only symptom was that nobody got an
answer.

Socket Mode removes the requirement rather than working around it. The client dials *out* to
Slack and holds the connection open, so Cortex needs no inbound network path at all. For anybody
running this against their own API keys, that is the difference between `docker compose up` and
acquiring a domain and a certificate.

## Authentication differs, and pretending otherwise would be worse than the gap

The webhook authenticates every delivery with an HMAC over the raw body, because anyone on the
internet can POST to it. Socket Mode has no per-message signature: the socket is opened with an
app-level token over TLS to Slack's own host, and nothing else can write to it. That is genuinely
sufficient, and adding a signature check here would be theatre — there is no field to check.

What follows from it is a real operational difference, recorded because it will matter to whoever
deploys this: **the app-level token is the entire authentication.** Anyone holding it can receive
this workspace's events. It is stored in the environment like the other secrets, never logged, and
never included in an error message.

## The three-second rule, and why acknowledgement comes first

Slack expects an acknowledgement within three seconds and redelivers otherwise. An investigation
takes ninety. So each envelope is acknowledged the moment it is understood, before any database
work — the same shape the webhook uses when it returns 200 and lets a worker deliver later. Doing
the work first would guarantee redelivery, and redelivery means the same question investigated
twice and billed twice.

## Reconnection is normal, not exceptional

Slack refreshes these connections routinely and sends a `disconnect` frame with `reason:
"refresh_requested"` before doing so. A client that treats disconnection as an error logs noise
and, worse, may back off when it should reconnect immediately. So a refresh is expected and
reconnects at once; only repeated *failures* to open a connection back off, and they back off with
a ceiling, because a Slack outage must not turn into a client that gives up permanently.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import structlog
import websockets

from cortex.tools.http import DEFAULT_TIMEOUT

log = structlog.get_logger(__name__)

#: Where a Socket Mode connection is requested. Returns a short-lived `wss://` URL.
CONNECTIONS_OPEN = "https://slack.com/api/apps.connections.open"

#: Backoff bounds for *failing* to connect. A ceiling matters more than the growth rate: an
#: unbounded doubling means a client that is still asleep an hour after Slack recovered.
_BACKOFF_FIRST_SECONDS = 1.0
_BACKOFF_MAX_SECONDS = 60.0

#: Jitter, so a fleet of clients that all lost the same connection do not retry in lockstep and
#: reproduce the outage against Slack's own endpoint.
_BACKOFF_JITTER = 0.25


class SocketModeError(RuntimeError):
    """Opening a connection failed in a way retrying will not fix — a rejected token."""


async def open_connection_url(token: str, *, client: httpx.AsyncClient | None = None) -> str:
    """Ask Slack for a WebSocket URL.

    Raises `SocketModeError` with Slack's own error string, which never contains the token and is
    the difference between "fix your token" and a generic failure. `invalid_auth` here is the one
    failure worth stopping on rather than retrying.
    """
    owned = client is None
    http = client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
    try:
        response = await http.post(CONNECTIONS_OPEN, headers={"Authorization": f"Bearer {token}"})
        # Slack answers `ok: false` inside an HTTP 200, so the status code says nothing on its
        # own -- a lesson this codebase already paid for on the outbound side.
        body = response.json() if response.content else {}
    finally:
        if owned:
            await http.aclose()

    if not isinstance(body, dict) or not body.get("ok"):
        error = (body or {}).get("error", f"http {response.status_code}")
        raise SocketModeError(f"slack refused a socket connection: {error}")
    url = body.get("url")
    if not isinstance(url, str) or not url:
        raise SocketModeError("slack returned no socket url")
    return url


@asynccontextmanager
async def _connect(url: str) -> AsyncIterator[Any]:
    async with websockets.connect(url, max_queue=32) as socket:
        yield socket


async def run_socket_mode(
    token: str,
    *,
    handle: Callable[[dict[str, Any]], Awaitable[None]],
    connect: Callable[[str], Any] = _connect,
    stop: asyncio.Event | None = None,
) -> None:
    """Hold a Socket Mode connection open and dispatch events until asked to stop.

    `handle` receives the *payload* of an events envelope, already acknowledged. It is awaited,
    but a failure inside it is logged rather than raised: one unparseable event must not close a
    connection that is delivering every other event correctly.

    `connect` is injected so the reconnection policy can be tested without a network.
    """
    stop = stop or asyncio.Event()
    delay = _BACKOFF_FIRST_SECONDS

    while not stop.is_set():
        try:
            url = await open_connection_url(token)
            async with connect(url) as socket:
                # A connection that opened is a working connection: reset the backoff so a long
                # stable session is not followed by a needlessly slow first retry.
                delay = _BACKOFF_FIRST_SECONDS
                log.info("slack.socket.connected")
                await _pump(socket, handle=handle, stop=stop)
        except SocketModeError as exc:
            # A rejected token will be rejected again. Still retried, but loudly and slowly:
            # stopping the process would take the service down for a fixable config error, and
            # an operator who fixes the token should not have to restart anything.
            log.error("slack.socket.refused", reason=str(exc))
        except Exception as exc:  # noqa: BLE001 - a transport is allowed to fail; see docstring
            log.warning("slack.socket.dropped", error=type(exc).__name__, detail=str(exc)[:200])

        if stop.is_set():
            break
        await _sleep_with_jitter(delay, stop)
        delay = min(delay * 2, _BACKOFF_MAX_SECONDS)


async def _pump(
    socket: Any, *, handle: Callable[[dict[str, Any]], Awaitable[None]], stop: asyncio.Event
) -> None:
    """Read frames until the socket closes or a disconnect is requested."""
    async for raw in socket:
        if stop.is_set():
            return
        try:
            frame = json.loads(raw)
        except ValueError:
            log.warning("slack.socket.unparseable_frame")
            continue
        if not isinstance(frame, dict):
            continue

        kind = frame.get("type")
        if kind == "hello":
            # Logged rather than skipped silently, because "connected" and "receiving events" are
            # different states and only this frame distinguishes them. A socket can be open and
            # idle forever when Socket Mode is toggled off in the app settings: Slack still hands
            # out a URL to any app-level token holding `connections:write`, so opening a
            # connection proves nothing about where events are being delivered. `num_connections`
            # is the other half -- more than one means a stray client is competing for events,
            # and each event goes to exactly one of them.
            info = frame.get("connection_info")
            log.info(
                "slack.socket.hello",
                app_id=(info or {}).get("app_id"),
                num_connections=frame.get("num_connections"),
            )
            continue
        if kind == "disconnect":
            # Routine. Slack refreshes connections on a schedule and says so first; returning
            # here reconnects immediately rather than treating a planned event as a failure.
            log.info("slack.socket.refresh", reason=frame.get("reason"))
            return

        envelope_id = frame.get("envelope_id")
        if envelope_id:
            # **Before any work.** Slack redelivers anything unacknowledged within three seconds
            # and an investigation takes ninety, so acknowledging last would guarantee a second
            # copy of every question -- investigated twice, billed twice, answered twice in the
            # same thread.
            await socket.send(json.dumps({"envelope_id": envelope_id}))

        if kind != "events_api":
            # `slash_commands` and `interactive` arrive on the same socket. Acknowledged above so
            # Slack stops asking, and otherwise ignored: this app subscribes to mentions.
            continue

        payload = frame.get("payload")
        if not isinstance(payload, dict):
            continue
        try:
            await handle(payload)
        except Exception as exc:  # noqa: BLE001 - see the docstring
            # One bad event must not close a socket that is delivering the rest correctly.
            log.error(
                "slack.socket.handler_failed",
                error=type(exc).__name__,
                detail=str(exc)[:200],
            )


async def _sleep_with_jitter(delay: float, stop: asyncio.Event) -> None:
    """Wait, but wake immediately if asked to stop, so shutdown is not held up by a backoff."""
    jittered = delay * (1 + random.uniform(-_BACKOFF_JITTER, _BACKOFF_JITTER))
    try:
        await asyncio.wait_for(stop.wait(), timeout=max(jittered, 0.0))
    except TimeoutError:
        return
