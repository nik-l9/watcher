"""The Slack Socket Mode transport.

Two properties carry the weight, and both fail *silently* in production if they are wrong: the
envelope must be acknowledged before any work, or Slack redelivers and the same question is
investigated twice; and a routine refresh must reconnect rather than back off, or the service
drifts into a slow loop that looks alive and receives nothing.

Everything here runs against a fake socket. The one thing a fake cannot check — that the token
actually opens a connection — was verified against the live API once, by hand.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from cortex.inbound.socket_mode import (
    SocketModeError,
    open_connection_url,
    run_socket_mode,
)


class _FakeSocket:
    """A socket that yields scripted frames and records what was sent back.

    `done` is set once every frame has been consumed, which is how a test ends the run loop.
    Signalling from the socket rather than counting `is_set()` calls matters: the production loop
    polls `stop` per frame as well as per connection, so a counting stub cuts a socket off
    mid-stream and the test then asserts against events that were never delivered.
    """

    def __init__(self, frames: list[dict[str, Any]], done: asyncio.Event | None = None) -> None:
        self._frames = [json.dumps(f) for f in frames]
        self.sent: list[dict[str, Any]] = []
        self._done = done

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    def __aiter__(self) -> _FakeSocket:
        self._iter = iter(self._frames)
        return self

    async def __anext__(self) -> str:
        try:
            return next(self._iter)
        except StopIteration:
            if self._done is not None:
                self._done.set()
            raise StopAsyncIteration from None


def _connector(sockets: list[_FakeSocket]):
    """A `connect` stand-in handing out each socket in turn, then refusing."""
    remaining = list(sockets)

    @asynccontextmanager
    async def _connect(url: str):
        if not remaining:
            raise ConnectionError("no more sockets")
        yield remaining.pop(0)

    return _connect


def _script(*frame_lists: list[dict[str, Any]]) -> tuple[list[_FakeSocket], asyncio.Event]:
    """Sockets plus the stop event the final one sets when it runs dry."""
    stop = asyncio.Event()
    sockets = [
        _FakeSocket(frames, done=stop if index == len(frame_lists) - 1 else None)
        for index, frames in enumerate(frame_lists)
    ]
    return sockets, stop


def _mention(text: str = "<@U1> why did signups fall?") -> dict[str, Any]:
    return {
        "envelope_id": "env-1",
        "type": "events_api",
        "payload": {
            "team_id": "T0EXAMPLETEAM",
            "event": {"type": "app_mention", "text": text, "channel": "C1", "ts": "1.0"},
        },
    }


class TestOpeningAConnection:
    async def test_a_url_is_returned_on_success(self) -> None:
        def _handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["authorization"] == "Bearer xapp-test"
            return httpx.Response(200, json={"ok": True, "url": "wss://example/link"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
            assert await open_connection_url("xapp-test", client=client) == "wss://example/link"

    async def test_a_refusal_inside_a_200_is_still_a_refusal(self) -> None:
        """Slack answers `ok: false` with HTTP 200, so the status code says nothing on its own.
        This codebase already paid for that lesson on the outbound side."""

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": False, "error": "invalid_auth"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
            with pytest.raises(SocketModeError, match="invalid_auth"):
                await open_connection_url("xapp-bad", client=client)

    async def test_the_token_never_appears_in_the_error(self) -> None:
        """The message is logged and this one is genuinely secret: it is the whole
        authentication for the transport."""

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": False, "error": "invalid_auth"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
            with pytest.raises(SocketModeError) as raised:
                await open_connection_url("xapp-super-secret", client=client)
        assert "xapp-super-secret" not in str(raised.value)

    async def test_a_missing_url_is_refused_rather_than_returned_empty(self) -> None:
        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True})

        async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
            with pytest.raises(SocketModeError, match="no socket url"):
                await open_connection_url("xapp-test", client=client)


class TestAcknowledgementComesFirst:
    """Slack redelivers anything unacknowledged within three seconds; an investigation takes
    ninety. Acknowledging after the work would guarantee a second copy of every question —
    investigated twice, billed twice, answered twice in the same thread."""

    async def test_the_envelope_is_acknowledged_before_the_handler_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "cortex.inbound.socket_mode.open_connection_url",
            _url_stub,
        )
        (socket,), stop = _script([{"type": "hello"}, _mention()])

        seen: list[int] = []

        async def _handle(payload: dict[str, Any]) -> None:
            # How many acks had been sent by the time the work started. Asserting the *order*,
            # not merely that both occurred.
            seen.append(len(socket.sent))

        await run_socket_mode("xapp-test", handle=_handle, connect=_connector([socket]), stop=stop)

        assert socket.sent == [{"envelope_id": "env-1"}]
        assert seen == [1], "the handler ran before the acknowledgement was sent"

    async def test_a_failing_handler_does_not_close_the_socket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One unparseable event must not take down a connection delivering every other event
        correctly."""
        monkeypatch.setattr("cortex.inbound.socket_mode.open_connection_url", _url_stub)
        (socket,), stop = _script([_mention(), {**_mention(), "envelope_id": "env-2"}])

        handled: list[str] = []

        async def _handle(payload: dict[str, Any]) -> None:
            handled.append("called")
            if len(handled) == 1:
                raise RuntimeError("boom")

        await run_socket_mode("xapp-test", handle=_handle, connect=_connector([socket]), stop=stop)

        assert len(handled) == 2, "the second event was dropped after the first one failed"
        assert socket.sent == [{"envelope_id": "env-1"}, {"envelope_id": "env-2"}]

    async def test_a_non_event_frame_is_acknowledged_and_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Slash commands and interactive payloads share the socket. Acknowledged so Slack stops
        asking; not dispatched, because this app subscribes to mentions."""
        monkeypatch.setattr("cortex.inbound.socket_mode.open_connection_url", _url_stub)
        (socket,), stop = _script(
            [{"envelope_id": "env-9", "type": "slash_commands", "payload": {}}]
        )

        handled: list[Any] = []

        async def _handle(payload: dict[str, Any]) -> None:
            handled.append(payload)

        await run_socket_mode("xapp-test", handle=_handle, connect=_connector([socket]), stop=stop)
        assert socket.sent == [{"envelope_id": "env-9"}]
        assert handled == []


class TestReconnection:
    async def test_a_refresh_reconnects_rather_than_erroring(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Slack refreshes these connections routinely and says so first. Treating a planned
        event as a failure would back off when it should reconnect at once, and the service
        would look alive while receiving nothing."""
        monkeypatch.setattr("cortex.inbound.socket_mode.open_connection_url", _url_stub)
        slept: list[float] = []
        monkeypatch.setattr(
            "cortex.inbound.socket_mode._sleep_with_jitter",
            _record_sleep(slept),
        )

        sockets, stop = _script(
            [{"type": "disconnect", "reason": "refresh_requested"}], [_mention()]
        )

        handled: list[Any] = []

        async def _handle(payload: dict[str, Any]) -> None:
            handled.append(payload)

        await run_socket_mode("xapp-test", handle=_handle, connect=_connector(sockets), stop=stop)

        assert len(handled) == 1, "the event on the refreshed connection was never delivered"

    async def test_repeated_failures_back_off_but_stay_bounded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unbounded doubling means a client still asleep an hour after Slack recovered."""
        from cortex.inbound.socket_mode import _BACKOFF_MAX_SECONDS

        async def _always_fails(token: str, *, client: object = None) -> str:
            raise SocketModeError("slack refused a socket connection: server_error")

        monkeypatch.setattr("cortex.inbound.socket_mode.open_connection_url", _always_fails)
        slept: list[float] = []
        monkeypatch.setattr("cortex.inbound.socket_mode._sleep_with_jitter", _record_sleep(slept))

        stop = asyncio.Event()

        async def _handle(payload: dict[str, Any]) -> None:  # pragma: no cover - never called
            raise AssertionError("no event should arrive")

        await run_socket_mode(
            "xapp-test", handle=_handle, connect=_connector([]), stop=_stop_after(stop, after=10)
        )

        assert slept, "a failure must back off rather than spin"
        assert slept == sorted(slept), "backoff must grow"
        assert max(slept) <= _BACKOFF_MAX_SECONDS

    async def test_a_successful_connection_resets_the_backoff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise a long stable session is followed by a needlessly slow first retry — the
        backoff would remember a failure from hours earlier."""
        from cortex.inbound.socket_mode import _BACKOFF_FIRST_SECONDS

        calls = {"n": 0}

        async def _fails_then_works(token: str, *, client: object = None) -> str:
            calls["n"] += 1
            if calls["n"] <= 3:
                raise SocketModeError("slack refused a socket connection: server_error")
            return "wss://example/link"

        monkeypatch.setattr("cortex.inbound.socket_mode.open_connection_url", _fails_then_works)
        slept: list[float] = []
        monkeypatch.setattr("cortex.inbound.socket_mode._sleep_with_jitter", _record_sleep(slept))

        async def _handle(payload: dict[str, Any]) -> None:
            return None

        # Three failures, then a connection that closes normally, then another failure. The
        # delay after the good connection must be the first-retry value again.
        await run_socket_mode(
            "xapp-test",
            handle=_handle,
            connect=_connector([_FakeSocket([])]),
            stop=_stop_after(asyncio.Event(), after=5),
        )

        assert slept[2] > _BACKOFF_FIRST_SECONDS, "backoff should have grown across failures"
        assert slept[3] == pytest.approx(_BACKOFF_FIRST_SECONDS), (
            "a working connection must reset the backoff"
        )


async def _url_stub(token: str, *, client: object = None) -> str:
    return "wss://example/link"


def _record_sleep(sink: list[float]):
    async def _sleep(delay: float, stop: asyncio.Event) -> None:
        sink.append(delay)

    return _sleep


def _stop_after(event: asyncio.Event, after: int = 1) -> asyncio.Event:
    """An event that sets itself once it has been checked `after` times.

    `run_socket_mode` loops until stopped, so a test needs a way to let a bounded number of
    connection attempts happen and then exit. Subclassing keeps the production loop free of test
    hooks.
    """

    class _CountingEvent(asyncio.Event):
        def __init__(self) -> None:
            super().__init__()
            self._checks = 0

        def is_set(self) -> bool:  # type: ignore[override]
            self._checks += 1
            if self._checks > after * 2:
                return True
            return super().is_set()

    return _CountingEvent()
