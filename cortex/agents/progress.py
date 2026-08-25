"""Progress reporting for a running investigation.

A 90-second investigation that prints nothing is indistinguishable from a hung one. That is
not only a comfort problem: during development it has already cost real time — an unbounded
provider call went unnoticed because silence looked normal, and a Slack search that returned
nothing looked identical to a search that never ran.

**Advisory, never load-bearing.** Losing a progress event must not affect an investigation,
and a reporter that raises must not fail one. The authoritative record is the Postgres row
and the evidence store; this is a window onto them.

**One event type, two renderers.** The CLI writes lines to stderr, so stdout stays the report
and stays pipeable. The worker publishes the same events as `InvestigationProgress` messages
for the UI to stream. Both read the same events, so the terminal and the web view cannot
drift into describing the loop differently — there is only one description.
"""

from __future__ import annotations

import enum
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TextIO

from cortex.contracts.messages import QUEUE_EVENTS, InvestigationProgress


class Phase(enum.StrEnum):
    """What the loop is doing. Named for what a *reader* wants to know, not for the
    internal call being made."""

    RECALLING = "recalling"
    #: Establishing what exists -- repositories, projects, events -- before investigating.
    SURVEYING = "surveying"
    THINKING = "thinking"
    CALLING = "calling"
    OBSERVED = "observed"
    TOOL_FAILED = "tool_failed"
    DRAFTING = "drafting"
    GATING = "gating"
    VERIFYING = "verifying"
    DONE = "done"


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    phase: Phase
    #: One line a human reads. Never a stack trace, never a raw payload.
    detail: str = ""
    #: 1-based loop step, where the phase belongs to one.
    step: int | None = None
    #: Evidence rows gathered so far, so a reader can see the investigation accumulating.
    observations: int = 0


#: How long a progress message stays useful.
#:
#: Sixty seconds. It describes what the analyst is doing *now*; a UI that reconnects wants
#: current state, which it reads from the investigation row, not a replay of stale frames.
EVENT_TTL_SECONDS = 60

#: Anything that accepts events. Deliberately a plain callable rather than an interface, so
#: a test can pass `events.append` and a caller can pass a lambda.
ProgressSink = Callable[[ProgressEvent], None]


def emit(sink: ProgressSink | None, event: ProgressEvent) -> None:
    """Send one event, swallowing anything the sink does wrong.

    A progress reporter is a window, and a window that breaks must not stop the work. An
    exception from a renderer — a closed pipe, a broken terminal — would otherwise abort an
    investigation that was proceeding perfectly.
    """
    if sink is None:
        return
    try:
        sink(event)
    except Exception:  # noqa: BLE001 - see the docstring
        return


@dataclass(slots=True)
class TerminalProgress:
    """Renders events as timestamped lines on stderr.

    Lines rather than an overwritten status line: the output survives being piped to a file,
    and the sequence of calls is itself the most useful thing to read afterwards when an
    investigation went somewhere unexpected.
    """

    stream: TextIO = field(default_factory=lambda: sys.stderr)
    clock: Callable[[], float] = time.monotonic
    started: float = field(default=0.0)
    #: Tools already reported this step, so two calls to the same capability in one turn do
    #: not print twice.
    _seen: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.started = self.clock()

    def __call__(self, event: ProgressEvent) -> None:
        line = self._format(event)
        if line is None:
            return
        elapsed = self.clock() - self.started
        step = f"step {event.step}" if event.step else "      "
        self.stream.write(f"  [{elapsed:5.1f}s] {step:8} {line}\n")
        # Flushed on every line: a buffered progress report is no progress report, and
        # stderr is line-buffered only when attached to a terminal.
        self.stream.flush()

    def _format(self, event: ProgressEvent) -> str | None:
        match event.phase:
            case Phase.RECALLING:
                return "checking memory for prior context"
            case Phase.SURVEYING:
                return f"found {event.detail}" if event.detail else "checking what data exists"
            case Phase.THINKING:
                return "deciding what to look at"
            case Phase.CALLING:
                return f"calling {event.detail}"
            case Phase.OBSERVED:
                return f"got {event.detail} ({event.observations} observation(s) so far)"
            case Phase.TOOL_FAILED:
                return f"! {event.detail}"
            case Phase.DRAFTING:
                return f"drafting the report from {event.observations} observation(s)"
            case Phase.GATING:
                return "checking every citation resolves"
            case Phase.VERIFYING:
                return "re-reading each claim against its own evidence"
            case Phase.DONE:
                return event.detail or "done"
        return None  # pragma: no cover - the match is exhaustive over Phase


@dataclass(slots=True)
class QueueProgress:
    """Publishes events as `InvestigationProgress` messages for the UI to stream.

    The contract for these has existed since M0 and nothing ever sent one, so the
    gateway's "streaming steps" endpoint had nothing to stream. This is the second renderer
    over the same events the terminal uses, which is the point of having one event type: the
    web view and the CLI cannot drift into describing the loop differently, because there is
    only one description.

    **Advisory, and rate-limited by construction.** A loop can emit several events per
    second; a UI needs to know the shape of what is happening, not every keystroke. Only
    phases that change what a reader would say the analyst is *doing* are published, and a
    repeated phase is dropped.
    """

    #: Something with `.send_task(name, args=[...], queue=...)` — a Celery app, or a test's
    #: recorder. Injected rather than imported so this class needs no broker to test.
    producer: object
    tenant_id: uuid.UUID
    investigation_id: uuid.UUID
    correlation_id: uuid.UUID | None = None
    #: The last phase published, so a burst of identical phases becomes one message.
    _last: str = ""
    _step: int = 0
    #: Published messages so far. The next one is `_sequence + 1`.
    _sequence: int = 0

    def __call__(self, event: ProgressEvent) -> None:
        # OBSERVED and CALLING both mean "working through step N". Publishing both doubles
        # the traffic and tells a reader nothing new, so the step number is the unit.
        key = f"{event.phase}:{event.step}"
        if key == self._last:
            return
        self._last = key
        self._step = event.step or self._step
        # Counted on *published* messages, not on events received. A sequence that skipped
        # the numbers belonging to suppressed duplicates would show a consumer a gap for
        # every burst the filter above collapsed — and a gap is supposed to mean something
        # was lost.
        self._sequence += 1

        message = InvestigationProgress(
            tenant_id=self.tenant_id,
            investigation_id=self.investigation_id,
            status=event.phase.value,
            step=self._step,
            sequence=self._sequence,
            note=event.detail[:500] or None,
            **({"correlation_id": self.correlation_id} if self.correlation_id else {}),
        )
        self.producer.send_task(  # type: ignore[attr-defined]
            "cortex.events.progress",
            args=[message.model_dump(mode="json")],
            queue=QUEUE_EVENTS,
            # Expires, because nothing consumes this queue yet — the UI is its consumer and
            # does not exist. Without a TTL every investigation would leave a permanent pile
            # of messages in Redis, which is a slow leak nobody would notice until the
            # broker filled. An expiry is also correct on its own terms: a progress event
            # describes a moment, and a minute later it is of no use to anyone.
            expires=EVENT_TTL_SECONDS,
        )
