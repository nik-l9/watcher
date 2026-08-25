"""Progress reporting.

Two properties matter, and only one of them is about the output.

The visible one: a reader should be able to see what the analyst is *looking at*, not only
that it is busy. "calling slack__search_messages" says almost nothing; the query string is
what lets someone notice the analyst is searching for the wrong thing while interrupting is
still cheap.

The load-bearing one: progress is advisory. A reporter that raises must not fail an
investigation, because a window that breaks must not stop the work.
"""

from __future__ import annotations

import io

from cortex.agents.llm import ToolRequest
from cortex.agents.progress import (
    Phase,
    ProgressEvent,
    TerminalProgress,
    emit,
)


class TestEmitIsAlwaysSafe:
    def test_no_sink_is_fine(self) -> None:
        emit(None, ProgressEvent(Phase.THINKING))

    def test_a_sink_that_raises_does_not_propagate(self) -> None:
        """A closed pipe or a broken terminal would otherwise abort an investigation that
        was proceeding perfectly."""

        def _broken(event: ProgressEvent) -> None:
            raise RuntimeError("stderr is gone")

        emit(_broken, ProgressEvent(Phase.THINKING))

    def test_events_reach_a_working_sink(self) -> None:
        seen: list[ProgressEvent] = []
        emit(seen.append, ProgressEvent(Phase.CALLING, detail="github__commits", step=2))
        assert seen[0].phase is Phase.CALLING
        assert seen[0].step == 2


class TestTerminalRendering:
    def _render(self, *events: ProgressEvent) -> str:
        stream = io.StringIO()
        ticks = iter([0.0, 1.5, 3.0, 4.5, 6.0, 7.5])
        reporter = TerminalProgress(stream=stream, clock=lambda: next(ticks))
        for event in events:
            reporter(event)
        return stream.getvalue()

    def test_each_phase_reads_as_a_sentence(self) -> None:
        output = self._render(
            ProgressEvent(Phase.THINKING, step=1),
            ProgressEvent(Phase.CALLING, detail="github__commits repo='acme/web'", step=1),
            ProgressEvent(Phase.OBSERVED, detail="github__commits", step=1, observations=1),
        )
        assert "deciding what to look at" in output
        assert "calling github__commits repo='acme/web'" in output
        assert "1 observation(s) so far" in output

    def test_elapsed_time_is_shown(self) -> None:
        """The number that distinguishes "slow" from "hung"."""
        output = self._render(ProgressEvent(Phase.THINKING, step=1))
        assert "1.5s" in output

    def test_a_failed_tool_is_marked(self) -> None:
        output = self._render(
            ProgressEvent(Phase.TOOL_FAILED, detail="github__deployment_history failed", step=2)
        )
        assert "!" in output
        assert "deployment_history" in output

    def test_the_step_number_is_shown_when_there_is_one(self) -> None:
        assert "step 3" in self._render(ProgressEvent(Phase.THINKING, step=3))
        # Phases outside the loop have no step, and must not print "step None".
        assert "None" not in self._render(ProgressEvent(Phase.GATING))


class TestDescribingACall:
    """The arguments are the interesting part, not the capability name."""

    def test_the_query_is_shown(self) -> None:
        from cortex.agents.investigator import _describe_call

        described = _describe_call(
            ToolRequest(
                id="t1",
                name="slack__search_messages",
                arguments={"query": "Apollo deanonymizer", "limit": 20},
            )
        )
        assert "slack__search_messages" in described
        assert "Apollo deanonymizer" in described
        # `limit` is noise: it says nothing about what is being looked for.
        assert "limit" not in described

    def test_a_long_argument_is_truncated(self) -> None:
        from cortex.agents.investigator import _describe_call

        described = _describe_call(
            ToolRequest(id="t1", name="sql__query", arguments={"sql": "SELECT " + "x" * 200})
        )
        assert len(described) < 120
        assert "..." in described

    def test_a_list_argument_is_readable(self) -> None:
        from cortex.agents.investigator import _describe_call

        described = _describe_call(
            ToolRequest(
                id="t1",
                name="posthog__funnel",
                arguments={"steps": ["$pageview", "user signed up"]},
            )
        )
        assert "user signed up" in described

    def test_a_call_with_nothing_interesting_still_names_itself(self) -> None:
        from cortex.agents.investigator import _describe_call

        described = _describe_call(
            ToolRequest(id="t1", name="posthog__list_projects", arguments={})
        )
        assert described == "posthog__list_projects"


class TestPublishingForTheUi:
    """The second renderer over the same events.

    `InvestigationProgress` has been in the contracts since M0 and nothing ever sent one, so
    the gateway's streaming endpoint had nothing to stream.
    """

    class _Producer:
        """A stand-in for the Celery app.

        `**kwargs` rather than a narrow signature: the first version named exactly the
        arguments the producer happened to pass, so adding `expires` broke three tests that
        were not about `expires` at all. A fake that mirrors the real API's shape does not
        have to be edited every time the caller learns something.
        """

        def __init__(self) -> None:
            self.sent: list[tuple[str, dict, str]] = []

        def send_task(self, name: str, *, args: list, queue: str, **kwargs: object) -> None:
            self.sent.append((name, args[0], queue))

    def _sink(self) -> tuple[_Producer, object]:
        import uuid

        from cortex.agents.progress import QueueProgress

        producer = self._Producer()
        return producer, QueueProgress(
            producer=producer,
            tenant_id=uuid.uuid4(),
            investigation_id=uuid.uuid4(),
        )

    def test_an_event_becomes_a_contract_message_on_the_events_queue(self) -> None:
        from cortex.contracts.messages import QUEUE_EVENTS, InvestigationProgress

        producer, sink = self._sink()
        sink(ProgressEvent(Phase.CALLING, detail="github__commits", step=2))  # type: ignore[operator]

        assert len(producer.sent) == 1
        name, payload, queue = producer.sent[0]
        assert name == "cortex.events.progress"
        assert queue == QUEUE_EVENTS
        # Validates against the contract, so a producer cannot drift from its consumer.
        message = InvestigationProgress.model_validate(payload)
        assert message.status == "calling"
        assert message.step == 2

    def test_a_repeated_phase_within_a_step_is_dropped(self) -> None:
        """A loop emits several events per second; a UI needs the shape of what is
        happening, not every keystroke."""
        producer, sink = self._sink()
        sink(ProgressEvent(Phase.CALLING, detail="a", step=1))  # type: ignore[operator]
        sink(ProgressEvent(Phase.CALLING, detail="b", step=1))  # type: ignore[operator]
        sink(ProgressEvent(Phase.OBSERVED, detail="a", step=1))  # type: ignore[operator]
        sink(ProgressEvent(Phase.CALLING, detail="c", step=2))  # type: ignore[operator]

        statuses = [(p["status"], p["step"]) for _, p in ((n, p) for n, p, _ in producer.sent)]
        assert statuses == [("calling", 1), ("observed", 1), ("calling", 2)]

    def test_phases_outside_the_loop_keep_the_last_step(self) -> None:
        """Drafting and verifying have no step of their own. Publishing step 0 would make a
        UI jump back to the start at the moment the analyst is finishing."""
        producer, sink = self._sink()
        sink(ProgressEvent(Phase.CALLING, detail="a", step=3))  # type: ignore[operator]
        sink(ProgressEvent(Phase.DRAFTING, observations=7))  # type: ignore[operator]

        assert producer.sent[-1][1]["step"] == 3

    def test_a_broken_producer_does_not_reach_the_investigation(self) -> None:
        """Published through `emit`, so a dead broker costs a frame rather than a report."""
        import uuid

        from cortex.agents.progress import QueueProgress

        class _Dead:
            def send_task(self, *args: object, **kwargs: object) -> None:
                raise RuntimeError("broker unreachable")

        sink = QueueProgress(
            producer=_Dead(), tenant_id=uuid.uuid4(), investigation_id=uuid.uuid4()
        )
        emit(sink, ProgressEvent(Phase.THINKING, step=1))

    def test_the_correlation_id_is_carried_when_given(self) -> None:
        """Propagated across every hop so one investigation can be traced end to end."""
        import uuid

        from cortex.agents.progress import QueueProgress

        correlation = uuid.uuid4()
        producer = self._Producer()
        sink = QueueProgress(
            producer=producer,
            tenant_id=uuid.uuid4(),
            investigation_id=uuid.uuid4(),
            correlation_id=correlation,
        )
        sink(ProgressEvent(Phase.THINKING, step=1))

        assert producer.sent[0][1]["correlation_id"] == str(correlation)

    def test_messages_expire_so_an_unconsumed_queue_cannot_grow_forever(self) -> None:
        """Nothing consumes this queue yet — the UI is its consumer and does not exist.
        Without a TTL every investigation would leave a permanent pile of messages in
        Redis: a slow leak nobody notices until the broker fills."""
        import uuid

        from cortex.agents.progress import EVENT_TTL_SECONDS, QueueProgress

        captured: dict[str, object] = {}

        class _Producer:
            def send_task(self, name: str, **kwargs: object) -> None:
                captured.update(kwargs)

        sink = QueueProgress(
            producer=_Producer(), tenant_id=uuid.uuid4(), investigation_id=uuid.uuid4()
        )
        sink(ProgressEvent(Phase.THINKING, step=1))

        assert captured["expires"] == EVENT_TTL_SECONDS


class TestTheSequenceAConsumerDedupsOn:
    """Why a sequence exists at all.

    The original design was for a terminal: it prints each line as it arrives and never
    reconnects, so an unnumbered stream is fine. A UI is a different consumer — it drops out,
    comes back, and receives some frames twice while missing others. Indexing the OpenHands
    frontend surfaced their handling of exactly this: their store dedups by event id and
    skips side-effects for events already seen, with a comment citing their own issue about a
    reconnect replaying a backlog from a stale anchor.

    Without a sequence, two identical "step 5 calling github__commits" frames are
    indistinguishable from one call reported twice, and a missing frame is invisible.
    """

    def _publish(self, events: list[ProgressEvent]) -> list[int]:
        import uuid

        from cortex.agents.progress import QueueProgress

        producer = TestPublishingForTheUi._Producer()
        sink = QueueProgress(
            producer=producer,
            tenant_id=uuid.uuid4(),
            investigation_id=uuid.uuid4(),
        )
        for event in events:
            sink(event)
        return [payload["sequence"] for _, payload, _ in producer.sent]

    def test_it_starts_at_one(self) -> None:
        """Zero is the field's default, so a first message numbered 0 would be
        indistinguishable from a producer that never set it."""
        assert self._publish([ProgressEvent(Phase.CALLING, detail="a", step=1)]) == [1]

    def test_it_increases_by_one_per_published_message(self) -> None:
        sequences = self._publish(
            [
                ProgressEvent(Phase.CALLING, detail="a", step=1),
                ProgressEvent(Phase.OBSERVED, detail="a", step=1),
                ProgressEvent(Phase.CALLING, detail="b", step=2),
            ]
        )
        assert sequences == [1, 2, 3]

    def test_a_suppressed_duplicate_does_not_consume_a_number(self) -> None:
        """The load-bearing detail. Counting received events rather than published ones would
        leave a hole for every burst the duplicate filter collapses — and a hole is supposed
        to mean a consumer lost something."""
        sequences = self._publish(
            [
                ProgressEvent(Phase.CALLING, detail="a", step=1),
                ProgressEvent(Phase.CALLING, detail="b", step=1),
                ProgressEvent(Phase.CALLING, detail="c", step=1),
                ProgressEvent(Phase.OBSERVED, detail="a", step=1),
            ]
        )
        assert sequences == [1, 2]

    def test_a_consumer_can_dedup_on_investigation_and_sequence(self) -> None:
        """The property a UI actually needs: the same frame delivered twice collapses, and
        two genuinely different frames do not."""
        import uuid

        from cortex.agents.progress import QueueProgress

        producer = TestPublishingForTheUi._Producer()
        investigation_id = uuid.uuid4()
        sink = QueueProgress(
            producer=producer,
            tenant_id=uuid.uuid4(),
            investigation_id=investigation_id,
        )
        sink(ProgressEvent(Phase.CALLING, detail="github__commits", step=5))
        sink(ProgressEvent(Phase.OBSERVED, detail="github__commits", step=5))

        delivered = [payload for _, payload, _ in producer.sent]
        # A replay: the same two frames arrive again after a reconnect.
        seen: set[tuple[str, int]] = set()
        kept = []
        for payload in delivered + delivered:
            key = (payload["investigation_id"], payload["sequence"])
            if key in seen:
                continue
            seen.add(key)
            kept.append(payload)
        assert len(kept) == 2
