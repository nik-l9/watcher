"""Retrieved-but-unread: the digest, and the ledger that can enumerate what was ignored.

The bug being closed here (ADR 0005 decision 8): `posthog.list_events` was fetched, contained
the answer -- thirteen server-side event types whose last activity fell inside one 72-minute
window while browser autocapture kept flowing -- and was never read. The analyst explained the
signup cessation with a pageview collapse that started seven days later.

So the first class below is the one that matters, and it is deliberately adversarial about the
obvious implementation. A *head* of that payload would have made things worse: PostHog returns
event definitions newest-activity-first, so the thirteen events that stopped are the last rows
in the payload, and any prefix of it contains none of them. A summary that could hide the
answer is our original bug wearing a summary's clothes.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from cortex.agents.reading import (
    BULK_CHARS,
    BULK_ROWS,
    READ_TOOL,
    ReadingLedger,
    UnknownHandle,
    digest_payload,
    read_tool_spec,
    render_payload,
)
from cortex.tools.base import Freshness
from cortex.tools.executor import ExecutedTool

#: The thirteen server-side product events whose `last_seen_at` all fell inside a 72-minute
#: window on 2026-08-04. Named from the ADR's IS/IS-NOT matrix.
STOPPED = [
    "signup_completed",
    "signup_started",
    "trial_started",
    "checkout_started",
    "checkout_completed",
    "invite_sent",
    "invite_accepted",
    "subscription_created",
    "subscription_cancelled",
    "workspace_created",
    "api_key_created",
    "seat_added",
    "onboarding_finished",
]


def _event_catalogue() -> dict[str, object]:
    """A PostHog event catalogue shaped like the one that contained the answer.

    Ordered as PostHog orders it -- newest activity first -- so the thirteen events that
    stopped are at the *end*. That ordering is the whole point: it is what makes a head of
    this payload actively misleading rather than merely incomplete.
    """
    events = [
        {
            "name": f"$autocapture_{i}" if i % 3 else f"$pageview_{i}",
            "description": None,
            "last_seen_at": f"2026-08-2{i % 3}T{i % 24:02d}:{i % 60:02d}:00Z",
            "verified": False,
        }
        for i in range(87)
    ]
    events += [
        {
            "name": name,
            "description": f"Server-side {name}.",
            # Real arithmetic, because `position * 5` produced "11:60:00Z" for the thirteenth
            # event -- an invalid timestamp in the fixture that stands for the actual incident.
            # Nothing here parses it today, which is exactly why it would have survived until
            # something did.
            "last_seen_at": (
                (
                    datetime(2026, 8, 4, 11, 0, tzinfo=UTC) + timedelta(minutes=position * 6)
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
            ),
            "verified": True,
        }
        for position, name in enumerate(STOPPED)
    ]
    return {
        "event_count": len(events),
        "total_available": len(events),
        "excluded_stale": False,
        "events": events,
    }


def _executed(payload: dict[str, object], capability: str = "list_events") -> ExecutedTool:
    return ExecutedTool(
        evidence_id=uuid.uuid4(),
        tool_name="posthog",
        capability=capability,
        payload=payload,
        source_ref="posthog://projects/1/event_definitions",
        freshness=Freshness.LIVE,
        payload_hash="deadbeef",
        duration_ms=42,
    )


class TestTheDigestIsNotAHead:
    def test_the_cluster_a_head_would_have_hidden_is_named(self) -> None:
        """The decisive fact, present in the digest without reading a single row.

        Thirteen events whose last activity is the earliest day in the column. That is the
        shape "a group of things stopped together" takes, and it is what the analyst missed.
        """
        digest = digest_payload(_event_catalogue())

        assert digest is not None
        assert "2026-08-04 x13" in digest.text
        assert digest.rows == 100

    def test_a_head_of_the_same_payload_contains_none_of_it(self) -> None:
        """The counterfactual, asserted rather than assumed.

        The first `BULK_ROWS` rows of this payload -- what an OBSK implementation that showed
        a literal head would have displayed -- mention none of the thirteen. Showing a prefix
        would have reproduced the original failure with a new mechanism to blame it on.
        """
        rows = _event_catalogue()["events"]
        assert isinstance(rows, list)
        head = render_payload(rows[:BULK_ROWS])

        assert not any(name in head for name in STOPPED)
        assert "2026-08-04" not in head

    def test_the_digest_does_not_depend_on_the_order_the_rows_arrived_in(self) -> None:
        """Reversed rows, identical digest.

        An ordering-sensitive summary is a summary whose content is decided by the upstream's
        sort order, which is not a property anyone reviews. Every fact in the digest is a
        count, a range or an extreme, so this holds by construction -- and this test is what
        keeps it holding the next time a field type is added.
        """
        forward = _event_catalogue()
        backward = _event_catalogue()
        rows = backward["events"]
        assert isinstance(rows, list)
        rows.reverse()

        first, second = digest_payload(forward), digest_payload(backward)
        assert first is not None and second is not None
        assert first.text == second.text

    def test_it_is_much_smaller_than_the_payload_it_replaces(self) -> None:
        """The reason for doing this at all. 14.7k characters of catalogue become under 1k."""
        digest = digest_payload(_event_catalogue())

        assert digest is not None
        assert digest.full_chars > 14_000
        assert digest.chars < digest.full_chars / 10

    def test_the_connectors_own_disclosures_are_never_behind_the_handle(self) -> None:
        """The last four bug fixes put their resolution in the payload's scalar keys:
        `series_ends_early`, `blast_radius`, `partial_buckets`, `total_available`. Hiding one
        of those behind a handle in order to ship this mechanism would undo a shipped fix, so
        only row collections are ever replaced -- everything else is rendered verbatim."""
        payload = dict(_event_catalogue())
        payload["series_ends_early"] = {"last_bucket": "2026-08-04", "missing_days": 18}
        payload["blast_radius"] = {"stopped_together": 13, "still_flowing": 87}

        digest = digest_payload(payload)

        assert digest is not None
        assert '"missing_days": 18' in digest.text
        assert '"stopped_together": 13' in digest.text
        assert '"total_available": 100' in digest.text


class TestWhenItApplies:
    """The rule is mechanical -- rendered size, and a list long enough to be rows -- because a
    model judging which of its own observations to hide is the judgement this whole mechanism
    exists to stop trusting."""

    def test_a_small_payload_is_untouched(self) -> None:
        assert digest_payload({"rows": [{"sessions": 1200, "conversion": 0.029}]}) is None

    def test_a_daily_series_is_read_row_by_row_not_digested(self) -> None:
        """The threshold sits deliberately above a daily series.

        A 60-day series renders at ~3.7k characters and the analyst has to read every bucket
        to measure anything -- which day it fell, by how much, whether it recovered. A digest
        of it would be a loss with no saving. The eval suite's largest fixture payload is
        5,390 characters for exactly this shape, so the suite cannot measure this change at
        all; that is stated in the module docstring rather than left to be discovered.
        """
        series = {
            "series": [
                {"date": f"2026-06-{i + 1:02d}", "sessions": 1000 - i * 7} for i in range(60)
            ]
        }

        assert len(render_payload(series)) < BULK_CHARS
        assert digest_payload(series) is None

    def test_a_large_payload_with_no_row_collection_is_shown_whole(self) -> None:
        """There is nothing a digest could honestly replace. A wall of prose stays a wall of
        prose rather than being summarised by something that cannot read it."""
        payload = {"body": "x" * (BULK_CHARS + 1000), "note": "one long document"}

        assert len(render_payload(payload)) > BULK_CHARS
        assert digest_payload(payload) is None

    def test_a_short_list_of_wide_rows_is_not_a_row_collection(self) -> None:
        """Twenty-four rows or fewer are read, not digested: a month of daily buckets, a
        week of hourly ones, a funnel's steps. Size alone must not turn ten rows into a
        digest just because each one is verbose."""
        payload = {"steps": [{"name": "step", "detail": "y" * 900} for _ in range(BULK_ROWS)]}

        assert len(render_payload(payload)) > BULK_CHARS
        assert digest_payload(payload) is None

    def test_a_digest_is_always_smaller_than_what_it_replaces(self) -> None:
        """The invariant behind the mechanism: it can never cost more context than the rows.

        Guarded in the code rather than argued for, and checked here across shapes -- rows of
        ids, rows of numbers, rows of long unique strings -- because a digest that grew would
        make every large observation worse while looking like an optimisation.
        """
        shapes: list[dict[str, object]] = [
            {"rows": [{"id": str(uuid.uuid4()), "n": i} for i in range(200)]},
            {"rows": [{"text": "word " * 40, "i": i} for i in range(60)]},
            {"rows": [[i, i * 2, f"label-{i}"] for i in range(400)]},
        ]
        for payload in shapes:
            digest = digest_payload(payload)
            assert digest is not None, payload.keys()
            assert digest.chars < digest.full_chars


class TestWhatOneFieldSays:
    def test_a_numeric_column_counts_its_zeros(self) -> None:
        """ "How many of these are zero" is the question behind most of the wrong answers this
        codebase has shipped: a series that fell to zero and a series that stopped being
        collected are identical in a min/max, and distinguishable in a count of zeros.

        The rows carry a text field purely to push the payload past `BULK_CHARS`. Written that
        way after the first version of this test used a bare 60-row daily series and got no
        digest at all -- correctly, because the threshold is deliberately set *above* a daily
        series, so that the analyst reads one row by row rather than being handed a summary of
        it. The test had contradicted the design it was checking.
        """
        payload = {
            "series": [
                {"day": i, "signups": 0 if i > 40 else 90 - i, "note": "padding " * 20}
                for i in range(60)
            ]
        }

        digest = digest_payload(payload)
        assert digest is not None, "the payload must be bulky enough to digest"
        assert "19 are zero" in digest.text

    def test_a_column_of_ordinary_strings_is_not_described_as_a_clock(self) -> None:
        """The temporal branch is what surfaces a cluster of dates, so it must not fire on a
        column of names -- a digest that reported the "earliest" repository name would be
        confidently meaningless."""
        payload = {"repos": [{"repo": f"acme/service-{i}", "note": "x" * 200} for i in range(50)]}

        digest = digest_payload(payload)
        assert digest is not None
        assert "timestamp" not in digest.text

    def test_a_low_cardinality_column_is_enumerated_completely(self) -> None:
        """Six distinct values or fewer come back with every value and its count. This is the
        cheapest possible answer to "which segment is in here", and it needs no read."""
        payload = {
            "rows": [
                {"plan": ["free", "pro", "enterprise"][i % 3], "blob": "z" * 200} for i in range(60)
            ]
        }

        digest = digest_payload(payload)
        assert digest is not None
        assert "'free' x20" in digest.text
        assert "'enterprise' x20" in digest.text


class TestTheLedgerEnumeratesWhatWasNotRead:
    """`unread()` is the point of the module. An agent that cannot enumerate what it has
    fetched cannot notice what it has ignored."""

    def test_a_summarised_observation_is_unread_until_it_is_read(self) -> None:
        ledger = ReadingLedger()
        executed = _executed(_event_catalogue())

        assert ledger.offer(executed) is not None
        unread = ledger.unread()
        assert [item.evidence_id for item in unread] == [executed.evidence_id]
        assert unread[0].source == "posthog.list_events"
        assert unread[0].rows == 100
        assert ledger.read_count == 0

    def test_reading_it_takes_it_off_the_list(self) -> None:
        ledger = ReadingLedger()
        executed = _executed(_event_catalogue())
        ledger.offer(executed)

        held, first_read = ledger.read(str(executed.evidence_id), step=2)

        assert first_read is True
        assert held.executed is executed
        assert ledger.unread() == ()
        assert ledger.read_count == 1

    def test_reading_the_same_handle_twice_is_not_progress(self) -> None:
        """The loop treats a first read as progress, so that a turn spent reading does not
        count toward the barren streak. A re-read must not buy another turn of that credit,
        or a loop could keep itself alive by re-reading one observation forever."""
        ledger = ReadingLedger()
        executed = _executed(_event_catalogue())
        ledger.offer(executed)

        ledger.read(str(executed.evidence_id), step=1)
        _, first_read = ledger.read(str(executed.evidence_id), step=2)

        assert first_read is False
        assert ledger.read_count == 1

    def test_an_observation_shown_in_full_says_so_rather_than_erroring(self) -> None:
        """The model asking to read something it already has in full is a reasonable mistake.
        "No such handle" would read as a bug it should work around; "you already have every
        row" is the fact, and it ends the exchange."""
        ledger = ReadingLedger()
        executed = _executed({"rows": [{"sessions": 1200}]})

        assert ledger.offer(executed) is None
        with pytest.raises(UnknownHandle, match="shown to you in full"):
            ledger.read(str(executed.evidence_id), step=0)

    def test_an_unrecognised_id_comes_back_with_the_ids_that_do_exist(self) -> None:
        """A correctable error rather than a dead end: the reply names what can be read, so
        the next turn can read the right thing instead of giving up on reading."""
        ledger = ReadingLedger()
        executed = _executed(_event_catalogue())
        ledger.offer(executed)

        with pytest.raises(UnknownHandle) as raised:
            ledger.read(str(uuid.uuid4()), step=0)

        assert str(executed.evidence_id) in str(raised.value)
        assert "posthog.list_events" in str(raised.value)

    @pytest.mark.parametrize("shape", ["  {id}  ", "evidence_id: {id}", "'{id}'"])
    def test_the_model_reformatting_the_id_does_not_lose_the_read(self, shape: str) -> None:
        """A read that fails on formatting is a read the model does not retry -- and then it
        reasons from the digest, which is the failure this mechanism exists to prevent. So
        the handle tolerates the wrappers a model actually emits."""
        ledger = ReadingLedger()
        executed = _executed(_event_catalogue())
        ledger.offer(executed)

        held, _ = ledger.read(shape.format(id=executed.evidence_id), step=0)

        assert held.executed is executed


class TestTheDisclosureIsAbsentWhenItIsNotNeeded:
    """Every new disclosure in this codebase has to be absent on the calls that do not need
    it. A notice that appears on every turn is read on none of them."""

    def test_no_reminder_when_nothing_was_summarised(self) -> None:
        ledger = ReadingLedger()
        ledger.offer(_executed({"rows": [{"sessions": 1200}]}))

        assert ledger.reminder() == ""
        assert ledger.has_handles is False

    def test_no_reminder_once_everything_has_been_read(self) -> None:
        ledger = ReadingLedger()
        executed = _executed(_event_catalogue())
        ledger.offer(executed)
        ledger.read(str(executed.evidence_id), step=0)

        assert ledger.reminder() == ""

    def test_the_reminder_names_the_handle_and_how_to_read_it(self) -> None:
        ledger = ReadingLedger()
        executed = _executed(_event_catalogue())
        ledger.offer(executed)

        reminder = ledger.reminder()

        assert "FETCHED BUT NOT READ (1)" in reminder
        assert str(executed.evidence_id) in reminder
        assert READ_TOOL in reminder

    def test_the_read_tool_is_offered_once_and_then_never_withdrawn(self) -> None:
        """`has_handles` is monotone because tool schemas sit in the cached prefix of every
        request. A spec that appeared and disappeared as the unread set emptied would
        invalidate the prompt cache at each transition -- and caching is worth 72% of the
        loop's wall clock, against about 120 tokens for the spec."""
        ledger = ReadingLedger()
        executed = _executed(_event_catalogue())
        ledger.offer(executed)
        assert ledger.has_handles is True

        ledger.read(str(executed.evidence_id), step=0)

        assert ledger.has_handles is True


class TestTheReadToolContract:
    def test_it_says_that_reading_is_free(self) -> None:
        """The description is the whole intervention on the model's side. If reading looks
        expensive the model reasons from the digest, and this change has then made the
        investigation worse rather than better."""
        spec = read_tool_spec()

        assert spec["name"] == READ_TOOL
        description = spec["description"]
        assert isinstance(description, str)
        assert "no upstream request" in description
        assert "does not count against your tool-call budget" in description

    def test_it_rejects_arguments_it_did_not_ask_for(self) -> None:
        """The same rule every registered capability is held to: an unknown argument is a
        hallucination and must surface as a validation error rather than be ignored."""
        schema = read_tool_spec()["input_schema"]
        assert isinstance(schema, dict)
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["evidence_id"]
