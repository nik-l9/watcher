"""Whether a series may answer a business question at all.

ADR 0005 decision 1, over the signals four earlier fixes already put in the payload. Eleven
checks are specified in the research; four are answerable today and five are named as
unevaluated rather than passed.

**That distinction is the design.** The research's warning is blunt: *a gate whose checks all
return `unknown` passes everything*. Sequenced wrongly it looks like a safeguard and behaves like
a pass-through.
"""

from __future__ import annotations

import pytest

from cortex.analysis.trust import (
    NOT_EVALUATED,
    Trust,
    UngradeableSource,
    assess_trust,
)

#: The one source the gate has checks for. Passed explicitly everywhere below, because the
#: guard exists to make a caller state which connector produced the payload.
_POSTHOG = "posthog.event_trend"


class TestOnlyACollectionFailureBlocks:
    def test_the_real_august_incident_is_broken(self) -> None:
        """53 server-side events ceasing while 8 kept recording. The one check the research
        allows to block, and the reasoning is that a group of unrelated events falling silent
        within a day of each other is not something changing user behaviour can produce."""
        verdict = assess_trust(
            {
                "blast_radius": {
                    "scope": "shared_with_other_events",
                    "stopped_count": 53,
                    "still_recording_count": 8,
                },
                "series_ends_early": {"last_bucket": "2026-08-03", "days_missing": 20},
            },
            source=_POSTHOG,
        )
        assert verdict.state is Trust.BROKEN
        assert not verdict.may_answer_the_business_question

    def test_one_event_stopping_alone_only_degrades(self) -> None:
        """Every other series still recording rules out a shared pipeline, which leaves a
        rename, broken instrumentation, or it genuinely stopping. That is a narrower question,
        not an unanswerable one."""
        verdict = assess_trust({"blast_radius": {"scope": "this_event_only"}}, source=_POSTHOG)
        assert verdict.state is Trust.DEGRADED
        assert verdict.may_answer_the_business_question

    def test_a_project_wide_stop_only_degrades(self) -> None:
        """With nothing still recording there is no healthy sibling to compare against, so a
        real outage cannot be ruled out -- which is exactly why this is the weakened form."""
        verdict = assess_trust({"blast_radius": {"scope": "project_wide"}}, source=_POSTHOG)
        assert verdict.state is Trust.DEGRADED

    def test_a_gap_alone_degrades_rather_than_blocks(self) -> None:
        verdict = assess_trust(
            {"series_ends_early": {"last_bucket": "2026-08-03", "days_missing": 12}},
            source=_POSTHOG,
        )
        assert verdict.state is Trust.DEGRADED

    def test_a_partial_bucket_degrades(self) -> None:
        verdict = assess_trust({"partial_buckets": [{"bucket": "2026-08-01"}]}, source=_POSTHOG)
        assert verdict.state is Trust.DEGRADED

    def test_a_clean_series_passes(self) -> None:
        clean = assess_trust({"event": "user signed up", "total": 4000}, source=_POSTHOG)
        assert clean.state is Trust.OK


class TestUnknownIsNeverAPass:
    def test_the_gate_names_the_rows_it_cannot_answer(self) -> None:
        """A gate showing only the checks it can run looks complete, and this one is not: five
        of eleven rows need metadata this project does not collect."""
        verdict = assess_trust({}, source=_POSTHOG)
        assert len(verdict.not_evaluated) == 5
        assert verdict.not_evaluated == NOT_EVALUATED

    def test_a_clean_verdict_says_it_is_not_a_clean_bill_of_health(self) -> None:
        note = assess_trust({}, source=_POSTHOG).note
        assert "not the same as the series being sound" in note
        assert "not evaluated rather than passed" in note

    def test_an_absent_blast_radius_is_recorded_as_unevaluated(self) -> None:
        """No sibling comparison was made, which is different from one having been made and
        found nothing."""
        checks = assess_trust({}, source=_POSTHOG).checks
        cessation = next(c for c in checks if "correlated" in c.check.value)
        assert not cessation.evaluated
        assert cessation.trips_to is None


class TestTheNoteSaysWhatToDo:
    def test_broken_forbids_attributing_the_movement(self) -> None:
        """The instruction, not just the state. The original wrong answer attributed an absence
        of measurement to a demand-side cause."""
        note = assess_trust(
            {
                "blast_radius": {
                    "scope": "shared_with_other_events",
                    "stopped_count": 13,
                    "still_recording_count": 11,
                }
            },
            source=_POSTHOG,
        ).note
        assert "cannot answer a business question" in note
        assert "only an absence of measurement" in note
        assert "what stopped, when, on which emitter" in note

    def test_degraded_asks_for_the_limitation_in_the_answer(self) -> None:
        """Not as a caveat. A caveat is what a reader skips."""
        note = assess_trust({"partial_buckets": [{"bucket": "x"}]}, source=_POSTHOG).note
        assert "narrowed question" in note
        assert "rather than as a caveat" in note

    def test_the_reason_travels_with_the_state(self) -> None:
        note = assess_trust(
            {
                "blast_radius": {
                    "scope": "shared_with_other_events",
                    "stopped_count": 53,
                    "still_recording_count": 8,
                }
            },
            source=_POSTHOG,
        ).note
        assert "53 series stopped together while 8 kept recording" in note


class TestItRefusesAPayloadItCannotGrade:
    """The gate's own docstring names the failure: a gate whose checks all return unknown
    passes everything. Generalising it to a second connector is how that happens.

    All three implemented checks read keys PostHog computes and nothing else does. Handed a GA4
    payload the gate returned `ok` for every series, with a note claiming five of eleven rows
    went unevaluated when in fact all eleven had -- reassuring text over a check that never ran.
    """

    def test_a_connector_computing_nothing_is_refused_rather_than_passed(self) -> None:
        with pytest.raises(UngradeableSource, match="computes none of this gate's inputs"):
            assess_trust({"deals": [], "count": 0}, source="hubspot.pipeline")

    def test_the_refusal_says_what_is_missing(self) -> None:
        """The reader of this error is whoever is wiring up the next connector."""
        with pytest.raises(UngradeableSource) as raised:
            assess_trust({}, source="github.recent_prs")
        message = str(raised.value)
        assert "blast_radius" in message
        assert "series_ends_early" in message
        assert "partial_buckets" in message

    def test_absence_of_a_disclosure_is_not_evidence_of_health(self) -> None:
        """Why the guard cannot be shape-sniffing.

        A *healthy* PostHog series carries none of the three disclosure keys either -- that is
        what healthy looks like -- so no test of the payload's contents can tell a sound series
        from a connector that never computes them. Only the caller knows, so the caller says.
        """
        healthy = assess_trust({"event": "user signed up", "total": 4000}, source=_POSTHOG)
        assert healthy.state is Trust.OK
        assert {"blast_radius", "series_ends_early", "partial_buckets"}.isdisjoint(
            {"event", "total"}
        )


class TestASourceIsGradedOnlyForWhatItComputes:
    """ "This source is gradeable" is too coarse to be honest.

    GA4's daily series has no partial bucket to disclose, ever, and GA4 exposes no sibling-series
    structure for a correlated cessation. Granting it the whole gate would report two checks as
    passing that were never run -- the same defect as grading a source that computes nothing,
    just two thirds as large.
    """

    def test_ga4_reports_the_checks_it_cannot_run_as_unrun(self) -> None:
        verdict = assess_trust({}, source="ga4.get_sessions")
        unrun = {c.check.value for c in verdict.checks if not c.evaluated}
        assert unrun == {"gate2_trailing_bucket", "gate3_correlated_cessation"}

    def test_ga4_can_degrade(self) -> None:
        verdict = assess_trust(
            {"series_ends_early": {"last_bucket": "2026-08-03", "days_missing": 12}},
            source="ga4.get_sessions",
        )
        assert verdict.state is Trust.DEGRADED

    def test_ga4_can_never_block(self) -> None:
        """Blocking is reserved for a correlated cessation, and GA4 has nothing to correlate.

        A blast_radius key in a GA4 payload could only be a mistake, so it must not be honoured:
        blocking on it would block on evidence the source cannot actually produce.
        """
        verdict = assess_trust(
            {
                "blast_radius": {
                    "scope": "shared_with_other_events",
                    "stopped_count": 53,
                    "still_recording_count": 8,
                },
                "series_ends_early": {"last_bucket": "2026-08-03", "days_missing": 12},
            },
            source="ga4.get_sessions",
        )
        assert verdict.state is Trust.DEGRADED
        assert verdict.may_answer_the_business_question

    def test_the_same_payload_blocks_on_posthog(self) -> None:
        """Same input, different source, different verdict -- which is the point of the mapping."""
        payload = {
            "blast_radius": {
                "scope": "shared_with_other_events",
                "stopped_count": 53,
                "still_recording_count": 8,
            }
        }
        assert assess_trust(payload, source=_POSTHOG).state is Trust.BROKEN

    def test_the_note_counts_unevaluated_rows_rather_than_asserting_a_figure(self) -> None:
        """It used to say "five of the gate's eleven rows" as a literal. On GA4 that is seven,
        and a hardcoded figure makes the note lie the moment a second source is added."""
        posthog = assess_trust({}, source=_POSTHOG).note
        ga4 = assess_trust({}, source="ga4.get_sessions").note
        assert "6 of the gate's 11 rows" in posthog
        assert "7 of the gate's 11 rows" in ga4
