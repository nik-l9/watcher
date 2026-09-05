"""Did the adversarial verifier remove the answer it was supposed to check?

`veto_precision` asks this of the sufficiency gate, and exists because that gate was caught
withholding correct planted causes while every other dimension read clean. The verifier is the
other mechanism that removes claims, it decides by asking a model rather than by resolving an id,
and nothing asked the same question of it — so the more fallible of the two was the unwatched one.

The error class is from arXiv 2608.18300 §5.3: a judge reaching a defensible verdict for the
wrong reason, which is invisible unless something scores its removals against a known answer.
Here that answer is the scenario's planted cause, so this stays a mechanical check on an LLM's
decision rather than an LLM's opinion about one.
"""

from __future__ import annotations

import uuid

import pytest

from cortex.eval.fixtures import alternatives_of, by_name
from cortex.eval.scorer import Scorer
from cortex.reports.schema import Claim, Confidence, InvestigationReport
from cortex.reports.verifier import ClaimVerdict, VerificationResult
from cortex.reports.verifier import Verdict as ClaimVerdictKind


def _verified(*removed: str, overstated: tuple[str, ...] = ()) -> VerificationResult:
    report = InvestigationReport(
        question="Why did signups fall?",
        executive_summary=[Claim(text="Something happened.", evidence_ids=[uuid.uuid4()])],
        confidence=Confidence.LOW,
    )
    verdicts = [
        ClaimVerdict(
            location=f"findings[{index}].claims[0]",
            claim_text=text,
            verdict=ClaimVerdictKind.UNSUPPORTED,
            reason="the cited evidence does not say this",
        )
        for index, text in enumerate(removed)
    ]
    verdicts += [
        ClaimVerdict(
            location=f"findings[{index}].claims[1]",
            claim_text=text,
            verdict=ClaimVerdictKind.OVERSTATED,
            reason="stronger than the evidence supports",
        )
        for index, text in enumerate(overstated)
    ]
    return VerificationResult(report=report, verdicts=verdicts)


def _cause_signal(scenario_name: str) -> str:
    """One alternative of the scenario's first required signal — the planted cause's own name."""
    truth = by_name(scenario_name).ground_truth
    return alternatives_of(truth.required_signals[0])[0]


class TestRemovingThePlantedCauseIsPenalised:
    def test_a_removed_claim_carrying_the_cause_scores_below_one(self) -> None:
        """The failure this dimension exists for. `campaign_traffic_drop` hit it on the
        sufficiency gate in runs 25 and 27; nothing was watching the verifier for the same
        thing."""
        scenario = by_name("campaign_traffic_drop")
        signal = _cause_signal("campaign_traffic_drop")
        scored = Scorer()._verifier_precision(
            scenario, _verified(f"The {signal} ending is what drove the fall.")
        )
        assert scored.score < 1.0
        assert "the summary lost the planted cause" in scored.detail

    def test_the_check_can_fail_at_all(self) -> None:
        """Guarding the mistake this nearly shipped with. `Verdict` names two different enums in
        this codebase — the report schema's hypothesis verdict and the verifier's per-claim one —
        and comparing against the wrong one matches nothing, so the dimension would have reported
        a clean 1.00 on every run forever. A dimension that cannot fail is not a safeguard.
        """
        scenario = by_name("campaign_traffic_drop")
        removed_everything = _verified(
            *(alternatives_of(r)[0] for r in scenario.ground_truth.required_signals)
        )
        assert Scorer()._verifier_precision(scenario, removed_everything).score == 0.0

    def test_removing_something_else_is_not_penalised(self) -> None:
        """A verifier removing an unsupported claim is the verifier working, and most removals
        are exactly that. Penalising them would push against the behaviour the suite wants."""
        scenario = by_name("campaign_traffic_drop")
        scored = Scorer()._verifier_precision(
            scenario, _verified("Tuesday's traffic was slightly below Monday's.")
        )
        assert scored.score == 1.0
        assert "none carrying the planted cause" in scored.detail


class TestWhatIsDeliberatelyNotScored:
    def test_an_overstated_claim_is_not_a_removal(self) -> None:
        """Overstatement downgrades confidence and leaves the claim in the report, so counting it
        here would penalise the verifier for a claim the reader still receives."""
        scenario = by_name("campaign_traffic_drop")
        signal = _cause_signal("campaign_traffic_drop")
        scored = Scorer()._verifier_precision(
            scenario, _verified(overstated=(f"The {signal} ending explains everything.",))
        )
        assert scored.score == 1.0
        assert "removed nothing" in scored.detail

    def test_a_scenario_with_no_findable_cause_is_exempt(self) -> None:
        """Where the honest answer is that the data cannot say, a removed causal claim is the
        system working — and `accuracy` already scores whether the report declined."""
        scored = Scorer()._verifier_precision(
            by_name("insufficient_evidence"), _verified("The deploy caused it.")
        )
        assert scored.score == 1.0
        assert "no findable cause" in scored.detail

    def test_removing_nothing_is_a_clean_pass(self) -> None:
        scored = Scorer()._verifier_precision(by_name("campaign_traffic_drop"), _verified())
        assert scored.score == 1.0


class TestTheRemovalBreakdownNamesTheMechanism:
    """`draft_reliability` merged two mechanisms into one number.

    A citation resolving to nothing is the drafter inventing an id; a claim its own evidence does
    not support is the drafter overreaching from real data. Different fixes, and 0.82 looks
    identical either way. The score still merges them — it is a survival rate — but the detail
    line no longer does.
    """

    def test_each_mechanism_is_named_when_it_removed_something(self) -> None:
        from cortex.reports.gate import GateResult, Rejection, RejectionReason

        evidence = [uuid.uuid4()]
        report = InvestigationReport(
            question="Why did signups fall?",
            executive_summary=[Claim(text="Signups fell.", evidence_ids=evidence)],
            confidence=Confidence.LOW,
        )
        gate = GateResult(
            report=report,
            rejections=[
                Rejection(
                    location="findings[0].claims[0]",
                    reason=RejectionReason.UNKNOWN_EVIDENCE,
                    detail="no such evidence id",
                    text="A fabricated claim.",
                )
            ],
        )
        scored = Scorer()._draft_reliability(report, gate, _verified("An unsupported claim."))
        assert "cited nothing that resolves" in scored.detail
        assert "unsupported by its own evidence" in scored.detail

    def test_a_clean_draft_says_nothing_extra(self) -> None:
        """A detail reading "gate 0, verifier 0" on every passing run trains a reader to skip the
        field, and the field exists to be read on the run where it is not zero."""
        from cortex.reports.gate import GateResult

        report = InvestigationReport(
            question="Why did signups fall?",
            executive_summary=[Claim(text="Signups fell.", evidence_ids=[uuid.uuid4()])],
            confidence=Confidence.LOW,
        )
        scored = Scorer()._draft_reliability(report, GateResult(report=report), _verified())
        assert scored.detail == "1/1 drafted claims survived review"

    def test_repeated_drafting_attempts_are_reported(self) -> None:
        """A repair or a brevity retry is recoverable and leaves no mark on any score. That is
        right, and it also means a drafter that has started needing two attempts every run looks
        exactly like one that does not."""
        from cortex.agents.timing import DRAFT, Phase, Timings
        from cortex.reports.gate import GateResult

        class _WithTwoDrafts:
            duration_ms = 1000
            timings = Timings(phases={DRAFT: Phase(calls=2, seconds=40.0)})

        report = InvestigationReport(
            question="Why did signups fall?",
            executive_summary=[Claim(text="Signups fell.", evidence_ids=[uuid.uuid4()])],
            confidence=Confidence.LOW,
        )
        scored = Scorer()._draft_reliability(
            report, GateResult(report=report), _verified(), _WithTwoDrafts()
        )
        assert "after 2 drafting attempts" in scored.detail


class TestItScoresWhatTheSummaryLostNotWhatWasRemoved:
    """Two things can be true at once, and matching on the removed text alone conflated them.

    Run 32: the verifier cut four sentences on `campaign_traffic_drop`, every one genuinely
    unsupported by its own citation — "one day before the session drop began" citing only the
    Slack message, "across three separate search queries" then listing four, "in the June 1–30
    window" when the payload queried the 13th to the 16th. Correct removals, all of them.

    And the delivered summary was left describing a 68.4% collapse in paid search without naming
    the exhausted campaign budget that caused it: the reader gets the mechanism and not the
    thing to act on. `accuracy` reads 1.00 throughout, because it searches the whole report and
    the cause survives in a finding.

    So the question is not whether a removal touched the cause. It is whether the summary still
    carries it.
    """

    def test_a_correct_removal_that_keeps_the_cause_in_the_summary_is_clean(self) -> None:
        scenario = by_name("campaign_traffic_drop")
        signal = _cause_signal("campaign_traffic_drop")
        verification = _verified(f"The {signal} ending drove the fall, one day before it began.")
        verification.report.executive_summary[0] = Claim(
            text=f"Signups fell because the {signal} budget was exhausted on 14 June.",
            evidence_ids=[uuid.uuid4()],
        )
        scored = Scorer()._verifier_precision(scenario, verification)
        assert scored.score == 1.0
        assert "still names the planted cause" in scored.detail

    def test_the_same_removal_scores_zero_when_the_summary_loses_it(self) -> None:
        """The run-32 shape exactly: the removal is right and the answer is poorer for it."""
        scenario = by_name("campaign_traffic_drop")
        signal = _cause_signal("campaign_traffic_drop")
        verification = _verified(f"The {signal} ending drove the fall, one day before it began.")
        verification.report.executive_summary[0] = Claim(
            text="Paid search session volume collapsed 68.4% in the second half of June.",
            evidence_ids=[uuid.uuid4()],
        )
        scored = Scorer()._verifier_precision(scenario, verification)
        assert scored.score == 0.0
        assert "appears nowhere in the delivered summary" in scored.detail

    def test_an_untouched_cause_reports_differently_from_a_surviving_one(self) -> None:
        """Two clean outcomes that call for different attention: nothing bearing the cause was
        touched, or something was and the summary held. One message for both would hide which."""
        scenario = by_name("campaign_traffic_drop")
        scored = Scorer()._verifier_precision(
            scenario, _verified("Tuesday's traffic was slightly below Monday's.")
        )
        assert scored.score == 1.0
        assert "none carrying the planted cause" in scored.detail


class TestTheExonerationTestIsNeitherTooLooseNorTooStrict:
    """Both directions were wrong today, in sequence, and the second was worse.

    First it asked whether the signal appeared anywhere in the summary, which exonerated the
    run-32 defect verbatim, a summary that *denied* the cause, and `"ends"` matching inside
    `"trends"`.

    Then it also required the claim to be causal — and rejected the right answer. On
    `measurement_stopped` the correct summary is "the GA4 sessions data simply stops being
    recorded after August 3": a statement of fact about a data incident, carrying no causal
    marker. That reading scored the dimension down on every attempt of that scenario and on two
    of `campaign_traffic_drop`, which sent me chasing a defect that was not there.

    So: word boundaries, and a claim that does not deny the signal. No causality requirement.
    """

    DENIALS_AND_NOISE = (
        ("campaign_traffic_drop", "The campaign was NOT the cause; the drop remains unexplained."),
        ("measurement_stopped", "Weekly signup trends were reviewed and nothing emerged."),
        ("measurement_stopped", "Recommendations depend on which team owns the tag."),
    )

    @pytest.mark.parametrize(("scenario", "summary"), DENIALS_AND_NOISE)
    def test_a_denial_or_a_substring_does_not_exonerate(self, scenario: str, summary: str) -> None:
        from cortex.eval.scorer import _still_asserted_in_summary

        report = InvestigationReport(
            question="q",
            executive_summary=[Claim(text=summary, evidence_ids=[uuid.uuid4()])],
            confidence=Confidence.LOW,
        )
        requirement = by_name(scenario).ground_truth.required_signals[0]
        assert not _still_asserted_in_summary(report, requirement)

    REAL_ANSWERS = (
        (
            "measurement_stopped",
            "Sessions did not collapse in August - the GA4 sessions data simply stops being "
            "recorded after August 3, 2026.",
        ),
        (
            "measurement_stopped",
            "Site sessions did not collapse in August: the GA4 sessions data feed itself "
            "stopped reporting after 2026-08-03.",
        ),
        (
            "campaign_traffic_drop",
            "A Slack message from 2026-06-14 announcing the spring campaign budget was "
            "exhausted and ads were being paused that day lines up with the drop.",
        ),
    )

    @pytest.mark.parametrize(("scenario", "summary"), REAL_ANSWERS)
    def test_a_correct_answer_exonerates_even_without_a_causal_marker(
        self, scenario: str, summary: str
    ) -> None:
        """Taken verbatim from run 34's delivered reports. A dimension that complains about
        these is worse than one that is slightly loose: it does not gate, so its only effect is
        to send a reader after a defect that is not there."""
        from cortex.eval.scorer import _still_asserted_in_summary

        report = InvestigationReport(
            question="q",
            executive_summary=[Claim(text=summary, evidence_ids=[uuid.uuid4()])],
            confidence=Confidence.LOW,
        )
        requirement = by_name(scenario).ground_truth.required_signals[0]
        assert _still_asserted_in_summary(report, requirement)
