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
        assert "removed the planted cause" in scored.detail

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
