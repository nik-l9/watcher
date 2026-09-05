"""Did the gate withhold the answer it was supposed to protect?

**The dimension that would have caught the failure nothing else did.** In run 15 the sufficiency
gate withheld the correct planted cause on both scenarios that have one -- eleven vetoes across
eight attempts -- and every scenario still scored accuracy 1.00, because the cause survives in the
findings once the summary claim is cut. A gate silently withholding correct answers passed every
dimension in the suite, and it was found by reading bundles by hand.

The ground truth needed already existed: `required_signals` names the planted cause and
`is_unanswerable` says whether one exists at all.
"""

from __future__ import annotations

import uuid

from cortex.eval.fixtures import by_name
from cortex.eval.scorer import Scorer
from cortex.reports.gate import Rejection, RejectionReason
from cortex.reports.schema import Claim, Confidence, InvestigationReport
from cortex.reports.sufficiency import AppliedSufficiency


def _applied(*withheld: str) -> AppliedSufficiency:
    evidence = [uuid.uuid4()]
    return AppliedSufficiency(
        report=InvestigationReport(
            question="Why did signups fall?",
            executive_summary=[Claim(text="Something happened.", evidence_ids=evidence)],
            confidence=Confidence.LOW,
        ),
        rejections=[
            Rejection(
                location=f"executive_summary[{index}]",
                reason=RejectionReason.NO_SURVIVING_EVIDENCE,
                detail="sufficiency: the evidence does not carry a cause",
                text=text,
            )
            for index, text in enumerate(withheld)
        ],
        withheld=len(withheld),
    )


def _score(scenario_name: str, *withheld: str):
    return Scorer()._veto_precision(by_name(scenario_name), _applied(*withheld))


class TestWithholdingThePlantedCauseIsPenalised:
    def test_the_run_15_failure_scores_down(self) -> None:
        """The sentence the gate actually withheld, on the scenario it withheld it from."""
        dimension = _score(
            "campaign_traffic_drop",
            "The spring paid-search ad campaign was paused on June 14 due to exhausted "
            "budget, cutting Paid Search traffic and thus signups.",
        )
        assert dimension.score < 1.0
        assert "the summary lost the planted cause" in dimension.detail

    def test_withholding_something_else_is_not_penalised(self) -> None:
        """The gate is *supposed* to withhold claims the evidence cannot carry. Penalising every
        veto would push against the behaviour the suite wants."""
        dimension = _score(
            "campaign_traffic_drop",
            "No Slack discussion was found, so nothing else can be ruled out.",
        )
        assert dimension.score == 1.0
        assert "none carrying the planted cause" in dimension.detail

    def test_withholding_nothing_is_clean(self) -> None:
        assert _score("campaign_traffic_drop").score == 1.0

    def test_the_score_reflects_how_much_was_lost(self) -> None:
        """A scenario whose cause needs two signals loses half its score for one withheld, not
        all of it -- the report may still carry the other half."""
        truth = by_name("onboarding_regression").ground_truth
        assert len(truth.required_signals) > 1, "this test needs a multi-signal scenario"
        # Only one of the two signals: "913" without "mobile". Withholding a sentence carrying
        # both, as the live gate did, correctly scores zero.
        one = _score("onboarding_regression", "PR #913 shipped that week.")
        assert 0.0 < one.score < 1.0
        both = _score("onboarding_regression", "The mobile onboarding modal rework (PR #913).")
        assert both.score == 0.0


class TestScenariosWithNoCauseToProtect:
    def test_an_unanswerable_scenario_is_not_scored(self) -> None:
        """Where no cause is establishable, withholding a causal claim is the gate working --
        and `accuracy` already scores whether the report correctly declined."""
        dimension = _score(
            "insufficient_evidence", "A deploy on 22 May caused the enterprise signup drop."
        )
        assert dimension.score == 1.0
        assert "no findable cause to protect" in dimension.detail

    def test_a_false_premise_scenario_is_not_scored(self) -> None:
        """Nothing to explain, so nothing to protect."""
        dimension = _score("partial_month_false_premise", "The pricing redesign caused it.")
        assert dimension.score == 1.0
        assert "no findable cause to protect" in dimension.detail

    def test_the_tempting_scenario_is_not_scored_either(self) -> None:
        """Built so no cause is establishable, however tempting the coincidence."""
        assert _score("tempting_coincidence", "The 16 June copy deploy caused it.").score == 1.0


class TestItNeverGates:
    def test_a_wrongly_withheld_claim_does_not_fail_the_build(self) -> None:
        """A wrongly withheld claim degrades an answer; it does not fabricate one. This project
        reserves gating for the second kind."""
        dimension = _score(
            "campaign_traffic_drop",
            "The spring paid-search ad campaign was paused on June 14, cutting signups.",
        )
        assert not dimension.gates


class TestItScoresWhatTheSummaryLostNotWhatWasWithheld:
    """The same sharpening `verifier_precision` already had, applied to the other mechanism.

    The two dimensions ask one question of the two things that remove claims, and measuring them
    differently made this one report harm where there was none. In run 33 it read 0.50 on
    `measurement_stopped` because a withheld claim said "collection stopped" — while the
    delivered summary said GA4's data "simply *stops* after 2026-08-03", which is the same answer
    in the present tense. A reader lost nothing.
    """

    def test_a_withheld_claim_whose_cause_survives_in_the_summary_is_clean(self) -> None:
        scenario = by_name("campaign_traffic_drop")
        applied = _applied("The campaign ending drove the fall.")
        applied.report.executive_summary[0] = Claim(
            text="Signups fell because the campaign budget was exhausted on 14 June.",
            evidence_ids=[uuid.uuid4()],
        )
        scored = Scorer()._veto_precision(scenario, applied)
        assert scored.score == 1.0
        assert "still names the planted cause" in scored.detail

    def test_it_still_penalises_a_summary_that_lost_it(self) -> None:
        """The run-15 failure this dimension was built for: the cause cut from the summary,
        surviving only in a finding, with `accuracy` reading 1.00 throughout."""
        scenario = by_name("campaign_traffic_drop")
        applied = _applied("The campaign ending drove the fall.")
        applied.report.executive_summary[0] = Claim(
            text="Paid search sessions fell sharply in the second half of June.",
            evidence_ids=[uuid.uuid4()],
        )
        scored = Scorer()._veto_precision(scenario, applied)
        assert scored.score == 0.0
        assert "appears nowhere in the delivered summary" in scored.detail
