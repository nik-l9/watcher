"""The threshold is a promise, so the arithmetic behind it is checked against known values.

A bound that is quietly too loose delivers reports it should have escalated, and nothing
downstream would reveal it: the reports look fine individually, and the error rate it claims
to hold is never measured again. So the binomial bound is tested against a closed form rather
than against itself, and the two ways of being wrong are tested for separately.
"""

from __future__ import annotations

import math

from cortex.eval.abstention import (
    Point,
    binomial_cdf,
    calibrate,
    instability_of,
    majority_premise,
    points_from,
    points_from_repeats,
    render,
    risk_upper_bound,
    score_of,
)


def point(score: float, expected: str, stated: str | None, name: str = "c") -> Point:
    return Point(scenario=name, score=score, expected=expected, stated=stated)


class TestTheBinomialArithmetic:
    def test_cdf_matches_a_hand_computable_case(self):
        # P(X <= 1) for n=3, p=0.5 is (1 + 3)/8. Compared with a tolerance because the sum
        # is taken in log space, which trades exactness for not overflowing at large n.
        assert math.isclose(binomial_cdf(1, 3, 0.5), 0.5, abs_tol=1e-12)

    def test_cdf_is_one_at_the_top_and_zero_below_it(self):
        assert binomial_cdf(3, 3, 0.7) == 1.0
        assert binomial_cdf(2, 3, 1.0) == 0.0

    def test_zero_errors_matches_the_closed_form_clopper_pearson(self):
        """With no errors the one-sided upper limit is exactly 1 - delta**(1/n)."""
        for n in (5, 10, 100):
            expected = 1 - 0.05 ** (1 / n)
            assert math.isclose(risk_upper_bound(0, n, 0.05), expected, abs_tol=1e-6)

    def test_no_observations_bound_nothing(self):
        """Zero errors out of zero is not evidence of safety.

        Returning 0.0 here would make a cut that delivers nothing look perfectly safe, and
        `calibrate` would then prefer it to every cut that actually delivers something.
        """
        assert risk_upper_bound(0, 0) == 1.0

    def test_more_evidence_tightens_the_bound(self):
        assert risk_upper_bound(0, 100) < risk_upper_bound(0, 10)

    def test_all_errors_bounds_at_one(self):
        assert risk_upper_bound(7, 7) == 1.0


class TestWhichErrorsCount:
    def test_a_wrong_verdict_that_misleads_counts(self):
        assert point(0.1, "false", "holds").misleading

    def test_declining_to_answer_is_not_misleading(self):
        # Costs an answer, tells no untruth. Counting it would let the bound be bought down
        # by escalating everything.
        assert not point(0.1, "holds", "unverifiable").misleading

    def test_a_benign_confusion_is_not_misleading(self):
        assert not point(0.1, "none_asserted", "holds").misleading

    def test_an_undelivered_report_misleads_nobody(self):
        p = point(0.1, "false", None)
        assert not p.delivered
        assert not p.misleading


class TestScoringACapturedReport:
    def test_a_fully_supported_report_scores_zero(self):
        bundle = {"verdicts": [{"verdict": "supported"}] * 4, "unverified": []}
        assert score_of(bundle) == 0.0

    def test_doubted_and_unjudged_claims_both_raise_the_score(self):
        bundle = {
            "verdicts": [{"verdict": "supported"}, {"verdict": "overstated"}],
            "unverified": ["a claim nobody could judge"],
        }
        assert math.isclose(score_of(bundle), 2 / 3)

    def test_a_refused_draft_scores_the_maximum(self):
        assert score_of({"rejected_because": "no surviving evidence", "verdicts": []}) == 1.0

    def test_a_report_nothing_checked_is_not_treated_as_clean(self):
        assert score_of({"verdicts": [], "unverified": []}) == 1.0


class TestFittingTheCut:
    def test_a_score_that_separates_perfectly_delivers_the_clean_reports(self):
        points = tuple(
            [point(0.0, "holds", "holds", f"good{i}") for i in range(40)]
            + [point(0.9, "false", "holds", f"bad{i}") for i in range(10)]
        )
        fitted = calibrate(points, alpha=0.10)
        assert fitted.cut == 0.0
        assert fitted.delivered == 40
        assert fitted.misleading_delivered == 0

    def test_the_most_permissive_feasible_cut_wins(self):
        """Every feasible cut satisfies the bound, so the useful one delivers most."""
        points = tuple(
            [point(0.1, "holds", "holds", f"a{i}") for i in range(30)]
            + [point(0.2, "holds", "holds", f"b{i}") for i in range(30)]
        )
        assert calibrate(points, alpha=0.10).cut == 0.2

    def test_a_set_that_is_mostly_wrong_delivers_nothing(self):
        points = tuple(point(0.1 * i, "false", "holds", f"x{i}") for i in range(10))
        fitted = calibrate(points, alpha=0.10)
        assert not fitted.feasible
        assert fitted.delivered == 0

    def test_too_few_examples_cannot_certify_a_low_rate(self):
        """Five clean reports cannot prove a 1% error rate, and must not claim to.

        1 - 0.05**(1/5) is about 45%, well above alpha, so no cut is feasible -- the right
        answer, and the one a normal approximation would get wrong at this n.
        """
        points = tuple(point(0.0, "holds", "holds", f"c{i}") for i in range(5))
        assert not calibrate(points, alpha=0.01).feasible

    def test_undelivered_attempts_are_outside_the_bound(self):
        points = tuple(
            [point(0.0, "holds", "holds", f"ok{i}") for i in range(40)]
            + [point(0.0, "holds", None, "crashed")]
        )
        fitted = calibrate(points, alpha=0.10)
        assert fitted.total == 40
        assert "1 attempt(s) produced no report" in render(fitted, points)

    def test_an_empty_calibration_set_is_not_a_pass(self):
        fitted = calibrate(())
        assert not fitted.feasible
        assert "No cut delivers anything" in render(fitted, ())


class TestPairingBundlesWithLabels:
    def test_only_labelled_bundles_are_used(self):
        bundles = [
            {"scenario": "gen_a", "report": {"premise": "holds"}, "verdicts": [], "unverified": []},
            {"scenario": "onboarding_regression", "report": {"premise": "holds"}},
        ]
        points = points_from(bundles, {"gen_a": "holds"})
        assert [p.scenario for p in points] == ["gen_a"]

    def test_a_bundle_with_no_premise_is_recorded_as_undelivered(self):
        points = points_from([{"scenario": "gen_a", "report": {}}], {"gen_a": "holds"})
        assert points[0].stated is None


def bundle(name: str, premise: str | None) -> dict:
    return {"scenario": name, "report": {"premise": premise} if premise else {}}


class TestScoringInstabilityAcrossAttempts:
    """The score that can see the defect `score_of` provably cannot.

    The measured failure is the same evidence yielding a different verdict on a re-run, not
    an unsupported claim. On the pilot, `score_of` ranked both wrong reports as safer than
    both right ones, because every claim in a premise-confused report is supported. This
    measures the re-roll itself.
    """

    def test_unanimous_attempts_score_zero(self):
        assert instability_of([bundle("a", "holds")] * 5) == 0.0

    def test_an_even_split_scores_a_half(self):
        attempts = [bundle("a", "false")] * 2 + [bundle("a", "unverifiable")] * 2
        assert instability_of(attempts) == 0.5

    def test_an_undelivered_attempt_counts_against_agreement(self):
        """Two answers out of seven is not as settled as two out of two.

        Dropping the silent attempts would make an unreliable case look unanimous, which is
        the opposite of what the score is for.
        """
        attempts = [bundle("a", "holds")] * 2 + [bundle("a", None)] * 5
        assert instability_of(attempts) > 0.5

    def test_no_attempt_delivering_is_maximally_unstable(self):
        assert instability_of([bundle("a", None)] * 3) == 1.0


class TestTheMajorityVerdict:
    def test_the_verdict_most_attempts_reached_wins(self):
        attempts = [bundle("a", "holds")] * 3 + [bundle("a", "false")]
        assert majority_premise(attempts) == "holds"

    def test_a_tie_breaks_toward_the_cautious_verdict(self):
        """Picking the confident half of a coin flip is how instability becomes a falsehood."""
        attempts = [bundle("a", "false")] * 2 + [bundle("a", "unverifiable")] * 2
        assert majority_premise(attempts) == "unverifiable"

    def test_a_case_that_never_answered_states_nothing(self):
        assert majority_premise([bundle("a", None)]) is None


class TestBuildingPointsFromRepeats:
    def test_attempts_are_grouped_into_one_point_per_case(self):
        bundles = [bundle("gen_a", "holds"), bundle("gen_a", "false"), bundle("gen_b", "holds")]
        points = points_from_repeats(bundles, {"gen_a": "holds", "gen_b": "holds"})
        assert [p.scenario for p in points] == ["gen_a", "gen_b"]
        assert points[0].score == 0.5
        assert points[1].score == 0.0

    def test_an_unstable_case_that_lands_right_is_still_scored_unstable(self):
        """The point of the score: being right by luck must not look safe."""
        bundles = [bundle("gen_a", "unverifiable"), bundle("gen_a", "false")]
        point = points_from_repeats(bundles, {"gen_a": "unverifiable"})[0]
        assert point.stated == "unverifiable"
        assert not point.misleading
        assert point.score == 0.5
