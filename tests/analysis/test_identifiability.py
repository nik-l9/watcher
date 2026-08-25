"""Whether a causal claim is available, decided before one is attempted."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from cortex.analysis.changepoints import detect_level_shifts
from cortex.analysis.identifiability import (
    DEFAULT_POWER,
    Control,
    Gate,
    Intervention,
    _z,
    assess_identifiability,
    minimum_separation,
)
from cortex.analysis.series import as_calendar
from tests.analysis.conftest import FIRST_BREAK, LAST_MEASURED, SERIES_START, replica_rows

#: A window that passes everything it can, so each test can fail exactly one gate.
CLEAN = {
    "shift": 117.0,
    "sigma": 15.0,
    "days_before": 42,
    "days_after": 28,
    "intervention": Intervention("deploy-91c3e", FIRST_BREAK, "web checkout"),
    "controls": (Control("eu-signups", correlation=0.82, attested_unexposed=True),),
    "ledger": (),
}


def _intact():
    return as_calendar(replica_rows(), start=SERIES_START, end=LAST_MEASURED)


def _gate(**overrides):
    return assess_identifiability(overrides.pop("calendar", _intact()), **{**CLEAN, **overrides})


def _check(result, gate: Gate):
    return next(c for c in result.checks if c.gate is gate)


class TestTheCleanCaseIsAvailableButStillNotCausal:
    def test_everything_passing_makes_a_claim_available(self) -> None:
        assert _gate().causal_claim_available

    def test_and_it_still_refuses_to_say_caused_by(self) -> None:
        """The one-sidedness, which is the easiest thing to forget. Passing removes obstacles
        and establishes nothing on its own."""
        summary = _gate().summary()
        assert "consistent with" in summary
        assert "not 'caused by'" in summary
        assert "establishes nothing on its own" in summary


class TestAnySingleRefusalDecides:
    """Worst-domain composition, following ROBINS-I. Confidence does not average -- a gate that
    averaged would let six easy passes outvote the one check saying the question is unanswerable.
    """

    def test_one_failure_refuses_the_whole_claim(self) -> None:
        result = _gate(ledger=None)
        assert not result.causal_claim_available
        assert len(result.refusals) == 1

    def test_six_passes_do_not_outvote_it(self) -> None:
        result = _gate(ledger=None)
        passed = [c for c in result.checks if c.passed]
        assert len(passed) >= 6
        assert not result.causal_claim_available

    def test_every_refusal_says_what_would_lift_it(self) -> None:
        """A refusal that does not say what is missing is the failure mode to design against --
        not saying "I don't know" but saying it uninformatively. The alternative to an
        informative refusal is not silence, it is a human inventing a cause."""
        result = assess_identifiability(
            as_calendar(replica_rows(), start=SERIES_START, end=date(2026, 8, 17)),
            intervention=None,
            shift=117.0,
            sigma=15.0,
            days_before=42,
            days_after=28,
            controls=(),
            ledger=None,
        )
        assert len(result.refusals) >= 4
        for refusal in result.refusals:
            assert refusal.lifts_it, refusal.gate


class TestG0AbsenceStopsTheAnalysis:
    def test_a_trailing_gap_refuses(self) -> None:
        gapped = as_calendar(replica_rows(), start=SERIES_START, end=date(2026, 8, 17))
        check = _check(_gate(calendar=gapped), Gate.COMPLETENESS)
        assert not check.passed
        assert "no rows after 2026-08-03" in check.detail

    def test_a_complete_window_passes(self) -> None:
        assert _check(_gate(), Gate.COMPLETENESS).passed


class TestG1TheSystemMayNotInventTheIntervention:
    def test_no_named_change_refuses(self) -> None:
        check = _check(_gate(intervention=None), Gate.INTERVENTION)
        assert not check.passed
        assert "We can say when the metric moved, not why" in check.detail

    def test_an_intervention_needs_an_identifier(self) -> None:
        with pytest.raises(ValueError, match="needs an identifier"):
            Intervention("   ", FIRST_BREAK, "web")


class TestG2SeparabilityIsExactNotAPreference:
    def test_two_candidates_on_the_same_day_are_not_identified(self) -> None:
        """Two identical indicator columns make the design rank-deficient. This is not "hard to
        separate" -- no estimator distinguishes them, exactly."""
        check = _check(_gate(candidates=(FIRST_BREAK, FIRST_BREAK)), Gate.SEPARABILITY)
        assert not check.passed
        assert "rank-deficient" in check.detail
        assert "no estimator distinguishes them" in check.detail

    def test_at_our_effect_size_one_day_is_enough(self) -> None:
        """The encouraging half of the finding: a 117/day shift against noise near 15 needs a
        single day. Two deploys 24 hours apart around 2026-06-17 are distinguishable."""
        check = _check(
            _gate(candidates=(FIRST_BREAK, FIRST_BREAK + timedelta(days=1))),
            Gate.SEPARABILITY,
        )
        assert check.passed

    def test_a_small_shift_needs_more_separation_than_the_window_has(self) -> None:
        result = _gate(
            shift=12.0,
            sigma=13.9,
            candidates=(FIRST_BREAK, FIRST_BREAK + timedelta(days=3)),
        )
        assert not _check(result, Gate.SEPARABILITY).passed
        assert result.required_separation == 29

    def test_one_candidate_has_nothing_to_separate_from(self) -> None:
        assert _check(_gate(candidates=(FIRST_BREAK,)), Gate.SEPARABILITY).passed


class TestTheSeparationFormula:
    @pytest.mark.parametrize(
        ("shift", "sigma", "days"),
        [(125, 19.0, 1), (60, 14.8, 2), (30, 14.0, 5), (20, 13.9, 11), (12, 13.9, 29)],
    )
    def test_it_reproduces_the_validated_table(self, shift, sigma, days) -> None:
        assert minimum_separation(shift, sigma) == days

    def test_the_default_power_is_the_one_that_reproduces_it(self) -> None:
        """0.90, not the more usual 0.80. Found by implementing: at 0.80 this returns 8 days
        where the simulation measured 11, and only 0.90 lands inside every row's implied
        interval. The prose formula's `z_{1-power}` would be negative at 0.80 and *shrink* the
        requirement, which is a typo for `z_{power}`."""
        assert DEFAULT_POWER == 0.90
        assert minimum_separation(20, 13.9, power=0.80) == 8
        assert minimum_separation(20, 13.9, power=0.90) == 11

    def test_the_power_term_is_not_optional(self) -> None:
        """Omitting it produces a gate that passes cases it should refuse."""
        naive = __import__("math").ceil(2 * (_z(0.975) * 13.9 / 20) ** 2)
        assert naive < minimum_separation(20, 13.9)

    def test_a_zero_shift_does_not_divide_by_zero(self) -> None:
        assert minimum_separation(0.0, 15.0) == 1
        assert minimum_separation(100.0, 0.0) == 1


class TestG3ControlsCannotBeSelectedByCorrelationAlone:
    def test_an_unattested_control_is_not_admissible(self) -> None:
        """A series that moved *because* of our change correlates beautifully, so correlation
        alone selects contaminated controls preferentially."""
        check = _check(
            _gate(controls=(Control("eu", correlation=0.98, attested_unexposed=False),)),
            Gate.CONTROLS,
        )
        assert not check.passed
        assert "nobody has attested they were unexposed" in check.detail
        assert "correlates beautifully" in check.detail

    def test_a_weakly_correlated_attested_control_is_not_admissible_either(self) -> None:
        assert not Control("noise", correlation=0.1, attested_unexposed=True).admissible()

    def test_no_controls_names_the_number_that_would_help(self) -> None:
        check = _check(_gate(controls=()), Gate.CONTROLS)
        assert not check.passed
        assert "Nineteen" in (check.lifts_it or "")


class TestG4TheFloorIsComputedFromShapesAlone:
    def test_zero_controls_makes_synthetic_control_impossible(self) -> None:
        """`p >= 1/(J+1)`, so with J=0, p=1 identically -- at any effect size, ever. The
        conformal route is what keeps this gate passable."""
        result = _gate(controls=())
        assert result.attainable_p == pytest.approx(1 / 70)
        assert _check(result, Gate.P_FLOOR).passed

    def test_a_short_window_cannot_reach_a_strict_alpha(self) -> None:
        """With T=70 the conformal floor is 0.0143, so alpha=0.01 is unreachable no matter what
        the data say."""
        result = _gate(controls=(), alpha=0.01)
        check = _check(result, Gate.P_FLOOR)
        assert not check.passed
        assert "at any effect size" in check.detail

    def test_it_needs_no_data_values(self) -> None:
        """The cheapest and sharpest check in the list."""
        result = _gate(days_before=3, days_after=2, controls=())
        assert result.attainable_p == pytest.approx(1 / 5)
        assert not _check(result, Gate.P_FLOOR).passed


class TestG5AskAboutTheChangeBeforeItsCause:
    def test_a_movement_inside_the_noise_refuses(self) -> None:
        check = _check(_gate(shift=2.0), Gate.DETECTION_FLOOR)
        assert not check.passed
        assert "There may be nothing to explain" in check.detail

    def test_our_own_shift_clears_it_by_an_order_of_magnitude(self) -> None:
        """117/day against a minimum detectable effect near 7.2 -- the research computed 7.09
        on its own replica, so this agrees with it to within the difference in noise estimate."""
        result = _gate()
        check = _check(result, Gate.DETECTION_FLOOR)
        assert check.passed
        assert result.minimum_detectable_effect == pytest.approx(7.17, abs=0.1)
        assert "16.3x the minimum detectable effect" in check.detail

    def test_no_noise_scale_means_no_floor_and_so_a_refusal(self) -> None:
        assert not _check(_gate(sigma=0.0), Gate.DETECTION_FLOOR).passed


class TestG6SensitivityIsAdvisoryNotPassFail:
    def test_a_ratio_below_one_does_not_refuse_the_claim(self) -> None:
        """Always reported, never a gate. It says report bounds rather than a point."""
        result = _gate(placebo_effect=200.0)
        check = _check(result, Gate.SENSITIVITY)
        assert check.advisory
        assert not check.passed
        assert result.causal_claim_available
        assert "report bounds, not a point" in check.detail

    def test_the_ratio_is_reported_when_a_placebo_was_run(self) -> None:
        result = _gate(placebo_effect=8.0)
        assert result.sensitivity_ratio == pytest.approx(117.0 / 8.0)

    def test_no_placebo_says_so_rather_than_implying_a_pass(self) -> None:
        assert "no in-time placebo was run" in _check(_gate(), Gate.SENSITIVITY).detail


class TestG7NobodyLookedIsNotTheSameAsNothingHappened:
    def test_a_missing_ledger_refuses(self) -> None:
        check = _check(_gate(ledger=None), Gate.CONFOUNDER_LEDGER)
        assert not check.passed
        assert "an empty ledger is a claim rather than a default" in check.detail

    def test_an_attested_empty_ledger_passes(self) -> None:
        """The distinction the parameter exists for: `None` is nobody looked, `()` is somebody
        looked and found nothing. Different claims, different verdicts."""
        check = _check(_gate(ledger=()), Gate.CONFOUNDER_LEDGER)
        assert check.passed
        assert "a human attests nothing else landed" in check.detail

    def test_several_coincident_changes_yield_an_unranked_set(self) -> None:
        check = _check(
            _gate(ledger=("pricing page rewrite", "email campaign", "ios release")),
            Gate.CONFOUNDER_LEDGER,
        )
        assert not check.passed
        assert "ranked by nothing" in check.detail
        assert "pricing page rewrite" in check.detail


class TestTheGateOnOurOwnSeries:
    """Reproduces the research's section 5.6 table, and it is the honest answer to the question
    that started all of this: the June break is real and its cause is not establishable."""

    def test_the_question_as_it_actually_arrives_refuses_four_ways(self) -> None:
        gapped = as_calendar(replica_rows(), start=SERIES_START, end=date(2026, 8, 17))
        segmentation = detect_level_shifts(gapped)
        before, after = segmentation.shift_at(FIRST_BREAK)
        result = assess_identifiability(
            gapped,
            intervention=None,
            shift=after - before,
            sigma=segmentation.noise_variance**0.5,
            days_before=42,
            days_after=28,
            controls=(),
            ledger=None,
        )
        refused = {c.gate for c in result.refusals}
        assert refused == {
            Gate.COMPLETENESS,
            Gate.INTERVENTION,
            Gate.CONTROLS,
            Gate.CONFOUNDER_LEDGER,
        }
        # And the two that carry the useful numbers both pass.
        assert _check(result, Gate.DETECTION_FLOOR).passed
        assert _check(result, Gate.P_FLOOR).passed
        assert result.attainable_p == pytest.approx(1 / 70)


def test_the_quantile_function_is_accurate() -> None:
    for probability, expected in ((0.975, 1.959964), (0.90, 1.281552), (0.80, 0.841621)):
        assert _z(probability) == pytest.approx(expected, abs=1e-6)


def test_an_impossible_probability_is_refused() -> None:
    with pytest.raises(ValueError, match=r"in \(0, 1\)"):
        _z(1.0)
