"""Minimum segment length is the parameter that decides whether this works at all."""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

from cortex.analysis.changepoints import (
    DEFAULT_MIN_LEN,
    detect_level_shifts,
    noise_scale,
)
from cortex.analysis.series import as_calendar
from tests.analysis.conftest import (
    FIRST_BREAK,
    LAST_MEASURED,
    SECOND_BREAK,
    SERIES_START,
    replica_rows,
)

TRUTH = (FIRST_BREAK, SECOND_BREAK)


def _segment(seed: int, min_len: int):
    calendar = as_calendar(replica_rows(seed), start=SERIES_START, end=LAST_MEASURED)
    return detect_level_shifts(calendar, min_len=min_len)


class TestItFindsTheRealBreaks:
    def test_both_breaks_recovered_at_the_default(self, replica_intact) -> None:
        assert detect_level_shifts(replica_intact).breaks == TRUTH

    def test_the_levels_either_side_are_reported(self, replica_intact) -> None:
        """A break with no magnitude is not a finding. 215 to 90 to 200 by construction."""
        segmentation = detect_level_shifts(replica_intact)
        before, after = segmentation.shift_at(FIRST_BREAK)
        assert 180 < before < 220
        assert 70 < after < 100
        before, after = segmentation.shift_at(SECOND_BREAK)
        assert 70 < before < 100
        assert 170 < after < 215

    def test_shift_at_a_non_break_is_none(self, replica_intact) -> None:
        assert detect_level_shifts(replica_intact).shift_at(date(2026, 6, 1)) is None


class TestMinimumSegmentLengthIsTheLever:
    """Recovery over 200 seeds, reproducing the research's own experiment.

    Its table: 0/200 at min_len=1, 109/200 at 7, 189/200 at 14. A weekly cycle is real
    structure that a change-in-mean model can only express as changepoints, so raising the
    penalty does not fix it -- only a floor of one full seasonal period does.

    Thresholds here are deliberately loose. The claim under test is the *ordering and
    magnitude*, not a specific count from a specific RNG.
    """

    @pytest.mark.parametrize(
        ("min_len", "at_most", "at_least"),
        [(1, 5, 0), (7, None, 80), (14, None, 150)],
    )
    def test_recovery_rises_steeply_with_the_floor(
        self, min_len: int, at_most: int | None, at_least: int
    ) -> None:
        exact = sum(1 for seed in range(200) if set(_segment(seed, min_len).breaks) == set(TRUTH))
        assert exact >= at_least
        if at_most is not None:
            assert exact <= at_most

    def test_no_floor_fits_the_weekend_dips(self) -> None:
        """The failure is not subtle: it spends twenty changepoints on the weekly cycle."""
        assert len(_segment(7, 1).breaks) > 10

    def test_a_higher_penalty_does_not_rescue_a_missing_floor(self, replica_intact) -> None:
        """Measured in the research: 2*log(n) to 5*log(n) took 23 detections down to 17."""
        loose = detect_level_shifts(replica_intact, min_len=1, penalty=2.0)
        strict = detect_level_shifts(replica_intact, min_len=1, penalty=5.0)
        assert len(strict.breaks) > 5
        assert len(loose.breaks) > 5

    def test_the_default_is_two_weeks_not_one(self) -> None:
        """189/200 against 109/200. The cost is declared in the module docstring rather than
        discovered later: nothing shorter than a fortnight can be a level shift."""
        assert DEFAULT_MIN_LEN == 14


class TestTheFloorIsAHardLimitOnDetectableDuration:
    def test_a_dip_shorter_than_the_floor_is_misdated_rather_than_missed(self) -> None:
        """The floor's real cost, and it is worse than the research's table implies.

        That table scored "recovery" as both endpoints within a day, and reported 0% for a
        10-day dip at `min_len=14`. What actually happens is not silence: the segmenter finds
        the dip and *stretches it* to satisfy the floor. A 10-day dip starting on 2026-06-15 is
        reported as starting 2026-06-12 -- three days early, with an end pushed out to match.

        A misdated level shift is more dangerous than an undetected one, because the onset date
        is what every downstream step keys on: `conformal.test_each_break` windows on it, and
        ADR 0005's elimination rule compares candidate causes against it. A cause that
        genuinely explains a 06-15 onset can be eliminated for not explaining 06-12.

        Pinned rather than fixed, because the fix is a separate short-window detector for
        incidents (research section 3.7) rather than a lower floor -- lowering it reintroduces
        the weekly-seasonality failure this floor exists to prevent.
        """
        generator = random.Random(11)
        rows = {
            SERIES_START + timedelta(days=offset): (60.0 if 40 <= offset < 50 else 200.0)
            + generator.gauss(0, 6)
            for offset in range(90)
        }
        calendar = as_calendar(rows, start=SERIES_START, end=LAST_MEASURED)
        true_onset = SERIES_START + timedelta(days=40)

        # A floor below the dip's length dates the onset exactly.
        tight = detect_level_shifts(calendar, min_len=7)
        assert tight.breaks[0] == true_onset

        floored = detect_level_shifts(calendar, min_len=14)
        assert len(floored.breaks) == 2
        # Found, and wrong by more than the tighter floor is.
        assert floored.breaks[0] < tight.breaks[0]
        assert (floored.breaks[1] - floored.breaks[0]).days >= 14


class TestItDoesNotInventStructure:
    def test_a_flat_series_has_no_breaks(self) -> None:
        rows = {SERIES_START + timedelta(days=offset): 100.0 for offset in range(90)}
        calendar = as_calendar(rows, start=SERIES_START, end=LAST_MEASURED)
        segmentation = detect_level_shifts(calendar)
        assert segmentation.breaks == ()
        assert segmentation.found_nothing

    def test_a_series_shorter_than_two_segments_is_one_segment(self) -> None:
        rows = {SERIES_START + timedelta(days=offset): 100.0 + offset for offset in range(20)}
        calendar = as_calendar(rows, start=SERIES_START, end=SERIES_START + timedelta(days=19))
        assert detect_level_shifts(calendar, min_len=14).breaks == ()

    def test_an_empty_calendar_yields_nothing(self) -> None:
        calendar = as_calendar({}, start=date(2026, 1, 1), end=date(2026, 3, 1))
        segmentation = detect_level_shifts(calendar)
        assert segmentation.segments == () and segmentation.breaks == ()

    def test_a_cessation_is_not_reported_as_a_level_shift(self, replica) -> None:
        """It is the absence of a measurement, it is already in `Freshness`, and calling it a
        changepoint would file it alongside a real movement."""
        breaks = detect_level_shifts(replica).breaks
        assert date(2026, 8, 4) not in breaks
        assert breaks == TRUTH

    def test_min_len_below_one_is_refused(self, replica_intact) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            detect_level_shifts(replica_intact, min_len=0)


class TestTheNoiseScale:
    def test_it_uses_differences_not_deviations(self) -> None:
        """A series with level shifts in it by assumption. A global mean would absorb them
        into the noise estimate, inflating it by orders of magnitude, and then nothing would
        look significant."""
        generator = random.Random(3)
        step = [10.0 + generator.gauss(0, 1) for _ in range(20)] + [
            200.0 + generator.gauss(0, 1) for _ in range(20)
        ]
        from statistics import mean, pvariance

        assert noise_scale(tuple(step)) < 10.0
        # For contrast: the variance about the global mean is dominated by the step itself.
        assert pvariance(step, mu=mean(step)) > 8000.0

    def test_a_degenerate_series_yields_no_scale_rather_than_a_guess(self) -> None:
        """Over half the differences exactly zero means the estimator has broken down. The
        tempting fallback -- average the non-zero differences -- estimates the *jumps* as the
        noise on a piecewise-constant series, inflates the penalty, and hides every break."""
        assert noise_scale(tuple([200.0] * 40 + [60.0] * 40)) == 0.0

    def test_a_series_with_no_estimable_noise_reports_one_segment(self) -> None:
        """The consequence of the above, stated where a caller meets it: we cannot call a split
        significant when we cannot say what noise looks like."""
        rows = {SERIES_START + timedelta(days=o): (200.0 if o < 45 else 60.0) for o in range(90)}
        calendar = as_calendar(rows, start=SERIES_START, end=LAST_MEASURED)
        segmentation = detect_level_shifts(calendar)
        assert segmentation.noise_variance == 0.0
        assert segmentation.breaks == ()

    def test_a_single_point_has_no_scale(self) -> None:
        assert noise_scale((5.0,)) == 0.0
        assert noise_scale(()) == 0.0
