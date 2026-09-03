"""The only significance statement available to us, and the trap in using it.

Reproduces the numbers verified against the same replica: the 2026-06-17 break reaches
`p = 0.0143`, at the floor for a 70-day window,
while in-time placebos are nowhere near rejecting.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

from cortex.analysis.changepoints import detect_level_shifts
from cortex.analysis.conformal import significance_of, significance_of_each
from cortex.analysis.series import as_calendar
from tests.analysis.conftest import (
    FIRST_BREAK,
    LAST_MEASURED,
    SECOND_BREAK,
    SERIES_START,
    replica_rows,
)


def _window(start: date, end: date):
    rows = {on: value for on, value in replica_rows().items() if start <= on <= end}
    return as_calendar(rows, start=start, end=end)


class TestItSeparatesRealBreaksFromPlacebos:
    def test_the_first_real_break_rejects_at_the_floor(self) -> None:
        result = significance_of(_window(SERIES_START, date(2026, 7, 14)), FIRST_BREAK)
        assert result is not None
        assert result.total_days == 70
        assert result.p_floor == 1 / 70
        assert result.p_value == 1 / 70
        assert result.significant()
        assert result.at_floor

    def test_the_second_real_break_rejects_too(self) -> None:
        result = significance_of(_window(FIRST_BREAK, LAST_MEASURED), SECOND_BREAK)
        assert result is not None
        assert result.significant()

    def test_an_in_time_placebo_does_not_reject(self) -> None:
        """The test earns nothing if it fires on a date where nothing happened. Two placebos
        inside the first flat regime."""
        for placebo in (date(2026, 5, 27), date(2026, 6, 3)):
            result = significance_of(_window(SERIES_START, date(2026, 6, 16)), placebo)
            assert result is not None
            assert not result.significant(), placebo
            assert result.p_value > 0.5, placebo


class TestTheWindowIsNotOptional:
    """The bug found by running this end to end rather than only on prepared windows.

    CWZ tests one break with `T0` pre-periods and `T*` post-periods, and requires both sides
    to be a single regime. Handed the whole two-break replica, a second break lands in the
    residuals, inflates the statistic under every permutation as well as the observed one, and
    the p-value collapses towards 1.
    """

    def test_the_whole_series_destroys_the_test(self, replica_intact) -> None:
        naive = significance_of(replica_intact, FIRST_BREAK)
        assert naive is not None
        # The same break that scores 0.0143 on its own window.
        assert naive.p_value > 0.2

    def test_test_each_break_windows_it_correctly(self, replica_intact) -> None:
        segmentation = detect_level_shifts(replica_intact)
        results = significance_of_each(replica_intact, segmentation)
        assert [r.at for r in results] == [FIRST_BREAK, SECOND_BREAK]
        assert results[0].p_value == 1 / 70
        assert all(r.significant() for r in results)

    def test_each_break_gets_its_neighbours_as_bounds(self, replica_intact) -> None:
        """The consequence worth stating: the floor `1/T` is set by how far apart the breaks
        are, so two breaks a fortnight apart cannot produce a small p-value however large the
        movement between them."""
        results = significance_of_each(replica_intact, detect_level_shifts(replica_intact))
        first, second = results
        assert first.total_days == 70
        assert second.total_days < first.total_days
        assert second.p_floor > first.p_floor


class TestTheFloorIsPartOfTheClaim:
    def test_the_floor_is_one_over_the_permutation_count(self) -> None:
        result = significance_of(_window(SERIES_START, date(2026, 7, 14)), FIRST_BREAK)
        assert result is not None
        assert result.p_floor == 1 / result.total_days

    def test_a_p_at_the_floor_is_labelled_as_such(self) -> None:
        """ "p = 0.014" and "p = 0.014, which is the floor" support different sentences, and the
        second cannot be compared against a p from a longer window."""
        result = significance_of(_window(SERIES_START, date(2026, 7, 14)), FIRST_BREAK)
        assert result is not None and result.at_floor
        assert "the floor for a 70-day window" in result.note()

    def test_the_note_refuses_to_imply_a_cause(self) -> None:
        result = significance_of(_window(SERIES_START, date(2026, 7, 14)), FIRST_BREAK)
        assert result is not None
        assert "says nothing about why" in result.note()


class TestItRefusesRatherThanGuesses:
    def test_a_break_with_nothing_before_it_is_untestable(self) -> None:
        assert significance_of(_window(SERIES_START, LAST_MEASURED), SERIES_START) is None

    def test_a_break_after_the_data_is_untestable(self) -> None:
        assert significance_of(_window(SERIES_START, LAST_MEASURED), date(2026, 9, 1)) is None

    def test_an_empty_calendar_is_untestable(self) -> None:
        calendar = as_calendar({}, start=date(2026, 1, 1), end=date(2026, 3, 1))
        assert significance_of(calendar, date(2026, 2, 1)) is None

    def test_a_window_too_short_for_the_proxy_is_untestable(self) -> None:
        """Intercept plus six weekday dummies needs spare observations. Below that the proxy
        fits the noise, the residuals go to zero, and any p-value would be an artefact."""
        rows = {date(2026, 1, 1) + timedelta(days=offset): 10.0 for offset in range(6)}
        calendar = as_calendar(rows, start=date(2026, 1, 1), end=date(2026, 1, 6))
        assert significance_of(calendar, date(2026, 1, 4)) is None

    def test_a_segmentation_with_no_breaks_yields_no_results(self, replica_intact) -> None:
        flat_rows = {
            SERIES_START + timedelta(days=offset): 100.0 + random.Random(offset).gauss(0, 5)
            for offset in range(90)
        }
        flat = as_calendar(flat_rows, start=SERIES_START, end=LAST_MEASURED)
        assert significance_of_each(flat, detect_level_shifts(flat)) == ()


class TestTheProxyIsFittedUnderTheNullOnEveryPeriod:
    def test_dropping_seasonality_changes_the_answer(self) -> None:
        """Not asserting which is better -- only that the weekday terms are load-bearing, so a
        future simplification cannot silently remove them."""
        window = _window(SERIES_START, date(2026, 7, 14))
        with_dow = significance_of(window, FIRST_BREAK)
        without = significance_of(window, FIRST_BREAK, seasonal=False)
        assert with_dow is not None and without is not None
        assert with_dow.statistic != without.statistic


class TestThePowerCliff:
    """Found by building this, not by reading the paper.

    The test has no power unless the post period is shorter than the pre period, because the
    intercept-plus-weekday proxy fits the weighted average of the two regimes and the longer
    regime gets the smaller residuals. Type I error is unaffected -- it stays at 1.7-3.3%
    against alpha = 0.05 in every configuration -- so this is a power cliff, not an invalidity.

    The research validated only at 42 pre against 28 post and never exercised the other side.
    """

    @staticmethod
    def _stepped(pre: int, seed: int = 4):
        generator = random.Random(seed)
        rows = {
            SERIES_START + timedelta(days=offset): (200.0 if offset < pre else 90.0)
            + generator.gauss(0, 6)
            for offset in range(90)
        }
        return as_calendar(rows, start=SERIES_START, end=LAST_MEASURED)

    def test_a_shorter_post_period_has_full_power(self) -> None:
        result = significance_of(self._stepped(70), SERIES_START + timedelta(days=70))
        assert result is not None and result.resolvable
        assert result.significant()

    def test_a_longer_post_period_has_none_and_says_so(self) -> None:
        """The dangerous case. Reporting p = 1.0 here would be technically correct and
        practically a lie."""
        result = significance_of(self._stepped(20), SERIES_START + timedelta(days=20))
        assert result is not None
        assert not result.resolvable
        assert not result.significant()

    def test_equal_periods_are_also_unresolvable(self) -> None:
        result = significance_of(self._stepped(45), SERIES_START + timedelta(days=45))
        assert result is not None and not result.resolvable

    def test_the_note_calls_it_an_inability_not_a_negative(self) -> None:
        result = significance_of(self._stepped(20), SERIES_START + timedelta(days=20))
        assert result is not None
        note = result.note()
        assert "No significance can be established" in note
        assert "could not have rejected whatever the data showed" in note
        assert "not evidence the change is unreal" in note
        assert "longer run of comparable days before the break would resolve it" in note

    def test_type_one_error_survives_the_cliff(self) -> None:
        """The reason this is a power problem and not a validity problem. Pure noise, no break,
        across both sides of the cliff."""
        for pre in (20, 70):
            rejects = 0
            for seed in range(60):
                generator = random.Random(seed)
                rows = {
                    SERIES_START + timedelta(days=offset): 200.0 + generator.gauss(0, 6)
                    for offset in range(90)
                }
                calendar = as_calendar(rows, start=SERIES_START, end=LAST_MEASURED)
                result = significance_of(calendar, SERIES_START + timedelta(days=pre))
                if result and result.significant():
                    rejects += 1
            assert rejects / 60 <= 0.10, pre
