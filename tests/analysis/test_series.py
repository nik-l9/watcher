"""A missing day is not a zero, and a `GROUP BY` cannot tell you which it was."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from cortex.analysis.changepoints import detect_level_shifts, noise_scale
from cortex.analysis.series import DayState, as_calendar
from tests.analysis.conftest import LAST_MEASURED, REQUESTED_END, SERIES_START, replica_rows


class TestTheFailureThatIsInvisibleInAQueryResult:
    """The reason this module exists.

    Our rows stop existing on 2026-08-04. A `GROUP BY date` over a range ending 08-17 returns
    90 rows for 104 days, and that array has no zeros, no gap and no anomaly in it. Every
    detector fed those numbers correctly reports nothing wrong at the end of the series,
    because nothing in its input says otherwise. The outage is not hidden in the data; it is
    absent from it.
    """

    def test_the_raw_query_result_contains_no_trace_of_the_outage(self) -> None:
        rows = replica_rows()
        assert len(rows) == 90
        # This is what a detector would have been handed before this module existed.
        values = [rows[day] for day in sorted(rows)]
        assert 0.0 not in values
        assert min(values) > 0
        # And the last value is an ordinary weekday number, not a decline.
        assert values[-1] > 50

    def test_reindexing_is_what_makes_the_gap_exist(self) -> None:
        calendar = as_calendar(replica_rows(), start=SERIES_START, end=REQUESTED_END)
        assert len(calendar) == 104
        assert len(calendar.measured) == 90
        assert calendar.freshness.absent_days == 14

    def test_the_gap_is_reported_before_any_analysis(self) -> None:
        calendar = as_calendar(replica_rows(), start=SERIES_START, end=REQUESTED_END)
        note = calendar.freshness.note()
        assert note is not None
        assert "no rows after 2026-08-03" in note
        # The distinction is stated, not implied. "Absent is not zero" is the whole module.
        assert "Absent is not zero" in note
        assert "Do not compute a rate for the whole requested period" in note


class TestZeroAndAbsentAreDifferentThings:
    def test_a_present_zero_is_a_measurement(self) -> None:
        """The database said nobody signed up. That is data."""
        calendar = as_calendar(
            {date(2026, 1, 1): 0.0, date(2026, 1, 2): 5.0},
            start=date(2026, 1, 1),
            end=date(2026, 1, 2),
        )
        assert calendar.days[0].state is DayState.ZERO
        assert calendar.days[0].measured is True
        assert calendar.days[0].value == 0.0

    def test_a_missing_row_is_not(self) -> None:
        calendar = as_calendar(
            {date(2026, 1, 2): 5.0}, start=date(2026, 1, 1), end=date(2026, 1, 2)
        )
        assert calendar.days[0].state is DayState.ABSENT
        assert calendar.days[0].measured is False
        assert calendar.days[0].value is None

    def test_they_do_not_share_a_representation(self) -> None:
        """Filling absent days with 0 is the single most consequential mistake available here,
        and it is the one every convenience API makes for you."""
        zeroed = as_calendar({date(2026, 1, 1): 0.0}, start=date(2026, 1, 1), end=date(2026, 1, 1))
        missing = as_calendar({}, start=date(2026, 1, 1), end=date(2026, 1, 1))
        assert zeroed.values == (0.0,)
        assert missing.values == ()
        assert zeroed.freshness.complete and not missing.freshness.complete


class TestATrailingGapIsNotTheSameAsAHole:
    def test_a_stopped_loader_is_counted_as_trailing(self) -> None:
        calendar = as_calendar(replica_rows(), start=SERIES_START, end=REQUESTED_END)
        assert calendar.freshness.trailing_absent_days == 14
        assert calendar.freshness.interior_absent_days == 0
        assert calendar.freshness.stops_early is True
        assert calendar.freshness.last_measured == LAST_MEASURED

    def test_an_interior_hole_is_counted_separately(self) -> None:
        """A hole and a stop have different causes and different consequences: a stop
        invalidates a rate for the requested period, a hole weakens it."""
        rows = replica_rows()
        del rows[date(2026, 6, 1)]
        del rows[date(2026, 6, 2)]
        calendar = as_calendar(rows, start=SERIES_START, end=LAST_MEASURED)
        assert calendar.freshness.interior_absent_days == 2
        assert calendar.freshness.trailing_absent_days == 0
        note = calendar.freshness.note()
        assert note is not None and "a hole rather than a stop" in note

    def test_a_complete_series_says_nothing(self) -> None:
        """Every ordinary call would otherwise carry a disclosure, which is how a disclosure
        stops being read on the call that needed it."""
        calendar = as_calendar(replica_rows(), start=SERIES_START, end=LAST_MEASURED)
        assert calendar.freshness.complete
        assert calendar.freshness.note() is None

    def test_an_empty_result_is_all_trailing(self) -> None:
        """Nothing measured at all: there is no point after which data exists, so the whole
        range is trailing rather than a hole in the middle of nothing."""
        calendar = as_calendar({}, start=date(2026, 1, 1), end=date(2026, 1, 10))
        assert calendar.freshness.trailing_absent_days == 10
        assert calendar.freshness.interior_absent_days == 0
        assert calendar.freshness.last_measured is None
        assert "no rows anywhere in this range" in (calendar.freshness.note() or "")


class TestTheContiguousSpanIsWhatMayBeAnalysed:
    def test_it_stops_at_the_outage(self) -> None:
        calendar = as_calendar(replica_rows(), start=SERIES_START, end=REQUESTED_END)
        assert calendar.contiguous_span() == (SERIES_START, LAST_MEASURED)

    def test_it_picks_the_longest_run_not_the_first(self) -> None:
        rows = {date(2026, 1, 1): 1.0, date(2026, 1, 5): 1.0, date(2026, 1, 6): 1.0}
        calendar = as_calendar(rows, start=date(2026, 1, 1), end=date(2026, 1, 6))
        assert calendar.contiguous_span() == (date(2026, 1, 5), date(2026, 1, 6))

    def test_nothing_measured_has_no_span(self) -> None:
        calendar = as_calendar({}, start=date(2026, 1, 1), end=date(2026, 1, 3))
        assert calendar.contiguous_span() is None


class TestTheNoiseScaleMustNotSeeTheGap:
    """The documented silent failure, reproduced.

    Zero-filling the absent days makes fourteen of the successive differences approximately
    zero. The median falls, the penalty falls with it, and the segmenter starts spending
    changepoints on the weekly cycle.
    """

    def test_zero_filling_collapses_the_noise_estimate(self) -> None:
        calendar = as_calendar(replica_rows(), start=SERIES_START, end=REQUESTED_END)
        honest = noise_scale(calendar.values)
        zero_filled = noise_scale(
            tuple(day.value if day.value is not None else 0.0 for day in calendar.days)
        )
        assert zero_filled < honest

    def test_the_segmenter_uses_the_measured_span_only(self) -> None:
        """Not asserted by inspection -- the segmentation over the gapped calendar must equal
        the segmentation over the intact one, because the gap contributes nothing either way."""
        gapped = as_calendar(replica_rows(), start=SERIES_START, end=REQUESTED_END)
        intact = as_calendar(replica_rows(), start=SERIES_START, end=LAST_MEASURED)
        assert detect_level_shifts(gapped).breaks == detect_level_shifts(intact).breaks


def test_an_inverted_range_is_refused() -> None:
    with pytest.raises(ValueError, match="precedes start"):
        as_calendar({}, start=date(2026, 2, 1), end=date(2026, 1, 1))


def test_the_calendar_has_no_missing_days_by_construction() -> None:
    calendar = as_calendar(replica_rows(), start=SERIES_START, end=REQUESTED_END)
    for earlier, later in zip(calendar.days, calendar.days[1:], strict=False):
        assert later.on - earlier.on == timedelta(days=1)
