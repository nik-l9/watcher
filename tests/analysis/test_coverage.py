"""A whole bucket finishing inside the range and reporting nothing.

Not "how long since the last bucket", which is what this measured first and which produced a
false positive the evaluation suite caught the day it started computing real disclosures over its
fixtures: `campaign_traffic_drop` has a weekly series whose last bucket starts on 22 June and so
covers through the 28th, against a range ending on the 30th. The only thing absent is the
in-progress week of the 29th. Measured from the bucket start that reads as 8 days missing against
7 days of slack, and the scenario would have been told its collection had stopped -- in a scenario
about a marketing campaign, where a data-incident verdict outranks and replaces the real answer.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from cortex.analysis.coverage import (
    BUCKET_DAYS,
    FRESHNESS_TOLERANCE_DAYS,
    freshness_disclosure,
    gap_disclosure,
)

ALTERNATIVES = "it may have stopped, or tracking may be broken"


def _trips(buckets: list[date], end: date, interval: str) -> bool:
    return (
        gap_disclosure(buckets, window_end=end, interval=interval, alternatives=ALTERNATIVES)
        is not None
    )


class TestAnIncompleteFinalBucketIsNormal:
    """A live source always has one. Disclosing it on every call costs the disclosure its value
    on the one call that matters."""

    def test_a_daily_series_ending_yesterday_is_silent(self) -> None:
        assert not _trips([date(2026, 8, 14)], date(2026, 8, 15), "day")

    def test_a_weekly_series_mid_week_is_silent(self) -> None:
        """The regression. Last bucket starts the 22nd and covers to the 28th; the range ends on
        the 30th, so only the in-progress week of the 29th is absent."""
        weeks = [date(2026, 6, 1), date(2026, 6, 8), date(2026, 6, 15), date(2026, 6, 22)]
        assert not _trips(weeks, date(2026, 6, 30), "week")

    def test_a_monthly_series_mid_month_is_silent(self) -> None:
        assert not _trips([date(2026, 7, 1), date(2026, 8, 1)], date(2026, 8, 12), "month")


class TestAMissingCompleteBucketIsDisclosed:
    def test_the_failure_this_exists_for(self) -> None:
        """The signups series that ran 1-3 August against a range ending on the 15th, from which
        the analyst computed "the August run rate" and found no decline."""
        found = gap_disclosure(
            [date(2026, 8, 1), date(2026, 8, 2), date(2026, 8, 3)],
            window_end=date(2026, 8, 15),
            interval="day",
            alternatives=ALTERNATIVES,
        )
        assert found is not None
        assert found["series_ends_early"]["days_missing"] == 12

    def test_whole_weeks_missing_are_disclosed(self) -> None:
        assert _trips([date(2026, 6, 1)], date(2026, 6, 30), "week")

    def test_the_boundary_is_one_complete_bucket(self) -> None:
        """Exactly one whole day finishing inside the range with nothing in it.

        With data to the 14th and a range to the 15th, the 15th is the range's own final bucket
        and may still be filling -- silent. Extend the range to the 16th and the 15th has now
        *finished*, having reported nothing, which is the thing worth saying.
        """
        assert not _trips([date(2026, 8, 14)], date(2026, 8, 15), "day")
        assert _trips([date(2026, 8, 14)], date(2026, 8, 16), "day")


class TestAMonthIsTreatedAsItsLongestPossibleLength:
    def test_february_is_not_reported_missing_after_28_days(self) -> None:
        """`BUCKET_DAYS` uses 31 for a month rather than the real length, which makes the test
        conservative: it will not claim a short month went missing on the strength of 28 days
        having passed."""
        assert BUCKET_DAYS["month"] == 31
        assert not _trips([date(2026, 1, 1)], date(2026, 3, 1), "month")


class TestNothingToSay:
    @pytest.mark.parametrize("interval", ["day", "week", "month"])
    def test_an_empty_series_is_silent(self, interval: str) -> None:
        """Already reported through the payload's own count; a second voice adds noise."""
        assert not _trips([], date(2026, 8, 15), interval)

    def test_an_unknown_interval_falls_back_to_a_day(self) -> None:
        assert _trips([date(2026, 8, 1)], date(2026, 8, 15), "fortnight")


class TestADayThatHasNotHappenedCannotBeMissing:
    """Without the clamp this fires on the most ordinary request there is.

    An analyst asking PostHog for 1-31 August on the 24th, with data through the 23rd, was told
    the series "has no data after 2026-08-23 -- 8 days are missing entirely... the event may have
    stopped firing". Seven of those eight days were in the future. Alarming, wrong, and triggered
    by a question about the current month.
    """

    def test_asking_about_the_rest_of_the_month_is_silent(self) -> None:
        assert not _trips_as_of(
            [date(2026, 8, day) for day in range(1, 24)],
            end=date(2026, 8, 31),
            as_of=date(2026, 8, 24),
        )

    def test_a_real_stop_still_trips_with_a_clamp_in_place(self) -> None:
        """The clamp must not become a way for a genuine gap to hide behind a wide request."""
        assert _trips_as_of(
            [date(2026, 8, 1), date(2026, 8, 3)],
            end=date(2026, 8, 31),
            as_of=date(2026, 8, 24),
        )

    def test_a_horizon_after_the_range_changes_nothing(self) -> None:
        """It clamps, it does not extend: a request narrower than the horizon keeps its own end,
        so an analyst asking about last week is not told about this week."""
        assert not _trips_as_of(
            [date(2026, 8, 1), date(2026, 8, 3)],
            end=date(2026, 8, 4),
            as_of=date(2026, 8, 24),
        )

    def test_without_a_horizon_the_behaviour_is_unchanged(self) -> None:
        """Defaulted rather than required, so nothing silently changes for a caller that has no
        opinion about when data stops being possible."""
        assert _trips([date(2026, 8, 1), date(2026, 8, 3)], date(2026, 8, 15), "day")


def _trips_as_of(buckets: list[date], *, end: date, as_of: date) -> bool:
    return (
        gap_disclosure(
            buckets,
            window_end=end,
            interval="day",
            alternatives=ALTERNATIVES,
            as_of=as_of,
        )
        is not None
    )


class TestTheSecondClock:
    """Staleness against the clock, because the bucket grid cannot see it.

    `gap_disclosure` asks whether a *complete bucket finished inside the range and reported
    nothing*. That is the right question about the grid, and a coarse interval answers it "no"
    for a very long time: its tolerance is two bucket spans, so at monthly granularity a series
    whose collection died on 3 August reports no gap when read on 7 September. August finished,
    and August had data in it.

    That is not a hypothetical. A live investigation asked for monthly signups through 7
    September, was told nothing, saw a small August bucket and wrote *"August's 580 ... likely
    reflect an incomplete trailing month/data lag rather than a genuine drop"*. Collection had
    stopped 35 days earlier and 52 unrelated events had stopped with it. **The same world asked
    daily produced the full warning** — so the disclosure was granularity-dependent, which is
    the one property a data-quality warning must never have.
    """

    ALTERNATIVES = "Collection may be broken, or the event may have been renamed"

    def _disclose(self, last: date | None, *, end: date, as_of: date | None = None) -> dict | None:
        return freshness_disclosure(
            last, window_end=end, as_of=as_of, alternatives=self.ALTERNATIVES
        )

    def test_a_monthly_series_cannot_hide_a_dead_pipeline(self) -> None:
        """The case that produced the wrong answer, at the granularity that hid it."""
        found = self._disclose(date(2026, 8, 3), end=date(2026, 9, 7), as_of=date(2026, 9, 7))
        assert found is not None
        assert found["data_freshness"] == {
            "last_event": "2026-08-03",
            "period_end": "2026-09-07",
            "days_short": 35,
        }
        assert "The period is not young; the data stops." in found["data_freshness_note"]

    def test_it_says_the_same_thing_at_every_granularity(self) -> None:
        """The whole point. `gap_disclosure` takes an interval and this does not, because a
        fortnight of missing data is a fortnight whether it is read in days or months."""
        import inspect

        assert "interval" not in inspect.signature(freshness_disclosure).parameters

    def test_a_source_running_a_day_behind_is_the_normal_state(self) -> None:
        """A daily series ending yesterday is a live source, not a defect. A warning that
        fires on every ordinary read is a warning nobody reads on the day it matters."""
        assert (
            self._disclose(date(2026, 9, 6), end=date(2026, 9, 7), as_of=date(2026, 9, 7)) is None
        )

    def test_the_tolerance_is_the_stated_one(self) -> None:
        """Pinned, so an edit to the constant cannot silently change when this fires."""
        assert FRESHNESS_TOLERANCE_DAYS == 2
        at_the_limit = date(2026, 9, 7) - timedelta(days=FRESHNESS_TOLERANCE_DAYS)
        assert self._disclose(at_the_limit, end=date(2026, 9, 7)) is None
        past_it = at_the_limit - timedelta(days=1)
        assert self._disclose(past_it, end=date(2026, 9, 7)) is not None

    def test_days_that_have_not_happened_are_not_missing(self) -> None:
        """The same clamp `gap_disclosure` applies. Without it, every question whose range runs
        into the future reports a collection failure for the part still to come — and asking
        about the current quarter is an ordinary request."""
        assert (
            self._disclose(date(2026, 9, 6), end=date(2026, 12, 31), as_of=date(2026, 9, 7)) is None
        )

    def test_a_series_with_no_last_observation_says_nothing(self) -> None:
        """An empty series is reported through the payload's own count, and a second voice
        saying the same thing adds noise rather than information."""
        assert self._disclose(None, end=date(2026, 9, 7)) is None

    def test_the_note_points_at_what_would_settle_it(self) -> None:
        """Not interpreted, like its neighbour: a stale source can be a broken pipeline, a
        rename, or a real stop, and the connector reports the shortfall and the candidates
        rather than guessing between them."""
        note = self._disclose(date(2026, 8, 3), end=date(2026, 9, 7))["data_freshness_note"]
        assert self.ALTERNATIVES in note
        assert "is not evidence of a fall in the metric until the stop is explained" in note
