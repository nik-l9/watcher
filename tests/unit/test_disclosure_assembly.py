"""Every disclosure a payload can carry actually reaches the payload.

Written because a disclosure that is computed and not wired in is indistinguishable from one
that was never written. `data_freshness` was correct in `cortex.analysis.coverage` and absent
from the assembler for the length of one edit, and the only thing that would have caught it is a
test at this seam rather than at the function's own.

The seam matters twice over: `series_disclosures` is the single assembler the eval harness calls
as well as the connectors, and the reason it exists is that the suite once measured *none* of
these — the harness replaces each capability's handler, so no connector method ran and two
consecutive clean runs said nothing about any disclosure at all.
"""

from __future__ import annotations

from datetime import date

from cortex.tools.disclosures import series_disclosures

#: The monthly world that produced the worst real answer this project has recorded: signup
#: collection stopped on 3 August, read on 7 September, at a granularity whose bucket grid
#: reports no gap because August finished and August had data in it.
DEAD_PIPELINE = [
    {
        "bucket": f"2026-{month:02d}-01T00:00:00",
        "value": 4000,
        "last_event": f"2026-{month:02d}-28T12:00:00",
    }
    for month in range(1, 8)
] + [{"bucket": "2026-08-01T00:00:00", "value": 619, "last_event": "2026-08-03T21:14:00"}]

REQUEST = {"start_date": "2026-01-01", "end_date": "2026-09-07", "interval": "month"}


class TestTheAssemblerCarriesTheFreshnessClock:
    def test_a_monthly_dead_pipeline_is_disclosed_through_the_assembler(self) -> None:
        found = series_disclosures(
            "posthog__event_trend", {"series": DEAD_PIPELINE}, REQUEST, as_of=date(2026, 9, 7)
        )
        assert found["data_freshness"]["days_short"] == 35
        assert found["data_freshness"]["last_event"] == "2026-08-03"

    def test_the_bucket_grid_alone_would_have_said_nothing(self) -> None:
        """The reason this disclosure exists, asserted rather than described.

        `series_ends_early` is the grid's answer and it is silent here — its tolerance is two
        bucket spans, so a monthly series may be 59 days stale before it complains. If this
        assertion ever starts failing, the two disclosures have converged and one of them is
        redundant.
        """
        found = series_disclosures(
            "posthog__event_trend", {"series": DEAD_PIPELINE}, REQUEST, as_of=date(2026, 9, 7)
        )
        assert "series_ends_early" not in found
        assert "data_freshness" in found

    def test_a_healthy_monthly_series_carries_neither(self) -> None:
        """A warning that fires on an ordinary read is a warning nobody reads on the day it
        matters."""
        healthy = [
            {
                "bucket": f"2026-{month:02d}-01T00:00:00",
                "value": 4000,
                "last_event": f"2026-{month:02d}-28T12:00:00",
            }
            for month in range(1, 9)
        ] + [{"bucket": "2026-09-01T00:00:00", "value": 800, "last_event": "2026-09-06T18:00:00"}]
        found = series_disclosures(
            "posthog__event_trend", {"series": healthy}, REQUEST, as_of=date(2026, 9, 7)
        )
        assert "data_freshness" not in found
        assert "series_ends_early" not in found

    def test_a_daily_read_of_the_same_world_agrees_with_the_monthly_one(self) -> None:
        """The property the granularity-dependent disclosure did not have. Both readings of one
        dead pipeline must report it; the original failure was that only the daily one did."""
        daily = [
            {
                "bucket": f"2026-08-{day:02d}T00:00:00",
                "value": 200,
                "last_event": f"2026-08-{day:02d}T20:00:00",
            }
            for day in range(1, 4)
        ]
        as_daily = series_disclosures(
            "posthog__event_trend",
            {"series": daily},
            {**REQUEST, "interval": "day"},
            as_of=date(2026, 9, 7),
        )
        as_monthly = series_disclosures(
            "posthog__event_trend", {"series": DEAD_PIPELINE}, REQUEST, as_of=date(2026, 9, 7)
        )
        assert as_daily["data_freshness"]["last_event"] == "2026-08-03"
        assert as_monthly["data_freshness"]["last_event"] == "2026-08-03"
