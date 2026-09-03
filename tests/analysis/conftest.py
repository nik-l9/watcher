"""A replica of our own signup series, for tests that need a known answer.

The replica the changepoint and conformal work was measured against: 90 daily
points from 2026-05-06, means of 215 / 90 / 200 with breaks at index 42 (2026-06-17) and 70
(2026-07-15), and a weekday multiplier large enough that a naive segmenter fits the weekend
dips instead of the real shifts.

Seeded, so the numbers below are assertions rather than expectations.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

from cortex.analysis.series import Calendar, as_calendar

#: Saturday and Sunday at roughly 60% of a weekday is what makes this a real test. Without it
#: the segmentation problem is trivial at every setting.
WEEKDAY_MULTIPLIER = {0: 1.08, 1: 1.12, 2: 1.10, 3: 1.05, 4: 0.95, 5: 0.60, 6: 0.58}

SERIES_START = date(2026, 5, 6)
FIRST_BREAK = date(2026, 6, 17)
SECOND_BREAK = date(2026, 7, 15)
LAST_MEASURED = date(2026, 8, 3)
#: Fourteen days past the last row, exactly as the real outage left it.
REQUESTED_END = date(2026, 8, 17)


def replica_rows(seed: int = 7, *, days: int = 90) -> dict[date, float]:
    generator = random.Random(seed)
    rows: dict[date, float] = {}
    for offset in range(days):
        on = SERIES_START + timedelta(days=offset)
        level = 215 if offset < 42 else (90 if offset < 70 else 200)
        mean = level * WEEKDAY_MULTIPLIER[on.weekday()]
        rows[on] = float(max(0, round(generator.gauss(mean, mean**0.5))))
    return rows


@pytest.fixture
def replica() -> Calendar:
    """The series with its trailing outage, as a query would actually return it."""
    return as_calendar(replica_rows(), start=SERIES_START, end=REQUESTED_END)


@pytest.fixture
def replica_intact() -> Calendar:
    """The measured span only, for tests that are not about the gap."""
    return as_calendar(replica_rows(), start=SERIES_START, end=LAST_MEASURED)
