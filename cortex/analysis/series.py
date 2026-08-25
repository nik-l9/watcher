"""A daily series as a calendar, where a missing day is a different thing from a zero.

**The failure this exists to prevent is silent, and it is a query shape rather than a bug in
any detector.** Our signup rows stop existing on 2026-08-04. A `GROUP BY date` over
2026-05-06..2026-08-17 returns 90 rows for 104 calendar days, and that array contains no
zeros, no gap, and no anomaly. Every changepoint method in the literature, fed those 90
numbers, correctly reports no change at the end of the series -- because nothing in its input
says otherwise. The outage is not hidden in the data; it is absent from it.

So the first act of any analysis is to reindex against a generated calendar, and to keep
three states rather than two:

- ``OBSERVED``   -- a row exists and the count is positive.
- ``ZERO``       -- a row exists and the count is zero. Nobody signed up.
- ``ABSENT``     -- no row exists. We do not know what happened.

``ZERO`` and ``ABSENT`` demand opposite conclusions and must not share a representation.
Filling absent days with 0 is the single most consequential mistake available here, and it is
the one every convenience API makes for you: it manufactures evidence of an event ("signups
went to zero!") out of the absence of evidence, and it corrupts the noise estimate that every
downstream test depends on.

Provenance: `docs/research/metric-attribution.md` section 3.7, which demonstrated each of
these on a replica of our own series.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

__all__ = ["Calendar", "Day", "DayState", "Freshness", "as_calendar"]


class DayState(enum.StrEnum):
    """Whether a day was measured, measured as nothing, or not measured at all."""

    OBSERVED = "observed"
    ZERO = "zero"
    ABSENT = "absent"


@dataclass(frozen=True, slots=True)
class Day:
    """One calendar day, with the distinction the whole module exists to preserve."""

    on: date
    value: float | None
    state: DayState

    @property
    def measured(self) -> bool:
        """Whether this day contributes a number. `ZERO` does; `ABSENT` does not."""
        return self.state is not DayState.ABSENT


@dataclass(frozen=True, slots=True)
class Freshness:
    """How far the data actually reaches, and where it is holed.

    Reported *before* any analysis rather than alongside it. A trailing gap is not a caveat on
    an answer computed over the requested range -- it changes which range can be answered at
    all, and section 3.7 is explicit that the correct output is "the table has no rows after
    <date>; everything below covers data through that date", not a number.
    """

    requested_from: date
    requested_to: date
    last_measured: date | None
    absent_days: int
    #: Absent days at the end of the range, which is the signature of a stopped loader rather
    #: than of a hole. Counted separately because it is the one that invalidates a rate.
    trailing_absent_days: int
    #: Absent days that have measured days on both sides.
    interior_absent_days: int

    @property
    def complete(self) -> bool:
        return self.absent_days == 0

    @property
    def stops_early(self) -> bool:
        return self.trailing_absent_days > 0

    def note(self) -> str | None:
        """A sentence for a reader, or None when there is nothing to disclose."""
        if self.complete:
            return None
        parts: list[str] = []
        if self.trailing_absent_days:
            reach = (
                f"has no rows after {self.last_measured.isoformat()}"
                if self.last_measured
                else "has no rows anywhere in this range"
            )
            parts.append(
                f"The series {reach}, though the range runs to "
                f"{self.requested_to.isoformat()} -- {self.trailing_absent_days} day(s) are "
                "absent from the end. Absent is not zero: no row exists, so we do not know "
                "what happened. Do not compute a rate for the whole requested period."
            )
        if self.interior_absent_days:
            parts.append(
                f"{self.interior_absent_days} day(s) inside the range have no rows either, "
                "which is a hole rather than a stop."
            )
        return " ".join(parts)


@dataclass(frozen=True, slots=True)
class Calendar:
    """A complete, gap-explicit daily series.

    Iterating gives every day in the requested range, in order, including the ones that were
    never measured. That is the point: a caller cannot accidentally skip a gap it did not know
    about, because the gap is an element.
    """

    days: tuple[Day, ...]
    freshness: Freshness

    def __len__(self) -> int:
        return len(self.days)

    def __iter__(self) -> Iterable[Day]:
        return iter(self.days)

    @property
    def measured(self) -> tuple[Day, ...]:
        """Only the days that carry a number.

        **The noise scale must be estimated over this, never over the full calendar.**
        Verified in section 3.7: zero-filling fourteen absent days and then computing
        `median(diff^2)/2` over the padded array drags the estimate down, because fourteen of
        the differences become approximately zero. The shrunken scale shrinks the penalty and
        the segmenter starts spending changepoints on weekly seasonality -- it returned five
        breakpoints, two of them artefacts, where the same call with the scale estimated on
        the observed span returned exactly the three real ones.
        """
        return tuple(day for day in self.days if day.measured)

    @property
    def values(self) -> tuple[float, ...]:
        """The measured numbers alone, for a routine that cannot express a gap.

        Deliberately awkward to reach, and not the default. Anything consuming this has lost
        the calendar and therefore cannot see an outage.
        """
        return tuple(day.value for day in self.measured if day.value is not None)

    def contiguous_span(self) -> tuple[date, date] | None:
        """The longest run of consecutive measured days, or None if nothing was measured.

        What a changepoint routine may legitimately be handed. Section 3.7's rule is that a
        detector run across a gap is describing a series that does not exist, so the honest
        move is to analyse the intact span and report the gap separately.
        """
        best: tuple[int, int] | None = None
        run_start: int | None = None
        for index, day in enumerate(self.days):
            if day.measured:
                run_start = index if run_start is None else run_start
                if best is None or index - run_start > best[1] - best[0]:
                    best = (run_start, index)
            else:
                run_start = None
        if best is None:
            return None
        return (self.days[best[0]].on, self.days[best[1]].on)


def as_calendar(
    rows: Mapping[date, float] | Sequence[tuple[date, float]],
    *,
    start: date,
    end: date,
) -> Calendar:
    """Reindex query output against a generated calendar.

    `rows` carries only the days the query returned. Every other day in `[start, end]` becomes
    `ABSENT`, which is the step that makes the outage visible at all.

    A day present with value 0 becomes `ZERO`, not `ABSENT`. The database said nothing
    happened; that is a measurement.
    """
    if end < start:
        raise ValueError(f"end {end} precedes start {start}")

    observed: dict[date, float] = dict(rows.items() if isinstance(rows, Mapping) else rows)

    days: list[Day] = []
    cursor = start
    while cursor <= end:
        if cursor in observed:
            value = float(observed[cursor])
            state = DayState.ZERO if value == 0 else DayState.OBSERVED
            days.append(Day(on=cursor, value=value, state=state))
        else:
            days.append(Day(on=cursor, value=None, state=DayState.ABSENT))
        cursor += timedelta(days=1)

    measured_indices = [i for i, day in enumerate(days) if day.measured]
    absent = len(days) - len(measured_indices)
    if measured_indices:
        last = measured_indices[-1]
        trailing = len(days) - 1 - last
        interior = absent - trailing
        last_measured: date | None = days[last].on
    else:
        # Nothing measured at all. Every day is trailing by the definition that matters --
        # there is no point after which data exists.
        trailing, interior, last_measured = len(days), 0, None

    return Calendar(
        days=tuple(days),
        freshness=Freshness(
            requested_from=start,
            requested_to=end,
            last_measured=last_measured,
            absent_days=absent,
            trailing_absent_days=trailing,
            interior_absent_days=interior,
        ),
    )
