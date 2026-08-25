"""When did the series change level.

Exact optimal partitioning (Jackson et al. 2005) with an L2 change-in-mean cost, a BIC-scale
penalty, and a **minimum segment length**, which is the parameter that actually decides whether
this works.

Why exact rather than PELT or binary segmentation: at n=90 the pure-Python O(n^2) inner loop
runs in about 2 ms, so approximation buys nothing. Measured in
`docs/research/metric-attribution.md` section 3.6, which also tested the alternatives and
rejected them -- CUSUM fires on weekly seasonality, and Prophet cannot represent a level shift
at all.

**The minimum segment length is the whole ballgame, and the numbers are stark.** Against a
replica of our own series with a realistic weekday cycle, scoring exact recovery of two known
breaks over 200 seeds:

    min_len =  1 day      0 / 200      median 20 changepoints found
    min_len =  5 days    19 / 200      median  7
    min_len =  7 days   109 / 200      median  2
    min_len = 10 days   141 / 200      median  2
    min_len = 14 days   189 / 200      median  2

A weekly cycle is *real* structure that a change-in-mean model can only express as
changepoints, so raising the penalty does not fix it -- going from 2*log(n) to 5*log(n) took 23
detections down to 17. One full seasonal period as a floor is what stops it.

Two findings from the same experiments that contradict the obvious moves:

- **Day-of-week normalisation is not a substitute and makes things worse when combined.**
  Deseasonalising with `min_len=1` scored 59/200. Deseasonalising *and* `min_len=14` scored
  167/200 -- worse than `min_len=14` on the raw series at 189/200. On 90 days each weekday has
  about 13 observations spread across three level regimes, so the weekday factors are
  themselves badly estimated and inject error. Prefer the structural constraint to the
  seasonal adjustment until the history is a year long.
- **`min_len` is a hard floor on detectable duration, and that is a declaration rather than a
  defect.** At `min_len=14` a 10-day dip is invisible (0% recovery); a 14-day dip is found 83%
  of the time. A five-day dip genuinely is better described as an incident than as a level
  shift, so it belongs to a different detector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from cortex.analysis.series import Calendar

__all__ = ["Segment", "Segmentation", "detect_level_shifts", "noise_scale"]

#: Default minimum segment length, in days.
#:
#: Two weeks, not one. 189/200 against 109/200, and the cost is declared rather than hidden:
#: nothing shorter than a fortnight can be reported as a level shift.
DEFAULT_MIN_LEN = 14

#: Penalty coefficient on `log(n) * s^2`.
#:
#: BIC scale. Section 3.6 found 2, 3 and 5 all recovered the breaks once `min_len` was set, so
#: this is not the sensitive knob -- 3 is the midpoint of a range that all worked.
DEFAULT_PENALTY = 3.0


def noise_scale(values: tuple[float, ...] | list[float]) -> float:
    """Robust noise variance, from successive differences.

    `s^2 = median(diff^2) / 2`. Differences rather than deviations from a mean, because the
    series has level shifts in it by assumption and a global mean would absorb them into the
    noise estimate.

    **Must be given measured days only.** Passing a zero-filled calendar is the documented
    silent failure: absent days become near-zero differences, the median falls, the penalty
    shrinks, and the segmenter starts reporting weekend dips as level shifts.
    """
    if len(values) < 2:
        return 0.0
    squared = sorted((values[i + 1] - values[i]) ** 2 for i in range(len(values) - 1))
    middle = len(squared) // 2
    median = squared[middle] if len(squared) % 2 else (squared[middle - 1] + squared[middle]) / 2.0
    # A zero median means more than half the successive differences are *exactly* zero, so
    # the estimator has degenerated. Returning 0 is deliberate and `detect_level_shifts` treats
    # it as "no significance can be attached to any split".
    #
    # The tempting fallback -- average the differences that are not zero -- is wrong, and
    # wrong in the direction that hides everything: on a piecewise-constant series the only
    # non-zero differences *are* the jumps, so it estimates the signal as the noise, inflates
    # the penalty, and reports no breaks at all. Measured while writing this: it returned a
    # scale of 9800 for a series whose steps were the thing being looked for.
    return median / 2.0


@dataclass(frozen=True, slots=True)
class Segment:
    """One run of days modelled as having a single level."""

    start: date
    end: date
    #: Inclusive index bounds into the analysed span, kept so a caller can slice the input.
    start_index: int
    end_index: int
    mean: float
    days: int


@dataclass(frozen=True, slots=True)
class Segmentation:
    """The result: segments, and the dates between them where the level changed."""

    segments: tuple[Segment, ...]
    #: The first day of each segment after the first -- the onset of each shift.
    breaks: tuple[date, ...]
    min_len: int
    penalty: float
    noise_variance: float

    @property
    def found_nothing(self) -> bool:
        return len(self.segments) <= 1

    def shift_at(self, when: date) -> tuple[float, float] | None:
        """The (before, after) means at a break, or None if `when` is not one."""
        for index, segment in enumerate(self.segments):
            if index and segment.start == when:
                return (self.segments[index - 1].mean, segment.mean)
        return None


def detect_level_shifts(
    calendar: Calendar,
    *,
    min_len: int = DEFAULT_MIN_LEN,
    penalty: float = DEFAULT_PENALTY,
) -> Segmentation:
    """Segment the calendar's longest intact measured span.

    Runs over the contiguous span rather than the whole calendar, because a detector run
    across a gap is describing a series that does not exist. The gap is the `Calendar`'s
    freshness note, and it is a separate finding rather than an input to this one.

    A cessation is deliberately **not** reported here. It is not a change in level -- it is the
    absence of a measurement, it is already in `Freshness`, and calling it a changepoint would
    put it in the same category as a real movement.
    """
    if min_len < 1:
        raise ValueError(f"min_len must be at least 1, got {min_len}")

    span = calendar.contiguous_span()
    if span is None:
        return Segmentation((), (), min_len, penalty, 0.0)

    days = [day for day in calendar.days if span[0] <= day.on <= span[1] and day.measured]
    values = [day.value for day in days if day.value is not None]
    n = len(values)

    variance = noise_scale(values)
    # No estimable noise scale means no penalty scale, and a zero penalty lets the segmenter
    # split anywhere at no cost. One segment is the honest answer: we cannot say a split is
    # significant when we cannot say what noise looks like. Real metric series always have
    # noise; a series that reaches here is constant, synthetic, or so heavily rounded that
    # over half its day-to-day differences are exactly zero.
    if n < 2 * min_len or variance <= 0.0:
        return _single(days, min_len, penalty, variance)

    beta = penalty * math.log(n) * variance

    # Prefix sums, so a segment's L2 cost is O(1). cost(i..j) = sum(y^2) - (sum y)^2 / len.
    prefix = [0.0] * (n + 1)
    prefix_sq = [0.0] * (n + 1)
    for i, value in enumerate(values):
        prefix[i + 1] = prefix[i] + value
        prefix_sq[i + 1] = prefix_sq[i] + value * value

    def cost(i: int, j: int) -> float:
        """L2 change-in-mean cost of values[i:j]."""
        length = j - i
        total = prefix[j] - prefix[i]
        return (prefix_sq[j] - prefix_sq[i]) - (total * total) / length

    # f[j] = optimal cost of segmenting values[0:j]; last[j] = start of its final segment.
    infinity = float("inf")
    f = [infinity] * (n + 1)
    last = [0] * (n + 1)
    f[0] = -beta  # so a single segment costs exactly cost(0, n), with no penalty for it
    for j in range(min_len, n + 1):
        for i in range(0, j - min_len + 1):
            if f[i] == infinity:
                continue
            # A candidate previous segment must itself be long enough. i == 0 is the start of
            # the series and carries no such requirement.
            if i and i < min_len:
                continue
            candidate = f[i] + cost(i, j) + beta
            if candidate < f[j]:
                f[j] = candidate
                last[j] = i
    if f[n] == infinity:
        return _single(days, min_len, penalty, variance)

    bounds: list[int] = [n]
    while bounds[-1] > 0:
        bounds.append(last[bounds[-1]])
    bounds.reverse()

    segments = tuple(
        Segment(
            start=days[start].on,
            end=days[stop - 1].on,
            start_index=start,
            end_index=stop - 1,
            mean=(prefix[stop] - prefix[start]) / (stop - start),
            days=stop - start,
        )
        # Pairwise over consecutive bounds, so the sequences differ in length by one by
        # construction -- `strict` would be a bug rather than a safeguard here.
        for start, stop in zip(bounds, bounds[1:])  # noqa: B905
    )
    return Segmentation(
        segments=segments,
        breaks=tuple(segment.start for segment in segments[1:]),
        min_len=min_len,
        penalty=penalty,
        noise_variance=variance,
    )


def _single(days: list, min_len: int, penalty: float, variance: float) -> Segmentation:
    """One segment covering everything -- the answer when there is nothing to split."""
    values = [day.value for day in days if day.value is not None]
    if not values:
        return Segmentation((), (), min_len, penalty, variance)
    return Segmentation(
        segments=(
            Segment(
                start=days[0].on,
                end=days[-1].on,
                start_index=0,
                end_index=len(values) - 1,
                mean=sum(values) / len(values),
                days=len(values),
            ),
        ),
        breaks=(),
        min_len=min_len,
        penalty=penalty,
        noise_variance=variance,
    )
