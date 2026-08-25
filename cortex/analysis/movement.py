"""The analysis layer as a connector sees it: rows in, a described movement out.

**Why this is computed in the connector rather than left to the investigation.** Seven fixes in
this codebase have now had the same shape -- a connector disclosed a data problem, supplied no
resolution, and the analyst resolved it with whatever was nearest. `series_ends_early` said
"twelve days are missing, this could be three things, do not guess", and the analyst answered it
with the nearest falling line on the screen. A fact the analyst has to make a second call to
obtain is a fact it will not obtain.

So this runs on the series that was just fetched, and its output travels in the same payload as
the numbers it describes. The analyst cannot hold the series without also holding when it moved
and whether that movement is more than noise.

**What it does not do.** It never names a cause, and it says so in its own note. Establishing
*when* something moved is a different act from establishing *why*, and the second needs an
intervention, a control series and a confounder ledger that no connector can supply -- see
`cortex.analysis.identifiability`.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from cortex.analysis.changepoints import DEFAULT_MIN_LEN, detect_level_shifts
from cortex.analysis.conformal import significance_of_each
from cortex.analysis.series import as_calendar

__all__ = ["describe_movement"]

#: Daily buckets only.
#:
#: `min_len` is a count of days calibrated against a weekly seasonal cycle, and the recovery
#: numbers behind it (0/200 at min_len=1 against 182/200 at 14) are numbers about days. A weekly
#: or monthly series has neither the resolution nor the seasonality this was tuned for, and
#: running it anyway would produce a confident answer from an uncalibrated method.
SUPPORTED_INTERVAL = "day"

#: Below two minimum segments there is nothing a segmenter can split.
MIN_DAYS = 2 * DEFAULT_MIN_LEN


def describe_movement(
    rows: list[dict[str, Any]],
    *,
    start_date: str,
    end_date: str,
    interval: str,
    breakdown_property: str | None = None,
) -> dict[str, Any] | None:
    """Describe when a daily series changed level, and whether each change beats noise.

    Returns None -- adding no field at all -- whenever the method does not apply. An absent
    field is honest; a field saying "not applicable" is noise on every ordinary call, and a
    disclosure that appears everywhere stops being read on the call that needed it.
    """
    if interval != SUPPORTED_INTERVAL or breakdown_property:
        return None

    observed: dict[date, float] = {}
    for row in rows:
        bucket = row.get("bucket")
        if not isinstance(bucket, str):
            return None
        try:
            on = date.fromisoformat(bucket[:10])
        except ValueError:
            return None
        try:
            observed[on] = float(row.get("value") or 0.0)
        except (TypeError, ValueError):
            return None
    if len(observed) < MIN_DAYS:
        return None

    try:
        calendar = as_calendar(
            observed, start=date.fromisoformat(start_date), end=date.fromisoformat(end_date)
        )
    except ValueError:
        return None

    segmentation = detect_level_shifts(calendar)
    if segmentation.found_nothing:
        # Said rather than omitted. "We looked for a level shift and found none" is a finding,
        # and it is the one that stops a reader reading noise as a trend.
        return {
            "movement": {
                "level_shifts": [],
                "min_segment_days": segmentation.min_len,
            },
            "movement_note": (
                "No sustained level shift was found in this series. Day-to-day variation is "
                f"within noise, and nothing lasting at least {segmentation.min_len} days "
                "separates itself from it. A movement shorter than that is an incident rather "
                "than a level shift and this check cannot see it."
            ),
        }

    significance = {result.at: result for result in significance_of_each(calendar, segmentation)}

    shifts: list[dict[str, Any]] = []
    for at in segmentation.breaks:
        levels = segmentation.shift_at(at)
        if levels is None:
            continue
        before, after = levels
        result = significance.get(at)
        entry: dict[str, Any] = {
            "at": at.isoformat(),
            "before_per_day": round(before, 1),
            "after_per_day": round(after, 1),
            "change_per_day": round(after - before, 1),
            "change_pct": round((after - before) / before * 100, 1) if before else None,
        }
        if result is None:
            entry["established"] = None
            entry["not_established_because"] = "no window was available to test this break"
        elif not result.resolvable:
            # An inability, not a negative. A non-significant result from a window that could
            # never have rejected reads as "we checked and it is not real", which is the
            # opposite of what happened.
            entry |= {
                "established": None,
                "not_established_because": (
                    f"the period after the break ({result.post_days} days) is not shorter than "
                    f"the period before it ({result.pre_days} days), and the test has no power "
                    "in that configuration"
                ),
                "window_days": result.total_days,
            }
        else:
            entry |= {
                "p_value": round(result.p_value, 4),
                "p_floor": round(result.p_floor, 4),
                "p_at_floor": result.at_floor,
                "established": result.significant(),
                "window_days": result.total_days,
            }
        shifts.append(entry)

    return {
        "movement": {
            "level_shifts": shifts,
            "min_segment_days": segmentation.min_len,
            "noise_scale_per_day": round(segmentation.noise_variance**0.5, 1),
        },
        "movement_note": _note(shifts, segmentation.min_len),
    }


def _note(shifts: list[dict[str, Any]], min_len: int) -> str:
    """What a reader is entitled to know, including what this cannot tell them."""
    established = [s for s in shifts if s.get("established")]
    untestable = [s for s in shifts if s.get("established") is None]
    parts = [
        f"{len(shifts)} sustained level shift(s) found; "
        f"{len(established)} distinguishable from noise by a permutation test that needs no "
        "control series."
    ]
    if untestable:
        parts.append(
            f"{len(untestable)} could not be tested at all -- `established` is null with a "
            "reason, which is not the same as being tested and found unreal. Treat those as "
            "movements of unknown significance, not as noise."
        )
    if any(s.get("p_at_floor") for s in established):
        parts.append(
            "A p-value marked `p_at_floor` is the smallest its window could produce, so it "
            "means 'as significant as this window can resolve' rather than an exact figure, "
            "and it cannot be compared against a p from a longer window."
        )
    parts.append(
        "**This establishes when the series moved, and nothing about why.** Do not attribute "
        "any of these shifts to a cause on the strength of this field. A causal claim needs a "
        "named change, a comparison series the change did not touch, and a statement of what "
        "else happened in the window -- none of which is in this payload. A shift whose date "
        "precedes a candidate cause's own date rules that cause out."
    )
    parts.append(
        f"Nothing shorter than {min_len} days can appear here; a briefer dip is an incident "
        "and needs a different check."
    )
    return " ".join(parts)
