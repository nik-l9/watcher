"""Does the data cover the range that was asked for.

Extracted from `cortex.tools.posthog`, where it produced the disclosure that stopped the worst
answer this project has recorded: a `user signed up` series running 1-3 August against a range
ending on the 15th, from which the analyst computed a healthy 206/day, called it "the August run
rate", and concluded there was no real decline. Signup collection had stopped twelve days
earlier -- either a catastrophic drop or a broken pipeline, urgent under either reading.

**Why it lives here now.** It was reachable only from PostHog, and the same hole exists in every
other connector. 31 of 42 captured GA4 `get_sessions` calls asked for a date-dimensioned series,
so three quarters of GA4's daily series could stop early and say nothing about it -- the identical
defect on a different source, waiting.

The seam is dates, not rows. Each connector knows its own payload shape and extracts the buckets
it returned; this module knows nothing about payloads and cannot be broken by a shape change.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta
from typing import Any

__all__ = ["BUCKET_DAYS", "gap_disclosure"]

#: How many days a whole bucket of each interval spans.
#:
#: A month is 31 here rather than the real length of the month in question. The test below only
#: asks whether a *complete* bucket could have finished inside the range, and using the longest
#: possible month makes that test conservative -- it will not claim a February went missing on
#: the strength of 28 days having passed.
BUCKET_DAYS = {"day": 1, "week": 7, "month": 31}


def gap_disclosure(
    buckets: Iterable[date],
    *,
    window_end: date,
    interval: str,
    alternatives: str,
    as_of: date | None = None,
) -> dict[str, Any] | None:
    """Report that the series stops before the requested range does, or None if it does not.

    `alternatives` is the whole "it could be any of these" clause, supplied by the caller rather
    than assembled here. The candidate explanations are genuinely different per source -- an
    event can stop firing or be renamed, a GA4 metric cannot be renamed but its property can stop
    receiving hits -- and a generic sentence covering both would name causes that do not apply
    and omit the one that does.

    **Deliberately not interpreted.** A series can end early because an event was deprecated,
    because a deploy broke the SDK, or because nothing happened. The connector cannot tell which
    and must not guess: it reports the gap and its size, and establishing the cause is the
    investigation's job.

    Distinct from a partial trailing bucket, and the distinction is the point. `partial_buckets`
    answers "the period is young"; this answers "collection stopped". The two invite opposite
    conclusions from the same shape of gap -- the first is a reason to be calm, the second a
    reason to escalate -- so a connector disclosing only the first actively encourages the wrong
    one.
    """
    # A day that has not happened cannot be missing. Without this, an analyst asking PostHog
    # for 1-31 August on the 24th is told the series "has no data after 2026-08-23 -- 8 days are
    # missing entirely... the event may have stopped firing", when seven of those eight days are
    # in the future. Alarming, wrong, and it fires on the most ordinary request there is: a
    # question about the current month.
    #
    # `as_of` rather than `date.today()` because the latest date data *could* exist for is not
    # always today. In the evaluation suite it is the last day the scenario's world has any data
    # for, which is what stops a fixture that answers every range with one fixed series from
    # looking like a collection failure to any analyst who asks a wider range than was planted.
    if as_of is not None and as_of < window_end:
        window_end = as_of

    last: date | None = None
    for bucket in buckets:
        if last is None or bucket > last:
            last = bucket
    if last is None:
        # An empty series is already reported through the payload's own count, and the executor
        # marks it empty. A second voice saying the same thing adds noise, not information.
        return None

    # The question is not "how long since the last bucket" but "did a whole bucket finish inside
    # this range and report nothing". Those differ, and the difference was a false positive: a
    # weekly series whose last bucket starts on the 22nd covers through the 28th, so a range
    # ending on the 30th is missing only the in-progress week of the 29th -- nothing is wrong.
    # Measured from the bucket *start* that reads as 8 days missing against 7 days of slack, and
    # `campaign_traffic_drop` would have been told its series had stopped collecting.
    #
    # Strictly before the range end, so the final bucket may be incomplete without tripping: a
    # daily series ending yesterday is the normal state of a live source, not a gap.
    span = BUCKET_DAYS.get(interval, 1)
    next_bucket_end = last + timedelta(days=2 * span - 1)
    if not next_bucket_end < window_end:
        return None

    missing = (window_end - last).days

    return {
        "series_ends_early": {
            "last_bucket": last.isoformat(),
            "requested_end": window_end.isoformat(),
            "days_missing": missing,
        },
        "series_gap_note": (
            f"This series has no data after {last.isoformat()}, though the range runs to "
            f"{window_end.isoformat()} -- {missing} days are missing entirely. That is not the "
            f"same as a low value: {alternatives}. Do not compute a rate for the whole "
            "requested period from these rows, and do not treat the gap as a decline without "
            "establishing which of those it is."
        ),
    }
