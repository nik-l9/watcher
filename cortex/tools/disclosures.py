"""What a connector discloses about a series, assembled in one place.

**Why this is not inside each connector.** Every disclosure in this project exists because an
answer went wrong without it, and none of them were reaching the evaluation suite. The eval
replaces each capability's handler with one that returns a canned fixture payload, so the real
connector method never runs -- which meant `series_ends_early`, `partial_buckets`, `data_trust`
and the movement description were exercised by unit tests and by nothing else. Two consecutive
10/10 runs said nothing about any of them, and a disclosure could have been wrong in production
with the suite still passing.

The obvious fix -- hand-write the disclosures into the fixtures -- measures the wrong thing. It
tests whether the analyst reacts to a disclosure, not whether the connector computes one. A
fixture author who forgets is indistinguishable from a connector that does not disclose.

So the computation moves here and both callers use it: the connectors in production, the eval
handler over its fixtures. One assembler rather than two, because two would drift, and drift
between what production discloses and what the suite measures is the failure this whole module
is trying to prevent.

**What cannot move.** PostHog's `blast_radius` needs a second query to the sibling events, so it
is not computable from a payload and stays in the connector. A fixture supplies it directly,
which is what that query would have returned.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from cortex.analysis.movement import describe_movement
from cortex.analysis.trust import trust_disclosure

__all__ = ["TRUST_SOURCES", "grade_series", "series_disclosures"]

#: Capability to the `assess_trust` source name, for those the gate has checks for.
TRUST_SOURCES = {
    "posthog__event_trend": "posthog.event_trend",
    "ga4__get_sessions": "ga4.get_sessions",
    "mixpanel__event_trend": "mixpanel.event_trend",
}


def series_disclosures(
    qualified: str,
    payload: dict[str, Any],
    params: dict[str, Any],
    *,
    as_of: date | None = None,
) -> dict:
    """Everything computable about this payload from the payload and the request alone.

    Returns `{}` for a capability that returns no series, and for a series request that carries
    no disclosure -- an ordinary complete series adds no fields, so a reader who sees one knows
    it means something.

    `as_of` is the last date data could exist for, and defaults to today because that is what it
    is in production: a day that has not happened cannot be missing. The evaluation suite passes
    its scenario's own last data day instead, which is what stops a fixture that answers every
    range with one fixed series from looking like a collection failure to an analyst who asks a
    wider range than was planted.
    """
    horizon = as_of if as_of is not None else date.today()
    if qualified == "ga4__get_sessions":
        from cortex.tools.ga4 import _gap as ga4_gap

        end = params.get("end_date")
        if not isinstance(end, str):
            return {}
        return ga4_gap(payload.get("rows") or [], end, as_of=horizon) or {}

    if qualified == "mixpanel__event_trend":
        from cortex.tools.mixpanel import _gap as mixpanel_gap
        from cortex.tools.mixpanel import _partial_bucket_note

        start, end = params.get("from_date"), params.get("to_date")
        unit = params.get("unit", "day")
        if not isinstance(start, str) or not isinstance(end, str):
            return {}
        series = payload.get("series") or []
        return {
            **(_partial_bucket_note(start, end, unit) or {}),
            **(mixpanel_gap(series, end, unit, as_of=horizon) or {}),
        }

    if qualified == "posthog__event_trend":
        from cortex.tools.posthog import _bucket_coverage, _series_freshness, _series_gap

        start, end = params.get("start_date"), params.get("end_date")
        interval = params.get("interval", "day")
        if not isinstance(start, str) or not isinstance(end, str):
            return {}
        rows = payload.get("series") or []
        return {
            **(_bucket_coverage(rows, start, end, interval) or {}),
            **(_series_gap(rows, end, interval, as_of=horizon) or {}),
            # The second clock. `_series_gap` asks the bucket grid whether a complete bucket
            # went missing; this asks how stale the data is against the requested period, which
            # is the question a coarse interval hides. Both, because they say different things
            # and the granularity-dependent one alone let "the period is young" stand as the
            # only reading of a pipeline that had been dead for 35 days.
            **(_series_freshness(rows, end, as_of=horizon) or {}),
            **(
                describe_movement(
                    rows,
                    start_date=start,
                    end_date=end,
                    interval=interval,
                    breakdown_property=params.get("breakdown_property"),
                )
                or {}
            ),
        }

    return {}


def grade_series(qualified: str, payload: dict[str, Any]) -> dict:
    """The data-trust verdict, run last because it reads the disclosures rather than the rows.

    Returns `{}` for a capability the gate has no checks for, rather than asking the gate and
    catching its refusal: a source with no checks is an ordinary fact about the surface, not an
    error, and `assess_trust` raising is reserved for a caller that believed otherwise.
    """
    source = TRUST_SOURCES.get(qualified)
    if source is None:
        return {}
    return trust_disclosure(payload, source=source)
