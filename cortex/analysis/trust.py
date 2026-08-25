"""Is this series trustworthy enough to answer a business question from.

ADR 0005 decision 1, and the ordered gate in `docs/research/data-trust-gate.md` section 6.1.
Eleven checks are specified there; this implements the ones whose inputs already exist and
reports the rest as unknown rather than as passing.

**Why that distinction is the whole design.** The research's warning is blunt: *a gate whose
checks all return `unknown` passes everything*. Sequenced wrongly it looks like a safeguard and
behaves like a pass-through. So a check with no input says so, and `unknown` never reads as
`ok` -- a caller asking "may I answer the business question" gets three states and the third is
"I cannot tell you".

## What is implemented, and what it rests on

Four checks are answerable from a PostHog trend payload today, because four earlier fixes put
their inputs there:

- **Gate 3, emitter-partitioned correlated cessation.** `blast_radius` already computes it: a
  group of series ceasing together while a sibling emitter keeps recording. This is the one check
  the research allows to BLOCK, and the reasoning is that a group of unrelated events falling
  silent within a day of each other is not something changing user behaviour can produce.
- **Gate 3b, the weakened form.** One event stopping alone, or a project-wide stop with no
  surviving emitter to compare against, degrades rather than blocks -- with nothing still
  recording, a real outage cannot be ruled out.
- **Gate 4, single-series cessation.** `series_ends_early`.
- **Gate 2, trailing bucket completeness.** `partial_buckets`.

Five are not answerable and say so: a declared trust policy per series (gate 0), an incident log
(gate 1), source-versus-destination reconciliation (gate 5, which needs an emitter-side count
that does not exist), schema-change history (gate 6), and emitter deploys inside the range (gate
7). Gates 8 and 9 are distributional and belong to `cortex.analysis.movement`, which already
answers them.

**At most one check here BLOCKs**, which is the research's two-BLOCK ceiling with room left: ExP
sets its own blocking bar conservatively on purpose, and Airbnb reported a single-tier framework
being disabled by its own false positives.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

__all__ = [
    "COMPUTED_CHECKS",
    "Trust",
    "TrustCheck",
    "TrustVerdict",
    "UngradeableSource",
    "assess_trust",
    "trust_disclosure",
]


class Trust(enum.StrEnum):
    """What may be done with this series."""

    #: Answer the business question.
    OK = "ok"
    #: Answer it, with the limitation stated in the summary rather than as a caveat.
    DEGRADED = "degraded"
    #: The business question is the wrong question. The answer is the data verdict.
    BROKEN = "broken"


class Check(enum.StrEnum):
    """The gate rows, named as the research numbers them."""

    TRAILING_BUCKET = "gate2_trailing_bucket"
    CORRELATED_CESSATION = "gate3_correlated_cessation"
    SINGLE_CESSATION = "gate4_single_cessation"


#: Which checks each source's payloads can actually answer.
#:
#: Per-source rather than one allowlist, because "this source is gradeable" is too coarse to be
#: honest. GA4's `get_sessions` returns a daily series, and a day bucket is a day -- gate 2 has
#: nothing to disclose about it, ever. GA4 also has no sibling-event structure, so gate 3, the
#: only check permitted to BLOCK, cannot be computed there at all. Granting GA4 the whole gate
#: would report two checks as passing that were never run.
#:
#: A mapping rather than payload sniffing, and the point is *where the thinking happens*. Adding
#: a row here is the moment someone has to name which disclosures that connector computes.
COMPUTED_CHECKS: dict[str, frozenset[Check]] = {
    "posthog.event_trend": frozenset(
        {Check.CORRELATED_CESSATION, Check.SINGLE_CESSATION, Check.TRAILING_BUCKET}
    ),
    # Only gate 4. The series is daily, so gate 2 has nothing to say, and GA4 exposes no
    # sibling-event structure for gate 3 -- which means a GA4 series can DEGRADE and can never
    # BLOCK. That is the honest ceiling rather than a gap to close: blocking is reserved for a
    # correlated cessation, and GA4 has nothing to correlate against.
    "ga4.get_sessions": frozenset({Check.SINGLE_CESSATION}),
    # Two of three. Mixpanel returns a bucket/value series like PostHog's, so the same gap and
    # bucket-coverage disclosures apply; it computes no blast radius, so gate 3 cannot run and
    # a Mixpanel series can degrade but not block.
    "mixpanel.event_trend": frozenset({Check.SINGLE_CESSATION, Check.TRAILING_BUCKET}),
}


@dataclass(frozen=True, slots=True)
class TrustCheck:
    """One row's outcome, or the honest absence of one."""

    check: Check
    #: None when the check had no input. Never silently treated as a pass.
    trips_to: Trust | None
    detail: str

    @property
    def evaluated(self) -> bool:
        return self.trips_to is not None


@dataclass(frozen=True, slots=True)
class TrustVerdict:
    state: Trust
    checks: tuple[TrustCheck, ...] = ()
    #: Checks the research specifies that this code cannot answer, by name.
    not_evaluated: tuple[str, ...] = ()
    #: What to say when the state is not `ok`.
    note: str = ""

    @property
    def may_answer_the_business_question(self) -> bool:
        return self.state is not Trust.BROKEN


#: Gate rows with no input available here, named so a reader can see the gate's own holes.
#:
#: Listed rather than omitted. A gate that shows only the checks it can run looks complete, and
#: this one is not: five of eleven rows need metadata this project does not yet collect.
NOT_EVALUATED = (
    "gate0_trust_policy_declared (no per-series cadence or owning emitter is recorded)",
    "gate1_open_incident_overlap (no incident log exists to check the range against)",
    "gate5_source_vs_destination (needs an emitter-side count; no emitter reports one)",
    "gate6_schema_change_in_range (no schema-version history per series)",
    "gate7_emitter_deploy_in_range (deploys are not linked to the emitter they touch)",
)


class UngradeableSource(Exception):
    """This gate was handed a payload it has no checks for.

    Raised rather than returning `OK`, because `OK` is the answer that gets a wrong report
    delivered. Every implemented check reads a key the caller's connector has to have computed,
    and *absence of that key is not evidence of health* -- it is silence. Worse, absence is read
    as two different facts depending on the source: when `blast_radius` is missing from a PostHog
    trend the gate says "this series did not stop early", which is sound there because blast
    radius is computed whenever a cessation exists, and false for any connector that never
    computes it at all.

    Failing loudly means the trap is sprung at wiring time, in a test, by whoever is generalising
    the gate -- not in production silence six weeks later.
    """


def assess_trust(payload: dict[str, Any], *, source: str) -> TrustVerdict:
    """Grade a trend payload's trustworthiness.

    Reads only fields earlier work already puts in the payload, so this adds no request. Returns
    `OK` with the unevaluated rows named when nothing trips -- which is not the same as "this
    series is sound", and the note says so.

    `source` is required, and decides *which* checks run: each reads a key that only some
    connectors compute, and absence of the key is silence rather than health. Grading a payload
    against a check its source never computes reports a pass for something never run, which on a
    GA4 series meant `OK` on everything while the note claimed five of eleven rows went
    unevaluated when in fact seven had.
    """
    computed = COMPUTED_CHECKS.get(source)
    if not computed:
        raise UngradeableSource(
            f"{source!r} computes none of this gate's inputs (blast_radius, series_ends_early, "
            f"partial_buckets). Sources that compute at least one: "
            f"{sorted(COMPUTED_CHECKS)}. Grading it anyway would return 'ok' for every series, "
            "which is the pass-through this gate exists to prevent."
        )
    checks: list[TrustCheck] = []

    radius = payload.get("blast_radius")
    scope = radius.get("scope") if isinstance(radius, dict) else None
    if Check.CORRELATED_CESSATION not in computed:
        checks.append(
            TrustCheck(
                Check.CORRELATED_CESSATION,
                None,
                f"{source} exposes no sibling-series structure, so a correlated cessation "
                "cannot be ruled out or confirmed here",
            )
        )
    elif scope is None:
        checks.append(
            TrustCheck(
                Check.CORRELATED_CESSATION,
                None,
                "no sibling-event comparison was made; this series did not stop early",
            )
        )
    elif scope == "shared_with_other_events":
        stopped = radius.get("stopped_count", "several")
        live = radius.get("still_recording_count", "others")
        checks.append(
            TrustCheck(
                Check.CORRELATED_CESSATION,
                Trust.BROKEN,
                f"{stopped} series stopped together while {live} kept recording, which is a "
                "collection failure rather than a change in behaviour",
            )
        )
    elif scope == "this_event_only":
        checks.append(
            TrustCheck(
                Check.CORRELATED_CESSATION,
                Trust.DEGRADED,
                "this series stopped alone while every other kept recording, so a shared "
                "pipeline is ruled out -- a rename or broken instrumentation, or it genuinely "
                "stopped happening",
            )
        )
    else:  # project_wide
        checks.append(
            TrustCheck(
                Check.CORRELATED_CESSATION,
                Trust.DEGRADED,
                "no series in this project recorded anything after the gap, so there is no "
                "healthy sibling to compare against and a real outage cannot be ruled out",
            )
        )

    gap = payload.get("series_ends_early")
    if Check.SINGLE_CESSATION not in computed:
        checks.append(
            TrustCheck(
                Check.SINGLE_CESSATION,
                None,
                f"{source} does not report whether its series stops before the requested range",
            )
        )
    elif isinstance(gap, dict):
        checks.append(
            TrustCheck(
                Check.SINGLE_CESSATION,
                Trust.DEGRADED,
                f"the series stops at {gap.get('last_bucket')} with "
                f"{gap.get('days_missing')} day(s) missing from the requested range",
            )
        )

    if Check.TRAILING_BUCKET not in computed:
        checks.append(
            TrustCheck(
                Check.TRAILING_BUCKET,
                None,
                f"{source} returns no bucket-coverage disclosure; a short bucket at the edge "
                "of the range would not be visible here",
            )
        )
    elif payload.get("partial_buckets"):
        checks.append(
            TrustCheck(
                Check.TRAILING_BUCKET,
                Trust.DEGRADED,
                "a bucket at the edge of the range is not yet complete and must be excluded "
                "from any rate or comparison",
            )
        )

    tripped = [check.trips_to for check in checks if check.trips_to is not None]
    state = (
        Trust.BROKEN
        if Trust.BROKEN in tripped
        else Trust.DEGRADED
        if Trust.DEGRADED in tripped
        else Trust.OK
    )
    return TrustVerdict(
        state=state,
        checks=tuple(checks),
        not_evaluated=NOT_EVALUATED,
        note=_note(state, checks),
    )


#: Rows the research specifies, counted so the OK note cannot assert a figure that goes stale.
#: Gates 0-9 plus 3b, the weakened form of the correlated-cessation check.
_GATE_ROWS = 11


def _note(state: Trust, checks: list[TrustCheck]) -> str:
    """What a reader, or the drafting call, needs to do about it."""
    reasons = " ".join(f"{c.detail}." for c in checks if c.trips_to is not None)
    if state is Trust.BROKEN:
        return (
            "**This series cannot answer a business question.** "
            + reasons
            + " The answer to any question about why this metric moved is the data incident: "
            "what stopped, when, on which emitter, and what is still flowing. Do not attribute "
            "the movement to product, marketing or demand -- there is no movement here to "
            "attribute, only an absence of measurement."
        )
    if state is Trust.DEGRADED:
        return (
            "This series can answer a narrowed question. "
            + reasons
            + " State the limitation in the answer itself rather than as a caveat, and say what "
            "was excluded."
        )
    unevaluated = len(NOT_EVALUATED) + sum(1 for c in checks if c.trips_to is None)
    return (
        "Nothing in this payload trips a data-trust check. That is not the same as the series "
        f"being sound: {unevaluated} of the gate's {_GATE_ROWS} rows were not evaluated rather "
        "than passed, either because this project collects no such metadata or because this "
        "source does not compute the disclosure."
    )


def trust_disclosure(payload: dict[str, Any], *, source: str) -> dict[str, Any]:
    """The payload fields that report a non-OK verdict, or `{}` when there is nothing to report.

    Shared because three connectors were assembling the same dict by hand, and the third copy is
    where they start to diverge. What a reader is told about a degraded series should not depend
    on which connector produced it.

    An `OK` verdict returns nothing on purpose. The disclosure is worth reading precisely because
    it is rare; a `data_trust: ok` on every ordinary response is a field readers learn to skip,
    which would cost it the one call that matters. The unevaluated rows are still reachable by
    calling `assess_trust` directly, which is what a caller wanting the gate's own holes should do.
    """
    verdict = assess_trust(payload, source=source)
    if verdict.state is Trust.OK:
        return {}
    return {
        "data_trust": {
            "state": verdict.state.value,
            "may_answer_the_business_question": verdict.may_answer_the_business_question,
            "tripped": [
                {"check": c.check.value, "detail": c.detail} for c in verdict.checks if c.evaluated
            ],
            "not_evaluated": list(verdict.not_evaluated),
        },
        "data_trust_note": verdict.note,
    }
