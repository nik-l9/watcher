"""What it would take to establish the cause this report names, and why it cannot.

ADR 0005 decision 5, reaching a report at last. `cortex.analysis.identifiability` has held the
eight predicates and ninety tests since it was written and has been called from nowhere, which
made it the most expensive kind of unfinished work: a thing that looks done.

**This attaches a disclosure; it does not delete a claim.** The division of labour matters and is
easy to get wrong. The sufficiency gate (decision 6) has the veto -- it asks whether the evidence
supports a definitive answer and withholds the claim when it does not. This asks a different
question, *whether a causal claim is available in principle from what we can observe*, and the
honest answer today is almost always no: we have zero admissible control series, so Abadie's
placebo p-value floor `1/(J+1)` is 1 and no synthetic-control claim exists at any effect size.

Wiring that as a second veto would refuse every causal answer Cortex ever gives, permanently,
until per-region or per-plan series exist. That is *defensible* -- the research's composition rule
is worst-domain, any single refusal refuses the claim -- and it is not useful. A reader is better
served by "here is the candidate, the movement is established, and here is precisely the one thing
missing that would let anyone call it the cause" than by silence. So the gate speaks and the
sufficiency gate cuts.

## What can be checked from a report and its audit trail

Three of the eight predicates need only the report's own structure and the calls it made, and they
are the three that refuse:

- **G1, a named intervention.** `Hypothesis.cause_at` is one. A hypothesis that dates its proposed
  cause has named an intervention; one that does not has not, and the system must never invent one
  by picking its own largest changepoint and testing that.
- **G3, admissible controls.** None exist, and the number that would change it is quotable:
  nineteen comparable, independently-unaffected series buys `p <= 0.05`.
- **G7, a confounder ledger.** The distinction this hinges on is computable from the audit trail
  rather than guessed -- see `ledger_from_calls`.

The rest -- completeness, separability, the p floor, the detection floor, sensitivity -- need the
series itself, its noise scale and its window. Those live in a connector payload rather than in the
report, so they are answered where the movement is described (`cortex.analysis.movement`) and are
deliberately not re-derived here from prose.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.analysis.identifiability import (
    Control,
    Gate,
    Intervention,
    assess_identifiability,
)
from cortex.db.models import Evidence, ToolCall
from cortex.reports.schema import Hypothesis, InvestigationReport, Risk, Verdict
from cortex.tenancy.context import TenantContext

__all__ = [
    "CHANGE_RECORD_CAPABILITIES",
    "apply_corroboration",
    "identifiability_notes",
    "identifiability_risks",
    "ledger_from_calls",
]

#: Capabilities whose whole purpose is to answer "what else changed here".
#:
#: A call to one of these is somebody looking. No call to any of them is nobody looking, and
#: the difference is the entire content of G7: an empty ledger is a claim, not a default.
CHANGE_RECORD_CAPABILITIES = frozenset(
    {
        ("github", "recent_prs"),
        ("github", "deployment_history"),
        ("github", "commits"),
        ("posthog", "annotations"),
        ("posthog", "feature_flags"),
        ("posthog", "experiments"),
    }
)


@dataclass(frozen=True, slots=True)
class Ledger:
    """What the investigation established about other changes in the window."""

    #: None when nobody looked. A tuple -- possibly empty -- when somebody did.
    entries: tuple[str, ...] | None
    #: Which capabilities were actually consulted, for the disclosure.
    consulted: tuple[str, ...]

    @property
    def attested(self) -> bool:
        return self.entries is not None


def ledger_from_calls(calls: list[ToolCall]) -> Ledger:
    """Whether anybody looked for other changes, from the audit trail.

    **The distinction G7 exists for, made mechanically instead of assumed.** "Nobody looked" and
    "somebody looked and found nothing" are different claims about the world, and a system that
    conflates them reports the second when it means the first -- which is how an empty result
    becomes evidence of absence.

    A call to a change-record capability is somebody looking, whether or not it returned rows: a
    query for deploys in the onset window that comes back empty is a genuine finding about that
    window. What it is not is a licence to say nothing else happened, which is why the entries are
    left empty rather than being filled with an assertion nobody made -- `assess_identifiability`
    reads an empty tuple as "attested, nothing found" and that is exactly the claim the call
    supports.

    A *failed* call is not looking. A 403 tells you about the credential, not the window, and
    counting it would turn a permissions problem into a statement about the world.
    """
    consulted = sorted(
        {
            f"{call.tool_name}.{call.capability}"
            for call in calls
            if call.succeeded and (call.tool_name, call.capability) in CHANGE_RECORD_CAPABILITIES
        }
    )
    return Ledger(entries=() if consulted else None, consulted=tuple(consulted))


def _asserted_causes(report: InvestigationReport) -> list[Hypothesis]:
    """Hypotheses that name a dated cause and stand behind it.

    Only `SUPPORTED` ones. A contradicted or inconclusive hypothesis is not an assertion, so
    demanding identifiability for it would refuse the report for correctly declining to commit --
    the same reading `_asserted_text` and the accuracy dimension already use.
    """
    return [
        hypothesis
        for hypothesis in report.hypotheses
        if hypothesis.verdict is Verdict.SUPPORTED and hypothesis.cause_at is not None
    ]


def identifiability_risks(
    report: InvestigationReport,
    calls: list[ToolCall],
    *,
    controls: tuple[Control, ...] = (),
) -> list[Risk]:
    """One risk per asserted dated cause that cannot be established, saying what is missing.

    Empty when the report asserts no dated cause, which is the common case and costs nothing.
    """
    asserted = _asserted_causes(report)
    if not asserted:
        return []

    ledger = ledger_from_calls(calls)
    risks: list[Risk] = []
    for hypothesis in asserted:
        assert hypothesis.cause_at is not None  # narrowed by `_asserted_causes`
        verdict = assess_identifiability(
            _no_calendar(),
            intervention=Intervention(
                change_id=hypothesis.statement[:120],
                at=hypothesis.cause_at,
                scope="as stated in the hypothesis",
            ),
            # The series-dependent predicates are answered where the movement is described, not
            # re-derived from prose here. Passed as unknown so they do not silently pass either.
            shift=None,
            sigma=0.0,
            days_before=0,
            days_after=0,
            controls=controls,
            ledger=ledger.entries,
        )
        blocking = [
            check
            for check in verdict.refusals
            if check.gate in (Gate.INTERVENTION, Gate.CONTROLS, Gate.CONFOUNDER_LEDGER)
        ]
        if not blocking:
            continue
        risks.append(Risk(description=_describe(hypothesis, blocking, ledger)))
    return risks


def _describe(hypothesis: Hypothesis, blocking: list, ledger: Ledger) -> str:
    """The sentence a reader gets, with the thing that would lift each refusal.

    Written as what is missing rather than as a verdict on the analyst. "No comparison series
    exists" is a fact about the estate; "the analyst failed to find one" would be a fact about the
    report, and it would be wrong.
    """
    lines = [
        f"'{hypothesis.statement[:140]}' is consistent with the evidence and cannot be "
        "established as the cause from it. What is missing:"
    ]
    for check in blocking:
        lines.append(f" - {check.detail}" + (f" ({check.lifts_it})" if check.lifts_it else ""))
    if ledger.attested and ledger.consulted:
        lines.append(" Change records consulted: " + ", ".join(ledger.consulted) + ".")
    return " ".join(lines)


#: A fixed day, so an empty calendar is deterministic rather than dependent on when the report
#: was drafted. Any date works: the calendar is empty and only its emptiness is read.
_NO_DAY = date(2000, 1, 1)


def _no_calendar():  # type: ignore[no-untyped-def]
    """An empty calendar, so the completeness predicate reports "not measured" rather than pass.

    Constructed here rather than threaded from the connector because this module deliberately
    does not re-derive series facts from prose. G0's verdict is discarded by the caller for the
    same reason.
    """
    from cortex.analysis.series import as_calendar

    return as_calendar({}, start=_NO_DAY, end=_NO_DAY)


async def identifiability_notes(
    session: AsyncSession,
    tenant: TenantContext,
    *,
    investigation_id: uuid.UUID,
    report: InvestigationReport,
) -> list[Risk]:
    """Load the audit trail and describe what each asserted dated cause is missing.

    Here rather than in `cortex.reports.identifiability` so that module stays a pure function of
    a report and its calls -- easy to test, and impossible to accidentally make issue a query per
    hypothesis.
    """
    if not any(
        hypothesis.verdict is Verdict.SUPPORTED and hypothesis.cause_at is not None
        for hypothesis in report.hypotheses
    ):
        # No asserted dated cause, so nothing to assess and no reason to touch the database.
        return []
    calls = (
        (
            await session.execute(
                select(ToolCall).where(
                    ToolCall.tenant_id == tenant.tenant_id,
                    ToolCall.investigation_id == investigation_id,
                )
            )
        )
        .scalars()
        .all()
    )
    return identifiability_risks(report, list(calls))


async def apply_corroboration(
    session: AsyncSession,
    tenant: TenantContext,
    *,
    investigation_id: uuid.UUID,
    report: InvestigationReport,
) -> InvestigationReport:
    """Apply decision 3's stay-broad rule, loading the evidence-to-source mapping.

    Runs before `identifiability_notes` at every call site, and the order matters: this can turn
    a supported cause into an inconclusive one, and the disclosure only speaks about causes the
    report still stands behind. Reversed, a claim downgraded here would still collect a paragraph
    explaining what would establish it -- which is true and beside the point once the report has
    stopped asserting it.
    """
    from cortex.reports.corroboration import corroborate

    if not any(
        hypothesis.verdict is Verdict.SUPPORTED and hypothesis.cause_at is not None
        for hypothesis in report.hypotheses
    ):
        return report
    rows = (
        (
            await session.execute(
                select(Evidence.id, Evidence.tool_name).where(
                    Evidence.tenant_id == tenant.tenant_id,
                    Evidence.investigation_id == investigation_id,
                )
            )
        )
        .tuples()
        .all()
    )
    return corroborate(report, {eid: tool for eid, tool in rows}).report
