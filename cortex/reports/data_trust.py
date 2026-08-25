"""Enforce the data-trust gate on a drafted report, in code rather than by instruction.

ADR 0005 decision 1 says a `broken` series means the business question is the wrong question and
the answer is the data verdict. `cortex.analysis.trust` computes that state and the drafting
guidance asks for it — and asking is not enforcing. This is the enforcement, and unlike the
sufficiency gate it needs no model judgement: the state is already in the payload, so a claim
attributing a movement to a cause while citing a series whose measurement stopped is a
*mechanical* contradiction.

That matters on this codebase's own rule: an LLM decides only what cannot be decided in code.
Whether the evidence *carries* a cause is a judgement. Whether a cited series was being measured
at all is a fact.

## What it does, and the one thing it deliberately does not

Causal claims citing a broken series are withheld, recorded as rejections, and replaced by the
data verdict as a risk the reader cannot miss. Descriptive claims survive untouched: "the series
stops on 2026-08-03" is true and is the answer, and removing it would leave the reader with
nothing.

It does **not** raise when the summary empties. The gate and the verifier both refuse a report
whose summary is gone, because an empty summary is no answer — but here the reason the summary
emptied *is itself the answer*, and throwing the investigation away would discard the one finding
worth delivering. So the verdict is promoted into the summary instead.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import Evidence
from cortex.db.threads import citable_investigation_ids
from cortex.reports.gate import Rejection, RejectionReason
from cortex.reports.schema import Claim, Finding, InvestigationReport, Risk
from cortex.reports.verifier import causal_claims
from cortex.tenancy.context import TenantContext

__all__ = ["AppliedDataTrust", "enforce_data_trust"]


@dataclass(frozen=True, slots=True)
class AppliedDataTrust:
    """A report with the gate enforced, and what enforcing it removed."""

    report: InvestigationReport
    rejections: list[Rejection] = field(default_factory=list)
    #: Evidence ids whose series was marked broken, for the disclosure.
    broken_sources: tuple[str, ...] = ()

    @property
    def withheld(self) -> int:
        return len(self.rejections)


async def enforce_data_trust(
    session: AsyncSession,
    tenant: TenantContext,
    *,
    investigation_id: uuid.UUID,
    report: InvestigationReport,
) -> AppliedDataTrust:
    """Withhold causal claims that cite a series which was not being measured.

    Cheap on the ordinary report: it returns before touching the database unless the report
    actually asserts a cause.
    """
    asserted = causal_claims(report)
    if not asserted:
        return AppliedDataTrust(report=report)

    citable = await citable_investigation_ids(session, tenant, investigation_id)
    rows = (
        (
            await session.execute(
                select(Evidence.id, Evidence.payload).where(
                    Evidence.tenant_id == tenant.tenant_id,
                    Evidence.investigation_id.in_(citable),
                )
            )
        )
        .tuples()
        .all()
    )
    broken = {
        evidence_id
        for evidence_id, payload in rows
        if isinstance(payload, dict) and _is_broken(payload)
    }
    if not broken:
        return AppliedDataTrust(report=report)

    # Only the claims that both assert a cause *and* rest on a broken series. A causal claim
    # citing healthy evidence is a different question and the sufficiency gate owns it.
    withheld = {location for location, claim in asserted if broken.intersection(claim.evidence_ids)}
    if not withheld:
        return AppliedDataTrust(report=report)

    rejections: list[Rejection] = []

    def _reject(location: str, claim: Claim) -> None:
        rejections.append(
            Rejection(
                location=location,
                reason=RejectionReason.NO_SURVIVING_EVIDENCE,
                detail=(
                    "data trust: this claim attributes a movement to a cause while citing a "
                    "series whose measurement stopped. There is no measured movement here to "
                    "attribute."
                ),
                text=claim.text,
            )
        )

    summary_kept: list[Claim] = []
    for index, claim in enumerate(report.executive_summary):
        location = f"executive_summary[{index}]"
        if location in withheld:
            _reject(location, claim)
            continue
        summary_kept.append(claim)

    findings_kept: list[Finding] = []
    for f_index, finding in enumerate(report.findings):
        claims_kept: list[Claim] = []
        for c_index, claim in enumerate(finding.claims):
            location = f"findings[{f_index}].claims[{c_index}]"
            if location in withheld:
                _reject(location, claim)
                continue
            claims_kept.append(claim)
        if claims_kept:
            findings_kept.append(finding.model_copy(update={"claims": claims_kept}))

    verdict = Claim(
        text=(
            "The measurement for at least one series in this answer stopped inside the "
            "requested range, so the movement being asked about was not observed. Any cause "
            "attributed to it has been withheld: what is established is that recording "
            "stopped, not that behaviour changed."
        ),
        # Cites the broken series itself, so the verdict resolves like any other claim rather
        # than being an uncited assertion the gate would have to make an exception for.
        evidence_ids=sorted(broken)[:3],
    )

    # Promoted into the summary rather than raising. An empty summary is normally no answer, and
    # both the gate and the verifier refuse one -- but here the reason it emptied *is* the
    # answer, and discarding the investigation would throw away the only finding worth having.
    summary = summary_kept if summary_kept else [verdict]
    risks = [*report.risks, Risk(description=_disclosure(rejections, broken))]
    if summary_kept:
        summary = [verdict, *summary_kept]

    return AppliedDataTrust(
        report=report.model_copy(
            update={
                "executive_summary": summary,
                "findings": findings_kept,
                "risks": risks,
            }
        ),
        rejections=rejections,
        broken_sources=tuple(str(e) for e in sorted(broken)),
    )


def _is_broken(payload: dict[str, Any]) -> bool:
    trust = payload.get("data_trust")
    return isinstance(trust, dict) and trust.get("state") == "broken"


def _disclosure(rejections: list[Rejection], broken: set[uuid.UUID]) -> str:
    """What was removed and why, in the reader's own terms."""
    return (
        f"{len(rejections)} causal claim(s) were withheld because they rest on a series whose "
        f"measurement stopped inside the requested range ({len(broken)} such observation(s)). "
        "This is a mechanical check, not a judgement about the argument: a movement that was "
        "not measured cannot have a cause attributed to it, however well the rest of the "
        "reasoning holds. The descriptive claims are untouched -- that recording stopped, and "
        "when, is the answer here."
    )
