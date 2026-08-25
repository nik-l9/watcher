"""Grounding gate — the structural half of "never hallucinate".

The gate resolves every citation in a drafted report against the evidence store and
removes anything that does not hold up. It runs in code, before render, and it does
not ask a model for an opinion. That distinction is the point: a prompt saying
"only cite real evidence" is a request, whereas failing to find an `evidence_id` in
a tenant-scoped query is a fact.

Four ways a citation fails:

  1. **Unknown** — the id does not exist. The model invented a UUID.
  2. **Foreign** — the id belongs to a different investigation, or a different
     tenant. This is what makes the gate a security control and not merely a quality
     filter; see docs/security-findings.md F-01.
  3. **Tampered** — the row exists but its payload no longer hashes to the recorded
     digest, so the observation changed after it was cited.
  4. **Empty** — a claim whose citations were all removed by the above.

A claim is dropped, never rewritten. Rewriting would mean generating fresh prose at
exactly the moment the model has been demonstrated unreliable.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import Evidence
from cortex.db.threads import citable_investigation_ids
from cortex.ingest.health import health_notes
from cortex.reports.charts import charts_from_evidence, validate_chart
from cortex.reports.schema import (
    Claim,
    Confidence,
    DataQualityNote,
    Finding,
    Hypothesis,
    InvestigationReport,
    Risk,
    Source,
    Verdict,
)
from cortex.tenancy.context import TenantContext
from cortex.tools.executor import canonical_hash


class RejectionReason(enum.StrEnum):
    UNKNOWN_EVIDENCE = "unknown_evidence"
    FOREIGN_EVIDENCE = "foreign_evidence"
    TAMPERED_EVIDENCE = "tampered_evidence"
    NO_SURVIVING_EVIDENCE = "no_surviving_evidence"


@dataclass(frozen=True, slots=True)
class Rejection:
    """One thing the gate removed, and why.

    Persisted on the report row. The eval suite scores hallucinations from these, so
    they are a product signal rather than a debug log — a rising unknown_evidence
    count means the drafting prompt is degrading.
    """

    location: str
    reason: RejectionReason
    detail: str
    text: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "location": self.location,
            "reason": self.reason.value,
            "detail": self.detail,
            "text": self.text,
        }


@dataclass(slots=True)
class GateResult:
    report: InvestigationReport
    rejections: list[Rejection] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.rejections

    @property
    def hallucination_count(self) -> int:
        """Citations that could not be resolved at all.

        Distinct from the total rejection count: an empty claim is a consequence of
        an earlier bad citation, not an independent fabrication, so counting both
        would double-count one mistake.
        """
        return sum(
            1
            for r in self.rejections
            if r.reason in (RejectionReason.UNKNOWN_EVIDENCE, RejectionReason.FOREIGN_EVIDENCE)
        )


class ReportRejected(Exception):
    """The report retained nothing citable.

    Raised rather than returning an empty report: a summary with no surviving claims
    is not a degraded answer, it is no answer, and presenting it as one would be the
    exact failure the gate exists to prevent.
    """


class GroundingGate:
    def __init__(self, *, verify_hashes: bool = True) -> None:
        # Hash verification costs a full payload read per cited row. It is on by
        # default and can be disabled for large synthetic eval runs where the
        # evidence store is known-good.
        self._verify_hashes = verify_hashes

    async def apply(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        *,
        investigation_id: uuid.UUID,
        report: InvestigationReport,
    ) -> GateResult:
        """Strip every unresolvable citation, then rebuild the Sources section."""
        cited = report.cited_evidence_ids()
        valid, evidence_by_id = await self._resolve(session, tenant, investigation_id, cited)

        rejections: list[Rejection] = []
        for missing in sorted(cited - set(valid), key=str):
            # One exception for absent and foreign, so a report cannot be used to
            # probe which evidence ids exist in other tenants.
            rejections.append(
                Rejection(
                    location="citation",
                    reason=RejectionReason.UNKNOWN_EVIDENCE,
                    detail=f"evidence {missing} is not part of this investigation",
                )
            )

        if self._verify_hashes:
            for evidence_id, row in list(evidence_by_id.items()):
                if canonical_hash(row.payload) != row.payload_hash:
                    valid.discard(evidence_id)
                    evidence_by_id.pop(evidence_id, None)
                    rejections.append(
                        Rejection(
                            location="citation",
                            reason=RejectionReason.TAMPERED_EVIDENCE,
                            detail=(
                                f"evidence {evidence_id} no longer matches its recorded "
                                "hash and cannot be cited"
                            ),
                        )
                    )

        summary = self._filter_claims(
            report.executive_summary, valid, "executive_summary", rejections
        )
        findings = self._filter_findings(report.findings, valid, rejections)
        hypotheses = self._filter_hypotheses(report.hypotheses, valid, rejections)
        charts = [c for c in report.charts if self._chart_survives(c, valid, rejections)]
        if not charts:
            # Derived from the cited evidence, on the same principle as the Sources section:
            # a model asked to write a series inline can write a number nobody observed, and
            # a chart is the part of a report a reader trusts without checking the citation.
            # Built here rather than in the loop because charts are withheld from the
            # drafting schema entirely — the compiled grammar exceeds the provider's limit
            # with them — so the model cannot produce one even when asked.
            charts = charts_from_evidence(
                [row for row_id, row in evidence_by_id.items() if row_id in valid]
            )
        recommendations = [
            r
            for r in report.recommendations
            if self._survives(
                set(r.evidence_ids), valid, f"recommendation[{r.action[:60]}]", r.action, rejections
            )
        ]
        risks = self._filter_risks(report.risks, valid, rejections)
        notes = [
            n.model_copy(update={"evidence_ids": [e for e in n.evidence_ids if e in valid]})
            for n in report.data_quality
        ]
        # Connector staleness is appended here rather than left to the model, because it is
        # the one disclosure the analyst cannot know it needs to make. A sync that has been
        # failing since Tuesday produces a graph that looks complete and is three days
        # behind, and an investigation reading it would honestly report "no deploys that
        # week" from data that was never fetched. The plan requires this in the confidence
        # section; requiring it in code is what makes it true on every report rather than
        # on the ones where the drafter thought of it.
        notes.extend(await health_notes(session, tenant))
        # And the same treatment for emptiness. A claim can cite a perfectly real evidence
        # row that contains nothing, and the citation gate above will pass it: the row
        # exists, the hash matches, the id belongs to this investigation. What it cannot
        # check is whether the row said anything. This is the third and last layer of the
        # same defence -- the capability declares where its results live, the loop labels an
        # empty observation as empty, and here the report discloses when its conclusions
        # rest on one.
        notes.extend(_empty_evidence_notes(evidence_by_id, valid))

        if not summary:
            raise ReportRejected(
                "no claim in the executive summary survived grounding; "
                f"{len(rejections)} citation(s) were rejected"
            )

        gated = report.model_copy(
            update={
                "executive_summary": summary,
                "findings": findings,
                "hypotheses": hypotheses,
                "charts": charts,
                "recommendations": recommendations,
                "risks": risks,
                "data_quality": notes,
                # Derived, never authored: a model-written source could carry an
                # invented permalink into the one section a reader treats as
                # independently verifiable.
                "sources": self._build_sources(evidence_by_id),
                "confidence": self._adjust_confidence(report.confidence, rejections),
            }
        )
        return GateResult(report=gated, rejections=rejections)

    # ------------------------------------------------------------------ internals

    async def _resolve(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        investigation_id: uuid.UUID,
        cited: set[uuid.UUID],
    ) -> tuple[set[uuid.UUID], dict[uuid.UUID, Evidence]]:
        """Load the cited evidence, scoped to this tenant and this thread.

        Both predicates matter. tenant_id alone would let a report cite evidence from
        the same tenant's unrelated investigation; investigation_id alone would trust
        a foreign-key value that F-01 showed can be attacker-supplied.

        **The scope is the thread rather than the single investigation**, so a follow-up may
        cite an observation its parent already paid for instead of re-fetching it. That set is
        this investigation plus its *ancestors*, computed from stored rows by
        `citable_investigation_ids` -- never from anything in the request or the report, so a
        report cannot widen the evidence it is allowed to cite. For an investigation with no
        parent the set has one element and this behaves exactly as it did before, which is what
        makes the widening safe: a sibling, a cousin, and an unrelated investigation of the same
        tenant all stay uncitable.
        """
        if not cited:
            return set(), {}

        citable = await citable_investigation_ids(session, tenant, investigation_id)
        rows = (
            (
                await session.execute(
                    select(Evidence).where(
                        Evidence.tenant_id == tenant.tenant_id,
                        Evidence.investigation_id.in_(citable),
                        Evidence.id.in_(cited),
                    )
                )
            )
            .scalars()
            .all()
        )
        by_id = {row.id: row for row in rows}
        return set(by_id), by_id

    def _survives(
        self,
        cited: set[uuid.UUID],
        valid: set[uuid.UUID],
        location: str,
        text: str,
        rejections: list[Rejection],
    ) -> bool:
        """Whether an element retains at least one resolvable citation."""
        if cited & valid:
            return True
        rejections.append(
            Rejection(
                location=location,
                reason=RejectionReason.NO_SURVIVING_EVIDENCE,
                detail="every cited evidence id was rejected",
                text=text[:500],
            )
        )
        return False

    def _filter_claims(
        self,
        claims: list[Claim],
        valid: set[uuid.UUID],
        location: str,
        rejections: list[Rejection],
    ) -> list[Claim]:
        kept: list[Claim] = []
        for index, claim in enumerate(claims):
            surviving = [e for e in claim.evidence_ids if e in valid]
            if not surviving:
                rejections.append(
                    Rejection(
                        location=f"{location}[{index}]",
                        reason=RejectionReason.NO_SURVIVING_EVIDENCE,
                        detail="every cited evidence id was rejected",
                        text=claim.text[:500],
                    )
                )
                continue
            # Narrowed to the citations that resolved, so a partially-grounded claim
            # keeps only the support it actually has.
            kept.append(claim.model_copy(update={"evidence_ids": surviving}))
        return kept

    def _filter_findings(
        self, findings: list[Finding], valid: set[uuid.UUID], rejections: list[Rejection]
    ) -> list[Finding]:
        kept: list[Finding] = []
        for index, finding in enumerate(findings):
            claims = self._filter_claims(
                finding.claims, valid, f"findings[{index}].claims", rejections
            )
            if not claims:
                rejections.append(
                    Rejection(
                        location=f"findings[{index}]",
                        reason=RejectionReason.NO_SURVIVING_EVIDENCE,
                        detail="no claim in this finding survived grounding",
                        text=finding.title[:500],
                    )
                )
                continue
            kept.append(finding.model_copy(update={"claims": claims}))
        return kept

    def _filter_hypotheses(
        self,
        hypotheses: list[Hypothesis],
        valid: set[uuid.UUID],
        rejections: list[Rejection],
    ) -> list[Hypothesis]:
        """Keep hypotheses, downgrading any whose evidence did not survive.

        Downgraded to inconclusive rather than dropped: that a hypothesis was
        considered is itself informative, and silently deleting it would hide the
        analyst's reasoning from the reader.
        """
        kept: list[Hypothesis] = []
        for index, hypothesis in enumerate(hypotheses):
            supporting = [e for e in hypothesis.supporting_evidence_ids if e in valid]
            contradicting = [e for e in hypothesis.contradicting_evidence_ids if e in valid]

            verdict = hypothesis.verdict
            if verdict is Verdict.SUPPORTED and not supporting:
                verdict = Verdict.INCONCLUSIVE
                rejections.append(
                    Rejection(
                        location=f"hypotheses[{index}]",
                        reason=RejectionReason.NO_SURVIVING_EVIDENCE,
                        detail="downgraded to inconclusive: supporting evidence was rejected",
                        text=hypothesis.statement[:500],
                    )
                )
            elif verdict is Verdict.CONTRADICTED and not contradicting:
                verdict = Verdict.INCONCLUSIVE
                rejections.append(
                    Rejection(
                        location=f"hypotheses[{index}]",
                        reason=RejectionReason.NO_SURVIVING_EVIDENCE,
                        detail="downgraded to inconclusive: contradicting evidence was rejected",
                        text=hypothesis.statement[:500],
                    )
                )

            kept.append(
                hypothesis.model_copy(
                    update={
                        "verdict": verdict,
                        "supporting_evidence_ids": supporting,
                        "contradicting_evidence_ids": contradicting,
                    }
                )
            )
        return kept

    def _filter_risks(
        self, risks: list[Risk], valid: set[uuid.UUID], rejections: list[Rejection]
    ) -> list[Risk]:
        """Risks are kept even when uncited.

        Deliberately asymmetric with claims: a risk that names absent data cites
        nothing by nature, and dropping it would remove a caveat while leaving the
        conclusion it qualifies. Removing a caveat is never the safe direction.
        """
        del rejections
        return [
            risk.model_copy(update={"evidence_ids": [e for e in risk.evidence_ids if e in valid]})
            for risk in risks
        ]

    def _chart_survives(
        self, chart: object, valid: set[uuid.UUID], rejections: list[Rejection]
    ) -> bool:
        cited = chart.all_evidence_ids  # type: ignore[attr-defined]
        location = f"charts[{chart.title[:60]}]"  # type: ignore[attr-defined]
        if not self._survives(
            cited,
            valid,
            location,
            chart.title,  # type: ignore[attr-defined]
            rejections,
        ):
            return False

        # A chart whose citations all resolve can still mislead, and in a way prose cannot:
        # a deploy marker at a position the data does not cover *invents* the coincidence
        # the reader is being shown. Dropped rather than repaired, because a chart with its
        # marker silently moved would assert something different from what the analyst
        # drafted.
        problems = validate_chart(chart)  # type: ignore[arg-type]
        for problem in problems:
            rejections.append(
                Rejection(
                    location=location,
                    text=chart.title,  # type: ignore[attr-defined]
                    reason=problem.problem[:300],
                )
            )
        return not problems

    @staticmethod
    def _build_sources(evidence_by_id: dict[uuid.UUID, Evidence]) -> list[Source]:
        return [
            Source(
                evidence_id=row.id,
                tool_name=row.tool_name,
                capability=row.capability,
                source_ref=row.source_ref,
                observed_at=row.observed_at.isoformat() if row.observed_at else None,
                from_cache=row.from_cache,
            )
            for row in sorted(evidence_by_id.values(), key=lambda r: (r.tool_name, r.capability))
        ]

    @staticmethod
    def _adjust_confidence(stated: Confidence, rejections: list[Rejection]) -> Confidence:
        """Cap confidence when the gate had to remove something.

        A model that cited fabricated evidence and declared high confidence was wrong
        about more than the citation, and letting the stated confidence stand would
        present a weakened report with its original certainty.
        """
        if not rejections:
            return stated
        if stated is Confidence.HIGH:
            return Confidence.MEDIUM
        if stated is Confidence.MEDIUM:
            return Confidence.LOW
        return stated


#: Payload keys that indicate the observation carried results, whatever they were called.
#:
#: Read from the persisted payload rather than from the capability, because the gate resolves
#: evidence rows and does not have the registry to hand. Deliberately generous: a false
#: "this was empty" note is a wasted line, while a missed one is the failure this exists to
#: catch.
_RESULT_KEYS = (
    "rows",
    "messages",
    "commits",
    "deployments",
    "issues",
    "pull_requests",
    "releases",
    "matches",
    "deals",
    "contacts",
    "companies",
    "activities",
    "stages",
    "events",
    "series",
    "steps",
    "annotations",
    "flags",
    "experiments",
    "comparison",
    "tables",
    "files",
)


def _empty_evidence_notes(
    evidence_by_id: dict[uuid.UUID, Evidence], cited: set[uuid.UUID]
) -> list[DataQualityNote]:
    """Disclose when the report's own citations point at observations that found nothing.

    The citation gate cannot catch this. A row that contains no results is a real row: it
    exists, its hash matches, it belongs to this investigation. Every grounding check passes,
    and the claim resting on it can still be wrong in the one way that matters — asserting
    that something did not happen when the truth is that we did not see it.

    Counted rather than listed per claim, because the useful statement is about the
    investigation's reach: "three of the eight things we looked at came back empty" tells a
    reader how much of this answer is built on silence.
    """
    empty = [
        row
        for row_id, row in evidence_by_id.items()
        if row_id in cited and _looks_empty(row.payload)
    ]
    if not empty:
        return []

    where = ", ".join(sorted({f"{row.tool_name}.{row.capability}" for row in empty}))
    return [
        DataQualityNote(
            note=(
                f"{len(empty)} of the {len(cited)} cited observation(s) returned no results "
                f"({where}). Conclusions resting on them describe what was not found, which "
                f"is not the same as what does not exist."
            ),
            evidence_ids=[row.id for row in empty][:10],
        )
    ]


def _looks_empty(payload: dict[str, object]) -> bool:
    """Whether a persisted observation carried any results."""
    if not isinstance(payload, dict):
        return False
    present = [payload[key] for key in _RESULT_KEYS if key in payload]
    if not present:
        # No recognisable result container: a scalar observation such as a single PR's state.
        # Not treated as empty, because there is nothing here to have been empty.
        return False
    return all(not value for value in present)
