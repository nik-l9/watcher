"""The investigation service — one place that runs an investigation to completion.

The loop, gate and verifier are separate components with separate tests. This is the
orchestration that connects them and owns the `Investigation` row's lifecycle, so both
the worker and the eval harness drive the same sequence rather than each assembling
their own.

Ordering is not arbitrary:

    loop → gate → verifier → sufficiency → persist

The gate runs before the verifier because verification costs a model call per claim,
and there is no point paying to read a claim whose citation does not exist. The gate
removes those for free.

The sufficiency gate runs last, on the report a reader would actually receive, for the
same reason the completeness judge does: a causal claim the verifier already removed must
not be paid for again, and a gate that judged the draft would be judging sentences that no
longer exist. It costs one model call, and only on a report that asserts a cause — see
`cortex.reports.sufficiency`.

Status transitions are written as the work happens, not at the end. An investigation
that dies mid-run should leave a row saying where it was, not a row still claiming to
be queued.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.agents.employee import Employee, gtm_data_analyst
from cortex.agents.investigator import (
    InvestigationCancelled,
    InvestigationFailed,
    Investigator,
)
from cortex.agents.llm import LLM, Usage
from cortex.agents.progress import Phase, ProgressEvent, ProgressSink, emit
from cortex.db.models import Investigation as InvestigationRow
from cortex.db.models import InvestigationStatus, Report
from cortex.memory.recall import HybridRecall
from cortex.reports.completeness import CompletenessJudge
from cortex.reports.data_trust import enforce_data_trust
from cortex.reports.gate import GroundingGate, ReportRejected
from cortex.reports.identifiability import apply_corroboration, identifiability_notes
from cortex.reports.sufficiency import (
    AppliedSufficiency,
    SufficiencyDecision,
    SufficiencyGate,
)
from cortex.reports.verifier import AdversarialVerifier
from cortex.tenancy.context import TenantContext
from cortex.tenancy.limits import RateLimiter
from cortex.tools.executor import ToolExecutor
from cortex.tools.registry import registry_for_tenant


@dataclass(slots=True)
class CompletedInvestigation:
    investigation_id: uuid.UUID
    report_id: uuid.UUID
    hallucinations: int
    gate_rejections: int
    verifier_rejections: int
    tokens: int
    duration_ms: int
    #: Causal claims the sufficiency gate withheld. Zero on a report that asserts no
    #: cause, and zero when the gate found the evidence sufficient — the two are
    #: distinguished by `SufficiencyDecision`, not by this count.
    sufficiency_vetoes: int = 0


class InvestigationService:
    def __init__(
        self,
        *,
        llm: LLM,
        employee: Employee | None = None,
        verify: bool = True,
        sufficiency: bool = True,
        sessionmaker: Any = None,
        recall: HybridRecall | None = None,
        progress: ProgressSink | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        self._llm = llm
        self._employee = employee or gtm_data_analyst()
        # Used only for the cancellation check, which needs its own transaction. Optional so
        # a caller that does not want cancellation is unaffected.
        self._sessionmaker = sessionmaker
        self._verify = verify
        # Its own flag rather than a second meaning for `verify`. The two are different
        # mechanisms: one asks whether each claim is supported by its citation, the other
        # asks whether the evidence supports a cause at all, and turning off the first while
        # iterating on the loop is not a reason to turn off the second.
        self._sufficiency = sufficiency
        # Passed through rather than constructed, so the worker decides whether memory is
        # available and this class stays testable without a graph or a vector store.
        self._recall = recall
        # Advisory. A dropped event costs a UI a frame; nothing about the investigation
        # depends on one arriving.
        self._progress = progress
        # Passed to the executor, which is the one chokepoint every tool call goes through —
        # the same reason scoping and evidence writing live there.
        self._limiter = limiter

    def _cancellation_check(self, investigation_id: uuid.UUID):  # type: ignore[no-untyped-def]
        """A callable the loop asks once per step: has this been cancelled?

        Reads through a fresh session deliberately. The loop's session began before the
        gateway's cancel could commit, so inside that transaction the row still reads
        `investigating` — the check would never see the cancellation it exists to see. This is
        the kind of thing that looks like it works in a test where both writes share a
        session.
        """

        async def _check() -> bool:
            async with self._sessionmaker() as session:
                status = await session.scalar(
                    select(InvestigationRow.status).where(InvestigationRow.id == investigation_id)
                )
            return status is InvestigationStatus.CANCELLED

        return _check

    async def run(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        *,
        investigation_id: uuid.UUID,
    ) -> CompletedInvestigation:
        """Run one investigation and persist its report.

        Raises `InvestigationFailed` or `ReportRejected` after recording the failure
        on the row, so a caller retrying a queued message can tell a transient outage
        from a report that was correctly refused.
        """
        row = await self._load(session, tenant, investigation_id)

        # Scoped to what this tenant has connected. Offering a capability whose credential
        # is absent costs a step, produces a CredentialMissing, and puts an irrelevant
        # caveat in the report -- see `registry_for_tenant`.
        registry = await registry_for_tenant(session, tenant)
        investigator = Investigator(
            llm=self._llm,
            registry=registry,
            executor=ToolExecutor(registry, limiter=self._limiter),
            employee=self._employee,
            recall=self._recall,
            progress=self._progress,
            # Reads the row the gateway marks. A separate session per check, because the
            # loop's own session holds an open transaction and would not see a commit made
            # by the gateway after it began.
            cancelled=(
                self._cancellation_check(investigation_id)
                if self._sessionmaker is not None
                else None
            ),
        )

        await self._mark(session, row, InvestigationStatus.INVESTIGATING, started=True)
        try:
            investigation = await investigator.investigate(
                session,
                tenant,
                investigation_id=investigation_id,
                question=row.question,
                # A follow-up is given its parent's conclusion and an inventory of its
                # evidence, and may cite those observations instead of gathering them again.
                parent_id=row.parent_id,
            )
        except InvestigationFailed as exc:
            await self._fail(session, row, str(exc))
            raise
        except InvestigationCancelled as exc:
            # Not a failure, and deliberately not routed through `_fail`: the row is already
            # CANCELLED (the gateway set it, which is how the loop knew), and overwriting it
            # with FAILED would turn "somebody stopped this" into "this broke". The evidence
            # gathered before the stop stays — it was really observed, and "what did it find
            # before I cancelled" is a fair question.
            row.steps_used = exc.steps
            row.error = f"cancelled by request after {exc.steps} step(s)"
            await session.flush()
            raise

        # Recorded before grounding: the hypotheses the loop tested are worth keeping
        # even if the report is subsequently refused, because they show what was ruled
        # out and on what basis.
        row.hypotheses = [h.model_dump(mode="json") for h in investigation.report.hypotheses]
        row.steps_used = len(investigation.steps)
        record_usage(row, investigation.usage, model=self._llm.model)
        await session.flush()

        await self._mark(session, row, InvestigationStatus.SYNTHESIZING)
        emit(self._progress, ProgressEvent(Phase.GATING))
        try:
            gated = await GroundingGate().apply(
                session,
                tenant,
                investigation_id=investigation_id,
                report=investigation.report,
            )
        except ReportRejected as exc:
            await self._fail(session, row, f"grounding rejected the report: {exc}")
            raise

        verification = None
        if self._verify:
            await self._mark(session, row, InvestigationStatus.VERIFYING)
            emit(self._progress, ProgressEvent(Phase.VERIFYING))
            try:
                verification = await AdversarialVerifier(self._llm).verify(
                    session,
                    tenant,
                    investigation_id=investigation_id,
                    report=gated.report,
                )
            except ReportRejected as exc:
                await self._fail(session, row, f"verification rejected the report: {exc}")
                raise

        final = verification.report if verification else gated.report

        # Decision 6. A separate call, asking only whether the evidence gathered supports a
        # definitive answer about cause, with veto power — and it can refuse the whole report
        # when every claim in the summary asserted one. Routed through `_fail` like the other
        # two refusals, so the row carries what was missing rather than a bare "rejected".
        sufficiency = SufficiencyDecision(needed=False)
        applied = AppliedSufficiency(report=final)
        if self._sufficiency:
            sufficiency = await SufficiencyGate(self._llm).assess(
                session,
                tenant,
                investigation_id=investigation_id,
                question=row.question,
                report=final,
            )
            try:
                applied = sufficiency.apply(final)
            except ReportRejected as exc:
                await self._fail(session, row, f"the evidence did not support an answer: {exc}")
                raise
            final = applied.report

        # Judged on the delivered report, after both grounding mechanisms have had their say.
        # Adds a caveat when the report does not cover the question, and edits nothing: this
        # is an LLM's opinion, and grounding decisions stay mechanical in this codebase.
        assessment = await CompletenessJudge(self._llm).assess(row.question, final)
        if assessment.risks:
            final = final.model_copy(update={"risks": [*final.risks, *assessment.risks]})

        # ADR 0005 decision 5. Mechanical, unlike the judge above: for every dated cause the
        # report stands behind, say what would be needed to establish it and is absent. A
        # disclosure rather than a veto -- the sufficiency gate holds the veto, and refusing every
        # causal answer until control series exist would be defensible and useless.
        # ADR 0005 decision 1, enforced rather than instructed, and first of the three
        # because it decides whether there is a movement to explain at all. A causal claim
        # citing a series whose measurement stopped is a mechanical contradiction, so this
        # needs no model call -- unlike the sufficiency gate that follows it.
        trust = await enforce_data_trust(
            session, tenant, investigation_id=investigation_id, report=final
        )
        final = trust.report

        # ADR 0005 decision 3, before the disclosure below: a cause resting on one
        # independent line of evidence is recorded as inconclusive rather than supported, because
        # insufficient evidence does not license a pick.
        final = await apply_corroboration(
            session, tenant, investigation_id=investigation_id, report=final
        )

        for risk in await identifiability_notes(
            session, tenant, investigation_id=investigation_id, report=final
        ):
            final = final.model_copy(update={"risks": [*final.risks, risk]})

        report = Report(
            tenant_id=tenant.tenant_id,
            investigation_id=investigation_id,
            body=final.model_dump(mode="json"),
            confidence=confidence_score(final.confidence.value),
            # Both rejection channels are stored separately: the eval suite needs to
            # distinguish a structural drop from a judged-unsupported claim.
            #
            # Sufficiency vetoes join the verifier's list rather than getting a column of
            # their own -- a new column is a migration, and both are "a model read this and
            # it did not hold up". They are told apart by the `sufficiency:` prefix on
            # `detail`, which is the same convention the verifier's own entries already use.
            gate_rejections=[r.as_dict() for r in gated.rejections],
            verifier_rejections=[
                *([r.as_dict() for r in verification.rejections] if verification else []),
                *[r.as_dict() for r in applied.rejections],
            ],
        )
        session.add(report)
        await session.flush()

        row.status = InvestigationStatus.COMPLETED
        row.completed_at = datetime.now(UTC)
        # Recorded again at the end, now including the verifier's own spend: a bill that
        # omitted the second grounding mechanism would understate every investigation by
        # whatever the verifier cost, which is not small.
        total = investigation.usage + sufficiency.usage
        if verification is not None:
            total = total + verification.usage
        record_usage(row, total, model=self._llm.model)
        await session.flush()

        return CompletedInvestigation(
            investigation_id=investigation_id,
            report_id=report.id,
            hallucinations=gated.hallucination_count
            + (verification.unsupported_count if verification else 0),
            gate_rejections=len(gated.rejections),
            verifier_rejections=(len(verification.rejections) if verification else 0)
            + len(applied.rejections),
            tokens=row.tokens_used,
            duration_ms=investigation.duration_ms,
            sufficiency_vetoes=applied.withheld,
        )

    # ------------------------------------------------------------------ internals

    @staticmethod
    async def _load(
        session: AsyncSession, tenant: TenantContext, investigation_id: uuid.UUID
    ) -> InvestigationRow:
        row = await session.get(InvestigationRow, investigation_id)
        # Tenant-scoped even though the worker already resolved the tenant from the
        # message: a redelivered or hand-crafted message must not be able to run
        # another tenant's investigation. Same reasoning as F-01.
        if row is None or row.tenant_id != tenant.tenant_id:
            raise InvestigationFailed(
                f"investigation {investigation_id} is not available to this tenant"
            )
        return row

    @staticmethod
    async def _mark(
        session: AsyncSession,
        row: InvestigationRow,
        status: InvestigationStatus,
        *,
        started: bool = False,
    ) -> None:
        row.status = status
        if started and row.started_at is None:
            row.started_at = datetime.now(UTC)
        await session.flush()

    @staticmethod
    async def _fail(session: AsyncSession, row: InvestigationRow, error: str) -> None:
        row.status = InvestigationStatus.FAILED
        row.error = error[:2000]
        row.completed_at = datetime.now(UTC)
        await session.flush()


def confidence_score(level: str) -> float:
    """A numeric confidence for sorting and dashboards.

    Public because `cortex.ask` writes a `Report` row too and must score it the same way.
    Two implementations of the same mapping would drift, and the one that drifts is
    whichever is edited second.

    The report keeps the coarse level as the thing a reader sees; this is only for
    ordering. Deliberately coarse-grained — a model asked for a percentage produces
    two decimal places and means nothing by them.
    """
    return {
        "high": 0.9,
        "medium": 0.6,
        "low": 0.3,
        "insufficient_evidence": 0.1,
    }.get(level, 0.5)


def record_usage(row: InvestigationRow, usage: Usage, *, model: str | None) -> None:
    """Write the token breakdown onto the row so spend can be computed from it.

    `tokens_used` is kept alongside for the existing API, but it is the derived figure now
    rather than the recorded one — input and output differ by 5x on every model we run, and a
    cache read costs a tenth of a fresh input token, so the total cannot be priced.

    The model is stored too. Pricing is per model, and without it a spend figure is a guess
    that silently reprices itself the next time the default model changes.
    """
    row.tokens_used = usage.total
    row.input_tokens = usage.input_tokens
    row.output_tokens = usage.output_tokens
    row.cache_read_tokens = usage.cache_read_input_tokens
    row.cache_write_tokens = usage.cache_creation_input_tokens
    row.model = model
