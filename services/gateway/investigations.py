"""Investigation endpoints.

The gateway accepts a question, persists it, and enqueues the work. It deliberately
does not run the investigation: a multi-minute LLM loop inside the request path would
couple user-facing availability to the slowest tool call, and would scale the HTTP tier
for a bottleneck that lives elsewhere.

So `POST` returns immediately with an id, and the client polls. Every response is
tenant-scoped by the same rule as everything else — an id from another tenant is
indistinguishable from one that does not exist.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.contracts.messages import QUEUE_INVESTIGATION, RunInvestigation
from cortex.db.models import Investigation, InvestigationStatus, Report, ToolCall
from cortex.db.spend import spend_for_tenant
from cortex.db.threads import ParentNotFound, ThreadTooDeep, check_parent
from cortex.db.titles import title_for
from cortex.runtime.celery_app import make_celery
from cortex.tenancy.context import TenantContext
from services.gateway.deps import get_session, require_tenant

router = APIRouter(tags=["investigations"])

#: A producer only. The gateway never registers or executes a task — it publishes to
#: the queue the investigation worker consumes.
_producer = make_celery("gateway-producer")


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=8, max_length=2000)
    #: Which specialist. V1 has one, but the field exists so a client written today
    #: does not need changing when the PMM analyst lands.
    employee: str = Field(default="gtm_data_analyst", pattern=r"^[a-z][a-z0-9_]*$")
    #: Ask this as a follow-up to an earlier investigation.
    #:
    #: The analyst is given that investigation's conclusion and an inventory of its evidence,
    #: and may cite those observations directly rather than gathering them again. Validated
    #: against the caller's tenant before the row is written: an id in a request body proves
    #: nothing about who owns it (F-01).
    parent_id: uuid.UUID | None = None


class InvestigationCreated(BaseModel):
    id: uuid.UUID
    status: str
    question: str


class InvestigationState(BaseModel):
    id: uuid.UUID
    status: str
    question: str
    #: A short label for lists and tabs. Derived from the question, not generated.
    title: str | None = None
    employee: str
    steps_used: int
    tokens_used: int
    error: str | None = None
    #: Present once the report exists, so a poller needs one call rather than two.
    report_id: uuid.UUID | None = None
    #: The investigation this one follows up on, if any. Exposed so a client can render a
    #: thread without a second query, and so a reader can tell a follow-up's narrower question
    #: from an oddly specific standalone one.
    parent_id: uuid.UUID | None = None
    #: The hypotheses the loop tested, including the ones it ruled out. Exposed while
    #: an investigation is still running so a caller can see it reasoning.
    hypotheses: list = Field(default_factory=list)


@router.post(
    "/investigations",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=InvestigationCreated,
)
async def ask(
    body: AskRequest,
    tenant: TenantContext = Depends(require_tenant),
    session: AsyncSession = Depends(get_session),
) -> InvestigationCreated:
    """Accept a question and queue the investigation.

    202, not 200: the work has been accepted, not performed. Returning 200 with an
    empty report would invite clients to treat a queued investigation as a finished one.
    """
    if body.parent_id is not None:
        try:
            await check_parent(session, tenant, body.parent_id)
        except ParentNotFound as exc:
            # 404, matching every other unknown-or-foreign id on this router, so a parent id
            # cannot be probed for existence.
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
        except ThreadTooDeep as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    row = Investigation(
        tenant_id=tenant.tenant_id,
        user_id=tenant.user_id,
        employee=body.employee,
        question=body.question,
        title=title_for(body.question),
        parent_id=body.parent_id,
    )
    session.add(row)
    # Flushed before enqueueing so the row exists by the time a worker picks the
    # message up. Enqueueing first would race: a fast worker could look for a row this
    # transaction has not written yet.
    await session.flush()

    _producer.send_task(
        "cortex.investigation.run",
        args=[
            RunInvestigation(
                tenant_id=tenant.tenant_id,
                investigation_id=row.id,
                requested_by_user_id=tenant.user_id,
            ).model_dump(mode="json")
        ],
        queue=QUEUE_INVESTIGATION,
    )

    return InvestigationCreated(id=row.id, status=row.status.value, question=row.question)


class InvestigationSummary(BaseModel):
    """One row in a list. Deliberately not `InvestigationState`.

    A list endpoint that returns the full state of every investigation invites a client to
    render a list by fetching everything and then throwing most of it away — and it grows
    every time state gains a field. `hypotheses` in particular is unbounded.
    """

    id: uuid.UUID
    #: What a list actually renders. Derived from the question, never generated.
    title: str | None = None
    status: str
    created_at: datetime
    report_id: uuid.UUID | None = None
    parent_id: uuid.UUID | None = None


class InvestigationPage(BaseModel):
    """A page of investigations, newest first.

    `total` is carried because a list without one cannot render "showing 20 of 340", and a
    client that has to guess whether more exist will guess wrong at the boundary.
    """

    investigations: list[InvestigationSummary]
    total: int
    limit: int
    offset: int


#: Rows per page. Bounded rather than caller-chosen without limit: an unbounded list
#: endpoint is a way to turn one request into a table scan of a tenant's whole history.
_MAX_PAGE = 100


@router.get("/investigations", response_model=InvestigationPage)
async def list_investigations(
    limit: int = 20,
    offset: int = 0,
    tenant: TenantContext = Depends(require_tenant),
    session: AsyncSession = Depends(get_session),
) -> InvestigationPage:
    """This tenant's investigations, newest first.

    The first endpoint that makes the product navigable: until now an investigation could
    only be read by an id the caller already had, so a UI had no way to show what had been
    asked before. It pairs with `title`, which is what a row renders — a list of raw
    question text is unusable.
    """
    if not 1 <= limit <= _MAX_PAGE:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, f"limit must be between 1 and {_MAX_PAGE}"
        )
    if offset < 0:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "offset must not be negative")

    total = await session.scalar(
        select(func.count())
        .select_from(Investigation)
        .where(Investigation.tenant_id == tenant.tenant_id)
    )

    rows = (
        (
            await session.execute(
                # Left-joined rather than queried per row: a page of twenty would otherwise
                # be twenty-one round trips, and the report id is what makes a completed row
                # clickable.
                select(Investigation, Report.id)
                .outerjoin(
                    Report,
                    (Report.investigation_id == Investigation.id)
                    & (Report.tenant_id == tenant.tenant_id),
                )
                .where(Investigation.tenant_id == tenant.tenant_id)
                # Tie-broken by id. Two investigations created in the same millisecond would
                # otherwise be free to swap places between pages, which silently drops one
                # row and shows another twice.
                .order_by(Investigation.created_at.desc(), Investigation.id.desc())
                .limit(limit)
                .offset(offset)
            )
        )
        .tuples()
        .all()
    )

    return InvestigationPage(
        investigations=[
            InvestigationSummary(
                id=row.id,
                title=row.title,
                status=row.status.value,
                created_at=row.created_at,
                report_id=report_id,
                parent_id=row.parent_id,
            )
            for row, report_id in rows
        ],
        total=int(total or 0),
        limit=limit,
        offset=offset,
    )


@router.get("/investigations/{investigation_id}", response_model=InvestigationState)
async def get_investigation(
    investigation_id: uuid.UUID,
    tenant: TenantContext = Depends(require_tenant),
    session: AsyncSession = Depends(get_session),
) -> InvestigationState:
    row = (
        await session.execute(
            select(Investigation).where(
                Investigation.id == investigation_id,
                Investigation.tenant_id == tenant.tenant_id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        # One response for absent and foreign, so ids cannot be enumerated.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown investigation")

    return await _state(session, tenant, row)


async def _state(
    session: AsyncSession, tenant: TenantContext, row: Investigation
) -> InvestigationState:
    """One investigation's state, as both the read and the cancel endpoint return it.

    Extracted rather than duplicated: two constructions of the same response drift, and the
    one that drifts is whichever is edited second.
    """
    report_id = None
    if row.status is InvestigationStatus.COMPLETED:
        report_id = (
            await session.execute(
                select(Report.id).where(
                    Report.investigation_id == row.id,
                    Report.tenant_id == tenant.tenant_id,
                )
            )
        ).scalar_one_or_none()

    return InvestigationState(
        id=row.id,
        status=row.status.value,
        question=row.question,
        title=row.title,
        employee=row.employee,
        steps_used=row.steps_used,
        tokens_used=row.tokens_used,
        error=row.error,
        report_id=report_id,
        parent_id=row.parent_id,
        hypotheses=row.hypotheses or [],
    )


class TraceStep(BaseModel):
    """One tool call, as a reader sees it.

    Params are included and responses are not. That split is already the rule for the audit
    row this is derived from: a response body belongs in `Evidence`, and copying it here
    would double the surface on which customer data can leak into a UI.
    """

    at: datetime
    tool: str
    capability: str
    succeeded: bool
    #: Present only on failure, and already sanitised at the point it was stored — vendor
    #: URLs and credentials are stripped before an error reaches this column.
    error: str | None = None
    duration_ms: int | None = None
    #: Present when the call produced an observation, so a reader can follow a trace step to
    #: the evidence a claim cites.
    evidence_id: uuid.UUID | None = None
    params: dict = Field(default_factory=dict)


class InvestigationTrace(BaseModel):
    status: str
    steps: list[TraceStep]
    #: Calls that returned nothing. Surfaced as a count rather than left to be noticed:
    #: "nothing happened" and "I could not look" are different facts, and a trace that
    #: hides the difference is how the second gets read as the first.
    empty_or_failed: int


@router.get("/investigations/{investigation_id}/trace", response_model=InvestigationTrace)
async def get_investigation_trace(
    investigation_id: uuid.UUID,
    tenant: TenantContext = Depends(require_tenant),
    session: AsyncSession = Depends(get_session),
) -> InvestigationTrace:
    """What the analyst actually did, in order.

    **Derived from the `tool_calls` audit rows rather than from a stored event stream**, and
    that is the design decision worth stating. Indexing the OpenHands frontend showed the
    split their UI settled on: a REST-served backlog for history plus a socket carrying only
    new events, because a transport that guarantees neither ordering nor completeness cannot
    be asked to reconstruct the past.

    Ours goes one step further. Their backlog is a persisted event log — a second record of
    what happened, which can disagree with the first. Every tool call here already writes an
    immutable audit row, so the backlog *is* the audit trail: it cannot drift from what
    happened, it survives a worker restart, and it needed no new table. The progress queue
    stays what its own docstring always said it was — advisory, describing the current
    moment, with the durable record elsewhere.

    Available while an investigation is still running, which is the point: this is the
    endpoint that answers "is it stuck or is it working".
    """
    row = (
        await session.execute(
            select(Investigation.status).where(
                Investigation.id == investigation_id,
                Investigation.tenant_id == tenant.tenant_id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown investigation")

    calls = (
        (
            await session.execute(
                select(ToolCall)
                .where(
                    ToolCall.investigation_id == investigation_id,
                    # Scoped by tenant as well as by investigation. The investigation is
                    # already known to be this tenant's, so this predicate is redundant
                    # today — and it is the one that still holds if `investigation_id` is
                    # ever nulled by the ON DELETE SET NULL on that column.
                    ToolCall.tenant_id == tenant.tenant_id,
                )
                # Tie-broken by id: several calls in one step share a timestamp to the
                # millisecond, and a trace whose order changes between reads reads as though
                # the analyst did something different the second time.
                .order_by(ToolCall.created_at.asc(), ToolCall.id.asc())
            )
        )
        .scalars()
        .all()
    )

    return InvestigationTrace(
        status=row.value,
        steps=[
            TraceStep(
                at=call.created_at,
                tool=call.tool_name,
                capability=call.capability,
                succeeded=call.succeeded,
                error=call.error,
                duration_ms=call.duration_ms,
                evidence_id=call.evidence_id,
                params=call.params or {},
            )
            for call in calls
        ],
        # A successful call with no evidence row returned nothing. Counted with outright
        # failures because they mean the same thing to a reader: this line of enquiry
        # produced no observation.
        empty_or_failed=sum(1 for call in calls if not call.succeeded or call.evidence_id is None),
    )


#: Statuses a cancellation cannot change, because the work is already over.
_TERMINAL = frozenset(
    {
        InvestigationStatus.COMPLETED,
        InvestigationStatus.FAILED,
        InvestigationStatus.CANCELLED,
    }
)


@router.post("/investigations/{investigation_id}/cancel", response_model=InvestigationState)
async def cancel_investigation(
    investigation_id: uuid.UUID,
    tenant: TenantContext = Depends(require_tenant),
    session: AsyncSession = Depends(get_session),
) -> InvestigationState:
    """Ask a running investigation to stop.

    Cooperative, and deliberately so. The loop runs in a worker process and this request
    arrives here, so there is nothing to interrupt directly — this marks the row, and the loop
    reads the row once per step and stops. That means cancellation lands within one step
    rather than instantly, which is the honest trade: the alternative is killing a worker
    mid-write and leaving the evidence store in a state nobody planned for.

    **Evidence gathered before the stop is kept.** It was really observed, and "what did it
    find before I cancelled" is a reasonable question. What is *not* produced is a report:
    drafting is the most expensive call in the investigation, and running it after the user
    asked to stop would bill them for the thing they cancelled.

    Idempotent. Cancelling an already-cancelled investigation returns its state rather than
    an error, because a user pressing a button twice has not done anything wrong.
    """
    row = (
        await session.execute(
            select(Investigation).where(
                Investigation.id == investigation_id,
                Investigation.tenant_id == tenant.tenant_id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        # One response for absent and foreign, so ids cannot be enumerated.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown investigation")

    if row.status not in _TERMINAL:
        row.status = InvestigationStatus.CANCELLED
        # Recorded on the row rather than left blank, so a reader of the audit trail can tell
        # a cancellation from a crash without cross-referencing anything.
        row.error = "cancelled by request"
        row.completed_at = datetime.now(UTC)
        await session.flush()

    return await _state(session, tenant, row)


class SpendView(BaseModel):
    """What this tenant has spent over a trailing window."""

    model_config = ConfigDict(extra="forbid")

    tenant: str
    window_days: int
    investigations: int
    completed: int
    failed: int
    cancelled: int
    usd: float
    usd_per_investigation: float
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cache_hit_rate: float
    #: True when some investigation carried no model name and was priced at the highest rate
    #: we know. Surfaced rather than hidden: an estimate presented as a measurement is the
    #: failure this codebase keeps guarding against.
    estimated: bool


@router.get("/spend", response_model=SpendView)
async def get_spend(
    days: int = 30,
    tenant: TenantContext = Depends(require_tenant),
    session: AsyncSession = Depends(get_session),
) -> SpendView:
    """This tenant's own spend. Never another's.

    Scoped to the caller's tenant rather than taking a tenant parameter, so there is no
    version of this endpoint that can read across tenants — the operator's cross-tenant view
    is a CLI against the database, not a route.
    """
    if not 1 <= days <= 365:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "days must be between 1 and 365")

    spend = await spend_for_tenant(session, tenant_id=tenant.tenant_id, days=days)
    if spend is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown or inactive tenant")

    return SpendView(
        tenant=spend.tenant_slug,
        window_days=days,
        investigations=spend.investigations,
        completed=spend.completed,
        failed=spend.failed,
        cancelled=spend.cancelled,
        usd=spend.usd,
        usd_per_investigation=round(spend.usd_per_investigation, 4),
        input_tokens=spend.input_tokens,
        output_tokens=spend.output_tokens,
        cache_read_tokens=spend.cache_read_tokens,
        cache_write_tokens=spend.cache_write_tokens,
        cache_hit_rate=round(spend.cache_hit_rate, 4),
        estimated=spend.estimated,
    )


@router.get("/reports/{report_id}")
async def get_report(
    report_id: uuid.UUID,
    tenant: TenantContext = Depends(require_tenant),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """The grounded report, as stored.

    Returned verbatim rather than re-rendered. The stored body is what the gate and
    verifier produced, and re-deriving it here would create a second path where a
    claim could reach a reader without passing through them.
    """
    row = (
        await session.execute(
            select(Report).where(Report.id == report_id, Report.tenant_id == tenant.tenant_id)
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown report")

    return {
        "id": str(row.id),
        "investigation_id": str(row.investigation_id),
        "confidence": row.confidence,
        "report": row.body,
        # Surfaced, not hidden. A reader is entitled to know that claims were removed
        # before this was shown to them.
        "removed_by_grounding": row.gate_rejections,
        "removed_by_verification": row.verifier_rejections,
    }
