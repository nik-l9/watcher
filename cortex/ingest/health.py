"""Connector health, turned into something a report has to say.

The plan is explicit that a stale connector must appear in a report's confidence section,
and the reason is the failure mode this codebase keeps running into: **absence of access is
indistinguishable from absence of evidence.** A tenant whose GitHub sync has been failing
since Tuesday has a graph that looks complete and is three days behind, and an
investigation reading it would honestly report "no deploys that week" from data that simply
was not fetched.

So staleness is not advisory here. It is converted into `DataQualityNote`s that travel with
the report, in the same section as sampling and thresholding disclosures, because they are
the same kind of statement: *what this answer cannot see.*

Only real problems produce a note. A healthy sync says nothing — a report that discloses
every connector's status on every question buries the one disclosure that matters.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import SyncState, SyncStatus
from cortex.ingest.state import is_stale
from cortex.reports.schema import DataQualityNote
from cortex.tenancy.context import TenantContext


async def health_notes(
    session: AsyncSession,
    tenant: TenantContext,
    *,
    now: datetime | None = None,
) -> list[DataQualityNote]:
    """Disclosures a report must carry about the freshness of its inputs.

    Empty when everything is current, which is the common case and the one that should add
    nothing to a report.
    """
    moment = now or datetime.now(UTC)
    rows = (
        (
            await session.execute(
                select(SyncState)
                .where(SyncState.tenant_id == tenant.tenant_id)
                .order_by(SyncState.provider, SyncState.stream)
            )
        )
        .scalars()
        .all()
    )

    notes: list[DataQualityNote] = []
    for row in rows:
        note = _describe(row, moment)
        if note:
            notes.append(DataQualityNote(note=note[:500]))
    return notes


def _describe(row: SyncState, now: datetime) -> str | None:
    """One stream's disclosure, or None when it is healthy.

    Three distinct conditions, phrased differently on purpose. "Failing" tells a reader the
    data has stopped arriving; "stale" tells them how old what they have is; "incomplete"
    tells them the data is current but truncated. A single "connector unhealthy" would
    collapse three different limits on the answer into one unactionable sentence.
    """
    label = f"{row.provider.value} {row.stream}"

    if row.status is SyncStatus.FAILED:
        last = (
            f"last succeeded {_age(row.last_success_at, now)} ago"
            if row.last_success_at
            else "has never succeeded"
        )
        return (
            f"The {label} sync is failing ({last}): {row.detail or 'no reason recorded'}. "
            f"Anything after that point is missing from stored history, so an absence here "
            f"is not evidence that nothing happened."
        )

    if is_stale(row, now=now):
        return (
            f"The {label} sync last succeeded {_age(row.last_success_at, now)} ago. Stored "
            f"history for it is that old, and a recent change may not appear."
        )

    if row.status is SyncStatus.PARTIAL and row.detail:
        return (
            f"The {label} sync is current but incomplete: {_report_safe(row.detail)}. Treat "
            f"counts drawn from it as a floor rather than a total."
        )

    return None


def _report_safe(detail: str) -> str:
    """A recorded failure, reduced to what belongs in front of a reader.

    `sync_state.detail` is a diagnostic field and holds whatever the upstream said. A real
    run put Voyage's rate-limit response — including a link to its billing dashboard and an
    explanation of its free-tier token allowance — on course for a customer-facing report.
    That is our vendor's internals leaking into someone's answer about their own funnel.

    So the body is cut at the first brace or URL and capped. What survives is the part that
    tells a reader something about their data; what goes is the part that tells them about
    our infrastructure.
    """
    text = " ".join(detail.split())
    for marker in ("{", "http://", "https://"):
        index = text.find(marker)
        if index > 0:
            text = text[:index]
    text = text.strip().rstrip(":").strip()
    return text[:200] if len(text) <= 200 else text[:197] + "..."


def _age(moment: datetime | None, now: datetime) -> str:
    if moment is None:
        return "an unknown time"
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    delta = now - moment
    hours = delta.total_seconds() / 3600
    if hours < 48:
        return f"{int(hours)} hours"
    return f"{int(hours / 24)} days"
