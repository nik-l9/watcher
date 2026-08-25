"""Watermarks and connector health — one row, one transaction.

The watermark says how far a stream has been ingested; the health record says whether the
last attempt worked. They live in the same row on purpose. Kept separately, they can
disagree — a watermark that advanced while the health row reads `failed` describes a sync
that both did and did not happen, and afterwards nothing can tell you which is true.

Two rules the rest of the ingest depends on:

  - **A watermark advances only on success.** A failed pull leaves the cursor where it was,
    so the next run re-reads that window. Re-reading is cheap and idempotent by design;
    skipping a window loses data silently, and nothing downstream would ever notice.
  - **The window overlaps deliberately.** Upstreams backfill late-arriving data — GitHub
    backfills a deployment status, PostHog processes an event minutes after it happened —
    so a sync starting exactly at the previous watermark would miss anything that landed
    behind it. `OVERLAP` re-reads a margin, and the uniqueness constraints on
    `metric_points` and the graph's `(label, key)` merge make that a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import CredentialProvider, SyncState, SyncStatus
from cortex.tenancy.context import TenantContext

#: How far back before the watermark to re-read.
#:
#: Six hours rather than minutes: the failure it guards against is an upstream that
#: attributes a record to a timestamp *before* the moment it became visible, and the lag
#: there is measured in hours for deployment statuses and analytics pipelines. The cost of
#: being generous is a re-read that writes nothing.
OVERLAP = timedelta(hours=6)

#: How much history a first sync pulls.
#:
#: Enough to establish a baseline — "is this week unusual" needs several weeks to compare
#: against — and bounded, because an unbounded first sync against a busy repository is an
#: hour of API calls and a rate limit.
FIRST_SYNC_WINDOW = timedelta(days=90)

#: When a stream is old enough that a report must say so.
STALE_AFTER = timedelta(hours=36)


@dataclass(frozen=True, slots=True)
class Window:
    """The period one stream will be asked for."""

    since: datetime
    until: datetime
    #: True when there was no previous watermark, so this is a backfill.
    first_sync: bool

    @property
    def since_date(self) -> str:
        """The `since` boundary as YYYY-MM-DD, which is what most connectors take."""
        return self.since.date().isoformat()

    @property
    def until_date(self) -> str:
        return self.until.date().isoformat()


async def read_state(
    session: AsyncSession,
    tenant: TenantContext,
    *,
    provider: CredentialProvider,
    label: str,
    stream: str,
) -> SyncState | None:
    return await session.scalar(
        select(SyncState).where(
            # Tenant-scoped like every other read: a watermark is not shared.
            SyncState.tenant_id == tenant.tenant_id,
            SyncState.provider == provider,
            SyncState.label == label,
            SyncState.stream == stream,
        )
    )


def window_for(state: SyncState | None, *, now: datetime) -> Window:
    """What period to ask the upstream for.

    `now` is passed in rather than read here so a sync is reproducible: a retry hours later
    must be able to cover the window the original attempt was given, and a function that
    reads the clock itself cannot be asked to.
    """
    if state is None or state.watermark is None:
        return Window(since=now - FIRST_SYNC_WINDOW, until=now, first_sync=True)
    return Window(since=state.watermark - OVERLAP, until=now, first_sync=False)


async def record_success(
    session: AsyncSession,
    tenant: TenantContext,
    *,
    provider: CredentialProvider,
    label: str,
    stream: str,
    watermark: datetime,
    items_written: int,
    now: datetime,
    detail: str | None = None,
    note: str | None = None,
) -> SyncState:
    """Advance the cursor and clear the failure count, in one write.

    `detail` means the data is **incomplete**; `note` means something is worth recording
    about a complete result. Only `detail` sets PARTIAL, because a report that calls a
    genuinely empty change log "incomplete" tells a reader their history might be missing
    entries when the truth is that nobody writes any.
    """
    state = await read_state(session, tenant, provider=provider, label=label, stream=stream)
    if state is None:
        state = SyncState(tenant_id=tenant.tenant_id, provider=provider, label=label, stream=stream)
        session.add(state)

    # Never moved backwards. A retry given an older window than a run that already
    # succeeded must not rewind the cursor and cause the next run to re-read weeks.
    state.watermark = max(watermark, state.watermark) if state.watermark else watermark
    # PARTIAL when the upstream refused part of what was asked for: the data is usable, so
    # this is not a failure, but a report drawing on it has to disclose the gap.
    state.status = SyncStatus.PARTIAL if detail else SyncStatus.OK
    state.detail = "; ".join(part for part in (detail, note) if part) or None
    state.last_run_at = now
    state.last_success_at = now
    state.consecutive_failures = 0
    state.items_written = items_written
    await session.flush()
    return state


async def record_failure(
    session: AsyncSession,
    tenant: TenantContext,
    *,
    provider: CredentialProvider,
    label: str,
    stream: str,
    detail: str,
    now: datetime,
) -> SyncState:
    """Record the failure and leave the watermark alone.

    `last_run_at` moves but `last_success_at` does not, and the gap between them is what
    makes staleness visible. A connector failing every night for a week still has a recent
    `last_run_at`, and reporting that as health would be a lie of omission.
    """
    state = await read_state(session, tenant, provider=provider, label=label, stream=stream)
    if state is None:
        state = SyncState(tenant_id=tenant.tenant_id, provider=provider, label=label, stream=stream)
        session.add(state)

    state.status = SyncStatus.FAILED
    state.detail = detail[:2000]
    state.last_run_at = now
    state.consecutive_failures = (state.consecutive_failures or 0) + 1
    await session.flush()
    return state


def is_stale(state: SyncState, *, now: datetime | None = None) -> bool:
    """Whether this stream's data is old enough that a report must disclose it.

    Measured from the last *success*, not the last run. A stream that has been failing
    since Tuesday is stale however often the job has attempted it since.
    """
    moment = now or datetime.now(UTC)
    if state.last_success_at is None:
        return True
    last = state.last_success_at
    if last.tzinfo is None:
        # Postgres returns aware datetimes, but a hand-built row in a test may not, and
        # comparing naive to aware raises rather than returning a wrong answer.
        last = last.replace(tzinfo=UTC)
    return moment - last > STALE_AFTER
