"""Age-based retention.

Tenant deletion already cascades. This is the other promise: *"we do not keep your Slack
messages for three years"*, which is the one a customer actually asks about.

Three properties are worth proving, and only one of them is "old rows go away":

  - **It cannot reach across tenants.** A deletion job with a missing predicate is the worst
    possible bug in a multi-tenant product — silent, irreversible, and discovered by the
    victim.
  - **A run that stopped short says so.** Batching means a large tenant may not finish. If an
    incomplete sweep looked like a completed one, "has this deletion request been honoured"
    would have no honest answer.
  - **What it deliberately does not delete.** Reports outlive their evidence, and that is a
    decision rather than an oversight.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import (
    CredentialProvider,
    Evidence,
    Investigation,
    MetricPoint,
    Report,
    Tenant,
    ToolCall,
)
from cortex.ingest.retention import (
    AUDIT_DAYS,
    EVIDENCE_DAYS,
    METRIC_POINT_DAYS,
    apply_retention,
    tenants_to_sweep,
)
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.tenancy.context import TenantContext

NOW = datetime(2026, 7, 31, 4, 30, tzinfo=UTC)


async def _tenant(session: AsyncSession, slug: str) -> TenantContext:
    tenant_id = uuid.uuid4()
    graph_name = graph_name_for_new_tenant(slug, tenant_id)
    session.add(Tenant(id=tenant_id, slug=slug, name=slug, graph_name=graph_name))
    await session.flush()
    return TenantContext(tenant_id=tenant_id, tenant_slug=slug, graph_name=graph_name)


async def _investigation(session: AsyncSession, tenant: TenantContext) -> uuid.UUID:
    row = Investigation(tenant_id=tenant.tenant_id, question="why did signups fall?")
    session.add(row)
    await session.flush()
    return row.id


async def _evidence(
    session: AsyncSession, tenant: TenantContext, investigation_id: uuid.UUID, *, age_days: int
) -> uuid.UUID:
    row = Evidence(
        tenant_id=tenant.tenant_id,
        investigation_id=investigation_id,
        tool_name="slack",
        capability="search_messages",
        params={},
        payload={"messages": []},
        payload_hash="x",
        observed_at=NOW - timedelta(days=age_days),
    )
    session.add(row)
    await session.flush()
    return row.id


async def _audit(
    session: AsyncSession, tenant: TenantContext, investigation_id: uuid.UUID, *, age_days: int
) -> None:
    session.add(
        ToolCall(
            tenant_id=tenant.tenant_id,
            investigation_id=investigation_id,
            tool_name="slack",
            capability="search_messages",
            params={},
            succeeded=True,
            duration_ms=10,
            created_at=NOW - timedelta(days=age_days),
        )
    )
    await session.flush()


async def _point(
    session: AsyncSession, tenant: TenantContext, *, age_days: int, metric: str = "signups"
) -> None:
    session.add(
        MetricPoint(
            tenant_id=tenant.tenant_id,
            provider=CredentialProvider.POSTHOG,
            metric=metric,
            segment_key="",
            segment={},
            observed_at=NOW - timedelta(days=age_days),
            value=1.0,
            created_at=NOW - timedelta(days=age_days),
        )
    )
    await session.flush()


async def _count(session: AsyncSession, model: type, tenant: TenantContext) -> int:
    return int(
        await session.scalar(
            select(func.count()).select_from(model).where(model.tenant_id == tenant.tenant_id)
        )
        or 0
    )


class TestExpiryByAge:
    async def test_old_evidence_goes_and_recent_evidence_stays(self, session: AsyncSession) -> None:
        tenant = await _tenant(session, "ret-basic")
        investigation_id = await _investigation(session, tenant)
        await _evidence(session, tenant, investigation_id, age_days=EVIDENCE_DAYS + 10)
        await _evidence(session, tenant, investigation_id, age_days=30)
        await session.commit()

        outcome = await apply_retention(session, tenant, now=NOW)

        assert outcome.deleted.get("evidence") == 1
        assert await _count(session, Evidence, tenant) == 1

    async def test_metric_points_are_kept_longer_than_evidence(self, session: AsyncSession) -> None:
        """Numbers, not content — and year-on-year comparison is a real GTM question that a
        365-day window cannot answer."""
        tenant = await _tenant(session, "ret-metrics")
        await _point(session, tenant, age_days=EVIDENCE_DAYS + 10)
        await _point(session, tenant, age_days=METRIC_POINT_DAYS + 10, metric="old")
        await session.commit()

        await apply_retention(session, tenant, now=NOW)

        # The one older than evidence's window survived; the one past 400 days did not.
        rows = await session.execute(
            select(MetricPoint.metric).where(MetricPoint.tenant_id == tenant.tenant_id)
        )
        remaining = rows.scalars().all()
        assert remaining == ["signups"]

    async def test_audit_rows_outlive_the_evidence_they_describe(
        self, session: AsyncSession
    ) -> None:
        """An audit trail that expires before the data it describes cannot explain how that
        data arrived."""
        tenant = await _tenant(session, "ret-audit")
        investigation_id = await _investigation(session, tenant)
        await _audit(session, tenant, investigation_id, age_days=EVIDENCE_DAYS + 10)
        await _audit(session, tenant, investigation_id, age_days=AUDIT_DAYS + 10)
        await session.commit()

        await apply_retention(session, tenant, now=NOW)

        assert await _count(session, ToolCall, tenant) == 1

    async def test_nothing_to_delete_is_reported_as_nothing(self, session: AsyncSession) -> None:
        tenant = await _tenant(session, "ret-fresh")
        investigation_id = await _investigation(session, tenant)
        await _evidence(session, tenant, investigation_id, age_days=1)
        await session.commit()

        outcome = await apply_retention(session, tenant, now=NOW)

        assert outcome.total == 0
        assert outcome.incomplete == []


class TestItCannotReachAnotherTenant:
    """A deletion job with a missing predicate is the worst bug in a multi-tenant product:
    silent, irreversible, and discovered by the victim."""

    async def test_another_tenants_expired_rows_are_untouched(self, session: AsyncSession) -> None:
        victim = await _tenant(session, "ret-victim")
        swept = await _tenant(session, "ret-swept")
        for tenant in (victim, swept):
            investigation_id = await _investigation(session, tenant)
            await _evidence(session, tenant, investigation_id, age_days=EVIDENCE_DAYS + 50)
            await _audit(session, tenant, investigation_id, age_days=AUDIT_DAYS + 50)
            await _point(session, tenant, age_days=METRIC_POINT_DAYS + 50)
        await session.commit()

        await apply_retention(session, swept, now=NOW)

        assert await _count(session, Evidence, swept) == 0
        assert await _count(session, ToolCall, swept) == 0
        assert await _count(session, MetricPoint, swept) == 0
        # The victim's rows are every bit as expired, and every one of them survives.
        assert await _count(session, Evidence, victim) == 1
        assert await _count(session, ToolCall, victim) == 1
        assert await _count(session, MetricPoint, victim) == 1

    async def test_the_sweep_list_is_only_active_tenants(self, session: AsyncSession) -> None:
        """A suspended tenant is mid-dispute or mid-offboarding. Deleting their data on a
        schedule would resolve that question on their behalf."""
        await _tenant(session, "ret-active")
        suspended_id = uuid.uuid4()
        session.add(
            Tenant(
                id=suspended_id,
                slug="ret-suspended",
                name="suspended",
                graph_name=graph_name_for_new_tenant("ret-suspended", suspended_id),
                is_active=False,
            )
        )
        await session.commit()

        slugs = {t.tenant_slug for t in await tenants_to_sweep(session)}

        assert "ret-active" in slugs
        assert "ret-suspended" not in slugs

    async def test_the_graph_name_comes_from_the_row(self, session: AsyncSession) -> None:
        """Recomputed, it would address a different graph for any tenant whose slug was ever
        recycled — and the sweep would then expire a graph nobody reads."""
        tenant = await _tenant(session, "ret-graph")
        await session.commit()

        found = next(t for t in await tenants_to_sweep(session) if t.tenant_slug == "ret-graph")
        assert found.graph_name == tenant.graph_name


class TestWhatItDeliberatelyKeeps:
    async def test_a_report_outlives_its_evidence(self, session: AsyncSession) -> None:
        """A report is the deliverable; deleting one is a product decision, not a background
        job. The report layer already strips a claim whose citation no longer resolves, so an
        old report degrades honestly rather than lying."""
        tenant = await _tenant(session, "ret-report")
        investigation_id = await _investigation(session, tenant)
        await _evidence(session, tenant, investigation_id, age_days=EVIDENCE_DAYS + 10)
        session.add(
            Report(
                tenant_id=tenant.tenant_id,
                investigation_id=investigation_id,
                body={"question": "why did signups fall?"},
                created_at=NOW - timedelta(days=EVIDENCE_DAYS + 10),
            )
        )
        await session.commit()

        await apply_retention(session, tenant, now=NOW)

        assert await _count(session, Evidence, tenant) == 0
        assert await _count(session, Report, tenant) == 1


class TestBatchingIsHonestAboutFinishing:
    async def test_a_sweep_cut_short_names_what_it_could_not_finish(
        self, session: AsyncSession
    ) -> None:
        """If an incomplete sweep looked like a completed one, "has this deletion request been
        honoured" would have no honest answer."""
        tenant = await _tenant(session, "ret-batched")
        investigation_id = await _investigation(session, tenant)
        for _ in range(5):
            await _evidence(session, tenant, investigation_id, age_days=EVIDENCE_DAYS + 1)
        await session.commit()

        # One row per batch and two batches, so the ceiling is reached with rows to spare.
        import cortex.ingest.retention as retention

        original_batch, original_max = retention._BATCH, retention._MAX_BATCHES
        retention._BATCH, retention._MAX_BATCHES = 1, 2
        try:
            outcome = await apply_retention(session, tenant, now=NOW)
        finally:
            retention._BATCH, retention._MAX_BATCHES = original_batch, original_max

        assert outcome.deleted["evidence"] == 2
        assert "evidence" in outcome.incomplete
        # And the rest is still there, waiting for tomorrow.
        assert await _count(session, Evidence, tenant) == 3

    async def test_finishing_exactly_at_the_ceiling_is_not_incomplete(
        self, session: AsyncSession
    ) -> None:
        """Hitting the last batch exactly is not the same as being cut short, and reporting it
        as incomplete would make the report cry wolf."""
        tenant = await _tenant(session, "ret-exact")
        investigation_id = await _investigation(session, tenant)
        for _ in range(2):
            await _evidence(session, tenant, investigation_id, age_days=EVIDENCE_DAYS + 1)
        await session.commit()

        import cortex.ingest.retention as retention

        original_batch, original_max = retention._BATCH, retention._MAX_BATCHES
        retention._BATCH, retention._MAX_BATCHES = 1, 2
        try:
            outcome = await apply_retention(session, tenant, now=NOW)
        finally:
            retention._BATCH, retention._MAX_BATCHES = original_batch, original_max

        assert outcome.deleted["evidence"] == 2
        assert outcome.incomplete == []


class TestSemanticMemory:
    class _Vectors:
        def __init__(self, removed: int = 0, error: Exception | None = None) -> None:
            self.removed = removed
            self.error = error
            self.cutoffs: list[datetime] = []

        async def delete_older_than(self, ctx, *, cutoff, kinds=None):  # type: ignore[no-untyped-def]
            self.cutoffs.append(cutoff)
            if self.error is not None:
                raise self.error
            return self.removed

    async def test_expired_documents_are_deleted_and_counted(self, session: AsyncSession) -> None:
        tenant = await _tenant(session, "ret-docs")
        await session.commit()
        vectors = self._Vectors(removed=7)

        outcome = await apply_retention(session, tenant, vectors=vectors, now=NOW)  # type: ignore[arg-type]

        assert outcome.deleted["documents"] == 7
        assert vectors.cutoffs[0] == NOW - timedelta(days=180)

    async def test_a_vector_store_outage_does_not_report_a_deletion_that_did_not_happen(
        self, session: AsyncSession
    ) -> None:
        """Best-effort, because the vector store is not the system of record and a Qdrant
        outage must not stop the Postgres deletion the promise rests on. What it must never do
        is claim a deletion it did not perform."""
        tenant = await _tenant(session, "ret-docs-down")
        investigation_id = await _investigation(session, tenant)
        await _evidence(session, tenant, investigation_id, age_days=EVIDENCE_DAYS + 5)
        await session.commit()
        vectors = self._Vectors(error=RuntimeError("qdrant unreachable"))

        outcome = await apply_retention(session, tenant, vectors=vectors, now=NOW)  # type: ignore[arg-type]

        assert "documents" not in outcome.deleted
        # The Postgres side still happened.
        assert outcome.deleted["evidence"] == 1


class TestTheFutureClockGuard:
    """A `now` in the future turns every window into "delete everything".

    This guard exists because I did it. A probe against the live database, commented "rolled
    back, so nothing is lost", passed `now=+500 days` and deleted 58 evidence rows, 72 audit
    rows and 2,231 metric points — retention commits per batch by design, so there was no
    transaction to roll back. The metric points were re-ingestible; the evidence and audit
    rows were not.
    """

    async def test_a_future_now_is_refused(self, session: AsyncSession) -> None:
        from cortex.ingest.retention import UnsafeRetentionWindow

        tenant = await _tenant(session, "ret-guard")
        investigation_id = await _investigation(session, tenant)
        await _evidence(session, tenant, investigation_id, age_days=1)
        await session.commit()

        with pytest.raises(UnsafeRetentionWindow, match="in the future"):
            await apply_retention(session, tenant, now=datetime.now(UTC) + timedelta(days=500))

        # And nothing was deleted on the way to the refusal.
        assert await _count(session, Evidence, tenant) == 1

    async def test_the_guard_can_be_overridden_deliberately(self, session: AsyncSession) -> None:
        """A test against a disposable database is the legitimate use. It has to be explicit,
        because the recovery from getting this wrong is a re-ingest at best."""
        tenant = await _tenant(session, "ret-guard-override")
        investigation_id = await _investigation(session, tenant)
        await _evidence(session, tenant, investigation_id, age_days=1)
        await session.commit()

        outcome = await apply_retention(
            session,
            tenant,
            now=datetime.now(UTC) + timedelta(days=500),
            allow_future_now=True,
        )
        assert outcome.deleted["evidence"] == 1

    async def test_ordinary_clock_skew_is_tolerated(self, session: AsyncSession) -> None:
        """Workers disagree about the time by seconds, not by days. Refusing a minute of skew
        would make the nightly job fail for no reason."""
        tenant = await _tenant(session, "ret-skew")
        await session.commit()

        outcome = await apply_retention(
            session, tenant, now=datetime.now(UTC) + timedelta(minutes=5)
        )
        assert outcome.total == 0
