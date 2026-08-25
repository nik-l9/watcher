"""Ingest worker service.

Consumes only cortex.ingest. Pulls metric time-series and graph entities from
connected sources on a schedule, so the analyst has baselines to compare against —
without baselines, anomaly detection and the weekly brief are impossible.

Separate from the investigation worker because its load is bursty and nightly
while investigations are interactive: one HubSpot backfill must never delay a user
waiting on an answer.

This module is deliberately thin. Every decision — what to sync, over what window, what to
write, what to record — lives in `cortex.ingest`, so it is testable without a broker. What
belongs here is the queue contract and the retry policy, and nothing else.
"""

from __future__ import annotations

import uuid
from typing import Any

from celery.schedules import crontab
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.contracts.messages import QUEUE_INGEST, SyncConnector
from cortex.db.models import CredentialProvider, Tenant
from cortex.ingest.dispatch import nightly_targets, syncer_for
from cortex.ingest.retention import apply_retention, tenants_to_sweep
from cortex.ingest.runner import IngestRunner
from cortex.ingest.state import Window
from cortex.runtime.celery_app import make_celery
from cortex.runtime.resources import Resources, run_with_resources
from cortex.tenancy.context import TenantContext

SERVICE_NAME = "ingest-worker"

celery = make_celery(SERVICE_NAME)

# Beat schedule. Fan-out per tenant and per connector happens inside the
# dispatcher task, so adding a tenant needs no schedule change.
celery.conf.beat_schedule = {
    "nightly-connector-sync": {
        "task": "cortex.ingest.dispatch_nightly",
        "schedule": crontab(hour=2, minute=0),
    },
    # After the sync, not before: expiring rows first and then re-fetching the same window
    # would delete and rewrite the same data every night.
    "nightly-retention-sweep": {
        "task": "cortex.ingest.apply_retention",
        "schedule": crontab(hour=4, minute=30),
    },
}

#: How long to wait before re-attempting a provider whose every stream failed.
#:
#: Ten minutes, because the failures worth retrying are transient — a rate limit, a brief
#: outage — and the ones that are not (a revoked token) will fail identically in ten
#: minutes and stop after `max_retries` rather than hammering the upstream all night.
_RETRY_SECONDS = 600


@celery.task(name="cortex.ingest.dispatch_nightly")
def dispatch_nightly() -> dict:
    """Fan out one SyncConnector message per (tenant, connector).

    One message per connector rather than per tenant: a HubSpot outage should not
    block that tenant's GA4 sync, and retries stay narrow.
    """

    async def _work(resources: Resources) -> dict:
        async with resources.session() as session:
            targets = await nightly_targets(session)

        for target in targets:
            # Sent as each target is read rather than collected and sent at the end: a
            # crash partway through then leaves the connectors already dispatched running,
            # instead of losing the whole night.
            celery.send_task(
                "cortex.ingest.sync_connector",
                args=[
                    SyncConnector(
                        tenant_id=target.tenant_id,
                        provider=target.provider.value,
                        credential_label=target.label,
                    ).model_dump(mode="json")
                ],
                queue=QUEUE_INGEST,
            )

        return {
            "dispatched": len(targets),
            # Listed, because "0 dispatched" is equally consistent with no tenants, no
            # credentials, and no syncer for the providers that are connected — three
            # different problems that would otherwise look identical in a log.
            "targets": [f"{t.tenant_slug}:{t.provider.value}" for t in targets],
        }

    return run_with_resources(_work)


@celery.task(name="cortex.ingest.apply_retention")
def retention_sweep() -> dict:
    """Delete every active tenant's expired rows.

    One task rather than one per tenant, unlike the sync. The work is bounded per tenant by
    the batch ceiling, it touches no upstream that could rate-limit us, and a single task
    keeps the whole sweep's outcome in one place — which is what someone asks for when they
    ask whether retention ran.
    """

    async def _work(resources: Resources) -> dict:
        outcomes = []
        async with resources.session() as session:
            tenants = await tenants_to_sweep(session)

        for tenant in tenants:
            # A session per tenant, because retention commits per batch: one long-lived
            # session across every tenant would hold a transaction open for the whole sweep.
            async with resources.session() as session:
                outcome = await apply_retention(session, tenant, vectors=resources.vectors)
            outcomes.append(outcome.summary())

        swept = [o for o in outcomes if o["total"]]
        return {
            "tenants": len(outcomes),
            # Only tenants that actually had something to delete, so a log line stays
            # readable at a hundred tenants — but incomplete ones are always listed, because
            # a sweep that stopped short must not look like one that finished.
            "deleted": swept,
            "incomplete": [o["tenant"] for o in outcomes if o["incomplete"]],
        }

    return run_with_resources(_work)


@celery.task(name="cortex.ingest.sync_connector", bind=True, max_retries=3)
def sync_connector(self: Any, payload: dict) -> dict:  # noqa: ANN401 - Celery bind
    message = SyncConnector.model_validate(payload)
    message.require_supported()

    async def _work(resources: Resources) -> dict:
        # Upper-cased: SQLAlchemy persists this enum by name, so the stored value is
        # `GITHUB` while the message carries the lowercase member value.
        provider = CredentialProvider(message.provider)
        syncer = syncer_for(provider)
        runner = IngestRunner(resources.graph, resources.vectors)

        # An explicit window when the message carries one — a backfill, or a retry that has
        # to cover the same period as the attempt it replaces. The contract uses absolute
        # dates rather than "the last N days" for exactly this: a retry hours later must not
        # silently sync a different window from the one that failed.
        window = (
            Window(since=message.since, until=message.until, first_sync=False)
            if message.since and message.until
            else None
        )

        async with resources.session() as session:
            tenant = await _tenant_context(session, message.tenant_id)
            outcome = await runner.run(
                session,
                tenant,
                syncer,
                label=message.credential_label,
                window=window,
            )
        return outcome.summary()

    result = run_with_resources(_work)
    # Retried only when *every* stream failed, which is the signature of an outage or a bad
    # credential. One failed stream is already recorded in sync_state with its watermark
    # left where it was, so the next night re-reads it; retrying the whole provider to
    # re-attempt one stream would re-read the ones that succeeded for no gain.
    if _all_streams_failed(result):
        raise self.retry(countdown=_RETRY_SECONDS, exc=RuntimeError(f"ingest failed: {result}"))
    return result


def _all_streams_failed(summary: dict) -> bool:
    streams = summary.get("streams") or {}
    return bool(streams) and not any(s.get("ok") for s in streams.values())


async def _tenant_context(session: AsyncSession, tenant_id: uuid.UUID) -> TenantContext:
    """Load the tenant's identity, including its assigned graph name.

    Read from the row rather than recomputed. The graph name is assigned once at tenant
    creation and carries a uuid suffix, so deriving it here would address a different graph
    for any tenant whose slug was ever recycled — and the ingest would populate a graph no
    investigation reads.
    """
    row = (await session.execute(select(Tenant).where(Tenant.id == tenant_id))).scalar_one_or_none()
    if row is None:
        raise LookupError(f"tenant {tenant_id} does not exist")
    return TenantContext(
        tenant_id=row.id,
        tenant_slug=row.slug,
        graph_name=row.graph_name,
    )
