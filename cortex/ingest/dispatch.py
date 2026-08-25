"""Deciding what to sync tonight, and doing one unit of it.

Two functions, deliberately separate from the Celery tasks that call them. The task layer
owns retries and queues; this owns the decisions, and keeping them apart is what makes the
decisions testable without a broker.

**Fan-out is per (tenant, provider), not per tenant.** A HubSpot outage must not delay a
tenant's GitHub sync, and a retry should re-attempt the connector that failed rather than
everything. One message per connector is also what keeps a retry's blast radius equal to
its cause.

**Only connected providers are enqueued.** The credential table is the source of truth for
what a tenant has connected, so a provider nobody connected produces no message rather than
a message that fails with "not connected" every night.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import Credential, CredentialProvider, Tenant
from cortex.ingest.base import Syncer
from cortex.ingest.github import GitHubSyncer
from cortex.ingest.posthog import PostHogSyncer
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.tenancy.context import TenantContext

#: Which syncer handles which provider.
#:
#: Providers absent from this map are connected but not yet ingested — GA4, HubSpot and
#: Slack are read live during an investigation and will gain syncers as their streams are
#: designed. Absence here is why they produce no nightly message, and it is a fact worth
#: reading off one table rather than inferring from what does not crash.
SYNCERS: dict[CredentialProvider, type[Syncer]] = {
    CredentialProvider.GITHUB: GitHubSyncer,
    CredentialProvider.POSTHOG: PostHogSyncer,
}


@dataclass(frozen=True, slots=True)
class SyncTarget:
    """One unit of nightly work."""

    tenant_id: uuid.UUID
    tenant_slug: str
    graph_name: str
    provider: CredentialProvider
    label: str

    @property
    def tenant(self) -> TenantContext:
        return TenantContext(
            tenant_id=self.tenant_id,
            tenant_slug=self.tenant_slug,
            # Read from the row rather than derived: the graph name is assigned once at
            # tenant creation, and recomputing it would produce a different name for a
            # tenant whose slug was ever recycled.
            graph_name=self.graph_name,
        )


async def nightly_targets(session: AsyncSession) -> list[SyncTarget]:
    """Every (tenant, provider, label) with both a credential and a syncer."""
    rows = (
        await session.execute(
            select(
                Tenant.id,
                Tenant.slug,
                Tenant.graph_name,
                Credential.provider,
                Credential.label,
            )
            .join(Credential, Credential.tenant_id == Tenant.id)
            .order_by(Tenant.slug, Credential.provider, Credential.label)
        )
    ).all()

    return [
        SyncTarget(
            tenant_id=tenant_id,
            tenant_slug=slug,
            graph_name=graph_name or graph_name_for_new_tenant(slug, tenant_id),
            provider=provider,
            label=label,
        )
        for tenant_id, slug, graph_name, provider, label in rows
        if provider in SYNCERS
    ]


def syncer_for(provider: CredentialProvider) -> Syncer:
    factory = SYNCERS.get(provider)
    if factory is None:
        raise KeyError(
            f"no ingest syncer for {provider.value}; connected providers without a syncer "
            f"are read live during an investigation. Have: "
            f"{', '.join(sorted(p.value for p in SYNCERS))}"
        )
    return factory()


def supported_providers() -> Sequence[str]:
    return sorted(p.value for p in SYNCERS)
