"""Which capabilities a tenant is offered.

Every real investigation was offered `ga4__*` and `bigquery__*`, spent steps calling them, got
`CredentialMissing`, and then disclosed the failure in the report. One live run built its entire
answer on a PostHog metric while apologising at length for absent GA4 data — which was not a
data problem at all: that tenant had no GA4 property and stored nothing in BigQuery.

Three costs, none of them cosmetic: steps spent against a wall-clock budget, an honest-looking
caveat about an irrelevance, and a transcript teaching the analyst that tools fail for no
reason.

**What these tests must not accidentally assert.** This is a planning improvement, not a
security boundary. `ToolExecutor` still resolves the credential per call and still raises
`CredentialMissing`; that check is unchanged and is the one that matters. The last test here
pins that down, so nobody later reads registry scoping as the thing keeping one tenant out of
another's data.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import Credential, CredentialProvider, Tenant
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.security.vault import encrypt_credential
from cortex.tenancy.context import TenantContext
from cortex.tools.registry import (
    NoToolsAvailable,
    gtm_analyst_registry,
    registry_for_tenant,
)


async def _tenant(session: AsyncSession, slug: str) -> TenantContext:
    tenant_id = uuid.uuid4()
    graph_name = graph_name_for_new_tenant(slug, tenant_id)
    session.add(Tenant(id=tenant_id, slug=slug, name=slug, graph_name=graph_name))
    await session.flush()
    return TenantContext(tenant_id=tenant_id, tenant_slug=slug, graph_name=graph_name)


async def _connect(
    session: AsyncSession, tenant: TenantContext, provider: CredentialProvider
) -> None:
    wrapped, ciphertext = encrypt_credential(tenant.tenant_id, provider.value, '{"token": "x"}')
    session.add(
        Credential(
            tenant_id=tenant.tenant_id,
            provider=provider,
            label="default",
            wrapped_data_key=wrapped,
            ciphertext=ciphertext,
        )
    )
    await session.flush()


class TestScoping:
    async def test_only_connected_providers_are_offered(self, session: AsyncSession) -> None:
        """The case this was found in: PostHog, GitHub and Slack connected, no GA4 and no
        BigQuery, so the analyst is never offered the two it cannot reach."""
        tenant = await _tenant(session, "scope-posthog")
        for provider in (
            CredentialProvider.POSTHOG,
            CredentialProvider.GITHUB,
            CredentialProvider.SLACK,
        ):
            await _connect(session, tenant, provider)

        registry = await registry_for_tenant(session, tenant)

        assert registry.tool_names == ["github", "posthog", "slack"]
        assert "ga4" not in registry.tool_names
        assert "bigquery" not in registry.tool_names

    async def test_the_capability_specs_shrink_with_it(self, session: AsyncSession) -> None:
        """The specs are what reaches the model. A registry that filtered `tool_names` but
        still emitted every spec would change nothing about what the analyst tries."""
        tenant = await _tenant(session, "scope-specs")
        await _connect(session, tenant, CredentialProvider.POSTHOG)

        registry = await registry_for_tenant(session, tenant)
        names = {spec["name"] for spec in registry.llm_tool_specs()}

        assert names
        assert all(name.startswith("posthog__") for name in names)
        assert len(names) < len(gtm_analyst_registry().llm_tool_specs())

    async def test_another_tenants_credentials_do_not_widen_the_registry(
        self, session: AsyncSession
    ) -> None:
        """The lookup is tenant-scoped. A shared query would offer one tenant a tool because
        a different tenant connected it — which is F-01's shape in a new place."""
        mine = await _tenant(session, "scope-mine")
        theirs = await _tenant(session, "scope-theirs")
        await _connect(session, mine, CredentialProvider.POSTHOG)
        await _connect(session, theirs, CredentialProvider.GA4)

        registry = await registry_for_tenant(session, mine)
        assert "ga4" not in registry.tool_names

    async def test_a_tenant_with_nothing_connected_is_refused(self, session: AsyncSession) -> None:
        """Not an empty registry. An analyst with no tools writes a fluent report grounded in
        nothing, which is the single output this system exists to prevent — and it could not
        disclose the cause from inside the loop, because "connect a credential" is not
        something the loop can observe."""
        tenant = await _tenant(session, "scope-empty")
        with pytest.raises(NoToolsAvailable, match="cortex.connect"):
            await registry_for_tenant(session, tenant)

    async def test_a_provider_less_tool_always_survives(self, session: AsyncSession) -> None:
        """The eval's `ScenarioTool` declares no provider, so a fixture run must be unaffected
        by which credentials a tenant happens to hold. Without this, scoping would silently
        empty the eval's registry and every scenario would fail for the wrong reason."""
        from cortex.eval.fixtures import SCENARIOS
        from cortex.eval.runner import scenario_registry

        tenant = await _tenant(session, "scope-fixture")
        fixtures = scenario_registry(SCENARIOS[0])

        registry = await registry_for_tenant(session, tenant, registry=fixtures)
        assert registry.tool_names == fixtures.tool_names


class TestItIsNotASecurityBoundary:
    async def test_the_executor_still_checks_the_credential(self, session: AsyncSession) -> None:
        """Scoping only decides what is *offered*. Availability must still be enforced at
        invocation, because a tool name can arrive from a model that was told about it on an
        earlier turn, from a replayed transcript, or from a caller that built its own registry.

        Asserted by calling a capability the tenant has not connected through the unfiltered
        registry — exactly the path scoping is *not* protecting — and requiring a refusal.
        """
        from cortex.db.models import Investigation
        from cortex.tools.base import CredentialMissing
        from cortex.tools.executor import ToolExecutor

        tenant = await _tenant(session, "scope-enforced")
        await _connect(session, tenant, CredentialProvider.POSTHOG)
        # A real investigation row: the executor checks tenant ownership of the investigation
        # *before* the credential (F-01), so a random id would fail the wrong assertion and
        # this test would pass without ever reaching the check it is about.
        investigation = Investigation(tenant_id=tenant.tenant_id, question="why did signups fall?")
        session.add(investigation)
        await session.flush()
        await session.commit()

        unfiltered = gtm_analyst_registry()
        executor = ToolExecutor(unfiltered)

        with pytest.raises(CredentialMissing):
            await executor.execute(
                session,
                tenant,
                investigation_id=investigation.id,
                qualified_name="ga4__get_sessions",
                params={"start_date": "2026-07-01", "end_date": "2026-07-07"},
            )
