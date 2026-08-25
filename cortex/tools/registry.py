"""The GTM Data Analyst's toolset.

Assembled in one place so the employee definition names a registry rather than
enumerating connectors, and so future specialists (PMM, Sales, CS) get their own
registry without touching connector code.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import Credential
from cortex.tenancy.context import TenantContext
from cortex.tools.amplitude import AmplitudeTool
from cortex.tools.base import ToolError, ToolRegistry
from cortex.tools.bigquery import BigQueryTool
from cortex.tools.ga4 import GA4Tool
from cortex.tools.github import GitHubTool
from cortex.tools.hubspot import HubSpotTool
from cortex.tools.mixpanel import MixpanelTool
from cortex.tools.posthog import PostHogTool
from cortex.tools.slack import SlackTool


class NoToolsAvailable(ToolError):
    """A tenant with no connected credentials, so nothing is reachable.

    Raised rather than returning an empty registry: an analyst with no tools writes a fluent
    report grounded in nothing, which is precisely the output this system exists to prevent.
    """


def gtm_analyst_registry() -> ToolRegistry:
    """Every tool the GTM Data Analyst may use. All read-only.

    A new registry per call rather than a module-level singleton: tools are cheap
    to construct, and a shared mutable registry is the kind of global state that
    later turns into a cross-tenant bug.
    """
    registry = ToolRegistry()
    for tool in (
        AmplitudeTool(),
        GA4Tool(),
        BigQueryTool(),
        GitHubTool(),
        HubSpotTool(),
        MixpanelTool(),
        PostHogTool(),
        SlackTool(),
    ):
        registry.register(tool)
    return registry


async def registry_for_tenant(
    session: AsyncSession, tenant: TenantContext, *, registry: ToolRegistry | None = None
) -> ToolRegistry:
    """The subset of tools this tenant can actually use.

    **Why this exists.** Every real investigation was offered `ga4__*` and `bigquery__*`, spent
    steps calling them, got `CredentialMissing`, and then disclosed the failure in the report's
    risks and data-quality sections. One live run built its whole answer on a PostHog metric
    while apologising at length for absent GA4 data. None of that was a data problem — the tenant
    had no GA4 property and stored nothing in BigQuery — it was the tool surface offering
    capabilities the tenant could not reach.

    The cost was threefold: wasted steps against a wall-clock budget, an honest-looking caveat
    about an absence that was never relevant, and a transcript teaching the analyst that tools
    fail for no reason.

    **Filtered rather than deleted.** The GA4 and BigQuery connectors stay in the product; a
    different tenant may well hold those credentials. Availability is a property of the tenant,
    so it is resolved per tenant rather than by editing the registry.

    **This is not a security boundary and must not be read as one.** `ToolExecutor` still
    resolves the credential per call, tenant-scoped, and still raises `CredentialMissing` — that
    is the check that matters, and it is unchanged. This only stops the analyst being *offered*
    what it cannot use, which is a planning improvement.

    Provider-less tools always survive: the eval's `ScenarioTool` declares no provider, so a
    fixture run is unaffected by which credentials a tenant happens to hold.
    """
    source = registry or gtm_analyst_registry()
    connected = set(
        (
            await session.execute(
                select(Credential.provider).where(Credential.tenant_id == tenant.tenant_id)
            )
        )
        .scalars()
        .all()
    )

    available = ToolRegistry()
    for name in source.tool_names:
        tool = source.get(name)
        if tool.provider is None or tool.provider in connected:
            available.register(tool)

    if not available.tool_names:
        # Failing loudly beats investigating with no tools. An analyst with an empty toolset
        # produces a fluent report grounded in nothing, which is the one output this system
        # exists to prevent -- and the cause ("connect something") is not something it could
        # discover or disclose from inside the loop.
        raise NoToolsAvailable(
            f"tenant {tenant.tenant_slug!r} has no connected credentials, so no capability is "
            "reachable. Run `python -m cortex.connect` before investigating."
        )
    return available
