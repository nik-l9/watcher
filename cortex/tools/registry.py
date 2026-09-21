"""The GTM Data Analyst's toolset.

Assembled in one place so the employee definition names a registry rather than
enumerating connectors, and so future specialists (PMM, Sales, CS) get their own
registry without touching connector code.
"""

from __future__ import annotations

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import Credential, CredentialProvider
from cortex.tenancy.context import TenantContext
from cortex.tools.amplitude import AmplitudeTool
from cortex.tools.base import ToolError, ToolRegistry
from cortex.tools.bigquery import BigQueryTool
from cortex.tools.ga4 import GA4Tool
from cortex.tools.github import GitHubTool
from cortex.tools.hubspot import HubSpotTool
from cortex.tools.mcp import MCPServer, MCPTool, MCPToolRefused
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


log = structlog.get_logger(__name__)


#: Where `cortex.connect` records what a server advertised, inside `Credential.metadata_`.
MCP_URL = "url"
MCP_TOOLS = "mcp_tools"
#: Set when an operator accepted a server whose tools are not declared read-only.
MCP_ACCEPT_UNANNOTATED = "mcp_accept_unannotated"
#: Tool names the operator restricted the server to, if any.
MCP_ALLOW = "mcp_allow"


async def mcp_tools_for_tenant(session: AsyncSession, tenant: TenantContext) -> list[MCPTool]:
    """One tool per MCP server this tenant has connected, built from stored descriptors.

    **Built from storage rather than discovered here.** `registry_for_tenant` runs at the
    start of every investigation, and `discover` is a network call against a server we do
    not operate. Doing it here would put a third party on the critical path of every run:
    latency on each investigation, and a server that is merely slow turning into an
    investigation that fails. So discovery happens once, in `cortex.connect`, where a human
    is watching and can read what was refused -- and this reads the result.

    **A server that has not been discovered yet is skipped, not an error.** A credential
    stored before this wiring existed has no descriptors, and the fix is to re-run connect,
    not to refuse to investigate. Logged at warning so the gap is visible rather than
    mysterious.
    """
    rows = (
        (
            await session.execute(
                select(Credential).where(
                    Credential.tenant_id == tenant.tenant_id,
                    Credential.provider == CredentialProvider.MCP,
                )
            )
        )
        .scalars()
        .all()
    )

    tools: list[MCPTool] = []
    for row in rows:
        meta = row.metadata_ or {}
        url, descriptors = meta.get(MCP_URL), meta.get(MCP_TOOLS)
        # **Three states, not two.** Never discovered, discovered and everything refused,
        # and discovered with something usable. Collapsing the first two produced a warning
        # telling an operator to re-run connect when connect had already run and refused
        # every tool -- advice whose only outcome is the identical result. PostHog's server
        # is the case that found this: it advertises one tool, `exec`, annotated
        # `readOnlyHint: false` and `destructiveHint: true`, so nothing is admissible and
        # re-running changes nothing.
        if MCP_TOOLS not in meta or not url:
            log.warning(
                "mcp.not_discovered",
                label=row.label,
                reason="no stored descriptors; re-run cortex-connect for this server",
            )
            continue
        if not descriptors:
            log.warning(
                "mcp.nothing_admissible",
                label=row.label,
                reason="the server was queried and offered no tool Cortex can admit; "
                "every tool must declare readOnlyHint: true",
            )
            continue
        server = MCPServer(
            name=f"mcp_{row.label}",
            url=str(url),
            allow=frozenset(meta.get(MCP_ALLOW) or ()),
            accept_unannotated=bool(meta.get(MCP_ACCEPT_UNANNOTATED)),
        )
        try:
            tools.append(MCPTool(server, list(descriptors), credential_label=row.label))
        except MCPToolRefused as refused:
            # Every tool the server offers failed admission. That is a fact about the
            # server, not a reason to abandon the investigation: the other connectors
            # still work, and the operator needs to see why rather than find an empty
            # tool list.
            log.warning("mcp.all_tools_refused", label=row.label, reason=str(refused))
    return tools


def surface_is_provably_read_only(registry: ToolRegistry) -> bool:
    """Whether every capability offered has been established read-only.

    **The product prints a promise before each investigation**, and a tenant who excepted a
    server has made that promise false for themselves. Answering it from the assembled
    registry rather than from a setting means the banner cannot drift from the tool surface
    it describes: if an unannotated tool is reachable, this is False, whatever the config
    says.
    """
    return not any(
        isinstance(registry.get(name), MCPTool) and registry.get(name).server.accept_unannotated
        for name in registry.tool_names
    )


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
    # Registered into the source set before filtering, so an MCP tool passes the same
    # credential check as every other tool rather than bypassing it.
    for mcp_tool in await mcp_tools_for_tenant(session, tenant):
        source.register(mcp_tool)
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
