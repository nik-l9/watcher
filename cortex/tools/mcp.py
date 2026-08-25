"""MCP connector — one adapter, any number of tools.

Cortex ships eight hand-written connectors and each cost a day. This is how breadth arrives
without writing the ninth: a tenant points Cortex at an MCP server, and that server's tools appear
in the registry as ordinary capabilities, audited and cited exactly like native ones.

**Why MCP and not an ELT platform.** Recorded in full in `docs/decisions/0004-connector-breadth.md`.
The short version is shape: an Airbyte connector emits a *stream of records for a warehouse*, while
a claim in a Cortex report has to resolve to one request, one response, hashed, with a `source_ref`
a human can follow. MCP is request/response, so it fits the evidence contract instead of fighting
it.

## Four things this refuses to do, and why each is a refusal rather than a fallback

An MCP server is somebody else's code, declaring its own tools. Everything below is written on the
assumption that a server is careless rather than hostile — but the guards hold either way, and they
default to refusing rather than guessing.

**1. A tool that is not declared read-only is not offered.** V1 ships no destructive capability
anywhere; that is what makes "humans approve" a property of the tool surface rather than a prompt.
MCP servers routinely expose writes, so admission requires an explicit `readOnlyHint: true`
annotation. A tool with no annotation is refused — not because it is probably dangerous, but
because *we cannot tell*, and a tool surface that admits the unknown has stopped being a guarantee.

**2. A tool that returns prose is not offered.** Tools return JSON, never markdown, because prose
cannot be cited: there is nothing structured left to point at, and a paragraph in the evidence
store is an unverifiable claim wearing a citation. MCP's `content` blocks are usually text, so this
adapter requires `structuredContent` and treats its absence as a failed call rather than as data.

**3. Nothing is inferred about emptiness.** `result_key` is how the executor marks an empty
observation as empty, which is the defence against "no rows" being read as "no such thing" — a bug
this codebase has shipped four times. An MCP server's structured output is an arbitrary object, so
there is no key to guess. Instead the payload is wrapped under a known key, which makes emptiness
detectable without pretending to understand the server's schema.

**4. Credentials stay per tenant.** A server is configured per tenant and its secret lives in the
vault like any other. The registry filter that offers a tenant only what it has connected applies
unchanged, because this is an ordinary `Tool` with an ordinary provider.

## Transport

Streamable HTTP only: JSON-RPC 2.0 over POST. Deliberately not stdio, which would mean spawning a
subprocess per tenant — an isolation question that deserves its own decision rather than being
smuggled in behind a connector.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from cortex.db.models import CredentialProvider
from cortex.tools.base import (
    Capability,
    Freshness,
    InvalidParams,
    ToolContext,
    ToolResult,
    UpstreamError,
)
from cortex.tools.base import Tool as BaseTool
from cortex.tools.http import DEFAULT_TIMEOUT, request_json

#: The MCP protocol version this adapter speaks.
PROTOCOL_VERSION = "2025-06-18"

#: The payload key every MCP result is wrapped under.
#:
#: A server's `structuredContent` is an arbitrary object, so there is no way to know which of its
#: keys holds "the results" — and `result_key` exists precisely so the executor can tell an empty
#: observation from a real absence. Wrapping under a known key gives the executor something true to
#: check without this module pretending to understand a schema it has never seen.
RESULT_KEY = "result"

#: How many tools one server may contribute.
#:
#: A cap rather than a limit on ambition: every admitted tool's schema goes into the opening prompt
#: of every investigation, and `registry_for_tenant` exists because an oversized tool surface
#: measurably costs steps and latency. A server offering more than this is asking for a curated
#: allowlist, which `MCPServer.allow` provides.
MAX_TOOLS = 24


@dataclass(frozen=True, slots=True)
class MCPServer:
    """One MCP server a tenant has connected."""

    #: Short identifier, used as the Cortex tool name and in every evidence row.
    name: str
    url: str
    #: Optional allowlist of tool names. Empty means "every tool that passes the guards".
    allow: frozenset[str] = field(default_factory=frozenset)


class MCPToolRefused(InvalidParams):
    """A server's tool was not admitted. Carries the reason, which is operator-actionable."""


def admissible(descriptor: dict[str, Any], server: MCPServer) -> tuple[bool, str]:
    """Whether one advertised tool may be offered, and why not when it may not.

    Separated from the class so it can be tested against real `tools/list` payloads without a
    server, and so the admission rules read as a list rather than as control flow.
    """
    name = descriptor.get("name")
    if not isinstance(name, str) or not name:
        return False, "the tool has no name"
    if server.allow and name not in server.allow:
        return False, f"{name} is not in this server's allowlist"

    annotations = descriptor.get("annotations")
    annotations = annotations if isinstance(annotations, dict) else {}
    if annotations.get("readOnlyHint") is not True:
        # The conservative direction on purpose. An unannotated tool is not assumed dangerous; it
        # is assumed *unknown*, and a surface that admits the unknown is no longer a guarantee
        # that nothing destructive can be reached.
        return False, (
            f"{name} does not declare readOnlyHint: true. Cortex offers no capability it cannot "
            "establish is read-only, so an unannotated tool is refused rather than assumed safe."
        )

    schema = descriptor.get("inputSchema")
    if not isinstance(schema, dict) or schema.get("type") != "object":
        # The schema is handed to the model verbatim as the tool-call contract and is used to
        # validate arguments before dispatch. Without an object schema there is nothing to
        # validate against, and unvalidated arguments reach somebody else's API.
        return False, f"{name} has no object inputSchema to validate arguments against"

    return True, ""


def _tighten(schema: dict[str, Any]) -> dict[str, Any]:
    """The server's schema, with `additionalProperties: false` applied.

    Every Cortex capability must reject unknown parameters, because that is what turns a
    hallucinated argument into a validation error instead of a silently ignored field. MCP servers
    do not generally set it, so it is applied here.

    **This is the one place the adapter rewrites what a server declared, and it is worth being
    uncomfortable about.** Tightening cannot invent an argument or change the meaning of one — it
    can only refuse a name the server never advertised. The cost is real though: a tool that
    genuinely accepts undeclared extras will reject calls using them. That is the right side to err
    on when the tool belongs to somebody else, and a server wanting those extras can declare them.

    Copied rather than mutated, so a caller holding the descriptor does not find it altered
    underneath them.
    """
    return {**schema, "additionalProperties": False}


class MCPTool(BaseTool):
    """Every admitted tool on one MCP server, exposed as Cortex capabilities.

    Constructed with the descriptors a `tools/list` call returned, rather than discovering them
    itself, because `Tool.capabilities()` is synchronous and called during registry construction.
    Discovery is an async, network-touching act and belongs to whoever builds the registry — see
    `discover`.
    """

    provider = CredentialProvider.MCP

    def __init__(self, server: MCPServer, descriptors: list[dict[str, Any]]) -> None:
        self.server = server
        self.name = server.name
        self._descriptors = [d for d in descriptors if admissible(d, server)[0]][:MAX_TOOLS]
        if not self._descriptors:
            raise MCPToolRefused(
                f"mcp server {server.name!r} advertised no tool Cortex can offer. Every tool must "
                "declare readOnlyHint: true and an object inputSchema."
            )
        super().__init__()

    def capabilities(self) -> list[Capability]:
        return [self._capability(d) for d in self._descriptors]

    def _capability(self, descriptor: dict[str, Any]) -> Capability:
        name = str(descriptor["name"])
        description = str(descriptor.get("description") or "").strip()
        return Capability(
            name=name,
            description=(
                # The server's own description, with its origin stated. The analyst reasons about
                # what a tool is for from this text, and "which system is this?" is part of that.
                f"{description}\n\n(Provided by the {self.server.name!r} MCP server.)"
                if description
                else f"A tool provided by the {self.server.name!r} MCP server."
            ),
            params_schema=_tighten(descriptor["inputSchema"]),
            handler=self._handler(name),
            result_key=RESULT_KEY,
        )

    def _handler(self, tool_name: str):  # type: ignore[no-untyped-def]
        async def _call(ctx: ToolContext, **arguments: Any) -> ToolResult:
            payload = await self._call_tool(ctx, tool_name, arguments)
            return ToolResult(
                payload={RESULT_KEY: payload},
                source_ref=f"mcp://{self.server.name}/{tool_name}",
                meta={"freshness": Freshness.LIVE, "server": self.server.name},
            )

        return _call

    async def _call_tool(self, ctx: ToolContext, tool_name: str, arguments: dict[str, Any]) -> Any:
        body = await self._rpc(ctx, "tools/call", {"name": tool_name, "arguments": arguments})
        result = body.get("result")
        if not isinstance(result, dict):
            raise UpstreamError(f"mcp: {self.server.name}/{tool_name} returned no result object")

        if result.get("isError"):
            # The server reporting a tool-level failure, which is distinct from a transport
            # failure and is reported as the tool's own error rather than as a Cortex bug.
            raise UpstreamError(
                f"mcp: {self.server.name}/{tool_name} reported an error: {_text_of(result)[:200]}"
            )

        structured = result.get("structuredContent")
        if structured is None:
            # **The prose refusal.** A text block cannot be cited: there is nothing structured to
            # point at, and storing a paragraph as evidence would put an unverifiable claim behind
            # a citation that looks exactly like a verifiable one.
            raise UpstreamError(
                f"mcp: {self.server.name}/{tool_name} returned no structuredContent. Cortex "
                "cites structured observations, so a tool that answers only in prose cannot be "
                "used as evidence."
            )
        return structured

    async def _rpc(self, ctx: ToolContext, method: str, params: dict[str, Any]) -> dict[str, Any]:
        async with self._client(ctx) as client:
            body = await request_json(
                client,
                "POST",
                "",
                tool=self.name,
                json_body={
                    "jsonrpc": "2.0",
                    # A constant id is sufficient: one request per connection, and the response is
                    # awaited before another is sent. Nothing here multiplexes.
                    "id": 1,
                    "method": method,
                    "params": params,
                },
            )
        error = body.get("error")
        if isinstance(error, dict):
            raise UpstreamError(
                f"mcp: {self.server.name} rejected {method}: "
                f"{error.get('message', 'unknown error')}"
            )
        return body

    def _client(self, ctx: ToolContext) -> httpx.AsyncClient:
        headers = {
            "Content-Type": "application/json",
            # Both are required by the Streamable HTTP transport: a server may answer a single
            # JSON body or an SSE stream, and refuses a client that does not accept both.
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": PROTOCOL_VERSION,
        }
        if ctx.credential:
            headers["Authorization"] = f"Bearer {ctx.credential}"
        return httpx.AsyncClient(base_url=self.server.url, headers=headers, timeout=DEFAULT_TIMEOUT)


async def discover(
    server: MCPServer, *, credential: str | None, client: httpx.AsyncClient | None = None
) -> list[dict[str, Any]]:
    """Ask a server what it offers, returning the raw descriptors.

    Async and network-touching, so it is deliberately *not* part of `capabilities()`. A registry is
    built synchronously and often; discovery happens once, when a tenant connects a server, and its
    result is what `MCPTool` is constructed from.
    """
    owned = client is None
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": PROTOCOL_VERSION,
    }
    if credential:
        headers["Authorization"] = f"Bearer {credential}"
    http = client or httpx.AsyncClient(
        base_url=server.url, headers=headers, timeout=DEFAULT_TIMEOUT
    )
    try:
        body = await request_json(
            http,
            "POST",
            "",
            tool=server.name,
            json_body={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
    finally:
        if owned:
            await http.aclose()

    error = body.get("error")
    if isinstance(error, dict):
        raise UpstreamError(
            f"mcp: {server.name} rejected tools/list: {error.get('message', 'unknown error')}"
        )
    tools = (body.get("result") or {}).get("tools")
    return [t for t in tools if isinstance(t, dict)] if isinstance(tools, list) else []


def _text_of(result: dict[str, Any]) -> str:
    """The text blocks of an MCP result, joined — used only to quote a server's error back."""
    blocks = result.get("content")
    if not isinstance(blocks, list):
        return json.dumps(result)[:200]
    return " ".join(
        str(b.get("text", "")) for b in blocks if isinstance(b, dict) and b.get("type") == "text"
    )
