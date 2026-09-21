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

import ipaddress
import json
import socket
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

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


#: Networks an MCP server may never live on, because a URL a *tenant* supplies is a URL an
#: attacker may supply. Without this, `MCPServer.url` is a server-side request forgery sink: a
#: tenant names `http://169.254.169.254/latest/meta-data/iam/security-credentials/` and the
#: gateway fetches cloud instance credentials on their behalf, from inside the trust boundary,
#: with the result handed back as an observation.
#:
#: Blocked by *resolved address*, not by hostname. A hostname check is bypassed by anything that
#: resolves inward -- `localtest.me`, a DNS record the attacker controls pointing at 127.0.0.1,
#: or an IPv6-mapped form of a v4 address. The address families below cover loopback, RFC1918,
#: link-local (which is where every cloud metadata service lives), CGNAT, and the v6 equivalents.
_FORBIDDEN_NETWORKS = tuple(
    ipaddress.ip_network(cidr)
    for cidr in (
        "127.0.0.0/8",  # loopback
        "10.0.0.0/8",  # RFC1918
        "172.16.0.0/12",  # RFC1918
        "192.168.0.0/16",  # RFC1918
        "169.254.0.0/16",  # link-local, and every cloud metadata endpoint
        "100.64.0.0/10",  # CGNAT
        "0.0.0.0/8",  # "this network"
        "::1/128",  # v6 loopback
        "fc00::/7",  # v6 unique-local
        "fe80::/10",  # v6 link-local
    )
)


def check_server_syntax(url: str) -> str:
    """The half of the URL check that costs nothing: scheme, credentials, a host at all.

    Split from the address check deliberately. This runs at construction, where it is free and
    catches the careless cases immediately; resolving DNS at construction would make building a
    dataclass do network I/O -- slow, flaky, and impossible to test without a network.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise InvalidParams(
            f"mcp: server url must use https, not {parsed.scheme or 'a missing scheme'!r}. "
            "A bearer token sent over http is readable in transit."
        )
    if parsed.username or parsed.password:
        raise InvalidParams("mcp: server url must not embed credentials; they end up in logs")
    if not parsed.hostname:
        raise InvalidParams(f"mcp: server url {url!r} has no host")
    return url


def check_server_url(url: str) -> str:
    """Return `url` if an MCP server may be reached at it, or raise saying why not.

    **`MCPServer.url` is tenant-supplied and reaches `httpx` directly**, which makes it the one
    place in this codebase where somebody else chooses what the server connects to. OWASP's SSRF
    guidance is to allowlist where the callable hosts are known and to validate the resolved
    address otherwise; an MCP server can legitimately live anywhere, so this is the second form.

    Three checks, and each blocks a different bypass:

      - **Scheme must be https.** `file://`, `gopher://` and `dict://` are the classic SSRF
        escalation schemes, and plain `http` would send the tenant's bearer token in clear.
      - **The host must resolve, and every address it resolves to must be public.** Checking the
        *resolved* address rather than the hostname is what stops `localtest.me`, an attacker's
        own DNS record pointing at 127.0.0.1, and `[::ffff:169.254.169.254]`. Every answer is
        checked, not the first: a name that resolves to one public and one private address would
        otherwise pass and then connect to either.
      - **No credentials in the URL.** `https://user:pass@host` puts a secret somewhere it will
        be logged.

    **What this does not close, stated rather than implied.** Between this check and the
    connection, DNS can change its answer -- the rebinding attack. Closing that needs the
    connection pinned to the address that was validated, which `httpx` does not expose a hook
    for; the residual risk is recorded in SECURITY.md rather than left for a reader to discover.
    """
    parsed = urlparse(check_server_syntax(url))
    host = parsed.hostname or ""
    try:
        resolved = socket.getaddrinfo(host, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise InvalidParams(f"mcp: server host {host!r} does not resolve ({exc})") from exc
    for entry in resolved:
        address = ipaddress.ip_address(entry[4][0])
        # v4-mapped v6 addresses are unwrapped first, so `::ffff:127.0.0.1` is caught as
        # loopback rather than passing as an unremarkable v6 address.
        if getattr(address, "ipv4_mapped", None) is not None:
            address = address.ipv4_mapped
        for network in _FORBIDDEN_NETWORKS:
            if address.version == network.version and address in network:
                raise InvalidParams(
                    f"mcp: server host {host!r} resolves to {address}, which is on a private or "
                    "link-local network. An MCP server must be reachable at a public address."
                )
    return url


@dataclass(frozen=True, slots=True)
class MCPServer:
    """One MCP server a tenant has connected."""

    #: Short identifier, used as the Cortex tool name and in every evidence row.
    name: str
    url: str
    #: Optional allowlist of tool names. Empty means "every tool that passes the guards".
    allow: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # Syntax at construction, address before connecting. A value that cannot be built
        # cannot be reached -- the reason `Capability` rejects a write capability here -- but
        # DNS belongs next to the socket, not next to the constructor.
        check_server_syntax(self.url)


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

    def __init__(
        self,
        server: MCPServer,
        descriptors: list[dict[str, Any]],
        *,
        credential_label: str | None = None,
    ) -> None:
        self.server = server
        self.name = server.name
        # One provider covers every MCP server, so the label is what distinguishes this
        # server's token from another's. Carried on the tool because the run-wide label
        # cannot name two servers at once. See `Tool.credential_label`.
        self.credential_label = credential_label
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
            # A session per call. Wasteful against a stateful server -- two extra round
            # trips -- but the alternative is a long-lived session held across an
            # investigation, which would need expiry and reconnection handling to be
            # correct. One call, one session, is the version that cannot go stale.
            session_id = await _handshake(client, self.server)
            body = _json_of(
                await request_json(
                    client,
                    "POST",
                    "",
                    tool=self.name,
                    json_body={
                        "jsonrpc": "2.0",
                        # A constant id is sufficient: one request per connection, and the
                        # response is awaited before another is sent. Nothing multiplexes.
                        "id": 1,
                        "method": method,
                        "params": params,
                    },
                    headers={SESSION_HEADER: session_id} if session_id else None,
                    raw_on_non_json=True,
                )
            )
        error = body.get("error")
        if isinstance(error, dict):
            raise UpstreamError(
                f"mcp: {self.server.name} rejected {method}: "
                f"{error.get('message', 'unknown error')}"
            )
        return body

    def _client(self, ctx: ToolContext) -> httpx.AsyncClient:
        # Every request goes through this factory, so judging the address here is what makes
        # "nothing reaches the network unvalidated" true of the whole class rather than of
        # the call sites someone remembered. Here rather than at construction because DNS
        # belongs next to the socket: its answer can change in between.
        check_server_url(self.server.url)
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


#: The header a server uses to hand back a session, and the client must echo on every later
#: request. Lower-cased at lookup because httpx normalises header names.
SESSION_HEADER = "mcp-session-id"


def _json_of(body: dict[str, Any] | str) -> dict[str, Any]:
    """One JSON-RPC response, whether the server answered JSON or an SSE stream.

    **Streamable HTTP lets a server answer either way for the same request**, and which one
    it picks is its business, not ours. A client that only parses JSON works against a fake
    and hangs against half the real servers, so the stream form is parsed here rather than
    treated as a transport error.
    """
    if isinstance(body, dict):
        return body
    for line in str(body).splitlines():
        if line.startswith("data:"):
            try:
                parsed = json.loads(line[5:].strip())
            except ValueError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return {}


async def _handshake(http: httpx.AsyncClient, server: MCPServer) -> str | None:
    """Initialise the session, returning the id the server wants echoed back.

    **Why this exists, and why its absence was invisible for so long.** The transport
    requires `initialize`, then a `notifications/initialized` acknowledgement, before any
    other method. `discover` went straight to `tools/list`. Every unit test passed, because
    a fake server answers whatever it is asked; every real server refused. HubSpot returned
    `400 Invalid request` and PostHog simply waited for a session that never came, which
    read as a timeout and looked like a network problem.

    A server that returns no session id is not an error: the header is optional, and a
    stateless server legitimately omits it.
    """
    response = await http.post(
        "",
        json={
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                # Cortex consumes tools and offers the server nothing: no sampling, no
                # roots, no elicitation. Declaring that plainly is also a small safety
                # property -- a server cannot ask us to call a model on its behalf.
                "capabilities": {},
                "clientInfo": {"name": "cortex", "version": "1"},
            },
        },
    )
    response.raise_for_status()
    session_id = response.headers.get(SESSION_HEADER)
    ack: dict[str, Any] = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    await http.post("", json=ack, headers={SESSION_HEADER: session_id} if session_id else None)
    return session_id


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
    # Re-checked here rather than trusted from construction: DNS can have changed its
    # answer since. See `check_server_url` for what this closes and what it does not.
    check_server_url(server.url)
    try:
        session_id = await _handshake(http, server)
        body = _json_of(
            await request_json(
                http,
                "POST",
                "",
                tool=server.name,
                json_body={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                headers={SESSION_HEADER: session_id} if session_id else None,
                raw_on_non_json=True,
            )
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
