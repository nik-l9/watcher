"""The MCP adapter.

An MCP server is somebody else's code declaring its own tools, so almost every test here is about
something the adapter **refuses**. That is the point of the module: breadth arrives without any
loosening of the three properties the tool framework guarantees — read-only, structured, and
honest about emptiness.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from cortex.tools.base import ToolContext, UpstreamError
from cortex.tools.mcp import (
    MAX_TOOLS,
    RESULT_KEY,
    MCPServer,
    MCPTool,
    MCPToolRefused,
    admissible,
    discover,
)

SERVER = MCPServer(name="acme_mcp", url="https://mcp.example/rpc")


def _tool(name: str = "search_tickets", **overrides: Any) -> dict[str, Any]:
    """A descriptor that passes every guard, so a test can break exactly one thing."""
    return {
        "name": name,
        "description": "Search support tickets.",
        "annotations": {"readOnlyHint": True},
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
        **overrides,
    }


def _ctx(tenant: object) -> ToolContext:
    return ToolContext(tenant=tenant, credential="mcp-token")  # type: ignore[arg-type]


def _rpc(result: dict[str, Any] | None = None, error: dict[str, Any] | None = None) -> dict:
    body: dict[str, Any] = {"jsonrpc": "2.0", "id": 1}
    if error is not None:
        body["error"] = error
    else:
        body["result"] = result or {}
    return body


class TestOnlyReadOnlyToolsAreOffered:
    """V1 ships no destructive capability anywhere, which is what makes "humans approve" a
    property of the tool surface rather than a prompt an agent could talk itself out of. MCP
    servers routinely expose writes."""

    def test_a_read_only_tool_is_admitted(self) -> None:
        ok, why = admissible(_tool(), SERVER)
        assert ok and why == ""

    def test_a_write_tool_is_refused(self) -> None:
        ok, why = admissible(_tool("delete_ticket", annotations={"readOnlyHint": False}), SERVER)
        assert not ok
        assert "readOnlyHint" in why

    def test_an_unannotated_tool_is_refused_rather_than_assumed_safe(self) -> None:
        """The conservative direction, deliberately. An unannotated tool is not assumed dangerous
        — it is *unknown*, and a surface that admits the unknown has stopped being a guarantee."""
        descriptor = _tool("mystery")
        descriptor.pop("annotations")
        ok, why = admissible(descriptor, SERVER)
        assert not ok
        assert "refused rather than assumed safe" in why

    @pytest.mark.parametrize("annotations", [{"readOnlyHint": "true"}, {"readOnlyHint": 1}, {}])
    def test_a_truthy_but_wrong_annotation_does_not_pass(self, annotations: dict[str, Any]) -> None:
        """`is not True` rather than a truthiness check: the string "true" and the integer 1 are
        both truthy and neither is the declaration the spec defines."""
        ok, _ = admissible(_tool("x", annotations=annotations), SERVER)
        assert not ok

    def test_a_tool_outside_the_allowlist_is_refused(self) -> None:
        server = MCPServer(name="s", url="https://x", allow=frozenset({"search_tickets"}))
        assert admissible(_tool("search_tickets"), server)[0]
        assert not admissible(_tool("something_else"), server)[0]

    def test_a_server_offering_nothing_admissible_is_refused_loudly(self) -> None:
        """Registering a tool with no capabilities would be a silent no-op; an operator who
        connected a server needs to know it contributed nothing, and why."""
        with pytest.raises(MCPToolRefused, match="readOnlyHint"):
            MCPTool(SERVER, [_tool("w", annotations={"readOnlyHint": False})])


class TestArgumentsAreValidatable:
    def test_a_tool_without_an_object_schema_is_refused(self) -> None:
        """The schema is the tool-call contract handed to the model *and* what arguments are
        validated against before dispatch. Without one, unvalidated arguments reach somebody
        else's API."""
        ok, why = admissible(_tool("x", inputSchema={"type": "string"}), SERVER)
        assert not ok
        assert "inputSchema" in why

    def test_the_servers_schema_survives_except_for_one_tightening(self) -> None:
        """Everything the server declared is preserved; the only addition is
        `additionalProperties: false`.

        Every Cortex capability must reject unknown parameters — that is what turns a hallucinated
        argument into a validation error rather than a silently ignored field — and MCP servers do
        not generally set it. Tightening cannot invent an argument or change what one means; it can
        only refuse a name the server never advertised, which is the right side to err on when the
        tool belongs to somebody else."""
        schema = {"type": "object", "required": ["query"], "properties": {"query": {}}}
        tool = MCPTool(SERVER, [_tool(inputSchema=schema)])
        tightened = tool.capability("search_tickets").params_schema

        assert tightened["required"] == ["query"]
        assert tightened["properties"] == {"query": {}}
        assert tightened["additionalProperties"] is False

    def test_the_servers_descriptor_is_not_mutated(self) -> None:
        """A caller holding the descriptor -- an operator inspecting what a server offered --
        must not find it altered underneath them."""
        descriptor = _tool()
        MCPTool(SERVER, [descriptor])
        assert "additionalProperties" not in descriptor["inputSchema"]


class TestResultsMustBeStructured:
    """Tools return JSON, never prose. A paragraph in the evidence store is an unverifiable claim
    wearing a citation that looks exactly like a verifiable one."""

    async def test_structured_content_becomes_the_payload(
        self, tenant: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tool = MCPTool(SERVER, [_tool()])
        _stub(
            monkeypatch,
            tool,
            _rpc({"structuredContent": {"tickets": [{"id": 7}], "count": 1}}),
        )
        result = await tool.capability("search_tickets").handler(_ctx(tenant), query="crash")

        assert result.payload == {RESULT_KEY: {"tickets": [{"id": 7}], "count": 1}}
        assert result.source_ref == "mcp://acme_mcp/search_tickets"

    async def test_a_prose_only_answer_is_refused(
        self, tenant: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tool = MCPTool(SERVER, [_tool()])
        _stub(
            monkeypatch,
            tool,
            _rpc({"content": [{"type": "text", "text": "There were three tickets."}]}),
        )
        with pytest.raises(UpstreamError, match="no structuredContent"):
            await tool.capability("search_tickets").handler(_ctx(tenant), query="crash")

    async def test_a_tool_level_error_is_reported_as_the_tools_error(
        self, tenant: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Distinct from a transport failure: the server worked and the tool did not, and an
        operator reading the trace needs to be able to tell those apart."""
        tool = MCPTool(SERVER, [_tool()])
        _stub(
            monkeypatch,
            tool,
            _rpc(
                {
                    "isError": True,
                    "content": [{"type": "text", "text": "rate limited by upstream"}],
                }
            ),
        )
        with pytest.raises(UpstreamError, match="rate limited by upstream"):
            await tool.capability("search_tickets").handler(_ctx(tenant), query="crash")

    async def test_a_json_rpc_error_is_surfaced(
        self, tenant: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tool = MCPTool(SERVER, [_tool()])
        _stub(monkeypatch, tool, _rpc(error={"code": -32601, "message": "method not found"}))
        with pytest.raises(UpstreamError, match="method not found"):
            await tool.capability("search_tickets").handler(_ctx(tenant), query="crash")


class TestEmptinessStaysDetectable:
    def test_every_capability_declares_the_wrapper_key(self) -> None:
        """`result_key` is how the executor marks an empty observation as empty — the defence
        against "no rows" being read as "no such thing", which this codebase has shipped four
        times. A server's structured output is an arbitrary object with no key to guess, so the
        payload is wrapped under a known one instead of inferring a schema."""
        tool = MCPTool(SERVER, [_tool(), _tool("list_projects")])
        assert {c.result_key for c in tool.capabilities()} == {RESULT_KEY}

    async def test_an_empty_result_is_preserved_as_empty(
        self, tenant: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not converted to an error or a default. An empty observation is a real finding, and
        the wrapper is what lets the executor say so."""
        tool = MCPTool(SERVER, [_tool()])
        _stub(monkeypatch, tool, _rpc({"structuredContent": {"tickets": [], "count": 0}}))
        result = await tool.capability("search_tickets").handler(_ctx(tenant), query="nothing")
        assert result.payload[RESULT_KEY] == {"tickets": [], "count": 0}


class TestTheToolSurfaceStaysBounded:
    def test_a_server_cannot_contribute_unlimited_tools(self) -> None:
        """Every admitted schema goes into the opening prompt of every investigation.
        `registry_for_tenant` exists because an oversized surface measurably costs steps and
        latency — a single server must not be able to undo that."""
        tool = MCPTool(SERVER, [_tool(f"tool_{i}") for i in range(MAX_TOOLS + 10)])
        assert len(tool.capabilities()) == MAX_TOOLS

    def test_the_server_name_identifies_the_capability_source(self) -> None:
        """The analyst reasons about what a tool is for from its description, and "whose system
        is this?" is part of that."""
        tool = MCPTool(SERVER, [_tool()])
        assert "acme_mcp" in tool.capability("search_tickets").description


class TestTheCredentialIsSentAndNeverLeaked:
    async def test_the_credential_becomes_a_bearer_token(
        self, tenant: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tool = MCPTool(SERVER, [_tool()])
        seen: list[httpx.Request] = []
        _stub(monkeypatch, tool, _rpc({"structuredContent": {}}), seen=seen)
        await tool.capability("search_tickets").handler(_ctx(tenant), query="x")

        assert seen[0].headers["authorization"] == "Bearer mcp-token"
        # Both are required by the Streamable HTTP transport; a server may answer with either.
        assert "text/event-stream" in seen[0].headers["accept"]

    async def test_a_server_with_no_credential_still_works(
        self, tenant: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not every MCP server is authenticated — a local one usually is not — and requiring a
        secret that does not exist would block the simplest case."""
        tool = MCPTool(SERVER, [_tool()])
        seen: list[httpx.Request] = []
        _stub(monkeypatch, tool, _rpc({"structuredContent": {}}), seen=seen)
        ctx = ToolContext(tenant=tenant, credential=None)  # type: ignore[arg-type]
        await tool.capability("search_tickets").handler(ctx, query="x")
        assert "authorization" not in seen[0].headers


class TestDiscovery:
    async def test_it_returns_the_raw_descriptors(self) -> None:
        """Raw rather than filtered: `admissible` decides admission, and keeping discovery honest
        means an operator can see what a server offered *and* what was refused."""

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_rpc({"tools": [_tool(), _tool("w", annotations={})]}))

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_handler), base_url=SERVER.url
        ) as client:
            found = await discover(SERVER, credential=None, client=client)
        assert [t["name"] for t in found] == ["search_tickets", "w"]

    async def test_a_rejected_list_call_raises(self) -> None:
        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_rpc(error={"message": "unauthorized"}))

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_handler), base_url=SERVER.url
        ) as client:
            with pytest.raises(UpstreamError, match="unauthorized"):
                await discover(SERVER, credential=None, client=client)


def _stub(
    monkeypatch: pytest.MonkeyPatch,
    tool: MCPTool,
    body: dict[str, Any],
    *,
    seen: list[httpx.Request] | None = None,
) -> None:
    """Replace the tool's client factory, so its real headers and request construction run."""
    original = tool._client

    def _factory(ctx: ToolContext) -> httpx.AsyncClient:
        def _handler(request: httpx.Request) -> httpx.Response:
            if seen is not None:
                seen.append(request)
            return httpx.Response(200, json=body)

        real = original(ctx)
        return httpx.AsyncClient(
            transport=httpx.MockTransport(_handler),
            base_url=real.base_url,
            headers=real.headers,
        )

    monkeypatch.setattr(tool, "_client", _factory)
