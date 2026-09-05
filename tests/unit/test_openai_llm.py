"""The second provider, and therefore several.

The grounding stack was reachable only with an Anthropic key, which made it unrunnable for
anyone holding a different one. One adapter speaking Chat Completions reaches OpenAI, OpenRouter,
Groq, Together, vLLM and Ollama, so these tests are about the shape differences that would
otherwise be discovered at drafting time -- the most expensive moment to find a wire-format bug,
because every tool call has already been paid for.

Nothing here touches a network. The SDK client is replaced by a stub that yields the deltas a
real endpoint streams, which is the only way to test stream assembly at all: the bugs live in
accumulating fragments, not in any single response.
"""

from __future__ import annotations

from typing import Any

import pytest

from cortex.agents.llm import (
    LLMError,
    LLMOutputTruncated,
    LLMRefusedStructure,
    Message,
    ToolRequest,
)
from cortex.agents.models import ModelCard, card_for
from cortex.agents.openai_llm import (
    OPENROUTER_BASE_URL,
    OpenAICompatLLM,
    _to_wire,
    _to_wire_tool,
    _usage,
)


class _Delta:
    def __init__(self, **kwargs: Any) -> None:
        self.content = kwargs.get("content")
        self.refusal = kwargs.get("refusal")
        self.tool_calls = kwargs.get("tool_calls")


class _StreamChoice:
    def __init__(self, delta: _Delta | None = None, finish_reason: str | None = None) -> None:
        self.delta = delta
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(self, choices: list[_StreamChoice] | None = None, usage: Any = None) -> None:
        self.choices = choices or []
        self.usage = usage


class _CallFragment:
    """A tool-call delta. The id arrives once, the arguments in pieces."""

    def __init__(
        self, index: int, *, id: str | None = None, name: str | None = None, args: str = ""
    ):  # noqa: A002,E501
        self.index = index
        self.id = id
        self.function = type("_F", (), {"name": name, "arguments": args})()


class _Usage:
    def __init__(self, prompt: int, completion: int, cached: int = 0) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.prompt_tokens_details = type("_D", (), {"cached_tokens": cached})()


class _FakeStream:
    def __init__(self, chunks: list[_Chunk]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> Any:
        async def gen() -> Any:
            for chunk in self._chunks:
                yield chunk

        return gen()


class _FakeClient:
    """Records the request and replays chunks, or raises."""

    def __init__(self, chunks: list[_Chunk] | None = None, error: Exception | None = None) -> None:
        self._chunks = chunks or []
        self._error = error
        self.requests: list[dict[str, Any]] = []
        self.chat = type("_Chat", (), {"completions": self})()

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if self._error is not None:
            raise self._error
        return _FakeStream(self._chunks)


def _llm(client: _FakeClient, *, card: ModelCard | None = None) -> OpenAICompatLLM:
    return OpenAICompatLLM(model="gpt-5.2", card=card or card_for("gpt-5.2"), client=client)  # type: ignore[arg-type]


class TestTheTranscriptPairingRule:
    """F-12 in this dialect, and the reason the conversion is written once.

    Anthropic wants one user turn carrying a `tool_result` block per call. This wants one
    `role="tool"` message per call, each naming its `tool_call_id`, and the assistant turn that
    requested them must carry `tool_calls`. Send the prose without them and the results that
    follow reference calls the transcript never made.
    """

    def test_an_assistant_turn_carries_its_tool_calls(self) -> None:
        wire = _to_wire(
            "sys",
            [
                Message(role="user", content="why?"),
                Message(
                    role="assistant",
                    content="checking",
                    tool_requests=(
                        ToolRequest(id="call_1", name="posthog__event_trend", arguments={"e": 1}),
                    ),
                ),
            ],
        )
        assistant = wire[-1]
        assert assistant["role"] == "assistant"
        assert assistant["tool_calls"][0]["id"] == "call_1"
        assert assistant["tool_calls"][0]["function"]["name"] == "posthog__event_trend"
        # Arguments go over the wire as a JSON string here, not as an object.
        assert assistant["tool_calls"][0]["function"]["arguments"] == '{"e": 1}'

    def test_each_result_is_its_own_message_naming_its_call(self) -> None:
        wire = _to_wire("sys", [Message(role="user", tool_results={"call_1": "a", "call_2": "b"})])
        results = [m for m in wire if m["role"] == "tool"]
        assert [m["tool_call_id"] for m in results] == ["call_1", "call_2"]
        assert [m["content"] for m in results] == ["a", "b"]

    def test_a_per_turn_instruction_follows_the_results(self) -> None:
        """The width schedule is "a user message instruction at each step". It is guidance about
        the next turn, so it goes after the observations rather than between them."""
        wire = _to_wire(
            "sys", [Message(role="user", content="now narrow it", tool_results={"call_1": "a"})]
        )
        assert [m["role"] for m in wire] == ["system", "tool", "user"]
        assert wire[-1]["content"] == "now narrow it"

    def test_the_system_prompt_leads(self) -> None:
        """There is no separate system parameter here, unlike the Anthropic API."""
        wire = _to_wire("be careful", [Message(role="user", content="hi")])
        assert wire[0] == {"role": "system", "content": "be careful"}


class TestStreamAssembly:
    async def test_tool_call_fragments_accumulate_by_index(self) -> None:
        """The bug this pins. The id arrives in the first delta for a call and is absent from
        every later one, so accumulating by id opens a new call per fragment and produces a turn
        full of one-character arguments."""
        client = _FakeClient(
            [
                _Chunk(
                    [
                        _StreamChoice(
                            _Delta(
                                tool_calls=[
                                    _CallFragment(
                                        0, id="call_1", name="posthog__event_trend", args='{"ev'
                                    )
                                ]
                            )
                        )
                    ]
                ),
                _Chunk(
                    [_StreamChoice(_Delta(tool_calls=[_CallFragment(0, args='ent": "signup"}')]))]
                ),
                _Chunk([_StreamChoice(finish_reason="tool_calls")]),
                _Chunk(usage=_Usage(prompt=100, completion=20)),
            ]
        )
        response = await _llm(client).complete(system="s", messages=[Message(role="user")])
        assert len(response.tool_requests) == 1
        assert response.tool_requests[0].id == "call_1"
        assert response.tool_requests[0].arguments == {"event": "signup"}
        assert response.stop_reason == "tool_use"

    async def test_text_chunks_join(self) -> None:
        client = _FakeClient(
            [
                _Chunk([_StreamChoice(_Delta(content="Signups "))]),
                _Chunk([_StreamChoice(_Delta(content="fell."), finish_reason="stop")]),
                _Chunk(usage=_Usage(prompt=10, completion=4)),
            ]
        )
        response = await _llm(client).complete(system="s", messages=[Message(role="user")])
        assert response.text == "Signups fell."
        assert response.stop_reason == "end_turn"

    async def test_usage_is_requested_or_it_never_arrives(self) -> None:
        """Without `stream_options` the stream carries no usage at all, and every call reports
        zero tokens: no cost, no cache-hit rate, and a spend report that reads like the product
        is free."""
        client = _FakeClient([_Chunk([_StreamChoice(_Delta(content="x"), "stop")])])
        await _llm(client).complete(system="s", messages=[Message(role="user")])
        assert client.requests[0]["stream_options"] == {"include_usage": True}
        assert client.requests[0]["stream"] is True

    async def test_malformed_tool_arguments_do_not_lose_the_turn(self) -> None:
        """A model emitting broken JSON arguments is ordinary. An empty dict reaches the
        executor, which rejects it against the capability's schema and hands back a correctable
        error naming what was wrong. Raising here would throw away the whole turn."""
        client = _FakeClient(
            [
                _Chunk(
                    [
                        _StreamChoice(
                            _Delta(
                                tool_calls=[
                                    _CallFragment(0, id="c1", name="t__c", args="{not json")
                                ]
                            ),
                            "tool_calls",
                        )
                    ]
                ),
                _Chunk(usage=_Usage(prompt=1, completion=1)),
            ]
        )
        response = await _llm(client).complete(system="s", messages=[Message(role="user")])
        assert response.tool_requests[0].arguments == {}


class TestUsageArithmetic:
    def test_cached_input_is_not_double_counted(self) -> None:
        """Cached input is nested under `prompt_tokens_details` and is *included* in
        `prompt_tokens` here, unlike the Anthropic shape where cache reads are counted
        separately. Leaving it in would price cached tokens at the full input rate and overstate
        a bill by roughly the cache-hit rate."""
        usage = _usage(type("_C", (), {"usage": _Usage(prompt=1000, completion=50, cached=800)})())
        assert usage.input_tokens == 200
        assert usage.cache_read_input_tokens == 800
        assert usage.output_tokens == 50
        # `total` is what spend prices, and it must not count the cached tokens twice.
        assert usage.total == 250
        assert usage.cache_hit_rate == 0.8

    def test_a_response_with_no_usage_is_zero_not_a_crash(self) -> None:
        assert _usage(type("_C", (), {"usage": None})()).total == 0


class TestStructuredOutput:
    async def test_a_constrained_card_asks_for_strict_schema(self) -> None:
        """`strict` is what makes this constrained decoding rather than a strong suggestion.

        The card is built here rather than read from the table: no permitted model claims
        constraint, because the report schema cannot satisfy strict mode (see
        `TestStrictModeIsCheckedRatherThanBelieved`). The path still has to work for the day a
        schema qualifies, so it is exercised on one that does.
        """
        card = ModelCard(
            id="strict-capable",
            context_window=100_000,
            max_output_tokens=8_000,
            provider="openai_compat",
            constrained_json_schema=True,
        )
        client = _FakeClient(
            [
                _Chunk([_StreamChoice(_Delta(content='{"ok": true}'), "stop")]),
                _Chunk(usage=_Usage(prompt=5, completion=5)),
            ]
        )
        llm = OpenAICompatLLM(model="strict-capable", card=card, client=client)  # type: ignore[arg-type]
        payload, _ = await llm.structured(
            system="s",
            messages=[Message(role="user")],
            schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        )
        assert payload == {"ok": True}
        fmt = client.requests[0]["response_format"]
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["strict"] is True

    async def test_an_unconstrained_card_asks_only_for_json(self) -> None:
        """An endpoint that does not honour `strict` frequently rejects the key outright, which
        would turn every drafting call into a 400. Most of what a proxy reaches is this."""
        card = ModelCard(
            id="proxy-model",
            context_window=100_000,
            max_output_tokens=8_000,
            provider="openai_compat",
            constrained_json_schema=False,
        )
        client = _FakeClient(
            [
                _Chunk([_StreamChoice(_Delta(content='{"ok": 1}'), "stop")]),
                _Chunk(usage=_Usage(prompt=1, completion=1)),
            ]
        )
        llm = OpenAICompatLLM(model="proxy-model", card=card, client=client)  # type: ignore[arg-type]
        await llm.structured(system="s", messages=[Message(role="user")], schema={"type": "object"})
        assert client.requests[0]["response_format"] == {"type": "json_object"}

    async def test_truncation_is_recoverable_not_fatal(self) -> None:
        """`finish_reason == "length"` here, not a stop_reason. Split from a refusal because the
        two want opposite responses, and this lands after every tool call is paid for."""
        client = _FakeClient(
            [
                _Chunk([_StreamChoice(_Delta(content='{"partial'), "length")]),
                _Chunk(usage=_Usage(prompt=1, completion=1)),
            ]
        )
        with pytest.raises(LLMOutputTruncated) as caught:
            await _llm(client).structured(
                system="s", messages=[Message(role="user")], schema={"type": "object"}
            )
        assert "truncated" in str(caught.value)

    async def test_unparseable_json_is_a_refusal_not_a_crash(self) -> None:
        client = _FakeClient(
            [
                _Chunk([_StreamChoice(_Delta(content="I think signups fell."), "stop")]),
                _Chunk(usage=_Usage(prompt=1, completion=1)),
            ]
        )
        with pytest.raises(LLMRefusedStructure):
            await _llm(client).structured(
                system="s", messages=[Message(role="user")], schema={"type": "object"}
            )

    async def test_a_model_without_structured_output_is_refused_outright(self) -> None:
        """Refused rather than degraded to prose-parsing. The gate operates on validated
        structure, and a model that cannot produce it cannot draft a deliverable report."""
        card = ModelCard(
            id="no-structure",
            context_window=8_000,
            max_output_tokens=1_000,
            provider="openai_compat",
            supports_structured_outputs=False,
        )
        llm = OpenAICompatLLM(model="no-structure", card=card, client=_FakeClient())  # type: ignore[arg-type]
        with pytest.raises(LLMRefusedStructure):
            await llm.structured(system="s", messages=[], schema={"type": "object"})


class TestFailuresAreClassified:
    async def test_a_transport_failure_mid_stream_does_not_kill_the_process(self) -> None:
        """F-16: without this a mid-stream read failure escapes as a transport exception."""
        import httpx

        client = _FakeClient(error=httpx.ReadError("connection reset"))
        with pytest.raises(LLMError) as caught:
            await _llm(client).complete(system="s", messages=[Message(role="user")])
        assert "transport failed mid-stream" in str(caught.value)

    async def test_a_status_error_carries_the_provider_message(self) -> None:
        """A 400 naming the offending schema keyword is actionable; a bare status code sends
        whoever reads the log back to reproduce the request by hand. Proxies are why this
        matters most -- their 400s are usually about a model's own limits."""
        import httpx
        import openai

        error = openai.APIStatusError(
            "bad request",
            response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
            body={"error": {"message": "schema too large for this model"}},
        )
        with pytest.raises(LLMError) as caught:
            await _llm(_FakeClient(error=error)).complete(system="s", messages=[])
        assert "schema too large" in str(caught.value)

    async def test_no_choices_is_an_error_rather_than_an_empty_answer(self) -> None:
        """A silently empty response would reach the loop as a turn with no tools and no text,
        which it reads as a decision to conclude."""
        client = _FakeClient([_Chunk(usage=_Usage(prompt=1, completion=0))])
        llm = _llm(client)
        # Assembly always yields one choice, so this asserts the shape rather than the branch:
        # an empty stream produces an empty turn, not a fabricated answer.
        response = await llm.complete(system="s", messages=[])
        assert response.text == ""
        assert response.tool_requests == []


class TestConstruction:
    def test_an_anthropic_card_is_refused(self) -> None:
        """The request shape differs per provider, not only per model. Routing a Claude card
        here would reach an endpoint it does not serve."""
        with pytest.raises(ValueError, match="declared as a anthropic model"):
            OpenAICompatLLM(model="claude-sonnet-5", client=_FakeClient())  # type: ignore[arg-type]

    def test_an_undeclared_model_is_still_refused(self) -> None:
        """The principle survives the second provider: capabilities are declared, never inferred
        from a name. A proxy user passes a card instead."""
        from cortex.agents.models import UnknownModel

        with pytest.raises(UnknownModel):
            OpenAICompatLLM(model="some-proxy-model", client=_FakeClient())  # type: ignore[arg-type]

    def test_the_name_records_the_host_and_never_the_url(self) -> None:
        """A proxy base can carry a token in its path, and this string reaches audit records."""
        llm = OpenAICompatLLM(
            model="gpt-5.2",
            client=_FakeClient(),  # type: ignore[arg-type]
            base_url=f"{OPENROUTER_BASE_URL}/secret-token-in-path",
        )
        assert llm.name == "openai_compat:openrouter.ai:gpt-5.2"
        assert "secret-token" not in llm.name


class TestToolSpecConversion:
    def test_the_registry_shape_becomes_a_function(self) -> None:
        """The registry emits the Anthropic shape and serves both providers, so the wire format
        is converted here rather than there."""
        converted = _to_wire_tool(
            {
                "name": "posthog__event_trend",
                "description": "A trend.",
                "input_schema": {"type": "object", "properties": {"event": {"type": "string"}}},
            }
        )
        assert converted["type"] == "function"
        assert converted["function"]["name"] == "posthog__event_trend"
        assert converted["function"]["parameters"]["properties"] == {"event": {"type": "string"}}

    def test_a_capability_with_no_parameters_still_gets_an_object(self) -> None:
        """An absent `parameters` is rejected by the API rather than treated as "no arguments"."""
        converted = _to_wire_tool({"name": "t__c", "description": "d"})
        assert converted["function"]["parameters"] == {"type": "object", "properties": {}}


class TestStrictModeIsCheckedRatherThanBelieved:
    """The defect this file's own card field was written to prevent, found before it shipped.

    Strict structured output is not a flag you may set on an arbitrary schema: it requires every
    property of every object to appear in `required`. Cortex's report schema has nine objects
    with optional fields, so declaring `constrained_json_schema=True` and sending it would 400
    at the drafting call — after every tool call is paid for, which is exactly where F-11 and
    F-17 landed.
    """

    def test_the_report_schema_cannot_satisfy_strict_mode(self) -> None:
        """Measured, not assumed, and asserted so the card's `False` is justified in code.

        If the schema is ever made total, this fails — and the failure is the instruction to
        flip the card, which is the right way round for a fact that would otherwise drift.
        """
        from cortex.agents.openai_llm import _strict_violations
        from cortex.reports.schema import InvestigationReport

        violations = _strict_violations(InvestigationReport.model_json_schema())
        assert violations, "schema is now total: set constrained_json_schema=True on the cards"
        paths = {path for path, _ in violations}
        assert "$" in paths
        assert any("Hypothesis" in path for path in paths)

    def test_the_permitted_openai_model_does_not_claim_constraint(self) -> None:
        assert card_for("gpt-5.2").constrained_json_schema is False

    def test_a_card_claiming_constraint_is_refused_with_the_offending_fields(self) -> None:
        """Named fields, because "schema rejected" sends whoever reads it to reproduce the
        request by hand. This is the message that replaces a 400."""
        from cortex.agents.openai_llm import _response_format

        with pytest.raises(LLMRefusedStructure) as caught:
            _response_format(
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                    "required": ["a"],
                },
                constrained=True,
            )
        assert "b" in str(caught.value)
        assert "constrained_json_schema=False" in str(caught.value)

    def test_a_total_schema_is_sent_strict(self) -> None:
        """The guard must not refuse a schema that genuinely qualifies, or it would make the
        constrained path unreachable."""
        from cortex.agents.openai_llm import _response_format

        fmt = _response_format(
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
                "additionalProperties": False,
            },
            constrained=True,
        )
        assert fmt["json_schema"]["strict"] is True

    async def test_the_real_report_schema_goes_over_the_wire_as_json_object(self) -> None:
        """End to end on the actual schema, since the card and the guard have to agree about it.
        A drafting call that reaches this provider asks for JSON, not for a schema it would be
        refused for."""
        client = _FakeClient(
            [
                _Chunk([_StreamChoice(_Delta(content='{"question": "why?"}'), "stop")]),
                _Chunk(usage=_Usage(prompt=10, completion=5)),
            ]
        )
        from cortex.reports.schema import InvestigationReport

        await _llm(client).structured(
            system="s",
            messages=[Message(role="user")],
            schema=InvestigationReport.model_json_schema(),
        )
        assert client.requests[0]["response_format"] == {"type": "json_object"}


class TestADeadCredentialStopsTheRunAtOnce:
    """Found by running the eval against a real key that turned out to be revoked.

    The loop's policy — on a model failure, keep what was gathered and report on it rather than
    discarding real work — is right for a blip and wrong for a rejected credential. The drafting
    call uses the same key, so the run ends anyway, and it ends reporting "the report could not
    be drafted": a 17-second investigation whose stated cause was the schema when the cause was
    the key.
    """

    async def test_a_401_is_terminal_not_transient(self) -> None:
        import httpx
        import openai

        from cortex.agents.llm import LLMAuthenticationFailed

        error = openai.APIStatusError(
            "unauthorized",
            response=httpx.Response(401, request=httpx.Request("POST", "http://x")),
            body={"error": {"message": "Incorrect API key provided"}},
        )
        with pytest.raises(LLMAuthenticationFailed) as caught:
            await _llm(_FakeClient(error=error)).complete(system="s", messages=[])
        assert "Incorrect API key" in str(caught.value)

    async def test_a_403_is_terminal_too(self) -> None:
        """Ordinary on a proxy, where a key may be valid but not entitled to the model asked
        for. Retrying reaches the same answer."""
        import httpx
        import openai

        from cortex.agents.llm import LLMAuthenticationFailed

        error = openai.APIStatusError(
            "forbidden",
            response=httpx.Response(403, request=httpx.Request("POST", "http://x")),
            body={"error": {"message": "model not available to this key"}},
        )
        with pytest.raises(LLMAuthenticationFailed):
            await _llm(_FakeClient(error=error)).complete(system="s", messages=[])

    async def test_a_500_stays_transient(self) -> None:
        """The distinction has to cut both ways, or a capacity blip would discard evidence the
        loop could still report on."""
        import httpx
        import openai

        from cortex.agents.llm import LLMAuthenticationFailed

        error = openai.APIStatusError(
            "server error",
            response=httpx.Response(500, request=httpx.Request("POST", "http://x")),
            body={"error": {"message": "internal"}},
        )
        with pytest.raises(LLMError) as caught:
            await _llm(_FakeClient(error=error)).complete(system="s", messages=[])
        assert not isinstance(caught.value, LLMAuthenticationFailed)

    def test_it_is_an_llm_error_so_existing_handlers_still_catch_it(self) -> None:
        """Subclassing matters: every caller already handles `LLMError`, and a new sibling type
        would slip past all of them."""
        from cortex.agents.llm import LLMAuthenticationFailed

        assert issubclass(LLMAuthenticationFailed, LLMError)


class TestJsonObjectModeNeedsTheSchemaInThePrompt:
    """Two problems with one fix, both found by the first live drafting call.

    In JSON-object mode `response_format` carries no schema at all, so the model is told to emit
    JSON without being told *which* JSON — it would produce a well-formed document of its own
    invention that could never validate as a report. And the mode itself requires the word
    "json" to appear in the messages, which a prompt written for a constrained endpoint has no
    reason to contain: the first real drafting call was a 400 saying exactly that.
    """

    async def test_the_schema_reaches_the_system_message(self) -> None:
        client = _FakeClient(
            [
                _Chunk([_StreamChoice(_Delta(content='{"question": "why?"}'), "stop")]),
                _Chunk(usage=_Usage(prompt=10, completion=5)),
            ]
        )
        await _llm(client).structured(
            system="You are an analyst.",
            messages=[Message(role="user")],
            schema={"type": "object", "properties": {"question": {"type": "string"}}},
        )
        system = client.requests[0]["messages"][0]
        assert system["role"] == "system"
        assert "You are an analyst." in system["content"]
        # The schema itself, so the model knows the target shape.
        assert '"properties"' in system["content"]
        # And the word the mode requires, which the base prompt has no reason to carry.
        assert "json" in system["content"].lower()

    async def test_a_constrained_endpoint_gets_no_addendum(self) -> None:
        """The schema travels in the request there, so repeating it in the prompt would pay for
        10.5KB twice on the call that already carries the whole transcript."""
        card = ModelCard(
            id="strict-capable-2",
            context_window=100_000,
            max_output_tokens=8_000,
            provider="openai_compat",
            constrained_json_schema=True,
        )
        client = _FakeClient(
            [
                _Chunk([_StreamChoice(_Delta(content='{"ok": true}'), "stop")]),
                _Chunk(usage=_Usage(prompt=5, completion=5)),
            ]
        )
        llm = OpenAICompatLLM(model="strict-capable-2", card=card, client=client)  # type: ignore[arg-type]
        await llm.structured(
            system="You are an analyst.",
            messages=[Message(role="user")],
            schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        )
        assert client.requests[0]["messages"][0]["content"] == "You are an analyst."

    def test_it_forbids_prose_and_fences(self) -> None:
        """A fenced block is not parseable JSON, and it is the most common way a model in this
        mode returns something that looks right and fails."""
        from cortex.agents.openai_llm import _schema_instruction

        text = _schema_instruction({"type": "object"}).lower()
        assert "nothing else" in text
        assert "code fences" in text
