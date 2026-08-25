"""Anthropic provider.

Every test runs against a mock transport, so the SDK's own request construction and
response parsing are exercised without an API key or a network call. That matters
here more than usual: the failure modes this file guards are *API contract*
failures, and a hand-rolled fake of the SDK would encode my assumptions rather
than the SDK's behaviour.

The properties pinned are the ones that break loudly in production and silently in
a naive test: sampling parameters must not be sent, thinking must be adaptive, and
a refusal arrives as HTTP 200 with possibly-empty content.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx
import pytest

from cortex.agents.anthropic_llm import (
    ALLOWED_MODELS,
    DEFAULT_MODEL,
    FAST_MODEL,
    AnthropicLLM,
)
from cortex.agents.llm import LLMError, LLMRefusedStructure, Message


def _sse(events: list[dict[str, Any]]) -> bytes:
    """Render a Messages API streaming response."""
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def _stream(
    *,
    content: list[dict[str, Any]] | None = None,
    stop_reason: str = "end_turn",
    stop_details: dict[str, Any] | None = None,
    input_tokens: int = 120,
    output_tokens: int = 45,
) -> bytes:
    """A complete stream for one assistant turn."""
    blocks = content or [{"type": "text", "text": "Signups fell 18%."}]
    events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_01",
                "type": "message",
                "role": "assistant",
                "model": DEFAULT_MODEL,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        }
    ]
    for index, block in enumerate(blocks):
        if block["type"] == "text":
            events += [
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": block["text"]},
                },
                {"type": "content_block_stop", "index": index},
            ]
        else:  # tool_use
            events += [
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": block["id"],
                        "name": block["name"],
                        "input": {},
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block["input"]),
                    },
                },
                {"type": "content_block_stop", "index": index},
            ]

    delta: dict[str, Any] = {"stop_reason": stop_reason, "stop_sequence": None}
    if stop_details is not None:
        delta["stop_details"] = stop_details
    events += [
        {
            "type": "message_delta",
            "delta": delta,
            "usage": {"output_tokens": output_tokens},
        },
        {"type": "message_stop"},
    ]
    return _sse(events)


class _Recorder:
    """Captures every outgoing request body so the wire shape can be asserted."""

    def __init__(self, *responses: bytes | httpx.Response) -> None:
        self.bodies: list[dict[str, Any]] = []
        self._responses = list(responses)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        nxt = self._responses.pop(0) if self._responses else _stream()
        if isinstance(nxt, httpx.Response):
            return nxt
        return httpx.Response(200, content=nxt, headers={"content-type": "text/event-stream"})


def _llm(recorder: _Recorder, **kwargs: Any) -> AnthropicLLM:
    client = anthropic.AsyncAnthropic(
        api_key="test-key-not-real",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(recorder)),
    )
    return AnthropicLLM(client=client, **kwargs)


#: A schema the API would actually accept. `{}` is not one: structured outputs
#: require a type, so passing it here would have tested a path the provider can
#: never take.
_MINIMAL_SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


class TestModelIds:
    def test_cortex_runs_one_mid_tier_model(self) -> None:
        """A stale or date-suffixed id 404s. This is the verified current one.

        One model, and a mid-tier one, is a cost decision: an investigation spends
        100–170k tokens and the drafting call alone can be 30k of output, which on an
        Opus-tier model is roughly $0.80 per investigation before the verifier's
        per-claim calls.
        """
        assert DEFAULT_MODEL == "claude-sonnet-5"
        assert FAST_MODEL == DEFAULT_MODEL

    def test_another_model_is_refused_at_construction(self) -> None:
        """The restriction has to bite where it is cheap to notice.

        A default that callers may override is not a ceiling — one call site passes
        something more capable "just for drafting" and the bill returns. Refused at
        construction rather than discovered on an invoice.
        """
        with pytest.raises(ValueError, match="not permitted"):
            _llm(_Recorder(), model="claude-opus-5")

    def test_the_allowlist_and_the_default_agree(self) -> None:
        """A default outside the allowlist would raise on every construction."""
        assert DEFAULT_MODEL in ALLOWED_MODELS

    def test_no_date_suffix(self) -> None:
        for model in (DEFAULT_MODEL, FAST_MODEL):
            assert not any(
                ch.isdigit() and len(part) == 8 for part in model.split("-") for ch in part
            ), model

    def test_name_reports_provider_and_model(self) -> None:
        llm = _llm(_Recorder(), model="claude-sonnet-5")
        assert llm.name == "anthropic:claude-sonnet-5"


class TestRequestShape:
    async def test_sampling_parameters_are_never_sent(self) -> None:
        """temperature / top_p / top_k are rejected with a 400 on these models."""
        recorder = _Recorder()
        await _llm(recorder).complete(system="s", messages=[Message(role="user", content="q")])
        body = recorder.bodies[0]
        for banned in ("temperature", "top_p", "top_k"):
            assert banned not in body, banned

    async def test_thinking_is_adaptive_not_budgeted(self) -> None:
        """The budget_tokens form returns a 400; depth is set by effort instead."""
        recorder = _Recorder()
        await _llm(recorder).complete(system="s", messages=[Message(role="user", content="q")])
        body = recorder.bodies[0]
        assert body["thinking"] == {"type": "adaptive"}
        assert "budget_tokens" not in json.dumps(body)

    async def test_effort_is_inside_output_config(self) -> None:
        recorder = _Recorder()
        await _llm(recorder, effort="max").complete(
            system="s", messages=[Message(role="user", content="q")]
        )
        assert recorder.bodies[0]["output_config"]["effort"] == "max"

    async def test_tools_are_forwarded_when_given(self) -> None:
        recorder = _Recorder()
        spec = {
            "name": "ga4__get_sessions",
            "description": "d",
            "input_schema": {"type": "object", "additionalProperties": False, "properties": {}},
        }
        await _llm(recorder).complete(
            system="s", messages=[Message(role="user", content="q")], tools=[spec]
        )
        assert [t["name"] for t in recorder.bodies[0]["tools"]] == ["ga4__get_sessions"]

    async def test_tools_are_omitted_when_absent(self) -> None:
        """An empty tools array is not the same as no tools."""
        recorder = _Recorder()
        await _llm(recorder).complete(system="s", messages=[Message(role="user", content="q")])
        assert "tools" not in recorder.bodies[0]

    async def test_structured_call_sends_the_schema(self) -> None:
        recorder = _Recorder(_stream(content=[{"type": "text", "text": '{"ok": true}'}]))
        schema = {"type": "object", "additionalProperties": False, "properties": {}}
        payload, usage = await _llm(recorder).structured(
            system="s", messages=[Message(role="user", content="q")], schema=schema
        )
        body = recorder.bodies[0]
        assert body["output_config"]["format"] == {"type": "json_schema", "schema": schema}
        assert payload == {"ok": True}
        assert usage.input_tokens == 120


class TestToolResultEncoding:
    async def test_tool_results_become_one_user_turn(self) -> None:
        """Several results in one turn, not one turn each — splitting them teaches
        the model to stop making parallel calls."""
        recorder = _Recorder()
        await _llm(recorder).complete(
            system="s",
            messages=[
                Message(role="user", content="q"),
                Message(role="assistant", content="calling tools"),
                Message(role="user", tool_results={"a": "obs A", "b": "obs B"}),
            ],
        )
        messages = recorder.bodies[0]["messages"]
        assert len(messages) == 3
        final = messages[-1]
        assert final["role"] == "user"
        assert [b["type"] for b in final["content"]] == ["tool_result", "tool_result"]
        assert {b["tool_use_id"] for b in final["content"]} == {"a", "b"}

    async def test_plain_messages_pass_through(self) -> None:
        recorder = _Recorder()
        await _llm(recorder).complete(
            system="sys", messages=[Message(role="user", content="hello")]
        )
        body = recorder.bodies[0]
        assert body["system"] == "sys"
        assert body["messages"] == [{"role": "user", "content": "hello"}]


class TestResponseParsing:
    async def test_text_is_collected(self) -> None:
        recorder = _Recorder(_stream(content=[{"type": "text", "text": "Signups fell 18%."}]))
        response = await _llm(recorder).complete(
            system="s", messages=[Message(role="user", content="q")]
        )
        assert response.text == "Signups fell 18%."
        assert not response.wants_tools
        assert response.stop_reason == "end_turn"

    async def test_tool_calls_are_parsed_with_arguments(self) -> None:
        recorder = _Recorder(
            _stream(
                content=[
                    {"type": "text", "text": "Checking."},
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "ga4__get_sessions",
                        "input": {"start_date": "2026-07-01", "end_date": "2026-07-07"},
                    },
                ],
                stop_reason="tool_use",
            )
        )
        response = await _llm(recorder).complete(
            system="s", messages=[Message(role="user", content="q")]
        )
        assert response.wants_tools
        request = response.tool_requests[0]
        assert request.id == "toolu_01"
        assert request.name == "ga4__get_sessions"
        # Parsed by the SDK, never re-parsed from a string: escaping differs across
        # models and raw matching breaks.
        assert request.arguments["start_date"] == "2026-07-01"

    async def test_usage_is_reported(self) -> None:
        """Without this an investigation cannot be stopped on budget."""
        recorder = _Recorder(_stream(input_tokens=1234, output_tokens=567))
        response = await _llm(recorder).complete(
            system="s", messages=[Message(role="user", content="q")]
        )
        assert response.usage.input_tokens == 1234
        assert response.usage.output_tokens == 567
        assert response.usage.total == 1801


class TestRefusalHandling:
    async def test_refusal_raises_rather_than_returning_empty(self) -> None:
        """A refusal is HTTP 200 with possibly-empty content. Code that read
        content[0] unconditionally would crash; code that returned it silently
        would present an incomplete investigation as complete."""
        recorder = _Recorder(
            _stream(
                content=[],
                stop_reason="refusal",
                stop_details={"type": "refusal", "category": "cyber"},
            )
        )
        with pytest.raises(LLMError, match="declined by a safety classifier"):
            await _llm(recorder).complete(system="s", messages=[Message(role="user", content="q")])

    async def test_refusal_category_is_reported_when_present(self) -> None:
        recorder = _Recorder(
            _stream(
                content=[],
                stop_reason="refusal",
                stop_details={"type": "refusal", "category": "cyber"},
            )
        )
        with pytest.raises(LLMError, match="category=cyber"):
            await _llm(recorder).complete(system="s", messages=[Message(role="user", content="q")])

    async def test_missing_stop_details_does_not_crash(self) -> None:
        """stop_details is informational and can be absent even on a refusal, so it
        is reported but never branched on."""
        recorder = _Recorder(_stream(content=[], stop_reason="refusal"))
        with pytest.raises(LLMError, match="unspecified"):
            await _llm(recorder).complete(system="s", messages=[Message(role="user", content="q")])

    async def test_structured_refusal_is_its_own_error(self) -> None:
        recorder = _Recorder(_stream(content=[], stop_reason="refusal"))
        with pytest.raises(LLMRefusedStructure):
            await _llm(recorder).structured(
                system="s", messages=[Message(role="user", content="q")], schema=_MINIMAL_SCHEMA
            )


class TestStructuredFailureModes:
    async def test_truncation_is_an_error_not_a_partial_parse(self) -> None:
        """Truncated JSON would fail to parse, and a partially-parsed report would
        silently lose findings."""
        recorder = _Recorder(
            _stream(
                content=[{"type": "text", "text": '{"executive_summary": ['}],
                stop_reason="max_tokens",
            )
        )
        with pytest.raises(LLMRefusedStructure, match="truncated"):
            await _llm(recorder).structured(
                system="s", messages=[Message(role="user", content="q")], schema=_MINIMAL_SCHEMA
            )

    async def test_non_json_output_is_an_error(self) -> None:
        recorder = _Recorder(_stream(content=[{"type": "text", "text": "I cannot do that."}]))
        with pytest.raises(LLMRefusedStructure, match="not valid JSON"):
            await _llm(recorder).structured(
                system="s", messages=[Message(role="user", content="q")], schema=_MINIMAL_SCHEMA
            )

    async def test_json_array_is_rejected(self) -> None:
        recorder = _Recorder(_stream(content=[{"type": "text", "text": "[1, 2, 3]"}]))
        with pytest.raises(LLMRefusedStructure, match="expected object"):
            await _llm(recorder).structured(
                system="s", messages=[Message(role="user", content="q")], schema=_MINIMAL_SCHEMA
            )


def _error(status: int) -> httpx.Response:
    return httpx.Response(status, json={"type": "error", "error": {"type": "x", "message": "m"}})


def _mid_stream_error(
    error_type: str = "overloaded_error", message: str = "Overloaded"
) -> httpx.Response:
    """A 200 response whose body carries an SSE `error` event.

    This is the real shape of a provider capacity failure, and the reason it needed
    handling of its own: the status line says 200, so the SDK's retry policy — which
    keys on the response status — never applies. It is only visible once the body is
    being consumed.
    """
    return httpx.Response(
        200,
        content=_sse([{"type": "error", "error": {"type": error_type, "message": message}}]),
        headers={"content-type": "text/event-stream"},
    )


class TestMidStreamProviderErrors:
    """A live run lost a completed 127-second investigation to one `Overloaded` at the
    drafting step. The status was 200, so nothing retried it."""

    async def test_a_capacity_error_inside_a_200_is_retried(self) -> None:
        recorder = _Recorder(_mid_stream_error(), _stream())
        response = await _llm(recorder, backoff=(0.0, 0.0)).complete(
            system="s", messages=[Message(role="user", content="q")]
        )
        assert response.text
        assert len(recorder.bodies) == 2

    async def test_drafting_gets_the_same_policy(self) -> None:
        """The call that can least afford to lose its work.

        Asserted separately from `complete` because the two paths were duplicated
        before, and a policy applied to only one of them is worse than none — it would
        look handled.
        """
        recorder = _Recorder(
            _mid_stream_error(), _stream(content=[{"type": "text", "text": '{"ok": true}'}])
        )
        payload, _ = await _llm(recorder, backoff=(0.0, 0.0)).structured(
            system="s", messages=[Message(role="user", content="q")], schema=_MINIMAL_SCHEMA
        )
        assert payload == {"ok": True}
        assert len(recorder.bodies) == 2

    async def test_it_gives_up_after_the_retry_budget(self) -> None:
        recorder = _Recorder(*[_mid_stream_error()] * 4)
        with pytest.raises(LLMError) as caught:
            await _llm(recorder, backoff=(0.0, 0.0)).complete(
                system="s", messages=[Message(role="user", content="q")]
            )
        assert len(recorder.bodies) == 3, "one attempt plus two retries"
        # The count is in the message so an exhausted retry reads differently from a
        # first-try rejection.
        assert "attempts: 3" in str(caught.value)
        assert "Overloaded" in str(caught.value)

    @pytest.mark.parametrize("error_type", ["invalid_request_error", "authentication_error"])
    async def test_a_request_error_inside_a_200_is_not_retried(self, error_type: str) -> None:
        """Retrying these spends the deadline to reach the same answer."""
        recorder = _Recorder(*[_mid_stream_error(error_type, "nope")] * 4)
        with pytest.raises(LLMError) as caught:
            await _llm(recorder, backoff=(0.0, 0.0)).complete(
                system="s", messages=[Message(role="user", content="q")]
            )
        assert len(recorder.bodies) == 1
        assert "attempts: 1" in str(caught.value)


class TestTransportErrors:
    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    async def test_non_retryable_errors_raise_immediately(self, status: int) -> None:
        """A bad request or a rejected key will not improve on retry."""
        recorder = _Recorder(_error(status))
        with pytest.raises(LLMError):
            await _llm(recorder).complete(system="s", messages=[Message(role="user", content="q")])
        assert len(recorder.bodies) == 1, "must not retry a client error"

    @pytest.mark.parametrize("status", [429, 500, 529])
    async def test_retryable_errors_are_retried_by_the_sdk(self, status: int) -> None:
        """The SDK retries these with backoff, so a transient rate limit or overload
        self-heals rather than failing an investigation. Relied on deliberately —
        but the retries consume the investigation's wall-clock budget, which is why
        the loop bounds time rather than only step count.
        """
        recorder = _Recorder(_error(status), _stream())
        response = await _llm(recorder).complete(
            system="s", messages=[Message(role="user", content="q")]
        )
        assert response.text
        assert len(recorder.bodies) == 2, "the SDK should have retried once"

    @pytest.mark.parametrize("status", [429, 500, 529])
    async def test_retryable_errors_raise_once_retries_are_exhausted(self, status: int) -> None:
        recorder = _Recorder(*[_error(status)] * 6)
        with pytest.raises(LLMError):
            await _llm(recorder).complete(system="s", messages=[Message(role="user", content="q")])

    async def test_error_message_does_not_leak_the_api_key(self) -> None:
        recorder = _Recorder(
            httpx.Response(401, json={"type": "error", "error": {"type": "auth", "message": "m"}})
        )
        with pytest.raises(LLMError) as exc:
            await _llm(recorder).complete(system="s", messages=[Message(role="user", content="q")])
        assert "test-key-not-real" not in str(exc.value)

    async def test_structured_transport_error_becomes_llm_error(self) -> None:
        recorder = _Recorder(
            httpx.Response(500, json={"type": "error", "error": {"type": "x", "message": "m"}})
        )
        with pytest.raises(LLMError):
            await _llm(recorder).structured(
                system="s", messages=[Message(role="user", content="q")], schema=_MINIMAL_SCHEMA
            )


class TestStreamingIsUsed:
    async def test_requests_stream(self) -> None:
        """Large max_tokens must stream or the request hits an HTTP timeout, and the
        report draft is big enough to need it."""
        recorder = _Recorder()
        await _llm(recorder).complete(system="s", messages=[Message(role="user", content="q")])
        assert recorder.bodies[0]["stream"] is True

    async def test_structured_requests_stream(self) -> None:
        recorder = _Recorder(_stream(content=[{"type": "text", "text": "{}"}]))
        await _llm(recorder).structured(
            system="s", messages=[Message(role="user", content="q")], schema=_MINIMAL_SCHEMA
        )
        assert recorder.bodies[0]["stream"] is True


class TestInterfaceConformance:
    def test_implements_the_llm_interface(self) -> None:
        from cortex.agents.llm import LLM

        assert issubclass(AnthropicLLM, LLM)

    async def test_is_substitutable_for_the_recorded_provider(self) -> None:
        """The loop and verifier depend on the interface, never the provider."""
        import inspect

        from cortex.agents.llm import LLM, RecordedLLM

        for method in ("complete", "structured"):
            real = inspect.signature(getattr(AnthropicLLM, method))
            recorded = inspect.signature(getattr(RecordedLLM, method))
            base = inspect.signature(getattr(LLM, method))
            assert set(real.parameters) == set(recorded.parameters) == set(base.parameters), method


class TestStructuredSchemaTransform:
    """Structured outputs accept a subset of JSON Schema.

    These tests exist because the failure mode is expensive and late: an
    unsupported keyword is a 400 raised while drafting the report, after the
    investigation has already spent its tokens gathering evidence.
    """

    def test_the_sdk_transform_is_where_we_expect_it(self) -> None:
        """Pinned deliberately: this is a private SDK path.

        Reimplementing the transform would mean tracking the API's supported subset
        by hand. Importing the maintained one is the better trade, provided an SDK
        upgrade that moves it fails here rather than in production.
        """
        from anthropic.lib._parse._transform import transform_schema

        assert callable(transform_schema)

    def test_unsupported_bounds_are_moved_into_the_description(self) -> None:
        """Dropped, not silently lost: the model is still told the bound."""
        from anthropic.lib._parse._transform import transform_schema

        result = transform_schema(
            {"type": "integer", "minimum": 1, "maximum": 5, "description": "Priority"}
        )
        assert "minimum" not in result
        assert "maximum" not in result
        assert "1" in result["description"] and "5" in result["description"]

    async def test_the_report_schema_goes_on_the_wire_transformed(self) -> None:
        """The real drafting schema, through the real call path."""
        from cortex.reports.schema import llm_report_schema

        recorder = _Recorder(_stream(content=[{"type": "text", "text": "{}"}]))
        await _llm(recorder).structured(
            system="s",
            messages=[Message(role="user", content="q")],
            schema=llm_report_schema(),
        )
        sent = recorder.bodies[0]["output_config"]["format"]["schema"]

        rejected = {
            "minimum",
            "maximum",
            "multipleOf",
            "minLength",
            "maxLength",
            "pattern",
            "maxItems",
            "prefixItems",
        }
        offenders: list[str] = []

        def walk(node: object, path: str) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in rejected:
                        offenders.append(f"{path}/{key}")
                    if key == "minItems" and value not in (0, 1):
                        offenders.append(f"{path}/minItems={value}")
                    walk(value, f"{path}/{key}")
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    walk(value, f"{path}[{index}]")

        walk(sent, "")
        assert offenders == [], offenders


class TestTranscriptPairing:
    """Every tool_result must pair with a tool_use in the preceding assistant turn.

    This was a real bug: the loop recorded only the assistant's prose, so the results
    it appended next were orphaned and the API rejected the transcript with
    "unexpected `tool_use_id` found in `tool_result` blocks". Nothing caught it,
    because the recorded provider used in tests does not validate pairing.
    """

    def test_assistant_tool_requests_are_replayed_as_blocks(self) -> None:
        from cortex.agents.anthropic_llm import _to_wire
        from cortex.agents.llm import ToolRequest

        wire = _to_wire(
            [
                Message(role="user", content="why did signups fall?"),
                Message(
                    role="assistant",
                    content="Checking sessions.",
                    tool_requests=(
                        ToolRequest(id="toolu_1", name="ga4__get_sessions", arguments={"days": 7}),
                    ),
                ),
                Message(role="user", tool_results={"toolu_1": '{"sessions": 10}'}),
            ]
        )

        assert [block["type"] for block in wire[1]["content"]] == ["text", "tool_use"]
        assert wire[1]["content"][1]["id"] == "toolu_1"
        assert wire[1]["content"][1]["input"] == {"days": 7}
        assert wire[2]["content"][0]["tool_use_id"] == "toolu_1"

    def test_every_result_id_has_a_request_in_the_previous_turn(self) -> None:
        """The invariant the API enforces, asserted on our own output."""
        from cortex.agents.anthropic_llm import _to_wire
        from cortex.agents.llm import ToolRequest

        requests = tuple(
            ToolRequest(id=f"toolu_{n}", name="ga4__get_sessions", arguments={}) for n in range(3)
        )
        wire = _to_wire(
            [
                Message(role="user", content="q"),
                Message(role="assistant", content="", tool_requests=requests),
                Message(role="user", tool_results={r.id: "{}" for r in requests}),
            ]
        )

        for index, turn in enumerate(wire):
            if not isinstance(turn["content"], list):
                continue
            result_ids = {
                block["tool_use_id"] for block in turn["content"] if block["type"] == "tool_result"
            }
            if not result_ids:
                continue
            previous = wire[index - 1]["content"]
            request_ids = {
                block["id"]
                for block in previous
                if isinstance(block, dict) and block["type"] == "tool_use"
            }
            assert result_ids <= request_ids, (result_ids, request_ids)

    def test_a_text_only_assistant_turn_stays_a_string(self) -> None:
        """No gratuitous reshaping: a turn with no tool calls is unchanged."""
        from cortex.agents.anthropic_llm import _to_wire

        wire = _to_wire([Message(role="assistant", content="done")])
        assert wire == [{"role": "assistant", "content": "done"}]


class _TruncatedStream(httpx.AsyncByteStream):
    """A response body that dies partway through, as a read timeout does."""

    def __init__(self, prefix: bytes, exc: Exception) -> None:
        self._prefix = prefix
        self._exc = exc

    async def __aiter__(self):  # type: ignore[no-untyped-def]
        yield self._prefix
        raise self._exc


class TestMidStreamFailures:
    """A transport failure while consuming the body must become an `LLMError`.

    This was a real defect. The SDK maps timeouts to `APITimeoutError` for the request
    phase only; one raised while iterating the stream arrives as `httpx.ReadTimeout`,
    which is not an `anthropic.APIError`. It escaped both handlers, took down the
    whole eval process, and destroyed the results of the two scenarios that had
    already completed.
    """

    async def test_a_read_timeout_mid_stream_becomes_an_llm_error(self) -> None:
        partial = _sse(
            [
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_01",
                        "type": "message",
                        "role": "assistant",
                        "model": DEFAULT_MODEL,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 10, "output_tokens": 0},
                    },
                }
            ]
        )
        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_TruncatedStream(partial, httpx.ReadTimeout("timed out")),
        )

        with pytest.raises(LLMError) as caught:
            await _llm(_Recorder(response)).complete(
                system="s", messages=[Message(role="user", content="q")]
            )
        # Reported as a timeout, not as a generic transport failure: the two call for
        # different responses, and only one is worth retrying at a different budget.
        assert "no response within" in str(caught.value)

    async def test_a_broken_connection_mid_stream_becomes_an_llm_error(self) -> None:
        partial = _sse(
            [
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_01",
                        "type": "message",
                        "role": "assistant",
                        "model": DEFAULT_MODEL,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 10, "output_tokens": 0},
                    },
                }
            ]
        )
        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_TruncatedStream(partial, httpx.ReadError("connection reset")),
        )

        with pytest.raises(LLMError) as caught:
            await _llm(_Recorder(response)).complete(
                system="s", messages=[Message(role="user", content="q")]
            )
        assert "mid-stream" in str(caught.value)

    async def test_the_deadline_is_ours_and_bounds_the_whole_turn(self) -> None:
        """A response that never arrives is bounded by us, not by the socket.

        httpx applies `read` per read operation, so a stream that keeps trickling
        would never trip it. The bound that matters is total wall clock for one turn.
        """
        import asyncio

        class _NeverFinishes(httpx.AsyncByteStream):
            def __init__(self, prefix: bytes) -> None:
                self._prefix = prefix

            async def __aiter__(self):  # type: ignore[no-untyped-def]
                yield self._prefix
                while True:
                    await asyncio.sleep(0.01)
                    yield b": ping\n\n"

        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_NeverFinishes(b": ping\n\n"),
        )

        with pytest.raises(LLMError) as caught:
            await _llm(_Recorder(response), timeout=0.2).complete(
                system="s", messages=[Message(role="user", content="q")]
            )
        assert "no response within" in str(caught.value)

    async def test_structured_is_bounded_the_same_way(self) -> None:
        """The drafting call is the longest one, so it needs the bound most."""
        import asyncio

        class _NeverFinishes(httpx.AsyncByteStream):
            async def __aiter__(self):  # type: ignore[no-untyped-def]
                while True:
                    await asyncio.sleep(0.01)
                    yield b": ping\n\n"

        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_NeverFinishes(),
        )

        with pytest.raises(LLMError):
            await _llm(_Recorder(response), timeout=0.2).structured(
                system="s",
                messages=[Message(role="user", content="q")],
                schema=_MINIMAL_SCHEMA,
            )

    async def test_a_per_call_timeout_overrides_the_provider_default(self) -> None:
        """Drafting passes its own, longer deadline.

        A live run gathered evidence for 239 seconds and then lost the whole
        investigation because the single drafting call hit the 120-second per-turn
        ceiling. The override is what stops that, so the reported bound has to be the
        one the caller asked for and not the client's.
        """
        import asyncio

        class _NeverFinishes(httpx.AsyncByteStream):
            async def __aiter__(self):  # type: ignore[no-untyped-def]
                while True:
                    await asyncio.sleep(0.01)
                    yield b": ping\n\n"

        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_NeverFinishes(),
        )

        # Client default is a deadline this call must NOT be held to.
        with pytest.raises(LLMError) as caught:
            await _llm(_Recorder(response), timeout=30.0).structured(
                system="s",
                messages=[Message(role="user", content="q")],
                schema=_MINIMAL_SCHEMA,
                timeout=0.2,
            )
        assert "no response within 0s" in str(caught.value)

    async def test_the_socket_read_timeout_stays_above_every_deadline(self) -> None:
        """Otherwise the transport, not us, decides when a long draft is over."""
        from cortex.agents.anthropic_llm import (
            _READ_TIMEOUT_SECONDS,
            REQUEST_TIMEOUT_SECONDS,
        )
        from cortex.agents.investigator import DRAFT_TIMEOUT_SECONDS

        assert _READ_TIMEOUT_SECONDS > max(DRAFT_TIMEOUT_SECONDS, REQUEST_TIMEOUT_SECONDS)
