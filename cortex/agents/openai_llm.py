"""An OpenAI-compatible provider, which is several providers.

**Why this exists.** The grounding stack was reachable only with an Anthropic key, so anyone
holding a different one could not run this project at all. ADR 0001 set the trigger for
revisiting the provider layer at "the second provider"; this is it.

**Why one adapter rather than a routing library.** The Chat Completions shape is spoken by
OpenAI, OpenRouter, Groq, Together, Fireworks, DeepInfra, vLLM, Ollama and LM Studio, so a single
adapter taking a `base_url` reaches all of them. A routing library would sit *underneath* the
`LLM` interface, which already is the abstraction, and would bring its own retry and timeout
policy into the one code path where this project's expensive defects have all lived — four of them
(F-15, F-16, F-18, F-19) were exactly this: a deadline that did not hold, or an error that
escaped. Those are owned deliberately here, and two policies in one path is how you get a
worst case nobody can state.

**What differs from the Anthropic path, beyond field names.**

  - *Tool results are one message each.* Anthropic wants a single user turn carrying one
    `tool_result` block per call; this wants one `role="tool"` message per call, each naming its
    `tool_call_id`. Both fail the same way when the pairing breaks, which is F-12 — so the
    conversion is written once, here, and `tests/unit/test_openai_llm.py` asserts the pairing
    rather than trusting it.
  - *Usage does not arrive in a stream unless you ask.* `stream_options={"include_usage": True}`
    is required, and without it every call reports zero tokens: no cost, no cache-hit rate, and
    a spend report that reads like the product is free.
  - *Truncation is `finish_reason == "length"`*, not a `stop_reason`.
  - *Schema conformance is not always guaranteed.* See `ModelCard.constrained_json_schema`.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import httpx
import openai

from cortex.agents.llm import (
    LLM,
    LLMAuthenticationFailed,
    LLMError,
    LLMOutputTruncated,
    LLMRefusedStructure,
    LLMResponse,
    Message,
    ToolRequest,
    Usage,
)
from cortex.agents.models import ModelCard, card_for
from cortex.config.settings import get_settings

#: Total wall clock for one turn, matching the Anthropic adapter's. Set below the loop budget
#: so the loop, not the socket, decides when to stop.
REQUEST_TIMEOUT_SECONDS = 120.0

#: Transport failures and 429s. Retrying is the SDK's job; the count is ours, because each
#: attempt can burn the whole timeout and the worst case has to stay inside the step budget.
MAX_RETRIES = 2

#: Above our own deadline, deliberately, for the reason the Anthropic adapter states: httpx
#: applies `read` per read operation rather than to the whole response, so a long pause between
#: chunks can trip it while the call is progressing normally. The deadline that matters is
#: enforced here with `asyncio.timeout`, so it stays the bound we state rather than an artefact
#: of how a transport counts chunks.
_READ_TIMEOUT_SECONDS = 600.0

#: Attempts after the first, for a failure that says nothing about the request.
#:
#: F-19 on this path. A capacity or rate-limit error from an OpenAI-compatible endpoint often
#: arrives *inside* a 200, as an SSE error event, which the SDK raises as a bare `APIError` --
#: so the client's own status-keyed retry never sees it. The Anthropic adapter carries this for
#: the same reason, recorded there: one `Overloaded` at the drafting step discarded a completed
#: 127-second investigation. Drafting is where it hurts, because every tool call is already paid
#: for by then.
_TRANSIENT_RETRIES = 2

#: Backoff before each retry. Fixed rather than jittered: with two attempts there is no herd to
#: spread out, and a predictable delay is easier to reason about against the deadline.
_BACKOFF_SECONDS = (2.0, 8.0)

#: Statuses that will not start working on the next attempt. 401 is a bad or absent key; 403 is
#: a key without access to this model, which is ordinary on a proxy where entitlements differ
#: per account. Both are configuration, and neither improves by being retried.
_TERMINAL_STATUSES = frozenset({401, 403})

#: OpenRouter's endpoint, because it is the one worth naming: a single key reaching many models
#: is the cheapest way for someone to try this with whatever they already have.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class _Transient(Exception):
    """A failure worth one more attempt, carrying the error to raise if retries run out."""

    def __init__(self, error: LLMError) -> None:
        super().__init__(str(error))
        self.error = error


class OpenAICompatLLM(LLM):
    """Any endpoint speaking OpenAI Chat Completions.

    `card` is accepted explicitly because this adapter can reach models this project has never
    heard of. `models.py` refuses an undeclared model on purpose — the request shape differs
    between generations and a wrong guess is a 400 at drafting time, the most expensive moment
    to discover a configuration error. That principle is kept rather than weakened: an unknown
    model is still refused, but the caller may *declare* it by passing a card, which is the
    difference between inferring capabilities from a name and being told them.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        card: ModelCard | None = None,
        client: openai.AsyncOpenAI | None = None,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self._model = model
        self.model = model
        self.card = card if card is not None else card_for(model)
        if card is None and not self.card.permitted:
            # Checked only when the card came from the table. A caller passing one explicitly has
            # declared the model deliberately, which is how a proxy reaches a model this project
            # has no card for; a card the table marks unpermitted is one somebody chose not to
            # run, and constructing it anyway is discovered on an invoice.
            raise ValueError(
                f"{model!r} is declared but not permitted. Set permitted=True on its card once "
                "an eval run says it should be, or pass a card to declare it deliberately."
            )
        if self.card.provider != "openai_compat":
            raise ValueError(
                f"{model!r} is declared as a {self.card.provider} model. Calling it through the "
                "OpenAI-compatible adapter would reach an endpoint it does not serve; use that "
                "provider's adapter, or pass a card declaring this one."
            )
        self._deadline = timeout
        settings = get_settings()
        resolved_base = base_url or settings.openai_base_url
        self.name = f"openai_compat:{_host(resolved_base)}:{model}"
        self._client = client or openai.AsyncOpenAI(
            api_key=_resolved_key(api_key or settings.openai_api_key, base_url=resolved_base),
            base_url=resolved_base,
            max_retries=max_retries,
            # A read timeout above the deadline, for the reason above. Connect stays short: a
            # host that will not accept a socket is not going to answer, and waiting the full
            # deadline for that spends an investigation's budget learning nothing.
            timeout=httpx.Timeout(_READ_TIMEOUT_SECONDS, connect=10.0),
        )

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": _to_wire(system, messages),
            "max_completion_tokens": min(max_tokens, self.card.max_output_tokens),
        }
        if tools:
            kwargs["tools"] = [_to_wire_tool(spec) for spec in tools]
        completion = await self._call(deadline=self._deadline, **kwargs)
        choice = completion.choices[0] if completion.choices else None
        if choice is None:
            raise LLMError(
                f"{self.name}: the response stream carried no choices. This is a failed turn, "
                "not an empty answer -- treating it as one would let the loop read it as a "
                "decision to conclude and draft from whatever evidence it had."
            )

        message = choice.message
        if getattr(message, "refusal", None):
            # Surfaced rather than returned as an empty turn, matching the Anthropic adapter:
            # the same prompt will be declined again, and an investigation that silently drops a
            # refused step presents an incomplete picture as a complete one.
            raise LLMError(
                f"{self.name}: request declined by a content filter: {message.refusal[:200]}"
            )
        requests: list[ToolRequest] = []
        for call in message.tool_calls or []:
            function = getattr(call, "function", None)
            if function is None:
                continue
            requests.append(
                ToolRequest(
                    id=call.id,
                    name=function.name,
                    # Arguments arrive as a JSON *string* here, unlike the Anthropic path where
                    # the SDK has already parsed them. A model emitting malformed arguments is
                    # ordinary, so this cannot raise: an empty dict reaches the executor, which
                    # rejects it against the capability's schema and hands the model a
                    # correctable error naming what was wrong. Raising here would lose the turn.
                    arguments=_loads_object(function.arguments),
                )
            )
        stop_reason = _stop_reason(choice.finish_reason)
        if stop_reason == "tool_use" and not requests:
            # The provider said it stopped to call tools and named none. Raised rather than
            # returned, because a turn with no tool requests is exactly what the loop reads as
            # "the analyst is finished" -- so the contradiction the adapter has already computed
            # would otherwise be discarded and the run would end early looking healthy.
            raise LLMError(
                f"{self.name}: finish_reason said tool_calls and no usable tool call arrived"
            )
        return LLMResponse(
            text=message.content or "",
            tool_requests=requests,
            usage=_usage(completion),
            stop_reason=stop_reason,
        )

    async def structured(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        schema: dict[str, Any],
        max_tokens: int = 8192,
        timeout: float | None = None,
    ) -> tuple[dict[str, Any], Usage]:
        if not self.card.supports_structured_outputs:
            # Refused rather than degraded to prose-parsing, matching the Anthropic adapter. The
            # grounding gate operates on validated structure, and a model that cannot produce it
            # cannot draft a report anyone should deliver.
            raise LLMRefusedStructure(
                f"{self.name}: this model does not support structured outputs"
            )
        effective = min(max_tokens, self.card.max_output_tokens)
        constrained = self.card.constrained_json_schema
        completion = await self._call(
            deadline=timeout or self._deadline,
            model=self._model,
            # The schema goes in the prompt when it cannot go in the request. See
            # `_schema_instruction`: in JSON-object mode `response_format` carries no schema at
            # all, so this is the only thing telling the model what shape to produce.
            messages=_to_wire(
                system if constrained else system + _schema_instruction(schema), messages
            ),
            max_completion_tokens=effective,
            response_format=_response_format(schema, constrained=constrained),
        )
        choice = completion.choices[0] if completion.choices else None
        if choice is None:
            raise LLMRefusedStructure(f"{self.name}: response carried no choices")
        if choice.finish_reason == "length":
            # Truncated JSON would fail to parse, and a partially-parsed report would silently
            # lose findings. Named as recoverable rather than fatal: the caller retries with an
            # instruction to write less, which is worth doing because this failure lands after
            # every tool call has already been paid for.
            raise LLMOutputTruncated(
                f"{self.name}: output truncated at max_completion_tokens={effective}; "
                "the schema could not be completed"
            )
        refusal = getattr(choice.message, "refusal", None)
        if refusal:
            raise LLMRefusedStructure(f"{self.name}: declined to produce the requested structure")
        try:
            payload = json.loads(choice.message.content or "")
        except ValueError as exc:
            # Reachable on any endpoint whose card says `constrained_json_schema=False`, which is
            # most of what a proxy reaches. The caller's repair attempt is what makes that
            # survivable, and the card is what makes it expected rather than mysterious.
            raise LLMRefusedStructure(f"{self.name}: structured output was not valid JSON") from exc
        if not isinstance(payload, dict):
            raise LLMRefusedStructure(
                f"{self.name}: structured output was {type(payload).__name__}, expected object"
            )
        return payload, _usage(completion)

    async def _call(self, *, deadline: float, **kwargs: Any) -> Any:
        """One request, streamed, with the deadline and the error mapping in one place.

        Streamed for the reason the Anthropic adapter streams: a tool-calling turn on a
        reasoning model can outlive a non-streaming timeout, and a request that dies at the
        transport layer is indistinguishable from a hung one.
        """
        for attempt in range(_TRANSIENT_RETRIES + 1):
            try:
                return await self._attempt(deadline=deadline, **kwargs)
            except _Transient as exc:
                if attempt >= _TRANSIENT_RETRIES:
                    raise exc.error from exc.error
                await asyncio.sleep(_BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)])
        # Unreachable: the last attempt re-raises above. Present so a future change to the loop
        # bounds cannot turn this into a function that returns None.
        raise LLMError(f"{self.name}: retries exhausted without a response")

    async def _attempt(self, *, deadline: float, **kwargs: Any) -> Any:
        """One request. Raises `_Transient` for a failure worth trying again."""
        try:
            async with asyncio.timeout(deadline):
                stream = await self._client.chat.completions.create(
                    stream=True,
                    # Without this the stream carries no usage at all and every call reports
                    # zero tokens -- no cost, no cache-hit rate, and a spend report that reads
                    # like the product is free.
                    stream_options={"include_usage": True},
                    **kwargs,
                )
                return await _assemble(stream)
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise LLMError(f"{self.name}: no response within {deadline:.0f}s") from exc
        except httpx.HTTPError as exc:
            # Raised while consuming the stream body, where the SDK's mapping no longer applies.
            # Without this a mid-stream read failure escapes as a transport exception and takes
            # the process with it, which is F-16.
            raise LLMError(f"{self.name}: transport failed mid-stream: {exc!r}") from exc
        except openai.APIStatusError as exc:
            # The provider's own message is included: a 400 naming the offending schema keyword
            # is actionable, where a bare status code sends whoever reads the log back to
            # reproduce the request by hand. Proxies are the reason this matters most -- their
            # 400s are frequently about a model's own limits rather than about the request.
            detail = f"{self.name}: {exc.status_code} {exc.__class__.__name__}: {_message_of(exc)}"
            if exc.status_code in _TERMINAL_STATUSES:
                # A rejected credential rejects every later call, so the loop must not spend its
                # budget discovering that one step at a time.
                raise LLMAuthenticationFailed(detail) from exc
            raise LLMError(detail) from exc
        except openai.APIError as exc:
            # A mid-stream capacity error arrives here as a bare `APIError`, and it is the one
            # worth another attempt: it says nothing about the request, so the same bytes may
            # well succeed a moment later. An authentication failure never reaches this branch,
            # and a status error is classified above.
            if type(exc) is openai.APIError:
                raise _Transient(
                    LLMError(f"{self.name}: {exc.__class__.__name__}: {_message_of(exc)}")
                ) from exc
            # APITimeoutError is an APIError subclass, so a hung request lands here rather than
            # escaping as something the loop cannot classify.
            #
            # The message is carried, not just the class. A capacity or rate-limit error
            # delivered *inside* a 200 as an SSE error event arrives here as a bare `APIError`,
            # so the class name alone reads "APIError" and tells whoever finds it in the
            # investigation row nothing at all. That is F-13, and proxies are where it bites:
            # their mid-stream errors are the common failure and the message is the only thing
            # naming which upstream refused and why.
            raise LLMError(f"{self.name}: {exc.__class__.__name__}: {_message_of(exc)}") from exc


async def _assemble(stream: Any) -> Any:
    """Collapse a stream of deltas into one completion-shaped object.

    The SDK offers no equivalent of `get_final_message()` here, so this does the accumulation:
    text, tool calls by index, the finish reason, and the usage chunk that arrives last and
    carries no choices at all.
    """
    text: list[str] = []
    refusal: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage: Any = None
    #: Whether the provider ever sent a choice. A stream can end without one -- a dropped
    #: connection after the headers, a proxy emitting `[DONE]` with no data, a 200 with an empty
    #: body -- and the difference between that and a turn of empty text is the difference
    #: between a failure and a decision to stop. Tracked rather than inferred from empty text,
    #: because a model genuinely may answer with nothing.
    saw_choice = False

    async for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        for choice in getattr(chunk, "choices", None) or []:
            saw_choice = True
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            if getattr(delta, "content", None):
                text.append(delta.content)
            if getattr(delta, "refusal", None):
                refusal.append(delta.refusal)
            for call in getattr(delta, "tool_calls", None) or []:
                # Keyed by index rather than by id: the id arrives in the first delta for a call
                # and is absent from every later one, so accumulating by id would open a new
                # call per fragment and produce a turn full of one-character arguments.
                entry = calls.setdefault(call.index, {"id": "", "name": "", "arguments": ""})
                if call.id:
                    entry["id"] = call.id
                function = getattr(call, "function", None)
                if function is not None:
                    if getattr(function, "name", None):
                        entry["name"] = function.name
                    if getattr(function, "arguments", None):
                        entry["arguments"] += function.arguments

    return _Completion(
        text="".join(text),
        refusal="".join(refusal) or None,
        calls=[calls[index] for index in _ordered(calls)],
        finish_reason=finish_reason,
        usage=usage,
        empty=not saw_choice,
    )


def _ordered(calls: dict[Any, dict[str, Any]]) -> list[Any]:
    """Call indices in a stable order, tolerating one that is not an integer.

    `index` is required by the API and built with `construct_type`, which does not validate --
    so a nonconforming proxy can send `None`, and sorting mixed types raises a `TypeError` that
    escapes every handler in `_call` and kills the investigation with a traceback. Reaching
    nonconforming proxies is this adapter's whole reason for existing.
    """
    return sorted(calls, key=lambda index: (index is None, index if index is not None else 0))


class _Completion:
    """The assembled result, shaped like the non-streaming response the callers read.

    A small stand-in rather than the SDK's own model: constructing one of those from deltas
    means satisfying a validator for fields this code never looks at.
    """

    def __init__(
        self,
        *,
        text: str,
        refusal: str | None,
        calls: list[dict[str, Any]],
        finish_reason: str | None,
        usage: Any,
        empty: bool = False,
    ) -> None:
        self.usage = usage
        # No choices when the provider sent none, which makes the callers' `choice is None`
        # guards live rather than dead code. They were unreachable before, and the consequence
        # was that an empty stream returned a turn with no text and no tool calls -- which the
        # loop reads as the analyst deciding it is finished, so the run drafted a report from
        # partial evidence and recorded no error anywhere.
        self.choices = (
            []
            if empty
            else [_Choice(text=text, refusal=refusal, calls=calls, finish=finish_reason)]
        )


class _Choice:
    def __init__(
        self, *, text: str, refusal: str | None, calls: list[dict[str, Any]], finish: str | None
    ) -> None:
        self.finish_reason = finish
        self.message = _AssembledMessage(text=text, refusal=refusal, calls=calls)


class _AssembledMessage:
    def __init__(self, *, text: str, refusal: str | None, calls: list[dict[str, Any]]) -> None:
        self.content = text
        self.refusal = refusal
        self.tool_calls = [
            _AssembledCall(id=call["id"], name=call["name"], arguments=call["arguments"])
            for call in calls
            if call["name"]
        ]


class _AssembledCall:
    def __init__(self, *, id: str, name: str, arguments: str) -> None:  # noqa: A002
        self.id = id
        self.function = _AssembledFunction(name=name, arguments=arguments)


class _AssembledFunction:
    def __init__(self, *, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


def _to_wire(system: str, messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Convert Cortex messages to the Chat Completions shape.

    **The pairing rule, which is F-12 in this dialect.** An assistant turn that requested tools
    must carry `tool_calls`, and each result must follow as its own `role="tool"` message naming
    the same `tool_call_id`. Send the assistant's prose without its `tool_calls` and the results
    that follow reference calls the transcript never made, which is a 400 mid-run after the
    tokens are spent.
    """
    wire: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for message in messages:
        if message.tool_results:
            wire.extend(
                {"role": "tool", "tool_call_id": request_id, "content": result}
                for request_id, result in message.tool_results.items()
            )
            # A per-turn instruction may ride along after the results -- the width schedule is
            # "a user message instruction at each step". It becomes its own user turn here,
            # after the results, because it is guidance about the next turn rather than an
            # observation about the last one.
            if message.content:
                wire.append({"role": "user", "content": message.content})
            continue
        if message.tool_requests:
            wire.append(
                {
                    "role": "assistant",
                    "content": message.content or None,
                    "tool_calls": [
                        {
                            "id": request.id,
                            "type": "function",
                            "function": {
                                "name": request.name,
                                "arguments": json.dumps(request.arguments),
                            },
                        }
                        for request in message.tool_requests
                    ],
                }
            )
            continue
        wire.append({"role": message.role, "content": message.content})
    return wire


def _to_wire_tool(spec: dict[str, Any]) -> dict[str, Any]:
    """One registry tool spec in function-calling shape.

    The registry emits `{name, description, input_schema}`, which is the Anthropic shape. It is
    converted here rather than at the registry because the wire format is a provider's business
    and the registry serves both.
    """
    return {
        "type": "function",
        "function": {
            "name": spec["name"],
            "description": spec.get("description", ""),
            "parameters": spec.get("input_schema") or {"type": "object", "properties": {}},
        },
    }


def _response_format(schema: dict[str, Any], *, constrained: bool) -> dict[str, Any]:
    """How to ask for the schema, given what the endpoint can actually promise.

    `strict` is what turns this into constrained decoding rather than a strong suggestion, and
    it is only sent where the card says the endpoint honours it: an endpoint that does not
    frequently rejects the key outright, which would turn every drafting call into a 400.

    **A card claiming constraint is checked, not believed.** Strict mode is not merely a flag on
    an arbitrary schema -- it requires every property of every object to appear in `required`,
    and a schema with an optional field is rejected outright. Cortex's report schema has nine
    such objects, so sending it strict would 400 at the drafting call: after every tool call is
    paid for, which is where F-11 and F-17 both landed. Refusing here names the offending
    fields instead, and does so at the moment the request is built.
    """
    if not constrained:
        return {"type": "json_object"}
    violations = _strict_violations(schema)
    if violations:
        raise LLMRefusedStructure(
            "this model's card claims constrained schema output, but strict mode requires every "
            "property to be required and these objects have optional ones: "
            + "; ".join(f"{path} ({', '.join(fields)})" for path, fields in violations[:4])
            + (f"; and {len(violations) - 4} more" if len(violations) > 4 else "")
            + ". Set constrained_json_schema=False on the card, or make the schema total."
        )
    return {
        "type": "json_schema",
        "json_schema": {"name": "report", "strict": True, "schema": schema},
    }


def _strict_violations(schema: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """Object nodes whose properties are not all required, which strict mode forbids.

    `additionalProperties: false` is the other strict requirement and is not checked, because
    the report models already forbid extras -- so it holds by construction and a check for it
    would assert a property of Pydantic rather than of this schema.
    """
    found: list[tuple[str, list[str]]] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                properties = node.get("properties") or {}
                optional = sorted(set(properties) - set(node.get("required") or []))
                if optional:
                    found.append((path, optional))
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(schema, "$")
    return found


def _schema_instruction(schema: dict[str, Any]) -> str:
    """The schema as prompt text, for an endpoint that will not take it as a constraint.

    **Two problems, one fix.** In JSON-object mode `response_format` carries no schema, so the
    model is told to emit JSON without being told *which* JSON -- it would produce a
    well-formed document of its own invention that could never validate as a report. And the
    mode itself requires the word "json" to appear in the messages, which a prompt written for
    a constrained endpoint has no reason to contain: the first live drafting call against a real
    endpoint was a 400 saying exactly that.

    So the schema is sent as text. This costs real tokens -- the report schema is about 10.5KB,
    on the call that already carries the whole investigation transcript -- and that cost is the
    honest price of an endpoint without constrained decoding. It buys the only thing that makes
    the mode usable at all.
    """
    return (
        "\n\nRespond with a single JSON object and nothing else: no prose, no code fences.\n"
        "It must conform to this JSON Schema:\n"
        f"{json.dumps(schema, separators=(',', ':'))}\n"
        "Omit any property the schema does not require and you have no evidence for, rather "
        "than inventing a value for it."
    )


def _stop_reason(finish_reason: str | None) -> str:
    """Map a finish reason onto the vocabulary the loop already branches on."""
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "refusal",
    }.get(finish_reason or "", finish_reason or "end_turn")


def _usage(completion: Any) -> Usage:
    """Token counts, including the cached share.

    Cached input is reported nested under `prompt_tokens_details` and is *included* in
    `prompt_tokens`, unlike the Anthropic shape where cache reads are counted separately. So it
    is subtracted out here: leaving it in would double-count the cached tokens and price them at
    the full input rate, overstating a bill by roughly the cache-hit rate.
    """
    usage = getattr(completion, "usage", None)
    if usage is None:
        return Usage()
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    details = getattr(usage, "prompt_tokens_details", None)
    cached = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
    # **Only subtract where the cached share is genuinely included.** OpenAI counts cached tokens
    # inside `prompt_tokens`, so subtracting is right there. Several compatible providers report
    # `prompt_tokens` already net of the cache, and then `cached > prompt` -- where a blind
    # subtraction clamped to zero and moved the whole prompt into the cache-read bucket, priced
    # at a tenth. A hundred thousand fresh input tokens would have billed as ten thousand.
    #
    # The direction is what makes this worth a branch rather than a clamp: `spend.py` exists on
    # the principle that a report which overstates is one somebody checks, while one that
    # understates is one somebody trusts.
    fresh = prompt - cached if cached <= prompt else prompt
    return Usage(
        input_tokens=max(0, fresh),
        output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        cache_read_input_tokens=cached,
    )


def _loads_object(raw: Any) -> dict[str, Any]:
    """Parse tool arguments, defensively: malformed arguments must not lose the turn."""
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _message_of(exc: openai.APIStatusError) -> str:
    """The provider's error text, defensively — an error path must not itself raise."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"][:500]
        if isinstance(error, str):
            return error[:500]
    return str(exc)[:500]


def _resolved_key(key: str | None, *, base_url: str | None) -> str:
    """The API key, or a refusal that says which variable to set.

    **Two cases, and only one of them is an error.** A local endpoint -- Ollama, vLLM, LM Studio
    -- serves models without authentication and its client library conventionally sends a
    placeholder, so demanding a key there would make the cheapest way to try this project
    impossible. A request to a hosted endpoint with no key is a 401 after a round trip.

    Raised here rather than left to the SDK because its own message names `workload_identity`
    and `OPENAI_ADMIN_KEY`, neither of which this project uses -- and F-14 was a key-resolution
    bug that cost real time, so the failure is worth saying plainly.
    """
    if key:
        return key
    if base_url:
        # Any non-empty string satisfies a local server; the value is never checked.
        return "not-required-for-a-local-endpoint"
    raise LLMError(
        "openai_compat: no API key. Set OPENAI_API_KEY, or set OPENAI_BASE_URL to a local "
        "endpoint (Ollama, vLLM, LM Studio) which needs no key."
    )


def _host(base_url: str | None) -> str:
    """A short label for audit records. Never the full URL: a proxy base can carry a token."""
    if not base_url:
        return "api.openai.com"
    return httpx.URL(base_url).host or "unknown"
