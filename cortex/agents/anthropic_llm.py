"""Anthropic provider.

Notes that matter for correctness on current models, verified against the SDK
reference rather than recalled:

  - `temperature`, `top_p` and `top_k` are **rejected** on Opus 5 / Sonnet 5. Output
    variance is steered by prompt, not by sampling parameters.
  - Thinking is configured as `{"type": "adaptive"}`. The old
    `{"type": "enabled", "budget_tokens": N}` form returns a 400. Depth is
    controlled by `output_config.effort` instead.
  - A safety classifier can decline a request: HTTP 200 with
    `stop_reason == "refusal"` and a possibly-empty `content`. Code that reads
    `content[0]` unconditionally crashes on that path, so the stop reason is
    checked first.
  - Large `max_tokens` must stream, or the request hits an HTTP timeout. The
    report draft is big enough to need it.
  - Streaming does not imply a deadline. A request can sit open indefinitely, which
    silently defeats the loop's wall-clock budget because that budget is only checked
    between steps. Every call is therefore bounded by `REQUEST_TIMEOUT_SECONDS`.
  - Structured outputs accept only a subset of JSON Schema: no `minimum`/`maximum`,
    no `minLength`/`maxLength`, no `pattern`, and array minimums of only 0 or 1.
    A Pydantic model that uses any of them produces a 400 at the moment a report is
    drafted — after the investigation has already spent its tokens. The schema is
    therefore transformed before it goes on the wire, which moves each dropped bound
    into the field's description so the model is still told about it. Pydantic then
    enforces the real constraint when the response is parsed, so the guarantee is
    unchanged; only where it is checked moves.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import anthropic
import httpx
from anthropic.lib._parse._transform import transform_schema

from cortex.agents.llm import (
    LLM,
    LLMError,
    LLMOutputTruncated,
    LLMRefusedStructure,
    LLMResponse,
    Message,
    ToolRequest,
    Usage,
)
from cortex.agents.models import ALLOWED_MODELS as _ALLOWED_MODELS
from cortex.agents.models import ModelCard, card_for
from cortex.config.settings import get_settings

#: The only model Cortex runs.
#:
#: Restricted to one, and to a mid-tier one, as a cost decision: an investigation spends
#: 100–170k tokens, and the drafting call alone can be 30k of output. On Opus that is
#: roughly $0.80 an investigation before the verifier's per-claim calls; a suite of six
#: costs more than a subscription. Sonnet is a fraction of the input price and does the
#: work the eval measures. Sonnet 5 rather than 4.6 because it is both newer and, on
#: intro pricing through 2026-08-31, cheaper: $2/$10 per Mtok against $3/$15.
#:
#: Naming one model rather than a default plus overrides is deliberate. A per-call
#: override is how a cost ceiling quietly stops applying — one caller passes something
#: more capable "just for drafting" and the bill returns. `ALLOWED_MODELS` enforces it.
DEFAULT_MODEL = "claude-sonnet-5"

#: Kept as a name because callers refer to it for narrow, high-volume steps — evidence
#: extraction, single-claim verification. It resolves to the same model: there is no
#: cheaper tier in use, and pointing it at Haiku would change what the verifier catches
#: without measuring the effect first.
FAST_MODEL = DEFAULT_MODEL

#: Models this provider will construct, derived from `cortex.agents.models`. An
#: unexpected model is refused at construction rather than discovered on an invoice,
#: which is the only point at which the restriction is worth anything.
#:
#: Re-exported here because callers imported it from this module before the capability
#: table existed. It is the same object, not a copy.
ALLOWED_MODELS = _ALLOWED_MODELS

#: Thinking budget for a model whose card declares the pre-4.6 `budget` style. Only
#: reached if such a model is ever permitted; sized to leave room for a report draft
#: within a 64k output ceiling.
_THINKING_BUDGET_TOKENS = 8192


#: Per-request ceiling. The loop's wall-clock budget can only be checked between
#: steps, so a single unbounded call defeats it entirely: an investigation observed
#: hanging on one open streaming request outlived its 300-second budget many times
#: over while using no CPU. A 90-second product cannot wait indefinitely on one turn.
#: Set below the loop budget so the loop, not the socket, decides when to stop.
REQUEST_TIMEOUT_SECONDS = 120.0

#: Transport failures and 429s only. Retrying is the SDK's job, but the count is
#: ours: each attempt can burn the whole timeout, so the worst case is
#: (1 + retries) x timeout and has to stay inside the step budget.
MAX_RETRIES = 2

#: The socket read timeout is set above our own deadline on purpose. httpx applies
#: `read` per read operation, not to the whole response, so a long thinking pause
#: between chunks can trip it even though the call is progressing normally. The
#: deadline that matters -- total wall clock for one turn -- is enforced by us with
#: `asyncio.timeout`, so the bound is the one we state rather than an artefact of how
#: a transport counts chunks. It therefore has to stay above the longest deadline any
#: caller passes to `structured(timeout=...)`, or the transport would start deciding
#: again — which is what this constant exists to prevent.
_READ_TIMEOUT_SECONDS = 600.0

#: Provider-side error types worth trying again. A capacity error says nothing about
#: the request, so the same bytes may well succeed a moment later.
#: `invalid_request_error` and the authentication types are deliberately absent:
#: retrying those spends time and tokens to reach the same answer.
_TRANSIENT_ERROR_TYPES = frozenset({"overloaded_error", "api_error", "rate_limit_error"})

#: Attempts after the first, for transient errors only.
#:
#: These arrive **inside** a 200 response as an SSE `error` event, so the SDK's own
#: `max_retries` never sees them — its policy keys on the response status, and the
#: status is 200. A live run lost a completed 127-second investigation to a single
#: `Overloaded` at the drafting step: the same asymmetry F-18 describes, where the
#: failure lands after all the work is already done.
#:
#: Each attempt gets the full deadline, so the worst case is
#: (1 + retries) x deadline + backoff. Tolerable because a capacity error fails fast —
#: it does not consume the deadline it is entitled to.
_TRANSIENT_RETRIES = 2

#: Backoff before each retry. Fixed rather than jittered: with two attempts there is no
#: herd to spread out, and a predictable delay is easier to reason about against the
#: deadline above.
_BACKOFF_SECONDS = (2.0, 8.0)

#: Prompt caching, applied to every request.
#:
#: Measured, not guessed: an investigation made 11 sequential model calls resending
#: **168,000 input tokens** with none of it cached, and those calls were 72% of the wall
#: clock. The loop appends to a transcript and re-sends the whole thing each step, so by
#: construction almost every request is a long stable prefix plus one new turn — which is
#: precisely the shape prompt caching exists for.
#:
#: The top-level form is used rather than hand-placed breakpoints. It marks the last
#: cacheable block automatically, which gives the rolling behaviour the loop needs: step N
#: writes the cache, step N+1 reads the prefix and extends it. Manual breakpoints would
#: mean tracking which message to mark as the transcript grows, with only four available
#: and a silent cache miss as the failure mode.
#:
#: Caching is prefix-based, so ordering matters and ours already suits it: the system
#: prompt and the tool schemas are identical across every call, and only the tail changes.
#: A 5-minute TTL is enough — loop steps are ~9.5 seconds apart — and a 1-hour TTL costs
#: more to write, which is only worth it if a prefix must survive between investigations.
CACHE_CONTROL = {"type": "ephemeral"}

#: Thinking depth, and the default is deliberately the cheapest.
#:
#: Measured, not assumed. At `high` the suite scored 4/6 with latencies of 199-330s and
#: 178k-295k tokens per investigation. At `low`, with prompt caching on, it scored
#: **6/6** with latencies of 51-108s and 4.6k-9.3k tokens: grounding, accuracy, decoy
#: rejection, tool selection and completeness all 1.00 on every attempt, and zero
#: delivered hallucinations. Faster, roughly thirty times cheaper, and not worse on any
#: dimension the suite measures.
#:
#: `onboarding_regression` is the reason this needed measuring rather than reasoning
#: about: it requires naming a specific deploy sha, so it was the scenario most likely
#: to suffer from less thinking. It failed at `high` on Opus and passes twice at `low`.
DEFAULT_EFFORT = "low"


class AnthropicLLM(LLM):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        effort: str = DEFAULT_EFFORT,
        client: anthropic.AsyncAnthropic | None = None,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        max_retries: int = MAX_RETRIES,
        transient_retries: int = _TRANSIENT_RETRIES,
        backoff: Sequence[float] = _BACKOFF_SECONDS,
    ) -> None:
        if model not in ALLOWED_MODELS:
            # `card_for` first, so an undeclared model is told to declare itself rather
            # than told it is not permitted — two different mistakes with two different
            # fixes. A declared-but-unpermitted model falls through to the cost message.
            card_for(model)
            raise ValueError(
                f"model {model!r} is not permitted; Cortex runs "
                f"{', '.join(sorted(ALLOWED_MODELS))}. This is a cost restriction — "
                "set permitted=True on its ModelCard deliberately rather than passing a "
                "model here."
            )
        # Every request-shape decision below reads from the card rather than assuming the
        # current generation's behaviour. That is the whole point of the table: permitting
        # a second model must not require remembering which parameters it rejects.
        self.card: ModelCard = card_for(model)
        self.name = f"anthropic:{model}"
        self._model = model
        # Public, for callers that need to know *which* model without calling it — the spend
        # aggregation prices per model, and reaching into a private attribute for that would
        # make the pricing path depend on this class's internals.
        self.model = model
        self._effort = effort
        # Injectable so a test can assert the retry policy without sleeping through it.
        self._transient_retries = transient_retries
        self._backoff = tuple(backoff)
        # A client is injectable so tests can supply a transport without an API key.
        # The key comes from settings rather than the SDK's own os.environ lookup, so a
        # key present only in .env still works from a worker or the eval harness.
        self._deadline = timeout
        self._client = client or anthropic.AsyncAnthropic(
            api_key=api_key or get_settings().anthropic_api_key,
            timeout=httpx.Timeout(_READ_TIMEOUT_SECONDS, connect=10.0),
            max_retries=max_retries,
        )

    def _shape(self, *, max_tokens: int, cacheable: bool = True) -> dict[str, Any]:
        """The parameters whose acceptance depends on which model this is.

        Built from the card in one place, so both call paths agree. Before this existed
        the adaptive-thinking form and the effort setting were written out twice, and
        permitting a model that rejects either would have produced a 400 from one path
        and not the other.
        """
        card = self.card
        # Clamped rather than passed through: a max_tokens above the model's ceiling is a
        # 400, and the caller asking for more room than exists is better served by the
        # most room available plus a truncation error it already handles than by a
        # rejected request.
        shape: dict[str, Any] = {"max_tokens": min(max_tokens, card.max_output_tokens)}

        if card.thinking == "adaptive":
            shape["thinking"] = {"type": "adaptive"}
        elif card.thinking == "budget":
            # The pre-4.6 form. `budget_tokens` must leave room for the answer itself,
            # so it is capped below the request's own output ceiling.
            shape["thinking"] = {
                "type": "enabled",
                "budget_tokens": min(_THINKING_BUDGET_TOKENS, shape["max_tokens"] // 2),
            }

        # **Caching a request nothing will read back costs 25% extra, not nothing.** A write
        # bills at 1.25x fresh input and a read at 0.1x, so the premium is only repaid once
        # something reads it. The per-request log showed all seven verifier calls writing
        # ~1.8k and reading zero: each judges a different claim, so no later call shares
        # their prefix. For those, paying to store the request is a pure surcharge.
        if card.supports_prompt_cache and cacheable:
            shape["cache_control"] = CACHE_CONTROL
        return shape

    def _output_config(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """`output_config`, with `effort` only where the model accepts it."""
        config: dict[str, Any] = dict(extra or {})
        if self.card.supports_effort:
            config["effort"] = self._effort
        return config

    async def _final_message(self, *, deadline: float, **kwargs: Any) -> Any:
        """Stream one turn and return the assembled message.

        Shared by both call paths on purpose. The deadline, the error mapping and the
        retry policy were duplicated before, and drafting — the call that can least
        afford to lose its work — is exactly the one a copy would have missed.
        """
        for attempt in range(self._transient_retries + 1):
            try:
                async with (
                    asyncio.timeout(deadline),
                    self._client.messages.stream(**kwargs) as stream,
                ):
                    return await stream.get_final_message()
            except (TimeoutError, httpx.TimeoutException) as exc:
                raise LLMError(f"{self.name}: no response within {deadline:.0f}s") from exc
            except httpx.HTTPError as exc:
                # Raised while consuming the stream body, where the SDK's own mapping to
                # APITimeoutError no longer applies. Without this a mid-stream read
                # timeout escapes as a transport exception and takes the process with it.
                raise LLMError(f"{self.name}: transport failed mid-stream: {exc!r}") from exc
            except anthropic.APIStatusError as exc:
                # Only errors delivered inside a 200 are ours to retry. A 429 or 529
                # was already retried by the SDK, whose policy keys on status; retrying
                # it again here multiplies the two budgets together and can spend
                # (1 + sdk_retries) x (1 + our_retries) deadlines on one turn.
                retryable = (
                    exc.status_code == 200
                    and _error_type(exc) in _TRANSIENT_ERROR_TYPES
                    and attempt < self._transient_retries
                )
                if not retryable:
                    # The provider's own message is included: a 400 naming the offending
                    # schema keyword is actionable, while a bare status code sends
                    # whoever reads the log back to reproduce the request by hand. The
                    # attempt count is included too, so an exhausted retry reads
                    # differently from a first-try rejection.
                    raise LLMError(
                        f"{self.name}: {exc.status_code} {exc.__class__.__name__}: "
                        f"{_api_message(exc)} (attempts: {attempt + 1})"
                    ) from exc
                # An empty schedule means retry immediately, not crash on an index.
                await asyncio.sleep(
                    self._backoff[min(attempt, len(self._backoff) - 1)] if self._backoff else 0.0
                )
            except anthropic.APIError as exc:
                # APITimeoutError is an APIError subclass, so a hung request lands here
                # rather than escaping as a transport exception the loop cannot classify.
                raise LLMError(f"{self.name}: {exc.__class__.__name__}") from exc

        # Unreachable: the last attempt raises above. Present so a future change to the
        # loop bounds cannot turn this into a function that returns None.
        raise LLMError(f"{self.name}: retries exhausted without a response")

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
            "system": system,
            "messages": _to_wire(messages),
            **self._shape(max_tokens=max_tokens),
        }
        output_config = self._output_config()
        if output_config:
            kwargs["output_config"] = output_config
        if tools:
            kwargs["tools"] = list(tools)

        # Streamed and reassembled: a tool-calling turn with adaptive thinking can run
        # long enough to trip the non-streaming timeout.
        response = await self._final_message(deadline=self._deadline, **kwargs)

        if response.stop_reason == "refusal":
            # Surfaced rather than retried: the same prompt will be declined again,
            # and an investigation that silently drops a refused step would present
            # an incomplete picture as a complete one.
            raise LLMError(
                f"{self.name}: request declined by a safety classifier "
                f"(category={_refusal_category(response)})"
            )

        text_parts: list[str] = []
        tool_requests: list[ToolRequest] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_requests.append(
                    ToolRequest(
                        id=block.id,
                        name=block.name,
                        # Already parsed by the SDK. Never re-parse from a string:
                        # escaping differs across models and raw matching breaks.
                        arguments=dict(block.input) if isinstance(block.input, dict) else {},
                    )
                )

        return LLMResponse(
            text="\n".join(text_parts),
            tool_requests=tool_requests,
            usage=_usage(response),
            stop_reason=response.stop_reason or "end_turn",
        )

    async def structured(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        schema: dict[str, Any],
        max_tokens: int = 8192,
        timeout: float | None = None,
        cacheable: bool = True,
    ) -> tuple[dict[str, Any], Usage]:
        if not self.card.supports_structured_outputs:
            # Refused rather than degraded to prose-parsing. The grounding gate operates
            # on validated structure, and a model that cannot produce it cannot draft a
            # report we would be willing to deliver.
            raise LLMRefusedStructure(
                f"{self.name}: this model does not support structured outputs"
            )
        # `format` constrains the response to the schema. This is what makes the grounding
        # gate possible: it operates on validated structure, not on prose it would
        # otherwise have to parse. The drafting call also carries the entire investigation
        # transcript, so its prefix is the largest and most cacheable request we make —
        # and with a repair attempt it can be sent twice.
        response = await self._final_message(
            deadline=timeout or self._deadline,
            model=self._model,
            system=system,
            messages=_to_wire(messages),
            output_config=self._output_config(
                {"format": {"type": "json_schema", "schema": transform_schema(schema)}}
            ),
            **self._shape(max_tokens=max_tokens, cacheable=cacheable),
        )

        if response.stop_reason == "refusal":
            raise LLMRefusedStructure(
                f"{self.name}: declined to produce the requested structure "
                f"(category={_refusal_category(response)})"
            )
        if response.stop_reason == "max_tokens":
            # Truncated JSON would fail to parse, and a partially-parsed report
            # would silently lose findings.
            # The effective ceiling, not the requested one: they differ when the caller
            # asked for more room than the model has, and a message naming a limit the
            # request never carried sends whoever reads it looking in the wrong place.
            effective = min(max_tokens, self.card.max_output_tokens)
            raise LLMOutputTruncated(
                f"{self.name}: output truncated at max_tokens={effective}; "
                "the schema could not be completed"
            )

        text = "".join(block.text for block in response.content if block.type == "text")
        try:
            payload = json.loads(text)
        except ValueError as exc:
            raise LLMRefusedStructure(f"{self.name}: structured output was not valid JSON") from exc
        if not isinstance(payload, dict):
            raise LLMRefusedStructure(
                f"{self.name}: structured output was {type(payload).__name__}, expected object"
            )
        return payload, _usage(response)


def _error_body(exc: anthropic.APIStatusError) -> dict[str, Any]:
    """The `error` object from the response body, or an empty dict."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return error
    return {}


def _error_type(exc: anthropic.APIStatusError) -> str:
    """The provider's error type, e.g. `overloaded_error`.

    Branched on rather than the status code, because a mid-stream capacity error
    arrives inside a 200 and is therefore indistinguishable from success by status.
    """
    value = _error_body(exc).get("type")
    return value if isinstance(value, str) else ""


def _api_message(exc: anthropic.APIStatusError) -> str:
    """The provider's error text, defensively — an error path must not itself raise."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"][:500]
    return str(exc)[:500]


def _to_wire(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Convert Cortex messages to the Messages API shape.

    Tool results become a user turn carrying one `tool_result` block per request id,
    which is what the API expects — several results in one turn rather than one turn
    each. Splitting them teaches the model to stop making parallel calls.
    """
    wire: list[dict[str, Any]] = []
    for message in messages:
        if message.tool_results:
            blocks: list[dict[str, Any]] = [
                {
                    "type": "tool_result",
                    "tool_use_id": request_id,
                    "content": result,
                }
                for request_id, result in message.tool_results.items()
            ]
            # A text block may ride along after the results, which is how a per-turn instruction
            # reaches the model -- W&D's width schedule is "a user message instruction at each
            # step". The text goes *after* the results deliberately: it is guidance about the next
            # turn, and putting it before the observations would separate each tool_result from
            # the tool_use it pairs with in the reader's eye, which is the shape F-12 came from.
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            wire.append({"role": "user", "content": blocks})
            continue
        if message.tool_requests:
            # Replayed as blocks. The API pairs each tool_result with the tool_use of
            # the same id in the previous assistant turn; sending the prose alone
            # produces "unexpected tool_use_id found in tool_result blocks".
            content: list[dict[str, Any]] = []
            if message.content:
                content.append({"type": "text", "text": message.content})
            content.extend(
                {
                    "type": "tool_use",
                    "id": request.id,
                    "name": request.name,
                    "input": request.arguments,
                }
                for request in message.tool_requests
            )
            wire.append({"role": message.role, "content": content})
            continue
        wire.append({"role": message.role, "content": message.content})
    return wire


def _usage(response: Any) -> Usage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return Usage()
    return Usage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        # Absent on responses that had nothing cacheable, so read defensively rather
        # than assumed present — a missing field must read as zero, not raise.
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )


def _refusal_category(response: Any) -> str:
    """Read the refusal category defensively.

    `stop_details` is informational and may be absent or null even on a refusal, so
    it is never the thing branched on — only reported.
    """
    details = getattr(response, "stop_details", None)
    return str(getattr(details, "category", None) or "unspecified")
