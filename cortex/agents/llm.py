"""LLM provider abstraction.

Multi-provider from day one, per the plan, behind one narrow interface. The
interface is deliberately small — a chat call with optional tools, and a
structured-output call — because a wide abstraction over several providers ends up
matching none of them.

Two properties matter more than provider coverage:

  - **Token accounting.** Every call reports usage, so an investigation can be
    stopped when it exceeds its budget. Without this a runaway loop is discovered on
    an invoice.
  - **Deterministic testing.** A recorded provider replays fixed responses, so the
    investigation loop, the gate and the verifier are all testable without a network
    call or an API key. Every M2 test runs against it.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


class LLMError(Exception):
    """Provider failure. Never carries an API key."""


class LLMAuthenticationFailed(LLMError):
    """The credential is wrong, absent, or not entitled to this model.

    Split from a plain `LLMError` because the loop's recovery is right for one and wrong for the
    other. A transient failure with evidence already gathered should stop and report what was
    found -- discarding real work over a blip is worse than a partial answer. A rejected
    credential will reject every later call too, so the same policy spends the step budget
    reaching the same place and then surfaces "the report could not be drafted", which sends
    whoever reads it to look at the schema instead of at their key.

    Observed exactly that way: a dead key produced a 17-second investigation whose reported
    cause was drafting.
    """


class LLMRefusedStructure(LLMError):
    """The model would not produce output matching the requested schema."""


class LLMOutputTruncated(LLMRefusedStructure):
    """The model ran out of output room mid-structure.

    Split from a plain refusal because the two want opposite responses. A refusal means
    the model will not answer this at all, so retrying is waste. Truncation means it was
    answering and ran out of room -- retrying with less to write can succeed, which matters
    because the drafting call happens *after* every tool call has been paid for, so the
    alternative is discarding a completed investigation over its final paragraph.

    Subclasses `LLMRefusedStructure` so callers that only care that no structure arrived
    keep working unchanged.
    """


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    #: Input tokens served from a prompt cache, and those written to it.
    #:
    #: Carried because prompt caching cannot be evaluated without them. The loop resends
    #: its whole transcript every step, so the cacheable prefix grows with the
    #: investigation — but "we enabled caching" is a claim, and `cache_read_input_tokens`
    #: rising while `input_tokens` falls is the measurement. Priced differently too: a
    #: cache read costs a fraction of a fresh input token, so a cost model that ignores
    #: these overstates the bill once caching is on.
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Share of input served from cache. 0.0 when nothing was cacheable."""
        considered = self.input_tokens + self.cache_read_input_tokens
        return self.cache_read_input_tokens / considered if considered else 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=(self.cache_read_input_tokens + other.cache_read_input_tokens),
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
        )


@dataclass(frozen=True, slots=True)
class ToolRequest:
    """A tool call the model wants to make.

    `name` is the flattened `tool__capability` form the registry resolves.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str = ""
    tool_requests: list[ToolRequest] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "end_turn"

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_requests)


@dataclass(frozen=True, slots=True)
class Message:
    """One turn of conversation.

    `tool_results` carries observations back to the model, keyed by the request id
    the model supplied.

    `tool_requests` records what an assistant turn asked for. It has to be part of the
    transcript, not just of the response: a provider requires every tool_result to
    have a matching request in the preceding assistant turn, so a history that keeps
    only the assistant's prose leaves the results orphaned and the next call fails.
    """

    role: str
    content: str = ""
    tool_results: dict[str, str] = field(default_factory=dict)
    tool_requests: tuple[ToolRequest, ...] = ()


class LLM(ABC):
    """A provider. Implementations are thin — no retry policy, no prompt shaping."""

    #: Which model this provider talks to. Declared on the interface because callers need it
    #: for reasons unrelated to calling: pricing is per model, and a spend figure without it
    #: silently reprices itself the next time the default changes. Defaults to `None` so a
    #: scripted provider in a test need not pretend to be a model.
    model: str | None = None

    #: Identifies the provider and model in audit records.
    name: str

    @abstractmethod
    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """One turn. May return text, tool requests, or both."""

    @abstractmethod
    async def structured(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        schema: dict[str, Any],
        max_tokens: int = 8192,
        timeout: float | None = None,
    ) -> tuple[dict[str, Any], Usage]:
        """Output conforming to a JSON schema.

        Used for the report draft and every verifier verdict. Returning parsed JSON
        rather than text is what lets the gate operate on structure — the guarantee
        the whole grounding story rests on.

        `timeout` overrides the provider's default deadline for this one call. The two
        callers have genuinely different durations — a verifier verdict is a sentence,
        a report draft is thousands of tokens of synthesis — and a single constant
        either cuts the draft off mid-thought or lets a hung verdict stall the run.
        """


class RecordedLLM(LLM):
    """A provider that replays scripted responses, for tests and eval fixtures.

    Not a mock in the usual sense: it implements the real interface and records what
    it was asked, so a test can assert on the prompt as well as the outcome. Every
    M2 test uses it, which is what makes them deterministic and free.
    """

    name = "recorded"

    def __init__(
        self,
        completions: Sequence[LLMResponse] | None = None,
        structured_outputs: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self._completions = list(completions or [])
        self._structured = list(structured_outputs or [])
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        self.calls.append(
            {
                "kind": "complete",
                "system": system,
                "messages": list(messages),
                "tool_names": [t["name"] for t in (tools or [])],
            }
        )
        if not self._completions:
            # An exhausted script means the loop ran more turns than the test
            # anticipated, which is a test failure worth being loud about rather
            # than a silent empty response.
            raise LLMError("RecordedLLM has no completion left to replay")
        return self._completions.pop(0)

    async def structured(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        schema: dict[str, Any],
        max_tokens: int = 8192,
        timeout: float | None = None,
    ) -> tuple[dict[str, Any], Usage]:
        self.calls.append(
            {
                "kind": "structured",
                "system": system,
                "messages": list(messages),
                # Recorded so a test can assert the draft asked for the longer
                # deadline; an override nothing observes is an override that can
                # silently stop being passed.
                "timeout": timeout,
            }
        )
        if not self._structured:
            raise LLMError("RecordedLLM has no structured output left to replay")
        payload = self._structured.pop(0)
        return payload, Usage(input_tokens=100, output_tokens=len(json.dumps(payload)) // 4)
