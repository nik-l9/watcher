"""Tool framework.

A Tool is a connector to one external system. A Capability is one operation on it.
Every capability declares its parameters as a JSON schema and returns typed JSON —
never markdown, never prose. Prose is generated once, in the report layer, from
evidence. A tool that returned prose would make grounding unverifiable, because
there would be nothing structured left to cite.

Three properties hold for every capability, enforced here rather than by
convention:

  1. Read-only. V1 ships no write capability at all, so "humans approve" is
     guaranteed by the absence of destructive tools rather than by a prompt an
     agent could talk itself out of.
  2. Tenant-scoped. Every invocation takes a TenantContext, and credentials are
     resolved per tenant.
  3. Audited. Every invocation writes a ToolCall row whether it succeeded or not,
     and every successful one writes immutable Evidence that reports cite by id.

Connectors implement `_invoke` and never touch the database. Auditing, evidence
capture, timing and error handling live in the executor so no connector can forget
them.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from cortex.db.models import CredentialProvider
from cortex.tenancy.context import TenantContext


class ToolError(Exception):
    """Base for tool failures. Never carries credential material."""


class CapabilityNotFound(ToolError):
    pass


class ToolNotFound(ToolError):
    pass


class CredentialMissing(ToolError):
    """The tenant has not connected this provider."""


class UpstreamError(ToolError):
    """The third-party API failed. Retryable at the caller's discretion."""


class RateLimited(UpstreamError):
    """Upstream asked us to slow down. Carries a hint when the API provides one."""

    def __init__(self, message: str, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class TenantRateLimited(RateLimited):
    """*Cortex* asked the caller to slow down, not the upstream.

    A subclass of `RateLimited` so the investigation loop's existing handling applies
    unchanged — it already knows to route around a rate limit and report the gap. The
    distinct type exists so the two are separable in the audit trail: "HubSpot throttled us"
    and "we throttled ourselves on this tenant's behalf" call for completely different
    responses, and a single message would make the second look like the first.
    """


class InvalidParams(ToolError):
    """Parameters failed schema validation. Not retryable — the agent must fix them."""


class Freshness(enum.StrEnum):
    """Where an observation came from.

    Reports must disclose staleness in their confidence section, so this travels
    with the result rather than being inferred later.
    """

    LIVE = "live"
    SYNCED = "synced"


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What a capability returns.

    `payload` is the structured observation. `source_ref` is a human-followable
    pointer — a permalink, a SQL string, a commit sha — rendered in the report's
    Sources section so a reader can independently verify the claim.
    """

    payload: dict[str, Any]
    source_ref: str | None = None
    freshness: Freshness = Freshness.LIVE
    # Non-secret upstream metadata: page counts, sampling notices, quota state.
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.payload, dict):
            raise TypeError(
                f"ToolResult.payload must be a dict of structured data, got "
                f"{type(self.payload).__name__}. Tools return JSON, never prose."
            )


#: Declared when a capability's result cannot be empty in a meaningful sense.
#:
#: The value is the reason, so the decision is visible in review rather than being an
#: omission that looks like an oversight.
NEVER_EMPTY = "never-empty"


@dataclass(frozen=True, slots=True)
class Capability:
    """One operation on one tool.

    `params_schema` is a JSON Schema object. It is handed to the LLM verbatim as
    the tool-call schema and is also used to validate arguments before dispatch,
    so the model gets one description of the contract rather than two that can
    drift.

    **`result_key` is mandatory, and it exists because of a bug we have now shipped four
    times.** Slack returned nothing for an over-narrow query; GitHub returned nothing for an
    environment that does not exist; PostHog returned nothing because the wrong project was
    queried; Slack returned empty text because apps post with Block Kit. Every one of those
    handed the analyst an empty result that was indistinguishable from a real absence, and
    every one produced a confident, properly cited, wrong conclusion.

    Naming the pattern in four commit messages did not stop the fifth. So a capability must
    now declare which payload key holds its results — and the executor uses that to mark an
    empty observation as empty, in the text the analyst reads, every time, without the tool
    author having to remember. A capability whose result genuinely cannot be empty declares
    `NEVER_EMPTY` with a reason instead, which is a decision rather than a silence.
    """

    name: str
    description: str
    params_schema: dict[str, Any]
    handler: Callable[..., Awaitable[ToolResult]]
    read_only: bool = True
    #: The payload key holding this capability's results, or `NEVER_EMPTY` plus a reason.
    result_key: str = ""

    #: True when this capability enumerates *what exists* rather than measuring it: the
    #: repositories, projects, channels or named queries a tenant can reach.
    #:
    #: Declared rather than inferred from the name, and load-bearing rather than descriptive.
    #: `Investigator` runs every declared discovery capability once, before the first step, and
    #: puts the results in the opening prompt.
    #:
    #: **Why the survey is deterministic instead of a tool the analyst may choose to call.**
    #: Asked which PR shipped most recently to the company website, the analyst searched
    #: `acme/acme` -- because that was the only repository name it had ever seen -- and
    #: honestly reported that it could not confirm an answer. `acme/company-website` exists,
    #: and the ingest already knew about it. Nothing in the tool surface could have told the
    #: analyst that, since all eight GitHub capabilities take a repository name as a parameter.
    #:
    #: Adding a capability it *might* call would be half a fix, and the literature is specific
    #: about why. "Agents Explore but Agents Ignore" (arXiv 2604.17609) measures the gap between
    #: discovering something and acting on it: on Terminal-Bench agents discovered solutions
    #: 78.6-81.2% of the time and interacted with them 37.1-50.3%; in AppWorld one model observed
    #: documentation naming the solution in **97.54%** of attempts and called it in **0.53%**.
    #:
    #: Its tool-availability finding runs the *opposite* way to the obvious reading, and is worth
    #: stating precisely: richer tooling made agents investigate **less** -- a bash-only scaffold
    #: roughly doubled interaction against one with a structured editor. So "add a discovery tool"
    #: is not merely insufficient, it is the kind of change that paper found can suppress the
    #: behaviour being sought.
    #:
    #: Its second factor bears directly on a decision already taken here. Curiosity scales with
    #: test-time compute: interaction@1 tripled from 11% at low reasoning to 37% at high. Cortex
    #: runs `effort="low"` on measured evidence (6/6 against 4/6, thirty times cheaper), so the
    #: analyst is at the low end of exactly the axis that governs whether it investigates an
    #: unexpected observation. That is an argument for establishing the environment mechanically
    #: rather than for buying curiosity back at thirty times the price.
    #:
    #: "Look Before You Leap" (arXiv 2605.16143) supplies the shape -- explore on a fixed budget,
    #: synthesise a summary, inject it, then act. Its own numbers do **not** show that shape
    #: helping an untrained model: explore-then-act moved ALFWorld 54.4% -> 54.1% and 30.9% ->
    #: 28.7% zero-shot, gaining only with exploration-aware training (+1.6, +2.2). So the case
    #: for the survey here is not borrowed from that result. It rests on a measurement of our
    #: own: the question above went from "I could not confirm this" to a correct, cited answer,
    #: and from 132 seconds to 52.
    #:
    #: So the environment is established as a fact rather than left to curiosity. It is the same
    #: treatment `registry_for_tenant` gives the toolset: resolve deterministically what the
    #: analyst would otherwise have to guess.
    discovery: bool = False

    def is_empty(self, payload: dict[str, Any]) -> bool:
        """Whether this result found nothing.

        Read from the declared key rather than guessed from the payload's shape. Guessing
        was the first design and it was wrong in both directions: a payload with a `count`
        of 0 and a populated `environments_available` list is *informative* emptiness, and a
        payload with rows under an unexpected key looked empty when it was not.
        """
        if self.result_key in ("", NEVER_EMPTY) or self.result_key.startswith(NEVER_EMPTY):
            return False
        value = payload.get(self.result_key)
        if value is None:
            # The declared key is absent entirely. Treated as empty, because a result that
            # does not contain the thing it promised has not found it — and a silent False
            # here would be exactly the hole this whole mechanism exists to close.
            return True
        if isinstance(value, dict | list | tuple | str):
            return len(value) == 0
        return False

    def empty_hint(self, payload: dict[str, Any]) -> str:
        """What the tool itself can say about why nothing came back.

        Tools already return this — `environments_available`, `note`, the PostHog project
        allowlist in an error. The value here is collecting it in one place so the analyst
        reads it in the same sentence as the emptiness, rather than having to notice a field
        several lines down a JSON blob.
        """
        parts: list[str] = []
        for key in ("note", "environments_available", "total_available", "total_matching"):
            value = payload.get(key)
            if value in (None, "", [], {}):
                continue
            parts.append(f"{key}={value}")
        return "; ".join(parts)

    def __post_init__(self) -> None:
        if not self.read_only:
            # V1 has no write path. This is the enforcement point: a write
            # capability cannot be declared, so it cannot be invoked, so a
            # destructive action cannot happen by prompt injection or by mistake.
            raise ValueError(
                f"capability {self.name!r} declares read_only=False; V1 integrations "
                "are read-only by design. See the non-negotiables in README.md."
            )
        if self.params_schema.get("type") != "object":
            raise ValueError(f"capability {self.name!r} params_schema must be an object schema")
        if self.params_schema.get("additionalProperties") is not False:
            # Rejecting unknown params surfaces a hallucinated argument as a
            # validation error instead of silently ignoring it.
            raise ValueError(
                f"capability {self.name!r} params_schema must set "
                '"additionalProperties": false so hallucinated arguments are rejected'
            )
        if not self.result_key:
            # Mandatory, because four separate bugs have now shipped where an empty result
            # was indistinguishable from a real absence. The author of a capability is the
            # only person who knows which key holds its results; asking them once here is
            # cheaper than the analyst reporting "nothing happened" about data it never saw.
            raise ValueError(
                f"capability {self.name!r} must declare result_key: the payload key holding "
                f"its results, so an empty result can be marked as empty. If it genuinely "
                f'cannot be empty, pass result_key=f"{{NEVER_EMPTY}}: <reason>".'
            )


class Tool(ABC):
    """A connector to one external system.

    Subclasses declare `name`, the credential provider they need, and their
    capabilities. They implement each capability as an async method returning a
    ToolResult and never write to the database — see ToolExecutor.
    """

    #: Stable identifier used in evidence rows and in LLM tool names.
    name: str
    #: Which credential this tool needs. None for tools requiring no credential.
    provider: CredentialProvider | None

    #: The credential label this tool resolves against, overriding the caller's.
    #:
    #: **Why a tool may pin its own.** A label normally selects *which account* an
    #: investigation runs as -- two GA4 properties, say -- so it is chosen per run and
    #: applies to every tool. That breaks down for MCP, where one provider covers every
    #: server and each server is a separate credential under its own label. Two servers
    #: connected at once would both look up the run's single label and both miss. A tool
    #: built from a specific credential therefore carries that credential's label with it,
    #: and the run-wide choice applies only to tools that have not pinned one.
    credential_label: str | None = None

    def __init__(self) -> None:
        if not getattr(self, "name", None):
            raise ValueError(f"{type(self).__name__} must declare a name")
        self._capabilities = {c.name: c for c in self.capabilities()}
        if not self._capabilities:
            raise ValueError(f"tool {self.name!r} declares no capabilities")

    @abstractmethod
    def capabilities(self) -> list[Capability]:
        """Every operation this tool exposes."""

    def capability(self, name: str) -> Capability:
        try:
            return self._capabilities[name]
        except KeyError:
            available = ", ".join(sorted(self._capabilities))
            raise CapabilityNotFound(
                f"tool {self.name!r} has no capability {name!r}; available: {available}"
            ) from None

    @property
    def capability_names(self) -> list[str]:
        return sorted(self._capabilities)

    def llm_tool_specs(self) -> list[dict[str, Any]]:
        """Tool-call specs for the LLM, one per capability.

        Flattened to `tool__capability` rather than nested, because a single flat
        list of concrete operations produces markedly better tool selection than
        asking a model to pick a tool and then an operation.
        """
        return [
            {
                "name": f"{self.name}__{c.name}",
                "description": c.description,
                "input_schema": c.params_schema,
            }
            for c in (self._capabilities[n] for n in self.capability_names)
        ]


class ToolRegistry:
    """The set of tools available to an employee."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            available = ", ".join(sorted(self._tools)) or "none"
            raise ToolNotFound(f"no tool named {name!r}; registered: {available}") from None

    def resolve(self, qualified_name: str) -> tuple[Tool, Capability]:
        """Resolve a flattened `tool__capability` name from an LLM tool call."""
        tool_name, sep, capability_name = qualified_name.partition("__")
        if not sep:
            raise ToolNotFound(
                f"{qualified_name!r} is not a qualified tool call; expected 'tool__capability'"
            )
        tool = self.get(tool_name)
        return tool, tool.capability(capability_name)

    @property
    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    def llm_tool_specs(self) -> list[dict[str, Any]]:
        return [spec for name in self.tool_names for spec in self._tools[name].llm_tool_specs()]

    def discovery_capabilities(self) -> list[str]:
        """Qualified names of every capability that enumerates what exists.

        Sorted, so the survey runs in the same order every time -- a transcript that reorders
        between runs makes two investigations of the same question hard to compare, and defeats
        prompt caching on the one prefix that is identical across every investigation.
        """
        return sorted(
            f"{name}__{capability}"
            for name in self.tool_names
            for capability in self._tools[name].capability_names
            if self._tools[name].capability(capability).discovery
        )

    def required_providers(self) -> set[CredentialProvider]:
        """Which credentials a tenant must connect for this registry to be usable."""
        return {t.provider for t in self._tools.values() if t.provider is not None}


@dataclass(frozen=True, slots=True)
class ToolContext:
    """What a capability is given at invocation time.

    Carries the tenant and the decrypted credential. The credential is a secret:
    it must not be logged, echoed into a ToolResult, or persisted.
    """

    tenant: TenantContext
    credential: str | None = None
    credential_metadata: dict[str, Any] = field(default_factory=dict)
    #: Metadata of the tenant's *other* credentials for this provider, keyed by label.
    #:
    #: No secrets: metadata only. It exists because one provider can hold several credentials
    #: doing different jobs, and a capability sometimes needs to know something about a
    #: credential it is not using. Slack is the case that forced it -- Cortex reads with one
    #: token and posts with another, so the reader could not recognise the poster's own
    #: messages and cited its own earlier report as independent corroboration.
    peer_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
