"""What each model can actually do.

`ALLOWED_MODELS` was a bare `frozenset({"claude-sonnet-5"})` and every fact about that
model lived somewhere else: as a hardcoded `{"type": "adaptive"}` in the request builder,
as prose in a module docstring, as a `DRAFT_MAX_TOKENS` constant chosen by trial, and as a
price in `cortex/db/spend.py`. A set of names says a model is permitted; it does not say
anything the code needs in order to call it correctly.

**The failure that motivates this.** Widening the set is a one-word change, and every
request-shape assumption we make is silently wrong for most of the models somebody might
add:

  - `thinking: {"type": "adaptive"}` is a **400** on pre-4.6 models, which want
    `{"type": "enabled", "budget_tokens": N}`. The reverse is also a 400 on Sonnet 5.
  - `output_config.effort` does not exist on older models.
  - `temperature` / `top_p` / `top_k` are **rejected** on Opus 5 and Sonnet 5, and
    accepted everywhere else.
  - Haiku 4.5 has a **200K** context window against 1M for the rest, so a transcript
    that is comfortable on Sonnet 5 overflows on Haiku — and the request that overflows
    is the drafting call, at the end, after all the tokens are spent.
  - Output ceilings differ, so `max_tokens=32768` is a 400 on a model that caps lower.

Each of those fails at the moment a report is drafted, which is the most expensive moment
in an investigation to discover a configuration error. Declaring the facts turns all of
them into a `ValueError` at construction instead.

Two choices worth stating, because the obvious implementation makes the opposite one:

  - **Exact ids, not name patterns.** A registry spanning hundreds of models from dozens of
    providers has no option but to match patterns. This one runs a single provider and a
    handful of models, where a pattern's failure mode — a new model quietly matching a rule
    written for an older one — is pure downside.
  - **No `supports_function_calling` flag.** Every model here has it, and a flag that is
    always true documents nothing while inviting a caller to branch on it.

Pricing stays in `cortex/db/spend.py`. That is not an oversight: this table describes how
to *call* a model, and that one prices a customer's bill. A test asserts every permitted
model appears in both, so the two cannot drift apart unnoticed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: How a model is told to think.
#:
#: "adaptive" — `thinking: {"type": "adaptive"}`, depth steered by `output_config.effort`.
#: "budget"   — `thinking: {"type": "enabled", "budget_tokens": N}`, the pre-4.6 form.
#: "none"     — the parameter is not sent at all.
ThinkingStyle = Literal["adaptive", "budget", "none"]

#: Adapters that exist. Not a free-form string: a typo in a card would otherwise route a
#: request to a provider that is never constructed, and the failure would surface as a model
#: that simply never answers.
Provider = Literal["anthropic", "openai_compat"]


class UnknownModel(ValueError):
    """A model with no capability declaration.

    Refused rather than defaulted. A default would have to guess a request shape, and
    every wrong guess is a 400 at drafting time — the one place a failure costs a whole
    investigation's tokens.
    """


@dataclass(frozen=True, slots=True)
class ModelCard:
    """Declared facts about one model. Not inferred from its name."""

    id: str

    #: Total tokens the model will accept in one request, input plus output. Used to
    #: decide when a transcript has to be condensed rather than resent.
    context_window: int

    #: The largest `max_tokens` this model accepts.
    max_output_tokens: int

    thinking: ThinkingStyle = "adaptive"

    #: Whether `output_config.effort` is accepted.
    supports_effort: bool = True

    #: Whether `temperature` / `top_p` / `top_k` are accepted. False on the current
    #: frontier models, which reject them outright.
    supports_sampling_params: bool = False

    #: Whether `output_config.format` with a JSON schema is accepted. The grounding gate
    #: depends on this: without it the report would have to be parsed out of prose.
    supports_structured_outputs: bool = True

    #: Whether the top-level `cache_control` form is accepted. Measured at 98-100% hit
    #: rate on our transcript shape, and worth roughly an order of magnitude on the bill.
    supports_prompt_cache: bool = True

    #: False for a model that is described here but must not be constructed. Cortex runs
    #: exactly one model as a cost decision (see `DEFAULT_MODEL`); the others are declared
    #: so that permitting one is a single flag rather than research.
    permitted: bool = False

    #: Which adapter can call this model. The request shape differs per provider, not only
    #: per model, so the two facts have to travel together: a card naming an OpenAI-shaped
    #: model reaching the Anthropic adapter is a 404 on an endpoint that does not exist.
    provider: Provider = "anthropic"

    #: Whether the provider *guarantees* the response matches the JSON schema, as opposed to
    #: merely promising valid JSON.
    #:
    #: The distinction is load-bearing and is why `supports_structured_outputs` is not enough
    #: on its own. Constrained decoding cannot emit a non-conforming document. A JSON-object
    #: mode can, and often does — a missing required field, an enum value invented, a number
    #: as a string. Both are usable here, because the drafting call already retries a draft it
    #: cannot parse, but only one of them makes that retry rare. A card claiming constraint it
    #: does not have turns a routine repair into an unexplained drafting failure rate.
    constrained_json_schema: bool = True

    def room_for_output(self, prompt_tokens: int) -> int:
        """Output tokens that still fit alongside a prompt this size.

        Never negative, and never above the model's own ceiling. Returning a number
        rather than raising because the caller's options depend on the number: a small
        positive value means condense, zero means the prompt alone does not fit.
        """
        return max(0, min(self.max_output_tokens, self.context_window - prompt_tokens))


#: Every model we have facts for, permitted or not.
#:
#: The 1M context windows and the sampling-parameter rejection are properties of the
#: current generation rather than of any one model, but they are written out per model
#: anyway: the moment one is written as a shared default, a model that differs inherits
#: the wrong value silently.
CARDS: dict[str, ModelCard] = {
    "claude-sonnet-5": ModelCard(
        id="claude-sonnet-5",
        context_window=1_000_000,
        max_output_tokens=64_000,
        permitted=True,
    ),
    "claude-opus-5": ModelCard(
        id="claude-opus-5",
        context_window=1_000_000,
        max_output_tokens=64_000,
    ),
    "claude-opus-4-8": ModelCard(
        id="claude-opus-4-8",
        context_window=1_000_000,
        max_output_tokens=64_000,
    ),
    "claude-sonnet-4-6": ModelCard(
        id="claude-sonnet-4-6",
        context_window=1_000_000,
        max_output_tokens=64_000,
    ),
    "claude-haiku-4-5": ModelCard(
        id="claude-haiku-4-5",
        # A fifth of the others. The reason this table exists: a transcript that is
        # comfortable on Sonnet 5 does not fit here, and nothing in the loop currently
        # knows that.
        context_window=200_000,
        max_output_tokens=64_000,
        # Pre-4.6: the adaptive form is rejected, and effort does not exist.
        thinking="budget",
        supports_effort=False,
        supports_sampling_params=True,
    ),
    "claude-fable-5": ModelCard(
        id="claude-fable-5",
        context_window=1_000_000,
        max_output_tokens=64_000,
    ),
    # ---------------------------------------------------------------- OpenAI-compatible
    #
    # Two entries, not a catalogue. This project runs one model as a cost decision, and these
    # exist so that someone holding an OpenAI key can run it at all -- which was impossible
    # before, and is the single largest barrier to anyone adopting this.
    #
    # A model reached through a proxy is deliberately *not* listed. There are hundreds, their
    # capabilities differ, and guessing from a name is exactly what `models.py` refuses to do:
    # pass `OpenAICompatLLM(card=...)` and declare it. Being told is not the same as inferring.
    "gpt-5.2": ModelCard(
        id="gpt-5.2",
        context_window=400_000,
        max_output_tokens=128_000,
        provider="openai_compat",
        # `output_config.effort` is Anthropic's knob and does not exist here; this provider's
        # reasoning control is a different parameter, and sending the wrong one is a 400.
        supports_effort=False,
        # Sampling parameters are accepted, unlike the current Anthropic frontier -- but nothing
        # here sends them, so this records the fact rather than acting on it.
        supports_sampling_params=True,
        # False, and measured rather than assumed. `response_format.json_schema` with
        # `strict: true` is a real guarantee -- but strict mode requires every property of every
        # object to be listed in `required`, and Cortex's report schema has nine objects with
        # optional fields (`confidence`, `charts`, `cause_at`, ...). Sending it strict is a 400
        # at the drafting call. So this asks for JSON-object mode, and the existing repair
        # attempt carries the occasional non-conforming draft. Flipping this to True needs the
        # schema made total first, which changes what a drafted report may omit.
        constrained_json_schema=False,
        # Caching is automatic on this provider rather than requested per block, so there is no
        # `cache_control` to send. The usage figures still report the cached share.
        supports_prompt_cache=False,
        permitted=True,
    ),
    # Declared, not permitted. The cheaper tier is the obvious candidate for the default --
    # `DEFAULT_EFFORT` was chosen the same way, and less turned out to be better on every
    # dimension the suite measures -- but that was a measurement, and this has not been
    # measured. Permitting it is one flag away from an eval run that says so.
    "gpt-5.2-mini": ModelCard(
        id="gpt-5.2-mini",
        context_window=400_000,
        max_output_tokens=128_000,
        provider="openai_compat",
        supports_effort=False,
        supports_sampling_params=True,
        constrained_json_schema=False,
        supports_prompt_cache=False,
    ),
}

#: Models this provider will construct, derived from the cards rather than maintained
#: beside them. A second list is a list that disagrees.
ALLOWED_MODELS = frozenset(model_id for model_id, card in CARDS.items() if card.permitted)


def card_for(model: str) -> ModelCard:
    """The capability declaration for `model`, or `UnknownModel`."""
    card = CARDS.get((model or "").strip().lower())
    if card is None:
        raise UnknownModel(
            f"no capability declaration for model {model!r}. Add a ModelCard to "
            "cortex/agents/models.py rather than calling an undeclared model: the "
            "request shape (thinking style, effort, sampling parameters) differs "
            "between generations and a wrong guess is a 400 at drafting time."
        )
    return card
