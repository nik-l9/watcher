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

Adapted from OpenHands' `model_features.py` (feature flags matched against model names)
and `verified_models.py` (a curated list per provider). Two deliberate differences:

  - **Exact ids, not name patterns.** Their registry has to cope with hundreds of models
    from dozens of providers reached through LiteLLM, so pattern matching is the only
    option. We run one provider and a handful of models, where a pattern's failure mode —
    a new model quietly matching a rule written for an older one — is pure downside.
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
