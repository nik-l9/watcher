"""Choosing the adapter for a model, in one place.

**Why a factory rather than a constructor at each call site.** There are six of them -- both CLI
paths, the eval harness, the worker and two benchmarks -- and each named `AnthropicLLM` directly.
Adding a provider by editing six sites is how five of them end up supporting it and the sixth
silently does not, which surfaces as "the model works from the CLI but not from Slack".

The choice is read from the model's own card rather than from a separate provider flag. A model
id and the endpoint that serves it are one fact, so splitting them into two settings creates a
combination that cannot work: `--provider openai --model claude-sonnet-5` is a 404 on an endpoint
that never had that model, and nothing would catch it before the request.
"""

from __future__ import annotations

from cortex.agents.anthropic_llm import DEFAULT_EFFORT, DEFAULT_MODEL, AnthropicLLM
from cortex.agents.llm import LLM
from cortex.agents.models import ModelCard, card_for
from cortex.agents.openai_llm import OpenAICompatLLM

__all__ = ["build_llm"]


def build_llm(
    *,
    model: str | None = None,
    effort: str = DEFAULT_EFFORT,
    card: ModelCard | None = None,
    base_url: str | None = None,
) -> LLM:
    """The provider that can serve `model`.

    `card` lets a caller declare a model this project has never heard of, which is how an
    OpenAI-compatible proxy reaches its hundreds of models without this file guessing
    capabilities from a name. `models.py` still refuses an unknown model when no card is given:
    being told is not the same as inferring.

    `effort` is Anthropic's reasoning control and is dropped for providers that have no
    equivalent. Silently, on purpose -- the alternative is that every caller has to know which
    knobs each provider takes, which is the knowledge this function exists to hold.
    """
    resolved = model or DEFAULT_MODEL
    declared = card if card is not None else card_for(resolved)
    if declared.provider == "openai_compat":
        return OpenAICompatLLM(model=resolved, card=declared, base_url=base_url)
    return AnthropicLLM(model=resolved, effort=effort)
