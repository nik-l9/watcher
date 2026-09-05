"""Choosing the adapter, in one place rather than six.

Six call sites named `AnthropicLLM` directly -- both CLI paths, the eval harness, the worker and
two benchmarks. Adding a provider by editing six sites is how five of them support it and the
sixth silently does not, which surfaces as "it works from the CLI but not from Slack".
"""

from __future__ import annotations

import pytest

from cortex.agents.anthropic_llm import AnthropicLLM
from cortex.agents.models import ModelCard, UnknownModel
from cortex.agents.openai_llm import OpenAICompatLLM
from cortex.agents.provider import build_llm


class TestTheAdapterComesFromTheModel:
    def test_a_claude_model_gets_the_anthropic_adapter(self) -> None:
        assert isinstance(build_llm(model="claude-sonnet-5"), AnthropicLLM)

    def test_the_default_still_works_with_no_arguments(self) -> None:
        """The worker calls it this way, so a regression here stops Slack answering."""
        assert isinstance(build_llm(), AnthropicLLM)

    def test_an_openai_model_gets_the_compatible_adapter(self) -> None:
        llm = build_llm(model="gpt-5.2", base_url="http://localhost:11434/v1")
        assert isinstance(llm, OpenAICompatLLM)

    def test_the_provider_is_not_a_separate_flag(self) -> None:
        """A model id and the endpoint serving it are one fact. Two settings would allow a
        combination that cannot work -- an OpenAI endpoint asked for a Claude model -- and
        nothing would catch it before the request."""
        import inspect

        assert "provider" not in inspect.signature(build_llm).parameters


class TestDeclaringAModelThisProjectHasNeverHeardOf:
    def test_a_card_admits_a_proxy_model(self) -> None:
        """A proxy reaches hundreds of models. Enumerating them is not possible and guessing
        capabilities from a name is what `models.py` refuses to do, so the caller declares."""
        card = ModelCard(
            id="qwen/qwen3-max",
            context_window=256_000,
            max_output_tokens=32_000,
            provider="openai_compat",
            constrained_json_schema=False,
        )
        llm = build_llm(model="qwen/qwen3-max", card=card, base_url="https://openrouter.ai/api/v1")
        assert isinstance(llm, OpenAICompatLLM)
        assert llm.card.constrained_json_schema is False

    def test_without_a_card_an_unknown_model_is_still_refused(self) -> None:
        with pytest.raises(UnknownModel):
            build_llm(model="qwen/qwen3-max")


class TestEffortIsAnAnthropicKnob:
    def test_it_is_dropped_rather_than_sent_to_a_provider_without_one(self) -> None:
        """Silently, on purpose. The alternative is every caller knowing which knobs each
        provider takes, which is the knowledge this factory exists to hold -- and sending the
        wrong one is a 400 at drafting time."""
        llm = build_llm(model="gpt-5.2", effort="high", base_url="http://localhost:11434/v1")
        assert isinstance(llm, OpenAICompatLLM)
        assert not hasattr(llm, "_effort")
