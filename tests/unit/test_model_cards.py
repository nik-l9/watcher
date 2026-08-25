"""What each model can do, and whether the request shape follows from it.

The tests worth having here are not "does the table contain Sonnet 5". They are about the
one change this table exists to make safe: **permitting a second model.** That is a
one-word edit, and every request-shape assumption in the provider is silently wrong for
some model somebody might add — the adaptive-thinking form is a 400 on pre-4.6 models, the
reverse is a 400 on Sonnet 5, `effort` does not exist on older models, and Haiku's context
window is a fifth of the rest.

Each of those failures lands at the drafting call, at the end of an investigation, after
all its tokens are already spent. So the assertions below check that the *shape* is derived
from the card rather than hardcoded, using the declared-but-unpermitted cards as the test
subjects — which is also what makes those cards worth carrying.
"""

from __future__ import annotations

import pytest

from cortex.agents.anthropic_llm import ALLOWED_MODELS, DEFAULT_MODEL, AnthropicLLM
from cortex.agents.models import CARDS, ModelCard, UnknownModel, card_for
from cortex.db.spend import RATES


def _llm(model: str) -> AnthropicLLM:
    """A provider for a model that may not be permitted.

    The permission check is a cost decision and is tested separately; these tests are
    about request shape, so the card is installed directly. `client` is a sentinel: no
    request is made.
    """
    llm = AnthropicLLM.__new__(AnthropicLLM)
    llm.card = card_for(model)
    llm._model = model
    llm._effort = "low"
    llm.name = f"anthropic:{model}"
    return llm


class TestTheTable:
    def test_the_default_model_is_permitted(self) -> None:
        assert DEFAULT_MODEL in ALLOWED_MODELS

    def test_exactly_one_model_is_permitted(self) -> None:
        """A cost decision, stated in `DEFAULT_MODEL`'s docstring. This test is what makes
        widening it a deliberate act rather than a side effect: adding a permitted model
        fails here and the failure is where the reasoning lives."""
        assert ALLOWED_MODELS == frozenset({"claude-sonnet-5"})

    def test_allowed_models_is_derived_from_the_cards(self) -> None:
        """Not maintained beside them. A second list is a list that disagrees."""
        assert ALLOWED_MODELS == {model_id for model_id, card in CARDS.items() if card.permitted}

    def test_an_undeclared_model_is_refused(self) -> None:
        with pytest.raises(UnknownModel):
            card_for("some-new-model")

    def test_the_refusal_says_what_to_do(self) -> None:
        """An error that names the file to edit is the difference between a two-minute fix
        and a search through the provider for wherever the check lives."""
        with pytest.raises(UnknownModel, match="ModelCard"):
            card_for("gpt-9")

    def test_every_card_id_matches_its_key(self) -> None:
        """A mismatch would make `card_for` return facts about a different model, which is
        precisely the class of error this table exists to prevent."""
        for key, card in CARDS.items():
            assert card.id == key

    def test_every_permitted_model_has_a_price(self) -> None:
        """This table says how to *call* a model; `cortex/db/spend.py` prices it. Kept
        apart on purpose — a benchmark tweak must not reprice a customer's history — so
        this is the test that stops them drifting. An unpriced model bills at the fallback
        rate, which is deliberately expensive and therefore visible, but a permitted model
        with no rate is still a bug in the table rather than a pricing decision."""
        for model_id in ALLOWED_MODELS:
            assert model_id in RATES, f"{model_id} is permitted but has no rate"


class TestRoomForOutput:
    def test_it_leaves_the_prompt_alone(self) -> None:
        card = card_for("claude-sonnet-5")
        assert card.room_for_output(0) == card.max_output_tokens

    def test_a_prompt_near_the_window_shrinks_the_answer(self) -> None:
        """The Haiku case: a transcript that is comfortable on Sonnet 5 leaves almost no
        room here, and nothing in the loop knew that before this table existed."""
        card = card_for("claude-haiku-4-5")
        assert card.room_for_output(195_000) == 5_000

    def test_an_oversized_prompt_leaves_nothing_rather_than_a_negative(self) -> None:
        """A negative max_tokens is a 400. Zero is a number the caller can branch on."""
        assert card_for("claude-haiku-4-5").room_for_output(500_000) == 0


class TestTheRequestShapeFollowsTheCard:
    def test_a_current_model_gets_adaptive_thinking(self) -> None:
        shape = _llm("claude-sonnet-5")._shape(max_tokens=1024)
        assert shape["thinking"] == {"type": "adaptive"}

    def test_a_pre_46_model_gets_the_budget_form(self) -> None:
        """`{"type": "adaptive"}` is a 400 on these. The card is the only thing that
        knows, and it is why the shape is built in one place rather than written out at
        each call site."""
        shape = _llm("claude-haiku-4-5")._shape(max_tokens=1024)
        assert shape["thinking"]["type"] == "enabled"
        assert "budget_tokens" in shape["thinking"]

    def test_the_thinking_budget_leaves_room_for_an_answer(self) -> None:
        """A budget equal to max_tokens spends the entire response on thinking and
        truncates the output — which, for the drafting call, discards the investigation."""
        shape = _llm("claude-haiku-4-5")._shape(max_tokens=1024)
        assert shape["thinking"]["budget_tokens"] < shape["max_tokens"]

    def test_max_tokens_is_clamped_to_the_models_ceiling(self) -> None:
        """`DRAFT_MAX_TOKENS` is 32768 and was chosen against Sonnet 5. Sent unclamped to a
        model with a lower ceiling it is a 400 — at the drafting call, with everything
        already spent."""
        card = card_for("claude-haiku-4-5")
        shape = _llm("claude-haiku-4-5")._shape(max_tokens=999_999)
        assert shape["max_tokens"] == card.max_output_tokens

    def test_a_request_under_the_ceiling_is_untouched(self) -> None:
        assert _llm("claude-sonnet-5")._shape(max_tokens=4096)["max_tokens"] == 4096

    def test_effort_is_sent_where_it_is_supported(self) -> None:
        assert _llm("claude-sonnet-5")._output_config() == {"effort": "low"}

    def test_effort_is_omitted_where_it_is_not(self) -> None:
        """Sent to a model that does not have it, `output_config.effort` is a rejected
        request rather than an ignored parameter."""
        assert "effort" not in _llm("claude-haiku-4-5")._output_config()

    def test_the_schema_survives_a_model_without_effort(self) -> None:
        """The structured-output path merges `format` into the same object. Dropping
        `effort` must not drop the schema with it — a report drafted without the schema is
        prose, and the grounding gate cannot operate on prose."""
        config = _llm("claude-haiku-4-5")._output_config({"format": {"type": "json_schema"}})
        assert config["format"] == {"type": "json_schema"}

    def test_prompt_cache_is_applied_where_supported(self) -> None:
        assert "cache_control" in _llm("claude-sonnet-5")._shape(max_tokens=1024)

    def test_prompt_cache_is_omitted_where_not(self) -> None:
        card = ModelCard(
            id="no-cache", context_window=100, max_output_tokens=100, supports_prompt_cache=False
        )
        llm = _llm("claude-sonnet-5")
        llm.card = card
        assert "cache_control" not in llm._shape(max_tokens=64)


class TestConstruction:
    def test_a_declared_but_unpermitted_model_is_refused_on_cost(self) -> None:
        """Two different mistakes with two different fixes, so they get two different
        messages: this one is a cost decision, not a missing declaration."""
        with pytest.raises(ValueError, match="cost restriction"):
            AnthropicLLM(model="claude-opus-5", api_key="x")

    def test_an_undeclared_model_is_refused_as_undeclared(self) -> None:
        with pytest.raises(UnknownModel):
            AnthropicLLM(model="claude-nonexistent-7", api_key="x")

    def test_the_permitted_model_carries_its_card(self) -> None:
        llm = AnthropicLLM(model=DEFAULT_MODEL, api_key="x")
        assert llm.card.id == DEFAULT_MODEL
        assert llm.card.context_window > 0
