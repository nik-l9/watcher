"""The check that decides whether the loop may finish.

This exists because every phase after the loop can only subtract. A live investigation wrote
"No GitHub, PostHog, or Slack evidence was gathered to explain why the January bulk-import
deals exist" into its own risks -- during `_draft`, a call with no tools. It found the
question after losing the ability to answer it.

The danger is the opposite failure. A reflection turn was tried in this loop and measured
worse, because it asked unconditionally and carried no content. So most of what is asserted
here is restraint: the conditions under which this must *not* reopen the loop.
"""

from __future__ import annotations

import pytest

from cortex.agents.gaps import MAX_GAPS, GapCheck, GapVerdict, instruction
from cortex.agents.llm import Usage


class _Stub:
    """An LLM that returns one canned structured payload, and records what it was asked."""

    def __init__(self, payload: dict | None = None, *, raises: bool = False) -> None:
        self._payload = payload or {"sufficient": True, "gaps": []}
        self._raises = raises
        self.seen: str = ""
        self.calls = 0

    async def structured(self, *, system, messages, schema, max_tokens=8192, timeout=None):
        self.calls += 1
        self.seen = messages[0].content
        if self._raises:
            raise RuntimeError("provider unavailable")
        return self._payload, Usage(input_tokens=10, output_tokens=5)


async def _assess(
    llm,
    gathered=("hubspot__pipeline() -> returned data",),
    capabilities=("slack__search_messages",),
):
    return await GapCheck(llm).assess(
        question="how much pipeline is real?",
        gathered=list(gathered),
        capabilities=list(capabilities),
    )


class TestItReopensOnlyForSomethingActionable:
    @pytest.mark.asyncio
    async def test_a_named_reachable_gap_reopens_the_loop(self) -> None:
        verdict = await _assess(
            _Stub({"sufficient": False, "gaps": ["Slack decisions about these deals"]})
        )
        assert verdict.reopens
        assert verdict.gaps == ("Slack decisions about these deals",)

    @pytest.mark.asyncio
    async def test_sufficient_does_not_reopen(self) -> None:
        assert not (await _assess(_Stub({"sufficient": True, "gaps": []}))).reopens

    @pytest.mark.asyncio
    async def test_insufficient_with_no_named_gap_does_not_reopen(self) -> None:
        # The contentless reflection turn, which is the shape that already measured worse.
        # "Something is missing" with nothing named gives the loop nothing to do.
        verdict = await _assess(_Stub({"sufficient": False, "gaps": []}))
        assert verdict.sufficient
        assert not verdict.reopens

    @pytest.mark.asyncio
    async def test_a_provider_failure_lets_the_investigation_finish(self) -> None:
        # An unavailable check must never end a run that worked.
        verdict = await _assess(_Stub(raises=True))
        assert verdict.ran is False
        assert not verdict.reopens

    @pytest.mark.asyncio
    async def test_nothing_gathered_means_no_call_at_all(self) -> None:
        llm = _Stub({"sufficient": False, "gaps": ["anything"]})
        verdict = await GapCheck(llm).assess(question="q", gathered=[], capabilities=["x"])
        assert llm.calls == 0
        assert not verdict.reopens

    @pytest.mark.asyncio
    async def test_gaps_are_capped(self) -> None:
        many = [f"gap {n}" for n in range(MAX_GAPS + 4)]
        verdict = await _assess(_Stub({"sufficient": False, "gaps": many}))
        assert len(verdict.gaps) == MAX_GAPS


class TestWhatTheCheckIsShown:
    @pytest.mark.asyncio
    async def test_it_is_told_which_capabilities_exist(self) -> None:
        # The single most important input: a gap no listed capability can close is not a gap,
        # and half the gaps in recorded runs were of that kind -- "a Google Ads change history
        # log" for a tenant with no ad platform connected.
        llm = _Stub()
        await _assess(llm, capabilities=["slack__search_messages", "posthog__event_trend"])
        assert "slack__search_messages" in llm.seen
        assert "posthog__event_trend" in llm.seen

    @pytest.mark.asyncio
    async def test_it_is_told_which_calls_came_back_empty(self) -> None:
        # So it does not ask for a query that has already returned nothing.
        llm = _Stub()
        await _assess(llm, gathered=["github__commits(repo='acme/web') -> returned nothing"])
        assert "returned nothing" in llm.seen

    @pytest.mark.asyncio
    async def test_it_never_sees_the_reasoning(self) -> None:
        # Evidence, not argument. A check shown the analyst's conclusion grades the
        # conclusion, and what needs grading is what was gathered.
        llm = _Stub()
        await _assess(llm)
        assert "QUESTION" in llm.seen and "OBSERVATIONS ALREADY GATHERED" in llm.seen


class TestTheInstructionHandedBack:
    def test_it_names_the_gaps(self) -> None:
        text = instruction(["Slack decisions about the frozen deals", "stage history"])
        assert "Slack decisions about the frozen deals" in text
        assert "stage history" in text

    def test_it_does_not_ask_for_a_rethink(self) -> None:
        # It is not criticism. The analyst was not wrong; the loop is simply not over.
        text = instruction(["something"]).lower()
        for scolding in ("reconsider", "you were wrong", "think again", "mistake"):
            assert scolding not in text

    def test_it_tells_the_analyst_to_keep_what_it_has(self) -> None:
        # Carrying confirmed findings forward is what stops the second pass re-fetching the
        # first pass's work, which is also the latency answer.
        assert "already established" in instruction(["something"])

    def test_it_caps_what_it_asks_for(self) -> None:
        text = instruction([f"gap {n}" for n in range(10)])
        assert f"{MAX_GAPS}." in text
        assert f"{MAX_GAPS + 1}." not in text


class TestTheDefaultIsToCarryOn:
    def test_a_bare_verdict_does_not_reopen(self) -> None:
        assert not GapVerdict().reopens
