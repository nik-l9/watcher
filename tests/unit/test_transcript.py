"""Transcript invariants.

F-12 is the reason this file exists. `Message` had no way to carry an assistant turn's tool
calls, so every `tool_result` referenced a `tool_use` that was not in the transcript, and the
provider rejected the request with `unexpected tool_use_id found in tool_result blocks` — mid
investigation, after the tokens were spent, naming no turn.

That instance was fixed a while ago. The *class* was not: nothing stated the invariant, so the
next change to transcript construction could reintroduce it. These tests state it.

The design is borrowed from OpenHands' `View` properties, which pair an `enforce` step with
*manipulation indices* marking where the list may be cut. `safe_cut_points` is that second
half, and it exists before condensation does — because the first thing a naive truncation does
is cut a tool call away from its result, and that failure is far harder to attribute than the
original was.
"""

from __future__ import annotations

import pytest

from cortex.agents.llm import Message, ToolRequest
from cortex.agents.transcript import (
    InvalidTranscript,
    assert_valid,
    check,
    safe_cut_points,
)


def _call(call_id: str = "t1", name: str = "ga4__get_sessions") -> ToolRequest:
    return ToolRequest(id=call_id, name=name, arguments={"start_date": "2026-07-01"})


def _healthy() -> list[Message]:
    """A well-formed two-round transcript, as the loop actually builds one."""
    return [
        Message(role="user", content="Why did signups fall?"),
        Message(role="assistant", content="Checking sessions.", tool_requests=(_call("t1"),)),
        Message(role="user", tool_results={"t1": "evidence_id: abc"}),
        Message(role="assistant", content="Checking the funnel.", tool_requests=(_call("t2"),)),
        Message(role="user", tool_results={"t2": "evidence_id: def"}),
    ]


class TestAHealthyTranscriptPasses:
    def test_no_problems(self) -> None:
        assert check(_healthy()) == []

    def test_assert_valid_is_silent(self) -> None:
        assert_valid(_healthy())

    def test_several_calls_in_one_turn_are_fine(self) -> None:
        """The loop issues parallel calls in a single turn, and all of them are answered
        together in the next."""
        messages = [
            Message(role="assistant", tool_requests=(_call("t1"), _call("t2"))),
            Message(role="user", tool_results={"t1": "a", "t2": "b"}),
        ]
        assert check(messages) == []


class TestF12StatedAsAnInvariant:
    def test_a_result_whose_call_is_absent_is_caught(self) -> None:
        """The original bug: the assistant turn recorded prose only, so the result below it
        referenced a tool call that was nowhere in the transcript."""
        messages = [
            Message(role="assistant", content="Checking sessions."),  # no tool_requests
            Message(role="user", tool_results={"t1": "evidence_id: abc"}),
        ]
        problems = check(messages)
        assert any("orphaned tool result" in str(p) for p in problems)
        # And it names the turn, which the provider's own error never did.
        assert any("turn 1" in str(p) for p in problems)

    def test_a_result_answering_an_older_turn_is_still_invalid(self) -> None:
        """The provider's requirement is positional, not merely referential: a `tool_result`
        must pair with a `tool_use` in the *immediately* preceding turn."""
        messages = [
            Message(role="assistant", tool_requests=(_call("t1"),)),
            Message(role="user", content="a follow-up question"),
            Message(role="user", tool_results={"t1": "evidence_id: abc"}),
        ]
        assert any("orphaned tool result" in str(p) for p in check(messages))

    def test_an_unanswered_call_is_caught(self) -> None:
        """A call with no result leaves the model waiting for an answer it never sees — and
        it re-issues the call, which then reads as the analyst repeating itself."""
        messages = [
            Message(role="assistant", tool_requests=(_call("t1"), _call("t2"))),
            Message(role="user", tool_results={"t1": "evidence_id: abc"}),
        ]
        problems = check(messages)
        assert any("unanswered tool call" in str(p) for p in problems)
        assert any("t2" in str(p) for p in problems)

    def test_assert_valid_raises_before_anything_is_sent(self) -> None:
        messages = [Message(role="user", tool_results={"t1": "orphan"})]
        with pytest.raises(InvalidTranscript, match="orphaned tool result"):
            assert_valid(messages)

    @pytest.mark.parametrize(
        ("role", "kwargs", "expected"),
        [
            ("assistant", {"tool_results": {"t1": "a"}}, "misplaced tool result"),
            ("user", {"tool_requests": (_call("t1"),)}, "misplaced tool call"),
        ],
    )
    def test_results_and_calls_must_sit_on_the_right_role(
        self, role: str, kwargs: dict, expected: str
    ) -> None:
        """Easy to get wrong when building a transcript by hand, which is what every test
        helper and every future condenser does."""
        problems = check([Message(role=role, **kwargs)])  # type: ignore[arg-type]
        assert any(expected in str(p) for p in problems)


class TestWhereATranscriptMayBeCut:
    """The half that exists before condensation does."""

    def test_the_boundary_between_rounds_is_safe(self) -> None:
        messages = _healthy()
        safe = safe_cut_points(messages)
        # Cutting to keep from turn 3 keeps a complete round: call then result.
        assert 3 in safe
        # And the end and the start are trivially safe.
        assert 0 in safe
        assert len(messages) in safe

    def test_cutting_between_a_call_and_its_result_is_not_safe(self) -> None:
        """This is the truncation that would reintroduce F-12, and the reason these indices
        are computed rather than assumed."""
        messages = _healthy()
        # Keeping from turn 2 starts the transcript at a tool *result* whose call was in
        # turn 1 — exactly the orphan the provider rejects.
        assert 2 not in safe_cut_points(messages)
        assert 4 not in safe_cut_points(messages)

    def test_every_safe_cut_actually_produces_a_valid_transcript(self) -> None:
        """The property the set claims. Asserted by construction rather than by inspection,
        because an index set that is merely plausible is worth nothing."""
        messages = _healthy()
        for index in safe_cut_points(messages):
            remainder = messages[index:]
            orphans = [p for p in check(remainder) if "orphaned" in str(p)]
            assert orphans == [], (index, orphans)

    def test_an_empty_transcript_has_no_problems(self) -> None:
        assert check([]) == []
        assert safe_cut_points([]) == {0}
