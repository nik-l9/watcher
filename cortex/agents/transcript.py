"""Invariants the transcript must satisfy before it is sent to a provider.

Borrowed in design from OpenHands' `View` and its `ViewPropertyBase` properties, found by
indexing their SDK as a graph. Their idea, and it is a good one: the list of events sent to a
model is not just a list — it satisfies **properties** that the provider's API requires, and
those properties are stated as code rather than maintained by care.

They keep four (`ToolCallMatchingProperty`, `BatchAtomicityProperty`,
`ObservationUniquenessProperty`, `ToolLoopAtomicityProperty`), each with two mechanisms:
`enforce`, which removes offending events, and *manipulation indices*, which mark where the
list may be cut without breaking the property. A condenser that respects those indices cannot
produce an invalid payload.

**Why this matters here specifically.** F-12 was this exact class of bug: `Message` had no way
to carry an assistant turn's tool calls, so every `tool_result` referenced a `tool_use` that
was not in the transcript and the provider rejected the whole request. That instance is fixed.
The *class* was not — nothing stated the invariant, so the next change to transcript
construction can reintroduce it, and the symptom is an opaque 400 mid-investigation.

**And it is about to matter more.** Our transcript grows unbounded. Memory recall, four
connectors and richer observations all push toward needing to drop or summarise turns, and the
moment anything truncates a transcript by position it will cut a tool call away from its
result. `safe_cut_points` exists so that when condensation arrives it has somewhere correct to
cut, rather than discovering this the way we discovered F-12.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cortex.agents.llm import Message


@dataclass(frozen=True, slots=True)
class TranscriptProblem:
    """One violated invariant, phrased for whoever has to fix it."""

    invariant: str
    detail: str

    def __str__(self) -> str:
        return f"{self.invariant}: {self.detail}"


def check(messages: Sequence[Message]) -> list[TranscriptProblem]:
    """Every invariant this transcript violates, or an empty list.

    Returns problems rather than raising, so a caller can log a warning in production and a
    test can assert on the specific violation. The loop raises on them; see `assert_valid`.
    """
    problems: list[TranscriptProblem] = []
    problems.extend(_unpaired_tool_calls(messages))
    problems.extend(_orphaned_tool_results(messages))
    problems.extend(_misplaced_results(messages))
    return problems


def assert_valid(messages: Sequence[Message]) -> None:
    """Raise if the transcript would be rejected by the provider.

    Checked before the call rather than diagnosed after it. F-12 surfaced as
    `unexpected tool_use_id found in tool_result blocks` from the provider, mid-run, after the
    tokens were spent — a message that says nothing about which turn is wrong. This raises
    with the turn index instead, before anything is sent.
    """
    problems = check(messages)
    if problems:
        raise InvalidTranscript(
            "the transcript violates an invariant the provider requires: "
            + "; ".join(str(problem) for problem in problems[:3])
        )


class InvalidTranscript(ValueError):
    """The transcript would be rejected by the provider, or silently misread by it."""


def safe_cut_points(messages: Sequence[Message]) -> set[int]:
    """Indices at which the transcript may be truncated without breaking an invariant.

    An index `i` is safe when keeping `messages[i:]` leaves no tool result without its call.
    Cutting anywhere else separates a `tool_use` from its `tool_result`, which is exactly the
    F-12 failure — and a truncation that produces it would be much harder to attribute than
    the original, because the transcript was valid when it was built.

    Returned as a set rather than a single "safe boundary" for the same reason OpenHands
    returns indices: a condenser wants to choose *which* safe cut best fits its budget, not be
    handed one.
    """
    safe: set[int] = set()
    for index in range(len(messages) + 1):
        if not _orphaned_tool_results(messages[index:]):
            safe.add(index)
    return safe


def _unpaired_tool_calls(messages: Sequence[Message]) -> list[TranscriptProblem]:
    """Every tool call must be answered in the next turn.

    A call with no result leaves the model waiting for an answer it will never see, and some
    providers reject it outright. It also produces a subtler failure: the model re-issues the
    call, which now reads as the analyst repeating itself.
    """
    problems: list[TranscriptProblem] = []
    for index, message in enumerate(messages):
        if not message.tool_requests:
            continue
        answered: set[str] = set()
        following = messages[index + 1] if index + 1 < len(messages) else None
        if following is not None:
            answered = set(following.tool_results or {})
        missing = [request.id for request in message.tool_requests if request.id not in answered]
        if missing:
            problems.append(
                TranscriptProblem(
                    "unanswered tool call",
                    f"turn {index} requested {len(message.tool_requests)} tool call(s) but "
                    f"{len(missing)} received no result ({', '.join(missing[:3])})",
                )
            )
    return problems


def _orphaned_tool_results(messages: Sequence[Message]) -> list[TranscriptProblem]:
    """Every tool result must answer a call in the immediately preceding turn.

    This is F-12 stated as an invariant. The provider's requirement is positional, not merely
    referential: a `tool_result` must pair with a `tool_use` in the *previous* assistant
    message, so a result whose call appears three turns earlier is still invalid.
    """
    problems: list[TranscriptProblem] = []
    for index, message in enumerate(messages):
        if not message.tool_results:
            continue
        previous = messages[index - 1] if index > 0 else None
        offered = {request.id for request in (previous.tool_requests if previous else ())}
        orphans = [key for key in message.tool_results if key not in offered]
        if orphans:
            problems.append(
                TranscriptProblem(
                    "orphaned tool result",
                    f"turn {index} carries {len(orphans)} result(s) whose tool call is not in "
                    f"the preceding turn ({', '.join(orphans[:3])})",
                )
            )
    return problems


def _misplaced_results(messages: Sequence[Message]) -> list[TranscriptProblem]:
    """Tool results belong on a user turn, and tool calls on an assistant turn.

    Cheap to check and easy to get wrong when constructing a transcript by hand, which is what
    every test helper and every future condenser does.
    """
    problems: list[TranscriptProblem] = []
    for index, message in enumerate(messages):
        if message.tool_results and message.role != "user":
            problems.append(
                TranscriptProblem(
                    "misplaced tool result",
                    f"turn {index} is role={message.role!r} but carries tool results",
                )
            )
        if message.tool_requests and message.role != "assistant":
            problems.append(
                TranscriptProblem(
                    "misplaced tool call",
                    f"turn {index} is role={message.role!r} but carries tool calls",
                )
            )
    return problems
