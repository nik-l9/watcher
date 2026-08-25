"""A short label for an investigation.

A list of investigations rendered as raw question text is unusable: *"Why did signups fall in
the week of 15 July 2026, and did anyone report it in Slack before the metric moved?"* is a
question, not a row in a table. Nothing produced a shorter form, so any UI listing
investigations would have had to invent one — in the frontend, where it could not be searched,
stored, or kept stable.

**Derived, not generated, and that is a domain difference rather than a shortcut.** OpenHands
generates titles with an LLM and falls back to truncation, which is right for them: a coding
task can arrive as a paragraph of specification, and the first fifty characters of one are
meaningless. A GTM question is already a short human sentence someone typed into a box, so
truncating it at a word boundary produces a better title than a model would, for free, with no
latency and no failure mode.

If that ever stops being true — a question arriving as a pasted brief rather than a sentence —
the trigger is visible: `was_truncated` says how often the raw question did not fit.
"""

from __future__ import annotations

import re

#: Characters a title may run to. Sized for a table row and a browser tab, not a paragraph.
MAX_LENGTH = 72

#: Openers that carry no information in a list where every row is a question. Dropped only
#: when the remainder still reads as a phrase, so "Why did signups fall" keeps its subject.
_LEADING_NOISE = re.compile(
    r"^(?:can you |could you |please |i want to know |tell me |help me )+", re.I
)

_WHITESPACE = re.compile(r"\s+")


def title_for(question: str) -> str:
    """A short label for this question.

    Never empty, and never longer than `MAX_LENGTH`. A question that is already short comes
    back unchanged apart from whitespace and trailing punctuation.
    """
    cleaned = _WHITESPACE.sub(" ", (question or "").strip())
    cleaned = _LEADING_NOISE.sub("", cleaned).strip()
    if not cleaned:
        # An empty question cannot happen through the API, which requires a non-empty string.
        # A label is still produced rather than an empty cell, because a blank row in a list
        # is indistinguishable from a rendering bug.
        return "Untitled investigation"

    # The trailing question mark is dropped: in a list where every row is a question, it is
    # pure noise, and it costs a character that a word does not.
    cleaned = cleaned.rstrip("?！!。.").strip() or cleaned

    if len(cleaned) <= MAX_LENGTH:
        return _capitalise(cleaned)

    # Cut at a word boundary. Mid-word truncation reads as corruption rather than as
    # abbreviation, which makes a reader distrust the row rather than click it.
    window = cleaned[: MAX_LENGTH - 1]
    boundary = window.rfind(" ")
    if boundary >= MAX_LENGTH // 2:
        window = window[:boundary]
    return _capitalise(window.rstrip(" ,;:-—") + "…")


def was_truncated(question: str) -> bool:
    """Whether the title lost anything.

    Exposed so the decision above can be revisited on evidence: if questions start arriving as
    pasted briefs rather than sentences, this is the number that says so — and generating
    titles with a model becomes worth its cost.
    """
    return title_for(question).endswith("…")


def _capitalise(text: str) -> str:
    """Uppercase the first letter without touching the rest.

    `str.capitalize` lowercases everything after the first character, which would turn
    "Why did GA4 sessions drop" into "Why did ga4 sessions drop" — mangling exactly the
    product and metric names a reader scans a list for.
    """
    return text[:1].upper() + text[1:] if text else text
