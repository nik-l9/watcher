"""Short labels for investigations.

Derived rather than generated, which is a domain difference rather than a shortcut: generating
a title with a model earns its cost where a task arrives as a paragraph of specification and the
first fifty characters of one are meaningless. A GTM question is a short sentence somebody typed
into a box, so truncating at a word boundary beats a model — free, instant, and with no failure
mode.

`was_truncated` exists so that decision can be revisited on evidence rather than opinion.
"""

from __future__ import annotations

import pytest

from cortex.db.titles import MAX_LENGTH, title_for, was_truncated


class TestShortQuestionsSurviveIntact:
    def test_a_normal_question_keeps_its_words(self) -> None:
        assert title_for("Why did signups fall last week?") == "Why did signups fall last week"

    def test_the_trailing_question_mark_goes(self) -> None:
        """In a list where every row is a question, it is pure noise — and it costs a
        character that a word does not."""
        assert not title_for("Why did signups fall?").endswith("?")

    def test_acronyms_keep_their_case(self) -> None:
        """`str.capitalize` would lowercase everything after the first character, mangling
        exactly the product and metric names a reader scans a list for."""
        assert title_for("why did GA4 sessions drop?") == "Why did GA4 sessions drop"
        assert "PostHog" in title_for("did PostHog events stop firing?")

    def test_the_first_letter_is_capitalised(self) -> None:
        assert title_for("why did signups fall?").startswith("Why")


class TestConversationalOpenersAreDropped:
    @pytest.mark.parametrize(
        ("question", "noise"),
        [
            ("can you check if we integrated Apollo?", "can you"),
            ("could you check if we integrated Apollo?", "could you"),
            ("please check if we integrated Apollo?", "please"),
            ("tell me if we integrated Apollo?", "tell me"),
        ],
    )
    def test_noise_openers_go(self, question: str, noise: str) -> None:
        """Every row in this list is a request. "Can you" on all of them carries nothing and
        pushes the informative words out of the visible width.

        What is asserted is that the noise is gone and the informative words survive — not
        which word ends up first. "Tell me if we integrated Apollo" becomes "If we integrated
        Apollo", a fragment, and a fragment naming the subject beats a sentence that does
        not."""
        title = title_for(question)
        assert noise not in title.lower()
        assert "Apollo" in title

    def test_a_question_word_is_not_an_opener(self) -> None:
        """ "Why" is the subject of a causal question, not noise — dropping it would turn
        "Why did signups fall" into "Did signups fall", a different question."""
        assert title_for("Why did signups fall?").startswith("Why")


class TestLongQuestions:
    def test_it_cuts_at_a_word_boundary(self) -> None:
        """Mid-word truncation reads as corruption rather than abbreviation, which makes a
        reader distrust the row instead of clicking it."""
        question = (
            "Why did signups fall in the week of 15 July 2026, and did anyone report it in "
            "Slack before the metric moved?"
        )
        title = title_for(question)

        assert len(title) <= MAX_LENGTH
        assert title.endswith("…")
        # The character before the ellipsis ends a word rather than splitting one.
        assert question.startswith(title[:-1])
        assert not title[:-1].endswith(" ")

    def test_truncation_is_reported(self) -> None:
        assert was_truncated("a" * 200) is True
        assert was_truncated("Why did signups fall?") is False

    def test_a_single_enormous_word_is_still_bounded(self) -> None:
        """No word boundary to cut at. The length limit still holds, because a title that
        overflows its column is a rendering bug wherever it is rendered."""
        title = title_for("x" * 500)
        assert len(title) <= MAX_LENGTH

    def test_trailing_punctuation_is_trimmed_before_the_ellipsis(self) -> None:
        """ "…, …" reads as a mistake. The comma carries nothing at a cut point."""
        question = "Why did signups fall, " + "and did anyone notice " * 10
        assert ", …" not in title_for(question)


class TestItAlwaysProducesSomething:
    @pytest.mark.parametrize("question", ["", "   ", "\n\t "])
    def test_an_empty_question_gets_a_label(self, question: str) -> None:
        """Unreachable through the API, which requires a non-empty string. A blank row in a
        list is indistinguishable from a rendering bug, so it gets words either way."""
        assert title_for(question) == "Untitled investigation"

    def test_a_question_of_only_punctuation_is_not_blanked(self) -> None:
        """Stripping punctuation must not leave an empty string, which would render as a
        blank row."""
        assert title_for("???") != ""

    def test_whitespace_is_collapsed(self) -> None:
        """A pasted question carries newlines, and a title with a newline in it breaks the
        row it sits in."""
        title = title_for("Why did\n\n  signups   fall?")
        assert title == "Why did signups fall"
        assert "\n" not in title
