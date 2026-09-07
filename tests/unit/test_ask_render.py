"""What the terminal report says about itself.

The header is the first line a reader takes at face value, and it said something it did not
know. `Took 81s over 4 steps, 4,740 tokens, 4 observations gathered` printed
`len(report.sources)` -- the number of observations *cited*. That live investigation gathered
eleven and cited four, so the line hid the seven the answer did not rest on.

The gap between gathered and cited is not a detail. It is the quantity
arXiv:2608.23623 ("When May an Agent Stop? Evidence-Carrying Termination") gates completion
on -- every answer slot bound to tool receipts that replay -- and reporting one number under
the other name removed the reader's only view of it.
"""

from __future__ import annotations

import uuid

from cortex.ask import render
from cortex.reports.schema import Claim, Confidence, InvestigationReport, Source


def _report(*, cited: int) -> InvestigationReport:
    ids = [uuid.uuid4() for _ in range(max(cited, 1))]
    return InvestigationReport(
        question="Which month was best for signups?",
        executive_summary=[Claim(text="July, at 4,603.", evidence_ids=ids[:1])],
        confidence=Confidence.LOW,
        sources=[
            Source(
                evidence_id=identifier,
                tool_name="posthog",
                capability="event_trend",
                source_ref=f"posthog://trend/{index}",
            )
            for index, identifier in enumerate(ids[:cited])
        ],
    )


class TestTheHeaderCountsBothNumbers:
    def test_it_names_what_was_gathered_and_what_was_cited(self) -> None:
        header = render(
            _report(cited=4), steps=4, seconds=81, tokens=4740, gathered=11
        ).splitlines()
        line = next(entry for entry in header if entry.startswith("Took "))
        assert "11 observations gathered, 4 cited" in line

    def test_a_caller_that_does_not_know_says_only_what_it_knows(self) -> None:
        """Rather than labelling the cited count as the gathered one, which is what it did."""
        header = render(_report(cited=4), steps=4, seconds=81, tokens=4740).splitlines()
        line = next(entry for entry in header if entry.startswith("Took "))
        assert "4 observations cited" in line
        assert "gathered" not in line

    def test_the_counts_are_not_conflated_when_they_are_equal(self) -> None:
        """A report citing everything it gathered still states both, so the reader learns the
        two are different quantities rather than inferring it from a run where they differ."""
        header = render(_report(cited=3), steps=2, seconds=40, tokens=900, gathered=3).splitlines()
        line = next(entry for entry in header if entry.startswith("Took "))
        assert "3 observations gathered, 3 cited" in line
