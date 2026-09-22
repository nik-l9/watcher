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

from cortex.ask import _render_truth, render
from cortex.eval.fixtures import SCENARIOS, by_name
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


class TestTheTruthBlockIsReadableByAStranger:
    """`--show-truth` is what makes a recorded run checkable rather than admirable.

    It crashed for every scenario whose label carries an any-of requirement --
    `TypeError: sequence item 0: expected str instance, tuple found` -- which is most of
    the answerable ones. Nothing covered it, so it stayed broken.
    """

    def test_an_any_of_requirement_renders_instead_of_raising(self) -> None:
        truth = by_name("onboarding_regression").ground_truth
        assert any(isinstance(signal, tuple) for signal in truth.required_signals), (
            "this scenario is the regression's subject; if its label loses its any-of "
            "requirement, point this test at another one rather than deleting it"
        )
        block = _render_truth(truth)
        assert "any of [" in block
        assert "91c3e4a" in block and "913" in block

    def test_every_scenario_renders(self) -> None:
        for scenario in SCENARIOS:
            block = _render_truth(scenario.ground_truth)
            assert scenario.ground_truth.cause in block, scenario.name

    def test_an_unanswerable_scenario_says_so_rather_than_looking_like_a_miss(self) -> None:
        # Printing only `cause` made "the data cannot establish a reason" read as an
        # analyst that had failed to find something.
        block = _render_truth(by_name("insufficient_evidence").ground_truth)
        assert "NO CAUSE" in block

    def test_a_false_premise_scenario_names_the_verdict_and_the_refutation(self) -> None:
        truth = by_name("partial_month_false_premise").ground_truth
        block = _render_truth(truth)
        assert "REFUSE THE PREMISE" in block
        assert "summary must refute with" in block

    def test_a_cause_scenario_is_not_labelled_a_refusal(self) -> None:
        block = _render_truth(by_name("onboarding_regression").ground_truth)
        assert "NAME THE CAUSE" in block
        assert "REFUSE" not in block

    def test_no_requirements_reads_as_none_not_as_an_empty_line(self) -> None:
        block = _render_truth(by_name("insufficient_evidence").ground_truth)
        assert "signals a correct answer names: none" in block
