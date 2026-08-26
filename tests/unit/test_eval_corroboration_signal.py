"""A corroboration signal must score whether the report corroborated, not which series it used.

`measurement_stopped` needs two independent lines: GA4's sessions stopping, and a PostHog series
that kept recording past the stop. The fixture plants two healthy PostHog series — signups and
pageviews — because a world where every other event returns nothing argues for the site outage the
scenario exists to rule out.

The accuracy signal only knew about signups. Run 25's second attempt corroborated with pageviews
— the more direct refutation, since pageviews measure the quantity GA4 lost while signups only
imply it — and scored 0.50 for choosing the better series. That is the failure this file pins.
"""

from __future__ import annotations

import uuid

from cortex.eval.fixtures import alternatives_of, by_name
from cortex.eval.scorer import Scorer
from cortex.reports.schema import Claim, Confidence, InvestigationReport

_SCENARIO = by_name("measurement_stopped")
#: The stop itself is the first requirement; the second line is the one under test.
_STOP, _SECOND_LINE = _SCENARIO.ground_truth.required_signals


def _report(summary: str) -> InvestigationReport:
    return InvestigationReport(
        question=_SCENARIO.question,
        executive_summary=[Claim(text=summary, evidence_ids=[uuid.uuid4()])],
        confidence=Confidence.HIGH,
    )


class TestEitherPostHogSeriesIsACorrectSecondLine:
    def test_every_alternative_on_its_own_scores_full_accuracy(self) -> None:
        """Alternatives, not a checklist. Scored one at a time so a report naming a single
        series is not quietly depending on another alternative appearing elsewhere."""
        for alternative in alternatives_of(_SECOND_LINE):
            summary = (
                f"GA4 sessions stopped being collected after 3 August; {alternative} data "
                "from PostHog kept arriving through the 15th, so traffic did not stop."
            )
            scored = Scorer()._accuracy(_SCENARIO, _report(summary))
            assert scored.score == 1.0, f"{alternative}: {scored.detail}"

    def test_pageviews_are_among_them(self) -> None:
        """The specific alternative run 25 was penalised for. Named explicitly so removing it
        from the tuple fails here rather than only in the next full evaluation."""
        assert "pageview" in alternatives_of(_SECOND_LINE)


class TestWideningTheSignalDidNotMakeItAFreePass:
    def test_naming_only_the_gap_still_falls_short(self) -> None:
        """A report that notices GA4 stopped and stops there has found a symptom. It cannot
        distinguish a broken tag from a site that went dark, and must not score as if it could."""
        scored = Scorer()._accuracy(
            _SCENARIO, _report("GA4 stopped collecting sessions after 3 August.")
        )
        assert scored.score == 0.5
        assert "pageview" in scored.detail

    def test_naming_only_the_second_line_falls_short_too(self) -> None:
        """Symmetric, and the reason the requirement is a pair. PostHog holding steady explains
        nothing on its own — the answer is the collection failure it corroborates."""
        scored = Scorer()._accuracy(
            _SCENARIO, _report("PostHog pageviews held steady through 15 August.")
        )
        assert scored.score == 0.5
        assert any(alt in scored.detail for alt in alternatives_of(_STOP))
