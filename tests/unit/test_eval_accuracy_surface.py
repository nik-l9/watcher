"""What counts as the report having *claimed* something.

`_accuracy` string-matches the planted cause's signals against the report, which is the right
call — the alternative is a model deciding whether two prose explanations agree, and then the
headline number depends on its opinion. But the signal was matched against `_report_text`, which
flattens every string in the document including the statements of hypotheses the report
*contradicted*. A report that named the planted cause in order to reject it scored as having
found it.

These are pure-function tests over `_asserted_text` because the defect is entirely in the search
space. Nothing here needs a scenario, a session or a database.
"""

from __future__ import annotations

import uuid

from cortex.eval.scorer import _asserted_text, _report_text
from cortex.reports.schema import (
    Claim,
    Confidence,
    Finding,
    Hypothesis,
    InvestigationReport,
    Recommendation,
    Risk,
    Verdict,
)


def _evidence() -> list[uuid.UUID]:
    return [uuid.uuid4()]


def _report(**overrides: object) -> InvestigationReport:
    defaults: dict[str, object] = {
        "question": "why did mobile signups fall?",
        "executive_summary": [
            Claim(text="A pricing page change did it.", evidence_ids=_evidence())
        ],
        "confidence": Confidence.HIGH,
    }
    return InvestigationReport(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestRejectingACauseIsNotFindingIt:
    def test_a_contradicted_hypothesis_is_not_an_assertion(self) -> None:
        """The defect, exactly as it shipped. The report blames a pricing change and explicitly
        rules the deploy out; the old surface matched `91c3e` and credited it."""
        report = _report(
            hypotheses=[
                Hypothesis(
                    statement="The deploy 91c3e broke mobile checkout.",
                    verdict=Verdict.CONTRADICTED,
                    reasoning="Timing does not line up: 91c3e shipped after the fall began.",
                    contradicting_evidence_ids=_evidence(),
                )
            ]
        )
        assert "91c3e" in _report_text(report).lower()
        assert "91c3e" not in _asserted_text(report).lower()

    def test_an_inconclusive_hypothesis_is_not_an_assertion_either(self) -> None:
        """A report that could not settle a cause has declined to stand behind it. That is the
        same rule `_DISMISSED_VERDICTS` already applied to decoys."""
        report = _report(
            hypotheses=[
                Hypothesis(
                    statement="The deploy 91c3e may have broken checkout.",
                    verdict=Verdict.INCONCLUSIVE,
                    reasoning="No error logs either way.",
                )
            ]
        )
        assert "91c3e" not in _asserted_text(report).lower()

    def test_a_supported_hypothesis_still_counts(self) -> None:
        """Naming the cause as a hypothesis the evidence upheld *is* finding it. Excluding all
        hypotheses would fix the over-credit by introducing an under-credit."""
        report = _report(
            hypotheses=[
                Hypothesis(
                    statement="The deploy 91c3e broke mobile checkout.",
                    verdict=Verdict.SUPPORTED,
                    supporting_evidence_ids=_evidence(),
                )
            ]
        )
        assert "91c3e" in _asserted_text(report).lower()


class TestACaveatIsNotAClaim:
    def test_a_risk_does_not_count_as_having_found_the_cause(self) -> None:
        """ "This could be a tracking artefact" is not the claim "this was a tracking artefact",
        and telling those apart is the whole point of the dimension."""
        report = _report(risks=[Risk(description="Deploy 91c3e might be involved.")])
        assert "91c3e" not in _asserted_text(report).lower()

    def test_the_question_itself_does_not_count(self) -> None:
        """The question is the input. Echoing a signal the user supplied is not a finding — and
        it was in the old surface, so a question naming the cause scored itself."""
        report = _report(question="did deploy 91c3e break mobile checkout?")
        assert "91c3e" not in _asserted_text(report).lower()

    def test_findings_and_recommendations_do_count(self) -> None:
        report = _report(
            findings=[
                Finding(
                    title="Checkout broke on 91c3e",
                    claims=[Claim(text="Mobile conversion halved.", evidence_ids=_evidence())],
                    confidence=Confidence.HIGH,
                )
            ],
            recommendations=[
                Recommendation(
                    action="Revert abcdef1",
                    rationale="It broke checkout.",
                    priority=1,
                    evidence_ids=_evidence(),
                )
            ],
        )
        asserted = _asserted_text(report).lower()
        assert "91c3e" in asserted and "abcdef1" in asserted


class TestTheMentionSurfaceStillIncludesEverything:
    def test_report_text_keeps_rejected_hypotheses(self) -> None:
        """`_summary_placement` asks whether a refutation reached the summary or was buried
        further down. A signal buried in a discarded hypothesis was still mentioned, which is
        the failure it measures — so that surface must stay wide."""
        report = _report(
            hypotheses=[
                Hypothesis(
                    statement="Tracking stopped on 2026-08-03.",
                    verdict=Verdict.CONTRADICTED,
                    reasoning="See the series.",
                    contradicting_evidence_ids=_evidence(),
                )
            ]
        )
        assert "2026-08-03" in _report_text(report)
