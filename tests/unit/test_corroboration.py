"""A cause needs two independent lines, or the report does not narrow to it.

From DOE-NE-STD-1004-92, and found in none of the other seven frameworks surveyed: every cause
node carries two independent lines of evidence, each named, and if only one is available the tree
does not narrow -- "insufficient evidence does not license a pick; it licenses continued breadth".

The failure behaviour is the whole rule, and it is the opposite of what a model does under
uncertainty: given one thin line an LLM narrows anyway and hedges in prose -- "likely", "appears
to" -- which reads as a cause to anyone who skims. So this acts on the verdict.
"""

from __future__ import annotations

import uuid
from datetime import date

from cortex.reports.corroboration import REQUIRED_LINES, corroborate, lines_for
from cortex.reports.schema import (
    Claim,
    Confidence,
    Hypothesis,
    InvestigationReport,
    Verdict,
)

POSTHOG, GITHUB, SLACK = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
TOOL_OF = {POSTHOG: "posthog", GITHUB: "github", SLACK: "slack"}


def _hypothesis(
    *,
    supporting: list[uuid.UUID],
    verdict: Verdict = Verdict.SUPPORTED,
    dated: bool = True,
    reasoning: str = "Timing lines up.",
) -> Hypothesis:
    return Hypothesis(
        statement="The 16 June copy deploy caused the signup drop.",
        verdict=verdict,
        supporting_evidence_ids=supporting if verdict is Verdict.SUPPORTED else [],
        contradicting_evidence_ids=supporting if verdict is Verdict.CONTRADICTED else [],
        reasoning=reasoning,
        cause_at=date(2026, 6, 16) if dated else None,
        effect_onset=date(2026, 6, 17) if dated else None,
    )


def _report(*hypotheses: Hypothesis) -> InvestigationReport:
    return InvestigationReport(
        question="Why did signups fall in mid-June?",
        executive_summary=[Claim(text="Signups fell.", evidence_ids=[POSTHOG])],
        hypotheses=list(hypotheses),
        confidence=Confidence.MEDIUM,
    )


class TestOneLineDoesNotLicenseAPick:
    def test_a_single_source_is_downgraded(self) -> None:
        result = corroborate(_report(_hypothesis(supporting=[POSTHOG])), TOOL_OF)
        assert result.report.hypotheses[0].verdict is Verdict.INCONCLUSIVE
        assert not result.narrowed

    def test_two_rows_from_one_source_are_still_one_line(self) -> None:
        """The distinction the rule turns on. Two PostHog trends are one line looked at twice,
        and counting rows rather than sources would let a report narrow on a single connector."""
        second_posthog = uuid.uuid4()
        result = corroborate(
            _report(_hypothesis(supporting=[POSTHOG, second_posthog])),
            {**TOOL_OF, second_posthog: "posthog"},
        )
        assert result.report.hypotheses[0].verdict is Verdict.INCONCLUSIVE

    def test_two_sources_are_left_alone(self) -> None:
        result = corroborate(_report(_hypothesis(supporting=[POSTHOG, GITHUB])), TOOL_OF)
        assert result.report.hypotheses[0].verdict is Verdict.SUPPORTED
        assert result.narrowed
        assert result.broadened == ()

    def test_the_threshold_is_the_one_the_source_states(self) -> None:
        """Two, not three. Inventing a stricter bar would refuse nearly everything while
        attributing the refusal to a standard nobody set."""
        assert REQUIRED_LINES == 2

    def test_unresolvable_evidence_counts_as_no_line(self) -> None:
        """An id with no evidence row is not a source. The gate would have removed it, so
        reaching here means it cannot be resolved to anything."""
        assert lines_for(_hypothesis(supporting=[uuid.uuid4()]), TOOL_OF) == ()


class TestContradictingEvidenceIsNotCorroboration:
    def test_ruling_something_else_out_does_not_license_narrowing(self) -> None:
        """Contradicting evidence killed a *different* candidate. Counting it here would let a
        report narrow on the strength of an elimination, which is an argument for breadth."""
        hypothesis = Hypothesis(
            statement="The 16 June deploy caused it.",
            verdict=Verdict.SUPPORTED,
            supporting_evidence_ids=[POSTHOG],
            contradicting_evidence_ids=[GITHUB, SLACK],
            cause_at=date(2026, 6, 16),
            effect_onset=date(2026, 6, 17),
        )
        result = corroborate(_report(hypothesis), TOOL_OF)
        assert result.report.hypotheses[0].verdict is Verdict.INCONCLUSIVE


class TestItOnlyTouchesAssertedDatedCauses:
    def test_an_inconclusive_hypothesis_is_untouched(self) -> None:
        """It has already declined to narrow."""
        hypothesis = _hypothesis(supporting=[POSTHOG], verdict=Verdict.INCONCLUSIVE)
        result = corroborate(_report(hypothesis), TOOL_OF)
        assert result.broadened == ()
        assert result.report.hypotheses[0].reasoning == "Timing lines up."

    def test_a_contradicted_hypothesis_is_untouched(self) -> None:
        hypothesis = _hypothesis(supporting=[POSTHOG], verdict=Verdict.CONTRADICTED)
        assert corroborate(_report(hypothesis), TOOL_OF).broadened == ()

    def test_an_undated_hypothesis_is_untouched(self) -> None:
        """No date, no named intervention, nothing for this rule to be about."""
        hypothesis = _hypothesis(supporting=[POSTHOG], dated=False)
        assert corroborate(_report(hypothesis), TOOL_OF).broadened == ()

    def test_a_clean_report_is_returned_unchanged(self) -> None:
        report = _report(_hypothesis(supporting=[POSTHOG, GITHUB]))
        assert corroborate(report, TOOL_OF).report is report


class TestTheDowngradeExplainsItself:
    def test_the_analysts_own_reasoning_survives(self) -> None:
        """Appended rather than replaced: the argument being overruled is what a reader needs to
        judge the downgrade."""
        result = corroborate(_report(_hypothesis(supporting=[POSTHOG])), TOOL_OF)
        reasoning = result.report.hypotheses[0].reasoning
        assert reasoning.startswith("Timing lines up.")
        assert "Insufficient evidence does not license a pick" in reasoning

    def test_the_lines_it_does_have_are_named(self) -> None:
        """The source's third rule: "the bases for rejected and accepted causes should be
        stated". A count alone would not let a reader see what was thin about it."""
        result = corroborate(_report(_hypothesis(supporting=[POSTHOG])), TOOL_OF)
        assert "posthog" in result.report.hypotheses[0].reasoning

    def test_a_risk_records_every_downgrade(self) -> None:
        result = corroborate(
            _report(
                _hypothesis(supporting=[POSTHOG]),
                Hypothesis(
                    statement="The pricing change caused it.",
                    verdict=Verdict.SUPPORTED,
                    supporting_evidence_ids=[GITHUB],
                    cause_at=date(2026, 6, 10),
                    effect_onset=date(2026, 6, 17),
                ),
            ),
            TOOL_OF,
        )
        assert len(result.broadened) == 2
        disclosure = " ".join(risk.description for risk in result.report.risks)
        assert "2 candidate cause(s) were recorded as inconclusive" in disclosure
        assert "deliberately does not narrow" in disclosure

    def test_the_disclosure_admits_what_independence_means_here(self) -> None:
        """Counting by source is a floor, not a definition. Real evidential independence needs a
        comparison series this estate does not have, and claiming otherwise would overstate the
        check."""
        result = corroborate(_report(_hypothesis(supporting=[POSTHOG])), TOOL_OF)
        disclosure = " ".join(risk.description for risk in result.report.risks)
        assert "a floor rather than a definition" in disclosure
        assert "two readings from one connector are one line" in disclosure
