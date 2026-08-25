"""What it would take to establish the cause a report names, and why it cannot.

`cortex.analysis.identifiability` held eight predicates and ninety tests and was called from
nowhere, which is the most expensive kind of unfinished work: a thing that looks done. This is
the wiring, and the division of labour is the part worth getting right.

**This attaches a disclosure; it does not delete a claim.** The sufficiency gate holds the veto.
This answers a different question -- whether a causal claim is available *in principle* from what
we can observe -- and today the answer is almost always no, because we have zero admissible
control series and Abadie's placebo floor `1/(J+1)` is therefore 1. Wiring that as a second veto
would refuse every causal answer Cortex gives until per-region series exist: defensible under the
worst-domain composition rule, and useless.
"""

from __future__ import annotations

import uuid
from datetime import date

from cortex.analysis.identifiability import Control
from cortex.db.models import ToolCall
from cortex.reports.identifiability import (
    CHANGE_RECORD_CAPABILITIES,
    identifiability_risks,
    ledger_from_calls,
)
from cortex.reports.schema import (
    Claim,
    Confidence,
    Hypothesis,
    InvestigationReport,
    Verdict,
)


def _call(tool: str, capability: str, *, succeeded: bool = True) -> ToolCall:
    return ToolCall(tool_name=tool, capability=capability, succeeded=succeeded, params={})


def _report(*hypotheses: Hypothesis) -> InvestigationReport:
    evidence = [uuid.uuid4()]
    return InvestigationReport(
        question="Why did signups fall in mid-June?",
        executive_summary=[Claim(text="Signups fell.", evidence_ids=evidence)],
        hypotheses=list(hypotheses),
        confidence=Confidence.MEDIUM,
    )


def _dated(verdict: Verdict = Verdict.SUPPORTED) -> Hypothesis:
    evidence = [uuid.uuid4()]
    return Hypothesis(
        statement="The 16 June copy deploy caused the signup drop.",
        verdict=verdict,
        supporting_evidence_ids=evidence if verdict is Verdict.SUPPORTED else [],
        contradicting_evidence_ids=evidence if verdict is Verdict.CONTRADICTED else [],
        cause_at=date(2026, 6, 16),
        effect_onset=date(2026, 6, 17),
    )


class TestNobodyLookedIsNotNothingHappened:
    """The distinction G7 exists for, made from the audit trail instead of assumed.

    Conflating them reports the second when it means the first, which is how an empty result
    becomes evidence of absence.
    """

    def test_no_change_record_call_means_nobody_looked(self) -> None:
        ledger = ledger_from_calls([_call("posthog", "event_trend")])
        assert ledger.entries is None
        assert not ledger.attested

    def test_a_change_record_call_means_somebody_looked(self) -> None:
        """Even returning nothing. A query for deploys in the window that comes back empty is a
        genuine finding about that window -- what it is not is a licence to say nothing else
        happened, which is why the entries stay empty rather than being filled with an assertion
        nobody made."""
        ledger = ledger_from_calls([_call("github", "recent_prs")])
        assert ledger.entries == ()
        assert ledger.attested

    def test_a_failed_call_is_not_looking(self) -> None:
        """A 403 tells you about the credential, not the window. Counting it would turn a
        permissions problem into a statement about the world."""
        ledger = ledger_from_calls([_call("github", "recent_prs", succeeded=False)])
        assert ledger.entries is None

    def test_every_change_record_capability_counts(self) -> None:
        for tool, capability in CHANGE_RECORD_CAPABILITIES:
            assert ledger_from_calls([_call(tool, capability)]).attested, (tool, capability)

    def test_what_was_consulted_is_recorded_for_the_reader(self) -> None:
        ledger = ledger_from_calls([_call("posthog", "annotations"), _call("github", "commits")])
        assert ledger.consulted == ("github.commits", "posthog.annotations")


class TestItSaysWhatIsMissing:
    def test_an_unestablishable_cause_is_disclosed(self) -> None:
        risks = identifiability_risks(_report(_dated()), [_call("github", "recent_prs")])
        assert len(risks) == 1
        text = risks[0].description
        assert "cannot be established as the cause" in text
        assert "no admissible control series" in text

    def test_it_names_the_number_that_would_change_it(self) -> None:
        """A refusal that does not say what is missing is the failure mode to design against.
        Nineteen is quotable and actionable in a way "insufficient data" is not."""
        risks = identifiability_risks(_report(_dated()), [_call("github", "recent_prs")])
        assert "Nineteen" in risks[0].description

    def test_a_missing_ledger_is_its_own_line(self) -> None:
        risks = identifiability_risks(_report(_dated()), [])
        text = risks[0].description
        assert "no confounder ledger" in text
        assert "an empty ledger is a claim" in text

    def test_consulting_the_change_records_removes_that_line(self) -> None:
        with_ledger = identifiability_risks(_report(_dated()), [_call("posthog", "annotations")])[
            0
        ].description
        assert "no confounder ledger" not in with_ledger
        assert "Change records consulted: posthog.annotations" in with_ledger

    def test_an_admissible_control_removes_that_line_too(self) -> None:
        """The escape hatch is real, not hypothetical: one attested series is all it takes for
        this refusal to stop firing."""
        risks = identifiability_risks(
            _report(_dated()),
            [_call("posthog", "annotations")],
            controls=(Control("eu-signups", correlation=0.86, attested_unexposed=True),),
        )
        assert risks == []

    def test_it_is_phrased_as_what_is_missing_not_as_blame(self) -> None:
        """ "No comparison series exists" is a fact about the estate. "The analyst failed to find
        one" would be a fact about the report, and it would be wrong."""
        text = identifiability_risks(_report(_dated()), [_call("github", "commits")])[0].description
        assert "What is missing" in text
        assert "failed" not in text.lower()


class TestItOnlyAsksAboutAssertedCauses:
    def test_a_contradicted_hypothesis_is_not_assessed(self) -> None:
        """Not an assertion, so demanding identifiability for it would refuse the report for
        correctly declining to commit."""
        assert identifiability_risks(_report(_dated(Verdict.CONTRADICTED)), []) == []

    def test_an_inconclusive_hypothesis_is_not_assessed(self) -> None:
        assert identifiability_risks(_report(_dated(Verdict.INCONCLUSIVE)), []) == []

    def test_an_undated_hypothesis_is_not_assessed(self) -> None:
        """Without a date there is no named intervention, and the system must never invent one
        by picking its own largest changepoint and testing that."""
        undated = Hypothesis(
            statement="Something in marketing.",
            verdict=Verdict.SUPPORTED,
            supporting_evidence_ids=[uuid.uuid4()],
        )
        assert identifiability_risks(_report(undated), []) == []

    def test_a_report_with_no_hypotheses_costs_nothing(self) -> None:
        assert identifiability_risks(_report(), []) == []

    def test_two_asserted_causes_get_two_disclosures(self) -> None:
        second = Hypothesis(
            statement="The pricing change caused it.",
            verdict=Verdict.SUPPORTED,
            supporting_evidence_ids=[uuid.uuid4()],
            cause_at=date(2026, 6, 10),
            effect_onset=date(2026, 6, 17),
        )
        risks = identifiability_risks(_report(_dated(), second), [])
        assert len(risks) == 2
