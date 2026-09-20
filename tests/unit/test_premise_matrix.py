"""The premise matrix must separate the two errors that are not interchangeable.

A report that states `holds` where the data says `false` has delivered a confident falsehood.
One that states `unverifiable` where the data answers has only been unhelpful. A single
accuracy figure prices them identically, and an abstention threshold fitted on it will buy
one down by selling the other up -- which is the failure mode the whole calibration exists
to avoid. So the split is tested, not just the total.

An undelivered report is a third thing again, and must never be counted as a wrong verdict:
a run that could not answer is an outage, not a judgement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from cortex.eval.premise_matrix import (
    confusion,
    diverged,
    outcomes_for,
    premise_of,
    render,
)


@dataclass
class FakeOutcome:
    scenario: str
    report_json: str | None


def report(premise: str) -> str:
    return json.dumps({"premise": premise, "executive_summary": "..."})


LABELS = {
    "a_holds": "holds",
    "b_false": "false",
    "c_unver": "unverifiable",
    "d_none": "none_asserted",
}


class TestReadingAStatedVerdict:
    def test_a_delivered_report_yields_its_premise(self):
        assert premise_of(report("false")) == "false"

    def test_an_absent_or_unparseable_report_yields_nothing(self):
        assert premise_of(None) is None
        assert premise_of("not json") is None

    def test_a_value_outside_the_enum_is_not_accepted(self):
        # A field carrying something the enum does not define is a schema problem, and
        # silently counting it as a verdict would hide that behind a plausible number.
        assert premise_of(json.dumps({"premise": "probably"})) is None


class TestPairingAttemptsWithLabels:
    def test_hand_written_scenarios_are_dropped_rather_than_guessed_at(self):
        outcomes = outcomes_for(
            LABELS,
            [FakeOutcome("a_holds", report("holds")), FakeOutcome("onboarding_regression", None)],
        )
        assert [o.scenario for o in outcomes] == ["a_holds"]

    def test_an_undelivered_report_is_recorded_as_undelivered(self):
        outcomes = outcomes_for(LABELS, [FakeOutcome("a_holds", None)])
        assert outcomes[0].stated is None
        assert confusion(outcomes)[("holds", "none")] == 1


class TestTheTwoErrorsAreCountedApart:
    def _rendered(self, pairs):
        return render(outcomes_for(LABELS, [FakeOutcome(n, report(v)) for n, v in pairs]))

    def test_a_wrong_confident_verdict_is_named_as_such(self):
        text = self._rendered([("b_false", "holds")])
        assert "confidently wrong (all)   1" in text
        assert "over-abstained            0" in text

    def test_an_unnecessary_abstention_is_counted_separately(self):
        text = self._rendered([("a_holds", "unverifiable")])
        assert "over-abstained            1" in text
        assert "confidently wrong (all)   0" in text

    def test_a_correct_abstention_is_neither(self):
        text = self._rendered([("c_unver", "unverifiable")])
        assert "confidently wrong (all)   0" in text
        assert "over-abstained            0" in text
        assert "correct        1/1" in text

    def test_an_undelivered_report_is_not_counted_as_a_wrong_verdict(self):
        text = render(outcomes_for(LABELS, [FakeOutcome("a_holds", None)]))
        assert "delivered      0/1" in text
        assert "confidently wrong (all)   0" in text


class TestAnEmptyRunSaysSo:
    def test_no_generated_cases_is_not_a_zero_score(self):
        assert "No generated cases" in render(())


class TestSeverityIsGradedNotJustCorrectness:
    """Three wrong answers, three different costs.

    Observed on the first live run: a wh-question labelled `none_asserted` was answered
    `holds`, with the right month named. Counting that the same as accepting a false premise
    would push a threshold toward abstaining on ordinary questions to buy down an error that
    misled nobody. Folding it into "correct" would hide a real schema confusion. It gets its
    own line, and the two errors that do mislead get theirs.
    """

    def _rendered(self, pairs):
        return render(outcomes_for(LABELS, [FakeOutcome(n, report(v)) for n, v in pairs]))

    def test_a_wh_question_answered_holds_is_benign_not_wrong(self):
        text = self._rendered([("d_none", "holds")])
        assert "benign         1" in text
        assert "confidently wrong (all)   0" in text

    def test_going_along_with_a_false_premise_is_named_on_its_own_line(self):
        text = self._rendered([("b_false", "holds")])
        assert "accepted a false premise  1" in text

    def test_answering_a_window_with_no_data_is_named_on_its_own_line(self):
        text = self._rendered([("c_unver", "holds")])
        assert "answered the unreachable  1" in text

    def test_a_benign_confusion_is_not_an_accepted_false_premise(self):
        text = self._rendered([("d_none", "holds"), ("a_holds", "none_asserted")])
        assert "accepted a false premise  0" in text
        assert "benign         2" in text


class TestAVerdictTheEvidenceAlsoSupports:
    """Some cases genuinely admit two readings, and scoring one as wrong fits noise.

    An `unverifiable` case whose window the project *was* collecting through is the example:
    the event has no rows while its siblings have plenty, so either it was not instrumented
    yet -- the data cannot say -- or it was and never fired, in which case the asserted
    movement did not occur. PostHog cannot tell those apart. Five of the first six such cases
    answered `false`, each correctly describing the absence first.

    Counted on its own line: folding it into `correct` would hide that the label and the
    answer disagree, and counting it as an error would spend an abstention threshold's budget
    on a distinction the evidence does not make.
    """

    LABELS = {"amb": {"expected": "unverifiable", "also_acceptable": ["false"]}}

    def test_the_other_reading_is_neither_right_nor_wrong(self):
        text = render(outcomes_for(self.LABELS, [FakeOutcome("amb", report("false"))]))
        assert "defensible     1" in text
        assert "correct        0/1" in text
        assert "confidently wrong (all)   0" in text

    def test_it_is_not_counted_as_answering_the_unreachable(self):
        text = render(outcomes_for(self.LABELS, [FakeOutcome("amb", report("false"))]))
        assert "answered the unreachable  0" in text

    def test_a_verdict_outside_the_accepted_set_is_still_wrong(self):
        text = render(outcomes_for(self.LABELS, [FakeOutcome("amb", report("holds"))]))
        assert "answered the unreachable  1" in text

    def test_a_plain_string_label_still_works(self):
        """Captures written before the richer form existed are still worth re-scoring."""
        text = render(outcomes_for({"amb": "unverifiable"}, [FakeOutcome("amb", report("false"))]))
        assert "answered the unreachable  1" in text


class TestNamingWhichAnswerWentWrong:
    """Decomposing the verdict makes the diagnosis possible; this is what reads it out.

    "Said `false`, wanted `unverifiable`" names a symptom. "Answered `premise_measured` true
    where the evidence does not reach the window" names the step, and a fix can be aimed at a
    step.
    """

    def test_it_names_the_single_answer_that_differs(self):
        report = {
            "premise_asserted": True,
            "premise_measured": True,
            "premise_contradicted": True,
        }
        assert diverged("unverifiable", report) == ("premise_measured",)

    def test_an_answer_the_verdict_does_not_depend_on_is_not_counted(self):
        """Once `premise_measured` is false nothing turns on `premise_contradicted`, and
        reporting it would invent an error out of a field nobody read."""
        report = {
            "premise_asserted": True,
            "premise_measured": False,
            "premise_contradicted": True,
        }
        assert diverged("unverifiable", report) == ()

    def test_a_correct_report_diverges_nowhere(self):
        report = {
            "premise_asserted": True,
            "premise_measured": True,
            "premise_contradicted": False,
        }
        assert diverged("holds", report) == ()

    def test_a_report_without_the_answers_is_not_guessed_at(self):
        """An older capture carries only the verdict, and inventing answers for it would put
        a fabricated diagnosis in a scorecard."""
        assert diverged("holds", {"premise": "false"}) == ()


class TestTheScorecardNamesTheFailingStep:
    """Errors concentrated in one answer are a defect; spread across three they are noise.

    The two call for opposite responses -- one has something to fix, the other says the
    sample is too small to act on -- and a verdict-only matrix cannot tell them apart.
    """

    def _outcome(self, name, expected, asserted, measured, contradicted):
        return FakeOutcome(
            name,
            json.dumps(
                {
                    "premise": {
                        (True, True, True): "false",
                        (True, True, False): "holds",
                        (True, False, True): "unverifiable",
                        (True, False, False): "unverifiable",
                        (False, True, False): "none_asserted",
                    }[(asserted, measured, contradicted)],
                    "premise_asserted": asserted,
                    "premise_measured": measured,
                    "premise_contradicted": contradicted,
                }
            ),
        )

    def test_a_shared_failing_step_is_counted(self):
        outcomes = outcomes_for(
            {"c_unver": "unverifiable", "amb2": "unverifiable"},
            [
                self._outcome("c_unver", "unverifiable", True, True, True),
                self._outcome("amb2", "unverifiable", True, True, True),
            ],
        )
        text = render(outcomes)
        assert "wrong answers by step" in text
        assert "premise_measured       2" in text

    def test_a_correct_run_names_no_step(self):
        outcomes = outcomes_for(
            {"c_unver": "unverifiable"},
            [self._outcome("c_unver", "unverifiable", True, False, False)],
        )
        assert "wrong answers by step" not in render(outcomes)
