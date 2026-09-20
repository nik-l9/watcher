"""How much report a question deserves.

The asymmetry is the whole design, and it is what these tests are aimed at: a padded
answer to a lookup wastes a reader's time, while a thin answer to "why did signups fall" is
the product not working. So the classifier is allowed to be wrong in one direction only,
and the tests that matter most are the ones where a question *looks* like a lookup and is
not — "did signups fall last week, and why?" opens with "did" and must still get the full
report.
"""

from __future__ import annotations

import uuid

import pytest

from cortex.agents.investigator import _drafting_instruction
from cortex.eval.fixtures import SCENARIOS, by_name
from cortex.eval.scorer import Scorer
from cortex.reports.schema import (
    Claim,
    Confidence,
    InvestigationReport,
    PremiseVerdict,
)
from cortex.reports.shape import (
    AMBIGUITY_GUIDANCE,
    GUIDANCE,
    Ambiguity,
    Shape,
    ambiguities,
    shape_for,
)


class TestFactualQuestions:
    @pytest.mark.parametrize(
        "question",
        [
            "Check if we have integrated the Apollo company deanonymiser in our website",
            "do we have HubSpot connected?",
            "which plan is Acme on?",
            "when did the mobile onboarding ship?",
            "how many deploys went out in July?",
            "list the PostHog projects we have access to",
            "is there a Slack channel for growth?",
            "can you confirm we track signup events in GA4?",
            "who owns the onboarding funnel?",
        ],
    )
    def test_a_lookup_is_factual(self, question: str) -> None:
        assert shape_for(question) is Shape.FACTUAL


class TestCausalQuestions:
    @pytest.mark.parametrize(
        "question",
        [
            "Why did signups fall last week?",
            "what caused the drop in mobile conversion?",
            "explain the spike in trial starts",
            "signups are down 12%, diagnose it",
            "GA4 sessions moved sharply on the 14th — what happened?",
            "investigate the conversion regression",
        ],
    )
    def test_a_causal_question_is_causal(self, question: str) -> None:
        assert shape_for(question) is Shape.CAUSAL

    @pytest.mark.parametrize(
        "question",
        [
            "did signups fall last week, and why?",
            "do we know what caused the drop?",
            "check whether the deploy on the 14th caused the decline",
            "is there a reason conversion dropped?",
            "when did signups start falling, and what changed?",
        ],
    )
    def test_a_lookup_opener_does_not_win_over_a_causal_word(self, question: str) -> None:
        """The regression this file exists to prevent. Each of these opens with "did", "do
        we", "check" or "is there" and is a causal question. A leading-word rule alone
        would have shortened exactly the reports that must not be short."""
        assert shape_for(question) is Shape.CAUSAL

    @pytest.mark.parametrize("question", ["", "   ", "signups", "the onboarding modal"])
    def test_an_unrecognised_question_gets_the_full_report(self, question: str) -> None:
        """Defaulting to CAUSAL makes the only mistake available the harmless one."""
        assert shape_for(question) is Shape.CAUSAL

    def test_every_causal_eval_scenario_is_causal(self) -> None:
        """The suite measures accuracy, completeness and actionability on causal
        investigations. If this change reclassified one of them as a lookup, the guidance
        would tell the model to leave hypotheses and recommendations empty and the scores
        would fall for a reason that had nothing to do with investigating.

        The false-premise scenario is excluded because it is the exception on purpose: *"Did our
        signups fall from last month?"* is a check, and being classified as one is the behaviour
        it tests. `_completeness` knows the difference, so it is not scored for the sections a
        factual answer is told to leave out."""
        for scenario in SCENARIOS:
            if scenario.ground_truth.is_false_premise:
                continue
            assert shape_for(scenario.question) is Shape.CAUSAL, scenario.name

    def test_the_false_premise_scenario_is_factual(self) -> None:
        """The other half of the same claim, asserted rather than assumed: if this scenario ever
        routes as causal again, the analyst will be told to hunt a cause for a fall that did not
        happen, which is precisely the live failure."""
        premise = [s for s in SCENARIOS if s.ground_truth.is_false_premise]
        assert premise, "the suite has no false-premise scenario"
        for scenario in premise:
            assert shape_for(scenario.question) is Shape.FACTUAL, scenario.name


class TestTheGuidanceReachesTheModel:
    def test_a_factual_question_is_told_to_leave_hypotheses_empty(self) -> None:
        instruction = _drafting_instruction("do we have HubSpot connected?", [], "answered")
        assert "hypotheses empty" in instruction

    def test_a_causal_question_is_told_to_record_what_it_ruled_out(self) -> None:
        instruction = _drafting_instruction("why did signups fall?", [], "answered")
        assert "ruled out" in instruction
        assert "hypotheses empty" not in instruction

    def test_the_grounding_rules_survive_either_shape(self) -> None:
        """The shape guidance is added to the drafting instruction, not substituted for it.
        A shorter report is still a cited one — an instruction that displaced the citation
        rules would trade readability for the thing the product is."""
        for question in ("do we have HubSpot connected?", "why did signups fall?"):
            instruction = _drafting_instruction(question, [], "answered")
            assert "evidence_ids" in instruction
            assert "will be removed before the report is shown" in instruction.replace("\n", " ")

    def test_both_shapes_are_described(self) -> None:
        assert set(GUIDANCE) == set(Shape)


class TestAYesNoQuestionAboutAMovement:
    """The failure this split exists for, observed live in Slack.

    Asked *"@Cortex did our signups fell from last month?"* the analyst produced four paragraphs
    about mid-June, GitHub deploy searches and Slack incident hunts — and never said yes or no. The
    honest answer was "no: 619 events over 12 days against 4,849 over 31 is a calendar artefact,
    not a fall", and it was in the *fourth* bullet, after a figure that appeared to confirm the
    premise.

    The cause was a single combined pattern. `fall`, `fell` and `drop` sat in the causal set so that
    a bare *"signups are down 12%"* would be treated as a request to explain, and that made every
    yes/no **check** about a movement causal too. A causal report is exactly the wrong shape for a
    question that asks *whether* something happened: it instructs the analyst to hunt for a cause
    for an event nobody has established.

    So the words are split. An explicit causal *request* wins over everything; a movement word
    describes the subject and no longer overrides a lookup opener.
    """

    @pytest.mark.parametrize(
        "question",
        [
            "did our signups fell from last month?",
            "did signups fall last month?",
            "have signups dropped?",
            "is conversion down?",
            "are signups falling?",
            "has revenue declined since June?",
            "did mobile conversion drop last week?",
            "was there a spike in errors yesterday?",
        ],
    )
    def test_a_check_about_a_movement_is_factual(self, question: str) -> None:
        assert shape_for(question) is Shape.FACTUAL

    @pytest.mark.parametrize(
        "question",
        [
            "did signups fall last week, and why?",
            "did signups fall, and what changed?",
            "have signups dropped? what happened?",
            "is conversion down because of the deploy?",
            "has revenue declined — any reason?",
        ],
    )
    def test_a_causal_request_still_wins_over_the_opener(self, question: str) -> None:
        """The asymmetry that has to hold: a question can open as a check and still ask for a
        cause, and getting that wrong costs a real investigation."""
        assert shape_for(question) is Shape.CAUSAL

    @pytest.mark.parametrize(
        "question",
        [
            "signups are down 12% this month",
            "conversion fell off a cliff after Tuesday",
            "the trial-start trend broke last week",
        ],
    )
    def test_a_bare_statement_of_a_movement_is_still_causal(self, question: str) -> None:
        """Why the movement words cannot simply be deleted. A statement with no interrogative is a
        request to explain it, and this is the case they were added for."""
        assert shape_for(question) is Shape.CAUSAL


class TestFalsePremises:
    """A question can assume something untrue, and the answer has to say so first.

    This failure passes every grounding layer we have, which is why it needs its own guidance. Each
    claim really is supported by the evidence it cites; the gate resolves every id, the verifier
    finds nothing overstated. The report is simply answering a question whose premise nobody
    checked, and a grounded answer to the wrong question still misinforms.

    Our live example: *"did our signups fell from last month?"* assumes a fall. The delivered answer
    led with "619 events for Aug 1–12 compared to July's total of 4,849", which reads as
    confirmation, and corrected itself in the same sentence. A reader who stops after one line has
    been misinformed by a report that was accurate throughout.
    """

    def test_the_factual_guidance_names_the_premise_problem(self) -> None:
        guidance = GUIDANCE[Shape.FACTUAL]
        assert "assumes something that is not true" in guidance
        assert "in the first " in guidance

    def test_it_forbids_leading_with_a_confirming_figure(self) -> None:
        """The specific defect. Correcting a misleading number *after* stating it is not the same
        as not leading with it."""
        guidance = GUIDANCE[Shape.FACTUAL]
        assert "Do not lead with a figure" in guidance
        assert "correct it afterwards" in guidance

    def test_it_names_the_artefacts_that_look_like_movements(self) -> None:
        """The three that produced this incident and will produce the next one: unequal periods,
        incomplete data, and a metric that changed definition. Each is the answer, not a caveat."""
        guidance = GUIDANCE[Shape.FACTUAL]
        for artefact in ("different lengths", "incomplete", "changed definition"):
            assert artefact in guidance

    def test_both_shapes_carry_the_premise_check(self) -> None:
        """It was attached to FACTUAL alone, which is backwards.

        A yes/no question wears its premise openly and can be answered without granting it —
        "did signups fall" asks. A causal question has already granted it: "why did signups
        fall" cannot be answered at all unless they fell. So the check was missing from exactly
        the shape whose questions always carry a premise, and the paragraph's own second example
        ("why is checkout slower on mobile") is a causal question.
        """
        for shape in (Shape.FACTUAL, Shape.CAUSAL):
            assert "assumes something that is not true" in GUIDANCE[shape], shape

    def test_a_why_question_receives_it(self) -> None:
        """The routing that produced the live failure. Asked "did our signups fall from last
        month" the analyst went FACTUAL, checked the premise and answered correctly. Asked to
        compare three months and correlate the movement with product changes it went CAUSAL, was
        handed no premise instruction, and argued a shared cause for two incidents a week apart.
        """
        for question in (
            "why did signups fall?",
            "why did our signups drop last month",
            "what caused the signup drop",
            "can you compare the last 3 months signups, whats the trend, is there any spikes, "
            "or dips, and can we correlate those from any changes we made in product",
        ):
            assert shape_for(question) is Shape.CAUSAL, question
            assert "assumes something that is not true" in GUIDANCE[shape_for(question)]

    def test_the_causal_guidance_orders_the_effect_before_the_cause(self) -> None:
        """A cause correctly argued for an effect that did not occur is wrong in a way no
        amount of evidence behind it can fix, so establishing the effect is step zero rather
        than an assumption inherited from the question."""
        guidance = GUIDANCE[Shape.CAUSAL]
        assert "Establish the effect before explaining it" in guidance
        assert "already granted that it happened" in guidance

    def test_the_causal_guidance_refuses_to_merge_two_incidents(self) -> None:
        """The exact wrong answer: a signup cessation on 2026-08-04 explained by a pageview
        collapse that began on 08-11, while pageviews were at 14,197/day on 08-04. Two metrics
        turning down on different dates are two incidents."""
        guidance = GUIDANCE[Shape.CAUSAL]
        assert "Two things going wrong are not one thing going wrong" in guidance
        assert "separate incidents until something ties them together" in guidance
        assert "not corroboration" in guidance


class TestThePremiseVerdictIsStatedNotInferred:
    """The keyword list scored a correct refutation as a failure.

    A live attempt opened "No -- this appears to be an artifact of an incomplete month, not a
    real drop", and `accuracy` scored 0.50 for *never having refuted the premise*: the twelve
    accepted denial phrasings included "did not drop" and "have not dropped" but not "not a real
    drop". The placement dimension, reading the same sentence, scored 1.00 for refuting it in the
    executive summary. Two dimensions contradicting each other about one sentence.
    """

    _TEXT = (
        "No -- this appears to be an artifact of an incomplete month, not a real drop: "
        "August 2026 shows only 1,884 signups versus 4,849 in July, but the August figure "
        "only covers data through August 12."
    )

    def _report(self, **overrides: object) -> InvestigationReport:
        import uuid

        return InvestigationReport(
            question="Did our signups fall from last month?",
            executive_summary=[Claim(text=self._TEXT, evidence_ids=[uuid.uuid4()])],
            confidence=Confidence.HIGH,
            **overrides,  # type: ignore[arg-type]
        )

    def test_the_keyword_fallback_still_gets_this_wrong(self) -> None:
        """Kept as a fallback so an older report scores as it did, and pinned so nobody mistakes
        it for the primary path."""
        truth = by_name("partial_month_false_premise").ground_truth
        dimension = Scorer()._premise_accuracy(truth, self._report(), self._TEXT.lower())
        assert dimension.score == 0.5
        assert "never refuted the premise" in dimension.detail

    def test_the_stated_verdict_scores_it_correctly(self) -> None:
        truth = by_name("partial_month_false_premise").ground_truth
        report = self._report(
            premise_asserted=True, premise_measured=True, premise_contradicted=True
        )
        dimension = Scorer()._premise_accuracy(truth, report, self._TEXT.lower())
        assert dimension.score == 1.0
        assert dimension.detail == "refuted the premise"

    def test_stating_the_wrong_verdict_still_fails(self) -> None:
        """The field must not become a rubber stamp: a report that says the premise holds, on a
        scenario whose premise is false, is wrong however it phrases its prose."""
        truth = by_name("partial_month_false_premise").ground_truth
        report = self._report(
            premise_asserted=True, premise_measured=True, premise_contradicted=False
        )
        dimension = Scorer()._premise_accuracy(truth, report, self._TEXT.lower())
        assert dimension.score == 0.0
        assert "the report says the premise is holds" in dimension.detail

    def test_unverifiable_is_not_a_refutation(self) -> None:
        """ "The evidence cannot settle it" is a different answer from "the evidence contradicts
        it", and this scenario's evidence does settle it."""
        truth = by_name("partial_month_false_premise").ground_truth
        report = self._report(premise_asserted=True, premise_measured=False)
        assert Scorer()._premise_accuracy(truth, report, self._TEXT.lower()).score == 0.0

    def test_both_shapes_are_told_to_set_it(self) -> None:
        """Either kind of question can carry a premise, so the instruction cannot live on one."""
        for shape in (Shape.FACTUAL, Shape.CAUSAL):
            for field in ("`premise_checked`", "`premise_asserted`", "`premise_measured`"):
                assert field in GUIDANCE[shape], (shape, field)

    def test_both_shapes_are_warned_about_the_answer_that_gets_skipped(self) -> None:
        """`premise_measured` is the one the measurement says is being skipped: 15 of 16
        reports about windows their project never collected in said the evidence contradicts
        the premise. The guidance has to name that case, not just list the fields."""
        for shape in (Shape.FACTUAL, Shape.CAUSAL):
            guidance = GUIDANCE[shape]
            assert "no rows for the window asked about has not measured it" in guidance, shape
            assert "cannot tell from here" in guidance, shape

    def test_both_shapes_are_told_which_order_to_answer_them_in(self) -> None:
        """The prose has to agree with the schema, or it argues against the field order.

        `premise_checked` is declared before `premise` so that a single-pass decoder writes the
        reasoning before the verdict -- an attempt that answered the verdict first set `holds`
        and then wrote "No -- signups did not fall". An instruction naming them the other way
        round would be telling the model to do what the schema is arranged to prevent.
        """
        for shape in (Shape.FACTUAL, Shape.CAUSAL):
            guidance = GUIDANCE[shape]
            assert "`premise_checked` first, then answer the three questions" in guidance, shape
            # The three named in dependency order, matching the field order in the schema.
            order = [
                guidance.index(f"`{name}`")
                for name in ("premise_asserted", "premise_measured", "premise_contradicted")
            ]
            assert order == sorted(order), shape


class TestStatingTheAssumptionRatherThanAsking:
    """The free half of decision 9, and the research says it is the half that matters.

    Every term in Horvitz's expected-utility calculation pushes our ask band narrow: a question
    in a public Slack channel is latency-visible and breaks the one promise the product makes,
    and our users are by construction in a hurry, which *lowers* the threshold for acting. What
    is cheap is *stating* the reading taken — the gap between a wrong reading delivered silently
    and one delivered with its assumption named is enormous and entirely under our control.

    And the detection is a predicate, never a model's sense of vagueness. Six independent
    measurements say a model cannot estimate its own need to clarify: clarification-need F1 of
    0.33-0.37, one benchmark whose R-squared is negative, ambiguity detection at 54%.
    """

    def test_a_movement_with_nothing_to_compare_it_against_is_flagged(self) -> None:
        """ "Why did signups fall" is unanswerable as asked -- against last week, last month, last
        year? Each is a different question with a different answer."""
        assert ambiguities("why did signups fall?") == (Ambiguity.UNSTATED_BASELINE,)

    def test_a_named_period_settles_it(self) -> None:
        assert ambiguities("why did signups fall in August 2026?") == ()

    def test_an_explicit_comparison_settles_it(self) -> None:
        for question in (
            "why did signups fall versus July?",
            "why did signups drop week over week?",
            "why did signups fall compared to last month?",
            "did signups drop last month?",
        ):
            assert ambiguities(question) == (), question

    def test_a_question_naming_no_movement_is_not_flagged(self) -> None:
        """A lookup has no movement to measure, so there is no baseline to be missing."""
        assert ambiguities("which plan is Acme on?") == ()

    def test_the_guidance_says_state_it_rather_than_ask(self) -> None:
        """The distinction the research turns on. Asking costs the 90-second promise; stating
        costs nothing and lets a reader correct a wrong reading in one line."""
        guidance = AMBIGUITY_GUIDANCE[Ambiguity.UNSTATED_BASELINE]
        assert "Do not ask which was meant" in guidance
        assert "state one and let the reader correct you" in guidance

    def test_the_guidance_asks_for_it_in_two_places(self) -> None:
        """First sentence so a skimming reader sees it, risks so it survives summarisation."""
        guidance = AMBIGUITY_GUIDANCE[Ambiguity.UNSTATED_BASELINE]
        assert "first sentence" in guidance
        assert "in the risks" in guidance

    def test_it_reaches_the_live_prompt_only_when_it_applies(self) -> None:

        from cortex.agents.investigator import _drafting_instruction

        needle = "without saying what to measure it against"
        assert needle in _drafting_instruction("why did signups fall?", ["x"], "budget")
        assert needle not in _drafting_instruction(
            "why did signups fall in August 2026?", ["x"], "budget"
        )


class TestACausalReportMustNameTheChange:
    """A report that found the cause and described it has withheld the useful part.

    Run 26's `onboarding_regression` attempt identified the right pull request — it called
    `commit_diff` on the sha and `pull_request_activity` on the number — and then wrote "a
    mobile-only onboarding modal rework merged on 14 July". Correct, cited, and not actionable:
    the reader has to go looking for what the analyst already had in hand. The other attempt
    wrote "PR #913" and scored 0.99 against 0.88.
    """

    def test_the_causal_guidance_asks_for_the_identifier(self) -> None:
        guidance = GUIDANCE[Shape.CAUSAL]
        assert "Name the change, not its category" in guidance
        # The three forms the evidence actually carries, so the instruction cannot be read as
        # being about prose style.
        for form in ("number", "sha", "tag"):
            assert form in guidance, form

    def test_a_factual_question_is_not_asked_for_one(self) -> None:
        """Nothing to name: a factual question is not attributing a movement to a change, and
        an instruction about identifying causes would be noise in that shape's guidance."""
        assert "Name the change, not its category" not in GUIDANCE[Shape.FACTUAL]


class TestTheCompoundClaimRule:
    """A sentence relating two observations has to cite both, and losing these costs the answer.

    Each claim is verified against only the ids it carries. The drafter kept citing the one
    observation a sentence appears to be *about*: "the campaign ended on 14 June, one day before
    the drop began", citing the Slack message alone. The verifier removes it correctly — when the
    drop began is not in that message — and the delivered summary is left describing a 68.4%
    collapse in paid search with nothing about the exhausted budget behind it.

    Measured twice on `campaign_traffic_drop` in run 32, and invisible to `accuracy`, which
    searches the whole report and finds the cause surviving in a finding.
    """

    def test_the_rule_reaches_the_drafter(self) -> None:
        import uuid

        from cortex.agents.investigator import _drafting_instruction

        text = _drafting_instruction("Why did signups fall?", [uuid.uuid4()], "confident")
        assert "relates two observations must cite both" in text

    def test_it_names_the_relationships_that_need_two_ids(self) -> None:
        """Before/after/during/because, and comparisons. Naming the shapes rather than stating
        the principle alone, because the principle was already there — "the observations that
        establish it" — and the drafter still cited one."""
        import uuid

        from cortex.agents.investigator import _drafting_instruction

        text = _drafting_instruction("Why did signups fall?", [uuid.uuid4()], "confident")
        for relationship in ("before", "after", "during", "because"):
            assert relationship in text
        assert "comparing two figures" in text

    def test_it_offers_the_narrower_claim_as_the_way_out(self) -> None:
        """Without somewhere to go, the instruction is a rule the drafter can only break. A
        claim split into the half it can establish survives; the fuller one does not."""
        import uuid

        from cortex.agents.investigator import _drafting_instruction

        text = _drafting_instruction("Why did signups fall?", [uuid.uuid4()], "confident")
        assert "state only the half you can establish" in text.lower()

    def test_the_instruction_asks_for_a_finding_not_the_question(self) -> None:
        """Reason-before-verdict only works if the reason is a reason.

        `premise_checked` used to ask for "the assertion you tested", and across twenty
        attempts at one scenario every single one wrote the question back — *"Signups fell in
        August 2026 compared to July 2026."* A faithful restatement carrying no finding, so
        moving `premise` after it bought the verdict an input that says nothing, and two of
        those twenty still set a verdict their own summary contradicted.

        Asserted on the prose rather than on an outcome because the outcome needs sixty
        attempts a side to measure and this does not: what the field asks for is a fact about
        the instruction.
        """
        for shape in (Shape.FACTUAL, Shape.CAUSAL):
            guidance = GUIDANCE[shape]
            assert "write what checking" in guidance and "*found*" in guidance, shape
            assert "Do not write the question back" in guidance, shape


class TestPartialCoverageIsStillMeasured:
    """`premise_measured` must not be read as "every day is present".

    Measured: a report about `$autocapture` for March against April answered
    `premise_measured` false *and* `premise_contradicted` true, because the series stopped
    three days before the requested end date. Its reading of the data was right -- the metric
    rose -- and the derivation turned that correct refutation into an abstention.

    The distinction matters in both directions, which is why the guidance cannot simply say
    "partial windows are fine": `partial_month_false_premise` exists because a twelve-day
    month compared against a thirty-one-day one makes an apparent fall an artefact, and there
    the right answer is that the premise is false, not that nothing was measured.
    """

    def test_both_shapes_say_a_short_tail_is_still_measurable(self) -> None:
        for shape in (Shape.FACTUAL, Shape.CAUSAL):
            guidance = GUIDANCE[shape]
            assert "not whether the window is complete" in guidance, shape
            assert "`data_quality`" in guidance, shape

    def test_both_shapes_name_the_incoherent_pair(self) -> None:
        """The combination that produced the bug is called out by name, since a model that
        answers both that way has found the movement and is describing a gap."""
        for shape in (Shape.FACTUAL, Shape.CAUSAL):
            assert "you have almost certainly measured" in GUIDANCE[shape], shape

    def test_an_incoherent_pair_still_abstains_rather_than_asserting(self) -> None:
        """Derivation unchanged: if the two answers disagree, the cautious one wins.

        The guidance is what stops the pair being emitted. Were it emitted anyway, deriving
        `false` from "I could not measure it" would assert a contradiction the report has just
        said it could not observe.
        """
        report = InvestigationReport(
            question="Why did signups fall in March?",
            executive_summary=[Claim(text="A claim.", evidence_ids=[uuid.uuid4()])],
            premise_asserted=True,
            premise_measured=False,
            premise_contradicted=True,
        )
        assert report.premise is PremiseVerdict.UNVERIFIABLE


class TestAQuestionAskingForANumberClaimsNothing:
    """`none_asserted` was unreachable in practice, which makes a schema value dead.

    Measured over thirty-two cases: eight of eight questions that assert nothing -- "which
    month had the highest daily rate", "between A and B, which was higher" -- were answered
    `holds`, every one diverging on `premise_asserted`, and every one naming the right answer.
    Eight for eight is systematic rather than noise.

    It misleads no reader, which is why it is scored benign and why it waited behind the
    errors that do. But a verdict on an assertion nobody made is still a verdict nobody can
    check.
    """

    def test_both_shapes_say_a_question_asking_for_a_number_asserts_nothing(self) -> None:
        for shape in (Shape.FACTUAL, Shape.CAUSAL):
            guidance = GUIDANCE[shape]
            assert "asks for a number is not making a claim" in guidance, shape
            assert "Which month had the" in guidance, shape

    def test_both_shapes_give_the_contrasting_case(self) -> None:
        """A rule with only one side of the distinction invites over-applying it: the fix for
        a dead `none_asserted` must not make `holds` and `false` unreachable instead."""
        for shape in (Shape.FACTUAL, Shape.CAUSAL):
            guidance = GUIDANCE[shape]
            assert "could contradict" in guidance, shape
            assert "signups fell" in guidance, shape

    def test_a_question_claiming_nothing_derives_none_asserted(self) -> None:
        report = InvestigationReport(
            question="Which month had the highest daily rate of signups?",
            executive_summary=[
                Claim(
                    text="June 2026 had the highest daily rate, at 412 signups a day.",
                    evidence_ids=[uuid.uuid4()],
                )
            ],
            premise_asserted=False,
        )
        assert report.premise is PremiseVerdict.NONE_ASSERTED


class TestATruncatedMonthIsStillMeasurable:
    """The regression this wording caused, and the lesson it had been erasing.

    `partial_month_false_premise` exists because August holds twelve days against July's
    thirty-one, so the totals cannot be compared and the daily rate can -- and the rate is
    flat, which makes the asserted fall false rather than unmeasurable. After
    `premise_measured` was introduced, both partial-month scenarios answered it false and
    came out `unverifiable`, taking the hand-written suite from 9/9 to 7/9.

    Declining is not the safe answer here. A reader told "we cannot tell" goes looking for a
    decline that the data already rules out.
    """

    def test_both_shapes_say_a_half_month_is_measurable_by_rate(self) -> None:
        for shape in (Shape.FACTUAL, Shape.CAUSAL):
            guidance = GUIDANCE[shape]
            assert "half-finished month is measurable" in guidance, shape
            assert "compare daily rates instead" in guidance, shape

    def test_both_shapes_keep_the_genuine_absence_case(self) -> None:
        """The fix must not undo what it is qualifying: no rows still means not measured."""
        for shape in (Shape.FACTUAL, Shape.CAUSAL):
            guidance = GUIDANCE[shape]
            assert "no rows at all for the window" in guidance, shape
            assert "cannot tell from here" in guidance, shape

    def test_a_truncated_window_whose_rate_settles_it_derives_false(self) -> None:
        report = InvestigationReport(
            question="Did our signups fall from last month?",
            executive_summary=[
                Claim(
                    text="Signups did not fall; the daily rate is flat at about 157.",
                    evidence_ids=[uuid.uuid4()],
                )
            ],
            premise_asserted=True,
            premise_measured=True,
            premise_contradicted=True,
        )
        assert report.premise is PremiseVerdict.FALSE
