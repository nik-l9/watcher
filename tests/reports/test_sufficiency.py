"""The sufficiency gate — may these causal claims stand at all?

ADR 0005 decision 6. A second call, asked separately from the question about the business, with
veto power. The evidence for it is measured rather than intuited: with sufficient context
frontier models still emit a wrong answer 13-25% of the time, they hallucinate more often than
they abstain in exactly that case, and supplying *any* context collapses abstention (Claude 3.5
Sonnet 84.1% to 52%). Our own failure was that shape — thirteen silent series and a stack of
unrelated tool results, and the stack licensed a confident answer.

These tests were written after the module, which is worth saying: the implementation arrived
without them, and two of the behaviours below are the ones a reviewer would most want pinned —
that no causal claim means no call at all, and that a provider failure withholds nothing.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

import pytest

from cortex.agents.llm import LLMError, Usage
from cortex.reports.gate import ReportRejected
from cortex.reports.schema import (
    Claim,
    Confidence,
    Finding,
    Hypothesis,
    InvestigationReport,
    Verdict,
)
from cortex.reports.sufficiency import SUFFICIENCY_PROMPT, SufficiencyDecision, SufficiencyGate
from cortex.reports.verifier import causal_claims, is_causal_claim


def _evidence() -> list[uuid.UUID]:
    return [uuid.uuid4()]


def _report(
    *, summary: list[str], hypotheses: list[Hypothesis] | None = None
) -> InvestigationReport:
    return InvestigationReport(
        question="Why did signups fall in mid-June?",
        executive_summary=[Claim(text=text, evidence_ids=_evidence()) for text in summary],
        hypotheses=hypotheses or [],
        confidence=Confidence.MEDIUM,
    )


class _Recording:
    """An LLM that records whether it was asked anything, and answers a fixed verdict."""

    def __init__(
        self,
        *,
        sufficient: bool = True,
        missing: tuple[str, ...] = (),
        fail: bool = False,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._sufficient = sufficient
        self._missing = missing
        self._fail = fail

    async def structured(self, **kwargs: Any) -> tuple[dict[str, Any], Usage]:
        self.calls.append(kwargs)
        if self._fail:
            raise LLMError("provider unavailable")
        return (
            {
                "sufficient": self._sufficient,
                "missing": list(self._missing),
                "reason": "scripted",
            },
            Usage(input_tokens=10, output_tokens=5),
        )


class TestTheCausalPredicateGatesEverything:
    """One definition, imported by both Phase 1 interventions, because two would drift and the
    drift would be silent."""

    @pytest.mark.parametrize(
        "text",
        [
            "Signups fell because of the 16 June deploy.",
            "The drop was caused by a pricing change.",
            "Conversion was driven by the new CTA.",
            "The refresh led to a fall in signups.",
        ],
    )
    def test_an_assertion_of_cause_is_causal(self, text: str) -> None:
        assert is_causal_claim(text)

    @pytest.mark.parametrize(
        "text",
        [
            "Signups fell 53% starting 2026-06-17.",
            "Pageview volume rose slightly over the same window.",
            "Conversion fell from 5.98% to 4.02%.",
        ],
    )
    def test_a_description_is_not(self, text: str) -> None:
        """Establishing when something moved is a different act from explaining why, and a gate
        that fired on descriptions would refuse the answers that are already honest."""
        assert not is_causal_claim(text)

    @pytest.mark.parametrize(
        "text",
        [
            "The pageview collapse did not cause the signup drop.",
            "We cannot establish a cause from this data.",
            "There is no evidence that the deploy caused the fall.",
            "The drop cannot be attributed to the refresh.",
        ],
    )
    def test_declining_to_name_a_cause_is_not_a_causal_claim(self, text: str) -> None:
        """The refusals this project works hardest to produce contain the word "cause" more
        often than the assertions do. A predicate that counted them would veto exactly the
        reports that were already right."""
        assert not is_causal_claim(text)

    @pytest.mark.parametrize(
        "text",
        [
            "The refresh broke the signup path.",
            "PR 565 broke checkout.",
            "The deploy introduced friction in the signup flow.",
            "Conversion regressed after the refresh.",
        ],
    )
    def test_a_transitive_causal_verb_counts(self, text: str) -> None:
        """The connective list missed these entirely, so a causal claim phrased this way
        escaped both Phase 1 interventions -- no re-derivation and no sufficiency gate. Not a
        hypothetical phrasing: a real report on our own data proposed that PR 567 changed
        download CTAs and was "a plausible source of added friction"."""
        assert is_causal_claim(text)

    @pytest.mark.parametrize(
        "text",
        [
            "The series broke down into three regimes.",
            "Signups broken down by channel show no change.",
        ],
    )
    def test_our_own_descriptive_vocabulary_is_not_causal(self, text: str) -> None:
        """`cortex.analysis.changepoints` segments a series into regimes, and a report saying so
        was scored as asserting a cause by the `broke ` marker. Our own vocabulary tripped our
        own predicate."""
        assert not is_causal_claim(text)

    @pytest.mark.parametrize(
        "text",
        ["Signups reduced by 53%.", "Pageviews increased slightly.", "The change made it worse."],
    )
    def test_ambiguous_verbs_are_left_out_on_purpose(self, text: str) -> None:
        """ "signups reduced" is a movement, "the outage reduced signups" is a cause, and a
        substring match cannot tell a subject from an object. Including them would veto
        descriptive reports -- the failure `_DECLINES_A_CAUSE` prevents from the other side.

        The third case is a real false negative and is recorded as one rather than fixed.
        """
        assert not is_causal_claim(text)

    def test_it_finds_causal_claims_in_findings_too(self) -> None:
        report = InvestigationReport(
            question="Why did signups fall?",
            executive_summary=[Claim(text="Signups fell 53%.", evidence_ids=_evidence())],
            findings=[
                Finding(
                    title="The deploy",
                    claims=[Claim(text="The drop was caused by PR 565.", evidence_ids=_evidence())],
                    confidence=Confidence.MEDIUM,
                )
            ],
            confidence=Confidence.MEDIUM,
        )
        locations = [location for location, _ in causal_claims(report)]
        assert locations == ["findings[0].claims[0]"]


class TestADescriptiveReportPaysNothing:
    """The whole latency argument for this gate.

    It adds a model call to a loop already running 192s against a 90-second budget, so a report
    that names no cause must not make it -- and the tracking-outage answer and the "this is
    within normal variation" answer are both descriptive.
    """

    async def test_no_causal_claim_means_no_call(self) -> None:
        llm = _Recording()
        decision = await SufficiencyGate(llm).assess(  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            investigation_id=uuid.uuid4(),
            question="Did signups fall?",
            report=_report(summary=["Signups fell 53% starting 2026-06-17."]),
        )
        assert llm.calls == []
        assert not decision.needed
        assert not decision.vetoes

    async def test_a_report_that_declines_a_cause_pays_nothing_either(self) -> None:
        llm = _Recording()
        decision = await SufficiencyGate(llm).assess(  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            investigation_id=uuid.uuid4(),
            question="Why did signups fall?",
            report=_report(summary=["We cannot establish a cause from this data."]),
        )
        assert llm.calls == []
        assert not decision.needed

    async def test_an_unneeded_gate_changes_nothing_about_the_report(self) -> None:
        report = _report(summary=["Signups fell 53%."])
        applied = SufficiencyDecision(needed=False).apply(report)
        assert applied.report is report
        assert applied.rejections == []
        assert applied.withheld == 0


class TestTheVeto:
    def test_an_insufficient_verdict_withholds_the_causal_claim(self) -> None:
        report = _report(
            summary=[
                "Signups fell 53% starting 2026-06-17.",
                "The fall was caused by the 16 June marketing site refresh.",
            ]
        )
        decision = SufficiencyDecision(
            needed=True, ran=True, sufficient=False, missing=("a comparable control series",)
        )
        applied = decision.apply(report)
        remaining = [claim.text for claim in applied.report.executive_summary]
        assert remaining == ["Signups fell 53% starting 2026-06-17."]
        assert applied.withheld == 1
        assert applied.rejections

    def test_the_withheld_sentence_survives_in_the_rejection(self) -> None:
        """Recorded rather than paraphrased. The exact text that was drafted is what a reader
        needs to judge whether the right claim went, and a paraphrase is a new claim nothing
        checked."""
        report = _report(
            summary=[
                "Signups fell 53%.",
                "The fall was caused by the 16 June refresh.",
            ]
        )
        applied = SufficiencyDecision(
            needed=True, ran=True, sufficient=False, missing=("a control series",)
        ).apply(report)
        assert any(
            "caused by the 16 June refresh" in rejection.text for rejection in applied.rejections
        )

    def test_a_supported_hypothesis_is_downgraded_not_deleted(self) -> None:
        """A hypothesis is the record of what was considered. Deleting it would hide the
        consideration; downgrading it says the evidence does not carry it.

        `cause_at` is the path exercised here because it is the stronger signal: a hypothesis
        carrying a machine-readable cause date names an intervention by construction, whatever
        its prose does.
        """
        report = _report(
            summary=["Signups fell 53%.", "The fall was caused by the refresh."],
            hypotheses=[
                Hypothesis(
                    statement="The 16 June refresh is responsible for the fall.",
                    verdict=Verdict.SUPPORTED,
                    supporting_evidence_ids=_evidence(),
                    cause_at=date(2026, 6, 16),
                    effect_onset=date(2026, 6, 17),
                )
            ],
        )
        applied = SufficiencyDecision(
            needed=True, ran=True, sufficient=False, missing=("a control series",)
        ).apply(report)
        assert len(applied.report.hypotheses) == 1
        assert applied.report.hypotheses[0].verdict is not Verdict.SUPPORTED

    def test_a_hypothesis_with_no_dates_is_caught_by_its_prose(self) -> None:
        """The weaker path, which is what most drafts will use."""
        report = _report(
            summary=["Signups fell 53%.", "The fall was caused by the refresh."],
            hypotheses=[
                Hypothesis(
                    statement="The refresh broke the signup path.",
                    verdict=Verdict.SUPPORTED,
                    supporting_evidence_ids=_evidence(),
                )
            ],
        )
        applied = SufficiencyDecision(
            needed=True, ran=True, sufficient=False, missing=("a control series",)
        ).apply(report)
        assert applied.report.hypotheses[0].verdict is not Verdict.SUPPORTED

    def test_a_sufficient_verdict_withholds_nothing(self) -> None:
        report = _report(summary=["The fall was caused by the 16 June refresh."])
        applied = SufficiencyDecision(needed=True, ran=True, sufficient=True).apply(report)
        assert [c.text for c in applied.report.executive_summary] == [
            "The fall was caused by the 16 June refresh."
        ]
        assert applied.withheld == 0

    def test_emptying_the_summary_rejects_the_report(self) -> None:
        """The same rule the gate and the verifier apply: a report with no answer left is not a
        shorter report, it is not a report."""
        report = _report(summary=["The fall was caused by the 16 June refresh."])
        with pytest.raises(ReportRejected):
            SufficiencyDecision(
                needed=True, ran=True, sufficient=False, missing=("a control series",)
            ).apply(report)


class TestAGateThatDidNotRunWithholdsNothing:
    """A gate that could break an investigation by being unavailable would be a worse trade than
    one that occasionally declines to judge.

    The provider-failure path itself is exercised in `tests/db/test_sufficiency_live.py`, because
    the gate loads evidence before it calls anything and so needs a real session.
    """

    def test_a_gate_that_did_not_run_discloses_itself(self) -> None:
        """`sufficient` defaults true, so a caller reading it without checking `ran` would treat
        an outage as a pass. The risk note is what stops that being invisible to a reader."""
        report = _report(summary=["The fall was caused by the refresh."])
        applied = SufficiencyDecision(needed=True, ran=False).apply(report)
        assert applied.withheld == 0
        assert any("was not checked" in risk.description for risk in applied.report.risks)


class TestARefusalCarriesWhatWouldLiftIt:
    def test_the_missing_items_are_named(self) -> None:
        decision = SufficiencyDecision(
            needed=True,
            ran=True,
            sufficient=False,
            missing=("a comparable series the change did not touch", "a dated deploy"),
        )
        sentence = decision.missing_sentence
        assert "comparable series" in sentence and "dated deploy" in sentence

    def test_it_falls_back_to_the_model_reason(self) -> None:
        """A refusal that names nothing is the failure mode the ADR spends a paragraph on: the
        alternative to an informative refusal is not silence, it is a human inventing a cause."""
        decision = SufficiencyDecision(
            needed=True, ran=True, sufficient=False, reason="nothing dated inside the window"
        )
        assert decision.missing_sentence == "nothing dated inside the window"


class TestTheBarIsDerivableNotProven:
    """The bar was set too high once, and this pins where it landed.

    The first prompt asked whether the observations *establish* a cause, required them to "rule
    out the other changes in the same window", and told the model that "not sufficient is the
    common answer". Measured on run 15: eleven vetoes across eight attempts, including the
    correct planted cause on both scenarios that have one -- "the mobile onboarding modal rework
    (PR #913, merged 2026-07-14) broke mobile signup completion", withheld. It also starved the
    two phases downstream, which act only on a causal assertion and so never ran.

    That is decision 5's question, and decision 5 is a disclosure precisely because the honest
    answer to it is almost always no. Asking it here gave decision 5's question decision 6's
    veto.
    """

    def test_it_says_outright_that_proof_is_not_the_bar(self) -> None:
        assert "not** deciding whether a cause is proven" in SUFFICIENCY_PROMPT
        assert "would refuse every real answer" in SUFFICIENCY_PROMPT

    def test_it_no_longer_demands_ruling_out_every_other_change(self) -> None:
        """That is identifiability, and it belongs to the disclosure rather than the veto."""
        assert "rule out the other changes" not in SUFFICIENCY_PROMPT

    def test_it_does_not_nudge_towards_refusing(self) -> None:
        """ "Not sufficient is the common answer" is a thumb on the scale, not a rule."""
        assert "is the common answer" not in SUFFICIENCY_PROMPT

    def test_it_states_what_sufficient_looks_like(self) -> None:
        """A prompt listing only failure modes biases towards finding one."""
        assert "Sufficient looks like" in SUFFICIENCY_PROMPT
        assert "plausibly explains it" in SUFFICIENCY_PROMPT

    def test_the_genuine_vetoes_survive(self) -> None:
        """Narrowing the bar must not lose the four things that really are insufficient."""
        for rule in (
            "Only the movement",
            "returned no rows",
            "stopped recording it",
            "dated after the movement began",
        ):
            assert rule in SUFFICIENCY_PROMPT, rule

    def test_it_still_forbids_reconstructing_the_conclusion(self) -> None:
        """The property that makes this a second reading rather than a review of the answer."""
        assert "must not try to reconstruct one" in SUFFICIENCY_PROMPT


class TestAnEliminationIsNotAnAssertion:
    """The gate was withholding the analyst's ruling-out work.

    Found by reading five sufficiency rejections together, across two providers. Every withheld
    claim carried the same verdict string, and several of them asserted no cause at all:

      - "no code change on the web front-end explains a session change"
      - "limited to a CI runner pin and a dependency bump, ruling out a deploy-caused regression"

    Both are eliminations backed by the diff they name. `_DECLINES_A_CAUSE` already held "rules
    out" and "ruled out" and simply lacked the gerund, and nothing at all matched a negated
    subject before the verb.

    Why it matters more than a scoring detail: ruling a candidate out *is* the analysis. It is
    what Kepner-Tregoe's IS/IS-NOT step produces and what makes a surviving cause worth
    believing. A gate that removes eliminations deletes the reasoning and keeps the conclusion,
    which is the opposite of its purpose — and on `measurement_stopped` it removed every summary
    claim and refused the whole report, which was correct and corroborated.
    """

    def test_ruling_out_is_not_asserting(self) -> None:
        assert not is_causal_claim(
            "Code changes were limited to a CI runner pin and a dependency bump, ruling out a "
            "deploy regression."
        )

    def test_an_elimination_naming_a_causal_word_is_over_caught_on_purpose(self) -> None:
        """The known cost of the direction this predicate errs in, recorded rather than hidden.

        "ruling out a deploy-**caused** funnel regression" still reads as causal: the marker is
        inside a hyphenated adjective and no substring test can tell that from an assertion. So
        the sufficiency gate may withhold this sentence.

        That is the acceptable failure. A false negative here disables four defences at once —
        the gate makes no call, the fresh-context re-derivation is skipped, data-trust
        enforcement returns early, and nothing enters the withheld set — so an unsupported
        causal claim reaches a reader unchecked. A false positive costs one elimination in one
        report. The first version of this fix chose the other direction and leaked eight
        assertions.
        """
        assert is_causal_claim(
            "Code changes were limited to a CI runner pin, ruling out a deploy-caused regression."
        )

    def test_a_negated_subject_is_not_asserting(self) -> None:
        """The noun between the negation and the verb is arbitrary, so no phrase list can catch
        this. It is the one pattern here expressed as a regex."""
        assert not is_causal_claim(
            "No commits were found since 2026-07-15, so no code change on the web front-end "
            "explains a session change."
        )
        assert not is_causal_claim("Nothing in the diff caused the drop.")

    def test_asserting_a_cause_still_counts(self) -> None:
        """The fix must not blunt the gate on what it exists for."""
        assert is_causal_claim(
            "The mobile signup drop was caused by PR #913, which reworked the onboarding modal."
        )
        assert is_causal_claim("The drop was driven by the paid campaign ending on 14 June.")

    def test_an_elimination_does_not_excuse_the_rest_of_the_sentence(self) -> None:
        """The hole the first version of this fix shipped with, caught before it landed.

        Returning False on any sentence *containing* an elimination blinds the gate to a
        compound one. So the elimination is cut out and the question asked of what remains,
        rather than short-circuited on — a gate that stops reading at the first "no" is worse
        than one that is slightly over-strict.
        """
        assert is_causal_claim(
            "No single deploy explains it, but the campaign ending caused the majority of the fall."
        )

    def test_a_negation_far_from_the_verb_does_not_excuse_it(self) -> None:
        """The window is narrow on purpose: a sentence that says "no" early and asserts a cause
        much later is an assertion, and widening it would start excusing those."""
        assert is_causal_claim(
            "No dashboards were available to the team at the time, and after a week of manual "
            "checks the eventual finding was that the onboarding modal caused the drop."
        )


class TestTheTwoJudgesAreOrderedOppositely:
    """The verifier reasons before deciding; this gate decides before listing. Both are right.

    Structured output generates fields in order, so whichever comes first anchors the rest. The
    two judges can fail in opposite directions, which is why they are ordered opposite ways:

    - The verifier judges *quality*, where deciding first means writing a justification for what
      you already said. So `reason` precedes `verdict`.
    - This gate judges *sufficiency*, where naming what is missing is easy and always possible —
      more evidence can always be wished for. A `missing`-first order leaves a list the model
      must then agree with, and the agreeing answer is "insufficient". That is over-abstention,
      and a suite whose hard bar is a hallucination count of zero cannot see it.

    Asserted together so that a future reader flipping one for consistency has to read why they
    differ, rather than discovering it from an eval run.
    """

    def test_the_sufficiency_gate_decides_before_it_lists(self) -> None:
        from cortex.reports.sufficiency import SUFFICIENCY_SCHEMA

        order = list(SUFFICIENCY_SCHEMA["properties"])
        assert order.index("sufficient") < order.index("missing")

    def test_the_verifier_reasons_before_it_decides(self) -> None:
        from cortex.reports.verifier import VERDICT_SCHEMA

        order = list(VERDICT_SCHEMA["properties"])
        assert order.index("reason") < order.index("verdict")

    def test_the_gate_still_has_somewhere_to_reason(self) -> None:
        """arXiv 2605.23970 §8: a rigid format with no room to reason destroys the gate — one
        ablation committed on 48 of 48 unknowable items. `reason` is required here, so that
        failure mode does not apply and is not the argument for changing the order."""
        from cortex.reports.sufficiency import SUFFICIENCY_SCHEMA

        assert "reason" in SUFFICIENCY_SCHEMA["required"]


class TestANegationElsewhereDoesNotExcuseAnAssertion:
    """Eight real causal assertions scanned as non-causal after the first version of this fix.

    The pattern spanned any negation within forty characters of a causal verb and excised the
    whole match, verb included — so it deleted the *assertion* rather than the elimination.
    "Neither team noticed, but PR 913 caused the drop" is structurally the sentence the fix's own
    test protects; it just leads with a negation belonging to another clause.

    The consequence was the reason to treat it as urgent: `is_causal_claim` gates the sufficiency
    gate, the fresh-context re-derivation, data-trust enforcement and the withheld set, so a
    false negative turns off all four for that sentence.
    """

    ASSERTIONS = (
        "Neither team noticed, but PR 913 caused the drop.",
        "There is no doubt the campaign ending caused the fall.",
        "No one disputes that the pricing change caused the churn.",
        "There was no warning before the deploy caused the outage.",
        "None other than the June deploy caused the mobile collapse.",
        "With no prior notice, the vendor outage caused the signup drop.",
        "No fewer than three deploys caused the regression.",
        "No dashboards existed, so the modal caused the drop.",
        "Deploys were ruled out, but the campaign ending caused the majority of the fall.",
        "There is no evidence of a deploy; the campaign ending caused the drop.",
        "The drop is unrelated to mobile, and was caused by the campaign budget running out.",
    )

    @pytest.mark.parametrize("text", ASSERTIONS)
    def test_it_is_still_a_causal_claim(self, text: str) -> None:
        assert is_causal_claim(text), text

    ELIMINATIONS = (
        "No commits were found, so no code change on the web front-end explains a session change.",
        "Nothing in the diff caused the drop.",
        "No deploy is responsible for the drop.",
        "None of these is the reason for the fall.",
        "There is no evidence that the deploy caused the fall.",
    )

    @pytest.mark.parametrize("text", ELIMINATIONS)
    def test_a_genuine_elimination_is_still_excused(self, text: str) -> None:
        """Both directions, in one class. Fixing the leak must not simply restore the behaviour
        that withheld the analyst's ruling-out work."""
        assert not is_causal_claim(text), text

    def test_the_phrase_list_no_longer_short_circuits(self) -> None:
        """The excise-don't-short-circuit fix was applied only to the pattern; the thirty-entry
        phrase list still returned False on the whole sentence — and the same commit added two
        entries to it, widening the hole it claimed to close."""
        assert is_causal_claim(
            "Code changes were limited to a dependency bump, ruling out a deploy regression, "
            "and the campaign ending caused the fall."
        )


class TestAPlainCausalAssertionIsCaught:
    """ "The most likely proximate cause is a signup-blocking UI bug" scanned as *not* causal.

    `_CAUSAL_MARKERS` carried "cause of" and "root cause" and not the copula, so a report naming
    a cause in the most ordinary English form available scored as having declined — and the
    sufficiency gate would not have fired on it either, since `is_causal_claim` is what decides
    whether the gate is called at all.

    Found in a decline-twin report, which is the point of having twins: the parent scenarios
    never phrased it this way.
    """

    ASSERTIONS = (
        "The most likely proximate cause is a signup-blocking UI bug on iOS 18 Safari.",
        "The primary cause is the campaign ending on 14 June.",
        "The cause was the onboarding modal change shipped in PR 913.",
        "The culprit is the paid-search budget running out.",
        "The main cause is a measurement break rather than a demand drop.",
    )

    @pytest.mark.parametrize("text", ASSERTIONS)
    def test_the_copula_forms_are_causal(self, text: str) -> None:
        assert is_causal_claim(text), text

    DECLINES = (
        "No cause could be established from the evidence gathered.",
        "The data cannot establish a cause.",
        "Signups fell 18% last week.",
    )

    @pytest.mark.parametrize("text", DECLINES)
    def test_declining_still_reads_as_declining(self, text: str) -> None:
        """Widening the marker list must not start catching the refusals, which is the failure
        that made the gate withhold the analyst's own eliminations."""
        assert not is_causal_claim(text), text


class TestAWithheldCauseCannotSurviveInAHeading:
    """The third place a withheld causal story can hide, found by the first real-data run.

    `causal_hypotheses` exists because a cause surviving only as a supported hypothesis has not
    been withheld, it has been moved. A finding's *title* is the same argument one field over,
    and worse on delivery: it is the line a reader reads, it carries no `evidence_ids`, so
    grounding cannot see it and the verifier has nothing to judge it against.

    Live, against real PostHog and GitHub data, the gate withheld the cause — *"nothing here
    dates or documents that rename/migration"* — and the delivered report still led its findings
    with *"the Canvas frontend 'conversation_created' event undercounts due to a consent-banner
    and telemetry-client regression"*. Neither claim beneath that title mentioned a consent
    banner. `delivered_hallucinations` counted zero, because titles are not claims.
    """

    LIVE_TITLE = (
        "The Canvas frontend 'conversation_created' event undercounts due to a "
        "consent-banner and telemetry-client regression, not falling usage"
    )

    def _insufficient(self) -> SufficiencyDecision:
        return SufficiencyDecision(
            needed=True,
            ran=True,
            sufficient=False,
            missing=("a dated migration record for the event rename",),
        )

    def _with_finding(self, title: str, claim: str) -> InvestigationReport:
        report = _report(summary=["Conversation volume did not fall in August 2026."])
        return report.model_copy(
            update={
                "findings": [
                    Finding(title=title, claims=[Claim(text=claim, evidence_ids=_evidence())])
                ]
            }
        )

    def test_the_heading_is_cut_back_to_what_it_observed(self) -> None:
        applied = self._insufficient().apply(
            self._with_finding(
                self.LIVE_TITLE,
                "The weekly count fell from 933 in the week of Aug 2 to 121 by Aug 30.",
            )
        )
        title = applied.report.findings[0].title
        assert title == "The Canvas frontend 'conversation_created' event undercounts"
        assert not is_causal_claim(title), title

    def test_the_drafted_heading_survives_in_the_rejection(self) -> None:
        """Recorded rather than paraphrased, like every other withholding here: what the
        analyst wrote is what a reader of the audit row needs to see."""
        applied = self._insufficient().apply(
            self._with_finding(self.LIVE_TITLE, "The weekly count fell from 933 to 121.")
        )
        titles = [r for r in applied.rejections if r.location.endswith(".title")]
        assert len(titles) == 1
        assert titles[0].text == self.LIVE_TITLE
        assert "the title named a cause" in titles[0].detail

    def test_a_heading_that_is_only_a_cause_falls_back_to_its_own_claim(self) -> None:
        """Truncation can leave nothing usable. The claim beneath is grounded and, by this
        point, non-causal — so it is the honest replacement, and it is not a sentence anybody
        invented on the analyst's behalf."""
        applied = self._insufficient().apply(
            self._with_finding(
                "Because the budget ran out", "Paid search sessions fell 74% after 15 June."
            )
        )
        assert applied.report.findings[0].title == "Paid search sessions fell 74% after 15 June."

    def test_a_descriptive_heading_is_left_alone(self) -> None:
        """The gate withholds causes, not findings. A heading that reports an observation is
        the analyst's work and must survive intact."""
        descriptive = "Signups fell sharply and abruptly starting 15 June"
        applied = self._insufficient().apply(
            self._with_finding(descriptive, "Daily signups ran at 57/day, then 33/day.")
        )
        assert applied.report.findings[0].title == descriptive
        assert not [r for r in applied.rejections if r.location.endswith(".title")]

    def test_a_sufficient_verdict_touches_no_heading(self) -> None:
        report = self._with_finding(self.LIVE_TITLE, "The weekly count fell from 933 to 121.")
        applied = SufficiencyDecision(needed=True, ran=True, sufficient=True).apply(report)
        assert applied.report is report
