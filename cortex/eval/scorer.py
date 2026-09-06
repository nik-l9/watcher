"""Scoring one investigation against a labeled scenario.

The dimensions come from the doc: accuracy, grounding, completeness, actionability,
tool selection, latency — plus a hallucination count that must be zero.

**What "hallucination count must be zero" counts.** Two numbers, because one was
measuring the wrong thing. The gate filters unresolvable citations out of the report and
the verifier drops the claims it judges unsupported, so neither ever reaches a reader.
Counting those failed a report that contained nothing unsupported — the suite failing
the product for defending itself — and it rewarded a thin draft over a rich one, since
fewer claims means fewer chances one gets trimmed. So:

  - `hallucinations` counts what shipped **unchecked**: claims the verifier could not
    judge, which are kept on purpose and disclosed, but reach the reader unverified.
    This gates.
  - `caught_and_removed` counts what a mechanism stopped. This does not gate, but it is
    always printed, and `draft_reliability` scores it — a drafter that needs more
    trimming is degrading even while the defenses hold.

`draft_reliability` gates against a floor rather than a standard, because gating on
delivery alone would pass an analyst whose every claim was fabricated and caught.

The split that matters: **grounding and hallucination are computed mechanically
against the evidence store; nothing about them asks a model.** They are the
product's central claim, so scoring them with an LLM judge would mean the claim
rests on the same kind of component it is supposed to constrain. Accuracy is scored
by string-matching required signals against the report for the same reason.

Two dimensions gate the run: **grounding** and **accuracy**. `summary_placement` is scored on
false-premise scenarios only and deliberately does *not* gate — see `_summary_placement`, and the
run it failed for a formatting reason before the split.

Original text follows.

Two dimensions gate the run: **grounding** and **accuracy**. Grounding catches a
fabricated citation; accuracy catches a fabricated *conclusion* — a report whose every
citation resolves but whose causal story was invented. The second is the more dangerous
failure, because it looks correct, and a grounding-only gate misses it entirely. Both
are safe to gate precisely because neither consults a model.

Actionability, completeness and decoy rejection are softer. They are weighted but do
not gate, because a soft score that can fail a build teaches people to distrust the
build.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.agents.investigator import Investigation
from cortex.agents.timing import DRAFT, GATE, SUFFICIENCY, VERIFY
from cortex.db.models import Evidence, ToolCall
from cortex.db.threads import citable_investigation_ids
from cortex.eval.fixtures import (
    GroundTruth,
    Requirement,
    Scenario,
    alternatives_of,
    describe_requirement,
)
from cortex.reports.gate import GateResult
from cortex.reports.schema import (
    Confidence,
    InvestigationReport,
    PremiseVerdict,
    Verdict,
)
from cortex.reports.shape import Shape, shape_for
from cortex.reports.sufficiency import AppliedSufficiency
from cortex.reports.verifier import (
    _DECLINES_A_CAUSE,
    VerificationResult,
    causal_claims,
    causal_hypotheses,
)

# Aliased because `Verdict` above is the *hypothesis* verdict from the report schema, and this
# is the per-claim one. Same spelling, different enum: comparing a claim against the wrong one
# never matches, which would leave the dimension below reporting a clean 1.00 forever.
from cortex.reports.verifier import Verdict as ClaimVerdictKind
from cortex.tenancy.context import TenantContext
from cortex.tools.executor import canonical_hash

#: Hypothesis verdicts that mean "this explanation did not survive the evidence".
#:
#: `INCONCLUSIVE` counts alongside `CONTRADICTED` on purpose. An analyst that tested a
#: decoy and could not settle it has still refused to assert it, which is the behaviour
#: being measured — the failure mode is naming a decoy as *the cause*, not failing to
#: disprove it.
_DISMISSED_VERDICTS = (Verdict.CONTRADICTED, Verdict.INCONCLUSIVE)

#: The share of drafted claims that must survive review.
#:
#: A floor, not a standard. Gating on delivered hallucinations alone leaves a hole: an
#: analyst that fabricates every claim would pass as long as both mechanisms caught
#: every one, and "the defenses held this time" is not the same as "the analyst is
#: sound". A single trimmed claim out of five is ordinary and must not fail a build; a
#: report where most of the draft had to be removed is a broken analyst whose next
#: fabrication ships the first time a verdict is wrong.
#:
#: Set at half deliberately, far from the observed one-in-five, so it catches collapse
#: without penalising the richer drafts that scored better on every other dimension.
_DRAFT_RELIABILITY_FLOOR = 0.5

#: Verbs that mean "change the system". Acting on these when no cause is established is
#: the failure the unanswerable scenario exists to catch.
#:
#: Matched only as the **leading** verb of a recommendation — see `_proposes_action`.
_ACTION_VERBS = (
    "roll back",
    "rollback",
    "roll out",
    "revert",
    "disable",
    "enable",
    "remove",
    "deploy",
    "restore",
    "ship",
    "merge",
    "launch",
)

#: How far into a recommendation an action verb is still read as the thing being proposed.
_ACTION_VERB_WINDOW_WORDS = 4

#: Verbs that mean "find out more". A recommendation led by one of these is a diagnostic
#: step, which is the *correct* output on unanswerable data — never a proposal to act.
#:
#: Listed explicitly rather than inferred from the absence of an action verb, because the
#: point is to decide from the sentence's own verb rather than from whichever nouns happen
#: to appear later in it.
_INVESTIGATIVE_VERBS = (
    "confirm",
    "check",
    "verify",
    "validate",
    "monitor",
    "watch",
    "measure",
    "re-measure",
    "instrument",
    "review",
    "investigate",
    "examine",
    "audit",
    "re-run",
    "rerun",
    "compare",
    "quantify",
    "document",
    "collect",
    "gather",
    "ask",
    "escalate",
)

#: Prefixes that invert the recommendation. Kept narrow and anchored to the start of the
#: sentence: a general negation search would misread "remove the flag, do not wait", which
#: *is* advice to act.
_NEGATIONS = (
    "do not ",
    "don't ",
    "avoid ",
    "hold off",
    "refrain from ",
    "no need to ",
    "should not ",
    "must not ",
    "resist ",
)


def _proposes_action(action: str) -> bool:
    """Whether a recommendation proposes changing the system.

    Decided from the sentence's **leading verb**, not from whether an action word appears
    anywhere in it. Two earlier attempts failed on the same class of collision:

      - substring matching scored "Confirm HubSpot, GitHub *deployment* and BigQuery
        connector health" as a proposal to deploy, because "deployment" contains "deploy";
      - word-boundary matching then scored "Confirm the correct repository and date range
        are being queried in GitHub before ruling out a *deploy-related* cause" the same
        way, because a hyphen is a word boundary.

    Both are recommendations to *look*, and both were penalised as recommendations to
    *act*. The pattern is that the nouns in a sentence say nothing about what it proposes,
    while the verb it opens with says almost everything — so an investigative opener is
    never an action, whatever it goes on to mention.
    """
    text = action.strip().lower()
    if any(re.match(rf"{re.escape(verb)}\b", text) for verb in _INVESTIGATIVE_VERBS):
        return False
    # A short window rather than the first word exactly, so an adverbial opener
    # ("Immediately roll back the modal") still reads as a proposal to act. Four words is
    # wide enough for that and far too narrow to reach the nouns that caused the
    # collisions above.
    opening = " ".join(text.split()[:_ACTION_VERB_WINDOW_WORDS])
    return any(re.search(rf"\b{re.escape(verb)}\b", opening) for verb in _ACTION_VERBS)


def _is_advice_against(action: str) -> bool:
    """Whether a recommendation counsels *inaction*.

    On an unanswerable scenario, "do not roll back on the strength of this report" is the
    best possible recommendation: it names the tempting wrong move and warns against it,
    which is more useful to a reader than silence.
    """
    text = action.strip().lower()
    return any(text.startswith(prefix) for prefix in _NEGATIONS)


@dataclass(frozen=True, slots=True)
class Dimension:
    name: str
    score: float
    detail: str
    #: A failing gate fails the run. Only the mechanical dimensions gate.
    gates: bool = False
    #: The score a gating dimension must reach. Perfection for the two that catch a
    #: fabricated citation or conclusion; lower where the dimension is a floor against
    #: gross failure rather than a standard to meet.
    threshold: float = 1.0

    @property
    def passed(self) -> bool:
        return not self.gates or self.score >= self.threshold


@dataclass(slots=True)
class Scorecard:
    scenario: str
    dimensions: list[Dimension] = field(default_factory=list)

    #: Unsupported claims that reached the reader. Gates the run, and is the metric the
    #: doc's "hallucination count must be zero" is about.
    #:
    #: This used to count every claim either mechanism removed, which failed a report
    #: that contained nothing unsupported: the gate filters unresolvable citations out
    #: entirely and the verifier drops the claims it judges unsupported, so neither ever
    #: ships. Counting them meant the suite failed the product for successfully
    #: defending itself — and, worse, rewarded a thin draft over a rich one, because
    #: fewer claims means fewer chances one gets trimmed.
    #:
    #: What can actually reach a reader is a claim the verifier could not judge. Those
    #: are kept on purpose (a verifier outage must not silently delete grounded work)
    #: and disclosed as a risk, but they ship unchecked. That is the real exposure.
    hallucinations: int = 0

    #: Claims a mechanism caught and removed before delivery. Does not gate.
    #:
    #: Non-gating but never hidden: a rise here means the drafter is producing more
    #: claims its own evidence does not support, which is a genuine regression even
    #: while the defenses hold. It is the leading indicator for the number above.
    caught_and_removed: int = 0

    duration_ms: int = 0
    tokens: int = 0

    def dimension(self, name: str) -> Dimension:
        for d in self.dimensions:
            if d.name == name:
                return d
        raise KeyError(name)

    @property
    def passed(self) -> bool:
        return self.hallucinations == 0 and all(d.passed for d in self.dimensions)

    @property
    def overall(self) -> float:
        """Weighted mean. Grounding and accuracy dominate deliberately."""
        weights = {
            "grounding": 3.0,
            "accuracy": 3.0,
            "tool_selection": 1.5,
            "decoy_rejection": 2.0,
            "completeness": 1.0,
            "actionability": 1.0,
            # Weighted, not gating: a trimmed claim is the defenses working, but a
            # drafter that needs trimming more often is a real regression.
            "draft_reliability": 1.5,
            # Delivery rather than correctness, and weighted below the gating pair on purpose:
            # burying a right answer is a real defect and a smaller one than getting it wrong.
            "summary_placement": 1.0,
            "latency": 0.5,
        }
        total = sum(weights.get(d.name, 1.0) for d in self.dimensions)
        if not total:
            return 0.0
        earned = sum(d.score * weights.get(d.name, 1.0) for d in self.dimensions)
        return round(earned / total, 4)

    @property
    def failures(self) -> list[str]:
        reasons = [f"{d.name}: {d.detail}" for d in self.dimensions if not d.passed]
        if self.hallucinations:
            reasons.insert(
                0,
                f"{self.hallucinations} unverified claim(s) reached the report "
                "(delivered hallucinations must be 0)",
            )
        return reasons


class Scorer:
    def __init__(self, *, latency_budget_ms: int = 90_000) -> None:
        # The doc's 90-second north star. Scored, not gated: a slow correct answer is
        # worth more than a fast wrong one, and gating on wall clock would make the
        # suite fail on a loaded CI runner.
        self._latency_budget_ms = latency_budget_ms

    async def score(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        *,
        investigation_id: uuid.UUID,
        scenario: Scenario,
        investigation: Investigation,
        gate_result: GateResult,
        verification: VerificationResult | None = None,
        sufficiency: AppliedSufficiency | None = None,
    ) -> Scorecard:
        report = verification.report if verification else gate_result.report
        card = Scorecard(
            scenario=scenario.name,
            duration_ms=investigation.duration_ms,
            tokens=investigation.usage.total,
        )

        # Caught: a fabricated citation the gate filtered out, or a claim the verifier
        # judged unsupported and dropped. Neither reaches a reader.
        card.caught_and_removed = gate_result.hallucination_count + (
            verification.unsupported_count if verification else 0
        )

        # Delivered: what shipped without being checked. With verification on, that is
        # the claims the verifier could not judge. With verification off there is no
        # second mechanism at all, so every claim in the report shipped unchecked —
        # reported as such rather than as a clean zero, since a suite that scores best
        # when a grounding mechanism is disabled is worse than no suite.
        card.hallucinations = len(verification.unverified) if verification else _claim_count(report)

        card.dimensions = [
            await self._grounding(session, tenant, investigation_id, report),
            self._accuracy(scenario, report),
            self._decoy_rejection(scenario, report),
            await self._tool_selection(session, tenant, investigation_id, scenario),
            self._completeness(scenario, report),
            self._actionability(scenario, report),
            self._draft_reliability(report, gate_result, verification, investigation),
            self._latency(investigation),
        ]
        if sufficiency is not None:
            card.dimensions.append(self._veto_precision(scenario, sufficiency))
        if verification is not None:
            card.dimensions.append(self._verifier_precision(scenario, verification))
        if scenario.ground_truth.is_false_premise:
            # Only meaningful where there is a premise to refute, so it is appended rather than
            # scored as 1.0 everywhere else -- a dimension that is perfect by default on three
            # scenarios out of four would drag every mean towards it and mean nothing.
            card.dimensions.append(
                self._summary_placement(scenario.ground_truth, report, _report_text(report).lower())
            )
        return card

    # ------------------------------------------------------------------ dimensions

    def _draft_reliability(
        self,
        report: InvestigationReport,
        gate_result: GateResult,
        verification: VerificationResult | None,
        investigation: Investigation | None = None,
    ) -> Dimension:
        """How much of the first draft survived review.

        Non-gating, because a removed claim is the system working. Scored anyway,
        because the alternative is that the drafter degrades invisibly for as long as
        the defenses keep holding — and the day one of them misjudges, the claim ships.

        **The detail names which mechanism removed what, and how many drafts it took.** The
        score cannot: two mechanisms remove claims for unrelated reasons, and one number over
        both says a draft got worse without saying how. A citation that resolves to nothing is
        the drafter inventing an id; a claim its own evidence does not support is the drafter
        overreaching from real data. Those have different fixes, and 0.82 looks identical either
        way. Same argument the sufficiency gate's rejections are already kept separate under.

        The draft attempt count is here for the reason a large deployment of this pattern gives
        for watching its pre-revision pass rate separately (arXiv 2608.18300 §6.1): a drafter
        needing three tries to emit a parseable report and a reviewer becoming stricter both land
        on this dimension, and only the attempt count separates them.
        """
        surviving = _claim_count(report)
        caught = gate_result.hallucination_count + (
            verification.unsupported_count if verification else 0
        )
        if verification is None:
            return Dimension(
                "draft_reliability",
                1.0,
                "not measured: verification was disabled",
            )
        drafted = surviving + caught
        if not drafted:
            return Dimension("draft_reliability", 0.0, "the draft contained no claims")
        score = round(surviving / drafted, 4)
        return Dimension(
            "draft_reliability",
            score,
            f"{surviving}/{drafted} drafted claims survived review"
            + _removal_breakdown(gate_result, verification)
            + _draft_attempts(investigation),
            gates=True,
            threshold=_DRAFT_RELIABILITY_FLOOR,
        )

    async def _grounding(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        investigation_id: uuid.UUID,
        report: InvestigationReport,
    ) -> Dimension:
        """Fraction of citations that resolve to real, untampered evidence.

        Recomputed from the database rather than trusted from the gate's own result:
        the gate is the thing being evaluated, so scoring it with its own output would
        be circular. Gates the run — anything below 1.0 means a rendered report cited
        something that does not exist.
        """
        cited = report.cited_evidence_ids()
        if not cited:
            return Dimension(
                name="grounding",
                score=0.0,
                detail="the report cites no evidence at all",
                gates=True,
            )

        # Thread-scoped, matching the gate and the verifier. Every scenario in the suite is a
        # standalone investigation with no parent, so this changes nothing today -- it is here
        # because three places now check the same property, and a scorer narrower than the gate
        # would score a follow-up's grounding at zero while the report was perfectly grounded.
        citable = await citable_investigation_ids(session, tenant, investigation_id)
        rows = (
            (
                await session.execute(
                    select(Evidence).where(
                        Evidence.tenant_id == tenant.tenant_id,
                        Evidence.investigation_id.in_(citable),
                        Evidence.id.in_(cited),
                    )
                )
            )
            .scalars()
            .all()
        )
        resolvable = {row.id for row in rows if canonical_hash(row.payload) == row.payload_hash}
        score = len(resolvable & cited) / len(cited)
        missing = sorted(str(c) for c in cited - resolvable)
        return Dimension(
            name="grounding",
            score=round(score, 4),
            detail=(
                "every citation resolves"
                if score == 1.0
                else f"{len(missing)} of {len(cited)} citations unresolvable: "
                + ", ".join(missing[:3])
            ),
            gates=True,
        )

    def _accuracy(self, scenario: Scenario, report: InvestigationReport) -> Dimension:
        """Did it find the planted cause?

        String-matched against required signals rather than judged. A deploy sha or a
        segment name either appears or it does not, and the alternative — an LLM
        deciding whether two prose explanations agree — would make the headline
        accuracy number depend on a model's opinion.
        """
        # The asserted surface, not every string in the document. Naming a cause in order to
        # reject it is not finding it. See `_asserted_text`.
        text = _asserted_text(report).lower()
        truth = scenario.ground_truth

        if truth.is_false_premise:
            return self._premise_accuracy(truth, report, text)

        if truth.is_unanswerable:
            # Correct here means declining to name a cause. Scored on the confidence
            # the report claims, because a confident causal story with real citations
            # is exactly the failure this scenario exists to catch.
            honest = report.confidence in (
                Confidence.LOW,
                Confidence.INSUFFICIENT_EVIDENCE,
            )
            # Whether a cause was asserted, decided by the same predicate the sufficiency gate
            # and the verifier use.
            #
            # **This was a keyword search for humility and it scored a correct report down.**
            # The old test looked for one of seven phrases -- "cannot", "insufficient", "within
            # normal", and so on. A report on `tempting_coincidence` declared confidence
            # `insufficient_evidence`, asserted no cause anywhere, marked the tempting deploy
            # `inconclusive`, and said "no commits were found ... coincides closely with". It
            # hedged in meaning and used none of the seven words, so it scored 0.5 for
            # "claimed a cause on unanswerable data" while claiming no cause at all.
            #
            # The predicate is strictly better and was already in the codebase: an asserted
            # causal claim is what "named a cause" means, and `causal_hypotheses` counts only
            # *supported* ones, so a candidate recorded as inconclusive is correctly not an
            # assertion. Sharing the definition also means the dimension and the gate can no
            # longer disagree about what a causal claim is.
            declined = not causal_claims(report) and not causal_hypotheses(report)
            score = 1.0 if (honest and declined) else (0.5 if honest or declined else 0.0)
            return Dimension(
                name="accuracy",
                score=score,
                detail=(
                    "correctly declined to name a cause"
                    if score == 1.0
                    else f"claimed a cause on unanswerable data (confidence={report.confidence})"
                ),
                gates=True,
            )

        # Any-of aware: a requirement is met when *any* of its alternatives appears. A
        # report identifying the change as "PR #913" is as correct as one quoting the
        # deploy sha, and scoring it 0.5 measured which name it chose.
        met = [
            requirement
            for requirement in truth.required_signals
            if any(alt.lower() in text for alt in alternatives_of(requirement))
        ]
        score = len(met) / len(truth.required_signals) if truth.required_signals else 1.0
        missing = [
            describe_requirement(requirement)
            for requirement in truth.required_signals
            if requirement not in met
        ]
        return Dimension(
            name="accuracy",
            score=round(score, 4),
            detail=(
                "named the planted cause"
                if score == 1.0
                else f"missing required signal(s): {', '.join(missing)}"
            ),
            # Gates, like grounding. A report whose every citation resolves but whose
            # conclusion is invented is the failure mode a grounding-only gate misses
            # entirely — and it is the more dangerous one, because it looks correct.
            # Safe to gate because it is string-matched against labeled signals, not
            # judged by a model.
            gates=True,
        )

    def _premise_accuracy(
        self, truth: GroundTruth, report: InvestigationReport, text: str
    ) -> Dimension:
        """Did it work out that the question's premise is false?

        **Detection only. Where the refutation appears is scored separately**, by
        `_summary_placement`, and the split matters more than it looks.

        The first version of this dimension folded the two together: full credit for refuting in
        the executive summary, half for refuting in a later finding, and it gated. That made a
        *formatting* property able to fail a run, and it did -- an investigation that correctly
        established the premise was false, made every discriminating call, and recommended no
        action was failed because the twelve-versus-thirty-one-day mechanism sat in a finding
        rather than the summary. Gating on where a correct answer was printed measures conformity
        to a house style, not the capability the scenario exists to test.

        Placement still matters and is still scored, because a correct answer nobody reads is the
        live failure this scenario was built from. It just does not gate.
        """
        # The report's own verdict wins, where it gave one.
        #
        # **The keyword list got this wrong and the failure is instructive.** A report whose
        # first two words were "No -- ... not a real drop" scored 0.50 for *never having refuted
        # the premise*, because the twelve accepted denial phrasings included "did not drop" and
        # "have not dropped" but not "not a real drop". Meanwhile `_summary_placement`, reading
        # the same report, scored 1.00 for refuting it in the executive summary -- two dimensions
        # contradicting each other about the same sentence.
        #
        # `report.premise` is the structural answer, and the same move that made the elimination
        # rule structural rather than a matter of phrasing. The list survives as a fallback so an
        # older report, or one from a model that left the field at its default, scores as it did
        # before -- and so this change cannot silently re-base an existing measurement.
        if report.premise is not PremiseVerdict.NONE_ASSERTED:
            stated = report.premise is PremiseVerdict.FALSE
            return Dimension(
                name="accuracy",
                score=1.0 if stated else 0.0,
                detail=(
                    "refuted the premise"
                    if stated
                    else f"the report says the premise is {report.premise.value}, "
                    "and this scenario's premise is false"
                ),
                gates=True,
            )

        scores: list[float] = []
        missing: list[str] = []
        for requirement in truth.refutation_signals:
            if any(alt in text for alt in alternatives_of(requirement)):
                scores.append(1.0)
            else:
                scores.append(0.0)
                missing.append(describe_requirement(requirement))

        score = sum(scores) / len(scores) if scores else 1.0
        return Dimension(
            name="accuracy",
            score=round(score, 4),
            detail=(
                "refuted the premise"
                if not missing
                else f"never refuted the premise: no {', '.join(missing)}"
            ),
            # Gates, like the causal branch: a report that accepts a false premise is
            # confidently wrong, and no grounding mechanism can see it.
            gates=True,
        )

    def _summary_placement(
        self, truth: GroundTruth, report: InvestigationReport, text: str
    ) -> Dimension:
        """Did the refutation reach the *first* thing a reader sees?

        Separated from accuracy so it can be scored honestly without failing a run. The failure
        it measures is real and was delivered: asked whether signups had fallen, the analyst led
        with "619 events for Aug 1-12 compared to July's total of 4,849", refuted it in the same
        sentence, and reached the actual answer in the fourth bullet. Every claim was accurate. A
        reader who stopped after one line had been misinformed.

        So this is a delivery dimension, not a correctness one, and it is weighted rather than
        gating -- the same treatment `actionability` and `completeness` get, and for the same
        reason: a soft score that can fail a build teaches people to distrust the build.

        **It reads `report.premise` first, and skipping that step put a false statement in a
        scorecard.** On run 36 this dimension reported *"refuted the premise in the executive
        summary"* at 1.00 for a report whose premise verdict was `unverifiable` and whose summary
        said "this is not confirmation of a genuine full-month decline". It scored on the strength
        of one word: `partial` appeared in "the partial data available", one of the twelve
        accepted alternatives, and the other requirement matched nowhere so was skipped as absent.
        A dimension that skips what is missing and scores what is left becomes "did you use one of
        these words early".

        `accuracy` had the mirror image of this bug and fixed it the same way -- see the note
        there about a report reading "No -- ... not a real drop" scoring 0.50 for never having
        refuted the premise while this dimension scored it 1.00 for refuting it in the summary.
        Two dimensions contradicting each other about one sentence was the symptom then too. The
        structural verdict is the authority on *whether* the premise was refuted; this dimension
        only answers *where*.
        """
        if report.premise not in (PremiseVerdict.NONE_ASSERTED, PremiseVerdict.FALSE):
            return Dimension(
                name="summary_placement",
                score=0.0,
                detail=(f"nothing to place: the report says the premise is {report.premise.value}"),
            )
        summary = " ".join(c.text for c in report.executive_summary).lower()
        placed: list[float] = []
        buried: list[str] = []
        for requirement in truth.refutation_signals:
            alternatives = alternatives_of(requirement)
            if any(alt in summary for alt in alternatives):
                placed.append(1.0)
            elif any(alt in text for alt in alternatives):
                placed.append(0.0)
                buried.append(describe_requirement(requirement))
            else:
                # Absent entirely. Accuracy already fails this; scoring it here too would
                # double-count one failure across two dimensions.
                continue

        if not placed:
            return Dimension(
                name="summary_placement",
                score=0.0,
                detail="nothing to place: the premise was never refuted",
            )
        score = sum(placed) / len(placed)
        return Dimension(
            name="summary_placement",
            score=round(score, 4),
            detail=(
                "refuted the premise in the executive summary"
                if not buried
                else f"refuted the premise, but not in the summary: {', '.join(buried)}"
            ),
        )

    def _decoy_rejection(self, scenario: Scenario, report: InvestigationReport) -> Dimension:
        """Did it avoid naming a decoy as the cause?

        The dimension that separates investigation from pattern-matching. Mentioning a
        decoy is fine and often good — ruling one out is exactly what the loop is asked
        to do. Naming it *as the cause* is the failure, so only the executive summary
        and recommendations are examined.
        """
        conclusion = " ".join(
            [c.text for c in report.executive_summary]
            + [r.action for r in report.recommendations]
            + [r.rationale for r in report.recommendations]
        ).lower()

        # A decoy the analyst tested and ruled out. This is the *ideal* behaviour, not a
        # failure: the employee contract asks explicitly for explanations it expects to
        # be wrong, and the report's own hypothesis list is where that work is recorded
        # in structure rather than prose.
        rejected = {
            decoy.lower()
            for decoy in scenario.ground_truth.decoys
            for hypothesis in report.hypotheses
            if hypothesis.verdict in _DISMISSED_VERDICTS
            and decoy.lower() in f"{hypothesis.statement} {hypothesis.reasoning}".lower()
        }

        mentioned = [d for d in scenario.ground_truth.decoys if d.lower() in conclusion]
        asserted = [d for d in mentioned if d.lower() not in rejected]

        if not asserted:
            detail = "no decoy named as the cause"
            if rejected:
                detail += f"; {len(rejected)} tested and ruled out"
            return Dimension(name="decoy_rejection", score=1.0, detail=detail)

        score = max(0.0, 1.0 - len(asserted) / max(len(scenario.ground_truth.decoys), 1))
        return Dimension(
            name="decoy_rejection",
            score=round(score, 4),
            detail=(
                f"decoy(s) asserted in the conclusion without being ruled out: "
                f"{', '.join(asserted)}"
            ),
        )

    async def _tool_selection(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        investigation_id: uuid.UUID,
        scenario: Scenario,
    ) -> Dimension:
        """Did it make the calls that actually separate cause from decoy?

        Read from the audit trail, not from the report. An analyst that names the right
        cause without making the discriminating call guessed, and a guess that happens
        to be right is not a repeatable capability.
        """
        used = set(
            (
                await session.execute(
                    select(ToolCall.tool_name, ToolCall.capability).where(
                        ToolCall.tenant_id == tenant.tenant_id,
                        ToolCall.investigation_id == investigation_id,
                        ToolCall.succeeded.is_(True),
                    )
                )
            ).all()
        )
        qualified = {f"{tool}__{capability}" for tool, capability in used}
        required = scenario.ground_truth.required_capabilities
        if not required:
            return Dimension(name="tool_selection", score=1.0, detail="no required capabilities")

        # Any-of aware, for the same reason as accuracy: reaching the change through
        # `commits` rather than `deployment_history` is a different route to the same
        # discriminating fact, and often the better-evidenced one. Requiring a specific
        # endpoint scores conformity to the label's expected path.
        hit = [
            requirement
            for requirement in required
            if any(alt in qualified for alt in alternatives_of(requirement))
        ]
        missing = sorted(describe_requirement(r) for r in required if r not in hit)
        return Dimension(
            name="tool_selection",
            score=round(len(hit) / len(required), 4),
            detail=(
                "made every discriminating call"
                if not missing
                else f"never called: {', '.join(missing)}"
            ),
        )

    def _completeness(self, scenario: Scenario, report: InvestigationReport) -> Dimension:
        """Does the report have the shape the doc requires?

        Structural, not semantic: an empty risks section on a report drawn from
        sampled data is a real omission, and it can be checked without a judge.

        **Which sections count depends on the shape of the question.** `cortex.reports.shape`
        tells a factual question to leave hypotheses and recommendations empty, because nothing
        needs explaining and there is nothing to act on. Scoring those as missing would dock a
        report a third of this dimension for following the instruction it was given — the same
        mistake, in the same dimension family, as scoring "do not roll back" as advice to roll
        back.
        """
        present = {
            "summary": bool(report.executive_summary),
            "findings": bool(report.findings),
            "risks": bool(report.risks),
            "sources": bool(report.sources),
        }
        if shape_for(scenario.question) is Shape.CAUSAL:
            present["hypotheses"] = bool(report.hypotheses)
            present["recommendations"] = bool(report.recommendations)
        score = sum(present.values()) / len(present)
        missing = sorted(k for k, v in present.items() if not v)
        return Dimension(
            name="completeness",
            score=round(score, 4),
            detail="all sections present" if not missing else f"missing: {', '.join(missing)}",
        )

    def _actionability(self, scenario: Scenario, report: InvestigationReport) -> Dimension:
        """Is there something to do about it?

        On an unanswerable scenario a recommendation to *act* is wrong, and the right
        output is a next diagnostic step or nothing at all. Scored accordingly rather
        than rewarding recommendations unconditionally.
        """
        if scenario.ground_truth.declines_a_cause:
            # A recommendation *against* acting is the ideal output here, not a failure.
            # A live run was scored 0.00 for "Do not roll back or alter the signup or
            # onboarding flow on the strength of this report" — advice to hold, matched as
            # advice to act because the substring "roll back" was present. The same
            # blindness as the decoy-rejection bug, and the same fix: read the sentence,
            # not the keyword.
            acted = [
                r
                for r in report.recommendations
                if _proposes_action(r.action) and not _is_advice_against(r.action)
            ]
            return Dimension(
                name="actionability",
                score=0.0 if acted else 1.0,
                detail=(
                    f"recommended acting on an unestablished cause: {acted[0].action}"
                    if acted
                    else "correctly avoided recommending action"
                ),
            )

        if not report.recommendations:
            return Dimension(name="actionability", score=0.0, detail="no recommendation to act on")
        # Specific enough to act on: a recommendation naming an artefact beats
        # "investigate further".
        specific = [
            r
            for r in report.recommendations
            if len(r.action) > 15
            and not r.action.lower().startswith(("investigate", "look into", "consider"))
        ]
        score = len(specific) / len(report.recommendations)
        return Dimension(
            name="actionability",
            score=round(score, 4),
            detail=(f"{len(specific)}/{len(report.recommendations)} recommendations are specific"),
        )

    def _veto_precision(self, scenario: Scenario, sufficiency: AppliedSufficiency) -> Dimension:
        """Did the sufficiency gate withhold the answer it was supposed to protect?

        **The dimension that would have caught the failure nothing else did.** In run 15 the gate
        withheld the correct planted cause on both scenarios that have one -- eleven vetoes across
        eight attempts -- and every scenario still scored accuracy 1.00, because the cause survives
        in the findings once the summary claim is cut. A gate silently withholding correct answers
        passed every dimension in the suite. It was found by reading bundles by hand.

        The ground truth needed already exists: `required_signals` names the planted cause, and
        `is_unanswerable` says whether one exists at all. So the check is mechanical -- of the
        claims the gate withheld, did any contain a signal the scenario expects a correct answer
        to carry?

        Scored only on scenarios that *have* a findable cause. Where none exists, withholding a
        causal claim is the gate working, and `accuracy` already scores whether the report
        correctly declined; penalising a veto there would push against the behaviour the suite
        wants. Never gates, because a wrongly withheld claim degrades an answer rather than
        fabricating one, and this project reserves gating for the second kind.
        """
        truth = scenario.ground_truth
        if truth.declines_a_cause or not truth.required_signals:
            return Dimension(
                name="veto_precision",
                score=1.0,
                detail="no findable cause to protect; a veto here is the gate working",
            )
        if not sufficiency.rejections:
            return Dimension(name="veto_precision", score=1.0, detail="the gate withheld nothing")

        withheld = [(r.text or "").lower() for r in sufficiency.rejections]
        # Scored on what the summary lost, matching `verifier_precision`. The two dimensions ask
        # the same question of the two mechanisms that remove claims, and measuring them
        # differently made one of them report harm where there was none: a withheld claim whose
        # cause still appears in the delivered summary has cost the reader nothing.
        #
        # This is the sharper question in both directions. It caught a real loss on the verifier
        # -- a summary describing a 68.4% collapse without naming the exhausted budget behind it
        # -- and it clears a reading here where the summary says the data "stops after
        # 2026-08-03" in its own words.
        cut = [
            requirement
            for requirement in truth.required_signals
            if any(alt.lower() in text for text in withheld for alt in alternatives_of(requirement))
        ]
        protected = [
            describe_requirement(requirement)
            for requirement in cut
            if not _still_asserted_in_summary(sufficiency.report, requirement)
        ]
        if not protected:
            return Dimension(
                name="veto_precision",
                score=1.0,
                detail=(
                    f"{len(withheld)} claim(s) withheld, the summary still names the planted cause"
                    if cut
                    else f"{len(withheld)} claim(s) withheld, none carrying the planted cause"
                ),
            )
        return Dimension(
            name="veto_precision",
            score=round(1.0 - len(protected) / len(truth.required_signals), 4),
            detail=(
                f"the summary lost the planted cause: {', '.join(protected)} was in a claim the "
                "gate withheld and appears nowhere in the delivered summary"
            ),
        )

    def _verifier_precision(
        self, scenario: Scenario, verification: VerificationResult
    ) -> Dimension:
        """Did the adversarial verifier remove the answer it was supposed to check?

        **The gap this closes.** `veto_precision` asks that question of the sufficiency gate, and
        it exists because that gate was found withholding correct planted causes while every other
        dimension read clean. The verifier is the other mechanism that removes claims, it decides
        by asking a model rather than by resolving an id, and nothing asked the same question of
        it. So the more fallible of the two was the unwatched one.

        Prompted by arXiv 2608.18300 §5.3, which measures this directly: a judge reaching a
        *defensible verdict for the wrong reason* is a distinct error class, and it is invisible
        unless something scores the judge's removals against a known answer. Here that answer is
        the scenario's planted cause, which is ground truth rather than an opinion — so this stays
        a mechanical check on an LLM's decision, and never an LLM's opinion about one.

        **Why the citation gate is not scored this way.** Its reasons are typed and computed: a
        claim it removes cited an id that does not resolve, which makes removal correct however
        important the claim looked. Only a judgement call needs watching for confident mistakes.

        Never gates, for `veto_precision`'s reason: a wrongly removed claim degrades an answer
        rather than fabricating one, and gating is reserved for the second kind.
        """
        truth = scenario.ground_truth
        if truth.declines_a_cause or not truth.required_signals:
            return Dimension(
                name="verifier_precision",
                score=1.0,
                detail="no findable cause to protect; a removal here is the verifier working",
            )
        removed = [
            v.claim_text.lower()
            for v in verification.verdicts
            if v.verdict is ClaimVerdictKind.UNSUPPORTED
        ]
        if not removed:
            return Dimension(
                name="verifier_precision", score=1.0, detail="the verifier removed nothing"
            )
        # **Scored on what the summary lost, not on what was removed.** A removal can be
        # entirely correct and still cost the answer: in run 32 the verifier cut four
        # over-claiming sentences on `campaign_traffic_drop` -- each genuinely unsupported by
        # its own citation -- and the delivered summary was left describing a 68.4% collapse in
        # paid search without naming the exhausted campaign budget that caused it. The reader
        # gets the mechanism and not the thing to act on.
        #
        # Matching on the removed text alone conflated that with the opposite case, a bad claim
        # that merely mentioned the cause, and I read it wrong twice before checking the
        # summaries. `accuracy` cannot see it either: it searches the whole report, so a cause
        # surviving in a finding scores 1.00 while the summary no longer carries it.
        cut = [
            requirement
            for requirement in truth.required_signals
            if any(alt.lower() in text for text in removed for alt in alternatives_of(requirement))
        ]
        protected = [
            describe_requirement(requirement)
            for requirement in cut
            if not _still_asserted_in_summary(verification.report, requirement)
        ]
        if not protected:
            # Two different clean outcomes, reported differently: nothing bearing the cause was
            # touched, or something was and the summary still carries it. A single message for
            # both would hide which, and they call for different attention.
            return Dimension(
                name="verifier_precision",
                score=1.0,
                detail=(
                    f"{len(removed)} claim(s) removed, the summary still names the planted cause"
                    if cut
                    else f"{len(removed)} claim(s) removed, none carrying the planted cause"
                ),
            )
        return Dimension(
            name="verifier_precision",
            score=round(1.0 - len(protected) / len(truth.required_signals), 4),
            detail=(
                f"the summary lost the planted cause: {', '.join(protected)} was in a claim the "
                "verifier judged unsupported and appears nowhere in the delivered summary"
            ),
        )

    def _latency(self, investigation: Investigation) -> Dimension:
        """Linear against the 90-second budget, floored at zero. Never gates.

        **Scored on the time a reader waits, which is not `duration_ms`.** That field stops when
        the loop returns, and the citation gate, the verifier and the sufficiency gate all run
        after it -- on one measured scenario, 21.4 seconds of a 119.8-second wall clock. The
        dimension had been scoring 98.3s against the 90-second budget and calling it 0.91 while
        the answer took two minutes to arrive.

        That is not a rounding error, it is a fifth of the time, and it is concentrated in
        exactly the phases Phase 1 added to. Every "under budget" claim this suite has made
        about a run with post-loop phases was measuring the wrong interval.

        Falls back to `duration_ms` when no phase clock is available. A replayed bundle carries
        one -- rebuilt from the phase seconds it stored -- so re-scoring reports the same latency
        the run did. It did not, once: the same attempt scored 0.47 live and 1.00 on re-score,
        and since replay is what a grader change is checked with, a replay that reads better than
        the run is the more expensive of the two errors. A bundle written before phases were
        captured has nothing to rebuild from and still scores as it did originally, which keeps
        it comparable with its own run.
        """
        measured = _wall_clock_ms(investigation)
        ratio = measured / self._latency_budget_ms
        score = max(0.0, min(1.0, 1.0 - max(0.0, ratio - 1.0)))
        loop_only = measured != investigation.duration_ms
        return Dimension(
            name="latency",
            score=round(score, 4),
            detail=(
                f"{measured}ms against a {self._latency_budget_ms}ms budget"
                + (f" ({investigation.duration_ms}ms of it in the loop)" if loop_only else "")
            ),
        )


def _still_asserted_in_summary(report: InvestigationReport, requirement: Requirement) -> bool:
    """Whether the delivered summary still *asserts* this signal as a cause.

    **Not "does the word appear".** That was the test, and it exonerated three things it should
    not have -- each a summary a drafter can plausibly write:

      - "Paid search sessions collapsed 68.4% in the period following the spring campaign."
        The word is present, descriptively. This is the run-32 defect verbatim, the mechanism
        without the cause, and it scored clean.
      - "The campaign was NOT the cause; the drop remains unexplained."
        The word is present, negated. The dimension read the reader as informed.
      - `"ends"`, an alternative on `measurement_stopped`, matches inside "trends", "depends",
        "recommends" and "weekends".

    So the signal must appear on a word boundary, in a claim that does not deny it.

    **It deliberately does not also require the claim to be causal**, which was the first
    attempt and was worse. On `measurement_stopped` the right answer is "the GA4 sessions data
    simply stops being recorded after August 3" -- a statement of fact about a data incident,
    with no causal marker in it at all. Requiring `is_causal_claim` rejected the correct summary
    on every attempt of that scenario, and on two of `campaign_traffic_drop`. Reaching for the
    stricter predicate cost more than the looseness it was fixing: this dimension does not gate,
    so a false complaint sends someone chasing a defect that is not there, which has already
    happened twice today.

    What remains undetectable, and is worth stating rather than implying: "collapsed 68.4% in
    the period following the spring campaign" mentions the signal without asserting it, and no
    substring test separates that from asserting it. The negated form is caught; the merely
    descriptive one is not. Nor can this tell which noun a causal verb attaches to -- "the
    deploy caused the fall; the campaign end was coincidental" passes -- and that belongs to
    `decoy_rejection` rather than here.
    """
    for claim in report.executive_summary:
        lowered = claim.text.lower()
        # A claim denying the signal is the cause does not carry it for the reader. Reuses the
        # verifier's own list so the two cannot disagree about what a denial looks like.
        if any(phrase in lowered for phrase in _DECLINES_A_CAUSE):
            continue
        if any(
            re.search(rf"\b{re.escape(alt.lower())}\b", lowered)
            for alt in alternatives_of(requirement)
        ):
            return True
    return False


def _removal_breakdown(gate_result: GateResult, verification: VerificationResult | None) -> str:
    """Which mechanism removed which claims, named only when something was removed.

    Silent on a clean draft: a detail line reading "gate 0, verifier 0" on every passing run
    trains a reader to skip the field, and the field exists to be read on the run where it is
    not zero.
    """
    gate = gate_result.hallucination_count
    unsupported = verification.unsupported_count if verification else 0
    parts = []
    if gate:
        parts.append(f"{gate} cited nothing that resolves")
    if unsupported:
        parts.append(f"{unsupported} unsupported by its own evidence")
    if not parts:
        return ""
    return " (" + "; ".join(parts) + ")"


def _draft_attempts(investigation: Investigation | None) -> str:
    """How many drafting calls it took, named only when it took more than one.

    A repair or a brevity retry is recoverable and deliberately not an error, so it leaves no
    mark on any score. That is right, and it also means a drafter that has started needing two
    attempts every run looks exactly like one that does not.
    """
    phases = getattr(getattr(investigation, "timings", None), "phases", None)
    calls = getattr(phases.get(DRAFT), "calls", 0) if phases else 0
    return f", after {calls} drafting attempts" if calls > 1 else ""


def _wall_clock_ms(investigation: Investigation) -> int:
    """Total time a reader waits: the loop, plus every phase that runs after it.

    `duration_ms` covers the loop alone. Summing the phase clock instead would double-count,
    because the loop's own phases are in there too -- so the post-loop phases are added to it by
    name. Named explicitly rather than by subtraction: a phase added later and forgotten here
    would silently stop being counted, and a list is a thing a reader can check.
    """
    phases = getattr(getattr(investigation, "timings", None), "phases", None)
    if not phases:
        return investigation.duration_ms
    after_the_loop = sum(
        phases[name].seconds for name in (GATE, VERIFY, SUFFICIENCY) if name in phases
    )
    return investigation.duration_ms + int(after_the_loop * 1000)


def _claim_count(report: InvestigationReport) -> int:
    """Claims subject to verification.

    Scoped to exactly what the verifier judges — executive summary and finding claims —
    so `draft_reliability` compares like with like. Counting recommendations or risks
    here would put unreviewed sections in the denominator and quietly inflate the score.
    """
    return len(report.executive_summary) + sum(len(f.claims) for f in report.findings)


def _report_text(report: InvestigationReport) -> str:
    """Every piece of prose in the report, for asking whether something was *mentioned*.

    Deliberately includes rejected hypotheses. `_summary_placement` asks whether a refutation
    reached the summary or was buried further down, and a signal buried inside a discarded
    hypothesis was still mentioned -- that is the failure it measures.

    Not for asking what the report *claims*. See `_asserted_text`.
    """
    parts: list[str] = [report.question]
    parts += [c.text for c in report.executive_summary]
    for finding in report.findings:
        parts.append(finding.title)
        parts += [c.text for c in finding.claims]
    for hypothesis in report.hypotheses:
        parts += [hypothesis.statement, hypothesis.reasoning]
    parts += [r.description for r in report.risks]
    for recommendation in report.recommendations:
        parts += [recommendation.action, recommendation.rationale]
    parts += [n.note for n in report.data_quality]
    return " ".join(p for p in parts if p)


def _asserted_text(report: InvestigationReport) -> str:
    """What the report actually claims to be true.

    **`_report_text` was wrong for accuracy and inflated it.** It flattens every string in the
    report, including `statement` for hypotheses the report *contradicted*. So a report that
    named the planted cause in order to reject it matched the required signal and scored as
    having found it -- the eval credited the precise failure it exists to punish, and every
    accuracy figure quoted before this was an upper bound rather than a measurement.

    A contradicted hypothesis is a sentence the report says is false; an inconclusive one is a
    sentence it declines to stand behind. Neither is an assertion, so neither belongs in the
    search space for "did it find the cause". Their `reasoning` is excluded for the same reason
    and one more: the reasoning for rejecting a cause names that cause.

    `_DISMISSED_VERDICTS` is reused rather than re-derived, and the connection is the point --
    this codebase already decided that a dismissed hypothesis is not an assertion, in order to
    stop crediting a decoy that was considered and dropped. The same rule was simply never
    applied to the signal search, so it let the *planted* cause through the door it had closed
    against decoys.

    Risks and data-quality notes are excluded too. "This could be a tracking artefact" as a
    caveat is not the same claim as "this was a tracking artefact", and the whole point of the
    dimension is to tell those apart.
    """
    parts: list[str] = [c.text for c in report.executive_summary]
    for finding in report.findings:
        parts.append(finding.title)
        parts += [c.text for c in finding.claims]
    for hypothesis in report.hypotheses:
        if hypothesis.verdict not in _DISMISSED_VERDICTS:
            parts += [hypothesis.statement, hypothesis.reasoning]
    for recommendation in report.recommendations:
        parts += [recommendation.action, recommendation.rationale]
    return " ".join(p for p in parts if p)
