"""Sufficiency gate — does the evidence we hold support a definitive answer about cause?

ADR 0005, decision 6. One call, asked separately from the question about the business,
with veto power over the report's causal claims.

## Why a separate call rather than a better prompt

The failure this exists to stop is measured. *Sufficient Context: A New Lens on Retrieval
Augmented Generation Systems* (Joren et al., ICLR 2025, arXiv:2411.06037) defines
sufficient context as a property of (question, context) — the retrieved context alone
plausibly supports a definitive answer — **without reference to the ground-truth answer**,
builds an autorater for it reported at 93% accuracy, and finds that with sufficient
context frontier models still emit a wrong answer 13–25% of the time (14–16% for the best).
Two of its findings are the whole argument for this module:

  - **In the sufficient-context case models hallucinate more often than they abstain.**
    Having the answer available does not make a model cautious; it makes it confident.
  - **Adding context suppresses abstention, hard.** Claude 3.5 Sonnet abstains on 84.1% of
    questions without retrieval and 52% with it; Gemini 1.5 Pro goes from 100% to 18.6%.
    Their proposed mechanism is increased confidence in the presence of *any* contextual
    information — which is exactly what a pile of tool results is.

Their intervention combines the sufficiency signal with model self-confidence to decide
generate-versus-abstain, and reports up to **+2–10% correct-among-answered**. Small, and
aimed squarely at our own failure: we handed an agent thirteen silent series and a stack of
unrelated tool results, and the stack itself licensed a confident answer.

So this is a second call, not a longer prompt. It never sees the report's conclusion — the
same reason the causal half of `cortex.reports.verifier` does not: a judge shown the answer
grades the answer, and what we need graded is the evidence.

## What it can and cannot decide

**Whether the report asserts a cause at all is decided in code**, by `is_causal_claim`
imported from the verifier so the two interventions cannot drift apart. That predicate is
also the latency control: a report that names no cause never makes this call, so the
descriptive and refusing reports — the tracking-outage answer, the "this is within normal
variation" answer — pay nothing.

**Whether the evidence carries a cause is what the model decides**, because it is the one
part no predicate can reach.

**The bar was set too high once and this records where it landed.** The first prompt asked
whether the observations *establish* a cause, and required them to "rule out the other changes in
the same window" -- then told the model that "not sufficient is the common answer". Measured on
run 15: eleven vetoes across eight attempts, including the correct planted cause on both scenarios
that have one ("the mobile onboarding modal rework (PR #913, merged 2026-07-14) broke mobile signup
completion", withheld). It also starved the two phases downstream, which act only on a causal
assertion and therefore never ran at all.

That bar is decision 5's question -- *is a causal claim available in principle* -- and decision 5
is a **disclosure** precisely because the honest answer is almost always no. Asking it here gave
decision 5's question decision 6's veto, and refused correct answers.

Joren et al.'s definition is the narrower one this should have been all along: sufficient context
means the context *plausibly supports* a definitive answer, judged without reference to the
ground-truth answer. Not proven, not identified -- derivable. So the prompt now asks whether a
reader of these observations alone could reach a supported answer, or would have to supply the
connection themselves. The prompt's rules are the ones this project has already paid
for: describing a movement is not explaining it, a coincident change is not a cause, an
empty result is nobody looking rather than nothing happening, and the instrument has to be
ruled out before the world is blamed.

Deliberately **not** in the prompt: decision 3's "two independent lines of evidence". It is
Phase 3, it has its own known weakness — in a single-warehouse architecture two lines are
usually two views of one source — and smuggling it in here would make this gate refuse
nearly everything while attributing the refusal to the wrong decision.

## What the veto does

A refusal must carry the thing that would lift it, so `missing` is what reaches the reader.
The veto itself is mechanical once the verdict is in:

  - causal claims are removed from the executive summary and findings, exactly as the
    verifier removes an unsupported claim, and each removal is recorded as a `Rejection` on
    the report row so the sentence itself survives in the audit record rather than in the
    prose;
  - supported causal hypotheses are downgraded to inconclusive, because a supported
    hypothesis is an assertion the reader and the eval both count — withholding the summary
    claim and leaving the hypothesis moves the causal story rather than withdrawing it;
  - confidence becomes `insufficient_evidence`;
  - and if nothing is left in the executive summary, the report is refused outright, which
    is the same rule the gate and the verifier already apply: a summary with no surviving
    claim is not a degraded answer, it is no answer.

**Recommendations are left alone**, and that is a judgement rather than an oversight. A
recommendation's rationale is causal by nature, matching the predicate on almost anything,
and dropping actions on this verdict would be a far larger blast radius than +2–10%
correct-among-answered justifies. Its causal warrant is withdrawn visibly instead, by the
confidence and the risk that travel with it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from cortex.agents.llm import LLM, LLMError, Message, Usage
from cortex.db.models import Evidence
from cortex.reports.gate import Rejection, RejectionReason, ReportRejected
from cortex.reports.schema import (
    Claim,
    Confidence,
    Finding,
    InvestigationReport,
    Risk,
)
from cortex.reports.schema import Verdict as HypothesisVerdict
from cortex.reports.verifier import (
    causal_claims,
    causal_hypotheses,
    is_causal_claim,
    load_cited_evidence,
    render_evidence,
    without_the_cause,
)
from cortex.tenancy.context import TenantContext

#: The gate is asked one question about a lot of evidence, so it needs little room to
#: answer. Same reasoning as the completeness judge's budget.
_MAX_TOKENS = 1024

#: Per-observation payload budget, and it is deliberately a third of the verifier's.
#:
#: The two calls need different things. A verdict on "fell 18%" turns on whether 18 appears
#: in the payload, so it needs depth. Sufficiency turns on what was looked at and what came
#: back at all, so it needs breadth — and a report cites many rows.
_MAX_EVIDENCE_CHARS = 2000

#: Total prompt budget for the evidence section. Beyond it, observations are omitted and
#: **the omission is stated**: a gate that silently saw less than the report holds would
#: refuse for the wrong reason, and "we could not show you everything" is the difference
#: between a refusal about the data and a refusal about the prompt.
_MAX_EVIDENCE_TOTAL_CHARS = 60_000

SUFFICIENCY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["sufficient", "missing", "reason"],
    "properties": {
        # Asked first, so the judgement is made before there is a list on the page to
        # rationalise. The fields of a structured output are generated in order.
        #
        # **This order is deliberately the opposite of the verifier's, and the difference is
        # the direction each one can fail.** `VERDICT_SCHEMA` emits its reason before its
        # verdict, because a verifier that decides first writes a justification for what it
        # already said. Here the risk runs the other way: naming what is missing is easy and
        # always possible -- more evidence can always be wished for -- so a `missing`-first
        # order would leave the model a list it must then agree with, and the agreeing answer
        # is "insufficient". That is over-abstention, and a gate that refuses everything
        # scores perfectly on a suite whose hard bar is a hallucination count of zero.
        #
        # arXiv 2605.23970 recommends grounding before scoring and reports revision
        # susceptibility falling from 75-85% to 5-22%, which argues for flipping this. It is
        # not flipped, because that paper measures a *judge of quality* and this is a judge of
        # *sufficiency*, where the cheap answer is refusal rather than approval. Its §8 warning
        # -- that a rigid format with no room to reason destroys the gate entirely -- does not
        # apply: `reason` exists and is required.
        #
        # The deciding evidence does not exist yet. It is a paired should-answer /
        # should-decline scenario set, where flipping this order would show up as the decline
        # twins improving while their answerable parents got worse. Until that exists, changing
        # this trades a measured concern for an unmeasured one.
        "sufficient": {
            "type": "boolean",
            "description": (
                "True only if these observations, on their own, support a definitive answer "
                "about what caused the movement. False if they establish what happened but "
                "not why."
            ),
        },
        "missing": {
            "type": "array",
            "description": (
                "What would have to be observed for a cause to be established here — a "
                "comparable series the change did not touch, a dated deploy or config "
                "change inside the onset window, a per-emitter ingest record. Each item "
                "concrete enough that somebody could go and fetch it. Empty when "
                "sufficient."
            ),
            "items": {"type": "string"},
        },
        "reason": {
            "type": "string",
            "description": (
                "One or two sentences: what these observations do establish, and where they stop."
            ),
        },
    },
}

SUFFICIENCY_PROMPT = """\
You decide one thing: whether these observations *contain* what an answer about cause would \
need, or whether the answer is simply not in here. No analyst's conclusion is included below \
and you must not try to reconstruct one — you are judging the evidence, not an answer.

You are **not** deciding whether a cause is proven. Observational data almost never proves \
one, and a bar that high would refuse every real answer. You are deciding whether someone \
reading only these observations could reach a supported answer about why, or whether they \
would have to supply the connection themselves.

Sufficient looks like: the movement is here, and so is something that plausibly explains it, \
dated compatibly, with enough detail to say which thing it was.

Not sufficient looks like:
- Only the movement. A series that fell, however precisely dated, is a description — nothing \
here says what changed.
- An observation that returned no rows, offered as support. That shows nobody saw anything, \
not that nothing happened.
- Observations that cannot separate "the thing stopped happening" from "we stopped recording \
it". The instrument comes before the world.
- A candidate dated after the movement began. That is not a cause whatever else is here.

When it is not sufficient, say what is missing concretely enough that someone could go and \
fetch it.\
"""


@dataclass(frozen=True, slots=True)
class AppliedSufficiency:
    """A report with the gate's verdict applied, and what applying it removed."""

    report: InvestigationReport
    rejections: list[Rejection] = field(default_factory=list)
    #: Claims removed. Downgraded hypotheses are in `rejections` and not counted here: a
    #: hypothesis is still in the report, saying it is no longer established.
    withheld: int = 0


@dataclass(frozen=True, slots=True)
class SufficiencyDecision:
    """Whether the evidence carries a causal answer, and what is missing if it does not."""

    #: Whether the report asserts a cause. False means the gate did not run and had no
    #: reason to: nothing about this report is disclosed, and no call was made.
    needed: bool
    #: False when the check could not be obtained. `sufficient` is then meaningless and a
    #: caller must not read it as a pass.
    ran: bool = False
    sufficient: bool = True
    missing: tuple[str, ...] = ()
    reason: str = ""
    error: str | None = None
    usage: Usage = field(default_factory=Usage)

    @property
    def vetoes(self) -> bool:
        return self.needed and self.ran and not self.sufficient

    @property
    def missing_sentence(self) -> str:
        """What would lift the refusal, in one sentence.

        Falls back to the model's own reason, because a refusal that names nothing is the
        failure mode this ADR spends a paragraph on: the alternative to an informative
        refusal is not silence, it is a human inventing a cause.
        """
        if self.missing:
            return "; ".join(self.missing[:4])
        return self.reason

    def apply(self, report: InvestigationReport) -> AppliedSufficiency:
        """Withhold what the evidence cannot carry.

        Must be given the report `assess` was given: claim locations are recomputed here
        rather than carried, so that the sentence a rejection names is the sentence that was
        removed even if a caller reordered something in between.

        Raises `ReportRejected` when the veto empties the executive summary — the same rule
        the gate and the verifier apply, for the same reason.
        """
        if not self.needed:
            return AppliedSufficiency(report=report)
        if not self.ran:
            return AppliedSufficiency(
                report=report.model_copy(
                    update={
                        "risks": [
                            *report.risks,
                            Risk(
                                description=(
                                    "This report names a cause, and whether the evidence "
                                    "gathered can support one was not checked "
                                    f"({self.error or 'the check did not run'}). The claims "
                                    "are grounded in real observations; that they add up to "
                                    "a cause is the drafter's judgement alone."
                                )
                            ),
                        ]
                    }
                )
            )
        if self.sufficient:
            # Nothing is disclosed on a report that passed. A line saying "the sufficiency
            # check passed" would appear on nearly every causal report, and a disclosure
            # that appears everywhere stops being read on the one that needed it.
            return AppliedSufficiency(report=report)

        rejections: list[Rejection] = []
        withheld = {location for location, _ in causal_claims(report)}

        summary_kept: list[Claim] = []
        for index, claim in enumerate(report.executive_summary):
            location = f"executive_summary[{index}]"
            if location in withheld:
                rejections.append(self._rejection(location, claim.text))
                continue
            summary_kept.append(claim)

        findings_kept: list[Finding] = []
        for f_index, finding in enumerate(report.findings):
            claims_kept: list[Claim] = []
            for c_index, claim in enumerate(finding.claims):
                location = f"findings[{f_index}].claims[{c_index}]"
                if location in withheld:
                    rejections.append(self._rejection(location, claim.text))
                    continue
                claims_kept.append(claim)
            if not claims_kept:
                continue
            # The title too, and it is the leak this whole block used to have. A finding's
            # title carries no `evidence_ids`, so grounding cannot see it and the verifier has
            # nothing to judge it against -- and the drafter puts the answer there, because
            # that is what a title is for. Withholding the causal *claims* and delivering the
            # cause in the heading above them withholds nothing.
            title = finding.title
            if is_causal_claim(title):
                neutral = without_the_cause(title) or claims_kept[0].text[:200]
                rejections.append(
                    Rejection(
                        location=f"findings[{f_index}].title",
                        reason=RejectionReason.NO_SURVIVING_EVIDENCE,
                        detail=(
                            "sufficiency: the title named a cause the evidence does not "
                            f"support. Missing: {self.missing_sentence}"
                        )[:1000],
                        text=title[:500],
                    )
                )
                title = neutral
            findings_kept.append(finding.model_copy(update={"claims": claims_kept, "title": title}))

        if not summary_kept:
            raise ReportRejected(
                "the evidence gathered does not support a definitive answer about cause, "
                "and every claim in the executive summary asserted one. What would settle "
                f"it: {self.missing_sentence}"
            )

        withheld_claims = len(rejections)
        hypotheses = list(report.hypotheses)
        for index, hypothesis in causal_hypotheses(report):
            hypotheses[index] = hypothesis.model_copy(
                update={"verdict": HypothesisVerdict.INCONCLUSIVE}
            )
            rejections.append(
                Rejection(
                    location=f"hypotheses[{index}]",
                    reason=RejectionReason.NO_SURVIVING_EVIDENCE,
                    detail=(
                        "sufficiency: downgraded to inconclusive: the evidence does not "
                        f"support a definitive cause. Missing: {self.missing_sentence}"
                    )[:1000],
                    text=hypothesis.statement[:500],
                )
            )

        applied = report.model_copy(
            update={
                "executive_summary": summary_kept,
                "findings": findings_kept,
                "hypotheses": hypotheses,
                "confidence": Confidence.INSUFFICIENT_EVIDENCE,
                "risks": [
                    *report.risks,
                    Risk(
                        description=(
                            f"{withheld_claims} causal claim(s) were withheld: a separate "
                            "check of the evidence gathered, made without seeing this "
                            "report, found that it does not support a definitive answer "
                            f"about cause. {self.reason} What would settle it: "
                            f"{self.missing_sentence}. What remains describes what "
                            "happened, and stops short of why."
                        )[:1000]
                    ),
                ],
            }
        )
        return AppliedSufficiency(report=applied, rejections=rejections, withheld=withheld_claims)

    def _rejection(self, location: str, text: str) -> Rejection:
        return Rejection(
            location=location,
            # The gate's vocabulary, as the verifier's rejections use it: the enum lives in
            # `gate.py` and the detail prefix is what tells the three mechanisms apart on a
            # persisted row.
            reason=RejectionReason.NO_SURVIVING_EVIDENCE,
            detail=(
                "sufficiency: the evidence does not support a definitive answer about "
                f"cause. Missing: {self.missing_sentence}"
            )[:1000],
            text=text[:500],
        )


class SufficiencyGate:
    def __init__(
        self,
        llm: LLM,
        *,
        max_evidence_chars: int = _MAX_EVIDENCE_CHARS,
        max_total_chars: int = _MAX_EVIDENCE_TOTAL_CHARS,
    ) -> None:
        self._llm = llm
        self._max_evidence_chars = max_evidence_chars
        self._max_total_chars = max_total_chars

    async def assess(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        *,
        investigation_id: uuid.UUID,
        question: str,
        report: InvestigationReport,
    ) -> SufficiencyDecision:
        """Decide whether this report's causal claims may stand.

        Never raises. The gate's veto is real, but a gate that could break an investigation
        by being unavailable would be a worse trade than one that occasionally declines to
        judge — so a provider failure produces `ran=False`, which discloses itself and
        withholds nothing.

        `question` is the question as asked, not the report's restatement of it. A report
        that quietly narrowed the question would otherwise have its narrowed version graded.
        """
        if not causal_claims(report) and not causal_hypotheses(report):
            # No call. The whole latency argument for this gate is here: a report that
            # names no cause has nothing for it to veto, and paying a round trip to be told
            # so would put the cost on every investigation instead of the causal ones.
            return SufficiencyDecision(needed=False)

        evidence = await load_cited_evidence(
            session, tenant, investigation_id=investigation_id, report=report
        )
        rendered = self._render(question, evidence)
        if rendered is None:
            return SufficiencyDecision(
                needed=True,
                ran=False,
                error="none of the report's cited evidence could be loaded",
            )

        try:
            payload, usage = await self._llm.structured(
                system=SUFFICIENCY_PROMPT,
                messages=[Message(role="user", content=rendered)],
                schema=SUFFICIENCY_SCHEMA,
                max_tokens=_MAX_TOKENS,
            )
        except LLMError as exc:
            return SufficiencyDecision(
                needed=True, ran=False, error=f"{type(exc).__name__}: {exc}"[:300]
            )

        if "sufficient" not in payload:
            # Not read as a refusal. A parse failure that vetoed would delete a report's
            # conclusions on the strength of a malformed field, and a check that fails
            # destructively gets turned off.
            return SufficiencyDecision(
                needed=True,
                ran=False,
                error="the sufficiency verdict came back without its verdict field",
                usage=usage,
            )

        missing = tuple(
            str(item).strip()[:300] for item in payload.get("missing") or [] if str(item).strip()
        )
        reason = str(payload.get("reason") or "").strip()[:500]
        sufficient = bool(payload.get("sufficient"))
        if not sufficient and not missing and not reason:
            # A refusal that names nothing is the uninformative refusal the ADR spends a
            # paragraph warning about, and it is indistinguishable from a shrug. Treated as
            # a check that did not run, which discloses itself and withholds nothing.
            return SufficiencyDecision(
                needed=True,
                ran=False,
                error="the sufficiency check refused without naming anything missing",
                usage=usage,
            )

        return SufficiencyDecision(
            needed=True,
            ran=True,
            sufficient=sufficient,
            missing=missing,
            reason=reason,
            usage=usage,
        )

    # ------------------------------------------------------------------ internals

    def _render(self, question: str, evidence: dict[uuid.UUID, Evidence]) -> str | None:
        rows = sorted(evidence.values(), key=lambda r: (r.tool_name, r.capability, str(r.id)))
        blocks = render_evidence(rows, max_chars=self._max_evidence_chars)
        if not blocks:
            return None

        kept: list[str] = []
        budget = self._max_total_chars
        for block in blocks:
            if len(block) > budget:
                break
            kept.append(block)
            budget -= len(block)
        omitted = len(blocks) - len(kept)
        if not kept:
            # One observation larger than the whole budget. Shown truncated rather than
            # dropped: an empty evidence section would read as "we hold nothing", which is
            # a refusal for a reason that is about this prompt and not about the data.
            kept = [blocks[0][: self._max_total_chars]]
            omitted = len(blocks) - 1

        header = f"OBSERVATIONS THE INVESTIGATION HOLDS ({len(blocks)} item(s)"
        header += f", {omitted} not shown here):\n" if omitted else "):\n"
        return (
            header
            + "\n\n".join(kept)
            + f"\n\nTHE QUESTION ASKED: {question}\n\n"
            + "Do these observations, on their own, support a definitive answer about what "
            + "caused the movement this question is about?"
        )
