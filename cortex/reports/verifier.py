"""Adversarial verifier — the second grounding mechanism.

The gate proves a citation *resolves*. It cannot prove the citation *supports the
claim*: a real evidence row about mobile sessions attached to a sentence about
enterprise revenue passes the gate and is still wrong. Catching that needs
reading, so this pass uses a model — but under conditions that make it useful
rather than decorative:

  - **Adversarial framing.** The prompt asks whether the evidence *fails* to
    support the claim. A verifier asked "is this supported?" agrees with almost
    anything; asked to find the gap, it finds real ones.
  - **Cited evidence only.** Each claim is judged against the rows it cites and
    nothing else. Given the whole evidence set, a verifier reasons from the
    investigation's overall story and confirms claims the specific citation does
    not support.
  - **No access to the drafter's reasoning.** It sees the claim and the data, not
    the argument that produced them.

Verdicts are structured, not prose. `unsupported` removes the claim;
`overstated` keeps it and lowers confidence — a claim that is directionally right
but too strong is worth keeping with a caveat, whereas removing it would lose a
true finding.

## Causal claims are re-derived rather than confirmed (ADR 0005, decision 7)

The three properties above are closer to the literature than expected, and one thing
was still missing: the prompt **shows the model the claim and asks whether the evidence
supports it**, which anchors the judgement on the proposed conclusion. CoVe
(arXiv:2309.11495) measured the difference the anchoring makes — its "factored" variant
answers each verification question *as a separate prompt that does not contain the
original answer*, verbatim rationale "not prone to simply copying or repeating it", and
that one detail is what separates list precision 0.17 → 0.36 and FACTSCORE 55.9 → 71.4
from intrinsic self-correction, which is measurably negative.

So a claim that asserts *X caused Y* — and only such a claim, decided by
`is_causal_claim`, in code — gets a second reading of its own cited evidence in a
context that contains **no claim at all**: `OPEN_QUESTION_PROMPT` asks the evidence what,
if anything, it establishes about the cause of the movement the investigation was asked
about. The claim's verdict is then reconciled with that reading **mechanically**, by
`_reconcile`, because two of the three interesting outcomes are decidable in code:

  - the open reading establishes no cause → the causal claim is `overstated`;
  - it names a cause dated after the movement → `unsupported`, by arithmetic, which is
    the elimination rule of decision 2 applied to a claim rather than a hypothesis;
  - it names a cause whose date holds → the claim stands.

Two deliberate limits, because both are easy to get wrong in the flattering direction.

**It runs only where it can change something.** The re-derivation is paid for only when
the anchored pass returned `supported` for a causal claim. A claim already judged
overstated or unsupported cannot be lowered further by this check, and false accepts on
causal claims are the failure the intervention exists to reduce — so the call is
skipped everywhere else, and reports with no causal claim pay nothing at all. The
consequence worth naming: a `supported` verdict on a causal claim now requires the
anchored judge *and* an independent, claim-free reading of the same rows to agree.

**Whether two prose causes are the same cause is not decided here.** If the open reading
names a cause, this code does not string-match it against the claim. A token overlap
between "the pricing change" and "the ingest path change" is the word "change", and a
model asked to compare them would need the claim in its context — which is the anchoring
this whole path exists to remove. The open reading's named cause travels into the verdict
`reason` instead, so a disagreement is visible in the persisted rejection rather than
silently acted on.
"""

from __future__ import annotations

import asyncio
import datetime
import enum
import json
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.agents.llm import LLM, LLMAuthenticationFailed, LLMError, Message, Usage
from cortex.db.models import Evidence
from cortex.db.threads import citable_investigation_ids
from cortex.reports.gate import Rejection, RejectionReason
from cortex.reports.schema import (
    Claim,
    Confidence,
    Finding,
    Hypothesis,
    InvestigationReport,
)
from cortex.reports.schema import Verdict as HypothesisVerdict
from cortex.tenancy.context import TenantContext

#: How many claims are judged at once.
#:
#: Bounded deliberately. Claims are independent, so they could all run at once — but
#: a report can carry twenty-plus claims, and twenty simultaneous requests is a
#: reliable way to provoke the provider capacity errors that then surface as
#: unverified claims, which is the exact failure this is meant to reduce. Four keeps
#: the wall clock down without turning the verifier into a load generator.
_VERDICT_CONCURRENCY = 4

#: Phrases that make a sentence assert a cause.
#:
#: A predicate rather than a model call, on this codebase's rule that an LLM decides only
#: what cannot be decided in code. "Does this sentence claim that X caused Y" is a
#: question about its grammar, and paying a round trip per claim to have a model answer it
#: would cost more than the verification it is deciding whether to run.
#:
#: Deliberately requires an explicit causal connective. The cost of the two errors is
#: asymmetric but neither is free: a missed causal claim gets the verification we ship
#: today, and a false positive spends one call and can only ever *lower* a verdict — so
#: the list is generous with connectives and strict about `_DECLINES_A_CAUSE` below.
_CAUSAL_MARKERS = (
    "caused",
    "causing",
    "cause of",
    "root cause",
    "because",
    "due to",
    "led to",
    "leading to",
    "drove",
    "driven by",
    "driving",
    "resulted in",
    "result of",
    "triggered",
    "triggering",
    "responsible for",
    "attributable to",
    "attributed to",
    "stems from",
    "stemming from",
    "explains",
    "explained by",
    "explanation for",
    "brought about",
    "knock-on",
    "thanks to",
    "the reason",
    # Transitive causal verbs, which the connective list misses entirely.
    #
    # Found by testing: "the refresh broke the signup path" and "PR 565 broke checkout" both
    # scored as *descriptive*, so a causal claim phrased this way escaped both Phase 1
    # interventions -- no fresh-context re-derivation and no sufficiency gate. It is not a
    # hypothetical phrasing: a real report on our own data proposed that PR 567 "changed
    # download CTAs ... a plausible source of added friction", which is this shape.
    #
    # Only the unambiguous ones. "reduced", "increased" and "made" are deliberately left out
    # because they are as often intransitive description as causal assertion -- "signups
    # reduced" is a movement, "the outage reduced signups" is a cause, and a substring match
    # cannot tell a subject from an object. Including them would veto descriptive reports,
    # which is the failure mode `_DECLINES_A_CAUSE` exists to prevent from the other side.
    "broke ",
    "broken by",
    "introduced friction",
    "introduced a defect",
    "introduced a regression",
    "regressed",
)

#: Phrases that mean the sentence *declines* to assert a cause.
#:
#: Checked first and decisive. "The data cannot establish a cause" contains a causal
#: connective and asserts nothing, and routing it down the causal path would end with the
#: report disclosing that a claim was "stronger than the evidence warrants" — a caveat
#: attached to the one behaviour this whole ADR is trying to produce. Declining to name a
#: cause is a correct answer, and it must not be penalised for saying the word.
_DECLINES_A_CAUSE = (
    "did not cause",
    "does not cause",
    "was not caused",
    "were not caused",
    "not caused by",
    "no evidence",
    "cannot establish",
    "could not establish",
    "cannot be established",
    "does not establish",
    "does not explain",
    "cannot explain",
    "cannot be attributed",
    "not attributable",
    "cannot be determined",
    "could not identify",
    "rules out",
    "ruled out",
    # The gerund was missing while the other two were present, and it is the form a report
    # actually uses: "limited to a CI runner pin and a dependency bump, ruling out a
    # deploy-caused funnel regression". The sufficiency gate withheld that sentence -- an
    # elimination, backed by the diff it names -- as though it asserted the cause it rules out.
    "ruling out",
    "rule out",
    "is not the cause",
    "not the cause",
    "no cause",
    "not because",
    "unrelated to",
    "does not explain",
    "do not explain",
    "cannot explain",
    "did not cause",
    "does not cause",
    "not caused by",
)


#: A negated subject before a causal marker: "no code change explains a session change",
#: "nothing in the diff caused it".
#:
#: A phrase list cannot catch this -- the noun between the negation and the verb is arbitrary --
#: so it is the one pattern here expressed as a regex. Kept narrow deliberately: the negation has
#: to be within a few words of the marker, because a sentence that says "no" early and asserts a
#: cause later is an assertion, and widening the window would start excusing those.
#: Idioms where a negation *strengthens* an assertion instead of eliminating one. Each was a
#: real leak: "there is no doubt the campaign ending caused the fall" reads as an elimination to
#: any pattern that only looks for a negation near a causal verb, and it is the opposite.
_ASSERTING_NEGATIONS = (
    "no doubt",
    "no question",
    "no denying",
    "no fewer",
    "none other",
    "no one disputes",
    "nobody disputes",
    "no one denies",
)

#: A negated subject immediately governing a causal verb: "no code change explains a session
#: change", "nothing in the diff caused the drop".
#:
#: **Deliberately narrow, and the direction of the error is the reason.** A false negative here
#: disables four separate defences for that sentence -- the sufficiency gate makes no call at
#: all, the fresh-context re-derivation is skipped, data-trust enforcement returns early, and
#: nothing enters the withheld set -- so an unsupported causal claim reaches a reader unchecked.
#: A false positive merely withholds an elimination, which is a worse report and not a wrong one.
#:
#: The first version of this spanned any negation within 40 characters of a causal verb and
#: excised the whole match, verb included. That deleted the assertion rather than the
#: elimination: "neither team noticed, but PR 913 caused the drop" scanned as non-causal. So the
#: span may not cross a clause boundary -- a comma was all it took -- and the idioms above are
#: excluded outright.
_NEGATED_CAUSE = re.compile(
    r"\b(?:no|nothing|neither|none)\b[^.;,]{0,40}?"
    r"\b(?:explains?|explained|caused?|causes|drove|drives|triggered|triggers|led to|"
    r"responsible for|attributable to|due to|driven by|stems from|the reason)\b"
)


#: Descriptive idioms that happen to contain a causal marker.
#:
#: Distinct from `_DECLINES_A_CAUSE`, which is about sentences that *refuse* to name a cause.
#: These do not refuse anything; they are simply not causal, and they collide with a marker by
#: accident. Kept as its own set because conflating the two would make the refusal list mean two
#: things, and the next reader would add the wrong kind of phrase to it.
#:
#: "broke down into" is ours: `cortex.analysis.changepoints` segments a series into regimes, and
#: a report describing that ("the series broke down into three regimes") was scored as asserting
#: a cause by the `broke ` marker added above. Our own vocabulary tripped our own predicate.
_DESCRIPTIVE_IDIOMS = (
    "broke down into",
    "broke down as",
    "broken down by",
    "broken down into",
)


def is_causal_claim(text: str) -> bool:
    """Whether this sentence asserts that one thing caused another.

    The gate on both Phase 1 interventions: the fresh-context re-derivation below, and
    the sufficiency gate in `cortex.reports.sufficiency`. One definition, imported by
    both, because two would drift and the drift would be silent — the verifier would
    re-derive claims the sufficiency gate never guarded, or the reverse, and either way
    the disagreement would only be visible by reading both prompts.
    """
    lowered = text.lower()
    if any(idiom in lowered for idiom in _DESCRIPTIVE_IDIOMS):
        return False
    # Eliminations are excised, then the question is asked of what remains -- on *both* paths.
    # The phrase list short-circuited while the regex excised, so "deploys were ruled out, but
    # the campaign ending caused the majority of the fall" scanned as non-causal on the strength
    # of its first clause. Two mechanisms doing the same job by different rules is how they
    # disagree silently.
    #
    # The negated-cause pass runs *first*, and the order is load-bearing: several entries in
    # `_DECLINES_A_CAUSE` are themselves negations ("no evidence", "no cause"), so excising them
    # first destroys the negation the pattern needs and leaves the bare verb behind. "There is
    # no evidence that the deploy caused the fall" then scanned as an assertion.
    remainder = lowered
    if not any(idiom in lowered for idiom in _ASSERTING_NEGATIONS):
        remainder = _NEGATED_CAUSE.sub(_excise_if_one_clause, remainder)
    for marker in _DECLINES_A_CAUSE:
        remainder = remainder.replace(marker, " ")
    # Eliminations are excised, not short-circuited on, and the difference matters. Ruling a
    # candidate out *is* the analysis -- it is what Kepner-Tregoe's IS/IS-NOT step produces and
    # what makes a remaining cause worth believing -- so a gate that removes eliminations
    # deletes the reasoning and keeps the conclusion, which is the opposite of its purpose.
    #
    # But returning False on any sentence *containing* an elimination would blind the gate to
    # "no single deploy explains it, but the campaign ending caused the fall", which asserts a
    # cause in its second clause. So each elimination is cut out and the question is asked of
    # what remains. The first version short-circuited and let that sentence through, which is
    # the dangerous direction for a gate whose job is catching unsupported assertions.
    return any(marker in remainder for marker in _CAUSAL_MARKERS)


#: Words that open a new clause. A negation on one side of these does not govern a causal verb on
#: the other: "there was no warning **before** the deploy caused the outage" asserts the cause,
#: and only the subordinator separates it from "no deploy caused the outage", which denies one.
#: Commas are already excluded by the pattern; these are the ones that need no punctuation.
_CLAUSE_BREAKS = (
    " before ",
    " after ",
    " while ",
    " since ",
    " when ",
    " because ",
    " so ",
    " but ",
    " and ",
    " though ",
    " although ",
    " until ",
    " whereas ",
)


def _excise_if_one_clause(match: re.Match[str]) -> str:
    """Remove a negated cause only where the negation and the verb share a clause.

    Called as the replacement for every `_NEGATED_CAUSE` match, so the decision is made per
    match rather than per sentence: one elimination in a sentence does not excuse a separate
    assertion elsewhere in it, and one assertion does not preserve a separate elimination.
    """
    span = match.group(0)
    if any(break_word in span for break_word in _CLAUSE_BREAKS):
        return span
    return " "


def causal_claims(report: InvestigationReport) -> list[tuple[str, Claim]]:
    """Every claim in the report's *asserted* prose that names a cause, with its location.

    Locations use the verifier's own vocabulary, so a sufficiency rejection and a verifier
    rejection point at the same sentence in the same words on the persisted report row.
    """
    found: list[tuple[str, Claim]] = [
        (f"executive_summary[{index}]", claim)
        for index, claim in enumerate(report.executive_summary)
        if is_causal_claim(claim.text)
    ]
    found += [
        (f"findings[{f_index}].claims[{c_index}]", claim)
        for f_index, finding in enumerate(report.findings)
        for c_index, claim in enumerate(finding.claims)
        if is_causal_claim(claim.text)
    ]
    return found


def causal_hypotheses(report: InvestigationReport) -> list[tuple[int, Hypothesis]]:
    """Supported hypotheses that assert a cause, with their index.

    Included because a supported hypothesis is an assertion the reader and the eval both
    read as one — `_asserted_text` counts supported hypotheses and excludes contradicted
    ones — so a causal story that survives only in this field has not been withheld at
    all, it has been moved.

    `cause_at` is the stronger signal and is checked first: a hypothesis carrying a
    machine-readable cause date names an intervention by construction, whatever its prose
    does.
    """
    return [
        (index, hypothesis)
        for index, hypothesis in enumerate(report.hypotheses)
        if hypothesis.verdict is HypothesisVerdict.SUPPORTED
        and (hypothesis.cause_at is not None or is_causal_claim(hypothesis.statement))
    ]


class Verdict(enum.StrEnum):
    SUPPORTED = "supported"
    OVERSTATED = "overstated"
    UNSUPPORTED = "unsupported"


VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    # **`reason` is emitted before `verdict`, and the order is the point.**
    #
    # Structured output is one left-to-right pass over these properties, so a field placed above
    # its own inputs has to be answered before those inputs exist. With the verdict first, the
    # model committed to supported/unsupported/overstated and then wrote a justification for
    # whatever it had already said -- which is rationale anchoring: the verdict holds while the
    # reasoning is rewritten to fit it. arXiv 2605.23970 measures exactly this and finds that
    # grounding the evidence before scoring cuts revision susceptibility from 75-85% to 5-22%.
    #
    # This codebase has already paid for the same mistake once: the report schema emitted its
    # executive summary and confidence before the findings they summarise, and the fix was to
    # move them last. That precedent is why this one is worth making without waiting for a
    # measurement -- it is the same defect in a smaller schema.
    #
    # This is the cheap half of that paper's intervention. The fuller version locks a *quotation*
    # from the evidence before scoring, which costs output tokens on every claim -- roughly ten
    # per report against a verify phase already costing 12s -- so it is deliberately not taken
    # here without a measurement to justify the latency.
    "required": ["reason", "verdict"],
    "properties": {
        "reason": {
            "type": "string",
            "description": (
                "First, before deciding: name what the cited evidence actually shows about this "
                "claim, and the specific gap if there is one. One sentence."
            ),
        },
        "verdict": {
            "type": "string",
            "enum": [v.value for v in Verdict],
            "description": (
                "The judgement that follows from the reason above. "
                "supported: the cited evidence establishes the claim as written. "
                "overstated: the claim's substance holds but it goes beyond the evidence -- "
                "stronger, more certain or more causal than the data warrants, or carrying a "
                "figure the evidence does not show. "
                "unsupported: the cited evidence does not establish the claim, is "
                "about something else, or contradicts it."
            ),
        },
    },
}

SYSTEM_PROMPT = """\
You are verifying one claim from an analyst's report against the exact evidence it \
cites. Your job is to find the gap, not to agree.

Rules:
- Judge the claim ONLY against the evidence shown. Do not use background knowledge, \
and do not reason from what the wider investigation was probably about.
- A number that appears nowhere in the evidence is unsupported, even if it looks \
plausible.
- Correlation stated as causation is overstated. Evidence that two things moved \
together does not establish that one caused the other.
- A claim about a segment, metric, or period the evidence does not cover is \
unsupported, however reasonable it sounds.
- Precision matters: "fell 18%" requires the evidence to show 18%, not "fell". A figure the \
evidence does not show makes a claim **overstated**, not unsupported, when the rest of it holds: \
say which number is wrong in your reason. Removing an otherwise sound finding over one wrong \
figure loses more than it protects.
- If the evidence genuinely establishes the claim as written, say supported. Being \
adversarial does not mean rejecting sound claims.\
"""

OPEN_QUESTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "establishes_cause",
        "cause",
        "cause_date",
        "movement_date",
        "what_the_evidence_shows",
    ],
    "properties": {
        # Asked first on purpose: the fields are generated in order, so the judgement is
        # made before there is a named candidate on the page to justify.
        "establishes_cause": {
            "type": "boolean",
            "description": (
                "True only if these observations establish what caused the movement. "
                "False if they show the movement but not its cause, which is the common "
                "case."
            ),
        },
        "cause": {
            "type": "string",
            "description": (
                "The cause these observations establish, named as specifically as they "
                "allow. Empty string if they establish none."
            ),
        },
        "cause_date": {
            "type": "string",
            "description": (
                "YYYY-MM-DD, the date the named cause happened, taken from the "
                "observations. Empty string if there is no named cause or the "
                "observations do not date it. Never estimate it."
            ),
        },
        "movement_date": {
            "type": "string",
            "description": (
                "YYYY-MM-DD, the date the movement being explained began, taken from the "
                "observations. Empty string if they do not date it."
            ),
        },
        "what_the_evidence_shows": {
            "type": "string",
            "description": (
                "One or two sentences on what these observations do establish, whether or "
                "not that includes a cause."
            ),
        },
    },
}

#: The claim-free half of causal verification. Nothing in this prompt, or in the message
#: that carries it, contains the claim under test or the report it came from.
OPEN_QUESTION_PROMPT = """\
You are reading observations gathered during a data investigation. No analyst's \
conclusion is included below and you must not try to reconstruct one. Answer from the \
observations alone.

Rules:
- Name a cause only if these observations establish it: something has to have changed, \
the observations have to date it, and its date has to be at or before the start of the \
movement it would explain.
- Two series moving together over a window is not a cause. Neither is a change dated \
after the movement began.
- An observation that returned no rows says nobody saw anything, not that nothing \
happened. It cannot establish a cause.
- If these observations do not establish a cause, say so plainly. That is the right \
answer more often than not, and it is more useful than a candidate.
- Dates come from the observations, in YYYY-MM-DD form. Never estimate one.\
"""


@dataclass(frozen=True, slots=True)
class FreshReading:
    """What the cited evidence says when read without the claim it was cited for."""

    establishes_cause: bool
    cause: str = ""
    cause_date: datetime.date | None = None
    movement_date: datetime.date | None = None
    shows: str = ""

    @property
    def postdates_the_movement(self) -> bool:
        """Whether the independently-named cause happened after the movement began.

        The elimination rule of decision 2, one level down. `Hypothesis` already refuses a
        cause dated after its effect, but a claim carries no dates — so a report can state
        an impossible cause in its summary while its hypotheses are clean. Here the dates
        come from a reading that never saw the claim, which is the only way to compare
        them without asking the drafter to grade its own arithmetic.
        """
        if self.cause_date is None or self.movement_date is None:
            return False
        return self.cause_date > self.movement_date


@dataclass(slots=True)
class ClaimVerdict:
    location: str
    claim_text: str
    verdict: Verdict
    reason: str

    @property
    def rejected(self) -> bool:
        return self.verdict is Verdict.UNSUPPORTED


@dataclass(slots=True)
class VerificationResult:
    report: InvestigationReport
    verdicts: list[ClaimVerdict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    #: Claims the verifier could not judge — an LLM failure, not a claim failure.
    unverified: list[str] = field(default_factory=list)
    #: How many causal claims were re-derived in a claim-free context (decision 7).
    #:
    #: Counted because the intervention's cost is exactly this number of extra calls, and
    #: because a value of zero on a report full of causal language means `is_causal_claim`
    #: stopped matching — a silent regression that no verdict would reveal.
    fresh_context_checks: int = 0

    @property
    def rejections(self) -> list[Rejection]:
        """Verifier rejections, in the gate's vocabulary, for the report row."""
        return [
            Rejection(
                location=v.location,
                reason=RejectionReason.NO_SURVIVING_EVIDENCE,
                detail=f"verifier: {v.verdict.value}: {v.reason}",
                text=v.claim_text[:500],
            )
            for v in self.verdicts
            if v.verdict is not Verdict.SUPPORTED
        ]

    @property
    def unsupported_count(self) -> int:
        return sum(1 for v in self.verdicts if v.verdict is Verdict.UNSUPPORTED)

    @property
    def overstated_count(self) -> int:
        return sum(1 for v in self.verdicts if v.verdict is Verdict.OVERSTATED)


# ------------------------------------------------- shared with the sufficiency gate


async def load_cited_evidence(
    session: AsyncSession,
    tenant: TenantContext,
    *,
    investigation_id: uuid.UUID,
    report: InvestigationReport,
) -> dict[uuid.UUID, Evidence]:
    """Load the evidence a report cites, scoped to this tenant and this thread.

    Both predicates matter. An id from elsewhere must not be readable here -- and this
    lookup must not be narrower than the gate's either. It was, briefly: a follow-up citing
    its parent's evidence passed the gate and was then rejected wholesale by the verifier,
    which loaded nothing and could therefore find nothing supported. Two mechanisms checking
    the same claims against different sets of evidence is a bug in whichever is narrower, and
    it fails closed -- so it destroys good reports rather than admitting bad ones, which is
    why it surfaced immediately.

    Module-level, and shared with the sufficiency gate for that same reason: three mechanisms
    now read the cited evidence, and a fourth copy of this query is a fourth chance for one of
    them to be scoped differently from the others.
    """
    cited = report.cited_evidence_ids()
    if not cited:
        return {}
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
    return {row.id: row for row in rows}


def render_evidence(rows: Iterable[Evidence], *, max_chars: int) -> list[str]:
    """Render evidence rows for a model to read, one block each.

    Provenance travels with the payload: which capability produced it, with what
    parameters, when it was observed, and whether it came from a nightly sync rather than a
    live call. A verdict about staleness cannot be reached without those.
    """
    blocks: list[str] = []
    for row in rows:
        payload = json.dumps(row.payload, indent=2, sort_keys=True, default=str)
        truncated = len(payload) > max_chars
        if truncated:
            payload = payload[:max_chars]
        blocks.append(
            f"--- evidence {row.id} ---\n"
            f"tool: {row.tool_name}.{row.capability}\n"
            f"parameters: {json.dumps(row.params, sort_keys=True, default=str)}\n"
            f"observed_at: {row.observed_at.isoformat() if row.observed_at else 'unknown'}\n"
            f"from_nightly_sync: {row.from_cache}\n"
            # Disclosed, so a reader does not treat a cut-off payload as evidence that a
            # field is missing.
            + ("payload (TRUNCATED — judge only what is shown):\n" if truncated else "payload:\n")
            + payload
        )
    return blocks


def _as_date(value: object) -> datetime.date | None:
    """Parse an ISO date, or return None. Never raises.

    A model that answers "mid-June" instead of a date has told us nothing comparable, and
    the arithmetic that depends on it simply does not run — which is the safe direction:
    the claim keeps the verdict the anchored pass gave it.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def _reconcile(verdict: ClaimVerdict, reading: FreshReading) -> None:
    """Reconcile a causal claim's verdict with an independent reading of its evidence.

    In place, so the same `ClaimVerdict` object is what the caller's list and its
    location index both hold. One direction only: this lowers a verdict, never raises one.
    A claim the anchored judge already doubted is not rescued by a second reading that
    happens to name a cause, because the two calls disagree about *what* they were asked
    and the disagreement is not evidence of anything.
    """
    if not reading.establishes_cause:
        verdict.verdict = Verdict.OVERSTATED
        verdict.reason = (
            "asked as an open question, with this claim withheld, the same evidence does "
            f"not establish a cause: {reading.shows}"
        )[:500]
        return
    if reading.postdates_the_movement:
        assert reading.cause_date is not None and reading.movement_date is not None
        days = (reading.cause_date - reading.movement_date).days
        verdict.verdict = Verdict.UNSUPPORTED
        verdict.reason = (
            "read without this claim, the same evidence dates its only candidate cause "
            f"({reading.cause}) to {reading.cause_date.isoformat()}, {days} day(s) after "
            f"the movement began on {reading.movement_date.isoformat()}; a cause cannot "
            "post-date its effect"
        )[:500]


class AdversarialVerifier:
    def __init__(self, llm: LLM, *, max_evidence_chars: int = 6000) -> None:
        self._llm = llm
        # Evidence payloads can be large. Truncating what the verifier reads is
        # safer than skipping verification, but it is disclosed in the prompt so
        # the verifier does not treat an absent field as absent data.
        self._max_evidence_chars = max_evidence_chars

    async def verify(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        *,
        investigation_id: uuid.UUID,
        report: InvestigationReport,
    ) -> VerificationResult:
        evidence = await self._load(session, tenant, investigation_id, report)

        # Every claim is judged against only its own evidence, so the judgements are
        # independent and there is no reason to serialise them. Sequentially, a report
        # with twenty claims made twenty round trips end to end and spent minutes doing
        # it — a large share of an investigation's wall clock, against a 90-second
        # target. Bounded rather than unbounded: firing twenty concurrent requests at a
        # provider is a good way to manufacture the capacity errors that then get
        # reported as unverified claims.
        jobs: list[tuple[str, Claim]] = [
            (f"executive_summary[{index}]", claim)
            for index, claim in enumerate(report.executive_summary)
        ]
        jobs += [
            (f"findings[{f_index}].claims[{c_index}]", claim)
            for f_index, finding in enumerate(report.findings)
            for c_index, claim in enumerate(finding.claims)
        ]

        semaphore = asyncio.Semaphore(_VERDICT_CONCURRENCY)

        async def _judge_one(location: str, claim: Claim) -> tuple[str, ClaimVerdict, Usage, bool]:
            async with semaphore:
                verdict, claim_usage, failed = await self._judge(claim, location, evidence)
            return location, verdict, claim_usage, failed

        judged = await asyncio.gather(*(_judge_one(loc, claim) for loc, claim in jobs))
        by_location = {location: (verdict, failed) for location, verdict, _, failed in judged}

        verdicts: list[ClaimVerdict] = []
        unverified: list[str] = []
        usage = Usage()
        for _, verdict, claim_usage, failed in judged:
            usage = usage + claim_usage
            if failed:
                # The reason travels with the location. Without it the run reports a
                # count and no cause, and the count is what fails the build.
                unverified.append(f"{verdict.location} ({verdict.reason})")
            else:
                verdicts.append(verdict)

        # Decision 7. Only claims that assert a cause, and only those the anchored pass
        # was willing to call `supported`: this check can lower a verdict and never raise
        # one, so anywhere else it would be a paid call whose outcome cannot matter. A
        # report with no causal claim therefore adds no latency at all.
        rederive = [
            (location, verdict)
            for location, (verdict, failed) in by_location.items()
            if not failed
            and verdict.verdict is Verdict.SUPPORTED
            and is_causal_claim(verdict.claim_text)
        ]
        fresh_error: str | None = None
        #: Whether a summary claim was kept only because a second reading disagreed with the
        #: first. Disclosed, because "two readers split on this" is a fact about the answer's
        #: reliability and a reader is entitled to it.
        reconsidered = False
        if rederive:
            claims_by_location = dict(jobs)

            async def _reread(location: str) -> tuple[str, FreshReading | None, Usage, str | None]:
                async with semaphore:
                    reading, reading_usage, error = await self._fresh_reading(
                        claims_by_location[location], report.question, evidence
                    )
                return location, reading, reading_usage, error

            for location, reading, reading_usage, error in await asyncio.gather(
                *(_reread(location) for location, _ in rederive)
            ):
                usage = usage + reading_usage
                if reading is None:
                    # Disclosed rather than treated as a pass: the anchored verdict stands,
                    # and the report says the second reading did not happen. "Nothing was
                    # wrong" and "nothing was checked" are different sentences.
                    fresh_error = error
                    continue
                _reconcile(by_location[location][0], reading)

        # Rebuilt in the report's own order, which the concurrent results do not
        # preserve. A report whose claims came back reordered would still be grounded and
        # would read as though someone had shuffled it.
        summary_kept: list[Claim] = []
        for index, claim in enumerate(report.executive_summary):
            verdict, failed = by_location[f"executive_summary[{index}]"]
            # Kept when unjudged, not dropped: a verifier outage must not silently delete
            # a grounded claim — the report discloses that it went unverified instead.
            if failed or not verdict.rejected:
                summary_kept.append(claim)

        findings_kept: list[Finding] = []
        for f_index, finding in enumerate(report.findings):
            claims_kept: list[Claim] = []
            for c_index, claim in enumerate(finding.claims):
                verdict, failed = by_location[f"findings[{f_index}].claims[{c_index}]"]
                if failed or not verdict.rejected:
                    claims_kept.append(claim)
            if claims_kept:
                findings_kept.append(finding.model_copy(update={"claims": claims_kept}))

        if not summary_kept:
            # One more reading before destroying the whole report.
            #
            # **The severity of a single wrong verdict scales with how short the summary is,
            # and this project asks for short summaries.** `GUIDANCE[Shape.FACTUAL]` says to
            # lead with the answer in one or two sentences, so a factual report often has a
            # one-claim summary -- and then one mistaken `unsupported` is the difference between
            # a delivered answer and no answer at all.
            #
            # Measured: a false-premise attempt was rejected entirely for a summary claim
            # reading "August 2026 shows only 1,884 signups versus 4,849 in July". Both figures
            # are in the cited series, exactly. The verdict was simply wrong, and it cost the
            # whole report on one of three attempts.
            #
            # So the judgement that would empty the summary is taken twice. This is not
            # leniency: the second reading is the same adversarial prompt over the same
            # evidence, and a claim that is genuinely unsupported fails it again. It is the
            # asymmetry that justifies the call -- discarding a sound report is far worse than
            # one extra request -- and the same reasoning the codebase already applies to a
            # verifier *error*, where the claim is kept and the report discloses that it went
            # unchecked. A confident wrong verdict deserves no more trust than a failed one.
            second_readings = await asyncio.gather(
                *(
                    self._judge(claim, f"executive_summary[{index}]", evidence)
                    for index, claim in enumerate(report.executive_summary)
                )
            )
            for index, (verdict, second_usage, failed) in enumerate(second_readings):
                usage = usage + second_usage
                if failed or not verdict.rejected:
                    summary_kept.append(report.executive_summary[index])
                    by_location[f"executive_summary[{index}]"] = (verdict, failed)
                    reconsidered = True

        if not summary_kept:
            # Same rule as the gate: an empty summary is no answer, and presenting
            # one as an answer is the failure both mechanisms exist to prevent.
            from cortex.reports.gate import ReportRejected

            raise ReportRejected(
                "the verifier found no claim in the executive summary to be supported "
                "by its cited evidence, on two independent readings"
            )

        overstated = sum(1 for v in verdicts if v.verdict is Verdict.OVERSTATED)
        verified = report.model_copy(
            update={
                "executive_summary": summary_kept,
                "findings": findings_kept,
                "confidence": _downgrade(report.confidence, verdicts),
                "risks": _add_risks(
                    report, verdicts, unverified, overstated, fresh_error, reconsidered
                ),
            }
        )
        return VerificationResult(
            report=verified,
            verdicts=verdicts,
            usage=usage,
            unverified=unverified,
            fresh_context_checks=len(rederive),
        )

    # ------------------------------------------------------------------ internals

    async def _load(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        investigation_id: uuid.UUID,
        report: InvestigationReport,
    ) -> dict[uuid.UUID, Evidence]:
        return await load_cited_evidence(
            session, tenant, investigation_id=investigation_id, report=report
        )

    async def _judge_call(self, rendered: str) -> tuple[dict[str, Any], Usage]:
        """One verdict, retried once on a transient failure.

        **Why a retry belongs here specifically.** A claim this call cannot judge ships to the
        reader unchecked and counts as a delivered hallucination, which fails the run. Run 30
        lost three claims on one attempt to three consecutive `APITimeoutError` -- no logic
        defect, just a slow minute at the provider, and the suite correctly reported three
        delivered hallucinations for it. Giving up after one attempt makes a transient blip
        indistinguishable from an unverifiable claim, and the two deserve different outcomes.

        One retry, not a policy. The cost is bounded and paid only on failure, and it is worth
        paying against a dimension that must read zero: a second full deadline on a rare timeout
        is cheaper than a build that fails for a reason nobody can act on.

        An authentication failure is never retried -- the credential will reject the next call
        too, and spending another deadline to learn that is waste.
        """
        last: LLMError | None = None
        for attempt in range(2):
            try:
                return await self._llm.structured(
                    system=SYSTEM_PROMPT,
                    messages=[Message(role="user", content=rendered)],
                    schema=VERDICT_SCHEMA,
                    max_tokens=1024,
                )
            except LLMAuthenticationFailed:
                raise
            except LLMError as exc:
                last = exc
                if attempt == 0:
                    # Short and fixed. With one retry there is no herd to spread out, and the
                    # delay has to stay small against a deadline the caller already waited on.
                    await asyncio.sleep(1.0)
        assert last is not None  # the loop either returns or records an error
        raise last

    async def _judge(
        self, claim: Claim, location: str, evidence: dict[uuid.UUID, Evidence]
    ) -> tuple[ClaimVerdict, Usage, bool]:
        rendered = self._render(claim, evidence)
        if rendered is None:
            # The gate should have removed this already; reaching here means the
            # claim cites nothing resolvable, which is unsupported by definition.
            return (
                ClaimVerdict(
                    location=location,
                    claim_text=claim.text,
                    verdict=Verdict.UNSUPPORTED,
                    reason="none of the cited evidence could be loaded",
                ),
                Usage(),
                False,
            )

        try:
            payload, usage = await self._judge_call(rendered)
        except LLMError as exc:
            return (
                ClaimVerdict(
                    location=location,
                    claim_text=claim.text,
                    verdict=Verdict.SUPPORTED,
                    # The cause is carried, not swallowed. An unverified claim ships to
                    # the reader unchecked and now fails the eval, so "why could this not
                    # be verified" is the diagnostic that matters most — and a bare
                    # "not verified" sends whoever reads it back to reproduce the run.
                    reason=f"not verified: {type(exc).__name__}: {exc}"[:300],
                ),
                Usage(),
                True,
            )

        try:
            verdict = Verdict(payload["verdict"])
        except (KeyError, ValueError):
            return (
                ClaimVerdict(
                    location=location,
                    claim_text=claim.text,
                    verdict=Verdict.SUPPORTED,
                    reason=(
                        "not verified: the verdict field was missing or unrecognised "
                        f"(got {payload.get('verdict')!r})"
                    )[:300],
                ),
                usage,
                True,
            )

        return (
            ClaimVerdict(
                location=location,
                claim_text=claim.text,
                verdict=verdict,
                reason=str(payload.get("reason", ""))[:500],
            ),
            usage,
            False,
        )

    async def _fresh_reading(
        self, claim: Claim, question: str, evidence: dict[uuid.UUID, Evidence]
    ) -> tuple[FreshReading | None, Usage, str | None]:
        """Read a causal claim's own evidence in a context that does not contain the claim.

        The independence properties of the anchored pass are kept and one is added. Same
        cited evidence and nothing else; no access to the drafter's reasoning; one claim per
        call — and now, no access to the claim either. The only thing carried over is the
        *question the user asked*, because sufficiency and this open question are both
        properties of (question, evidence) and an open reading with nothing to be open about
        would just describe rows. Where the user's own question names a candidate cause
        ("is conversion down because of the deploy?") that candidate came from the user and
        not from the drafter, which is a different and much weaker anchor than showing the
        model its own conclusion.

        Never raises. A second reading that cannot be obtained leaves the anchored verdict
        standing and is disclosed as a risk; failing the report because an extra check was
        unavailable would make this intervention strictly worse than not shipping it.
        """
        blocks = render_evidence(
            (row for row in (evidence.get(eid) for eid in claim.evidence_ids) if row is not None),
            max_chars=self._max_evidence_chars,
        )
        if not blocks:
            return None, Usage(), "none of the cited evidence could be loaded"

        rendered = (
            f"OBSERVATIONS ({len(blocks)} item(s)):\n" + "\n\n".join(blocks) + "\n\n"
            f"The investigation was asked: {question}\n\n"
            "From these observations alone: do they establish what caused the movement this "
            "question is about? If they do, name it, date it, and date the movement it "
            "would explain."
        )
        try:
            payload, usage = await self._llm.structured(
                system=OPEN_QUESTION_PROMPT,
                messages=[Message(role="user", content=rendered)],
                schema=OPEN_QUESTION_SCHEMA,
                max_tokens=1024,
            )
        except LLMError as exc:
            return None, Usage(), f"{type(exc).__name__}: {exc}"[:300]

        if "establishes_cause" not in payload:
            # A payload of the wrong shape is not a reading. Treated as an outage rather
            # than as "no cause established", because the second answer would silently
            # downgrade every causal claim in the report on the strength of a parse error.
            return None, usage, "the open reading came back without a verdict field"

        return (
            FreshReading(
                establishes_cause=bool(payload.get("establishes_cause")),
                cause=str(payload.get("cause") or "")[:300],
                cause_date=_as_date(payload.get("cause_date")),
                movement_date=_as_date(payload.get("movement_date")),
                shows=str(payload.get("what_the_evidence_shows") or "")[:300],
            ),
            usage,
            None,
        )

    def _render(self, claim: Claim, evidence: dict[uuid.UUID, Evidence]) -> str | None:
        blocks = render_evidence(
            (row for row in (evidence.get(eid) for eid in claim.evidence_ids) if row is not None),
            max_chars=self._max_evidence_chars,
        )
        if not blocks:
            return None

        return (
            f"CLAIM:\n{claim.text}\n\n"
            f"CITED EVIDENCE ({len(blocks)} item(s)):\n" + "\n\n".join(blocks) + "\n\n"
            "Does the cited evidence establish this claim exactly as written?"
        )


def _downgrade(stated: Confidence, verdicts: list[ClaimVerdict]) -> Confidence:
    """Lower confidence when the verifier found problems.

    An unsupported claim is evidence the drafter's judgement was off across the
    report, not only in the sentence that was removed.
    """
    if any(v.verdict is Verdict.UNSUPPORTED for v in verdicts):
        return Confidence.LOW
    if any(v.verdict is Verdict.OVERSTATED for v in verdicts):
        if stated is Confidence.HIGH:
            return Confidence.MEDIUM
        if stated is Confidence.MEDIUM:
            return Confidence.LOW
    return stated


def _add_risks(
    report: InvestigationReport,
    verdicts: list[ClaimVerdict],
    unverified: list[str],
    overstated: int,
    fresh_error: str | None = None,
    reconsidered: bool = False,
) -> list[Any]:
    """Disclose verification outcomes as risks the reader can see.

    A report that quietly dropped claims would look cleaner than it is, so what
    the verifier did is stated rather than only logged.
    """
    from cortex.reports.schema import Risk

    risks = list(report.risks)
    if reconsidered:
        risks.append(
            Risk(
                description=(
                    "Every claim in this summary was judged unsupported on a first reading and "
                    "kept on a second. Two independent readings of the same evidence "
                    "disagreed, so treat the summary as less settled than its wording suggests "
                    "-- and the alternative was delivering no answer at all."
                )
            )
        )
    removed = sum(1 for v in verdicts if v.verdict is Verdict.UNSUPPORTED)
    if removed:
        risks.append(
            Risk(
                description=(
                    f"{removed} claim(s) were removed because an independent check found "
                    "the cited evidence did not support them. Treat the remaining "
                    "conclusions with corresponding caution."
                )
            )
        )
    if overstated:
        risks.append(
            Risk(
                description=(
                    f"{overstated} claim(s) go beyond their evidence -- overstated, or "
                    "carrying a figure the evidence does not show -- and are retained with "
                    "reduced confidence rather than removed, because the rest of each one "
                    "holds."
                )
            )
        )
    if unverified:
        risks.append(
            Risk(
                description=(
                    f"{len(unverified)} claim(s) could not be independently verified "
                    "because the verification step failed. They are grounded in real "
                    "evidence but their support was not double-checked."
                )
            )
        )
    if fresh_error:
        # Absent unless a causal claim was actually re-read and the re-reading failed,
        # which keeps it off the reports that never needed it. A caveat that appears
        # everywhere is a caveat nobody reads on the report where it mattered.
        risks.append(
            Risk(
                description=(
                    "A causal claim in this report was not re-derived from its evidence "
                    f"independently of the claim itself ({fresh_error}). Its support was "
                    "checked only in the weaker form, against the claim as written."
                )
            )
        )
    return risks
