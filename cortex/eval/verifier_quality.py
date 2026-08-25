"""Measuring how often the verifier deletes a claim that was actually supported.

## The hole this closes

`AdversarialVerifier` re-reads every claim against only its own cited evidence and **deletes** the
ones it judges unsupported. The eval reports `draft_reliability` — the share of drafted claims that
survived — and has been reading between 0.62 and 1.00.

That number has been unreadable, and dangerously so. A `draft_reliability` of 0.62 means 38% of
claims were removed, and nothing distinguished:

  - the drafter overreaching, which is the reading assumed so far, and the defences working; from
  - the verifier being **wrong**, deleting true, properly-cited claims — which silently makes
    reports worse while every dashboard says the grounding machinery is doing its job.

A false-positive verifier is the more expensive failure, because it is invisible. A removed claim
leaves no trace in the delivered report, the run still passes, and the only symptom is an answer
that is quietly less complete than the evidence supported.

## How this measures it without human labelling

The trick is to construct claims whose truth is not a matter of judgement. For each real evidence
row, a claim is generated **mechanically from that row's own payload** — "posthog.event_trend
returned 14 rows", "the payload's `count` field is 82" — so the claim is true *by construction*, is
cited to exactly the row it came from, and needs no annotator to adjudicate.

The verifier is then asked to judge them. Any `UNSUPPORTED` verdict on such a claim is an
**over-rejection**, counted mechanically. The rate is `over_rejections / claims_judged`.

## A naming correction, because the first version of this file used the wrong word

This was originally called a "false-positive rate". That is the opposite polarity from the one the
attribution literature means. In *Do You Need a Frontier Model as a Citation Verifier?*
(arXiv 2607.08700), FPR is defined as FP/(FP+TN) — **a bad citation accepted** — and the error
measured here, an `unsupported` verdict on a genuinely supported claim, is their **false negative**.

The distinction is not pedantry, because the two polarities have opposite difficulty. That paper
finds adversarial-edit detection "uniformly high" at 86–100% (their FPR is easy), while
false-negative rates across eight judges span **0.183 to 0.470**, and it names over-rejection of
genuinely supported citations as the dominant failure mode. So this file measures the hard
direction and used to report it under the name of the easy one — which would have been quoted
internally as reassurance about a number nobody had measured.

The polarity that *is* still unmeasured here: whether the verifier accepts a claim its evidence
does not support. Nothing in this module tests that, and it is the one that maps to the product's
"never hallucinate" promise.

## What this does and does not prove

It measures the verifier on the **easy** end of the distribution: restatements of an observation,
where entailment is nearly syntactic. A real claim is an interpretation — "signups fell *because*
of the deploy" — and this says nothing about those, where the verifier's judgement is actually
being exercised.

That asymmetry is the point rather than a weakness. A verifier that removes a claim which merely
*restates* its own evidence is broken in a way no amount of prompt tuning excuses, and until now we
could not have detected it. A low false-positive rate here is a floor, not a clean bill of health,
and the docstring says so because the number will be quoted.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import Evidence
from cortex.reports.schema import Claim, Confidence, InvestigationReport
from cortex.reports.verifier import AdversarialVerifier
from cortex.tenancy.context import TenantContext

#: Claims generated per evidence row, at most. Two: one about the shape of the result and one
#: about a specific value in it. More would repeat the same syntactic pattern and inflate the
#: denominator without testing anything new.
MAX_CLAIMS_PER_ROW = 2

#: Rows sampled. Bounded because each claim costs a verifier call, and the point is a rate rather
#: than an exhaustive audit.
MAX_ROWS = 12


@dataclass(slots=True)
class OverRejectionReport:
    """What the verifier did to claims that were true by construction."""

    claims_judged: int = 0
    over_rejections: int = 0
    #: The claim text and the verifier's stated reason, for the ones it removed. A rate with no
    #: examples is impossible to act on: the reason is what says whether the verifier misread the
    #: payload or the claim generator wrote something ambiguous.
    examples: list[tuple[str, str]] = field(default_factory=list)
    #: Claims the verifier could not judge at all — a provider failure rather than a verdict.
    #: Excluded from the rate, because an unjudged claim is kept by the verifier and so is not a
    #: false positive.
    unjudged: int = 0

    @property
    def rate(self) -> float:
        return self.over_rejections / self.claims_judged if self.claims_judged else 0.0

    def render(self) -> str:
        if not self.claims_judged:
            return "verifier over-rejection rate: no claims could be generated"
        lines = [
            f"verifier over-rejection rate: {self.rate:.0%} "
            f"({self.over_rejections}/{self.claims_judged} true claims removed"
            + (f", {self.unjudged} unjudged)" if self.unjudged else ")")
        ]
        for text, reason in self.examples[:5]:
            lines.append(f"  removed: {text}")
            lines.append(f"    reason: {reason}")
        return "\n".join(lines)


#: How a generated claim relates to the evidence it cites. Recorded per claim, never shown to the
#: verifier, and reported per stratum rather than aggregated.
#:
#: This is NEI-CAP's construction variable (arXiv 2605.26663), and its finding is why a single
#: aggregate number here would be actively misleading: a verifier trained or prompted for one
#: construction scored **NEI-F1 1.000 on matched evidence and 0.000 on a near-miss retrieval**. Its
#: conclusion is blunt -- *"matched NEI performance does not imply transfer."* Our first 0/24 was
#: entirely the matched cell.
class Construction(enum.StrEnum):
    #: A restatement of something the payload literally contains. Should be SUPPORTED.
    RESTATEMENT = "restatement"
    #: The number or direction mutated while the citation is left alone. Should be REJECTED.
    #: P7's counterfactual negative (arXiv 2604.09537): *"semantically meaningful and often close
    #: to the correct evidence, but it supports the wrong state."*
    COUNTERFACTUAL = "counterfactual"
    #: A true claim repointed at an unrelated row's payload. Should be REJECTED.
    #: P7's swap control drops a trained verifier from AUROC 97.43 to 55.62 -- so a verifier that
    #: still says supported here is not reading its evidence at all.
    SWAPPED = "swapped"


@dataclass(slots=True)
class StratumResult:
    """What the verifier did to one construction."""

    construction: Construction
    #: What a correct verifier would do with every claim in this stratum.
    should_reject: bool
    judged: int = 0
    #: Verdicts that went the wrong way: over-rejections in RESTATEMENT, wrong acceptances in the
    #: two negative strata.
    errors: int = 0
    unjudged: int = 0
    examples: list[tuple[str, str]] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.errors / self.judged if self.judged else 0.0

    @property
    def error_name(self) -> str:
        # Named per stratum, because the two directions are different failures with opposite
        # difficulty -- and conflating them under one word is the mistake this module already made
        # once. See the naming correction above.
        return "wrongly accepted" if self.should_reject else "wrongly rejected"


def counterfactual_claims(row: Evidence) -> list[str]:
    """Claims that are FALSE by construction, cited to the row they misdescribe.

    A correct verifier must reject every one. This measures the polarity the restatement stratum
    cannot: whether the verifier *accepts* a claim its own cited evidence contradicts — which is
    the direction that maps to "never hallucinate".

    The mutations are deliberately small and plausible, following P7's counterfactual design: a
    changed magnitude, a flipped direction, a count that is wrong but of the right order. A wildly
    wrong claim would be rejected by a verifier that was barely reading.
    """
    payload = row.payload if isinstance(row.payload, dict) else {}
    claims: list[str] = []

    for key, value in payload.items():
        if isinstance(value, list) and value:
            # A count that is wrong but adjacent, so rejecting it requires actually counting.
            wrong = len(value) + 7
            claims.append(
                f"The {row.tool_name}.{row.capability} observation returned "
                f"{wrong} item(s) under '{key}'."
            )
            break

    for key, value in payload.items():
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int | float) and value not in (0, 0.0):
            # Order-of-magnitude wrong, same sign: plausible enough to need a real comparison.
            claims.append(
                f"In the {row.tool_name}.{row.capability} observation, '{key}' is "
                f"{type(value)(value * 10) if isinstance(value, int) else value * 10}."
            )
            break

    return claims[:MAX_CLAIMS_PER_ROW]


def claims_from_payload(row: Evidence) -> list[str]:
    """Claims that are true by construction about one evidence row.

    Each is a restatement of something the payload literally contains, so no annotator is needed
    to decide whether it is supported. Deliberately dull: the moment a generated claim requires
    interpretation, a removal stops being unambiguously a false positive.
    """
    payload = row.payload if isinstance(row.payload, dict) else {}
    claims: list[str] = []

    # 1. The shape of the result. True of every payload, and the weakest possible claim -- if the
    #    verifier rejects this, something is badly wrong.
    for key, value in payload.items():
        if isinstance(value, list):
            claims.append(
                f"The {row.tool_name}.{row.capability} observation returned "
                f"{len(value)} item(s) under '{key}'."
            )
            break

    # 2. One scalar, quoted exactly. A number or short string copied verbatim from the payload.
    for key, value in payload.items():
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int | float):
            claims.append(
                f"In the {row.tool_name}.{row.capability} observation, '{key}' is {value}."
            )
            break
        if isinstance(value, str) and 0 < len(value) <= 60:
            claims.append(
                f"In the {row.tool_name}.{row.capability} observation, '{key}' is '{value}'."
            )
            break

    return claims[:MAX_CLAIMS_PER_ROW]


async def measure_over_rejection(
    session: AsyncSession,
    tenant: TenantContext,
    *,
    investigation_id: uuid.UUID,
    verifier: AdversarialVerifier,
    question: str = "Measurement of the verifier itself.",
) -> OverRejectionReport:
    """The restatement stratum alone. Kept because it is the calibration floor.

    A verifier that removes a claim merely restating its own evidence is broken beyond excuse, so
    this stratum should read 0 and is worth watching on its own.
    """
    strata = await measure_strata(
        session,
        tenant,
        investigation_id=investigation_id,
        verifier=verifier,
        question=question,
        constructions=(Construction.RESTATEMENT,),
    )
    result = strata.get(Construction.RESTATEMENT)
    outcome = OverRejectionReport()
    if result is None:
        return outcome
    outcome.claims_judged = result.judged
    outcome.over_rejections = result.errors
    outcome.unjudged = result.unjudged
    outcome.examples = list(result.examples)
    return outcome


async def measure_strata(
    session: AsyncSession,
    tenant: TenantContext,
    *,
    investigation_id: uuid.UUID,
    verifier: AdversarialVerifier,
    question: str = "Measurement of the verifier itself.",
    constructions: tuple[Construction, ...] = tuple(Construction),
) -> dict[Construction, StratumResult]:
    """Judge each construction separately and report per stratum, never aggregated.

    Separate verifier passes per stratum, deliberately: a single report mixing true and false
    claims would let the verifier calibrate against the mixture — if it notices half the claims are
    wrong it can reject more freely — and the rate would measure the mixture rather than the
    verifier.
    """
    rows = (
        (
            await session.execute(
                select(Evidence)
                .where(
                    Evidence.investigation_id == investigation_id,
                    Evidence.tenant_id == tenant.tenant_id,
                )
                .order_by(Evidence.observed_at.asc())
                .limit(MAX_ROWS)
            )
        )
        .scalars()
        .all()
    )
    if len(rows) < 2:
        # The swapped stratum needs a second row to point at. Below that, nothing is measurable.
        rows = list(rows)

    results: dict[Construction, StratumResult] = {}
    for construction in constructions:
        claims = _claims_for(construction, rows)
        should_reject = construction is not Construction.RESTATEMENT
        result = StratumResult(construction=construction, should_reject=should_reject)
        if not claims:
            results[construction] = result
            continue

        # The negative strata need a genuinely supported claim in the executive summary, because
        # the verifier rejects a whole report whose summary is entirely unsupported -- correct
        # product behaviour that would otherwise hide the per-claim verdicts this reads. The anchor
        # is excluded from the count below, and doubles as a per-run control: if the anchor is
        # rejected, the restatement stratum is broken and the negative numbers mean nothing.
        anchor: Claim | None = None
        if should_reject:
            anchor = next(
                (
                    Claim(text=text, evidence_ids=[row.id])
                    for row in rows
                    for text in claims_from_payload(row)[:1]
                ),
                None,
            )
            if anchor is None:
                results[construction] = result
                continue

        report = _wrap(question, claims, anchor=anchor)
        try:
            verification = await verifier.verify(
                session, tenant, investigation_id=investigation_id, report=report
            )
        except Exception as exc:  # noqa: BLE001 - a rejected report is a measurable outcome
            # Reached when even the anchor was judged unsupported, so the whole report was
            # rejected. Recorded rather than raised: it says the restatement stratum is failing,
            # which is information, and it must not abort the other strata.
            result.examples.append(("(whole report rejected)", str(exc)[:200]))
            results[construction] = result
            continue
        judged = {verdict.claim_text: verdict for verdict in verification.verdicts}
        result.unjudged = max(0, len(claims) - len(judged))
        for claim in claims:
            verdict = judged.get(claim.text)
            if verdict is None:
                continue
            result.judged += 1
            # An error is a verdict that went the wrong way for this stratum. `rejected` is True
            # only for UNSUPPORTED, so an `overstated` verdict on a counterfactual counts as
            # accepted -- deliberately strict, because a downgraded-confidence false claim is
            # still a false claim in the delivered report.
            wrong = (not verdict.rejected) if should_reject else verdict.rejected
            if wrong:
                result.errors += 1
                result.examples.append((claim.text, verdict.reason))
        results[construction] = result
    return results


def _claims_for(construction: Construction, rows: list[Evidence]) -> list[Claim]:
    """The claims for one stratum, each cited according to its construction."""
    claims: list[Claim] = []
    if construction is Construction.RESTATEMENT:
        for row in rows:
            claims += [Claim(text=text, evidence_ids=[row.id]) for text in claims_from_payload(row)]
    elif construction is Construction.COUNTERFACTUAL:
        for row in rows:
            claims += [
                Claim(text=text, evidence_ids=[row.id]) for text in counterfactual_claims(row)
            ]
    elif construction is Construction.SWAPPED:
        # A true claim about one row, cited to a different row. Paired so the citation is always a
        # real, resolvable, hash-matching evidence id -- the gate cannot catch this, which is the
        # point: only the verifier stands between a swapped citation and the reader.
        for index, row in enumerate(rows):
            other = rows[(index + 1) % len(rows)] if len(rows) > 1 else None
            if other is None or other.id == row.id:
                continue
            for text in claims_from_payload(row)[:1]:
                claims.append(Claim(text=text, evidence_ids=[other.id]))
    return claims


def _wrap(
    question: str, claims: list[Claim], *, anchor: Claim | None = None
) -> InvestigationReport:
    """One report carrying the stratum's claims.

    `anchor` is a genuinely supported claim placed in the executive summary for the negative
    strata. Without it the verifier rejects the whole report -- correct behaviour for a report whose
    every summary claim is unsupported, and fatal to a measurement that needs to read the individual
    verdicts. The anchor is not counted.
    """
    summary = [anchor] if anchor is not None else [claims[0]]
    body = claims if anchor is not None else (claims[1:] or [claims[0]])
    return InvestigationReport(
        question=question,
        executive_summary=summary,
        findings=[
            {  # type: ignore[list-item]
                "title": "Generated claims under test",
                "claims": body,
                "confidence": Confidence.HIGH,
            }
        ],
        confidence=Confidence.HIGH,
    )


def render_strata(results: dict[Construction, StratumResult]) -> str:
    """Per stratum, with the two error directions named separately.

    Never a single aggregate. NEI-CAP's finding is that a matched-construction score carries no
    information about the others: 1.000 on the matched cell, 0.000 on a near miss.
    """
    lines = ["verifier quality by construction"]
    for construction in Construction:
        result = results.get(construction)
        if result is None:
            continue
        if not result.judged:
            lines.append(f"  {construction.value:<15} no claims generated")
            continue
        lines.append(
            f"  {construction.value:<15} {result.rate:>4.0%} {result.error_name} "
            f"({result.errors}/{result.judged}"
            + (f", {result.unjudged} unjudged)" if result.unjudged else ")")
        )
        for text, reason in result.examples[:2]:
            lines.append(f"      {text}")
            lines.append(f"        verifier said: {reason}")
    return "\n".join(lines)


def as_dict(outcome: OverRejectionReport) -> dict[str, Any]:
    """For a scorecard or a log line."""
    return {
        "claims_judged": outcome.claims_judged,
        "over_rejections": outcome.over_rejections,
        "unjudged": outcome.unjudged,
        "rate": round(outcome.rate, 4),
    }
