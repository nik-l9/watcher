"""A cause needs two independent lines of evidence, or the report does not narrow to it.

ADR 0005 decision 3, from DOE-NE-STD-1004-92 and found in none of the other seven frameworks
surveyed. Its Events-and-Causal-Factor rule: every cause node carries two independent lines
of evidence, each named, and if only
one is available **the tree does not narrow** -- "all possible causes should be evaluated as
potential causes". Insufficient evidence does not license a pick; it licenses continued breadth.

**The failure behaviour is the whole rule, and it is the opposite of what a model does under
uncertainty.** An LLM given one thin line narrows anyway and hedges in the prose -- "likely",
"appears to", "suggests" -- which reads as a cause to every reader who skims. So this acts on the
verdict rather than adding a caveat: a supported hypothesis resting on one line becomes
inconclusive, and the report says why. The same source's third rule is honoured too, that "the
bases for rejected and accepted causes should be stated", so the lines it *does* have are named
rather than merely counted.

## What counts as independent, and why this is the weak decision of the three

Independence is by **tool**, not by evidence row. Two PostHog trends are one line looked at twice;
a GitHub deploy record beside a PostHog annotation is two. That is a proxy and the honest
description of it is a floor rather than a definition -- real evidential independence is a causal
property, and `docs/rollout-plan.md` says plainly why we cannot reach it here: in a
single-warehouse architecture two ways of knowing something are usually two views of one source,
and with `J = 0` control series there is no unaffected second series to be the other line.

So this catches the thinnest claims -- a causal assertion resting entirely on one connector -- and
does not pretend to establish independence for the ones it passes. It fires on cardinality before
any content reasoning, which makes it the most robust of the checks and the least informative: it
says "not proven", never "impossible".
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from cortex.reports.schema import Hypothesis, InvestigationReport, Risk, Verdict

__all__ = ["Corroboration", "REQUIRED_LINES", "corroborate"]

#: Independent lines a cause needs before a report may narrow to it.
#:
#: Two, from the source. Not three: DOE's own rule is two, and inventing a stricter threshold
#: would refuse nearly everything while attributing the refusal to a standard nobody set.
REQUIRED_LINES = 2


@dataclass(frozen=True, slots=True)
class Corroboration:
    """The result of applying the rule to a report."""

    report: InvestigationReport
    #: Statements that were downgraded, with the lines they had.
    broadened: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def narrowed(self) -> bool:
        """Whether anything was left standing as a supported cause."""
        return any(
            hypothesis.verdict is Verdict.SUPPORTED and hypothesis.cause_at is not None
            for hypothesis in self.report.hypotheses
        )


def lines_for(hypothesis: Hypothesis, tool_of: dict[uuid.UUID, str]) -> tuple[str, ...]:
    """The distinct tools whose evidence supports this hypothesis, named.

    Supporting evidence only. Contradicting evidence is what killed a *different* candidate and
    counting it here would let a report narrow on the strength of having ruled something else
    out -- which is an argument for breadth, not against it.
    """
    return tuple(
        sorted({tool_of[eid] for eid in hypothesis.supporting_evidence_ids if eid in tool_of})
    )


def corroborate(report: InvestigationReport, tool_of: dict[uuid.UUID, str]) -> Corroboration:
    """Downgrade any dated cause the report narrows to on fewer than two independent lines.

    Only dated, supported hypotheses are considered -- the same reading the elimination rule and
    the identifiability disclosure use. An undated hypothesis has named no intervention, and an
    inconclusive one has already declined to narrow.
    """
    broadened: list[tuple[str, tuple[str, ...]]] = []
    updated: list[Hypothesis] = []
    for hypothesis in report.hypotheses:
        if hypothesis.verdict is not Verdict.SUPPORTED or hypothesis.cause_at is None:
            updated.append(hypothesis)
            continue
        lines = lines_for(hypothesis, tool_of)
        if len(lines) >= REQUIRED_LINES:
            updated.append(hypothesis)
            continue
        broadened.append((hypothesis.statement, lines))
        updated.append(
            hypothesis.model_copy(
                update={
                    "verdict": Verdict.INCONCLUSIVE,
                    "reasoning": _reasoning(hypothesis, lines),
                }
            )
        )

    if not broadened:
        return Corroboration(report=report)

    return Corroboration(
        report=report.model_copy(
            update={
                "hypotheses": updated,
                "risks": [*report.risks, _risk(broadened)],
            }
        ),
        broadened=tuple(broadened),
    )


def _reasoning(hypothesis: Hypothesis, lines: tuple[str, ...]) -> str:
    """Why this stopped being supported, appended to whatever the analyst wrote.

    Appended rather than replacing it: the analyst's own reasoning is what a reader needs to
    judge the downgrade, and overwriting it would hide the argument being overruled.
    """
    named = ", ".join(lines) if lines else "none that resolve"
    existing = f"{hypothesis.reasoning} " if hypothesis.reasoning else ""
    return (
        f"{existing}Recorded as inconclusive rather than supported: a cause needs two "
        f"independent lines of evidence and this rests on {len(lines)} ({named}). "
        "Insufficient evidence does not license a pick."
    )[:2000]


def _risk(broadened: list[tuple[str, tuple[str, ...]]]) -> Risk:
    """One risk covering every downgrade, naming the lines each had."""
    parts = [
        f"{len(broadened)} candidate cause(s) were recorded as inconclusive rather than "
        "supported, because a cause needs two independent lines of evidence and these rest on "
        "fewer. The report deliberately does not narrow to them:"
    ]
    for statement, lines in broadened:
        named = ", ".join(lines) if lines else "no resolvable source"
        parts.append(f" - '{statement[:120]}' ({len(lines)}: {named})")
    parts.append(
        "Independence is counted by source here, which is a floor rather than a definition -- "
        "two readings from one connector are one line, and genuine evidential independence "
        "needs a comparison series this estate does not yet have."
    )
    return Risk(description=" ".join(parts))
