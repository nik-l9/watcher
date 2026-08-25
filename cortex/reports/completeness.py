"""Did the report answer the question that was asked?

Nothing checked this. The loop stops when the model emits no tool call — its own judgement
that it is finished — and everything downstream verifies *what the report says* rather than
*whether it said enough*. A report can be perfectly grounded, survive the citation gate and
the adversarial verifier, and answer less than was asked. The eval catches that on fixtures
through `required_signals`; production had no equivalent.

Adapted from OpenHands' `conversation/goal/judge.py`, found by indexing their SDK. Their
judge is a pure `objective + transcript -> verdict` evaluator returning a score, a `complete`
flag and a description of what remains, and their prompt is the instructive part:

> Derive the concrete requirements implied by the objective. For EACH requirement, look for
> authoritative evidence in the transcript. Treat missing, uncertain, or
> merely-claimed-but-unverified evidence as NOT satisfied.

Their authoritative evidence is file contents and test output. Ours is stronger: a claim is
only in the delivered report if its citation resolved to an immutable, hashed evidence row. So
this judge reads the *delivered* report rather than the transcript — what survived, not what
was drafted.

**It discloses; it never edits.** This is an LLM's opinion, and the rule in this codebase is
that grounding and accuracy are decided mechanically. So an incomplete verdict adds a caveat a
reader can weigh and removes nothing. Two consequences follow deliberately:

  - A judge that goes wrong costs a spurious caveat, not a lost finding.
  - A judge that cannot run at all is not fatal. `assess` returns `assessed=False` rather
    than raising, and an unassessed report says so instead of implying it passed — the same
    distinction between "nothing was wrong" and "nothing was checked" that F-24 was about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cortex.agents.llm import LLM, LLMError, Message
from cortex.reports.schema import InvestigationReport, Risk

#: The judge is asked for a short answer about a long report, so it needs little room.
_MAX_TOKENS = 1024

#: Below this, the question is treated as unanswered rather than partially answered. A
#: single number is a blunt instrument, which is why the *list* of unaddressed parts is what
#: reaches the reader and the score only decides whether to say anything at all.
_COMPLETE_AT = 0.75

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["score", "complete", "unaddressed"],
    "properties": {
        "score": {
            "type": "number",
            "description": (
                "Probability from 0 to 1 that every part of the question is answered by a "
                "cited claim in the report."
            ),
        },
        "complete": {
            "type": "boolean",
            "description": "Whether every part of the question is answered.",
        },
        "unaddressed": {
            "type": "array",
            "description": (
                "Each part of the question the report does not answer, phrased as the "
                "reader would ask it. Empty when the report is complete. Do not list "
                "things the report answers with low confidence — a hedged answer is an "
                "answer."
            ),
            "items": {"type": "string"},
        },
    },
}

_SYSTEM = (
    "You audit whether an analyst's report answers the question it was given. You judge "
    "coverage only: whether each part of the question is addressed by a claim in the "
    "report.\n\n"
    "Derive the concrete sub-questions the question implies, then check each against the "
    "report's claims.\n\n"
    "Rules that decide most cases:\n"
    "  - A part answered with low confidence, or answered as 'the data cannot establish "
    "this', IS answered. Declining to invent a cause is a correct answer, not a gap.\n"
    "  - A part the report does not mention at all is unanswered.\n"
    "  - Do not judge whether the answer is *correct* — you cannot see the underlying "
    "data. Judge only whether the question was addressed.\n"
    "  - Do not ask for more than the question did. A question about one metric is not "
    "incomplete for lacking analysis of another."
)


@dataclass(frozen=True, slots=True)
class AnswerAssessment:
    """Whether the delivered report covers the question."""

    #: False when the judge could not run. `complete` is then meaningless, and a caller must
    #: not read it as a pass.
    assessed: bool
    complete: bool = True
    score: float = 1.0
    unaddressed: list[str] = field(default_factory=list)
    #: Why it could not be assessed, when it could not.
    error: str | None = None

    @property
    def risks(self) -> list[Risk]:
        """The caveats this assessment adds to a report.

        Attached as risks rather than data-quality notes because the subject is the *answer*,
        not the data: a data-quality note says what the numbers cannot support, and this says
        what the report did not get to. Both are things a reader needs, and conflating them
        would bury one in the other.
        """
        if not self.assessed:
            # Silence here would be the worst option: a reader would take the absence of a
            # caveat as coverage. Cheap to state, and honest.
            return [
                Risk(
                    description=(
                        "Whether this report addresses every part of the question was not "
                        f"checked ({self.error or 'the check did not run'}). Read the "
                        "question and the answer side by side."
                    )
                )
            ]
        if self.complete:
            return []
        parts = "; ".join(self.unaddressed[:4])
        return [
            Risk(
                description=(
                    f"The question also asked about {parts} — this report does not address "
                    f"that. What it does say is grounded; it is not the whole answer."
                )
            )
        ]


class CompletenessJudge:
    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    async def assess(self, question: str, report: InvestigationReport) -> AnswerAssessment:
        """Judge whether the report covers the question.

        Never raises. A judge that could break an investigation would be a worse trade than
        one that occasionally declines to answer, given it can only ever add a caveat.
        """
        claims = _claims(report)
        if not claims:
            # Nothing to judge. The loop already refuses to draft a report with no evidence,
            # so this is unreachable in production and cheap to be correct about.
            return AnswerAssessment(assessed=False, error="the report contains no claims")

        try:
            payload, _usage = await self._llm.structured(
                system=_SYSTEM,
                messages=[
                    Message(
                        role="user",
                        content=(
                            f"QUESTION AS ASKED:\n{question}\n\nWHAT THE REPORT CLAIMS:\n{claims}"[
                                :20000
                            ]
                        ),
                    )
                ],
                schema=_SCHEMA,
                max_tokens=_MAX_TOKENS,
            )
        except LLMError as exc:
            return AnswerAssessment(assessed=False, error=f"{type(exc).__name__}: {exc}")

        score = _as_score(payload.get("score"))
        unaddressed = [
            str(item).strip() for item in payload.get("unaddressed") or [] if str(item).strip()
        ]
        # The list decides, not the flag. A judge that returns `complete: false` with nothing
        # unaddressed has told a reader nothing actionable, and a caveat with no content is
        # noise that teaches people to skip the section.
        complete = bool(payload.get("complete")) or not unaddressed
        if score < _COMPLETE_AT and unaddressed:
            complete = False

        return AnswerAssessment(
            assessed=True,
            complete=complete,
            score=score,
            unaddressed=unaddressed if not complete else [],
        )


def _claims(report: InvestigationReport) -> str:
    """The report's assertions, as the judge sees them.

    Only claims that survived the gate and the verifier — what a reader will actually read.
    Judging the draft would credit the report for a finding that was removed for being
    unsupported.
    """
    lines: list[str] = []
    for claim in report.executive_summary:
        lines.append(f"- {claim.text}")
    for finding in report.findings:
        lines.append(f"- {finding.title}")
        lines.extend(f"  - {claim.text}" for claim in finding.claims)
    for hypothesis in report.hypotheses:
        lines.append(f"- [{hypothesis.verdict.value}] {hypothesis.statement}")
    for recommendation in report.recommendations:
        lines.append(f"- recommends: {recommendation.action}")
    # Confidence is included because "the data cannot establish a cause" is a *complete*
    # answer to a causal question, and a judge that cannot see the confidence would read it
    # as a gap.
    lines.append(f"- stated confidence: {report.confidence.value}")
    return "\n".join(lines)


def _as_score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        # An unparseable score should not decide anything, so it lands at the boundary and
        # lets the unaddressed list carry the verdict.
        return 1.0
