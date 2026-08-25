"""Whether the report answered the question.

Nothing checked this before. The loop stops when the model emits no tool call — its own
judgement that it is finished — and everything downstream verifies *what the report says*
rather than *whether it said enough*. A report can be perfectly grounded, survive the citation
gate and the adversarial verifier, and answer less than was asked.

Adapted from OpenHands' goal judge. The tests below are mostly about the two boundaries that
keep an LLM's opinion from doing damage: it may only add a caveat, and a judge that cannot run
must say so rather than imply a pass.
"""

from __future__ import annotations

import uuid
from typing import Any

from cortex.agents.llm import LLMError, Usage
from cortex.reports.completeness import AnswerAssessment, CompletenessJudge
from cortex.reports.schema import Confidence, InvestigationReport

EVIDENCE = uuid.uuid4()


class _Judge:
    """A provider that returns whatever verdict a test dictates."""

    def __init__(self, payload: dict[str, Any] | None = None, error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.prompts: list[str] = []

    async def structured(self, **kwargs: Any) -> tuple[dict[str, Any], Usage]:
        self.prompts.append(str(kwargs.get("messages", "")))
        if self.error is not None:
            raise self.error
        return self.payload or {}, Usage(input_tokens=100, output_tokens=20)


def _report(**overrides: Any) -> InvestigationReport:
    defaults: dict[str, Any] = {
        "question": "Why did signups fall, and did anyone report it?",
        "executive_summary": [
            {"text": "Signups fell 31% on mobile.", "evidence_ids": [str(EVIDENCE)]}
        ],
        "confidence": Confidence.MEDIUM,
    }
    return InvestigationReport(**{**defaults, **overrides})


class TestAnIncompleteAnswerIsDisclosed:
    async def test_unaddressed_parts_become_a_risk(self) -> None:
        judge = _Judge(
            {
                "score": 0.5,
                "complete": False,
                "unaddressed": ["whether anyone reported the drop in Slack"],
            }
        )
        assessment = await CompletenessJudge(judge).assess(  # type: ignore[arg-type]
            "Why did signups fall, and did anyone report it?", _report()
        )

        assert assessment.complete is False
        risks = assessment.risks
        assert len(risks) == 1
        assert "reported the drop in Slack" in risks[0].description
        # The phrasing has to be honest about what it is *not* saying: the part that was
        # answered is still grounded.
        assert "not the whole answer" in risks[0].description

    async def test_a_complete_answer_adds_nothing(self) -> None:
        """A caveat on every report is noise, and noise teaches readers to skip the
        section."""
        judge = _Judge({"score": 1.0, "complete": True, "unaddressed": []})
        assessment = await CompletenessJudge(judge).assess("Why did signups fall?", _report())  # type: ignore[arg-type]

        assert assessment.complete is True
        assert assessment.risks == []

    async def test_a_verdict_of_incomplete_with_nothing_named_is_not_a_caveat(self) -> None:
        """A judge that says "incomplete" and lists nothing has told the reader nothing
        actionable. The list decides, not the flag."""
        judge = _Judge({"score": 0.2, "complete": False, "unaddressed": []})
        assessment = await CompletenessJudge(judge).assess("Why did signups fall?", _report())  # type: ignore[arg-type]

        assert assessment.complete is True
        assert assessment.risks == []

    async def test_only_the_first_few_gaps_are_listed(self) -> None:
        """A risk section that reproduces a ten-item checklist is not read."""
        judge = _Judge(
            {
                "score": 0.1,
                "complete": False,
                "unaddressed": [f"part {n}" for n in range(10)],
            }
        )
        assessment = await CompletenessJudge(judge).assess("q", _report())  # type: ignore[arg-type]
        description = assessment.risks[0].description
        assert "part 0" in description
        assert "part 9" not in description


class TestItNeverEdits:
    async def test_the_assessment_carries_no_report(self) -> None:
        """This is an LLM's opinion, and grounding decisions in this codebase are mechanical.
        The structural guarantee is that the judge has no channel through which to return
        edited content: its result carries a verdict and risks, and no report. A judge that
        goes wrong therefore costs a spurious sentence, never a lost finding."""
        import dataclasses

        judge = _Judge({"score": 0.0, "complete": False, "unaddressed": ["everything"]})
        assessment = await CompletenessJudge(judge).assess("q", _report())  # type: ignore[arg-type]

        fields = {field.name for field in dataclasses.fields(assessment)}
        assert fields == {"assessed", "complete", "score", "unaddressed", "error"}
        assert not any("report" in name or "claim" in name for name in fields)

    async def test_the_caller_decides_what_to_do_with_the_verdict(self) -> None:
        """The judge appends nothing itself. `risks` is a suggestion the caller merges, which
        is why an incomplete verdict cannot silently rewrite a delivered report."""
        judge = _Judge({"score": 0.2, "complete": False, "unaddressed": ["the Slack part"]})
        report = _report()
        before = len(report.risks)

        assessment = await CompletenessJudge(judge).assess("q", report)  # type: ignore[arg-type]

        assert len(report.risks) == before, "the report the judge was given is untouched"
        assert len(assessment.risks) == 1


class TestWhenTheJudgeCannotRun:
    async def test_a_provider_failure_says_it_was_not_checked(self) -> None:
        """Silence would be the worst option: a reader takes the absence of a caveat as
        coverage. Same distinction as F-24 — "nothing was wrong" is not "nothing was
        checked"."""
        judge = _Judge(error=LLMError("provider unavailable"))
        assessment = await CompletenessJudge(judge).assess("q", _report())  # type: ignore[arg-type]

        assert assessment.assessed is False
        assert len(assessment.risks) == 1
        assert "was not checked" in assessment.risks[0].description

    async def test_it_does_not_raise(self) -> None:
        """A judge that could break an investigation is a worse trade than one that
        occasionally declines, given it can only ever add a caveat."""
        judge = _Judge(error=LLMError("boom"))
        assessment = await CompletenessJudge(judge).assess("q", _report())  # type: ignore[arg-type]
        assert isinstance(assessment, AnswerAssessment)

    async def test_a_malformed_score_does_not_decide_anything(self) -> None:
        judge = _Judge({"score": "not-a-number", "complete": True, "unaddressed": []})
        assessment = await CompletenessJudge(judge).assess("q", _report())  # type: ignore[arg-type]
        assert assessment.complete is True

    async def test_a_claimless_report_cannot_even_be_constructed(self) -> None:
        """The `no claims` guard in the judge is defensive and provably unreachable through
        the normal path: `Claim.evidence_ids` requires at least one id, and
        `executive_summary` requires at least one claim. Asserted rather than assumed,
        because a guard whose condition cannot occur is worth knowing about — it either
        documents an invariant or it is dead code, and this one documents an invariant."""
        import pytest as _pytest

        with _pytest.raises(Exception, match="at least 1 item"):
            _report(executive_summary=[{"text": "x", "evidence_ids": []}])


class TestWhatTheJudgeIsShown:
    async def test_it_sees_the_stated_confidence(self) -> None:
        """ "The data cannot establish a cause" is a *complete* answer to a causal question.
        A judge that could not see the confidence would read it as a gap."""
        judge = _Judge({"score": 1.0, "complete": True, "unaddressed": []})
        await CompletenessJudge(judge).assess(  # type: ignore[arg-type]
            "Why did signups fall?",
            _report(confidence=Confidence.INSUFFICIENT_EVIDENCE),
        )
        assert "insufficient_evidence" in judge.prompts[0]

    async def test_it_sees_hypotheses_and_recommendations(self) -> None:
        """A question can be answered by ruling something out, which lives in a hypothesis
        rather than in the summary."""
        judge = _Judge({"score": 1.0, "complete": True, "unaddressed": []})
        await CompletenessJudge(judge).assess(  # type: ignore[arg-type]
            "Was it the deploy?",
            _report(
                hypotheses=[
                    {
                        "statement": "The deploy caused it",
                        "verdict": "contradicted",
                        "contradicting_evidence_ids": [str(EVIDENCE)],
                    }
                ],
                recommendations=[
                    {
                        "action": "Ship the sticky footer fix",
                        "rationale": "the reviewer flagged it",
                        "evidence_ids": [str(EVIDENCE)],
                    }
                ],
            ),
        )
        prompt = judge.prompts[0]
        assert "contradicted" in prompt
        assert "sticky footer" in prompt

    async def test_the_question_is_shown_as_asked(self) -> None:
        """Judged against the user's words, not the report's restatement of them — a report
        that quietly narrowed the question would otherwise mark itself complete."""
        judge = _Judge({"score": 1.0, "complete": True, "unaddressed": []})
        await CompletenessJudge(judge).assess(  # type: ignore[arg-type]
            "Why did signups fall AND did anyone report it?",
            _report(question="Why did mobile signups fall?"),
        )
        assert "did anyone report it" in judge.prompts[0]
