"""The sufficiency gate against a real session.

Separated from `tests/reports/test_sufficiency.py` because the gate loads the cited evidence
before it calls anything, so the paths that reach the provider need a database. The pure
verdict-application logic is tested without one.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from cortex.agents.llm import LLMError, Usage
from cortex.db.models import Evidence, Investigation, InvestigationStatus, Tenant
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.reports.schema import Claim, Confidence, InvestigationReport
from cortex.reports.sufficiency import SufficiencyGate
from cortex.tenancy.context import TenantContext
from cortex.tools.executor import canonical_hash


class _Scripted:
    def __init__(self, *, sufficient: bool = True, fail: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self._sufficient = sufficient
        self._fail = fail

    async def structured(self, **kwargs: Any) -> tuple[dict[str, Any], Usage]:
        self.calls.append(kwargs)
        if self._fail:
            raise LLMError("provider unavailable")
        return (
            {
                "sufficient": self._sufficient,
                "missing": [] if self._sufficient else ["a comparable control series"],
                "reason": "scripted",
            },
            Usage(input_tokens=10, output_tokens=5),
        )


async def _causal_report(
    session: AsyncSession,
) -> tuple[TenantContext, uuid.UUID, InvestigationReport]:
    slug = f"suff-{uuid.uuid4().hex[:6]}"
    tenant_id = uuid.uuid4()
    graph_name = graph_name_for_new_tenant(slug, tenant_id)
    session.add(Tenant(id=tenant_id, slug=slug, name=slug, graph_name=graph_name))
    await session.flush()
    investigation = Investigation(
        tenant_id=tenant_id,
        question="Why did signups fall in mid-June?",
        status=InvestigationStatus.COMPLETED,
    )
    session.add(investigation)
    await session.flush()

    payload = {"rows": [{"day": "2026-06-17", "value": 91}]}
    evidence = Evidence(
        tenant_id=tenant_id,
        investigation_id=investigation.id,
        tool_name="posthog",
        capability="event_trend",
        params={"event": "user signed up"},
        payload=payload,
        payload_hash=canonical_hash(payload),
        source_ref="posthog://x",
        from_cache=False,
    )
    session.add(evidence)
    await session.flush()

    report = InvestigationReport(
        question="Why did signups fall in mid-June?",
        executive_summary=[
            Claim(
                text="Signups fell because of the 16 June marketing site refresh.",
                evidence_ids=[evidence.id],
            )
        ],
        confidence=Confidence.MEDIUM,
    )
    context = TenantContext(tenant_id=tenant_id, tenant_slug=slug, graph_name=graph_name)
    return context, investigation.id, report


class TestItReachesTheProviderOnlyWhenItMust:
    async def test_a_causal_report_is_assessed(self, session: AsyncSession) -> None:
        tenant, investigation_id, report = await _causal_report(session)
        llm = _Scripted(sufficient=False)
        decision = await SufficiencyGate(llm).assess(  # type: ignore[arg-type]
            session,
            tenant,
            investigation_id=investigation_id,
            question="Why did signups fall in mid-June?",
            report=report,
        )
        assert len(llm.calls) == 1
        assert decision.needed and decision.ran
        assert decision.vetoes
        assert "comparable control series" in decision.missing_sentence

    async def test_the_evidence_reaches_the_prompt(self, session: AsyncSession) -> None:
        """The gate grades the evidence, so the evidence has to be in front of it. A prompt
        carrying only the claim would be the anchored judgement this design exists to avoid."""
        tenant, investigation_id, report = await _causal_report(session)
        llm = _Scripted()
        await SufficiencyGate(llm).assess(  # type: ignore[arg-type]
            session,
            tenant,
            investigation_id=investigation_id,
            question="Why did signups fall in mid-June?",
            report=report,
        )
        rendered = str(llm.calls[0])
        assert "2026-06-17" in rendered
        assert "event_trend" in rendered

    async def test_the_conclusion_does_not_reach_the_prompt(self, session: AsyncSession) -> None:
        """The point of a second call. A judge shown the answer grades the answer, and what
        needs grading is the evidence -- so the drafted sentence must not travel with it."""
        tenant, investigation_id, report = await _causal_report(session)
        llm = _Scripted()
        await SufficiencyGate(llm).assess(  # type: ignore[arg-type]
            session,
            tenant,
            investigation_id=investigation_id,
            question="Why did signups fall in mid-June?",
            report=report,
        )
        rendered = str(llm.calls[0])
        assert "16 June marketing site refresh" not in rendered

    async def test_a_provider_failure_withholds_nothing(self, session: AsyncSession) -> None:
        """A gate that could break an investigation by being unavailable would be a worse trade
        than one that occasionally declines to judge."""
        tenant, investigation_id, report = await _causal_report(session)
        decision = await SufficiencyGate(_Scripted(fail=True)).assess(  # type: ignore[arg-type]
            session,
            tenant,
            investigation_id=investigation_id,
            question="Why did signups fall in mid-June?",
            report=report,
        )
        assert decision.needed
        assert not decision.ran
        assert not decision.vetoes
        assert decision.error

    async def test_a_descriptive_report_never_reaches_the_provider(
        self, session: AsyncSession
    ) -> None:
        """The latency argument, against a loop already at 192s on a 90-second budget."""
        tenant, investigation_id, report = await _causal_report(session)
        descriptive = report.model_copy(
            update={
                "executive_summary": [
                    Claim(
                        text="Signups fell 53% starting 2026-06-17.",
                        evidence_ids=report.executive_summary[0].evidence_ids,
                    )
                ]
            }
        )
        llm = _Scripted()
        decision = await SufficiencyGate(llm).assess(  # type: ignore[arg-type]
            session,
            tenant,
            investigation_id=investigation_id,
            question="Did signups fall?",
            report=descriptive,
        )
        assert llm.calls == []
        assert not decision.needed
