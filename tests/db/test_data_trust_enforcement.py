"""Withholding a cause for a movement that was never measured.

`cortex.analysis.trust` computes the state and the drafting guidance asks the analyst to respect
it. Asking is not enforcing. This is the enforcement, and unlike the sufficiency gate it needs no
model judgement: whether the evidence *carries* a cause is a judgement, whether a cited series was
being measured at all is a fact.
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import Evidence, Investigation, InvestigationStatus, Tenant
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.reports.data_trust import enforce_data_trust
from cortex.reports.schema import (
    Claim,
    Confidence,
    Finding,
    InvestigationReport,
)
from cortex.tenancy.context import TenantContext
from cortex.tools.executor import canonical_hash


async def _setup(
    session: AsyncSession, *, broken: bool
) -> tuple[TenantContext, uuid.UUID, uuid.UUID]:
    slug = f"trust-{uuid.uuid4().hex[:6]}"
    tenant_id = uuid.uuid4()
    graph_name = graph_name_for_new_tenant(slug, tenant_id)
    session.add(Tenant(id=tenant_id, slug=slug, name=slug, graph_name=graph_name))
    await session.flush()
    investigation = Investigation(
        tenant_id=tenant_id,
        question="Why did signups fall?",
        status=InvestigationStatus.COMPLETED,
    )
    session.add(investigation)
    await session.flush()

    payload: dict = {"event": "user signed up", "series": [{"bucket": "2026-08-03", "value": 195}]}
    if broken:
        payload["data_trust"] = {
            "state": "broken",
            "may_answer_the_business_question": False,
            "tripped": [{"check": "gate3_correlated_cessation", "detail": "53 stopped, 8 live"}],
        }
    evidence = Evidence(
        tenant_id=tenant_id,
        investigation_id=investigation.id,
        tool_name="posthog",
        capability="event_trend",
        params={},
        payload=payload,
        payload_hash=canonical_hash(payload),
        source_ref="posthog://x",
        from_cache=False,
    )
    session.add(evidence)
    await session.flush()
    return (
        TenantContext(tenant_id=tenant_id, tenant_slug=slug, graph_name=graph_name),
        investigation.id,
        evidence.id,
    )


def _report(evidence_id: uuid.UUID, *, causal: bool) -> InvestigationReport:
    text = (
        "Signups fell because the pricing page changed."
        if causal
        else "The series stops recording on 2026-08-03."
    )
    return InvestigationReport(
        question="Why did signups fall?",
        executive_summary=[
            Claim(text=text, evidence_ids=[evidence_id]),
            Claim(text="Recording stopped on 2026-08-03.", evidence_ids=[evidence_id]),
        ],
        findings=[
            Finding(
                title="What the series shows",
                claims=[Claim(text="195 signups on 03 August.", evidence_ids=[evidence_id])],
                confidence=Confidence.HIGH,
            )
        ],
        confidence=Confidence.MEDIUM,
    )


class TestACauseForAnUnmeasuredMovementIsWithheld:
    async def test_the_causal_claim_is_removed(self, session: AsyncSession) -> None:
        tenant, investigation_id, evidence_id = await _setup(session, broken=True)
        applied = await enforce_data_trust(
            session,
            tenant,
            investigation_id=investigation_id,
            report=_report(evidence_id, causal=True),
        )
        assert applied.withheld == 1
        remaining = [c.text for c in applied.report.executive_summary]
        assert "Signups fell because the pricing page changed." not in remaining

    async def test_the_descriptive_claim_survives(self, session: AsyncSession) -> None:
        """ "The series stops on 2026-08-03" is true and is the answer. Removing it would leave
        the reader with nothing."""
        tenant, investigation_id, evidence_id = await _setup(session, broken=True)
        applied = await enforce_data_trust(
            session,
            tenant,
            investigation_id=investigation_id,
            report=_report(evidence_id, causal=True),
        )
        assert any(
            "Recording stopped on 2026-08-03." == c.text for c in applied.report.executive_summary
        )

    async def test_the_verdict_leads_the_summary(self, session: AsyncSession) -> None:
        """A reader who stops after one line must not be left with the surviving description
        and no explanation of why the cause is gone."""
        tenant, investigation_id, evidence_id = await _setup(session, broken=True)
        applied = await enforce_data_trust(
            session,
            tenant,
            investigation_id=investigation_id,
            report=_report(evidence_id, causal=True),
        )
        assert "was not observed" in applied.report.executive_summary[0].text

    async def test_the_verdict_cites_the_broken_series(self, session: AsyncSession) -> None:
        """So it resolves like any other claim, rather than being an uncited assertion the gate
        would need an exception for."""
        tenant, investigation_id, evidence_id = await _setup(session, broken=True)
        applied = await enforce_data_trust(
            session,
            tenant,
            investigation_id=investigation_id,
            report=_report(evidence_id, causal=True),
        )
        assert applied.report.executive_summary[0].evidence_ids == [evidence_id]

    async def test_an_emptied_summary_is_not_a_rejection(self, session: AsyncSession) -> None:
        """The gate and the verifier both refuse a report with no summary, because an empty
        summary is no answer. Here the reason it emptied *is* the answer, so discarding the
        investigation would throw away the only finding worth delivering."""
        tenant, investigation_id, evidence_id = await _setup(session, broken=True)
        only_causal = InvestigationReport(
            question="Why did signups fall?",
            executive_summary=[
                Claim(text="Signups fell because of a deploy.", evidence_ids=[evidence_id])
            ],
            confidence=Confidence.MEDIUM,
        )
        applied = await enforce_data_trust(
            session, tenant, investigation_id=investigation_id, report=only_causal
        )
        assert len(applied.report.executive_summary) == 1
        assert "was not observed" in applied.report.executive_summary[0].text

    async def test_the_removal_is_disclosed_as_mechanical(self, session: AsyncSession) -> None:
        tenant, investigation_id, evidence_id = await _setup(session, broken=True)
        applied = await enforce_data_trust(
            session,
            tenant,
            investigation_id=investigation_id,
            report=_report(evidence_id, causal=True),
        )
        disclosure = " ".join(r.description for r in applied.report.risks)
        assert "mechanical check, not a judgement about the argument" in disclosure


class TestItLeavesEverythingElseAlone:
    async def test_a_healthy_series_is_untouched(self, session: AsyncSession) -> None:
        tenant, investigation_id, evidence_id = await _setup(session, broken=False)
        report = _report(evidence_id, causal=True)
        applied = await enforce_data_trust(
            session, tenant, investigation_id=investigation_id, report=report
        )
        assert applied.report is report
        assert applied.withheld == 0

    async def test_a_report_with_no_causal_claim_never_queries(self, session: AsyncSession) -> None:
        """Cheap on the ordinary report: it returns before touching the database."""
        tenant, investigation_id, evidence_id = await _setup(session, broken=True)
        report = _report(evidence_id, causal=False)
        applied = await enforce_data_trust(
            session, tenant, investigation_id=investigation_id, report=report
        )
        assert applied.report is report

    async def test_a_causal_claim_citing_healthy_evidence_is_kept(
        self, session: AsyncSession
    ) -> None:
        """Only claims resting on a broken series. A causal claim on healthy evidence is a
        different question, and the sufficiency gate owns it."""
        tenant, investigation_id, broken_id = await _setup(session, broken=True)
        healthy = {"event": "$pageview", "series": [{"bucket": "2026-08-03", "value": 14197}]}
        other = Evidence(
            tenant_id=tenant.tenant_id,
            investigation_id=investigation_id,
            tool_name="posthog",
            capability="event_trend",
            params={},
            payload=healthy,
            payload_hash=canonical_hash(healthy),
            source_ref="posthog://y",
            from_cache=False,
        )
        session.add(other)
        await session.flush()

        report = InvestigationReport(
            question="Why did signups fall?",
            executive_summary=[
                Claim(text="Pageviews fell because of a CDN change.", evidence_ids=[other.id])
            ],
            confidence=Confidence.MEDIUM,
        )
        applied = await enforce_data_trust(
            session, tenant, investigation_id=investigation_id, report=report
        )
        assert applied.withheld == 0
