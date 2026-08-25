"""Grounding gate.

The gate is where "never hallucinate" stops being a prompt and becomes a database
query. These run against real Postgres because the guarantee *is* the query: a
mocked evidence store would happily confirm a citation the real one would reject.

The tenant-scoping tests are the security half — see docs/security-findings.md F-01
for why evidence resolution has to be scoped by tenant *and* investigation.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import Evidence, Investigation, Tenant
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.reports.gate import (
    GroundingGate,
    RejectionReason,
    ReportRejected,
)
from cortex.reports.schema import (
    ChartPoint,
    ChartSeries,
    ChartSpec,
    ChartType,
    Claim,
    Confidence,
    Finding,
    Hypothesis,
    InvestigationReport,
    Recommendation,
    Risk,
    Verdict,
)
from cortex.tenancy.context import TenantContext
from cortex.tools.executor import canonical_hash


async def _tenant(session: AsyncSession, slug: str = "gate-test") -> TenantContext:
    tenant_id = uuid.uuid4()
    session.add(
        Tenant(
            id=tenant_id,
            slug=slug,
            name=slug,
            graph_name=graph_name_for_new_tenant(slug, tenant_id),
        )
    )
    await session.flush()
    return TenantContext(
        tenant_id=tenant_id,
        tenant_slug=slug,
        graph_name=graph_name_for_new_tenant(slug, tenant_id),
    )


async def _investigation(session: AsyncSession, ctx: TenantContext) -> uuid.UUID:
    inv = Investigation(tenant_id=ctx.tenant_id, question="Why did signups fall?")
    session.add(inv)
    await session.flush()
    return inv.id


async def _evidence(
    session: AsyncSession,
    ctx: TenantContext,
    investigation_id: uuid.UUID,
    *,
    payload: dict | None = None,
    tool: str = "ga4",
    capability: str = "get_sessions",
) -> Evidence:
    body = payload if payload is not None else {"sessions": 1200}
    row = Evidence(
        tenant_id=ctx.tenant_id,
        investigation_id=investigation_id,
        tool_name=tool,
        capability=capability,
        params={"days": 7},
        payload=body,
        payload_hash=canonical_hash(body),
        source_ref=f"{tool}://observation",
    )
    session.add(row)
    await session.flush()
    return row


def _report(summary: list[Claim], **overrides: object) -> InvestigationReport:
    return InvestigationReport(
        question="Why did signups fall?", executive_summary=summary, **overrides
    )  # type: ignore[arg-type]


class TestResolvableCitationsSurvive:
    async def test_fully_grounded_report_passes_untouched(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        report = _report([Claim(text="Signups fell 18%.", evidence_ids=[evidence.id])])
        result = await GroundingGate().apply(
            session, ctx, investigation_id=investigation_id, report=report
        )

        assert result.passed
        assert result.rejections == []
        assert len(result.report.executive_summary) == 1
        assert result.report.confidence is Confidence.MEDIUM

    async def test_sources_are_derived_from_the_evidence_store(self, session: AsyncSession) -> None:
        """Never model-authored: an invented permalink must not reach the one
        section a reader treats as verifiable."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        report = _report([Claim(text="Signups fell.", evidence_ids=[evidence.id])])
        result = await GroundingGate().apply(
            session, ctx, investigation_id=investigation_id, report=report
        )

        assert len(result.report.sources) == 1
        source = result.report.sources[0]
        assert source.evidence_id == evidence.id
        assert source.tool_name == "ga4"
        assert source.source_ref == "ga4://observation"
        assert source.observed_at is not None


class TestUnknownCitations:
    async def test_invented_evidence_id_is_rejected(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        real = await _evidence(session, ctx, investigation_id)

        report = _report(
            [
                Claim(text="Signups fell 18%.", evidence_ids=[real.id]),
                Claim(text="Traffic doubled.", evidence_ids=[uuid.uuid4()]),
            ]
        )
        result = await GroundingGate().apply(
            session, ctx, investigation_id=investigation_id, report=report
        )

        texts = [c.text for c in result.report.executive_summary]
        assert texts == ["Signups fell 18%."]
        assert result.hallucination_count == 1
        assert any(r.reason is RejectionReason.UNKNOWN_EVIDENCE for r in result.rejections)

    async def test_partially_grounded_claim_keeps_only_real_citations(
        self, session: AsyncSession
    ) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        real = await _evidence(session, ctx, investigation_id)
        invented = uuid.uuid4()

        report = _report([Claim(text="Signups fell 18%.", evidence_ids=[real.id, invented])])
        result = await GroundingGate().apply(
            session, ctx, investigation_id=investigation_id, report=report
        )

        surviving = result.report.executive_summary[0]
        assert surviving.evidence_ids == [real.id]
        assert invented not in surviving.evidence_ids

    async def test_report_with_no_surviving_summary_is_rejected(
        self, session: AsyncSession
    ) -> None:
        """A summary with nothing citable is not a degraded answer, it is no answer."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)

        report = _report([Claim(text="Signups fell 18%.", evidence_ids=[uuid.uuid4()])])
        with pytest.raises(ReportRejected):
            await GroundingGate().apply(
                session, ctx, investigation_id=investigation_id, report=report
            )


class TestTenantScoping:
    """F-01's other half. Even if evidence were planted on this investigation, or an
    id from another tenant were cited, the gate must not resolve it."""

    async def test_another_tenants_evidence_is_not_resolvable(self, session: AsyncSession) -> None:
        victim = await _tenant(session, "gate-victim")
        attacker = await _tenant(session, "gate-attacker")
        victim_investigation = await _investigation(session, victim)
        attacker_investigation = await _investigation(session, attacker)

        victim_evidence = await _evidence(session, victim, victim_investigation)
        attacker_evidence = await _evidence(session, attacker, attacker_investigation)

        report = _report(
            [
                Claim(text="My claim.", evidence_ids=[attacker_evidence.id]),
                Claim(text="Their claim.", evidence_ids=[victim_evidence.id]),
            ]
        )
        result = await GroundingGate().apply(
            session, attacker, investigation_id=attacker_investigation, report=report
        )

        assert [c.text for c in result.report.executive_summary] == ["My claim."]
        assert victim_evidence.id not in {s.evidence_id for s in result.report.sources}

    async def test_same_tenants_other_investigation_is_not_resolvable(
        self, session: AsyncSession
    ) -> None:
        """tenant_id alone is not enough — one tenant's unrelated investigation must
        not leak into this report's citations."""
        ctx = await _tenant(session)
        first = await _investigation(session, ctx)
        second = await _investigation(session, ctx)
        other_evidence = await _evidence(session, ctx, second)
        own_evidence = await _evidence(session, ctx, first)

        report = _report(
            [
                Claim(text="Own claim.", evidence_ids=[own_evidence.id]),
                Claim(text="Other investigation.", evidence_ids=[other_evidence.id]),
            ]
        )
        result = await GroundingGate().apply(session, ctx, investigation_id=first, report=report)
        assert [c.text for c in result.report.executive_summary] == ["Own claim."]

    async def test_rejection_detail_does_not_confirm_existence(self, session: AsyncSession) -> None:
        """Absent and foreign share one reason code, so a report cannot be used to
        probe which evidence ids exist elsewhere."""
        victim = await _tenant(session, "gate-victim")
        attacker = await _tenant(session, "gate-attacker")
        victim_evidence = await _evidence(session, victim, await _investigation(session, victim))
        attacker_investigation = await _investigation(session, attacker)
        own = await _evidence(session, attacker, attacker_investigation)

        report = _report(
            [
                Claim(text="Own claim.", evidence_ids=[own.id]),
                Claim(text="Foreign claim.", evidence_ids=[victim_evidence.id]),
                Claim(text="Invented claim.", evidence_ids=[uuid.uuid4()]),
            ]
        )
        result = await GroundingGate().apply(
            session, attacker, investigation_id=attacker_investigation, report=report
        )
        reasons = {r.reason for r in result.rejections if r.location == "citation"}
        assert reasons == {RejectionReason.UNKNOWN_EVIDENCE}


class TestTamperDetection:
    async def test_edited_payload_is_no_longer_citable(self, session: AsyncSession) -> None:
        """The hash exists so a citation can be proven to refer to data actually
        observed — an edit after the fact must break the citation."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)
        good = await _evidence(session, ctx, investigation_id, payload={"sessions": 999})

        # Simulate the payload changing after it was recorded.
        evidence.payload = {"sessions": 999999}
        await session.flush()

        report = _report(
            [
                Claim(text="Tampered claim.", evidence_ids=[evidence.id]),
                Claim(text="Intact claim.", evidence_ids=[good.id]),
            ]
        )
        result = await GroundingGate().apply(
            session, ctx, investigation_id=investigation_id, report=report
        )

        assert [c.text for c in result.report.executive_summary] == ["Intact claim."]
        assert any(r.reason is RejectionReason.TAMPERED_EVIDENCE for r in result.rejections)

    async def test_hash_verification_can_be_disabled(self, session: AsyncSession) -> None:
        """For large eval runs against a known-good store."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)
        evidence.payload = {"sessions": 999999}
        await session.flush()

        result = await GroundingGate(verify_hashes=False).apply(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report([Claim(text="Tampered claim.", evidence_ids=[evidence.id])]),
        )
        assert result.passed


class TestSectionHandling:
    async def test_findings_with_no_surviving_claim_are_dropped(
        self, session: AsyncSession
    ) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        good = await _evidence(session, ctx, investigation_id)

        report = _report(
            [Claim(text="Summary claim.", evidence_ids=[good.id])],
            findings=[
                Finding(title="Real", claims=[Claim(text="Real claim.", evidence_ids=[good.id])]),
                Finding(
                    title="Fabricated",
                    claims=[Claim(text="Fake claim.", evidence_ids=[uuid.uuid4()])],
                ),
            ],
        )
        result = await GroundingGate().apply(
            session, ctx, investigation_id=investigation_id, report=report
        )
        assert [f.title for f in result.report.findings] == ["Real"]

    async def test_hypotheses_are_downgraded_not_deleted(self, session: AsyncSession) -> None:
        """That a hypothesis was considered is informative; deleting it hides the
        analyst's reasoning from the reader."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        good = await _evidence(session, ctx, investigation_id)

        report = _report(
            [Claim(text="Summary claim.", evidence_ids=[good.id])],
            hypotheses=[
                Hypothesis(
                    statement="The deploy caused it",
                    verdict=Verdict.SUPPORTED,
                    supporting_evidence_ids=[uuid.uuid4()],
                )
            ],
        )
        result = await GroundingGate().apply(
            session, ctx, investigation_id=investigation_id, report=report
        )

        assert len(result.report.hypotheses) == 1
        assert result.report.hypotheses[0].verdict is Verdict.INCONCLUSIVE
        assert result.report.hypotheses[0].supporting_evidence_ids == []

    async def test_uncited_risks_are_kept(self, session: AsyncSession) -> None:
        """Removing a caveat while leaving the conclusion it qualifies is never the
        safe direction."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        good = await _evidence(session, ctx, investigation_id)

        report = _report(
            [Claim(text="Summary claim.", evidence_ids=[good.id])],
            risks=[Risk(description="GA4 data may be sampled.")],
        )
        result = await GroundingGate().apply(
            session, ctx, investigation_id=investigation_id, report=report
        )
        assert len(result.report.risks) == 1

    async def test_ungrounded_recommendations_are_dropped(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        good = await _evidence(session, ctx, investigation_id)

        report = _report(
            [Claim(text="Summary claim.", evidence_ids=[good.id])],
            recommendations=[
                Recommendation(
                    action="Roll back the modal", rationale="It broke", evidence_ids=[good.id]
                ),
                Recommendation(
                    action="Rewrite pricing", rationale="Hunch", evidence_ids=[uuid.uuid4()]
                ),
            ],
        )
        result = await GroundingGate().apply(
            session, ctx, investigation_id=investigation_id, report=report
        )
        assert [r.action for r in result.report.recommendations] == ["Roll back the modal"]

    async def test_ungrounded_charts_are_dropped(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        good = await _evidence(session, ctx, investigation_id)

        report = _report(
            [Claim(text="Summary claim.", evidence_ids=[good.id])],
            charts=[
                ChartSpec(
                    type=ChartType.LINE,
                    title="Fabricated",
                    series=[ChartSeries(name="s", points=[ChartPoint(x="2026-07-01", y=1.0)])],
                    evidence_ids=[uuid.uuid4()],
                )
            ],
        )
        result = await GroundingGate().apply(
            session, ctx, investigation_id=investigation_id, report=report
        )
        assert result.report.charts == []


class TestConfidenceAdjustment:
    async def test_high_confidence_is_capped_when_something_was_removed(
        self, session: AsyncSession
    ) -> None:
        """A model that cited fabricated evidence and declared high confidence was
        wrong about more than the citation."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        good = await _evidence(session, ctx, investigation_id)

        report = _report(
            [
                Claim(text="Real claim.", evidence_ids=[good.id]),
                Claim(text="Fake claim.", evidence_ids=[uuid.uuid4()]),
            ],
            confidence=Confidence.HIGH,
        )
        result = await GroundingGate().apply(
            session, ctx, investigation_id=investigation_id, report=report
        )
        assert result.report.confidence is Confidence.MEDIUM

    async def test_clean_report_keeps_its_stated_confidence(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        good = await _evidence(session, ctx, investigation_id)

        result = await GroundingGate().apply(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report(
                [Claim(text="Real claim.", evidence_ids=[good.id])], confidence=Confidence.HIGH
            ),
        )
        assert result.report.confidence is Confidence.HIGH


class TestRejectionRecords:
    async def test_rejections_serialise_for_the_report_row(self, session: AsyncSession) -> None:
        """The eval suite scores hallucinations from these, so they are a product
        signal rather than a debug log."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        good = await _evidence(session, ctx, investigation_id)

        report = _report(
            [
                Claim(text="Real claim.", evidence_ids=[good.id]),
                Claim(text="Fake claim.", evidence_ids=[uuid.uuid4()]),
            ]
        )
        result = await GroundingGate().apply(
            session, ctx, investigation_id=investigation_id, report=report
        )
        for rejection in result.rejections:
            record = rejection.as_dict()
            assert set(record) == {"location", "reason", "detail", "text"}
            assert all(isinstance(v, str) for v in record.values())

    async def test_empty_claim_is_not_double_counted_as_a_hallucination(
        self, session: AsyncSession
    ) -> None:
        """An empty claim is the consequence of a bad citation, not a second
        independent fabrication."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        good = await _evidence(session, ctx, investigation_id)

        report = _report(
            [
                Claim(text="Real claim.", evidence_ids=[good.id]),
                Claim(text="Fake claim.", evidence_ids=[uuid.uuid4()]),
            ]
        )
        result = await GroundingGate().apply(
            session, ctx, investigation_id=investigation_id, report=report
        )
        assert result.hallucination_count == 1
        assert len(result.rejections) > 1


class TestGateEdgePaths:
    async def test_a_report_citing_nothing_needs_no_query(self, session: AsyncSession) -> None:
        """Structurally impossible via the schema — a claim cannot exist without an
        evidence id — so this guards the resolver's own empty-input branch rather
        than a reachable report shape."""
        from cortex.reports.gate import GroundingGate as _Gate

        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        gate = _Gate()
        valid, rows = await gate._resolve(session, ctx, investigation_id, set())
        assert valid == set()
        assert rows == {}

    async def test_a_contradicted_hypothesis_is_downgraded(self, session: AsyncSession) -> None:
        """The mirror of the supported case. Without it, a contradicted hypothesis
        whose evidence was rejected would keep claiming to have ruled something out."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        good = await _evidence(session, ctx, investigation_id)

        report = _report(
            [Claim(text="Summary claim.", evidence_ids=[good.id])],
            hypotheses=[
                Hypothesis(
                    statement="Pricing caused it",
                    verdict=Verdict.CONTRADICTED,
                    contradicting_evidence_ids=[uuid.uuid4()],
                )
            ],
        )
        result = await GroundingGate().apply(
            session, ctx, investigation_id=investigation_id, report=report
        )
        assert result.report.hypotheses[0].verdict is Verdict.INCONCLUSIVE
        assert result.report.hypotheses[0].contradicting_evidence_ids == []
        assert any("contradicting evidence was rejected" in r.detail for r in result.rejections)

    async def test_a_surviving_contradicted_hypothesis_keeps_its_verdict(
        self, session: AsyncSession
    ) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        good = await _evidence(session, ctx, investigation_id)

        report = _report(
            [Claim(text="Summary claim.", evidence_ids=[good.id])],
            hypotheses=[
                Hypothesis(
                    statement="Pricing caused it",
                    verdict=Verdict.CONTRADICTED,
                    contradicting_evidence_ids=[good.id],
                )
            ],
        )
        result = await GroundingGate().apply(
            session, ctx, investigation_id=investigation_id, report=report
        )
        assert result.report.hypotheses[0].verdict is Verdict.CONTRADICTED

    @pytest.mark.parametrize("stated", [Confidence.LOW, Confidence.INSUFFICIENT_EVIDENCE])
    async def test_already_low_confidence_is_not_lowered_further(
        self, session: AsyncSession, stated: Confidence
    ) -> None:
        """There is nothing below these, and silently rewriting them would misreport
        what the analyst said."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        good = await _evidence(session, ctx, investigation_id)

        report = _report(
            [
                Claim(text="Real claim.", evidence_ids=[good.id]),
                Claim(text="Fake claim.", evidence_ids=[uuid.uuid4()]),
            ],
            confidence=stated,
        )
        result = await GroundingGate().apply(
            session, ctx, investigation_id=investigation_id, report=report
        )
        assert result.report.confidence is stated

    async def test_data_quality_notes_lose_only_rejected_citations(
        self, session: AsyncSession
    ) -> None:
        """A staleness disclosure is kept even when one of its citations fails —
        dropping a caveat is never the safe direction."""
        from cortex.reports.schema import DataQualityNote

        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        good = await _evidence(session, ctx, investigation_id)

        report = _report(
            [Claim(text="Summary claim.", evidence_ids=[good.id])],
            data_quality=[
                DataQualityNote(
                    note="HubSpot data is from a nightly sync.",
                    evidence_ids=[good.id, uuid.uuid4()],
                )
            ],
        )
        result = await GroundingGate().apply(
            session, ctx, investigation_id=investigation_id, report=report
        )
        assert len(result.report.data_quality) == 1
        assert result.report.data_quality[0].evidence_ids == [good.id]
