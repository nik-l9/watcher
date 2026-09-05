"""Adversarial verifier.

The gate proves a citation resolves; the verifier proves it *supports the claim*.
A real evidence row about mobile sessions attached to a sentence about enterprise
revenue passes the gate and is still wrong — this is the pass that catches it.

Driven by RecordedLLM against real Postgres, so the evidence loading, the
tenant scoping and the claim-removal logic are all exercised deterministically.
The prompt itself is asserted on, not just the outcome: an adversarial framing that
silently became agreeable would still pass an outcome-only test.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.agents.llm import LLMError, RecordedLLM, Usage
from cortex.db.models import Evidence, Investigation, Tenant
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.reports.gate import ReportRejected
from cortex.reports.schema import Claim, Confidence, Finding, InvestigationReport
from cortex.reports.verifier import (
    VERDICT_SCHEMA,
    AdversarialVerifier,
    Verdict,
)
from cortex.tenancy.context import TenantContext
from cortex.tools.executor import canonical_hash


async def _tenant(session: AsyncSession, slug: str = "verify-test") -> TenantContext:
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
    body = payload if payload is not None else {"sessions": 1200, "conversion": 0.029}
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


def _verdicts(*pairs: tuple[str, str]) -> RecordedLLM:
    """A scripted verifier, one verdict per claim in evaluation order."""
    return RecordedLLM(structured_outputs=[{"verdict": v, "reason": r} for v, r in pairs])


class TestSupportedClaimsSurvive:
    async def test_a_supported_claim_is_kept(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        llm = _verdicts(("supported", "the evidence shows 1200 sessions"))
        result = await AdversarialVerifier(llm).verify(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report([Claim(text="Sessions were 1200.", evidence_ids=[evidence.id])]),
        )

        assert len(result.report.executive_summary) == 1
        assert result.unsupported_count == 0
        assert result.rejections == []
        assert result.usage.total > 0

    async def test_a_clean_report_keeps_its_confidence(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        result = await AdversarialVerifier(_verdicts(("supported", "holds"))).verify(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report(
                [Claim(text="Sessions were 1200.", evidence_ids=[evidence.id])],
                confidence=Confidence.HIGH,
            ),
        )
        assert result.report.confidence is Confidence.HIGH
        assert result.report.risks == []


class TestUnsupportedClaimsRemoved:
    async def test_unsupported_claim_is_removed(self, session: AsyncSession) -> None:
        """The case the gate structurally cannot catch: a real citation attached to
        a claim it does not establish."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        good = await _evidence(session, ctx, investigation_id)
        unrelated = await _evidence(
            session,
            ctx,
            investigation_id,
            payload={"deals": []},
            tool="hubspot",
            capability="pipeline",
        )

        llm = _verdicts(
            ("supported", "the evidence shows 1200 sessions"),
            ("unsupported", "the evidence is about deals, not enterprise revenue"),
        )
        report = _report(
            [
                Claim(text="Sessions were 1200.", evidence_ids=[good.id]),
                Claim(text="Enterprise revenue fell 40%.", evidence_ids=[unrelated.id]),
            ]
        )
        result = await AdversarialVerifier(llm).verify(
            session, ctx, investigation_id=investigation_id, report=report
        )

        assert [c.text for c in result.report.executive_summary] == ["Sessions were 1200."]
        assert result.unsupported_count == 1
        assert result.report.confidence is Confidence.LOW

    async def test_removal_is_disclosed_as_a_risk(self, session: AsyncSession) -> None:
        """A report that quietly dropped claims would look cleaner than it is."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        good = await _evidence(session, ctx, investigation_id)
        other = await _evidence(session, ctx, investigation_id, payload={"x": 1})

        llm = _verdicts(("supported", "holds"), ("unsupported", "off topic"))
        result = await AdversarialVerifier(llm).verify(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report(
                [
                    Claim(text="Real claim.", evidence_ids=[good.id]),
                    Claim(text="Fabricated claim.", evidence_ids=[other.id]),
                ]
            ),
        )
        assert any("removed" in r.description for r in result.report.risks)

    async def test_rejections_are_recorded_for_the_report_row(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        good = await _evidence(session, ctx, investigation_id)
        other = await _evidence(session, ctx, investigation_id, payload={"x": 1})

        llm = _verdicts(("supported", "holds"), ("unsupported", "measures a different metric"))
        result = await AdversarialVerifier(llm).verify(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report(
                [
                    Claim(text="Real claim.", evidence_ids=[good.id]),
                    Claim(text="Fake claim.", evidence_ids=[other.id]),
                ]
            ),
        )
        assert len(result.rejections) == 1
        record = result.rejections[0].as_dict()
        assert "unsupported" in record["detail"]
        assert "different metric" in record["detail"]

    async def test_empty_summary_after_verification_is_rejected(
        self, session: AsyncSession
    ) -> None:
        """Same rule as the gate: an empty summary is no answer, and presenting one
        as an answer is what both mechanisms exist to prevent.

        Two verdicts are scripted, not one. A judgement that would empty the summary is now
        taken twice before the report is destroyed -- see
        `TestAJudgementThatWouldEmptyTheSummaryIsTakenTwice` for why -- so rejection requires
        both readings to agree.
        """
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        llm = _verdicts(
            ("unsupported", "the evidence does not establish this"),
            ("unsupported", "the second reading agrees"),
        )
        with pytest.raises(ReportRejected, match="no claim"):
            await AdversarialVerifier(llm).verify(
                session,
                ctx,
                investigation_id=investigation_id,
                report=_report([Claim(text="Invented claim.", evidence_ids=[evidence.id])]),
            )


class TestOverstatedClaimsKept:
    async def test_overstated_claim_is_kept_with_lower_confidence(
        self, session: AsyncSession
    ) -> None:
        """Directionally right but too strong is worth keeping with a caveat;
        removing it would lose a true finding."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        llm = _verdicts(("overstated", "the timeline shows correlation, not causation"))
        result = await AdversarialVerifier(llm).verify(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report(
                [Claim(text="The deploy caused the drop.", evidence_ids=[evidence.id])],
                confidence=Confidence.HIGH,
            ),
        )

        assert len(result.report.executive_summary) == 1
        assert result.overstated_count == 1
        assert result.report.confidence is Confidence.MEDIUM
        # Worded to cover both kinds of overshoot, since a claim carrying a figure the evidence
        # does not show is now `overstated` rather than removed.
        assert any("go beyond their evidence" in r.description for r in result.report.risks)
        assert any("retained with reduced confidence" in r.description for r in result.report.risks)

    async def test_overstated_appears_in_rejections_without_removal(
        self, session: AsyncSession
    ) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        llm = _verdicts(("overstated", "too causal"))
        result = await AdversarialVerifier(llm).verify(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report([Claim(text="The deploy caused it.", evidence_ids=[evidence.id])]),
        )
        assert len(result.rejections) == 1
        assert result.unsupported_count == 0
        assert len(result.report.executive_summary) == 1


class TestFindingsAreVerifiedToo:
    async def test_unsupported_finding_claims_are_removed(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        # Order: one summary claim, then two finding claims.
        llm = _verdicts(
            ("supported", "summary holds"),
            ("supported", "first finding holds"),
            ("unsupported", "second finding is unrelated"),
        )
        report = _report(
            [Claim(text="Summary claim.", evidence_ids=[evidence.id])],
            findings=[
                Finding(
                    title="Mobile",
                    claims=[
                        Claim(text="Mobile fell.", evidence_ids=[evidence.id]),
                        Claim(text="Desktop rose.", evidence_ids=[evidence.id]),
                    ],
                )
            ],
        )
        result = await AdversarialVerifier(llm).verify(
            session, ctx, investigation_id=investigation_id, report=report
        )
        assert [c.text for c in result.report.findings[0].claims] == ["Mobile fell."]

    async def test_a_finding_losing_every_claim_is_dropped(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        llm = _verdicts(("supported", "summary holds"), ("unsupported", "unrelated"))
        report = _report(
            [Claim(text="Summary claim.", evidence_ids=[evidence.id])],
            findings=[
                Finding(
                    title="Fabricated",
                    claims=[Claim(text="Fake claim.", evidence_ids=[evidence.id])],
                )
            ],
        )
        result = await AdversarialVerifier(llm).verify(
            session, ctx, investigation_id=investigation_id, report=report
        )
        assert result.report.findings == []


class TestPromptConstruction:
    """An adversarial framing that silently became agreeable would still pass an
    outcome-only test, so the prompt itself is asserted on."""

    async def test_system_prompt_asks_for_the_gap(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        llm = _verdicts(("supported", "holds"))
        await AdversarialVerifier(llm).verify(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report([Claim(text="Sessions were 1200.", evidence_ids=[evidence.id])]),
        )

        # Verdicts keep the provider's default deadline. They are short, one per
        # claim, and a long override here would let a single hung verdict stall a run
        # for minutes -- the opposite of why drafting was given a longer one.
        assert llm.calls[0]["timeout"] is None

        system = llm.calls[0]["system"]
        assert "find the gap, not to agree" in system
        assert "ONLY against the evidence shown" in system
        assert "Correlation stated as causation is overstated" in system

    async def test_only_the_cited_evidence_is_shown(self, session: AsyncSession) -> None:
        """Given the whole evidence set, a verifier reasons from the investigation's
        overall story and confirms claims the specific citation does not support."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        cited = await _evidence(session, ctx, investigation_id, payload={"cited": "yes"})
        uncited = await _evidence(session, ctx, investigation_id, payload={"UNCITED": "leak"})

        llm = _verdicts(("supported", "holds"))
        await AdversarialVerifier(llm).verify(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report([Claim(text="Cited claim.", evidence_ids=[cited.id])]),
        )

        prompt = llm.calls[0]["messages"][0].content
        assert str(cited.id) in prompt
        assert str(uncited.id) not in prompt
        assert "UNCITED" not in prompt

    async def test_prompt_carries_the_provenance_the_verdict_needs(
        self, session: AsyncSession
    ) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        llm = _verdicts(("supported", "holds"))
        await AdversarialVerifier(llm).verify(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report([Claim(text="Sessions were 1200.", evidence_ids=[evidence.id])]),
        )

        prompt = llm.calls[0]["messages"][0].content
        assert "ga4.get_sessions" in prompt
        assert "1200" in prompt
        assert "from_nightly_sync" in prompt
        assert "Sessions were 1200." in prompt

    async def test_truncation_is_disclosed_in_the_prompt(self, session: AsyncSession) -> None:
        """A cut-off payload must not read as evidence that a field is absent."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        big = await _evidence(
            session, ctx, investigation_id, payload={"rows": [{"i": i} for i in range(500)]}
        )

        llm = _verdicts(("supported", "holds"))
        await AdversarialVerifier(llm, max_evidence_chars=200).verify(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report([Claim(text="Rows exist.", evidence_ids=[big.id])]),
        )
        assert "TRUNCATED" in llm.calls[0]["messages"][0].content

    def test_verdict_schema_is_closed_and_enumerated(self) -> None:
        assert VERDICT_SCHEMA["additionalProperties"] is False
        assert set(VERDICT_SCHEMA["properties"]["verdict"]["enum"]) == {v.value for v in Verdict}
        assert VERDICT_SCHEMA["required"] == ["verdict", "reason"]


class TestTenantScoping:
    async def test_another_tenants_evidence_is_not_loaded(self, session: AsyncSession) -> None:
        """Same rule as the gate: an id from elsewhere must not be readable here."""
        victim = await _tenant(session, "verify-victim")
        attacker = await _tenant(session, "verify-attacker")
        victim_evidence = await _evidence(
            session, victim, await _investigation(session, victim), payload={"SECRET": 1}
        )
        attacker_investigation = await _investigation(session, attacker)
        own = await _evidence(session, attacker, attacker_investigation)

        # The foreign citation resolves to nothing, so it is unsupported without
        # the verifier ever seeing the payload.
        llm = _verdicts(("supported", "holds"))
        result = await AdversarialVerifier(llm).verify(
            session,
            attacker,
            investigation_id=attacker_investigation,
            report=_report(
                [
                    Claim(text="Own claim.", evidence_ids=[own.id]),
                    Claim(text="Foreign claim.", evidence_ids=[victim_evidence.id]),
                ]
            ),
        )

        assert [c.text for c in result.report.executive_summary] == ["Own claim."]
        assert all("SECRET" not in str(call) for call in llm.calls)


class TestUnloadableCitations:
    async def test_a_citation_that_cannot_be_loaded_is_unsupported(
        self, session: AsyncSession
    ) -> None:
        """The gate should have removed it; reaching here means it cites nothing
        resolvable, which is unsupported by definition — and no model call is made."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        good = await _evidence(session, ctx, investigation_id)

        llm = _verdicts(("supported", "holds"))
        result = await AdversarialVerifier(llm).verify(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report(
                [
                    Claim(text="Real claim.", evidence_ids=[good.id]),
                    Claim(text="Phantom claim.", evidence_ids=[uuid.uuid4()]),
                ]
            ),
        )
        assert [c.text for c in result.report.executive_summary] == ["Real claim."]
        # One scripted verdict consumed: the phantom needed no model call.
        assert len(llm.calls) == 1


class TestVerifierFailureIsSurvivable:
    async def test_a_provider_failure_keeps_the_claim(self, session: AsyncSession) -> None:
        """A verifier outage must not silently delete grounded work."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        # No scripted outputs: structured() raises LLMError.
        result = await AdversarialVerifier(RecordedLLM()).verify(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report([Claim(text="Sessions were 1200.", evidence_ids=[evidence.id])]),
        )

        assert len(result.report.executive_summary) == 1
        # Location plus cause: an unverified claim ships unchecked and fails the
        # eval, so the reason has to survive rather than only the count.
        assert len(result.unverified) == 1
        assert result.unverified[0].startswith("executive_summary[0] (not verified:")
        assert result.verdicts == []

    async def test_the_outage_is_disclosed_as_a_risk(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        result = await AdversarialVerifier(RecordedLLM()).verify(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report([Claim(text="Sessions were 1200.", evidence_ids=[evidence.id])]),
        )
        assert any(
            "could not be independently verified" in r.description for r in result.report.risks
        )

    async def test_a_malformed_verdict_counts_as_unverified(self, session: AsyncSession) -> None:
        """A provider returning an unrecognised verdict must not be read as a
        rejection — that would delete a claim on a parsing failure."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        llm = RecordedLLM(structured_outputs=[{"verdict": "maybe", "reason": "unsure"}])
        result = await AdversarialVerifier(llm).verify(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report([Claim(text="Sessions were 1200.", evidence_ids=[evidence.id])]),
        )
        assert len(result.report.executive_summary) == 1
        # Location plus cause: an unverified claim ships unchecked and fails the
        # eval, so the reason has to survive rather than only the count.
        assert len(result.unverified) == 1
        assert result.unverified[0].startswith("executive_summary[0] (not verified:")

    async def test_a_verdict_missing_its_field_counts_as_unverified(
        self, session: AsyncSession
    ) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        llm = RecordedLLM(structured_outputs=[{"reason": "no verdict field"}])
        result = await AdversarialVerifier(llm).verify(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report([Claim(text="Sessions were 1200.", evidence_ids=[evidence.id])]),
        )
        # Location plus cause: an unverified claim ships unchecked and fails the
        # eval, so the reason has to survive rather than only the count.
        assert len(result.unverified) == 1
        assert result.unverified[0].startswith("executive_summary[0] (not verified:")


class TestConcurrentJudging:
    """Claims are judged concurrently, and the report keeps its own order.

    Sequential judging made one round trip per claim. With reports carrying twenty-plus
    claims that was a large share of an investigation's wall clock against a 90-second
    target, and it was the shape of failure behind every scenario in one full eval run:
    the more claims, the more chances one verdict call fails and the claim ships
    unverified.
    """

    async def test_claims_are_judged_concurrently(self, session: AsyncSession) -> None:
        import asyncio

        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        class _Tracking(RecordedLLM):
            def __init__(self) -> None:
                super().__init__()
                self.in_flight = 0
                self.peak = 0

            async def structured(self, **kwargs):  # type: ignore[no-untyped-def]
                self.in_flight += 1
                self.peak = max(self.peak, self.in_flight)
                try:
                    # Yield so other judgements can start; sequential code cannot
                    # overlap here however long it waits.
                    await asyncio.sleep(0.01)
                    return {"verdict": "supported", "reason": "holds"}, Usage(
                        input_tokens=10, output_tokens=5
                    )
                finally:
                    self.in_flight -= 1

        llm = _Tracking()
        claims = [Claim(text=f"Claim number {n}.", evidence_ids=[evidence.id]) for n in range(8)]
        await AdversarialVerifier(llm).verify(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report(claims),
        )
        assert llm.peak > 1, "verdicts ran one at a time"

    async def test_report_order_survives_concurrency(self, session: AsyncSession) -> None:
        """Concurrent results arrive out of order. A report whose claims came back
        shuffled would still be grounded and would read as though someone had jumbled
        it."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        llm = RecordedLLM(structured_outputs=[{"verdict": "supported", "reason": "holds"}] * 6)
        texts = [f"Claim {n}." for n in range(6)]
        result = await AdversarialVerifier(llm).verify(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report([Claim(text=t, evidence_ids=[evidence.id]) for t in texts]),
        )
        assert [c.text for c in result.report.executive_summary] == texts

    async def test_one_unverifiable_claim_does_not_lose_the_others(
        self, session: AsyncSession
    ) -> None:
        """The run-failing case: one verdict call fails and the rest must still stand."""
        from cortex.agents.llm import LLMError

        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        class _FlakyOnce(RecordedLLM):
            """Fails for one claim on every attempt, keyed by its text.

            Keyed by claim rather than by call count because the judging call is retried once
            now: a counter would fail attempt one and be rescued by attempt two, which measures
            the retry rather than the property this test is about. Failing persistently for one
            claim is what keeps it about "one unverifiable claim does not lose the others".
            """

            async def structured(self, **kwargs):  # type: ignore[no-untyped-def]
                rendered = kwargs["messages"][0].content
                if "Claim 2." in rendered:
                    raise LLMError("capacity")
                return {"verdict": "supported", "reason": "holds"}, Usage(
                    input_tokens=10, output_tokens=5
                )

        result = await AdversarialVerifier(_FlakyOnce()).verify(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report(
                [Claim(text=f"Claim {n}.", evidence_ids=[evidence.id]) for n in range(4)]
            ),
        )
        # Every claim survives: three judged supported, one kept and disclosed.
        assert len(result.report.executive_summary) == 4
        assert len(result.unverified) == 1
        assert "capacity" in result.unverified[0]


class TestUsageAccounting:
    async def test_usage_accumulates_across_claims(self, session: AsyncSession) -> None:
        """Verification cost is real and must be attributable to the investigation."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        llm = _verdicts(("supported", "a"), ("supported", "b"), ("supported", "c"))
        result = await AdversarialVerifier(llm).verify(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report(
                [
                    Claim(text="One claim.", evidence_ids=[evidence.id]),
                    Claim(text="Claim two.", evidence_ids=[evidence.id]),
                    Claim(text="Claim three.", evidence_ids=[evidence.id]),
                ]
            ),
        )
        assert result.usage.input_tokens == 300
        assert len(result.verdicts) == 3


class TestNoCitationsAtAll:
    async def test_a_report_citing_nothing_loadable_is_rejected(
        self, session: AsyncSession
    ) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)

        with pytest.raises(ReportRejected):
            await AdversarialVerifier(RecordedLLM()).verify(
                session,
                ctx,
                investigation_id=investigation_id,
                report=_report([Claim(text="Phantom claim.", evidence_ids=[uuid.uuid4()])]),
            )


class TestLLMErrorType:
    def test_recorded_provider_raises_on_exhaustion(self) -> None:
        """An exhausted script means the verifier ran more claims than the test
        anticipated — worth being loud about rather than silently empty."""
        import asyncio

        with pytest.raises(LLMError):
            asyncio.run(RecordedLLM().structured(system="s", messages=[], schema={}))


class TestAJudgementThatWouldEmptyTheSummaryIsTakenTwice:
    """The severity of one wrong verdict scales with how short the summary is.

    `GUIDANCE[Shape.FACTUAL]` tells the analyst to lead with the answer in one or two
    sentences, so a factual report often has a one-claim summary -- and then a single mistaken
    `unsupported` is the difference between a delivered answer and no answer at all.

    Measured: a false-premise attempt was rejected entirely for a summary claim reading "August
    2026 shows only 1,884 signups versus 4,849 in July". Both figures are in the cited series,
    exactly. The verdict was simply wrong, and it cost the whole report on one attempt in three.

    Not leniency. The second reading is the same adversarial prompt over the same evidence, so a
    genuinely unsupported claim fails it again. It is the asymmetry that justifies the call, and
    the same reasoning the codebase already applies to a verifier *error* -- keep the claim,
    disclose that it went unchecked. A confident wrong verdict deserves no more trust than a
    failed one.
    """

    async def test_a_reversed_verdict_saves_the_report(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        # First reading rejects the only claim; second reading supports it.
        llm = _verdicts(
            ("unsupported", "I do not see 1200 anywhere"),
            ("supported", "the evidence shows 1200 sessions"),
        )
        report = _report([Claim(text="Sessions were 1200.", evidence_ids=[evidence.id])])
        result = await AdversarialVerifier(llm).verify(
            session, ctx, investigation_id=investigation_id, report=report
        )

        assert [c.text for c in result.report.executive_summary] == ["Sessions were 1200."]

    async def test_the_disagreement_is_disclosed(self, session: AsyncSession) -> None:
        """Two readers splitting on the only claim is a fact about the answer's reliability,
        and a report that quietly kept it would read as more settled than it is."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        llm = _verdicts(("unsupported", "no"), ("supported", "yes"))
        report = _report([Claim(text="Sessions were 1200.", evidence_ids=[evidence.id])])
        result = await AdversarialVerifier(llm).verify(
            session, ctx, investigation_id=investigation_id, report=report
        )

        disclosure = " ".join(risk.description for risk in result.report.risks)
        assert "judged unsupported on a first reading and kept on a second" in disclosure
        assert "less settled than its wording suggests" in disclosure

    async def test_two_rejections_still_reject_the_report(self, session: AsyncSession) -> None:
        """The check that stops this being a way through. An empty summary is no answer, and a
        claim the evidence genuinely does not support fails both readings."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        unrelated = await _evidence(
            session,
            ctx,
            investigation_id,
            payload={"deals": []},
            tool="hubspot",
            capability="pipeline",
        )

        llm = _verdicts(
            ("unsupported", "the evidence is about deals"),
            ("unsupported", "still about deals"),
        )
        report = _report([Claim(text="Enterprise revenue fell 40%.", evidence_ids=[unrelated.id])])
        with pytest.raises(ReportRejected, match="two independent readings"):
            await AdversarialVerifier(llm).verify(
                session, ctx, investigation_id=investigation_id, report=report
            )

    async def test_a_surviving_claim_costs_no_second_reading(self, session: AsyncSession) -> None:
        """The second reading exists only to avoid destroying a report. A summary with anything
        left in it must not pay for it."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        good = await _evidence(session, ctx, investigation_id)
        unrelated = await _evidence(
            session,
            ctx,
            investigation_id,
            payload={"deals": []},
            tool="hubspot",
            capability="pipeline",
        )

        llm = _verdicts(("supported", "shows 1200"), ("unsupported", "about deals"))
        report = _report(
            [
                Claim(text="Sessions were 1200.", evidence_ids=[good.id]),
                Claim(text="Enterprise revenue fell 40%.", evidence_ids=[unrelated.id]),
            ]
        )
        result = await AdversarialVerifier(llm).verify(
            session, ctx, investigation_id=investigation_id, report=report
        )

        assert [c.text for c in result.report.executive_summary] == ["Sessions were 1200."]
        disclosure = " ".join(risk.description for risk in result.report.risks)
        assert "kept on a second" not in disclosure


class TestAWrongFigureDoesNotRemoveASoundFinding:
    """The gap a real run walked into.

    An answer claimed "53 of 24+ distinct PostHog event types stopped recording". The payload said
    `stopped_count: 53, still_recording_count: 8`, and the note said so in words. The 24 appears
    nowhere in the cited evidence — and the verifier passed it as supported.

    It fell between the definitions. The precision rule said a figure must appear in the evidence
    but not which verdict a wrong one earns, and `overstated` was described in terms of strength,
    certainty and causality rather than accuracy. `unsupported` would have deleted a correct and
    important finding over one number; `supported` let the number through unremarked. The verdict
    now covers it, so the sentence survives with reduced confidence and the reason names the
    figure.
    """

    def test_the_verdict_description_covers_a_wrong_figure(self) -> None:
        from cortex.reports.verifier import VERDICT_SCHEMA

        description = VERDICT_SCHEMA["properties"]["verdict"]["description"]
        assert "carrying a figure the evidence does not show" in description
        assert "substance holds" in description

    def test_the_prompt_says_which_verdict_a_wrong_figure_earns(self) -> None:
        from cortex.reports.verifier import SYSTEM_PROMPT

        assert "makes a claim **overstated**, not unsupported" in SYSTEM_PROMPT
        assert "say which number is wrong in your reason" in SYSTEM_PROMPT

    def test_the_prompt_says_why_removal_would_be_worse(self) -> None:
        """The reasoning has to travel with the rule, or a later edit will "tighten" it back."""
        from cortex.reports.verifier import SYSTEM_PROMPT

        assert "loses more than it protects" in SYSTEM_PROMPT

    async def test_such_a_claim_is_kept_and_disclosed(self, session: AsyncSession) -> None:
        """The behaviour, not just the wording: overstated keeps the claim and warns."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        llm = _verdicts(("overstated", "the evidence shows 8 still recording, not 24"))
        result = await AdversarialVerifier(llm).verify(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report(
                [
                    Claim(
                        text="53 of 24+ event types stopped recording.",
                        evidence_ids=[evidence.id],
                    )
                ]
            ),
        )

        assert len(result.report.executive_summary) == 1, "a sound finding must not be deleted"
        assert result.overstated_count == 1
        disclosure = " ".join(r.description for r in result.report.risks)
        assert "go beyond their evidence" in disclosure


class TestATransientFailureIsRetried:
    """A timeout was costing a delivered hallucination.

    Run 30 lost three claims on one attempt to three consecutive `APITimeoutError`. No logic
    defect — a slow minute at the provider — and the suite correctly reported three delivered
    hallucinations, which must be zero, so the run failed for a reason nobody could act on.

    A claim this call cannot judge ships unchecked. Giving up after one attempt makes a
    transient blip indistinguishable from a genuinely unverifiable claim, and those deserve
    different outcomes.
    """

    async def test_a_claim_is_rescued_by_the_second_attempt(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        class _TimesOutOnce(RecordedLLM):
            def __init__(self) -> None:
                super().__init__()
                self.attempts = 0

            async def structured(self, **kwargs):  # type: ignore[no-untyped-def]
                self.attempts += 1
                if self.attempts == 1:
                    raise LLMError("APITimeoutError")
                return {"verdict": "supported", "reason": "holds"}, Usage(
                    input_tokens=10, output_tokens=5
                )

        llm = _TimesOutOnce()
        result = await AdversarialVerifier(llm).verify(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report([Claim(text="Signups fell 18%.", evidence_ids=[evidence.id])]),
        )
        assert result.unverified == []
        assert llm.attempts == 2

    async def test_a_persistent_failure_is_still_reported(self, session: AsyncSession) -> None:
        """One retry, not a policy. A claim that cannot be judged twice is disclosed rather
        than retried indefinitely — the count is what fails the build, and it should."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        class _AlwaysTimesOut(RecordedLLM):
            def __init__(self) -> None:
                super().__init__()
                self.attempts = 0

            async def structured(self, **kwargs):  # type: ignore[no-untyped-def]
                self.attempts += 1
                raise LLMError("APITimeoutError")

        llm = _AlwaysTimesOut()
        result = await AdversarialVerifier(llm).verify(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report([Claim(text="Signups fell 18%.", evidence_ids=[evidence.id])]),
        )
        assert len(result.unverified) == 1
        assert "APITimeoutError" in result.unverified[0]
        assert llm.attempts == 2

    async def test_a_rejected_credential_is_not_retried(self, session: AsyncSession) -> None:
        """The credential will reject the next call too, and spending another deadline to learn
        that is waste — the same distinction the investigation loop draws."""
        from cortex.agents.llm import LLMAuthenticationFailed

        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        evidence = await _evidence(session, ctx, investigation_id)

        class _Unauthorised(RecordedLLM):
            def __init__(self) -> None:
                super().__init__()
                self.attempts = 0

            async def structured(self, **kwargs):  # type: ignore[no-untyped-def]
                self.attempts += 1
                raise LLMAuthenticationFailed("401 Incorrect API key provided")

        llm = _Unauthorised()
        result = await AdversarialVerifier(llm).verify(
            session,
            ctx,
            investigation_id=investigation_id,
            report=_report([Claim(text="Signups fell 18%.", evidence_ids=[evidence.id])]),
        )
        assert llm.attempts == 1
        assert len(result.unverified) == 1
