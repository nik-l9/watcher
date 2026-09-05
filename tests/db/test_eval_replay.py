"""Re-scoring a captured attempt without paying for it again.

**The problem this solves, and it was invisible until someone tried to re-measure.** An eval run
rolls its transaction back on purpose -- it must not leave tenants and evidence behind in
whatever database it was pointed at -- so nothing about an attempt survives it. Not the report,
not the evidence, not the audit rows. Every change to a scoring dimension therefore cost a full
run against the live provider, paid in tokens and exposed to the between-scenario variance that
dominates this harness.

That is how the accuracy figures in `docs/eval-results.md` became upper bounds rather than
measurements: the surface `_accuracy` searched was fixed, and re-measuring meant re-running.

The property under test throughout is that **a re-score is the same computation as the original
score**. Anything less and the numbers are not comparable, which is the problem this exists to
solve rather than a new one to introduce.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.agents.timing import GATE, SUFFICIENCY, VERIFY, Phase, Timings
from cortex.db.models import (
    Evidence,
    Investigation,
    InvestigationStatus,
    Tenant,
    ToolCall,
)
from cortex.eval.fixtures import SCENARIOS
from cortex.eval.replay import (
    BUNDLE_VERSION,
    BundleVersionMismatch,
    load_bundle,
    replay_bundle,
    write_bundle,
)
from cortex.eval.scorer import Scorer
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.reports.gate import GateResult, Rejection, RejectionReason
from cortex.reports.schema import Claim, Confidence, InvestigationReport, Source
from cortex.tenancy.context import TenantContext
from cortex.tools.executor import canonical_hash


class _Investigation:
    """The fields the scorer reads off an investigation.

    The phase clock is here rather than left off because `test_every_dimension_matches_the
    _original` was passing vacuously without it: with no clock on either side, `latency` fell
    back to `duration_ms` in both the live and the replayed score and the two agreed by
    construction. Replay was in fact scoring the loop alone -- 0.47 live against 1.00 on
    re-score for one real attempt -- and the test that exists to catch exactly that could not
    see it. So the stub carries post-loop phases, which is what makes the two paths differ if
    replay stops rebuilding them.
    """

    duration_ms = 88_000
    steps_used = 6
    timings = Timings(
        phases={
            "loop_model": Phase(calls=4, seconds=70.0),
            GATE: Phase(calls=1, seconds=0.01),
            VERIFY: Phase(calls=1, seconds=12.0),
            SUFFICIENCY: Phase(calls=1, seconds=9.0),
        }
    )

    class usage:  # noqa: N801 - mirrors the real attribute name
        total = 41_000


async def _scored_attempt(
    session: AsyncSession, *, payload: dict | None = None
) -> tuple[TenantContext, uuid.UUID, InvestigationReport, GateResult, uuid.UUID]:
    """A tenant with one investigation, one evidence row, one audit row, and a report."""
    slug = f"cap-{uuid.uuid4().hex[:6]}"
    tenant_id = uuid.uuid4()
    graph_name = graph_name_for_new_tenant(slug, tenant_id)
    session.add(Tenant(id=tenant_id, slug=slug, name=slug, graph_name=graph_name))
    await session.flush()

    scenario = SCENARIOS[0]
    investigation = Investigation(
        tenant_id=tenant_id, question=scenario.question, status=InvestigationStatus.COMPLETED
    )
    session.add(investigation)
    await session.flush()

    body = payload or {"rows": [{"day": "2026-06-17", "value": 91}]}
    evidence = Evidence(
        tenant_id=tenant_id,
        investigation_id=investigation.id,
        tool_name="posthog",
        capability="event_trend",
        params={"event": "user signed up"},
        payload=body,
        payload_hash=canonical_hash(body),
        source_ref="posthog://projects/1/trend",
        from_cache=False,
    )
    session.add(evidence)
    await session.flush()
    session.add(
        ToolCall(
            tenant_id=tenant_id,
            investigation_id=investigation.id,
            evidence_id=evidence.id,
            tool_name="posthog",
            capability="event_trend",
            read_only=True,
            params={"event": "user signed up"},
            succeeded=True,
            duration_ms=120,
        )
    )
    await session.flush()

    report = InvestigationReport(
        question=scenario.question,
        executive_summary=[
            Claim(text="Signups fell after the deploy.", evidence_ids=[evidence.id])
        ],
        confidence=Confidence.MEDIUM,
        sources=[
            Source(
                evidence_id=evidence.id,
                tool_name="posthog",
                capability="event_trend",
                source_ref="posthog://projects/1/trend",
            )
        ],
    )
    context = TenantContext(tenant_id=tenant_id, tenant_slug=slug, graph_name=graph_name)
    return context, investigation.id, report, GateResult(report=report), evidence.id


class TestARescoreIsTheSameComputation:
    """The only property that makes a bundle worth having."""

    async def test_every_dimension_matches_the_original(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        tenant, investigation_id, _report, gate, _ = await _scored_attempt(session)
        scorer = Scorer()
        before = await scorer.score(
            session,
            tenant,
            investigation_id=investigation_id,
            scenario=SCENARIOS[0],
            investigation=_Investigation(),
            gate_result=gate,
            verification=None,
        )
        path = await write_bundle(
            session,
            tenant,
            investigation_id=investigation_id,
            scenario=SCENARIOS[0].name,
            attempt=1,
            investigation=_Investigation(),
            gate_result=gate,
            verification=None,
            directory=tmp_path,
        )

        replayed_tenant, replayed_id, view, gate2, verification2, _s = await replay_bundle(
            session, load_bundle(path)
        )
        after = await scorer.score(
            session,
            replayed_tenant,
            investigation_id=replayed_id,
            scenario=SCENARIOS[0],
            investigation=view,
            gate_result=gate2,
            verification=verification2,
        )

        assert {d.name: d.score for d in after.dimensions} == {
            d.name: d.score for d in before.dimensions
        }
        assert after.overall == before.overall

    async def test_grounding_survives_the_round_trip(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        """The dimension most likely to break silently. It re-hashes each payload and compares,
        so a bundle that dropped or altered a payload would score zero rather than error."""
        tenant, investigation_id, _report, gate, _ = await _scored_attempt(session)
        path = await write_bundle(
            session,
            tenant,
            investigation_id=investigation_id,
            scenario=SCENARIOS[0].name,
            attempt=1,
            investigation=_Investigation(),
            gate_result=gate,
            verification=None,
            directory=tmp_path,
        )
        replayed_tenant, replayed_id, view, gate2, _, _s = await replay_bundle(
            session, load_bundle(path)
        )
        card = await Scorer().score(
            session,
            replayed_tenant,
            investigation_id=replayed_id,
            scenario=SCENARIOS[0],
            investigation=view,
            gate_result=gate2,
            verification=None,
        )
        grounding = next(d for d in card.dimensions if d.name == "grounding")
        assert grounding.score == 1.0

    async def test_the_payload_hash_is_carried_not_recomputed(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        """Recomputing on replay would make the grounding check tautological: it would pass even
        for a payload the bundle had corrupted in transit."""
        tenant, investigation_id, _r, gate, _ = await _scored_attempt(session)
        path = await write_bundle(
            session,
            tenant,
            investigation_id=investigation_id,
            scenario=SCENARIOS[0].name,
            attempt=1,
            investigation=_Investigation(),
            gate_result=gate,
            verification=None,
            directory=tmp_path,
        )
        bundle = load_bundle(path)
        # Corrupt the payload, leave the hash. A recomputing replay would not notice.
        bundle.evidence[0]["payload"] = {"rows": [{"day": "2026-06-17", "value": 9999}]}
        replayed_tenant, replayed_id, view, gate2, _, _s = await replay_bundle(session, bundle)
        card = await Scorer().score(
            session,
            replayed_tenant,
            investigation_id=replayed_id,
            scenario=SCENARIOS[0],
            investigation=view,
            gate_result=gate2,
            verification=None,
        )
        grounding = next(d for d in card.dimensions if d.name == "grounding")
        assert grounding.score == 0.0, "a corrupted payload must fail its hash check"


class TestABundleCanBeReplayedMoreThanOnce:
    """Keeping the original evidence ids was the obvious choice and it was wrong.

    A bundle could then be replayed exactly once, into a database that had never seen it.
    Replaying it twice, or alongside the run that produced it, collides on `evidence_pkey`.
    """

    async def test_replaying_beside_the_original_does_not_collide(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        tenant, investigation_id, _r, gate, original_evidence = await _scored_attempt(session)
        path = await write_bundle(
            session,
            tenant,
            investigation_id=investigation_id,
            scenario=SCENARIOS[0].name,
            attempt=1,
            investigation=_Investigation(),
            gate_result=gate,
            verification=None,
            directory=tmp_path,
        )
        first, first_id, _v1, _g1, _, _s1 = await replay_bundle(session, load_bundle(path))
        second, second_id, _v2, _g2, _, _s2 = await replay_bundle(session, load_bundle(path))
        await session.flush()

        assert first.tenant_id != second.tenant_id
        assert first_id != second_id
        # And the original row is untouched.
        assert (
            await session.execute(select(Evidence).where(Evidence.id == original_evidence))
        ).scalar_one() is not None

    async def test_citations_are_rewritten_to_the_new_ids(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        """Re-minting ids without rewriting the citations would leave every claim pointing at an
        id that no longer exists, and grounding would score zero for a perfectly grounded
        report."""
        tenant, investigation_id, _r, gate, original_evidence = await _scored_attempt(session)
        path = await write_bundle(
            session,
            tenant,
            investigation_id=investigation_id,
            scenario=SCENARIOS[0].name,
            attempt=1,
            investigation=_Investigation(),
            gate_result=gate,
            verification=None,
            directory=tmp_path,
        )
        _t, replayed_id, _v, gate2, _, _s = await replay_bundle(session, load_bundle(path))
        cited = gate2.report.executive_summary[0].evidence_ids[0]
        assert cited != original_evidence
        rows = (
            (
                await session.execute(
                    select(Evidence).where(Evidence.investigation_id == replayed_id)
                )
            )
            .scalars()
            .all()
        )
        assert cited in {row.id for row in rows}


class TestWhatABundleCarries:
    async def test_the_gate_and_verifier_verdicts_survive(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        """`draft_reliability` scores the share of drafted claims that survived review, so a
        bundle that dropped the rejections would score a filtered report as a clean one."""
        tenant, investigation_id, report, _g, _ = await _scored_attempt(session)
        gate = GateResult(
            report=report,
            rejections=[
                Rejection(
                    location="executive_summary[1]",
                    reason=RejectionReason.UNKNOWN_EVIDENCE,
                    detail="not in this investigation",
                    text="Marketing spend was cut.",
                )
            ],
        )
        path = await write_bundle(
            session,
            tenant,
            investigation_id=investigation_id,
            scenario=SCENARIOS[0].name,
            attempt=1,
            investigation=_Investigation(),
            gate_result=gate,
            verification=None,
            directory=tmp_path,
        )
        bundle = load_bundle(path)
        assert len(bundle.gate_rejections) == 1
        _t, _i, _v, gate2, _, _s = await replay_bundle(session, bundle)
        assert len(gate2.rejections) == 1
        assert gate2.rejections[0].reason is RejectionReason.UNKNOWN_EVIDENCE
        assert gate2.rejections[0].text == "Marketing spend was cut."

    async def test_an_unverified_run_replays_as_unverified(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        """`--no-verify` runs one of the two grounding mechanisms only, and the report says so.
        A replay that invented a verification would score it as though both had run."""
        tenant, investigation_id, _r, gate, _ = await _scored_attempt(session)
        path = await write_bundle(
            session,
            tenant,
            investigation_id=investigation_id,
            scenario=SCENARIOS[0].name,
            attempt=1,
            investigation=_Investigation(),
            gate_result=gate,
            verification=None,
            directory=tmp_path,
        )
        _t, _i, _v, _g, verification, _s = await replay_bundle(session, load_bundle(path))
        assert verification is None

    async def test_the_audit_rows_travel(self, session: AsyncSession, tmp_path: Path) -> None:
        """`tool_selection` reads the audit trail, not the report, so a bundle without the
        calls would score every attempt as having used no tools."""
        tenant, investigation_id, _r, gate, _ = await _scored_attempt(session)
        path = await write_bundle(
            session,
            tenant,
            investigation_id=investigation_id,
            scenario=SCENARIOS[0].name,
            attempt=1,
            investigation=_Investigation(),
            gate_result=gate,
            verification=None,
            directory=tmp_path,
        )
        _t, replayed_id, _v, _g, _, _s = await replay_bundle(session, load_bundle(path))
        await session.flush()
        calls = (
            (
                await session.execute(
                    select(ToolCall).where(ToolCall.investigation_id == replayed_id)
                )
            )
            .scalars()
            .all()
        )
        assert [(c.tool_name, c.capability) for c in calls] == [("posthog", "event_trend")]

    async def test_latency_and_tokens_travel(self, session: AsyncSession, tmp_path: Path) -> None:
        tenant, investigation_id, _r, gate, _ = await _scored_attempt(session)
        path = await write_bundle(
            session,
            tenant,
            investigation_id=investigation_id,
            scenario=SCENARIOS[0].name,
            attempt=1,
            investigation=_Investigation(),
            gate_result=gate,
            verification=None,
            directory=tmp_path,
        )
        _t, _i, view, _g, _, _s = await replay_bundle(session, load_bundle(path))
        assert view.duration_ms == 88_000
        assert view.usage.total == 41_000
        assert view.steps_used == 6


class TestAnUnreadableBundleIsRefused:
    def test_a_version_mismatch_raises_rather_than_guessing(self, tmp_path: Path) -> None:
        """A mis-read bundle produces a scorecard that looks like a measurement and is not,
        which is worse than an error telling someone to re-run."""
        path = tmp_path / "stale.json"
        path.write_text('{"version": 999, "scenario": "x", "attempt": 1}')
        with pytest.raises(BundleVersionMismatch, match="Re-run the eval"):
            load_bundle(path)

    def test_the_current_version_is_readable(self, tmp_path: Path) -> None:
        assert BUNDLE_VERSION == 1


class TestARejectedAttemptIsStillCaptured:
    """The most interesting failures used to be the only ones leaving no trace.

    Run 15 refused two of ten attempts outright -- "no claim in the executive summary supported
    by its cited evidence, on two independent readings" -- and the drafts behind them were gone
    before anyone could ask why. A rejection rate could be counted and not investigated.
    """

    async def test_a_rejected_bundle_records_the_reason(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        tenant, investigation_id, report, gate, _ = await _scored_attempt(session)
        path = await write_bundle(
            session,
            tenant,
            investigation_id=investigation_id,
            scenario=SCENARIOS[0].name,
            attempt=1,
            investigation=_Investigation(),
            gate_result=gate,
            verification=None,
            directory=tmp_path,
            rejected_because="report rejected by the verifier: nothing survived",
        )
        bundle = load_bundle(path)
        assert bundle.rejected_because == "report rejected by the verifier: nothing survived"

    async def test_the_refused_draft_travels_with_it(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        """The draft is the whole point -- knowing a report was refused says nothing about why,
        and the sentence the verifier objected to is the first thing anyone wants."""
        tenant, investigation_id, report, gate, _ = await _scored_attempt(session)
        path = await write_bundle(
            session,
            tenant,
            investigation_id=investigation_id,
            scenario=SCENARIOS[0].name,
            attempt=1,
            investigation=_Investigation(),
            gate_result=gate,
            verification=None,
            directory=tmp_path,
            rejected_because="refused",
        )
        bundle = load_bundle(path)
        assert bundle.report["executive_summary"][0]["text"] == ("Signups fell after the deploy.")
        assert bundle.evidence, "the evidence rows must travel too, or the draft is unreadable"

    async def test_a_delivered_bundle_carries_no_rejection(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        tenant, investigation_id, _r, gate, _ = await _scored_attempt(session)
        path = await write_bundle(
            session,
            tenant,
            investigation_id=investigation_id,
            scenario=SCENARIOS[0].name,
            attempt=1,
            investigation=_Investigation(),
            gate_result=gate,
            verification=None,
            directory=tmp_path,
        )
        assert load_bundle(path).rejected_because is None

    async def test_an_older_bundle_loads_without_the_field(self, tmp_path: Path) -> None:
        """Bundles written before rejections were captured stay readable: absent means
        delivered, which is what they were."""
        import json

        path = tmp_path / "old.json"
        path.write_text(
            json.dumps(
                {
                    "version": BUNDLE_VERSION,
                    "scenario": "onboarding_regression",
                    "attempt": 1,
                    "duration_ms": 1,
                    "tokens": 1,
                    "steps_used": 1,
                    "report": {},
                    "evidence": [],
                    "tool_calls": [],
                    "gate_rejections": [],
                    "verdicts": [],
                    "unverified": [],
                    "verified": True,
                }
            )
        )
        assert load_bundle(path).rejected_because is None


class TestTheSummaryTravelsForVerifierPrecision:
    """`verifier_precision` asks whether the *delivered summary* still names the planted cause.

    That makes it the first dimension to read `verification.report.executive_summary` during
    scoring rather than the verdicts alone. The equality test above compares every dimension, but
    its stub carries no unsupported verdict, so the branch that can score below 1.00 was never
    exercised through a bundle — and a replay that lost the summary would report a clean 1.00
    forever, which is the failure mode this project keeps finding in its own safeguards.
    """

    async def test_a_bundle_reproduces_a_penalised_reading(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        from cortex.eval.fixtures import alternatives_of, by_name
        from cortex.reports.verifier import ClaimVerdict, VerificationResult
        from cortex.reports.verifier import Verdict as ClaimVerdictKind

        scenario = by_name("campaign_traffic_drop")
        signal = alternatives_of(scenario.ground_truth.required_signals[0])[0]

        tenant, investigation_id, report, gate, evidence_id = await _scored_attempt(session)
        # A summary that does NOT name the cause, and a removed claim that does: the run-32
        # shape, where every removal was correct and the answer was poorer for it.
        delivered = report.model_copy(
            update={
                "executive_summary": [
                    Claim(
                        text="Paid search session volume collapsed in the second half of June.",
                        evidence_ids=[evidence_id],
                    )
                ]
            }
        )
        verification = VerificationResult(
            report=delivered,
            verdicts=[
                ClaimVerdict(
                    location="findings[0].claims[0]",
                    claim_text=f"The {signal} ending drove the fall, one day before it began.",
                    verdict=ClaimVerdictKind.UNSUPPORTED,
                    reason="the cited evidence does not establish when the fall began",
                )
            ],
        )

        scorer = Scorer()
        before = await scorer.score(
            session,
            tenant,
            investigation_id=investigation_id,
            scenario=scenario,
            investigation=_Investigation(),
            gate_result=gate,
            verification=verification,
        )
        path = await write_bundle(
            session,
            tenant,
            investigation_id=investigation_id,
            scenario=scenario.name,
            attempt=1,
            investigation=_Investigation(),
            gate_result=gate,
            verification=verification,
            directory=tmp_path,
        )
        replayed_tenant, replayed_id, view, gate2, verification2, _s = await replay_bundle(
            session, load_bundle(path)
        )
        after = await scorer.score(
            session,
            replayed_tenant,
            investigation_id=replayed_id,
            scenario=scenario,
            investigation=view,
            gate_result=gate2,
            verification=verification2,
        )

        live = {d.name: d.score for d in before.dimensions}
        replay = {d.name: d.score for d in after.dimensions}
        # The reading is penalised, and it survives the round trip rather than reading clean.
        assert live["verifier_precision"] == 0.0
        assert replay["verifier_precision"] == live["verifier_precision"]
