"""Capture an eval attempt so it can be re-scored without paying for it again.

**Why this exists.** `cortex/eval/runner.py` builds each report in memory, scores it, prints a
scorecard, and then `cortex/eval/__main__.py` rolls the whole transaction back -- deliberately,
because an eval run must not leave tenants and evidence behind in whatever database it was
pointed at. The consequence nobody had noticed is that *nothing survives*: not the report, not
the evidence, not the audit rows. So every change to a scoring dimension costs a full re-run
against the live provider, paid in API spend and exposed to the between-scenario variance that
dominates this harness (sigma_sc = 0.05 against sigma_att = 0.0145).

That is how the accuracy figures in `docs/eval-results.md` became upper bounds rather than
measurements: `_accuracy` was searching a surface that included contradicted hypotheses, the
surface was fixed, and re-measuring meant re-running.

**The design, and the one property that matters.** A bundle carries everything the scorer reads
-- the delivered report, the evidence rows with their hashes, the audit rows, the gate's and
verifier's verdicts, and the investigation's own timing -- and re-scoring **replays those rows
into a scratch tenant and runs the unmodified `Scorer`**. Not a reimplementation of scoring over
a file. The whole point is that a re-score is the same computation as the original score; a
second scoring path would drift from the first and then the numbers would be incomparable, which
is the problem this is meant to solve rather than a new one to introduce.

No migration and no schema change: a bundle is a file, and replay writes to the same tables the
run would have.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import Evidence, Investigation, InvestigationStatus, Tenant, ToolCall
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.reports.gate import GateResult, Rejection, RejectionReason
from cortex.reports.schema import InvestigationReport
from cortex.reports.sufficiency import AppliedSufficiency
from cortex.reports.verifier import ClaimVerdict, VerificationResult
from cortex.reports.verifier import Verdict as ClaimVerdictKind
from cortex.tenancy.context import TenantContext

#: Bumped when a bundle's shape changes in a way that makes an older one unreadable.
#:
#: Refused rather than migrated. A silently mis-read bundle would produce a scorecard that looks
#: like a measurement and is not, which is worse than an error telling someone to re-run.
BUNDLE_VERSION = 1

__all__ = [
    "Bundle",
    "BundleVersionMismatch",
    "load_bundle",
    "replay_bundle",
    "write_bundle",
    "write_failure",
    "FAILURES_DIRNAME",
]


class BundleVersionMismatch(Exception):
    """A bundle written by a different version of this module."""


@dataclass(slots=True)
class Bundle:
    """One scored attempt, in a form that can be replayed.

    Everything here is what the *scorer* consumes, not everything the run produced. The model
    transcript, the prompts and the intermediate drafts are deliberately absent: they are large,
    they are the expensive part to store, and no dimension reads them.
    """

    scenario: str
    attempt: int
    report: dict[str, Any]
    #: `duration_ms` and token totals, which the latency dimension scores.
    duration_ms: int
    tokens: int
    steps_used: int
    #: Per-phase seconds, calls and output tokens, so "where did the 121 seconds go" is
    #: answerable from a bundle rather than only by re-running under `cortex.ask`.
    phases: dict[str, dict[str, float]] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    gate_rejections: list[dict[str, str]] = field(default_factory=list)
    sufficiency_rejections: list[dict[str, str]] = field(default_factory=list)
    verdicts: list[dict[str, str]] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    verified: bool = True
    #: Why this attempt was refused, when it was. None on a delivered report.
    #:
    #: A rejected attempt used to write no bundle at all, which made the most interesting
    #: failures the only ones that left no trace -- a 20% total-rejection rate could be counted
    #: and not looked at. The report captured here is the *draft* that was refused, not an
    #: answer anybody received, and `rescore` reports it as errored rather than scoring it.
    rejected_because: str | None = None
    version: int = BUNDLE_VERSION

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "scenario": self.scenario,
                "attempt": self.attempt,
                "duration_ms": self.duration_ms,
                "tokens": self.tokens,
                "steps_used": self.steps_used,
                "phases": self.phases,
                "report": self.report,
                "evidence": self.evidence,
                "tool_calls": self.tool_calls,
                "gate_rejections": self.gate_rejections,
                "sufficiency_rejections": self.sufficiency_rejections,
                "verdicts": self.verdicts,
                "unverified": self.unverified,
                "verified": self.verified,
                "rejected_because": self.rejected_because,
            },
            indent=1,
            sort_keys=True,
        )


async def write_bundle(
    session: AsyncSession,
    tenant: TenantContext,
    *,
    investigation_id: uuid.UUID,
    scenario: str,
    attempt: int,
    investigation: Any,
    gate_result: GateResult,
    verification: VerificationResult | None,
    directory: Path,
    sufficiency: AppliedSufficiency | None = None,
    rejected_because: str | None = None,
) -> Path:
    """Capture an attempt to `directory`, returning the file written.

    Called before the run's transaction is rolled back, which is the only moment the evidence
    rows exist. Reads them back out of the session rather than reconstructing them from the
    executor's return values, so that what is captured is what was *stored* -- including the
    payload hash the grounding dimension re-computes.
    """
    # The report as delivered, after all three filters. Sufficiency runs last, so its output is
    # the one a reader saw -- capturing the verifier's would replay a report that was never sent.
    delivered = (
        sufficiency.report
        if sufficiency is not None
        else (verification.report if verification else gate_result.report)
    )

    evidence_rows = (
        (
            await session.execute(
                Evidence.__table__.select().where(
                    Evidence.tenant_id == tenant.tenant_id,
                    Evidence.investigation_id == investigation_id,
                )
            )
        )
        .mappings()
        .all()
    )
    call_rows = (
        (
            await session.execute(
                ToolCall.__table__.select().where(
                    ToolCall.tenant_id == tenant.tenant_id,
                    ToolCall.investigation_id == investigation_id,
                )
            )
        )
        .mappings()
        .all()
    )

    bundle = Bundle(
        scenario=scenario,
        attempt=attempt,
        report=json.loads(delivered.model_dump_json()),
        duration_ms=int(getattr(investigation, "duration_ms", 0) or 0),
        tokens=int(getattr(getattr(investigation, "usage", None), "total", 0) or 0),
        steps_used=int(getattr(investigation, "steps_used", 0) or 0),
        phases=_phase_json(getattr(investigation, "timings", None)),
        evidence=[_evidence_json(row) for row in evidence_rows],
        tool_calls=[_call_json(row) for row in call_rows],
        gate_rejections=[r.as_dict() for r in gate_result.rejections],
        # Kept separately from the verifier's, because they answer different questions: the
        # verifier judged a claim against its own citations, the sufficiency gate judged whether
        # the evidence could carry a cause at all. Merging them would make `draft_reliability`
        # unable to say which mechanism removed what.
        sufficiency_rejections=[
            r.as_dict() for r in (sufficiency.rejections if sufficiency else [])
        ],
        verdicts=[_verdict_json(v) for v in (verification.verdicts if verification else [])],
        unverified=list(verification.unverified) if verification else [],
        verified=verification is not None,
        rejected_because=rejected_because,
    )

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{scenario}-attempt{attempt}.json"
    path.write_text(bundle.to_json())
    return path


def _phase_json(timings: Any) -> dict[str, dict[str, float]]:
    """Per-phase seconds, calls and tokens, flattened for a bundle.

    Empty when an investigation carried no clock, which is how a replayed bundle looks: replay
    reconstructs the three fields the scorer reads and does not re-run the phases, so there is
    nothing to time. Returning `{}` rather than zeros keeps "not measured" distinguishable from
    "measured as instant".
    """
    phases = getattr(timings, "phases", None)
    if not phases:
        return {}
    return {
        name: {
            "seconds": round(phase.seconds, 2),
            "calls": phase.calls,
            "output_tokens": getattr(phase.usage, "output_tokens", 0),
        }
        for name, phase in phases.items()
    }


def load_bundle(path: Path) -> Bundle:
    raw = json.loads(path.read_text())
    version = raw.get("version")
    if version != BUNDLE_VERSION:
        raise BundleVersionMismatch(
            f"{path.name} was written by bundle version {version!r}; this is "
            f"{BUNDLE_VERSION}. Re-run the eval rather than re-scoring a bundle whose shape "
            "this code does not understand -- a mis-read bundle produces a scorecard that "
            "looks like a measurement and is not."
        )
    return Bundle(
        scenario=raw["scenario"],
        attempt=raw["attempt"],
        report=raw["report"],
        duration_ms=raw["duration_ms"],
        tokens=raw["tokens"],
        steps_used=raw["steps_used"],
        # Absent from bundles written before phases were timed in the eval. Defaulted rather
        # than refused: an older bundle is a valid measurement of everything except this.
        phases=raw.get("phases", {}),
        evidence=raw["evidence"],
        tool_calls=raw["tool_calls"],
        gate_rejections=raw["gate_rejections"],
        # Absent from bundles written before the sufficiency gate existed. Defaulted rather than
        # version-bumped: an older bundle is still fully readable, it simply records a pipeline
        # with one fewer filter, and refusing it would throw away a measurement for nothing.
        sufficiency_rejections=raw.get("sufficiency_rejections", []),
        verdicts=raw["verdicts"],
        unverified=raw["unverified"],
        verified=raw["verified"],
        rejected_because=raw.get("rejected_because"),
        version=version,
    )


async def replay_bundle(
    session: AsyncSession, bundle: Bundle
) -> tuple[
    TenantContext,
    uuid.UUID,
    Investigation,
    GateResult,
    VerificationResult | None,
    AppliedSufficiency,
]:
    """Insert a bundle's rows into a fresh scratch tenant, ready for the real `Scorer`.

    A new tenant per replay, for the same reason the run itself provisions one per scenario:
    scoring reads the evidence store and the audit trail by tenant, so sharing one would let a
    previous replay's rows inflate this one's grounding and tool-selection scores.
    """
    slug = f"replay-{bundle.scenario.replace('_', '-')}-{uuid.uuid4().hex[:6]}"
    tenant_id = uuid.uuid4()
    graph_name = graph_name_for_new_tenant(slug, tenant_id)
    session.add(Tenant(id=tenant_id, slug=slug, name=bundle.scenario, graph_name=graph_name))
    await session.flush()

    # Evidence ids are re-minted, and every citation is rewritten through the same mapping.
    #
    # Keeping the original ids was the obvious choice and it is wrong: a bundle could then be
    # replayed exactly once, into a database that had never seen it. Replaying it twice, or
    # alongside the run that produced it, collides on `evidence_pkey`. Re-minting costs nothing
    # -- the grounding dimension asks whether a citation *resolves*, never what its value is --
    # and the rewrite is done over the raw JSON rather than over the parsed model so that no
    # section can be missed. A new citable section added to the report schema later would
    # otherwise silently keep its stale ids and score as ungrounded.
    remap = {row["id"]: str(uuid.uuid4()) for row in bundle.evidence}
    report = InvestigationReport.model_validate(_rewrite_ids(bundle.report, remap))
    investigation = Investigation(
        tenant_id=tenant_id,
        question=report.question,
        status=InvestigationStatus.COMPLETED,
    )
    session.add(investigation)
    await session.flush()

    for row in bundle.evidence:
        session.add(
            Evidence(
                id=uuid.UUID(remap[row["id"]]),
                tenant_id=tenant_id,
                investigation_id=investigation.id,
                tool_name=row["tool_name"],
                capability=row["capability"],
                params=row["params"],
                payload=row["payload"],
                # Carried, not recomputed. The grounding dimension re-hashes the payload and
                # compares; recomputing here would make that check tautological and it would
                # pass even for a payload this bundle had corrupted in transit.
                payload_hash=row["payload_hash"],
                source_ref=row["source_ref"],
                from_cache=row["from_cache"],
                observed_at=_moment(row["observed_at"]),
            )
        )
    for row in bundle.tool_calls:
        session.add(
            ToolCall(
                tenant_id=tenant_id,
                investigation_id=investigation.id,
                evidence_id=(
                    uuid.UUID(remap[row["evidence_id"]])
                    if row["evidence_id"] and row["evidence_id"] in remap
                    else None
                ),
                tool_name=row["tool_name"],
                capability=row["capability"],
                read_only=row["read_only"],
                params=row["params"],
                succeeded=row["succeeded"],
                error=row["error"],
                duration_ms=row["duration_ms"],
            )
        )
    await session.flush()

    # The scorer reads `duration_ms`, `usage.total` and `steps_used` off the investigation
    # object rather than off the row, so a stand-in carries them.
    investigation_view = _Replayed(
        duration_ms=bundle.duration_ms,
        tokens=bundle.tokens,
        steps_used=bundle.steps_used,
        report=report,
    )

    gate_result = GateResult(
        report=report,
        rejections=[_rejection(entry) for entry in bundle.gate_rejections],
    )
    verification = (
        VerificationResult(
            report=report,
            verdicts=[_verdict(entry) for entry in bundle.verdicts],
            unverified=list(bundle.unverified),
        )
        if bundle.verified
        else None
    )
    # Reconstructed so `veto_precision` can be re-scored from a bundle. The report carried here
    # is the delivered one, which is what the dimension needs -- it reads the withheld claims,
    # not the surviving report.
    applied = AppliedSufficiency(
        report=report,
        rejections=[_rejection(entry) for entry in bundle.sufficiency_rejections],
        withheld=len(bundle.sufficiency_rejections),
    )
    context = TenantContext(tenant_id=tenant_id, tenant_slug=slug, graph_name=graph_name)
    return context, investigation.id, investigation_view, gate_result, verification, applied


def _rewrite_ids(value: Any, remap: dict[str, str]) -> Any:
    """Replace every occurrence of a remapped evidence id, anywhere in the structure.

    A total traversal rather than a per-section rewrite. The report cites evidence from at least
    eight places today -- summary claims, finding claims, hypotheses on both sides, chart series
    and annotations, recommendations, risks, data-quality notes and the sources table -- and a
    rewrite that enumerated them would go stale the first time a ninth is added, silently, with
    the new section's citations left pointing at ids that no longer exist.
    """
    if isinstance(value, str):
        return remap.get(value, value)
    if isinstance(value, list):
        return [_rewrite_ids(item, remap) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_ids(item, remap) for key, item in value.items()}
    return value


@dataclass(slots=True)
class _Usage:
    total: int


@dataclass(slots=True)
class _Replayed:
    """The three investigation fields the scorer reads, and nothing else."""

    duration_ms: int
    tokens: int
    steps_used: int
    report: InvestigationReport

    @property
    def usage(self) -> _Usage:
        return _Usage(total=self.tokens)


def _evidence_json(row: Any) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "tool_name": row["tool_name"],
        "capability": row["capability"],
        "params": row["params"],
        "payload": row["payload"],
        "payload_hash": row["payload_hash"],
        "source_ref": row["source_ref"],
        "from_cache": row["from_cache"],
        "observed_at": row["observed_at"].isoformat() if row["observed_at"] else None,
    }


def _call_json(row: Any) -> dict[str, Any]:
    return {
        "tool_name": row["tool_name"],
        "capability": row["capability"],
        "read_only": row["read_only"],
        "params": row["params"],
        "succeeded": row["succeeded"],
        "error": row["error"],
        "duration_ms": row["duration_ms"],
        "evidence_id": str(row["evidence_id"]) if row["evidence_id"] else None,
    }


def _verdict_json(verdict: ClaimVerdict) -> dict[str, str]:
    return {
        "location": verdict.location,
        "claim_text": verdict.claim_text,
        "verdict": verdict.verdict.value,
        "reason": verdict.reason,
    }


def _verdict(entry: dict[str, str]) -> ClaimVerdict:
    return ClaimVerdict(
        location=entry["location"],
        claim_text=entry["claim_text"],
        verdict=ClaimVerdictKind(entry["verdict"]),
        reason=entry["reason"],
    )


def _rejection(entry: dict[str, str]) -> Rejection:
    return Rejection(
        location=entry["location"],
        reason=RejectionReason(entry["reason"]),
        detail=entry["detail"],
        text=entry.get("text", ""),
    )


def _moment(value: str | None) -> datetime:
    return datetime.fromisoformat(value) if value else datetime.now(UTC)


#: Where failed attempts go, kept out of the bundle namespace on purpose -- see `write_failure`.
FAILURES_DIRNAME = "failures"


async def write_failure(
    session: AsyncSession,
    tenant: TenantContext,
    *,
    investigation_id: uuid.UUID,
    scenario: str,
    attempt: int,
    error: str,
    directory: Path,
) -> Path:
    """Record an attempt that never produced a report, so the failure can be examined.

    **Why this exists.** Run 18 lost both `insufficient_evidence` attempts to a drafting
    failure and left nothing behind: a bundle needs a report, and there was no report. The
    only surviving trace was one line of error text, so the first question -- did the loop
    work and the draft blow up, or did the loop wander and gather so much that no draft could
    fit? -- could not be answered from the run at all, only guessed at and then re-run.

    The rows are still in the session here, uncommitted but present, and this is the last
    moment before rollback that they exist.

    Deliberately *not* a `Bundle`. A bundle is replayable and scoreable, and this is neither;
    naming it one would put an unscoreable artifact in the directory the scorer reads. What it
    carries is what the loop did before it died -- every tool call, in order, and the evidence
    minted -- which is the part that distinguishes the two explanations above.

    Written to a `failures/` subdirectory rather than alongside the bundles, because three
    separate places glob `*.json` over a capture directory and hand what they find to
    `load_bundle`. A filename convention would make all three -- and every future one --
    responsible for remembering an exclusion; a subdirectory means they cannot see it.
    """
    call_rows = (
        (
            await session.execute(
                ToolCall.__table__.select()
                .where(
                    ToolCall.tenant_id == tenant.tenant_id,
                    ToolCall.investigation_id == investigation_id,
                )
                .order_by(ToolCall.created_at)
            )
        )
        .mappings()
        .all()
    )
    evidence_rows = (
        (
            await session.execute(
                Evidence.__table__.select().where(
                    Evidence.tenant_id == tenant.tenant_id,
                    Evidence.investigation_id == investigation_id,
                )
            )
        )
        .mappings()
        .all()
    )

    record = {
        "scenario": scenario,
        "attempt": attempt,
        "failed": True,
        "error": error,
        # In call order, because "the loop repeated one tool nine times" and "the loop made
        # nine different calls" are different diagnoses and a bare count cannot tell them apart.
        "tool_calls": [_call_json(row) for row in call_rows],
        "observations": len(evidence_rows),
    }

    failures = directory / FAILURES_DIRNAME
    failures.mkdir(parents=True, exist_ok=True)
    path = failures / f"{scenario}-attempt{attempt}.json"
    path.write_text(json.dumps(record, indent=2, default=str))
    return path
