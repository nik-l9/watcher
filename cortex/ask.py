"""`python -m cortex.ask "why did signups fall?"` — run one investigation and read it.

The point of this module is that a person can use the product. Everything else so far
has been reachable only through the eval harness, which scores investigations rather
than showing them, so the system could be working without anyone having seen it work.

It runs the **production path**: the real `Investigator`, the real `ToolExecutor` writing
real `Evidence` rows, the real grounding gate and the real adversarial verifier.

Two modes, and the difference is only where the observations come from:

  - **default** — connectors serve a labelled fixture. The report is produced exactly as
    a customer's would be, over data whose true causal story is known, so the answer can
    be *checked* rather than admired. `--show-truth` prints the planted cause afterwards.
  - **`--real`** — connectors call real APIs with the credentials stored for a tenant by
    `python -m cortex.connect`. There is no ground truth, so judgement is the reader's,
    and the investigation is committed rather than rolled back because it is an audit
    record. Still read-only: no write capability exists in the registry.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import sys
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from cortex.agents.anthropic_llm import DEFAULT_EFFORT, DEFAULT_MODEL, AnthropicLLM
from cortex.agents.employee import gtm_data_analyst
from cortex.agents.investigator import InvestigationFailed, Investigator
from cortex.agents.progress import Phase, ProgressEvent, TerminalProgress, emit
from cortex.agents.provider import build_llm
from cortex.agents.service import confidence_score, record_usage
from cortex.agents.timing import GATE, VERIFY
from cortex.db.models import Investigation as InvestigationRow
from cortex.db.models import InvestigationStatus, Report, Tenant
from cortex.db.threads import ParentNotFound, ThreadTooDeep, check_parent
from cortex.db.titles import title_for
from cortex.eval.fixtures import SCENARIOS, Scenario, by_name
from cortex.eval.runner import scenario_registry
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.memory.recall import HybridRecall
from cortex.reports.charts import render_chart
from cortex.reports.completeness import CompletenessJudge
from cortex.reports.gate import GroundingGate, ReportRejected
from cortex.reports.schema import InvestigationReport
from cortex.reports.sufficiency import (
    AppliedSufficiency,
    SufficiencyDecision,
    SufficiencyGate,
)
from cortex.reports.verifier import AdversarialVerifier
from cortex.runtime.resources import open_resources
from cortex.tenancy.context import TenantContext
from cortex.tools.executor import ToolExecutor


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cortex.ask",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="What to investigate. Defaults to the chosen dataset's own question.",
    )
    parser.add_argument(
        "--dataset",
        default="campaign_traffic_drop",
        choices=[s.name for s in SCENARIOS],
        help="Which labelled dataset to investigate against. Each has a known cause and "
        "deliberate decoy correlations, so an answer can be checked rather than admired.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--effort",
        default=DEFAULT_EFFORT,
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Thinking depth. Lower is faster and cheaper.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the adversarial verifier. Faster, but one of the two grounding "
        "mechanisms is then not running, and the report says so.",
    )
    parser.add_argument(
        "--no-sufficiency",
        action="store_true",
        help="Skip the sufficiency gate. It asks, in a separate call, whether the evidence "
        "gathered supports a definitive answer about cause, and withholds causal claims "
        "when it does not. Skipping it removes a refusal, not a caveat.",
    )
    parser.add_argument(
        "--show-truth",
        action="store_true",
        help="After the report, print what the dataset's planted cause actually was.",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Investigate REAL data through real connectors, using the credentials stored "
        "for --tenant. Requires `python -m cortex.connect` first. There is no ground "
        "truth to check against, so judgement is yours.",
    )
    parser.add_argument(
        "--tenant",
        default=None,
        help="Tenant slug whose stored credentials to use. Required with --real.",
    )
    parser.add_argument(
        "--follow-up",
        default=None,
        metavar="INVESTIGATION_ID",
        help="Ask this as a follow-up to an earlier investigation. The analyst is given that "
        "investigation's conclusion and an inventory of its evidence, and may cite those "
        "observations directly instead of gathering them again.",
    )
    return parser.parse_args(argv)


async def _provision(session: AsyncSession, scenario: Scenario) -> tuple[TenantContext, uuid.UUID]:
    """A throwaway tenant per run, so nothing leaks between questions."""
    tenant_id = uuid.uuid4()
    # Underscores are legal in scenario names and illegal in tenant slugs, which the
    # naming validator enforces rather than trusting callers to remember.
    stem = scenario.name[:20].replace("_", "-").strip("-")
    slug = f"ask-{stem}-{tenant_id.hex[:6]}"
    graph_name = graph_name_for_new_tenant(slug, tenant_id)
    session.add(Tenant(id=tenant_id, slug=slug, name=slug, graph_name=graph_name))
    await session.flush()

    investigation = InvestigationRow(tenant_id=tenant_id, question=scenario.question)
    session.add(investigation)
    await session.flush()
    return (
        TenantContext(tenant_id=tenant_id, tenant_slug=slug, graph_name=graph_name),
        investigation.id,
    )


def _short(value: uuid.UUID, known: dict[uuid.UUID, str]) -> str:
    """Render a citation as the source it points at, not as a bare uuid.

    A report whose citations read `[a3f2…]` is technically grounded and practically
    unreadable, which defeats the purpose of citing anything.
    """
    label = known.get(value)
    return f"{label} {str(value)[:8]}" if label else str(value)[:8]


def _cites(ids: list[uuid.UUID], known: dict[uuid.UUID, str]) -> str:
    return f"  [{', '.join(_short(i, known) for i in ids)}]" if ids else ""


def render(report: InvestigationReport, *, steps: int, seconds: float, tokens: int) -> str:
    known = {s.evidence_id: f"{s.tool_name}.{s.capability}" for s in report.sources}
    out: list[str] = [
        "",
        "=" * 78,
        f"Q: {report.question}",
        "=" * 78,
        "",
        f"Confidence: {report.confidence.value.replace('_', ' ')}",
        f"Took {seconds:.0f}s over {steps} steps, {tokens:,} tokens, "
        f"{len(report.sources)} observations gathered",
        "",
        "ANSWER",
        "-" * 78,
    ]
    for claim in report.executive_summary:
        out.append(f"  {claim.text}{_cites(claim.evidence_ids, known)}")

    if report.findings:
        out += ["", "FINDINGS", "-" * 78]
        for finding in report.findings:
            out.append(f"  {finding.title}  ({finding.confidence.value})")
            for claim in finding.claims:
                out.append(f"    - {claim.text}{_cites(claim.evidence_ids, known)}")

    if report.charts:
        # Drawn rather than described. A chart nobody can see is a chart nobody checks, and
        # the frontend is deferred — so an off-by-one annotation or a wrong axis becomes
        # obvious here instead of waiting for a UI to reveal it.
        out += ["", "CHARTS", "-" * 78]
        for chart in report.charts:
            out.append(render_chart(chart))
            out.append("")

    if report.hypotheses:
        out += ["", "HYPOTHESES TESTED", "-" * 78]
        for hypothesis in report.hypotheses:
            out.append(f"  [{hypothesis.verdict.value}] {hypothesis.statement}")
            if hypothesis.reasoning:
                out.append(f"      {hypothesis.reasoning}")

    if report.recommendations:
        out += ["", "RECOMMENDATIONS", "-" * 78]
        for recommendation in sorted(report.recommendations, key=lambda r: r.priority):
            out.append(f"  P{recommendation.priority} {recommendation.action}")
            out.append(f"      why: {recommendation.rationale}")
            out.append(f"      {_cites(recommendation.evidence_ids, known).strip()}")

    if report.risks:
        out += ["", "RISKS AND CAVEATS", "-" * 78]
        for risk in report.risks:
            out.append(f"  - {risk.description}")

    if report.data_quality:
        out += ["", "DATA QUALITY", "-" * 78]
        for note in report.data_quality:
            out.append(f"  - {note.note}")

    out += ["", "SOURCES", "-" * 78]
    for source in report.sources:
        ref = f"  {source.source_ref}" if source.source_ref else ""
        out.append(f"  {str(source.evidence_id)[:8]}  {source.tool_name}.{source.capability}{ref}")

    out.append("")
    return "\n".join(out)


async def _existing_tenant(session: AsyncSession, slug: str) -> TenantContext:
    """Load a tenant that already holds credentials. Never creates one.

    Creating it here would silently produce a tenant with no credentials, and the
    investigation would then fail deep inside a tool call with a confusing upstream
    error instead of at the point the mistake was made.
    """
    from sqlalchemy import select

    tenant = await session.scalar(select(Tenant).where(Tenant.slug == slug))
    if tenant is None:
        raise SystemExit(
            f"No tenant {slug!r}. Connect one first:\n"
            f"  python -m cortex.connect --tenant {slug} --provider hubspot --provider github"
        )
    return TenantContext(tenant_id=tenant.id, tenant_slug=tenant.slug, graph_name=tenant.graph_name)


async def _sufficiency(
    llm: AnthropicLLM,
    session: AsyncSession,
    tenant: TenantContext,
    *,
    investigation_id: uuid.UUID,
    question: str,
    report: InvestigationReport,
    timings: Any,
    skip: bool,
) -> tuple[SufficiencyDecision, AppliedSufficiency]:
    """Run the sufficiency gate and print what it cost, in seconds.

    Printed rather than only totalled because the added latency is the number this change
    has to be judged on: the wall clock was 192s against a 90s target before it, and a
    trade paid in seconds is one to make deliberately rather than discover. Attributed to
    the `verify` phase in the breakdown — it is the second half of verification, and giving
    it a phase of its own means editing `cortex/agents/timing.py`, which this change does
    not own.

    May raise `ReportRejected`, when every claim in the summary asserted a cause the
    evidence cannot carry.
    """
    if skip:
        return SufficiencyDecision(needed=False), AppliedSufficiency(report=report)

    started = time.monotonic()
    with timings.measure(VERIFY):
        decision = await SufficiencyGate(llm).assess(
            session,
            tenant,
            investigation_id=investigation_id,
            question=question,
            report=report,
        )
    timings.record_usage(VERIFY, decision.usage)
    elapsed = time.monotonic() - started

    if not decision.needed:
        print(
            "  sufficiency gate: skipped, this report asserts no cause (0.0s)",
            file=sys.stderr,
        )
        return decision, AppliedSufficiency(report=report)
    if not decision.ran:
        print(
            f"  sufficiency gate: could not run ({decision.error}) in {elapsed:.1f}s",
            file=sys.stderr,
        )
    elif decision.sufficient:
        print(
            f"  sufficiency gate: the evidence supports a cause ({elapsed:.1f}s)",
            file=sys.stderr,
        )
    else:
        print(
            f"  sufficiency gate: the evidence does NOT support a cause ({elapsed:.1f}s); "
            f"missing: {decision.missing_sentence}",
            file=sys.stderr,
        )

    applied = decision.apply(report)
    if applied.withheld:
        print(f"  sufficiency gate withheld {applied.withheld} causal claim(s)", file=sys.stderr)
    return decision, applied


async def _main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if args.real:
        return await _ask_real(args)
    scenario = by_name(args.dataset)
    if args.question:
        # The dataset's fixed responses do not change with the question, so a question
        # far from the data will be answered honestly with what the data can support.
        scenario = dataclasses.replace(scenario, question=args.question)

    llm = build_llm(model=args.model, effort=args.effort)
    registry = scenario_registry(scenario)
    investigator = Investigator(
        llm=llm,
        registry=registry,
        executor=ToolExecutor(registry),
        employee=gtm_data_analyst(),
    )

    print(f"Investigating: {scenario.question}", file=sys.stderr)
    print(
        f"Dataset: {args.dataset} (synthetic, labelled)  model: {args.model}  "
        f"effort: {args.effort}",
        file=sys.stderr,
    )
    print("This takes a few minutes. Progress:", file=sys.stderr)

    async with open_resources() as resources:
        maker = async_sessionmaker(resources.engine, expire_on_commit=False)
        async with maker() as session:
            tenant, investigation_id = await _provision(session, scenario)
            try:
                investigation = await investigator.investigate(
                    session, tenant, investigation_id=investigation_id, question=scenario.question
                )
            except InvestigationFailed as exc:
                print(f"\nThe investigation could not complete: {exc}", file=sys.stderr)
                return 1
            print(
                f"  gathered {len(investigation.evidence_ids)} observations in "
                f"{investigation.duration_ms / 1000:.0f}s; grounding the report",
                file=sys.stderr,
            )

            try:
                with investigation.timings.measure(GATE):
                    gated = await GroundingGate().apply(
                        session,
                        tenant,
                        investigation_id=investigation_id,
                        report=investigation.report,
                    )
            except ReportRejected as exc:
                print(f"\nThe report was rejected by the grounding gate: {exc}", file=sys.stderr)
                return 1

            report = gated.report
            if gated.rejections:
                print(
                    f"  gate removed {len(gated.rejections)} unsupported citation(s)",
                    file=sys.stderr,
                )

            if not args.no_verify:
                print("  verifying each claim against its own evidence", file=sys.stderr)
                with investigation.timings.measure(VERIFY):
                    verification = await AdversarialVerifier(llm).verify(
                        session, tenant, investigation_id=investigation_id, report=report
                    )
                investigation.timings.record_usage(VERIFY, verification.usage)
                report = verification.report
                if verification.unsupported_count:
                    print(
                        f"  verifier removed {verification.unsupported_count} claim(s)",
                        file=sys.stderr,
                    )
                if verification.fresh_context_checks:
                    print(
                        f"  {verification.fresh_context_checks} causal claim(s) re-derived "
                        "from their evidence without the claim in view",
                        file=sys.stderr,
                    )

            try:
                _, applied = await _sufficiency(
                    llm,
                    session,
                    tenant,
                    investigation_id=investigation_id,
                    question=scenario.question,
                    report=report,
                    timings=investigation.timings,
                    skip=args.no_sufficiency,
                )
            except ReportRejected as exc:
                await session.rollback()
                print(f"\nThe report was refused: {exc}", file=sys.stderr)
                return 1
            report = applied.report

            # Rolled back deliberately: `ask` is for reading an answer, not for leaving
            # a tenant and its evidence behind in whatever database it was pointed at.
            await session.rollback()

    sys.stdout.write(
        render(
            report,
            steps=len(investigation.steps),
            seconds=investigation.duration_ms / 1000,
            tokens=investigation.usage.total,
        )
    )

    print(
        # Rendered without an explicit wall figure so the collector uses its own
        # start: `investigation.duration_ms` stops when the loop returns, which
        # leaves gate and verify outside the denominator and makes every share wrong.
        "\n" + investigation.timings.render(),
        file=sys.stderr,
    )

    if args.show_truth:
        truth: Any = scenario.ground_truth
        sys.stdout.write(
            "\n".join(
                [
                    "WHAT THE DATA ACTUALLY CONTAINED",
                    "-" * 78,
                    f"  planted cause: {truth.cause}",
                    f"  decoys placed to mislead: {', '.join(truth.decoys) or 'none'}",
                    f"  signals a correct answer names: "
                    f"{', '.join(truth.required_signals) or 'none'}",
                    "",
                ]
            )
        )
    return 0


async def _ask_real(args: argparse.Namespace) -> int:
    """Investigate real data through real connectors.

    Structurally the same as the fixture path — same loop, same evidence writes, same
    gate, same verifier — with two differences that matter. The registry is the
    production one, so a tool call reaches a real API and needs a stored credential; and
    the investigation is **committed** rather than rolled back, because a real
    investigation is an audit record, and its evidence is what its citations resolve to.
    """
    if not args.tenant:
        sys.stderr.write("--real needs --tenant, whose stored credentials will be used.\n")
        return 2
    if not args.question:
        sys.stderr.write("--real needs a question; there is no fixture question to fall back on.\n")
        return 2

    from cortex.tools.registry import NoToolsAvailable, registry_for_tenant

    llm = build_llm(model=args.model, effort=args.effort)

    print(f"Investigating REAL data for tenant {args.tenant}: {args.question}", file=sys.stderr)
    print("Read-only: no write capability exists in the tool registry.", file=sys.stderr)

    async with open_resources() as resources:
        # Memory is offered on real data and withheld on fixtures, because on a fixture
        # there is nothing ingested and a recall step that always returns "nothing
        # recalled" would add a call to every eval run for no information.
        # Progress to stderr, so stdout stays the report and stays pipeable. Without it a
        # 90-second investigation prints nothing and looks hung -- which is also how a real
        # hang looked, twice, during development.
        reporter = TerminalProgress()
        maker = async_sessionmaker(resources.engine, expire_on_commit=False)
        async with maker() as session:
            tenant = await _existing_tenant(session, args.tenant)

            # Built after the tenant is known, because which capabilities exist is a property
            # of the tenant: offering one whose credential is absent costs a step, returns
            # CredentialMissing, and leaves an irrelevant caveat in the report.
            try:
                registry = await registry_for_tenant(session, tenant)
            except NoToolsAvailable as exc:
                sys.stderr.write(f"{exc}\n")
                return 2
            print(
                f"Capabilities available to {tenant.tenant_slug}: {', '.join(registry.tool_names)}",
                file=sys.stderr,
            )

            investigator = Investigator(
                llm=llm,
                registry=registry,
                executor=ToolExecutor(registry, limiter=resources.limiter),
                employee=gtm_data_analyst(),
                recall=HybridRecall(resources.vectors, resources.graph),
                progress=reporter,
            )
            parent_id = None
            if args.follow_up:
                try:
                    parent_id = uuid.UUID(args.follow_up)
                except ValueError:
                    sys.stderr.write(f"--follow-up is not a uuid: {args.follow_up!r}\n")
                    return 2
                # Checked before any tokens are spent. A parent from another tenant, or one
                # that does not exist, must not reach the loop -- `parent_id` arrives from a
                # caller and proves nothing about ownership (F-01).
                try:
                    await check_parent(session, tenant, parent_id)
                except (ParentNotFound, ThreadTooDeep) as exc:
                    sys.stderr.write(f"{exc}\n")
                    return 2
                print(f"Following up on investigation {parent_id}", file=sys.stderr)

            row = InvestigationRow(
                tenant_id=tenant.tenant_id,
                question=args.question,
                title=title_for(args.question),
                parent_id=parent_id,
            )
            session.add(row)
            await session.flush()

            try:
                investigation = await investigator.investigate(
                    session,
                    tenant,
                    investigation_id=row.id,
                    question=args.question,
                    parent_id=parent_id,
                )
            except InvestigationFailed as exc:
                await session.commit()  # keep the evidence gathered before the failure
                print(f"\nThe investigation could not complete: {exc}", file=sys.stderr)
                return 1

            emit(reporter, ProgressEvent(Phase.GATING))
            try:
                with investigation.timings.measure(GATE):
                    gated = await GroundingGate().apply(
                        session, tenant, investigation_id=row.id, report=investigation.report
                    )
            except ReportRejected as exc:
                await session.commit()
                print(f"\nThe report was rejected by the grounding gate: {exc}", file=sys.stderr)
                return 1

            report = gated.report
            # Bound before the branch, so the usage record below does not depend on
            # short-circuit evaluation to avoid an unbound name.
            verification = None
            if not args.no_verify:
                emit(reporter, ProgressEvent(Phase.VERIFYING))
                with investigation.timings.measure(VERIFY):
                    verification = await AdversarialVerifier(llm).verify(
                        session, tenant, investigation_id=row.id, report=report
                    )
                investigation.timings.record_usage(VERIFY, verification.usage)
                report = verification.report

            try:
                decision, applied = await _sufficiency(
                    llm,
                    session,
                    tenant,
                    investigation_id=row.id,
                    question=args.question,
                    report=report,
                    timings=investigation.timings,
                    skip=args.no_sufficiency,
                )
            except ReportRejected as exc:
                # Committed before returning, as the gate's refusal is: the evidence was
                # really gathered, and a refusal is an audit record too.
                row.status = InvestigationStatus.FAILED
                row.error = f"the evidence did not support an answer: {exc}"[:2000]
                row.completed_at = datetime.now(UTC)
                await session.commit()
                print(f"\nThe report was refused: {exc}", file=sys.stderr)
                return 1
            report = applied.report

            # Judged last, on the report a reader will actually receive: a claim removed for
            # being unsupported must not count as coverage. Adds a caveat and edits nothing.
            assessment = await CompletenessJudge(llm).assess(args.question, report)
            if assessment.risks:
                report = report.model_copy(update={"risks": [*report.risks, *assessment.risks]})
            # Recorded on the row, so a CLI-driven investigation appears in the spend view.
            # It did not before: `cortex.ask` drives the investigator directly rather than
            # through `InvestigationService`, so the row kept its zeros and every real
            # investigation I have run cost $0.0000 according to the aggregate. A spend
            # report that silently omits the path actually in use is worse than none.
            total = investigation.usage + decision.usage
            if verification is not None:
                total = total + verification.usage
            record_usage(row, total, model=llm.model)
            # Marked terminal, because the row was left QUEUED forever otherwise: a finished
            # CLI investigation read as still queued through the API, and the spend view
            # counted ten runs with zero completed. `cortex.ask` owns the row it created, so
            # it owns closing it.
            row.status = InvestigationStatus.COMPLETED
            row.steps_used = len(investigation.steps)
            row.completed_at = datetime.now(UTC)

            # Persisted, because it was not before. `cortex.ask` drives the investigator
            # directly rather than through `InvestigationService`, and only the service wrote
            # a `Report` row -- so every real investigation ever run through this CLI printed
            # itself to a terminal and left nothing behind. The consequences were invisible
            # until there was a UI to notice them: `/reports/{id}` and the report view had
            # nothing to show for exactly the runs we do most, and the `--real` mode's own
            # docstring says the run is committed *because it is an audit record*.
            #
            # The rejection lists are stored, not counted, matching the service: the eval
            # suite needs to tell a structural drop from a judged-unsupported claim, and a
            # reader is entitled to know claims were removed.
            session.add(
                Report(
                    tenant_id=tenant.tenant_id,
                    investigation_id=row.id,
                    body=report.model_dump(mode="json"),
                    confidence=confidence_score(report.confidence.value),
                    gate_rejections=[r.as_dict() for r in gated.rejections],
                    # Sufficiency vetoes share the verifier's list, told apart by the
                    # `sufficiency:` prefix on `detail`. A column of their own would be a
                    # migration, and both channels answer the same question: what did a
                    # reading of the evidence remove?
                    verifier_rejections=[
                        *([r.as_dict() for r in verification.rejections] if verification else []),
                        *[r.as_dict() for r in applied.rejections],
                    ],
                )
            )

            emit(reporter, ProgressEvent(Phase.DONE, detail="writing the report"))
            await session.commit()

    sys.stdout.write(
        render(
            report,
            steps=len(investigation.steps),
            seconds=investigation.duration_ms / 1000,
            tokens=investigation.usage.total,
        )
    )
    # The phase breakdown, on the real path too. It printed only for fixture runs, which is
    # backwards: a fixture serves canned payloads over no network, so its timings say little
    # about where a real investigation's wall clock goes -- and real ones are what exceed the
    # 90-second target. Optimising the wrong phase costs a day and buys nothing, which is why
    # cortex/agents/timing.py exists at all.
    sys.stdout.write("\n" + investigation.timings.render() + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
