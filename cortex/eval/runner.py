"""The eval harness.

Runs each labeled scenario end to end — loop, gate, verifier, scorer — and prints a
scorecard. `make eval` calls this, and CI fails the build when a gating dimension
regresses.

Two properties make the result trustworthy:

  - **The connectors are replaced, not the framework.** Scenario responses are served
    through a real `Tool` and the real `ToolExecutor`, so evidence rows are written,
    hashes are computed, and the audit trail is populated exactly as in production.
    Stubbing the executor would make the grounding score meaningless, since the gate
    resolves against the rows the executor writes.
  - **The analyst is the real one.** Only the *upstream APIs* are synthetic. With a
    live provider this measures the shipped analyst; with a recorded provider it
    measures the harness itself, deterministically and for free.
"""

from __future__ import annotations

import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from cortex.agents.employee import Employee, gtm_data_analyst
from cortex.agents.investigator import InvestigationFailed, Investigator
from cortex.agents.llm import LLM
from cortex.agents.timing import GATE, SUFFICIENCY, VERIFY
from cortex.db.models import Investigation as InvestigationRow
from cortex.db.models import Tenant
from cortex.eval.fixtures import SCENARIOS, Scenario
from cortex.eval.replay import write_bundle, write_failure
from cortex.eval.scorer import Scorecard, Scorer
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.reports.data_trust import enforce_data_trust
from cortex.reports.gate import GateResult, GroundingGate, ReportRejected
from cortex.reports.identifiability import apply_corroboration, identifiability_notes
from cortex.reports.schema import InvestigationReport
from cortex.reports.sufficiency import (
    AppliedSufficiency,
    SufficiencyDecision,
    SufficiencyGate,
)
from cortex.reports.verifier import AdversarialVerifier
from cortex.tenancy.context import TenantContext
from cortex.tools.base import Capability, Tool, ToolContext, ToolRegistry, ToolResult
from cortex.tools.disclosures import grade_series, series_disclosures
from cortex.tools.executor import ToolExecutor
from cortex.tools.registry import gtm_analyst_registry


class ScenarioTool(Tool):
    """A stand-in for one real connector, serving a scenario's canned responses.

    Mirrors the real tool's name and capability names so the LLM sees the same tool
    surface it sees in production, and so `tool_selection` scores against the same
    qualified names. Only the transport is synthetic.
    """

    provider = None

    def __init__(self, name: str, capability_names: list[str], scenario: Scenario) -> None:
        self.name = name
        self._capability_names = capability_names
        self._scenario = scenario
        self.calls: list[str] = []
        super().__init__()

    def capabilities(self) -> list[Capability]:
        real = gtm_analyst_registry().get(self.name)
        out: list[Capability] = []
        for capability_name in self._capability_names:
            original = real.capability(capability_name)
            out.append(
                Capability(
                    name=capability_name,
                    # The real description and schema: tool selection is part of what
                    # is being measured, so the model must choose from the real surface.
                    description=original.description,
                    params_schema=original.params_schema,
                    # Mirrored too, so a scenario that plants nothing for a capability
                    # produces an observation the analyst reads as *empty* rather than as a
                    # real absence — which is the behaviour the suite should be measuring.
                    result_key=original.result_key,
                    # Mirrored for the same reason as everything above: the survey runs every
                    # discovery capability before the first step, so a fixture that dropped the
                    # flag would exercise a loop the product does not have -- no survey, a
                    # smaller opening prompt, and fewer tool calls than a real investigation
                    # makes. That is exactly the fixture drift that once let six new
                    # capabilities go unmeasured.
                    discovery=original.discovery,
                    handler=self._make_handler(capability_name),
                )
            )
        return out

    def _make_handler(self, capability_name: str) -> Any:
        qualified = f"{self.name}__{capability_name}"

        async def _handler(ctx: ToolContext, **params: Any) -> ToolResult:
            del ctx
            self.calls.append(qualified)
            # Params reach the fixture so a scenario about a change can answer
            # differently on each side of it. Without this every period gets the same
            # payload, which is how the campaign fixture came to contradict itself.
            payload = dict(self._scenario.response_for(qualified, params))
            if self._scenario.discloses_anything:
                # The same assembler production uses. Without this the suite measured none of
                # them: this handler replaces the connector method, so `series_ends_early`,
                # `partial_buckets`, `data_trust` and the movement description never reached a
                # scenario, and two consecutive 10/10 runs said nothing about any of them.
                #
                # Not hand-written into the fixtures instead, which was the obvious alternative
                # and measures the wrong thing -- whether the analyst reacts to a disclosure,
                # not whether the connector computes one. A fixture author who forgets looks
                # exactly like a connector that does not disclose.
                # Filtered by name, so a scenario can allow one disclosure and withhold the
                # rest. Pricing a single disclosure needs that: a bool turns on four at once
                # and a paired comparison then attributes four changes to one.
                found = dict(
                    series_disclosures(
                        qualified,
                        payload,
                        params,
                        # The scenario's world, not today's date: a fixture serves one
                        # fixed series for every range, so without this every scenario
                        # reports a collection failure to an analyst who asks wider than
                        # was planted.
                        as_of=self._scenario.as_of,
                    )
                )
                found.update(grade_series(qualified, {**payload, **found}))
                payload.update(
                    {key: value for key, value in found.items() if self._scenario.discloses(key)}
                )
            return ToolResult(
                payload={**payload, "_eval_params": params},
                source_ref=f"eval://{qualified}",
            )

        return _handler


def scenario_registry(scenario: Scenario) -> ToolRegistry:
    """The tool surface this scenario's tenant can actually reach, backed by the scenario.

    Filtered by `Scenario.connected_tools` for the same reason `registry_for_tenant` filters in
    production: a tenant is offered what it has credentials for, and no more. Mirroring the whole
    registry made the eval measure a surface the product does not present -- and made every
    scenario slower the moment a connector was added for somebody else's tenant.

    The filter is applied here rather than inside `ScenarioTool` so `tool_selection` still scores
    against the real qualified names, and so a scenario that starts planting a new source picks it
    up with no change to this function.
    """
    real = gtm_analyst_registry()
    connected = scenario.connected_tools
    registry = ToolRegistry()
    for tool_name in real.tool_names:
        if tool_name not in connected:
            continue
        registry.register(ScenarioTool(tool_name, real.get(tool_name).capability_names, scenario))
    return registry


@dataclass(slots=True)
class ScenarioOutcome:
    scenario: str
    card: Scorecard | None
    error: str | None = None
    #: Which attempt this was, when a scenario is run more than once. 1-based.
    attempt: int = 1

    #: Where the wall clock went, as "phase 12.3s x2" strings, ordered slowest first.
    #:
    #: On the scorecard rather than only in a bundle, because a latency line reading "121440ms
    #: against a 90000ms budget" tells nobody what to do about it.
    phases: tuple[str, ...] = ()

    #: The delivered report, as JSON, kept so a failure can be diagnosed.
    #:
    #: An eval run rolls its tenants and evidence back on purpose, which means a failing
    #: attempt used to leave nothing behind but a one-line reason. "missing required
    #: signal: 91c3e4a" does not say whether the analyst identified the change by another
    #: name or missed it entirely, and those call for opposite fixes -- one is a
    #: mislabelled fixture, the other a real miss. Answering it required paying for
    #: another run and hoping the failure recurred.
    report_json: str | None = None

    @property
    def passed(self) -> bool:
        return self.card is not None and self.card.passed


@dataclass(slots=True)
class EvalRun:
    outcomes: list[ScenarioOutcome] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.outcomes) and all(o.passed for o in self.outcomes)

    @property
    def total_hallucinations(self) -> int:
        """Unsupported claims that reached a report. The number that must be zero."""
        return sum(o.card.hallucinations for o in self.outcomes if o.card)

    @property
    def total_caught(self) -> int:
        """Claims a mechanism removed before delivery. Informational, never hidden."""
        return sum(o.card.caught_and_removed for o in self.outcomes if o.card)

    def render(self) -> str:
        """A scorecard a human reads without needing the code open."""
        lines = ["", "Cortex evaluation", "=" * 72]
        for outcome in self.outcomes:
            if outcome.card is None:
                lines += [f"\n{outcome.scenario}: ERRORED", f"  {outcome.error}"]
                continue
            card = outcome.card
            verdict = "PASS" if card.passed else "FAIL"
            lines.append(
                f"\n{outcome.scenario}: {verdict}  "
                f"overall={card.overall:.2f}  "
                # Both numbers, always. The delivered count is what gates; the caught
                # count is what warns. Printing only the first would hide the drafter
                # getting worse for as long as the defenses keep holding.
                f"delivered_hallucinations={card.hallucinations}  "
                f"caught={card.caught_and_removed}  "
                f"{card.duration_ms}ms  {card.tokens} tokens"
            )
            for dimension in card.dimensions:
                mark = " " if dimension.passed else "!"
                gate = " (gate)" if dimension.gates else ""
                lines.append(
                    f"  {mark} {dimension.name:<16} {dimension.score:>6.2f}{gate}  "
                    f"{dimension.detail}"
                )
                # Printed whenever the answer took longer than the budget, not only when the
                # dimension "failed" -- it never gates, so 98 seconds against a 90-second budget
                # scores 0.91 and passes. The first run this diagnostic was written for
                # suppressed it for exactly that reason.
                if dimension.name == "latency" and dimension.score < 1.0 and outcome.phases:
                    lines.append(f"      where it went: {', '.join(outcome.phases)}")

        lines += self._aggregate_lines()

        lines += ["", "-" * 72]
        passed = sum(1 for o in self.outcomes if o.passed)
        lines.append(
            f"{passed}/{len(self.outcomes)} scenarios passed  "
            f"delivered_hallucinations={self.total_hallucinations} (must be 0)  "
            f"caught_before_delivery={self.total_caught}"
        )
        if not self.passed:
            lines.append("")
            for outcome in self.outcomes:
                if outcome.passed:
                    continue
                reasons = outcome.card.failures if outcome.card else [outcome.error or "errored"]
                for reason in reasons:
                    lines.append(f"  FAIL {outcome.scenario}: {reason}")
        return "\n".join(lines) + "\n"

    def _aggregate_lines(self) -> list[str]:
        """Per-scenario spread, printed only when a scenario was attempted more than once.

        Without this a repeated run is just a longer list, and the thing repeats were
        added to reveal — how much a score moves between identical attempts — has to be
        eyeballed. The min and max are reported rather than only the mean, because a
        scenario averaging 0.7 by scoring 0.9 and 0.5 is a different situation from one
        scoring 0.7 twice, and only one of them supports a comparison.
        """
        by_scenario: dict[str, list[ScenarioOutcome]] = {}
        for outcome in self.outcomes:
            by_scenario.setdefault(outcome.scenario, []).append(outcome)
        if all(len(group) == 1 for group in by_scenario.values()):
            return []

        lines = ["", "-" * 72, "Across attempts"]
        for name, group in by_scenario.items():
            scored = [o.card.overall for o in group if o.card]
            passes = sum(1 for o in group if o.passed)
            errored = sum(1 for o in group if o.card is None)
            spread = (
                f"overall min={min(scored):.2f} mean={sum(scored) / len(scored):.2f} "
                f"max={max(scored):.2f}"
                if scored
                else "no attempt produced a score"
            )
            suffix = f", {errored} errored" if errored else ""
            lines.append(f"  {name:<24} {passes}/{len(group)} passed  {spread}{suffix}")
        return lines


class EvalHarness:
    def __init__(
        self,
        *,
        llm: LLM,
        employee: Employee | None = None,
        verify: bool = True,
        scorer: Scorer | None = None,
        investigator_factory: Callable[..., Any] | None = None,
        capture_to: Path | None = None,
        sufficiency: bool = True,
    ) -> None:
        self._llm = llm
        self._employee = employee or gtm_data_analyst()
        self._verify = verify
        # On by default, matching both delivery paths. An eval that ran a different pipeline
        # from the one that ships is measuring something nobody uses.
        self._sufficiency = sufficiency
        self._scorer = scorer or Scorer()
        # Where to write a replayable bundle per attempt, if anywhere.
        #
        # The run's transaction is rolled back on purpose -- an eval must not leave tenants and
        # evidence behind in whatever database it was pointed at -- so nothing about an attempt
        # survives it. That made every scoring change cost a full paid re-run, and it is why the
        # accuracy figures in docs/eval-results.md are upper bounds rather than measurements.
        # See `cortex.eval.replay`.
        self._capture_to = capture_to
        # Which loop runs is injectable so an alternative implementation can be scored
        # on identical scenarios, tools, evidence store, drafting call, gate and
        # verifier. Holding everything but the loop constant is the only way a score
        # difference means anything.
        self._investigator_factory = investigator_factory or Investigator

    async def run(
        self,
        session: AsyncSession,
        scenarios: tuple[Scenario, ...] = SCENARIOS,
        repeat: int = 1,
    ) -> EvalRun:
        """Score every scenario, `repeat` times each.

        Repeats exist because the same scenario has been observed passing at 0.85 and
        then failing on unchanged analyst code. At one attempt per scenario, a score is
        a sample of a distribution being reported as a measurement — which is fine for
        catching an outage and useless for comparing two implementations.
        """
        # Interleaved rather than run back to back, so a provider having a bad few
        # minutes degrades every scenario's attempts a little instead of destroying one
        # scenario's entire sample.
        plan = [(scenario, attempt) for attempt in range(1, repeat + 1) for scenario in scenarios]
        run = EvalRun()
        for index, (scenario, attempt) in enumerate(plan, start=1):
            # Progress goes to stderr, leaving stdout as the scorecard alone so it
            # stays pipeable. A live run takes minutes per scenario, and a run that
            # prints nothing until the end is indistinguishable from a hung one --
            # which is exactly how an unbounded provider call went unnoticed.
            label = f"[{index}/{len(plan)}] {scenario.name}"
            if repeat > 1:
                label += f" (attempt {attempt}/{repeat})"
            _progress(f"{label}: investigating")
            started = time.monotonic()
            try:
                outcome = await self.run_one(session, scenario, attempt)
            except Exception as exc:  # noqa: BLE001 - see below
                # One scenario's outage must not destroy the run. A crash here
                # previously lost the completed scenarios' results along with their
                # error reasons, which is how a mid-stream transport failure came to
                # be diagnosed from a traceback rather than from the scorecard.
                # Recorded as errored, never as a zero: a scenario that could not run
                # is a different problem from one that answered badly.
                outcome = ScenarioOutcome(
                    scenario=scenario.name, card=None, error=f"{type(exc).__name__}: {exc}"
                )
            outcome.attempt = attempt
            elapsed = time.monotonic() - started
            state = "errored" if outcome.error else ("passed" if outcome.passed else "failed")
            line = f"{label}: {state} in {elapsed:.0f}s"
            # The reason goes out with the progress line, not only into the scorecard,
            # so it survives a run that never reaches the end.
            _progress(f"{line} -- {outcome.error}" if outcome.error else line)
            run.outcomes.append(outcome)
        return run

    async def _capture_rejection(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        *,
        investigation_id: uuid.UUID,
        scenario: Scenario,
        attempt: int,
        investigation: Any,
        report: InvestigationReport,
        reason: str,
    ) -> None:
        """Capture the draft that was refused, so a refusal can be looked at afterwards.

        A rejected attempt used to write nothing, which made the most interesting failures the
        only ones leaving no trace: run 15 refused two of ten attempts outright and the reports
        behind them were gone before anyone could ask why. The evidence and audit rows still
        exist at this point -- the transaction has not been rolled back yet -- so this is the one
        moment they can be captured.

        The report stored is the draft, not an answer anybody received, and the bundle says so.
        Failure here is swallowed: a capture problem must not replace the rejection reason with
        a traceback about capturing.
        """
        if self._capture_to is None:
            return
        try:
            await write_bundle(
                session,
                tenant,
                investigation_id=investigation_id,
                scenario=scenario.name,
                attempt=attempt,
                investigation=investigation,
                gate_result=GateResult(report=report),
                verification=None,
                directory=self._capture_to,
                rejected_because=reason,
            )
        except Exception as exc:  # noqa: BLE001 - see the docstring
            _progress(f"  could not capture the rejected attempt: {type(exc).__name__}: {exc}")

    async def _capture_failure(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        *,
        investigation_id: uuid.UUID,
        scenario: Scenario,
        attempt: int,
        error: str,
    ) -> None:
        """Record an attempt that never reached a report.

        Swallows its own failure for the same reason `_capture_rejection` does: a problem
        writing the record must not replace the real error with a traceback about writing it.
        """
        if self._capture_to is None:
            return
        try:
            await write_failure(
                session,
                tenant,
                investigation_id=investigation_id,
                scenario=scenario.name,
                attempt=attempt,
                error=error,
                directory=self._capture_to,
            )
        except Exception as exc:  # noqa: BLE001 - see the docstring
            _progress(f"  could not capture the failed attempt: {type(exc).__name__}: {exc}")

    async def run_one(
        self, session: AsyncSession, scenario: Scenario, attempt: int = 1
    ) -> ScenarioOutcome:
        tenant, investigation_id = await self._provision(session, scenario)
        registry = scenario_registry(scenario)
        investigator = self._investigator_factory(
            llm=self._llm,
            registry=registry,
            executor=ToolExecutor(registry),
            employee=self._employee,
        )

        try:
            investigation = await investigator.investigate(
                session,
                tenant,
                investigation_id=investigation_id,
                question=scenario.question,
            )
        except InvestigationFailed as exc:
            # Recorded as an errored scenario rather than a zero: a loop that could
            # not run is a different problem from a loop that answered badly, and
            # collapsing them would hide an outage behind a quality regression.
            #
            # Captured as well as recorded. This branch used to leave nothing at all, so a
            # drafting failure was diagnosable only from its error string -- and the first
            # question about one is what the loop had done beforehand, which the string does
            # not say.
            await self._capture_failure(
                session,
                tenant,
                investigation_id=investigation_id,
                scenario=scenario,
                attempt=attempt,
                error=str(exc),
            )
            return ScenarioOutcome(scenario=scenario.name, card=None, error=str(exc))

        # Timed into the investigation's own clock, so the eval can say *where* a scenario's
        # wall time went rather than only how much there was.
        #
        # The loop already times survey, recall, its own model calls, tool round trips and
        # drafting. Gate, verifier and sufficiency all run out here in the harness, so they were
        # untimed -- and they are exactly the phases Phase 1 added to. A scenario at 121s against
        # a 90-second budget was diagnosable only by re-running it under `cortex.ask`.
        try:
            with investigation.timings.measure(GATE):
                gate_result = await GroundingGate().apply(
                    session,
                    tenant,
                    investigation_id=investigation_id,
                    report=investigation.report,
                )
        except ReportRejected as exc:
            reason = f"report rejected by the gate: {exc}"
            await self._capture_rejection(
                session,
                tenant,
                investigation_id=investigation_id,
                scenario=scenario,
                attempt=attempt,
                investigation=investigation,
                report=investigation.report,
                reason=reason,
            )
            return ScenarioOutcome(scenario=scenario.name, card=None, error=reason)

        verification = None
        if self._verify:
            try:
                with investigation.timings.measure(VERIFY):
                    verification = await AdversarialVerifier(self._llm).verify(
                        session,
                        tenant,
                        investigation_id=investigation_id,
                        report=gate_result.report,
                    )
                # Folded into the total and attributed to the phase, which the production path
                # already does (`service.py` sums both before billing) and this one did not.
                # The scorecard reads `investigation.usage.total`, so every token figure the
                # eval has ever printed omitted verification -- and every bundle recorded zero
                # output tokens for this phase, leaving its output-boundness unmeasured while
                # latency work leaned on exactly that number.
                investigation.usage = investigation.usage + verification.usage
                investigation.timings.record_usage(VERIFY, verification.usage)
            except ReportRejected as exc:
                reason = f"report rejected by the verifier: {exc}"
                await self._capture_rejection(
                    session,
                    tenant,
                    investigation_id=investigation_id,
                    scenario=scenario,
                    attempt=attempt,
                    investigation=investigation,
                    report=gate_result.report,
                    reason=reason,
                )
                return ScenarioOutcome(scenario=scenario.name, card=None, error=reason)

        # Decision 6, and it was missing here while running in both delivery paths.
        #
        # `ask.py` and `agents/service.py` both run the sufficiency gate; the eval did not, so
        # the suite measured every other Phase 1 change and silently skipped this one -- on the
        # scenario it exists to catch. `insufficient_evidence` failed a paid run with "claimed a
        # cause on unanswerable data" while the gate that vetoes exactly that never ran.
        #
        # Ordered after the verifier, as in `service.py`: the gate judges the report a reader
        # would actually receive, so a claim the verifier already removed must not be counted
        # against it.
        sufficiency = SufficiencyDecision(needed=False)
        applied = AppliedSufficiency(
            report=verification.report if verification else gate_result.report
        )
        if self._sufficiency:
            try:
                with investigation.timings.measure(SUFFICIENCY):
                    sufficiency = await SufficiencyGate(self._llm).assess(
                        session,
                        tenant,
                        investigation_id=investigation_id,
                        question=scenario.question,
                        report=applied.report,
                    )
                investigation.usage = investigation.usage + sufficiency.usage
                investigation.timings.record_usage(SUFFICIENCY, sufficiency.usage)
                applied = sufficiency.apply(applied.report)
            except ReportRejected as exc:
                reason = f"report rejected by the sufficiency gate: {exc}"
                await self._capture_rejection(
                    session,
                    tenant,
                    investigation_id=investigation_id,
                    scenario=scenario,
                    attempt=attempt,
                    investigation=investigation,
                    report=applied.report,
                    reason=reason,
                )
                return ScenarioOutcome(scenario=scenario.name, card=None, error=reason)

        # The verifier's own result carries the report the scorer reads, so a sufficiency veto
        # has to be folded back into it -- otherwise the scorer grades the pre-veto report and
        # the gate's effect is invisible to every dimension.
        if verification is not None:
            verification = replace(
                verification,
                report=applied.report,
                verdicts=verification.verdicts,
            )
        else:
            gate_result = replace(gate_result, report=applied.report)

        # Decision 5's disclosure, run here too so the suite measures the pipeline that ships
        # rather than one without it -- the same mistake the sufficiency gate was found in.
        # Decision 1, enforced. First of the three for the same reason as in the worker:
        # it decides whether there is a movement to explain.
        applied = replace(
            applied,
            report=(
                await enforce_data_trust(
                    session, tenant, investigation_id=investigation_id, report=applied.report
                )
            ).report,
        )

        # Decision 3, before the disclosure below. See `apply_corroboration` for why the
        # order matters.
        applied = replace(
            applied,
            report=await apply_corroboration(
                session, tenant, investigation_id=investigation_id, report=applied.report
            ),
        )

        for risk in await identifiability_notes(
            session, tenant, investigation_id=investigation_id, report=applied.report
        ):
            applied = replace(
                applied,
                report=applied.report.model_copy(update={"risks": [*applied.report.risks, risk]}),
            )
        if verification is not None:
            verification = replace(verification, report=applied.report)
        else:
            gate_result = replace(gate_result, report=applied.report)

        card = await self._scorer.score(
            session,
            tenant,
            investigation_id=investigation_id,
            scenario=scenario,
            investigation=investigation,
            gate_result=gate_result,
            verification=verification,
            sufficiency=applied,
        )
        delivered = verification.report if verification else gate_result.report

        # Written here and nowhere else, because this is the last moment the evidence and audit
        # rows exist -- `__main__` rolls the transaction back immediately after the run.
        if self._capture_to is not None:
            await write_bundle(
                session,
                tenant,
                investigation_id=investigation_id,
                scenario=scenario.name,
                attempt=attempt,
                investigation=investigation,
                gate_result=gate_result,
                verification=verification,
                sufficiency=applied,
                directory=self._capture_to,
            )

        return ScenarioOutcome(
            scenario=scenario.name,
            card=card,
            # The report as the reader would have received it, not the draft: a claim the
            # gate or verifier removed is not part of the answer being scored.
            report_json=delivered.model_dump_json(indent=1),
            phases=_phase_summary(investigation.timings),
        )

    @staticmethod
    async def _provision(
        session: AsyncSession, scenario: Scenario
    ) -> tuple[TenantContext, uuid.UUID]:
        """A fresh tenant per scenario.

        Isolated on purpose: scoring reads the evidence store and the audit trail by
        tenant, so sharing one would let an earlier scenario's rows inflate a later
        scenario's grounding and tool-selection scores.
        """
        slug = f"eval-{scenario.name.replace('_', '-')}-{uuid.uuid4().hex[:6]}"
        tenant_id = uuid.uuid4()
        session.add(
            Tenant(
                id=tenant_id,
                slug=slug,
                name=scenario.name,
                graph_name=graph_name_for_new_tenant(slug, tenant_id),
            )
        )
        await session.flush()

        row = InvestigationRow(tenant_id=tenant_id, question=scenario.question)
        session.add(row)
        await session.flush()

        return (
            TenantContext(
                tenant_id=tenant_id,
                tenant_slug=slug,
                graph_name=graph_name_for_new_tenant(slug, tenant_id),
            ),
            row.id,
        )


def _phase_summary(timings: Any) -> tuple[str, ...]:
    """Phases as "name 12.3s x2", slowest first, dropping anything under a tenth of a second.

    Slowest first because that is the only order anyone reads it in, and the sub-100ms phases
    are noise on a two-minute investigation -- the citation gate is pure code and always rounds
    to zero.
    """
    phases = getattr(timings, "phases", None)
    if not phases:
        return ()
    ranked = sorted(phases.items(), key=lambda item: item[1].seconds, reverse=True)
    return tuple(
        f"{name} {phase.seconds:.1f}s x{phase.calls}"
        for name, phase in ranked
        if phase.seconds >= 0.1
    )


def _progress(line: str) -> None:
    """One line of run progress, flushed. stderr so stdout stays the scorecard."""
    sys.stderr.write(f"{line}\n")
    sys.stderr.flush()
