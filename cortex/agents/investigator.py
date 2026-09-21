"""The investigation loop.

Replaces single-shot retrieval with the cycle a human analyst actually runs:

    question → hypotheses → evidence → contradiction → more evidence → conclusion

Three things make this more than a tool-calling loop, and they are the reason it
is written here rather than delegated to a framework:

  - **Evidence ids reach the model.** Every tool result is returned with the
    `evidence_id` the executor minted for it. That is what lets the drafted report
    cite observations, and therefore what makes the grounding gate able to check
    the citation rather than trust it. A loop that returned bare payloads would
    leave nothing to cite.
  - **Budgets are enforced, not suggested.** Steps, tool calls, tokens and wall
    clock are all bounded, and the loop stops on whichever binds first. An
    unbounded agent loop is discovered on an invoice.
  - **Spinning is detected.** A loop that keeps calling tools without gathering
    new evidence has stopped investigating. It is stopped early rather than left
    to exhaust the budget.

Tool failures are fed back rather than raised. A rate limit or a missing
credential is information the analyst can work around — reporting "HubSpot is not
connected, so revenue impact is unknown" is a better answer than a crash.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.agents.employee import Employee
from cortex.agents.llm import (
    LLM,
    LLMAuthenticationFailed,
    LLMError,
    LLMOutputTruncated,
    Message,
    ToolRequest,
    Usage,
)
from cortex.agents.progress import Phase, ProgressEvent, ProgressSink, emit
from cortex.agents.reading import (
    MAX_READS_PER_STEP,
    READ_TOOL,
    Digest,
    ReadingLedger,
    UnknownHandle,
    Unread,
    read_tool_spec,
    render_payload,
)
from cortex.agents.spin import SpinDetector
from cortex.agents.timing import DRAFT, LOOP_MODEL, LOOP_TOOLS, RECALL, SURVEY, Timings
from cortex.agents.transcript import assert_valid
from cortex.db.models import Evidence
from cortex.db.threads import citable_investigation_ids, prior_context
from cortex.memory.recall import HybridRecall
from cortex.reports.schema import InvestigationReport, PremiseVerdict, llm_report_schema
from cortex.reports.shape import AMBIGUITY_GUIDANCE, GUIDANCE, ambiguities, shape_for
from cortex.tenancy.context import TenantContext
from cortex.tools.base import ToolError, ToolRegistry
from cortex.tools.executor import ExecutedTool, ToolExecutor


class InvestigationCancelled(Exception):
    """The caller asked for the investigation to stop, and it did.

    Not a subclass of `InvestigationFailed`: nothing went wrong, and a caller distinguishing
    "this broke" from "somebody stopped it" needs the types separated. The evidence gathered
    before the stop is kept — it was really observed, and it is what a user asking "what did
    it find before I cancelled" wants to see.
    """

    def __init__(self, message: str, *, steps: int, evidence: int) -> None:
        super().__init__(message)
        self.steps = steps
        self.evidence = evidence


class InvestigationFailed(Exception):
    """The loop could not produce a draft. Distinct from a report being rejected."""


class StopReason(str):
    """Why the loop ended. A plain string subclass so it serialises trivially."""


CONCLUDED = StopReason("concluded")
STEP_LIMIT = StopReason("step_limit")
TOOL_CALL_LIMIT = StopReason("tool_call_limit")
TOKEN_LIMIT = StopReason("token_limit")
TIME_LIMIT = StopReason("time_limit")
STALLED = StopReason("stalled")
#: The caller asked for it to stop. Distinct from every other reason, because the others are
#: the loop's own judgement and this one is not.
CANCELLED = StopReason("cancelled")


#: Drafting gets its own deadline, longer than a loop turn's. It is one call that
#: synthesises the entire investigation into a structured report of up to 32k tokens,
#: and it was measured timing out at the 120-second per-turn ceiling *after* the loop
#: had already gathered its evidence successfully — the most expensive possible place
#: to fail, since everything spent up to that point is discarded with it.
DRAFT_TIMEOUT_SECONDS = 300.0

#: Wall clock reserved for drafting, subtracted from the loop's own budget.
#:
#: The loop's `max_seconds` and the drafting deadline were independent numbers, both 300, and
#: nothing subtracted one from the other. So a loop could spend its entire budget gathering, and
#: drafting would then start with a fresh 300 seconds — total 600 against a 90-second target — or,
#: worse, produce a transcript large enough that drafting could not finish inside its own deadline.
#:
#: That is not hypothetical. The reflection-turn experiment (docs/eval-results.md run 15) lost an
#: attempt entirely: "the report could not be drafted: no response within 300s", after 432 seconds
#: of work. The intervention was scored as the failure, and it may well be one — but the mechanism
#: that destroyed the run was a budgeting bug, and a clean re-test of reflection would need this
#: fixed first.
#:
#: Reserved rather than shared, because the asymmetry is total: evidence gathered without a report
#: is worth nothing, while a report drafted from slightly less evidence is worth almost as much.
#: When the reserve binds, the loop stops early with `TIME_LIMIT` and the report says it was cut
#: short — which is a disclosed, complete answer instead of a lost investigation.
DRAFT_RESERVE_SECONDS = 90.0

#: Repair attempts after the first draft, when the draft is well-formed JSON but fails a
#: model-level invariant the JSON Schema cannot express.
#:
#: One, not more. A model handed the exact validation error usually fixes it immediately;
#: a model that fails twice is not converging, and each attempt costs a full drafting call
#: against a deadline that already dominates the investigation's wall clock.
_DRAFT_REPAIR_ATTEMPTS = 1

#: Retries bought by asking for a shorter report after a truncated one. One, because a
#: second would mean the ceiling is the wrong problem: at 64k a report that still cannot
#: fit twice is not long, it is looping.
_DRAFT_BREVITY_RETRIES = 1

#: Tool calls requested per turn, as a schedule over the step index.
#:
#: **Descending, and the previous constant-3 was the wrong arm of the paper it cited.** W&D
#: (arXiv 2602.07359) Table 3, BrowseComp / GPT-5-Medium, compares schedules rather than only
#: constant widths:
#:
#:     Constant 1        66%   45.7 turns
#:     Constant 3        68%   23.8 turns   <- what this used to be
#:     Ascending         63%   36.5 turns
#:     Descending        74%   23.5 turns   <- this
#:     Automatic         72%   26.6 turns
#:
#: Descending buys **+6 points of accuracy at an identical turn count** -- free on latency, which
#: is the budget that actually binds here. The mechanism is the one the old comment guessed at and
#: then tried to handle with wording: early turns are exploratory and genuinely parallel, because
#: many unknowns are independent; late turns are confirmatory and dependent, so a fixed floor of
#: three late in a trajectory is exactly when a model invents filler calls.
#:
#: The paper is also explicit that the model should not be asked to choose: *"The Automatic
#: strategy did not perform better than Descending, indicating that the LLM itself cannot
#: determine the optimal number of tool calls in each iteration."* The previous instruction here
#: asked for "around 3" and forbade padding -- prose delegating the decision, which is the
#: Automatic arm at 72% and 26.6 turns rather than the Descending arm at 74% and 23.5.
#:
#: **Opening at 5 rather than 3**, on the interaction W&D states: *"when number of steps is small,
#: increasing number of tools per step improves accuracy; when number of steps is higher, having
#: more tools may not be beneficial."* Their counter-evidence for wide openings (Table 1: at a
#: 100-turn cap, 1 tool 70% beats 3 tools 68% beats 5 tools 60%) describes a regime we are nowhere
#: near -- our realised trajectories are 3 model calls, so we are firmly in the small-steps case
#: where width monotonically helps.
#:
#: **The tail is 2, not 1, and that is measured rather than inherited.** The first version of this
#: schedule was (5, 3, 2, 1) and eval 17 showed exactly what the narrowing cost:
#:
#:     insufficient_evidence latency        126,180ms -> 79,439 / 76,571ms
#:     insufficient_evidence tool_selection      1.00 -> 0.00, both attempts
#:
#: The wide opening delivered the latency win the whole exercise was for -- the scenario that had
#: never once finished inside the 90-second budget did so twice. The narrowing cost the
#: discriminating call: front-loading five means the analyst picks its five, gathers enough to
#: decline, and a tail of one never revisits a source it did not think of at step 1.
#:
#: So the tail holds at 2. That keeps W&D's mechanism -- early turns exploratory and parallel, late
#: turns confirmatory -- while leaving room for the one call a narrowing schedule forecloses.
#:
#: Stated plainly: this is tuned against a six-attempt eval, which the statistics say cannot
#: resolve a 0.02 effect (MDE ~0.08 unpaired). The direction is supported by a consistent
#: two-of-two failure with an explicable mechanism; the exact tail value is not.
TOOL_CALL_SCHEDULE = (5, 3, 2, 2)


def tool_calls_for_step(index: int) -> int:
    """How many calls to ask for at a zero-based step index.

    Holds at the final value rather than running out, so a long investigation keeps a floor of one
    request per turn instead of falling off the end of the schedule.
    """
    if index < 0:
        return TOOL_CALL_SCHEDULE[0]
    return TOOL_CALL_SCHEDULE[min(index, len(TOOL_CALL_SCHEDULE) - 1)]


#: Output ceiling for one report draft.
#:
#: Raised from 16k after a live run failed with "output truncated at max_tokens=16384;
#: the schema could not be completed" — and failed for a *good* reason: reports have grown
#: from a handful of claims to twenty-plus as the drafting prompt improved, and a rich
#: report with findings, hypotheses, risks, recommendations and data-quality notes simply
#: does not fit. Truncation is the worst way to hit a limit, because partial JSON parses to
#: nothing and the whole investigation is discarded.
#:
#: Raised again, from 32k, after a second live run died the same way -- and the repeat is the
#: interesting part. `max_tokens` bounds *thinking plus output*, not output, so the draft is
#: two demands on one budget: it is the longest-thinking call in the investigation (deciding
#: what the evidence concluded) and the one whose output is structurally all-or-nothing.
#: The scenario that died both times was `insufficient_evidence` -- the hardest one to think
#: about, because the model has to conclude that nothing can be concluded.
#:
#: This is now the model's whole ceiling rather than a fraction of it. Nothing is reserved
#: because nothing follows: drafting is the last provider call in the investigation, and a
#: ceiling costs nothing unused -- tokens are billed as generated.
#:
#: Room alone is still not a fix, only a bigger number to exhaust, which is why truncation
#: is separately *recoverable* now -- see `_BREVITY_RETRY_INSTRUCTION`.
DRAFT_MAX_TOKENS = 64_000

#: Handed back after a draft is truncated, to buy a second attempt that fits.
#:
#: Says what to give up, and in what order. A model told only "be shorter" compresses the
#: summary, which is the one part a reader always reads; the evidence is already gathered
#: and paid for, so the cheap thing to lose is the fourth hypothesis, not the answer.
_BREVITY_RETRY_INSTRUCTION = (
    "That report ran past the output limit and was lost before it could be read. "
    "Draft it again, short enough to finish.\n\n"
    "Give things up in this order: extra hypotheses beyond the ones the evidence "
    "actually distinguishes, then risks and recommendations that repeat each other, then "
    "detail inside findings. Keep the executive summary, keep every finding that carries "
    "the answer, and keep all citations -- a claim with no evidence_id is dropped later "
    "anyway, so shortening by removing citations loses the claim too."
)


@dataclass(slots=True)
class Step:
    """One turn, recorded for the investigation's audit trail."""

    index: int
    note: str
    tool_calls: list[str] = field(default_factory=list)
    evidence_ids: list[uuid.UUID] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    #: Observations this step read in full, having previously seen only a digest.
    #:
    #: Separate from `tool_calls` because a read is not a call: it reaches no upstream, mints
    #: no evidence and spends nothing. It is recorded because ADR 0005 decision 8 asks for
    #: reading to be a *logged* action — without this the trace shows a turn that appears to
    #: have done nothing.
    reads: list[uuid.UUID] = field(default_factory=list)


@dataclass(slots=True)
class Investigation:
    """The loop's outcome, before grounding."""

    report: InvestigationReport
    steps: list[Step]
    evidence_ids: list[uuid.UUID]
    usage: Usage
    stop_reason: StopReason
    duration_ms: int
    #: Per-phase wall clock, so a slow investigation can be attributed rather than
    #: guessed at. Default-constructed so existing callers keep working.
    timings: Timings = field(default_factory=Timings)

    #: Observations fetched, summarised, and never read in full.
    #:
    #: The enumerable set ADR 0005 decision 8 asks for. It is on the outcome rather than only
    #: in the transcript because "what did it fetch and ignore" is a question asked *after*
    #: the investigation — by whoever is deciding whether to trust the report. Nothing
    #: persists it yet: `investigations` has no trace column, and adding one is a migration
    #: this change is not allowed to write. Scoring it is an eval change.
    unread_observations: tuple[Unread, ...] = ()
    #: Digested observations the analyst went back and read. The other half of the count.
    observations_read: int = 0

    @property
    def gathered_evidence(self) -> bool:
        return bool(self.evidence_ids)


class Investigator:
    def __init__(
        self,
        *,
        llm: LLM,
        registry: ToolRegistry,
        executor: ToolExecutor,
        employee: Employee,
        clock: Callable[[], float] = time.monotonic,
        recall: HybridRecall | None = None,
        progress: ProgressSink | None = None,
        cancelled: Callable[[], Awaitable[bool]] | None = None,
        today: date | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._executor = executor
        self._employee = employee
        # **Pinned by the eval, real for production.** Every GTM question is relative to now,
        # so the opening prompt states the date -- and a fixture that plants a window relative
        # to a date it was written on drifts as real time passes. `partial_month_false_premise`
        # plants July plus twelve days of August and expects "last month" to mean July; run on
        # 20 September it means August, five weeks after the data stops, and the honest answer
        # becomes "cannot tell" rather than "the premise is false". The scenario was not
        # wrong; it had aged.
        self._today = today

        # Optional, and off by default. Memory is prior context, not evidence: a report can
        # only cite a resolvable evidence_id, so recall can shorten the path to a hypothesis
        # but can never be the grounds for one. Injected rather than constructed here so an
        # eval run -- which has no ingested memory and must not depend on any -- gets the
        # loop exactly as it was.
        self._recall = recall
        # Advisory only. A 90-second investigation that prints nothing is indistinguishable
        # from a hung one, which has already cost real debugging time here -- but losing an
        # event must never affect the investigation, so every emit is guarded.
        self._progress = progress
        # Asked once per step. Cooperative rather than pre-emptive, and it has to cross a
        # process boundary: the loop runs in a Celery worker while the cancel request arrives
        # at the gateway, so a threading flag on the conversation -- the usual shape -- cannot
        # see it. The check reads shared state instead, and the state it reads is the
        # investigation row, which this codebase already treats as authoritative.
        self._cancelled = cancelled
        # Injectable so the wall-clock budget can be tested deterministically. A
        # test that reached the time limit by actually sleeping would be slow and
        # flaky, which in practice means the limit goes untested — and an unenforced
        # time budget is the one most likely to matter under a hung upstream.
        self._now = clock

    async def _is_cancelled(self) -> bool:
        """Whether the caller has asked this investigation to stop.

        A failing check returns False rather than raising: a database blip must not cancel an
        investigation that nobody asked to cancel, and the next step asks again.
        """
        if self._cancelled is None:
            return False
        try:
            return bool(await self._cancelled())
        except Exception:  # noqa: BLE001 - see the docstring
            return False

    def _tool_specs(self, ledger: ReadingLedger) -> list[dict[str, object]]:
        """The tools offered this turn.

        The read tool appears only once something has actually gone behind a handle, and then
        stays for the rest of the investigation. Both halves are deliberate:

          - **Absent by default.** An investigation whose payloads are all small never sees
            it, so a capability that cannot do anything is never described. That is the same
            rule the empty-result warning and the survey follow: a disclosure has to be absent
            on the calls that do not need it or it becomes noise everywhere.
          - **Never withdrawn.** Tool schemas sit in the cached prefix of every request, ahead
            of the transcript, so a spec that appeared and disappeared would invalidate the
            prompt cache at each transition. Caching is worth 72% of the wall clock on the
            loop's model calls; the spec is worth about 120 tokens. `has_handles` is monotone
            for exactly this reason.
        """
        specs = self._registry.llm_tool_specs()
        if ledger.has_handles:
            return [*specs, read_tool_spec()]
        return specs

    def _serve_reads(
        self,
        ledger: ReadingLedger,
        reads: list[ToolRequest],
        step: Step,
        results: dict[str, str],
        spin: SpinDetector,
        index: int,
        observations: int,
    ) -> bool:
        """Serve this turn's read requests from the ledger. Returns whether anything new was read.

        No session, no executor, no await: a read is a dict lookup against an observation this
        investigation already fetched and already persisted. That is the property that makes
        reading cheap, and it is also why nothing here can touch `Evidence` or its hash --
        there is no write path to reach.
        """
        served = 0
        progressed = False
        for request in reads:
            spin.record_call(request.name, request.arguments)
            if served >= MAX_READS_PER_STEP:
                # Refused, but named, so nothing is lost: the model asks again next turn.
                results[request.id] = json.dumps(
                    {
                        "error": (
                            f"only {MAX_READS_PER_STEP} observations can be read in one turn; "
                            "this one was not read. Ask for it in your next message."
                        ),
                        "tool": request.name,
                    }
                )
                continue
            raw_id = request.arguments.get("evidence_id", "")
            try:
                held, first_read = ledger.read(raw_id, step=index)
            except UnknownHandle as exc:
                # Fed back like a tool failure, because it is one of the same kind: a
                # correctable mistake in the arguments rather than a reason to stop.
                step.errors.append(f"{request.name}: {exc}")
                spin.record_failure(request.name, str(exc))
                results[request.id] = json.dumps({"error": str(exc), "tool": request.name})
                continue
            served += 1
            progressed = progressed or first_read
            step.reads.append(held.executed.evidence_id)
            results[request.id] = _render_observation(held.executed, reread=not first_read)
            emit(
                self._progress,
                ProgressEvent(
                    Phase.OBSERVED,
                    detail=(
                        f"read {held.executed.tool_name}.{held.executed.capability} in full "
                        f"({held.digest.rows} rows)"
                    ),
                    step=index + 1,
                    observations=observations,
                ),
            )
        return progressed

    async def _prior_context(self, tenant: TenantContext, question: str) -> str | None:
        """What memory offers about this question, or nothing.

        Failure here is deliberately non-fatal. Recall is an accelerant — it can point the
        loop at the PR that shipped last time this metric moved — and an investigation that
        cannot run because a vector store is unreachable would be a worse product than one
        that runs without its memory. The absence is invisible to the report either way,
        because nothing recalled is ever cited.
        """
        if self._recall is None:
            return None
        emit(self._progress, ProgressEvent(Phase.RECALLING))
        try:
            recalled = await self._recall.recall(tenant, question)
        except Exception:  # noqa: BLE001 - see the docstring
            return None
        return None if recalled.is_empty else recalled.render()

    async def _survey(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        investigation_id: uuid.UUID,
        ledger: ReadingLedger | None = None,
    ) -> tuple[str | None, list[uuid.UUID], int]:
        """Establish what exists before investigating anything.

        Every capability declaring `discovery=True` is called once, with default parameters,
        before the first step. The results go into the opening prompt as observations already
        made, with their evidence ids, so the analyst can both reason from them and cite them.

        **Why this is not simply a tool the analyst may call.** Asked which pull request shipped
        most recently to the company website, it searched `acme/acme` -- the only
        repository name it had ever seen -- and honestly reported that it could not confirm an
        answer. `acme/company-website` exists and the ingest already knew about it. Nothing
        in the tool surface could have told the analyst, because all eight GitHub capabilities
        take a repository name as a parameter.

        Adding a discovery capability alone would have been half a fix, and the literature is
        specific about why. *Agents Explore but Agents Ignore* (arXiv 2604.17609) measures the
        gap between discovering something and acting on it: Terminal-Bench discovery 78.6-81.2%
        against interaction 37.1-50.3%, and in AppWorld a model that observed documentation
        naming the solution in 97.54% of attempts called it in 0.53%.

        Two of its three factors matter here, and one of them cuts against the obvious reading:

          - **Tool availability, inverted.** Richer tooling made agents investigate *less* -- a
            bash-only scaffold roughly doubled interaction against one with a structured editor.
            So adding a discovery tool is not just insufficient; it is the sort of change that
            paper found can suppress the very behaviour it is meant to encourage.
          - **Test-time compute.** Interaction@1 tripled from 11% at low reasoning to 37% at
            high. Cortex runs `effort="low"` on measured evidence, which puts the analyst at the
            low end of the axis that governs whether it investigates an unexpected observation.
            That argues for establishing the environment mechanically rather than buying
            curiosity back at thirty times the cost.

        *Look Before You Leap* (arXiv 2605.16143) supplies the shape -- explore on a fixed
        budget, synthesise a summary, inject it, act. Its own numbers do **not** show that shape
        helping an untrained model (54.4% -> 54.1%, and 30.9% -> 28.7% zero-shot; it gains only
        with exploration-aware training). So the justification for this survey is not borrowed
        from their result: it is that the question above went from "I could not confirm this" to
        a correct cited answer, and from 132 seconds to 52.

        So the environment is established as a fact, the same treatment `registry_for_tenant`
        gives the toolset. The budget is fixed at one call per discovery capability, which cannot
        grow with the question.

        Failures are non-fatal, individually. A connector that will not answer its own
        enumeration is a connector the analyst will discover is broken when it calls it for
        real, and refusing to investigate over it would be worse than proceeding.
        """
        names = self._registry.discovery_capabilities()
        if not names:
            return None, [], 0

        emit(self._progress, ProgressEvent(Phase.SURVEYING))
        # Concurrently, because the survey's calls are independent by definition -- they
        # enumerate different sources and none takes a parameter derived from another. Serially
        # they cost 1.8 seconds warm, and they sit on the critical path of every investigation
        # before any work has started.
        outcomes = await self._executor.execute_many(
            session,
            tenant,
            investigation_id=investigation_id,
            calls=[(name, {}) for name in names],
        )

        blocks: list[str] = []
        evidence_ids: list[uuid.UUID] = []
        for name, outcome in zip(names, outcomes, strict=True):
            if isinstance(outcome, ToolError):
                # Recorded for the analyst rather than swallowed: "I could not list your
                # repositories" is information, and it explains an absence the analyst would
                # otherwise attribute to the data.
                blocks.append(f"{name}: unavailable ({type(outcome).__name__})")
                continue
            evidence_ids.append(outcome.evidence_id)
            # The survey is where the bug's payload actually arrived: `posthog.list_events` is
            # a discovery capability, so the event catalogue that contained the answer was
            # fetched here, before the first step, and pasted whole into the opening prompt.
            # It is therefore the first place a digest applies rather than an afterthought.
            blocks.append(_render_observation(outcome, ledger.offer(outcome) if ledger else None))
            emit(
                self._progress,
                ProgressEvent(Phase.SURVEYING, detail=name, observations=len(evidence_ids)),
            )

        # The connector list is part of the header, and it is not decoration. The first version
        # named only the *resources* found, and the eval regressed on exactly the provider that
        # has no discovery capability: GA4 never appeared in the block, so "anything not on that
        # list, you have not observed" read as "GA4 is not part of this estate" and two of six
        # attempts stopped calling it -- tool_selection 0.67 and 0.00, both missing only GA4
        # capabilities. Naming every available connector separates "these are the resource names
        # you may use" from "these are the only sources you have", which was the confusion.
        connectors = ", ".join(self._registry.tool_names)
        header = (
            "WHAT EXISTS (established before investigating; these are real observations and "
            "you may cite their evidence ids)\n"
            f"Every source available to you: {connectors}. Use all of them as the question "
            "requires.\n"
            "The enumerations below cover only what can be *listed* -- repositories, projects, "
            "events, named queries. Where one appears, use those names verbatim rather than "
            "guessing. Where a source is not enumerated below, that says nothing about it: it "
            "simply has nothing to list, and you should still query it normally."
        )
        return "\n\n".join([header, *blocks]), evidence_ids, len(names)

    async def _inherited_evidence(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        investigation_id: uuid.UUID,
    ) -> list[uuid.UUID]:
        """Evidence ids this investigation may cite but did not gather.

        Empty for anything that is not a follow-up, which is why the drafting prompt and the
        no-evidence check below both behave exactly as they did before for a standalone
        investigation.
        """
        citable = await citable_investigation_ids(session, tenant, investigation_id)
        ancestors = citable - {investigation_id}
        if not ancestors:
            return []
        rows = (
            (
                await session.execute(
                    select(Evidence.id).where(
                        Evidence.tenant_id == tenant.tenant_id,
                        Evidence.investigation_id.in_(ancestors),
                    )
                )
            )
            .scalars()
            .all()
        )
        return list(rows)

    async def investigate(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        *,
        investigation_id: uuid.UUID,
        question: str,
        parent_id: uuid.UUID | None = None,
    ) -> Investigation:
        budget = self._employee.budget
        started = self._now()
        timings = Timings()
        # What has been fetched, what was shown as a digest, and what has been read. Empty and
        # inert for an investigation whose payloads are all small: nothing goes behind a
        # handle, the read tool is never offered, and no turn carries an unread notice.
        ledger = ReadingLedger()

        # Explore before acting: establish what exists, then investigate. Its evidence and its
        # tool calls both count -- they are real observations against real budgets.
        with timings.measure(SURVEY):
            survey, survey_evidence, survey_calls = await self._survey(
                session, tenant, investigation_id, ledger
            )
        with timings.measure(RECALL):
            prior = await self._prior_context(tenant, question)
        # A follow-up is told what came before it: the earlier conclusion, so "and what about
        # mobile?" is interpretable at all, and an inventory of the parent's evidence ids so an
        # observation already paid for can be cited rather than fetched again. Distinct from
        # `prior` above, which is recalled prose and never citable -- this names rows the gate
        # will resolve.
        thread = await prior_context(session, tenant, parent_id) if parent_id is not None else None
        messages: list[Message] = [
            Message(
                role="user",
                content=_opening(
                    question,
                    memory=prior,
                    thread=thread,
                    survey=survey,
                    # Absent unless the survey itself summarised something, which is the
                    # normal case: an opening prompt that explained the read tool with nothing
                    # to read would be a paragraph of instruction about a situation that does
                    # not exist.
                    unread=ledger.reminder(),
                    today=self._today,
                ),
            )
        ]
        steps: list[Step] = []
        evidence_ids: list[uuid.UUID] = [*survey_evidence]
        usage = Usage()
        tool_calls = survey_calls
        # Watches for a loop that is busy without progressing. The previous signal was a
        # bare barren-step counter, which resets whenever *any* evidence is written -- so an
        # analyst repeating one identical call gathered a new row each time and the counter
        # never fired. See cortex/agents/spin.py.
        spin = SpinDetector()
        stop_reason: StopReason = STEP_LIMIT

        for index in range(budget.max_steps):
            # Checked before the model call, which is where the money is. Checking after it
            # would take the cancellation but still pay for the turn.
            if await self._is_cancelled():
                raise InvestigationCancelled(
                    f"cancelled after {index} step(s)",
                    steps=index,
                    evidence=len(evidence_ids),
                )
            elapsed_ms = int((self._now() - started) * 1000)
            # The loop's share is its budget minus the drafting reserve. Without the reserve a
            # loop that used its whole budget left drafting to start from zero against its own
            # deadline, which is how one attempt was lost after 432 seconds -- see
            # DRAFT_RESERVE_SECONDS.
            loop_seconds = max(1.0, budget.max_seconds - DRAFT_RESERVE_SECONDS)
            if elapsed_ms > loop_seconds * 1000:
                stop_reason = TIME_LIMIT
                break
            if usage.total > budget.max_tokens:
                stop_reason = TOKEN_LIMIT
                break

            emit(self._progress, ProgressEvent(Phase.THINKING, step=index + 1))
            # Checked before the call, not diagnosed after it. F-12 surfaced as an opaque
            # provider 400 mid-run, after the tokens were spent; this names the offending
            # turn instead. See cortex/agents/transcript.py.
            assert_valid(messages)
            try:
                with timings.measure(LOOP_MODEL):
                    response = await self._llm.complete(
                        system=self._employee.system_prompt,
                        messages=messages,
                        tools=self._tool_specs(ledger),
                        max_tokens=8192,
                    )
            except LLMAuthenticationFailed as exc:
                # Stopped regardless of what was gathered. Reporting on partial evidence is the
                # right move for a transient failure and pointless for a rejected credential:
                # the drafting call uses the same credential, so the run ends either way -- and
                # it ends naming drafting rather than the key, which is the wrong place to look.
                raise InvestigationFailed(f"the analyst could not authenticate: {exc}") from exc
            except LLMError as exc:
                if not evidence_ids:
                    # Nothing gathered and the model is unavailable: there is no
                    # partial answer worth grounding.
                    raise InvestigationFailed(f"the analyst could not run: {exc}") from exc
                # Evidence exists, so stop and report what was found rather than
                # discarding it.
                steps.append(Step(index=index, note="analyst unavailable", errors=[str(exc)]))
                stop_reason = STALLED
                break

            usage = usage + response.usage
            timings.record_usage(LOOP_MODEL, response.usage)

            if not response.wants_tools:
                # No tool call means the analyst considers the investigation done.
                #
                # A reflection turn was tried here and removed on measurement: one injected
                # message asking the analyst to reconsider its empty observations before
                # concluding. The literature was encouraging — arXiv 2604.17609 moved
                # Terminal-Bench interaction@1 from 37.12% to 53.33% with exactly that shape —
                # and on this suite it was clearly harmful: 5/6 instead of 6/6, tool_selection
                # back to 0.67, insufficient_evidence at 206s against a 90s budget, and one
                # attempt lost entirely when the lengthened transcript pushed the drafting call
                # past its 300-second deadline after 432 seconds of work. See
                # docs/eval-results.md run 15. The finding did not transfer; the measurement is
                # what decides.
                steps.append(Step(index=index, note=response.text[:2000] or "concluded"))
                stop_reason = CONCLUDED
                break

            # A read is separated from a call before the budget is checked, because it is not
            # a call: it reaches no upstream, mints no evidence and spends no wall clock. If
            # it consumed a slot in a 40-call budget then reading would compete with
            # gathering, and a mechanism whose whole purpose is to make reading cheap would
            # have priced it at one observation. It is bounded instead by MAX_READS_PER_STEP
            # and by the spin detector, which stops a loop that reads the same handle three
            # times.
            reads = [r for r in response.tool_requests if r.name == READ_TOOL]
            calls = [r for r in response.tool_requests if r.name != READ_TOOL]

            if tool_calls + len(calls) > budget.max_tool_calls:
                stop_reason = TOOL_CALL_LIMIT
                break

            step = Step(index=index, note=response.text[:2000])
            results: dict[str, str] = {}

            # Announced before any are executed, so a reader sees the whole batch rather
            # than watching it appear one call at a time.
            for request in calls:
                tool_calls += 1
                step.tool_calls.append(request.name)
                spin.record_call(request.name, request.arguments)
                emit(
                    self._progress,
                    ProgressEvent(
                        Phase.CALLING,
                        detail=_describe_call(request),
                        step=index + 1,
                        observations=len(evidence_ids),
                    ),
                )

            # Reads are served from the ledger before the network runs, so a turn that both
            # reads and gathers spends no extra wall clock on the read.
            read_progress = self._serve_reads(
                ledger, reads, step, results, spin, index, len(evidence_ids)
            )

            # Executed as a batch, overlapping only the network time. A step's calls are
            # independent by construction -- the model chose them all before seeing any
            # result -- and serially they were 17% of a 143-second investigation: 8 calls at
            # a mean of 3.0 seconds. `execute_many` keeps every database write and every
            # F-01/F-05 guarantee serial; only `capability.handler` overlaps.
            with timings.measure(LOOP_TOOLS):
                outcomes = await self._executor.execute_many(
                    session,
                    tenant,
                    investigation_id=investigation_id,
                    calls=[(request.name, request.arguments) for request in calls],
                )

            for request, outcome in zip(calls, outcomes, strict=True):
                if isinstance(outcome, ToolError):
                    # Fed back, not raised. A missing credential or a rate limit is
                    # something the analyst can route around, and reporting the gap
                    # beats crashing the investigation.
                    message = f"{type(outcome).__name__}: {outcome}"
                    step.errors.append(f"{request.name}: {message}")
                    spin.record_failure(request.name, message)
                    emit(
                        self._progress,
                        ProgressEvent(
                            Phase.TOOL_FAILED,
                            detail=f"{request.name} failed: {message}"[:200],
                            step=index + 1,
                            observations=len(evidence_ids),
                        ),
                    )
                    results[request.id] = json.dumps({"error": message, "tool": request.name})
                    continue

                evidence_ids.append(outcome.evidence_id)
                step.evidence_ids.append(outcome.evidence_id)
                spin.record_observation(outcome.payload_hash)
                results[request.id] = _render_observation(outcome, ledger.offer(outcome))
                emit(
                    self._progress,
                    ProgressEvent(
                        Phase.OBSERVED,
                        detail=request.name,
                        step=index + 1,
                        observations=len(evidence_ids),
                    ),
                )

            steps.append(step)
            messages.append(
                Message(
                    role="assistant",
                    content=response.text,
                    # Carried so the results appended next have something to pair with.
                    tool_requests=tuple(response.tool_requests),
                )
            )
            # The next turn's width, injected with the results it follows. W&D's schedule is
            # per-step; a single instruction in the opening prompt would be the constant arm, which
            # is the one that scored 68% against Descending's 74%.
            messages.append(
                Message(
                    role="user",
                    tool_results=results,
                    # The unread notice rides with the width instruction rather than being its
                    # own turn: it is the same kind of thing -- what to do next -- and an extra
                    # message per step would grow the transcript this mechanism exists to
                    # shrink. Empty whenever nothing is unread.
                    content=_next_turn(index + 1, ledger),
                )
            )

            # Reading a summarised observation for the first time is progress, so it must not
            # count toward the barren streak. Otherwise an analyst that spends a turn reading
            # what it fetched -- exactly the behaviour this whole mechanism is asking for --
            # walks into a stall, and reading becomes the expensive option again. Re-reading
            # something already read is *not* progress, and the spin detector's identical-call
            # rule stops it at three.
            spin.record_step(gathered_evidence=bool(step.evidence_ids) or read_progress)
            detected = spin.spinning(max_barren_steps=budget.max_steps_without_new_evidence)
            if detected is not None:
                # Recorded on the step, so the report can say *why* it stopped rather than
                # only that it did. "Stopped early" and "stopped early because it asked the
                # same question three times" call for different fixes.
                step.errors.append(f"stalled — {detected.reason}")
                stop_reason = STALLED
                break

        # Inherited evidence counts. A follow-up may legitimately need no new calls -- "which
        # of those days was highest" is answerable from observations the parent already made --
        # and refusing to draft in that case would make the cheapest, best-behaved follow-up the
        # only one that cannot produce a report. The grounding guarantee is unchanged: every
        # claim still cites a row the gate resolves, and the gate resolves within the thread.
        #
        # Found by a live follow-up that concluded in one step and was then refused, which is
        # exactly the behaviour the feature is meant to reward.
        # The parent's evidence, if this is a follow-up. Loaded once and used twice: to decide
        # whether a draft is possible at all, and -- the part a live run caught -- to tell the
        # drafting call which ids it may cite. Listing only this run's evidence meant a follow-up
        # that correctly called nothing was told it could cite nothing, and then failed drafting
        # repeatedly against a schema requiring every claim to carry at least one id.
        inherited_ids = await self._inherited_evidence(session, tenant, investigation_id)
        if not evidence_ids and not inherited_ids:
            # Refused rather than drafted. A report with no evidence cannot be
            # grounded, and drafting one would produce a fluent answer with nothing
            # behind it — the exact failure Cortex exists to prevent.
            raise InvestigationFailed(
                f"the investigation gathered no evidence (stopped: {stop_reason}); "
                "there is nothing to ground a report on"
            )

        emit(
            self._progress,
            # Counts what the draft may actually cite, not only what this run fetched. A
            # follow-up that reused its parent's observations reported "drafting the report from
            # 0 observation(s)" while drafting from fourteen, which reads as a bug in the loop.
            ProgressEvent(Phase.DRAFTING, observations=len(evidence_ids) + len(inherited_ids)),
        )
        report, draft_usage = await self._draft(
            question,
            messages,
            # Both sets, this run's first. A follow-up may cite either, and the gate resolves
            # both -- so a drafting prompt that omitted the inherited ids would forbid exactly
            # the reuse the thread exists to enable.
            [*evidence_ids, *inherited_ids],
            stop_reason,
            timings,
            # The last point at which an ignored observation can still be disclosed. It cannot
            # be fixed here -- the loop is over -- but a report drafted over an unread
            # observation must not describe it as examined.
            unread=ledger.unread(),
            # Appended to, not read: a draft that had to be shortened is a step the reader
            # is entitled to see. It is the one event in the run that changes what the
            # report says without anything in the report saying so.
            steps=steps,
        )
        return Investigation(
            report=report,
            steps=steps,
            evidence_ids=evidence_ids,
            usage=usage + draft_usage,
            stop_reason=stop_reason,
            duration_ms=int((self._now() - started) * 1000),
            timings=timings,
            unread_observations=ledger.unread(),
            observations_read=ledger.read_count,
        )

    # ------------------------------------------------------------------ drafting

    async def _draft(
        self,
        question: str,
        messages: list[Message],
        evidence_ids: list[uuid.UUID],
        stop_reason: StopReason,
        timings: Timings | None = None,
        unread: tuple[Unread, ...] = (),
        steps: list[Step] | None = None,
    ) -> tuple[InvestigationReport, Usage]:
        """Turn the gathered evidence into a structured report.

        A separate call from the loop on purpose: the loop's job is to gather, and
        a model asked to gather and conclude in one turn tends to conclude early.
        """
        instruction = _drafting_instruction(question, evidence_ids, stop_reason, unread)
        draft_messages = [*messages, Message(role="user", content=instruction)]
        usage = Usage()

        # One repair attempt. The schema constrains shape but not the model-level
        # invariants, so a draft can be well-formed JSON and still invalid — and a whole
        # investigation's evidence was once discarded at this point over a single
        # hypothesis. Handing the validation error back is far cheaper than re-running
        # the loop, and a model told exactly what it violated usually fixes it.
        clock = timings if timings is not None else Timings()

        # Two independent budgets, deliberately not shared. A truncated draft and an invalid
        # draft fail for unrelated reasons, and spending the repair attempt on truncation
        # would leave a report that came back short *and* invalid with nothing left to fix
        # it -- the two most likely to co-occur, since both mean the model was struggling.
        repairs_left = _DRAFT_REPAIR_ATTEMPTS
        brevity_retries_left = _DRAFT_BREVITY_RETRIES
        while True:
            attempt = _DRAFT_REPAIR_ATTEMPTS - repairs_left
            try:
                # Measured across every attempt, including a repair: two drafting calls
                # is the honest cost of that path, and averaging it away would hide the
                # thing worth knowing.
                with clock.measure(DRAFT):
                    payload, draft_usage = await self._llm.structured(
                        system=self._employee.system_prompt,
                        messages=draft_messages,
                        schema=llm_report_schema(),
                        max_tokens=DRAFT_MAX_TOKENS,
                        timeout=DRAFT_TIMEOUT_SECONDS,
                        # **Cached only if the repair reads it back, and it mostly does not.**
                        # A write bills at 1.25x fresh input and a read at 0.1x, so caching
                        # this call pays only when a second draft follows: break-even is a
                        # repair rate of 28%. Measured across a full suite, nine drafts wrote
                        # and none read -- the comment above, which called this "the largest
                        # and most cacheable request we make", was right about the size and
                        # wrong about the rest. Caching it costs $0.028 an investigation.
                        cacheable=False,
                    )
            except LLMOutputTruncated as exc:
                # Distinguished from a refusal because the responses are opposite: a refusal
                # will not answer, so retrying is waste, while truncation *was* answering
                # and ran out of room. Discarding the investigation here throws away every
                # tool call already made, which is the whole cost of the run.
                if brevity_retries_left <= 0:
                    raise InvestigationFailed(f"the report could not be drafted: {exc}") from exc
                brevity_retries_left -= 1
                if steps is not None:
                    steps.append(
                        Step(
                            index=len(steps),
                            note=(
                                "the first report ran past the output limit; drafted again, shorter"
                            ),
                            errors=[str(exc)],
                        )
                    )
                draft_messages = [
                    *draft_messages,
                    Message(role="user", content=_BREVITY_RETRY_INSTRUCTION),
                ]
                continue
            except LLMError as exc:
                raise InvestigationFailed(f"the report could not be drafted: {exc}") from exc

            usage = usage + draft_usage
            clock.record_usage(DRAFT, draft_usage)
            payload.setdefault("question", question)
            try:
                # Validated again here even though the provider constrained the schema:
                # a near-miss must not reach the gate, which assumes a well-formed report.
                report = InvestigationReport.model_validate(payload)
                if _is_degenerate(report, evidence_ids):
                    raise _DegenerateDraft(
                        "the draft has no findings and no hypotheses, from "
                        f"{len(evidence_ids)} observation(s)"
                    )
                return report, usage
            except Exception as exc:
                if repairs_left <= 0:
                    raise InvestigationFailed(
                        f"the drafted report did not satisfy the report schema after "
                        f"{attempt + 1} attempt(s): {exc}"
                    ) from exc
                repairs_left -= 1
                draft_messages = [
                    *draft_messages,
                    Message(role="assistant", content=json.dumps(payload)[:20000]),
                    Message(
                        role="user",
                        content=(
                            "That report was rejected by validation:\n"
                            f"{exc}\n\n"
                            "Return the whole report again with only that defect fixed. "
                            "Change nothing else, and do not drop any finding."
                        ),
                    ),
                ]

        # No fallthrough: every path through the loop returns, raises, or continues with
        # one of the two budgets decremented, so the loop cannot spin.


class _DegenerateDraft(Exception):
    """A draft that satisfies the schema and is not a report.

    Raised into the existing repair loop rather than being a separate mechanism, because the
    repair loop already has what is needed: the draft, the evidence, and one more attempt.
    """


def _is_degenerate(report: InvestigationReport, evidence_ids: list[uuid.UUID]) -> bool:
    """Whether a schema-valid draft is a stub rather than an answer.

    **The failure this catches, measured.** Two attempts in fifteen returned a report whose only
    summary claim was the literal text "placeholder", with no findings, no hypotheses and no
    risks -- 95 and 112 output tokens against 1,400 to 4,700 on healthy drafts. The loop had run
    normally and gathered evidence in both cases; it was the drafting call that collapsed.

    Nothing noticed. A placeholder claim is *valid* -- `Claim.text` requires one character -- so
    the repair retry never fired, and the collapse surfaced three phases later as a citation gate
    rejection that discarded the whole investigation. The most expensive possible place to find it.

    The test is structural rather than a search for "placeholder", which would be the fifth
    keyword list standing in for meaning in this codebase. Across thirteen healthy drafts every
    one had at least one finding; the false-premise reports legitimately had zero *hypotheses*,
    which is what `GUIDANCE[Shape.FACTUAL]` asks for when nothing needs explaining. Neither had
    zero of both. So: no findings and no hypotheses, after a loop that actually ran, is not a
    report.

    Gated on evidence rather than on turns: an investigation that gathered nothing has nothing
    to write a finding from, and failing it for that would report the wrong problem -- the
    absence of evidence, which the loop already reports.

    **The exemption, added after this test rejected a correct report twice.** The paragraph above
    assumed no real report has zero of both, on thirteen drafts. Run 30 produced the
    counter-example on `partial_month_false_premise`: a question whose answer is "nothing
    happened" has no cause to hypothesise about *and* may carry its whole answer in the premise
    check and the summary -- which is what `GUIDANCE[Shape.FACTUAL]` asks for. The draft was
    refused twice and the investigation was discarded.

    So zero-of-both is no longer sufficient. A report that recorded a premise verdict and wrote
    the check behind it has done work a collapsed draft does not: both fields default to "nothing
    was decided" (`NONE_ASSERTED` and `""`), and the placeholder drafts left them there. That is
    a structural difference rather than a length threshold, which is what this test wanted in the
    first place.
    """
    if not evidence_ids or report.findings or report.hypotheses:
        return False
    return not _answers_by_premise(report)


#: A premise check that actually checked something cites what it measured. Every real one does:
#: "August is 12 days old: 157.0 signups/day against 156.4/day across all of July." A stub
#: writes "placeholder", "n/a", or restates the question back.
_CHECK_CITES_A_FIGURE = re.compile(r"\d")

#: Verdicts under which a report may legitimately carry no findings and no hypotheses.
#:
#: `HOLDS` is excluded, and that is the point: if the premise holds then the movement really
#: happened, and a report saying nothing about it is a stub whatever it wrote in the check.
#: `HOLDS` is also what a collapsing draft defaults to, having decided nothing.
_ANSWERABLE_BY_PREMISE_ALONE = frozenset({PremiseVerdict.FALSE, PremiseVerdict.UNVERIFIABLE})


def _answers_by_premise(report: InvestigationReport) -> bool:
    """Whether the report's answer *is* its premise verdict.

    True for a refutation of a false premise, which needs no findings and no hypotheses to be
    complete. False for a stub.

    **The first version of this was reachable by every stub it existed to catch.** It asked only
    that `premise` be set and `premise_checked` be non-empty -- and both are filled by habit
    rather than by work: `llm_report_schema()` emits them at indices 1 and 2, *before* findings,
    and `_PREMISE_CHECK` in `shape.py` instructs every report to set them. So a draft that
    collapsed on findings had already filled both. `premise=holds, premise_checked="."` was
    exempt, as was a check that merely restated the question.

    Three conditions now, each closing one of those: the verdict has to be one a report can
    legitimately answer with alone, the check has to cite a figure, and it has to be long enough
    to be a sentence. Deliberately structural -- a search for the word "placeholder" would be
    the fifth keyword list in this codebase standing in for meaning.
    """
    checked = report.premise_checked.strip()
    return (
        report.premise in _ANSWERABLE_BY_PREMISE_ALONE
        and len(checked) >= 40
        and bool(_CHECK_CITES_A_FIGURE.search(checked))
    )


def _opening(
    question: str,
    *,
    today: date | None = None,
    memory: str | None = None,
    thread: str | None = None,
    survey: str | None = None,
    unread: str = "",
) -> str:
    """The opening instruction, including today's date.

    The date is stated because a model does not know it, and every GTM question is
    relative to now: "last week", "the last 60 days", "in June". A real investigation
    against live HubSpot data queried 2025 ranges, got zero rows for both windows, and
    spent two steps discovering the current period from record timestamps before
    recovering. It disclosed the error honestly, which is the system working — but the
    two wasted steps came out of a budget meant for investigating.
    """
    stamp = (today or date.today()).isoformat()
    # Memory comes *before* the question, and labelled as memory. After it, the model has
    # already read the instruction that only tool results carry evidence ids; before it,
    # recalled prose is the last thing seen before drafting begins, which is exactly the
    # position from which an uncited claim gets written.
    prior = f"{memory}\n\n" if memory else ""
    # The thread block sits after memory and before the question, and unlike memory it names
    # evidence ids the gate will resolve. Both precede the question so that the last thing read
    # before investigating is the question itself.
    preceding = f"{thread}\n\n" if thread else ""
    # The survey comes first of all. It is the only block that constrains what the *other*
    # blocks can mean: a repository or project named nowhere in it does not exist, and a
    # question about "the website" has to be resolved against the list before anything else.
    environment = f"{survey}\n\n" if survey else ""
    follow_up = (
        "\nThis is a follow-up. Read the earlier conclusion as context, not as evidence: cite "
        "the observation ids listed above, never the earlier prose. Do not re-answer the "
        "earlier question -- answer this one, and gather only what the listed observations do "
        "not already establish.\n"
        if thread
        else ""
    )
    return (
        f"Today is {stamp}. Interpret every relative period in the question against that "
        "date, and state the absolute date range you used.\n\n"
        f"{environment}"
        f"{prior}"
        f"{preceding}"
        f"Question: {question}\n"
        f"{follow_up}\n"
        "Investigate this. Start by confirming and measuring the change, then find "
        "which segment drove it, then test explanations — including at least one "
        "you expect to be wrong. Call tools to gather evidence; when you have "
        "enough to explain the change and rule out the obvious alternatives, stop "
        "calling tools and say so.\n\n"
        + width_instruction(0)
        # After the instructions rather than inside the survey block, because it is an
        # instruction about what to do next and not part of the environment. Empty unless the
        # survey summarised something -- `posthog.list_events` is a discovery capability, so
        # this is exactly where our bug's payload arrived.
        + (f"\n\n{unread}" if unread else "")
    )


def _next_turn(index: int, ledger: ReadingLedger) -> str:
    """What to send with a turn's tool results: the next width, and anything left unread.

    Two instructions rather than one message each, and in this order, because the width
    instruction is about the next call and the unread notice is about a call already made --
    reading a result you already hold should be weighed before deciding what else to fetch.
    """
    reminder = ledger.reminder()
    width = width_instruction(index)
    return f"{width}\n\n{reminder}" if reminder else width


def width_instruction(index: int) -> str:
    """The per-turn request for how many calls to make.

    Stated as a number the loop decides rather than as advice the model weighs, because W&D
    measured the alternative: their Automatic arm, where the model chose its own width, scored 72%
    at 26.6 turns against Descending's 74% at 23.5, and they conclude the LLM "cannot determine the
    optimal number of tool calls in each iteration".

    The escape hatch survives, because their limitation section is explicit that they do not
    address tools whose arguments depend on a previous tool's output -- and we have exactly those
    chains. A number with no exemption would be met by guessing a repository name rather than
    looking it up.
    """
    wanted = tool_calls_for_step(index)
    if wanted <= 1:
        return (
            "This late in an investigation the remaining questions usually depend on what you "
            "have already found, so ask for as few calls as the next step actually needs — one is "
            "normal here. Do not pad."
        )
    return (
        f"Request about {wanted} tool calls in this response, in the SAME message, choosing "
        "observations that are independent of each other: several events, several time periods, "
        "several repositories, a metric and the change record that might explain it. Each round "
        "trip costs the reader roughly twenty seconds, so a turn that asks one question when it "
        f"could have asked {wanted} is a turn wasted.\n"
        "Ask for fewer only when you genuinely need one answer before you know what to ask next — "
        "a repository you have not identified, an event name you have not confirmed. Never invent "
        "calls to reach the number, and never guess a parameter you could look up first."
    )


def _describe_call(request: ToolRequest) -> str:
    """A tool call as one readable line: the capability plus its most telling argument.

    The arguments matter more than the name here. "calling slack__search_messages" says
    almost nothing; "calling slack__search_messages query='Apollo deanonymizer'" is the
    line that lets a watching human notice the analyst is searching for the wrong thing
    while it is still cheap to interrupt.

    Arguments are truncated and only a few are shown, because the point is a glance rather
    than a transcript -- the full parameters are on the evidence row either way.
    """
    interesting = ("query", "repo", "topic", "event", "metric", "number", "sql", "steps")
    parts = [
        f"{key}={_short(value)}"
        for key, value in request.arguments.items()
        if key in interesting and value not in (None, "", [])
    ]
    return f"{request.name} {' '.join(parts[:2])}".strip()


def _short(value: object) -> str:
    text = ", ".join(str(v) for v in value) if isinstance(value, list | tuple) else str(value)
    return repr(text if len(text) <= 60 else text[:57] + "...")


def _render_observation(
    executed: ExecutedTool, digest: Digest | None = None, *, reread: bool = False
) -> str:
    """Return one observation to the model, with its citable id.

    The `evidence_id` is first and explicitly labelled: it is the only part of this
    payload the report cannot be written without, and burying it inside a JSON blob
    measurably lowers how reliably it gets cited.

    With a `digest`, the rows are withheld and the digest is shown in their place. Everything
    else is identical -- same header, same id, same emptiness warning -- because the digest
    changes what the analyst has *read*, not what it has observed or may cite. The stored
    payload and its hash are untouched: a claim citing this id is still checked against every
    row by the grounding gate and the verifier, whether or not the loop read them.
    """
    header = (
        f"evidence_id: {executed.evidence_id}\n"
        f"source: {executed.tool_name}.{executed.capability}\n"
        f"freshness: {executed.freshness.value}\n"
    )
    if executed.source_ref:
        header += f"source_ref: {executed.source_ref}\n"
    if digest is None:
        note = (
            "\nYou asked to read this observation, so here are all of its rows. You have read "
            "it before; nothing has changed.\n"
            if reread
            else ""
        )
        return (
            f"{header}{_empty_warning(executed)}{note}\nobservation:\n"
            f"{render_payload(executed.payload)}"
        )
    return f"{header}{_empty_warning(executed)}{_digest_notice(executed, digest)}"


def _digest_notice(executed: ExecutedTool, digest: Digest) -> str:
    """The block that stands in for a bulk observation's rows.

    Three sentences it must carry, and each of them is load-bearing:

      - **What is missing.** How many rows, and how large the full thing is, so "some of it"
        is a number rather than a feeling.
      - **That the digest is not a prefix.** Showing the first N rows of a series whose
        interesting fact is at the end is our original bug in a new costume, so the analyst is
        told exactly what the digest covers: every row, no ordering.
      - **How to read it, and that reading is free.** The alternative to a cheap read is not a
        careful analyst; it is an analyst reasoning from the digest and reporting a conclusion
        it never checked.
    """
    return (
        f"\nSUMMARY ONLY -- {digest.rows} rows are NOT shown. The full observation is "
        f"{digest.full_chars:,} characters; below is a digest computed over EVERY row of it "
        "(counts, ranges, clusters), not the first few, so nothing here depends on the order "
        "the rows arrived in. The rows themselves are stored and unread.\n"
        f"To read all {digest.rows} rows, call {READ_TOOL} with "
        f"evidence_id={executed.evidence_id}. It makes no upstream request, takes no "
        "measurable time, and does not count against your tool-call budget. Until you do, do "
        "not describe this observation as examined, and do not conclude anything that depends "
        "on which rows are in it.\n"
        f"\ndigest:\n{digest.text}"
    )


def _empty_warning(executed: ExecutedTool) -> str:
    """The sentence an empty observation must arrive with.

    This exists because the same bug has now shipped four times: Slack returned nothing for
    an over-narrow query, GitHub returned nothing for an environment that does not exist,
    PostHog returned nothing because the wrong project was queried, and Slack returned empty
    text because apps post with Block Kit. In every case the analyst reasoned correctly from
    an empty result and reported a confident, properly cited, wrong conclusion — and the
    grounding machinery could not help, because the citation was real and the result
    genuinely was empty.

    Naming the pattern in four commit messages did not prevent the fifth. Stating it in the
    observation does something a comment cannot: the analyst cannot read this result without
    reading that empty is not the same as absent, and cannot fail to see the filters that
    produced it.

    Filters are echoed from the params the executor recorded, because "no deploys" and "no
    deploys *in production*" are different findings and only one of them is about the data.
    """
    if not executed.is_empty:
        return ""
    lines = [
        "",
        "THIS RESULT IS EMPTY. An empty result is not evidence that nothing happened — it",
        "is equally consistent with a filter that excluded everything, a name that does not",
        "exist upstream, or a source you cannot see. Before concluding an absence from this,",
        "widen or remove one filter and look again, or say plainly that you could not see.",
    ]
    if executed.empty_hint:
        # The tool's own account of what it *could* have returned. This is the line that
        # turns "nothing found" into "you asked for an environment that does not exist".
        lines.append(f"What the source says about it: {executed.empty_hint}")
    return "\n".join(lines) + "\n"


def _drafting_instruction(
    question: str,
    evidence_ids: list[uuid.UUID],
    stop_reason: StopReason,
    unread: tuple[Unread, ...] = (),
) -> str:
    ids = "\n".join(f"  - {eid}" for eid in evidence_ids)
    note = ""
    if stop_reason in (STEP_LIMIT, TOOL_CALL_LIMIT, TOKEN_LIMIT, TIME_LIMIT, STALLED):
        # Disclosed so the report can say the investigation was cut short. A report
        # that presents a truncated investigation as a complete one is misleading
        # even when every individual claim is grounded.
        note = (
            f"\nThis investigation stopped early ({stop_reason}) rather than because "
            "it was finished. Reflect that in your confidence, and record what "
            "remains unexamined as a risk.\n"
        )
    if unread:
        # The last place the omission can be disclosed. It cannot be fixed here -- the loop is
        # over and no tool can be called -- but a report drafted over an observation nobody
        # read must not present it as examined, and the thing left unexamined is a risk in
        # exactly the sense the section already means. Absent when nothing is unread, which is
        # the common case.
        listed = "\n".join(item.line for item in unread)
        note += (
            f"\n{len(unread)} observation(s) were fetched and never read in full -- you saw "
            f"only a digest of each:\n{listed}\n"
            "You may still cite them for what the digest states, and only for that. Do not "
            "describe them as examined, and record the unread rows as a risk: they are the "
            "part of the evidence this investigation held and did not look at.\n"
        )

    # Decided here rather than left to the model. A lookup answered with four findings,
    # three hypotheses and a recommendations section is correct and unreadable; the shape
    # is computed so the instruction can state which one applies, and so the choice is
    # testable without a model call. `shape_for` defaults to CAUSAL, so the mistake it can
    # make is a full report for a simple question rather than a thin one for a hard one.
    guidance = GUIDANCE[shape_for(question)]

    # Structural gaps in the question itself, decided in code for the reason the research is
    # emphatic about: a model cannot estimate its own need to clarify, and six independent
    # measurements say so. What it *can* do is state the reading it took, and the whole
    # intervention is that stating it is free while a silent wrong reading is not -- a reader who
    # sees "read as against the prior 30 days" re-asks in one line, and a reader who sees nothing
    # is misinformed without knowing it.
    for ambiguity in ambiguities(question):
        guidance += "\n" + AMBIGUITY_GUIDANCE[ambiguity]

    return (
        f"Now write the report answering: {question}\n"
        f"{note}\n"
        f"{guidance}\n\n"
        "You may cite only these evidence ids, which are the observations you "
        f"actually received:\n{ids}\n\n"
        "Every claim must carry the ids of the observations that establish it. A "
        "claim citing anything not on this list will be removed before the report "
        "is shown, and the finding will be lost with it.\n\n"
        # The compound-claim rule, added because losing these sentences costs the *answer*.
        # A sentence relating two observations is checked against exactly the ids it carries,
        # and the drafter kept citing only the one the sentence appears to be about: "the
        # campaign ended on 14 June, one day before the drop began" citing the Slack message
        # alone. The verifier removes it correctly -- when the drop began is not in that
        # message -- and the summary is left describing a 68.4% collapse in paid search with
        # nothing about the exhausted budget behind it. The reader gets the mechanism and not
        # the thing to act on, while `accuracy` still reads 1.00 because the cause survives in
        # a finding. Measured twice on `campaign_traffic_drop` in run 32.
        "**A claim that relates two observations must cite both.** Each claim is checked "
        "against only the ids it carries, by a reader who cannot see the rest of the "
        "report. So a sentence saying one thing happened before, after, during or because "
        "of another needs the id for each side of that relationship, and a sentence "
        "comparing two figures needs the id behind each figure. Citing only the "
        "observation the sentence is *about* is the common mistake: 'the campaign ended on "
        "the 14th, the day before the drop began' needs the record of the campaign ending "
        "*and* the series that shows when the drop began. If you cannot cite both, state "
        "only the half you can establish, in its own sentence — a narrower claim that "
        "survives is worth more than a fuller one that is removed.\n\n"
        "Write the evidence ids in the evidence_ids field only — never inside the "
        "claim text.\n\n"
        # These are Pydantic validators, so they cannot be expressed in the JSON Schema
        # the model is given, and it has no way to infer them. A live investigation
        # against real HubSpot data gathered all of its evidence and was then thrown away
        # at the final parse for exactly this: a contradicted hypothesis with no
        # contradicting evidence. Stating the rules is the cheap half of the fix.
        "Two rules the schema cannot express, and a report that breaks either is "
        "rejected whole:\n"
        "  - A hypothesis with verdict 'supported' must list at least one id in "
        "supporting_evidence_ids.\n"
        "  - A hypothesis with verdict 'contradicted' must list at least one id in "
        "contradicting_evidence_ids — the observation that actually breaks it.\n"
        "If you ruled something out by reasoning rather than by a specific observation, "
        "use verdict 'inconclusive' instead, which carries no such requirement.\n\n"
        # Charts are deliberately NOT requested here. They are withheld from the drafting
        # schema (see `llm_report_schema`) because the compiled grammar exceeds the
        # provider's limit with them, and because a chart's series must be derived from
        # stored evidence rather than authored — a model writing points inline can write a
        # number nobody observed. `charts_from_evidence` builds them in the gate instead,
        # from the same rows the claims cite. Asking for them here produced nothing, which
        # is how this comment came to exist.
    )
