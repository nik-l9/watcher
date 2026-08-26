"""The investigation loop.

Deterministic: every test drives a RecordedLLM against real Postgres, so the loop,
the executor and the evidence store are all exercised without a network call.

The properties worth pinning are the ones a framework would not give us — evidence
ids reaching the model so claims can cite them, budgets that actually bind, and a
refusal to draft a report with nothing behind it.
"""

from __future__ import annotations

import json
import uuid

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.agents.employee import Budget, Employee, gtm_data_analyst
from cortex.agents.investigator import (
    CONCLUDED,
    STALLED,
    STEP_LIMIT,
    TIME_LIMIT,
    TOKEN_LIMIT,
    TOOL_CALL_LIMIT,
    InvestigationFailed,
    Investigator,
    _opening,
)
from cortex.agents.llm import (
    LLMOutputTruncated,
    LLMRefusedStructure,
    LLMResponse,
    RecordedLLM,
    ToolRequest,
    Usage,
)
from cortex.db.models import Evidence, Investigation, Tenant, ToolCall
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.reports.schema import (
    Claim,
    Confidence,
    Finding,
    Hypothesis,
    InvestigationReport,
    Verdict,
)
from cortex.tenancy.context import TenantContext
from cortex.tools.base import Capability, Tool, ToolContext, ToolRegistry, ToolResult
from cortex.tools.executor import ToolExecutor

SCHEMA = {"type": "object", "additionalProperties": False, "properties": {}}


class _ProbeTool(Tool):
    name = "probe"
    provider = None

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.invocations = 0
        super().__init__()

    def capabilities(self) -> list[Capability]:
        return [
            Capability(
                name="observe",
                description="Return a structured observation for testing the loop.",
                params_schema=SCHEMA,
                handler=self.observe,
                result_key="rows",
            )
        ]

    async def observe(self, ctx: ToolContext) -> ToolResult:
        self.invocations += 1
        if self.fail:
            from cortex.tools.base import UpstreamError

            raise UpstreamError("probe: upstream is unavailable")
        # The payload carries `rows`, matching the declared result_key above. It did not until
        # the reflection turn was added: the capability declared "rows" and returned a payload
        # without one, so `Capability.is_empty` correctly judged every observation in this
        # suite empty. Nothing depended on that until something started reading the flag, at
        # which point the whole happy path began ending in a reflection and then a stall. The
        # F-24 mechanism was right and this fixture was lying to it.
        return ToolResult(
            payload={"rows": [{"sessions": 1200, "conversion": 0.029}]},
            source_ref="probe://observation",
        )


def _employee(**budget_overrides: int) -> Employee:
    base = gtm_data_analyst()
    budget = base.budget.model_copy(update=budget_overrides)
    return base.model_copy(update={"budget": budget})


def _tool_turn(n: int = 1) -> LLMResponse:
    return LLMResponse(
        text="Checking sessions.",
        tool_requests=[
            ToolRequest(id=f"call_{uuid.uuid4().hex[:8]}", name="probe__observe", arguments={})
            for _ in range(n)
        ],
        usage=Usage(input_tokens=500, output_tokens=100),
    )


def _final_turn(text: str = "I have enough to conclude.") -> LLMResponse:
    return LLMResponse(text=text, usage=Usage(input_tokens=400, output_tokens=80))


def _draft(evidence_ids: list[uuid.UUID]) -> dict:
    """A minimal but non-degenerate draft.

    The finding is not decoration. `_is_degenerate` rejects a draft with no findings *and* no
    hypotheses from an investigation that gathered evidence, because two live attempts returned
    exactly that -- one "placeholder" claim and nothing else -- and nothing noticed until the
    citation gate discarded the whole investigation. This builder produced that same shape, so
    nineteen tests about cancellation, memory recall and repair were failing on a check that had
    nothing to do with what they were testing.
    """
    cited = [str(e) for e in evidence_ids]
    return {
        "question": "Why did signups fall?",
        "executive_summary": [{"text": "Signups fell 18%.", "evidence_ids": cited}],
        "findings": [
            {
                "title": "Signups fell in the week measured",
                "claims": [{"text": "Down 18% week over week.", "evidence_ids": cited}],
                "confidence": "medium",
            }
        ],
        "confidence": "medium",
    }


async def _tenant(session: AsyncSession, slug: str = "loop-test") -> TenantContext:
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


def _investigator(llm: RecordedLLM, tool: Tool, employee: Employee | None = None) -> Investigator:
    registry = ToolRegistry()
    registry.register(tool)
    return Investigator(
        llm=llm,
        registry=registry,
        executor=ToolExecutor(registry),
        employee=employee or _employee(),
    )


class TestHappyPath:
    async def test_gathers_evidence_then_drafts(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        tool = _ProbeTool()

        # Two gathering turns, then the analyst concludes.
        llm = RecordedLLM(
            completions=[_tool_turn(), _tool_turn(), _final_turn()],
            structured_outputs=[{}],  # replaced below once ids are known
        )
        investigator = _investigator(llm, tool)

        # The draft must cite the ids the loop actually minted, so the script is
        # completed after the fact by a wrapper around structured().
        original = llm.structured

        async def _structured(**kwargs):  # type: ignore[no-untyped-def]
            rows = (
                (
                    await session.execute(
                        select(Evidence.id).where(Evidence.investigation_id == investigation_id)
                    )
                )
                .scalars()
                .all()
            )
            llm._structured = [_draft(list(rows))]  # type: ignore[attr-defined]
            return await original(**kwargs)

        llm.structured = _structured  # type: ignore[method-assign]

        result = await investigator.investigate(
            session, ctx, investigation_id=investigation_id, question="Why did signups fall?"
        )

        assert result.stop_reason == CONCLUDED
        assert tool.invocations == 2
        assert len(result.evidence_ids) == 2
        assert result.report.executive_summary[0].evidence_ids == result.evidence_ids
        assert result.usage.total > 0
        assert result.duration_ms >= 0

    async def test_every_observation_is_persisted_as_evidence(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        llm = RecordedLLM(completions=[_tool_turn(2), _final_turn()])
        investigator = _investigator(llm, _ProbeTool())

        original = llm.structured

        async def _structured(**kwargs):  # type: ignore[no-untyped-def]
            rows = (
                (
                    await session.execute(
                        select(Evidence.id).where(Evidence.investigation_id == investigation_id)
                    )
                )
                .scalars()
                .all()
            )
            llm._structured = [_draft(list(rows))]  # type: ignore[attr-defined]
            return await original(**kwargs)

        llm.structured = _structured  # type: ignore[method-assign]
        await investigator.investigate(
            session, ctx, investigation_id=investigation_id, question="Why?"
        )

        stored = await session.scalar(
            select(func.count())
            .select_from(Evidence)
            .where(Evidence.investigation_id == investigation_id)
        )
        audited = await session.scalar(
            select(func.count()).select_from(ToolCall).where(ToolCall.tenant_id == ctx.tenant_id)
        )
        assert stored == 2
        assert audited == 2

    async def test_drafting_asks_for_its_own_longer_deadline(self, session: AsyncSession) -> None:
        """Drafting is one long call, not a loop turn.

        A live run spent 239 seconds gathering evidence and then threw the entire
        investigation away because the draft hit the 120-second per-turn ceiling. The
        override is only worth having if it is actually passed, so it is asserted here
        rather than inferred from the provider accepting the argument.
        """
        from cortex.agents.investigator import DRAFT_TIMEOUT_SECONDS

        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        llm = RecordedLLM(completions=[_tool_turn(), _final_turn()])
        investigator = _investigator(llm, _ProbeTool())

        original = llm.structured

        async def _structured(**kwargs):  # type: ignore[no-untyped-def]
            rows = (
                (
                    await session.execute(
                        select(Evidence.id).where(Evidence.investigation_id == investigation_id)
                    )
                )
                .scalars()
                .all()
            )
            llm._structured = [_draft(list(rows))]  # type: ignore[attr-defined]
            return await original(**kwargs)

        llm.structured = _structured  # type: ignore[method-assign]
        await investigator.investigate(
            session, ctx, investigation_id=investigation_id, question="Why?"
        )

        drafts = [call for call in llm.calls if call["kind"] == "structured"]
        assert len(drafts) == 1
        assert drafts[0]["timeout"] == DRAFT_TIMEOUT_SECONDS


class TestEvidenceIdsReachTheModel:
    """The property the whole grounding story depends on: if the model never sees
    an evidence id, no claim can cite one and the gate has nothing to check."""

    async def test_tool_results_carry_the_evidence_id(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        llm = RecordedLLM(completions=[_tool_turn(), _final_turn()])
        investigator = _investigator(llm, _ProbeTool())

        original = llm.structured

        async def _structured(**kwargs):  # type: ignore[no-untyped-def]
            rows = (
                (
                    await session.execute(
                        select(Evidence.id).where(Evidence.investigation_id == investigation_id)
                    )
                )
                .scalars()
                .all()
            )
            llm._structured = [_draft(list(rows))]  # type: ignore[attr-defined]
            return await original(**kwargs)

        llm.structured = _structured  # type: ignore[method-assign]
        result = await investigator.investigate(
            session, ctx, investigation_id=investigation_id, question="Why?"
        )

        # The second completion call saw the tool result from the first.
        second_call = llm.calls[1]
        tool_result_messages = [m for m in second_call["messages"] if m.tool_results]
        assert tool_result_messages, "the loop must feed observations back to the model"

        rendered = "\n".join(v for m in tool_result_messages for v in m.tool_results.values())
        evidence_id = result.evidence_ids[0]
        assert str(evidence_id) in rendered
        # Labelled, not buried: a bare uuid inside a JSON blob is cited far less
        # reliably than an explicit `evidence_id:` line.
        assert f"evidence_id: {evidence_id}" in rendered
        assert "probe.observe" in rendered
        assert "1200" in rendered

    async def test_the_draft_prompt_lists_the_permitted_ids(self, session: AsyncSession) -> None:
        """The model is told exactly which ids it may cite, so an invented citation
        is a deviation from an explicit list rather than an unconstrained guess."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        llm = RecordedLLM(completions=[_tool_turn(), _final_turn()])
        investigator = _investigator(llm, _ProbeTool())

        original = llm.structured
        captured: dict[str, str] = {}

        async def _structured(**kwargs):  # type: ignore[no-untyped-def]
            captured["instruction"] = kwargs["messages"][-1].content
            rows = (
                (
                    await session.execute(
                        select(Evidence.id).where(Evidence.investigation_id == investigation_id)
                    )
                )
                .scalars()
                .all()
            )
            llm._structured = [_draft(list(rows))]  # type: ignore[attr-defined]
            return await original(**kwargs)

        llm.structured = _structured  # type: ignore[method-assign]
        result = await investigator.investigate(
            session, ctx, investigation_id=investigation_id, question="Why?"
        )

        assert str(result.evidence_ids[0]) in captured["instruction"]
        assert "may cite only these evidence ids" in captured["instruction"]


class TestBudgetsBind:
    async def test_step_limit_stops_the_loop(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        # Three gathering turns available, but only two steps allowed.
        llm = RecordedLLM(completions=[_tool_turn(), _tool_turn(), _tool_turn()])
        investigator = _investigator(llm, _ProbeTool(), _employee(max_steps=2))

        original = llm.structured

        async def _structured(**kwargs):  # type: ignore[no-untyped-def]
            rows = (
                (
                    await session.execute(
                        select(Evidence.id).where(Evidence.investigation_id == investigation_id)
                    )
                )
                .scalars()
                .all()
            )
            llm._structured = [_draft(list(rows))]  # type: ignore[attr-defined]
            return await original(**kwargs)

        llm.structured = _structured  # type: ignore[method-assign]
        result = await investigator.investigate(
            session, ctx, investigation_id=investigation_id, question="Why?"
        )
        assert result.stop_reason == STEP_LIMIT
        assert len(result.steps) == 2

    async def test_tool_call_limit_stops_before_exceeding_it(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        tool = _ProbeTool()
        # A turn requesting three calls, with only two permitted: the turn is
        # refused wholesale rather than partially executed.
        llm = RecordedLLM(completions=[_tool_turn(2), _tool_turn(3)])
        investigator = _investigator(llm, tool, _employee(max_tool_calls=2))

        original = llm.structured

        async def _structured(**kwargs):  # type: ignore[no-untyped-def]
            rows = (
                (
                    await session.execute(
                        select(Evidence.id).where(Evidence.investigation_id == investigation_id)
                    )
                )
                .scalars()
                .all()
            )
            llm._structured = [_draft(list(rows))]  # type: ignore[attr-defined]
            return await original(**kwargs)

        llm.structured = _structured  # type: ignore[method-assign]
        result = await investigator.investigate(
            session, ctx, investigation_id=investigation_id, question="Why?"
        )
        assert result.stop_reason == TOOL_CALL_LIMIT
        assert tool.invocations == 2

    async def test_stalling_is_detected(self, session: AsyncSession) -> None:
        """A loop calling tools that yield nothing has stopped investigating, and
        should stop rather than burn the whole budget."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        good = _ProbeTool()
        registry = ToolRegistry()
        registry.register(good)

        # First turn succeeds so evidence exists; then the tool starts failing.
        llm = RecordedLLM(completions=[_tool_turn(), _tool_turn(), _tool_turn(), _tool_turn()])
        investigator = Investigator(
            llm=llm,
            registry=registry,
            executor=ToolExecutor(registry),
            employee=_employee(max_steps_without_new_evidence=2),
        )

        original = llm.structured
        call_count = {"n": 0}

        async def _structured(**kwargs):  # type: ignore[no-untyped-def]
            rows = (
                (
                    await session.execute(
                        select(Evidence.id).where(Evidence.investigation_id == investigation_id)
                    )
                )
                .scalars()
                .all()
            )
            llm._structured = [_draft(list(rows))]  # type: ignore[attr-defined]
            call_count["n"] += 1
            return await original(**kwargs)

        llm.structured = _structured  # type: ignore[method-assign]

        # Flip the tool to failing after the first observation.
        real_observe = good.observe

        async def _observe(ctx_: ToolContext) -> ToolResult:
            if good.invocations >= 1:
                from cortex.tools.base import UpstreamError

                good.invocations += 1
                raise UpstreamError("probe: upstream is unavailable")
            return await real_observe(ctx_)

        good._capabilities["observe"] = good.capability("observe").__class__(  # type: ignore[attr-defined]
            name="observe",
            description="Return a structured observation for testing the loop.",
            params_schema=SCHEMA,
            handler=_observe,
            result_key="rows",
        )

        result = await investigator.investigate(
            session, ctx, investigation_id=investigation_id, question="Why?"
        )
        assert result.stop_reason == STALLED
        assert len(result.evidence_ids) == 1

    def test_budget_fields_are_all_required(self) -> None:
        """A missing budget would mean an unbounded loop discovered on an invoice."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Budget(max_steps=10, max_tool_calls=10, max_tokens=1000)  # type: ignore[call-arg]


class TestToolFailuresAreFedBack:
    async def test_failure_becomes_an_observation_not_a_crash(self, session: AsyncSession) -> None:
        """Reporting "HubSpot is not connected" beats crashing the investigation."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        llm = RecordedLLM(completions=[_tool_turn(), _final_turn()])
        registry = ToolRegistry()
        registry.register(_ProbeTool(fail=True))
        investigator = Investigator(
            llm=llm,
            registry=registry,
            executor=ToolExecutor(registry),
            employee=_employee(),
        )

        # No evidence was gathered, so the loop must refuse to draft.
        with pytest.raises(InvestigationFailed, match="gathered no evidence"):
            await investigator.investigate(
                session, ctx, investigation_id=investigation_id, question="Why?"
            )

        # The failed call is still audited.
        audited = await session.scalar(
            select(func.count()).select_from(ToolCall).where(ToolCall.succeeded.is_(False))
        )
        assert audited == 1

    async def test_error_text_is_returned_to_the_model(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        good = _ProbeTool()
        bad = _ProbeTool(fail=True)
        bad.name = "broken"  # type: ignore[misc]

        registry = ToolRegistry()
        registry.register(good)
        registry.register(bad)

        llm = RecordedLLM(
            completions=[
                LLMResponse(
                    text="Trying both.",
                    tool_requests=[
                        ToolRequest(id="a", name="probe__observe", arguments={}),
                        ToolRequest(id="b", name="broken__observe", arguments={}),
                    ],
                    usage=Usage(input_tokens=100, output_tokens=20),
                ),
                _final_turn(),
            ]
        )
        investigator = Investigator(
            llm=llm,
            registry=registry,
            executor=ToolExecutor(registry),
            employee=_employee(),
        )

        original = llm.structured

        async def _structured(**kwargs):  # type: ignore[no-untyped-def]
            rows = (
                (
                    await session.execute(
                        select(Evidence.id).where(Evidence.investigation_id == investigation_id)
                    )
                )
                .scalars()
                .all()
            )
            llm._structured = [_draft(list(rows))]  # type: ignore[attr-defined]
            return await original(**kwargs)

        llm.structured = _structured  # type: ignore[method-assign]
        result = await investigator.investigate(
            session, ctx, investigation_id=investigation_id, question="Why?"
        )

        # The good observation survived; the failure was reported, not fatal.
        assert len(result.evidence_ids) == 1
        assert result.steps[0].errors
        second_call = llm.calls[1]
        results = {
            k: v
            for m in second_call["messages"]
            if m.tool_results
            for k, v in m.tool_results.items()
        }
        assert json.loads(results["b"])["error"].startswith("UpstreamError")


class TestRefusesToDraftWithoutEvidence:
    async def test_no_tool_calls_at_all_is_refused(self, session: AsyncSession) -> None:
        """A fluent answer with nothing behind it is the failure Cortex exists to
        prevent, so an evidence-free investigation raises rather than drafts."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        llm = RecordedLLM(completions=[_final_turn("Signups fell because of the deploy.")])
        investigator = _investigator(llm, _ProbeTool())

        with pytest.raises(InvestigationFailed, match="nothing to ground"):
            await investigator.investigate(
                session, ctx, investigation_id=investigation_id, question="Why?"
            )

    async def test_a_malformed_draft_is_refused(self, session: AsyncSession) -> None:
        """A provider returning a near-miss must not reach the gate, which assumes
        a well-formed report."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        # An executive summary claim with no evidence_ids: unrepresentable. Scripted
        # twice, because drafting now gets one repair attempt with the validation error
        # handed back — a whole investigation's evidence was once discarded over a single
        # fixable defect. A model that fails twice is not converging.
        malformed = {"question": "Why?", "executive_summary": [{"text": "Signups fell."}]}
        llm = RecordedLLM(
            completions=[_tool_turn(), _final_turn()],
            structured_outputs=[malformed, malformed],
        )
        investigator = _investigator(llm, _ProbeTool())

        with pytest.raises(InvestigationFailed, match="report schema after 2 attempt"):
            await investigator.investigate(
                session, ctx, investigation_id=investigation_id, question="Why?"
            )

    async def test_a_repairable_draft_is_repaired(self, session: AsyncSession) -> None:
        """The defect that cost a real investigation everything it had gathered.

        A `contradicted` hypothesis with no contradicting evidence is rejected by a
        Pydantic validator, which cannot be expressed in the JSON Schema the model is
        given — so the model has no way to know the rule from the schema alone. Handing
        the error back is far cheaper than re-running the loop.
        """
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        llm = RecordedLLM(completions=[_tool_turn(), _final_turn()])
        investigator = _investigator(llm, _ProbeTool())

        original = llm.structured
        attempts: list[int] = []

        async def _structured(**kwargs):  # type: ignore[no-untyped-def]
            rows = (
                (
                    await session.execute(
                        select(Evidence.id).where(Evidence.investigation_id == investigation_id)
                    )
                )
                .scalars()
                .all()
            )
            attempts.append(1)
            good = _draft(list(rows))
            if len(attempts) == 1:
                # First draft breaks the invariant the schema cannot state.
                good = {
                    **good,
                    "hypotheses": [
                        {
                            "statement": "The deploy caused it",
                            "verdict": "contradicted",
                            "reasoning": "ruled out by reasoning, with no cited observation",
                        }
                    ],
                }
            llm._structured = [good]  # type: ignore[attr-defined]
            return await original(**kwargs)

        llm.structured = _structured  # type: ignore[method-assign]
        result = await investigator.investigate(
            session, ctx, investigation_id=investigation_id, question="Why?"
        )

        assert len(attempts) == 2, "the first draft should have been repaired, not discarded"
        assert result.report.executive_summary, "the repaired report must keep its findings"


class TestTruncationIsDisclosed:
    async def test_early_stop_is_stated_in_the_draft_instruction(
        self, session: AsyncSession
    ) -> None:
        """A truncated investigation presented as complete is misleading even when
        every individual claim is grounded."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        llm = RecordedLLM(completions=[_tool_turn(), _tool_turn()])
        investigator = _investigator(llm, _ProbeTool(), _employee(max_steps=1))

        original = llm.structured
        captured: dict[str, str] = {}

        async def _structured(**kwargs):  # type: ignore[no-untyped-def]
            captured["instruction"] = kwargs["messages"][-1].content
            rows = (
                (
                    await session.execute(
                        select(Evidence.id).where(Evidence.investigation_id == investigation_id)
                    )
                )
                .scalars()
                .all()
            )
            llm._structured = [_draft(list(rows))]  # type: ignore[attr-defined]
            return await original(**kwargs)

        llm.structured = _structured  # type: ignore[method-assign]
        result = await investigator.investigate(
            session, ctx, investigation_id=investigation_id, question="Why?"
        )

        assert result.stop_reason == STEP_LIMIT
        assert "stopped early" in captured["instruction"]
        assert "step_limit" in captured["instruction"]


class TestEmployeeContract:
    def test_the_shipped_employee_is_valid(self) -> None:
        from cortex.tools.registry import gtm_analyst_registry

        employee = gtm_data_analyst()
        employee.validate_against(set(gtm_analyst_registry().tool_names))
        assert employee.role == "gtm_data_analyst"
        assert employee.kpis
        assert employee.escalation

    def test_unknown_tool_is_caught_at_load(self) -> None:
        """A typo must surface at startup, not as a mid-investigation failure."""
        with pytest.raises(ValueError, match="not registered"):
            gtm_data_analyst().validate_against({"ga4"})

    def test_missing_employee_raises(self) -> None:
        from cortex.agents.employee import EmployeeNotFound, load_employee

        with pytest.raises(EmployeeNotFound):
            load_employee("does_not_exist")

    def test_hostile_role_name_is_refused(self) -> None:
        """The role becomes a filename, so it is validated rather than trusted."""
        from cortex.agents.employee import EmployeeNotFound, load_employee

        for hostile in ("../secrets", "a/b", "..", "a b"):
            with pytest.raises(EmployeeNotFound):
                load_employee(hostile)


class _FakeClock:
    """A controllable monotonic clock.

    Reaching the time limit by sleeping would make the test slow and flaky, which
    in practice means the limit goes untested — and the wall-clock budget is the one
    most likely to matter under a hung upstream.
    """

    def __init__(self, *, step: float = 0.0) -> None:
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        now = self.t
        self.t += self.step
        return now


class TestTimeAndTokenLimits:
    async def test_time_limit_stops_the_loop(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        # Each clock read advances 100s. The budget is 250s, of which DRAFT_RESERVE_SECONDS=90
        # is reserved for drafting, so the loop's own share is 160s: the first step runs at
        # elapsed=100, the second finds elapsed=200 over the share and stops. Sized against the
        # reserve deliberately -- the previous 120s budget now leaves a 30s loop share, so the
        # loop would stop before its first step and the test would pass for the wrong reason.
        clock = _FakeClock(step=100.0)
        llm = RecordedLLM(completions=[_tool_turn(), _tool_turn(), _tool_turn()])
        registry = ToolRegistry()
        registry.register(_ProbeTool())
        investigator = Investigator(
            llm=llm,
            registry=registry,
            executor=ToolExecutor(registry),
            employee=_employee(max_seconds=250),
            clock=clock,
        )

        original = llm.structured

        async def _structured(**kwargs):  # type: ignore[no-untyped-def]
            rows = (
                (
                    await session.execute(
                        select(Evidence.id).where(Evidence.investigation_id == investigation_id)
                    )
                )
                .scalars()
                .all()
            )
            llm._structured = [_draft(list(rows))]  # type: ignore[attr-defined]
            return await original(**kwargs)

        llm.structured = _structured  # type: ignore[method-assign]
        result = await investigator.investigate(
            session, ctx, investigation_id=investigation_id, question="Why?"
        )

        assert result.stop_reason == TIME_LIMIT
        assert result.evidence_ids, "evidence gathered before the limit is kept"

    async def test_token_limit_stops_the_loop(self, session: AsyncSession) -> None:
        """Deterministic: the recorded provider reports usage, so no real spend is
        needed to prove the budget binds."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        expensive = LLMResponse(
            text="Checking.",
            tool_requests=[ToolRequest(id="c1", name="probe__observe", arguments={})],
            usage=Usage(input_tokens=9000, output_tokens=3000),
        )
        llm = RecordedLLM(completions=[expensive, expensive, expensive])
        investigator = _investigator(llm, _ProbeTool(), _employee(max_tokens=10000))

        original = llm.structured

        async def _structured(**kwargs):  # type: ignore[no-untyped-def]
            rows = (
                (
                    await session.execute(
                        select(Evidence.id).where(Evidence.investigation_id == investigation_id)
                    )
                )
                .scalars()
                .all()
            )
            llm._structured = [_draft(list(rows))]  # type: ignore[attr-defined]
            return await original(**kwargs)

        llm.structured = _structured  # type: ignore[method-assign]
        result = await investigator.investigate(
            session, ctx, investigation_id=investigation_id, question="Why?"
        )

        assert result.stop_reason == TOKEN_LIMIT
        assert result.usage.total > 10000

    async def test_duration_is_measured_from_the_injected_clock(
        self, session: AsyncSession
    ) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        clock = _FakeClock(step=2.0)
        llm = RecordedLLM(completions=[_tool_turn(), _final_turn()])
        registry = ToolRegistry()
        registry.register(_ProbeTool())
        investigator = Investigator(
            llm=llm,
            registry=registry,
            executor=ToolExecutor(registry),
            employee=_employee(),
            clock=clock,
        )

        original = llm.structured

        async def _structured(**kwargs):  # type: ignore[no-untyped-def]
            rows = (
                (
                    await session.execute(
                        select(Evidence.id).where(Evidence.investigation_id == investigation_id)
                    )
                )
                .scalars()
                .all()
            )
            llm._structured = [_draft(list(rows))]  # type: ignore[attr-defined]
            return await original(**kwargs)

        llm.structured = _structured  # type: ignore[method-assign]
        result = await investigator.investigate(
            session, ctx, investigation_id=investigation_id, question="Why?"
        )
        assert result.duration_ms > 0


class TestProviderFailureMidLoop:
    async def test_failure_after_evidence_stops_and_keeps_the_evidence(
        self, session: AsyncSession
    ) -> None:
        """Evidence already gathered is worth reporting; discarding it because the
        next turn failed would throw away real work."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        # One gathering turn, then the script is exhausted so complete() raises.
        llm = RecordedLLM(completions=[_tool_turn()])
        investigator = _investigator(llm, _ProbeTool())

        original = llm.structured

        async def _structured(**kwargs):  # type: ignore[no-untyped-def]
            rows = (
                (
                    await session.execute(
                        select(Evidence.id).where(Evidence.investigation_id == investigation_id)
                    )
                )
                .scalars()
                .all()
            )
            llm._structured = [_draft(list(rows))]  # type: ignore[attr-defined]
            return await original(**kwargs)

        llm.structured = _structured  # type: ignore[method-assign]
        result = await investigator.investigate(
            session, ctx, investigation_id=investigation_id, question="Why?"
        )

        assert result.stop_reason == STALLED
        assert len(result.evidence_ids) == 1
        assert result.steps[-1].errors, "the outage is recorded in the trail"

    async def test_failure_before_any_evidence_raises(self, session: AsyncSession) -> None:
        """No evidence and no model: there is no partial answer worth grounding."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        investigator = _investigator(RecordedLLM(), _ProbeTool())

        with pytest.raises(InvestigationFailed, match="could not run"):
            await investigator.investigate(
                session, ctx, investigation_id=investigation_id, question="Why?"
            )

    async def test_drafting_failure_raises(self, session: AsyncSession) -> None:
        """Evidence was gathered but the draft call failed — reported as a failure
        rather than an empty report."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        # Completions for the loop, but no structured output for the draft.
        llm = RecordedLLM(completions=[_tool_turn(), _final_turn()])
        investigator = _investigator(llm, _ProbeTool())

        with pytest.raises(InvestigationFailed, match="could not be drafted"):
            await investigator.investigate(
                session, ctx, investigation_id=investigation_id, question="Why?"
            )


class TestInvestigationResult:
    async def test_gathered_evidence_reports_truthfully(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        llm = RecordedLLM(completions=[_tool_turn(), _final_turn()])
        investigator = _investigator(llm, _ProbeTool())

        original = llm.structured

        async def _structured(**kwargs):  # type: ignore[no-untyped-def]
            rows = (
                (
                    await session.execute(
                        select(Evidence.id).where(Evidence.investigation_id == investigation_id)
                    )
                )
                .scalars()
                .all()
            )
            llm._structured = [_draft(list(rows))]  # type: ignore[attr-defined]
            return await original(**kwargs)

        llm.structured = _structured  # type: ignore[method-assign]
        result = await investigator.investigate(
            session, ctx, investigation_id=investigation_id, question="Why?"
        )
        assert result.gathered_evidence is True


class TestPriorContextFromMemory:
    """Recall shortens the path to a hypothesis. It can never be the grounds for one.

    Memory has no evidence row behind it, so a claim citing it would be stripped by the
    gate and the reader would see a thinner report for no visible reason. These tests pin
    the two properties that keep that from happening: the block says what it is, and the
    loop survives memory being unavailable.
    """

    def test_recalled_context_reaches_the_opening_prompt(self) -> None:
        opening = _opening(
            "Why did signups fall?",
            memory="MEMORY (prior context, NOT evidence). ... mobile signups look off",
        )
        assert "NOT evidence" in opening
        # Before the question, so the instruction that only tool results carry evidence
        # ids is read before the loop starts rather than after it has drafted.
        assert opening.index("NOT evidence") < opening.index("Question:")

    def test_no_memory_leaves_the_prompt_unchanged(self) -> None:
        assert "MEMORY" not in _opening("Why did signups fall?")

    async def test_an_unavailable_memory_does_not_fail_the_investigation(
        self, session: AsyncSession
    ) -> None:
        """An investigation that cannot run because a vector store is unreachable is a
        worse product than one that runs without its memory."""

        class _Broken:
            async def recall(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                raise RuntimeError("qdrant unreachable")

        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        tool = _ProbeTool()
        llm = RecordedLLM(completions=[_tool_turn(), _final_turn()])
        registry = ToolRegistry()
        registry.register(tool)
        investigator = Investigator(
            llm=llm,
            registry=registry,
            executor=ToolExecutor(registry),
            employee=_employee(),
            recall=_Broken(),  # type: ignore[arg-type]
        )

        original = llm.structured

        async def _structured(**kwargs):  # type: ignore[no-untyped-def]
            rows = (
                (
                    await session.execute(
                        select(Evidence.id).where(Evidence.investigation_id == investigation_id)
                    )
                )
                .scalars()
                .all()
            )
            llm._structured = [_draft(list(rows))]  # type: ignore[attr-defined]
            return await original(**kwargs)

        llm.structured = _structured  # type: ignore[method-assign]

        result = await investigator.investigate(
            session, ctx, investigation_id=investigation_id, question="Why did signups fall?"
        )
        assert result.report is not None
        assert tool.invocations == 1

    async def test_recalled_context_is_offered_when_memory_has_something(
        self, session: AsyncSession
    ) -> None:
        """The point of wiring it in: what memory knows reaches the first turn."""

        class _Recall:
            def __init__(self) -> None:
                self.asked: list[str] = []

            async def recall(self, tenant, question, **kwargs):  # type: ignore[no-untyped-def]
                self.asked.append(question)
                return _Recalled()

        class _Recalled:
            is_empty = False

            def render(self) -> str:
                return "MEMORY (prior context, NOT evidence). PR 913 reworked onboarding."

        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        tool = _ProbeTool()
        llm = RecordedLLM(completions=[_tool_turn(), _final_turn()])
        registry = ToolRegistry()
        registry.register(tool)
        recall = _Recall()
        investigator = Investigator(
            llm=llm,
            registry=registry,
            executor=ToolExecutor(registry),
            employee=_employee(),
            recall=recall,  # type: ignore[arg-type]
        )

        original = llm.structured

        async def _structured(**kwargs):  # type: ignore[no-untyped-def]
            rows = (
                (
                    await session.execute(
                        select(Evidence.id).where(Evidence.investigation_id == investigation_id)
                    )
                )
                .scalars()
                .all()
            )
            llm._structured = [_draft(list(rows))]  # type: ignore[attr-defined]
            return await original(**kwargs)

        llm.structured = _structured  # type: ignore[method-assign]

        await investigator.investigate(
            session, ctx, investigation_id=investigation_id, question="Why did signups fall?"
        )

        assert recall.asked == ["Why did signups fall?"]
        first_turn = llm.calls[0]["messages"][0].content  # type: ignore[index]
        assert "PR 913 reworked onboarding" in first_turn


class TestCancellation:
    """Cooperative, and it has to cross a process boundary.

    The usual shape is a threading flag on the conversation, which works when the loop and the
    cancel request share a process. These do not: the loop runs in a Celery worker and the
    request arrives at the gateway. So the check reads shared state — and it is
    asked *before* the model call, because checking after it would take the cancellation and
    still pay for the turn.
    """

    async def test_a_cancelled_investigation_stops_without_drafting(
        self, session: AsyncSession
    ) -> None:
        """Drafting is the most expensive call in the investigation. Running it after the user
        asked to stop would bill them for the thing they cancelled."""
        from cortex.agents.investigator import InvestigationCancelled

        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        tool = _ProbeTool()
        llm = RecordedLLM(completions=[_tool_turn(), _final_turn()])
        registry = ToolRegistry()
        registry.register(tool)

        async def _cancelled() -> bool:
            return True

        investigator = Investigator(
            llm=llm,
            registry=registry,
            executor=ToolExecutor(registry),
            employee=_employee(),
            cancelled=_cancelled,
        )

        with pytest.raises(InvestigationCancelled) as raised:
            await investigator.investigate(
                session, ctx, investigation_id=investigation_id, question="Why did signups fall?"
            )

        assert raised.value.steps == 0
        # Nothing was asked of the model, and no tool ran: the check precedes both.
        assert tool.invocations == 0
        assert llm.calls == []

    async def test_evidence_gathered_before_the_stop_is_kept(self, session: AsyncSession) -> None:
        """It was really observed, and "what did it find before I cancelled" is a fair
        question."""
        from cortex.agents.investigator import InvestigationCancelled

        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        tool = _ProbeTool()
        llm = RecordedLLM(completions=[_tool_turn(), _tool_turn(), _final_turn()])
        registry = ToolRegistry()
        registry.register(tool)

        calls = {"n": 0}

        async def _cancelled() -> bool:
            # Allowed through the first step, cancelled before the second.
            calls["n"] += 1
            return calls["n"] > 1

        investigator = Investigator(
            llm=llm,
            registry=registry,
            executor=ToolExecutor(registry),
            employee=_employee(),
            cancelled=_cancelled,
        )

        with pytest.raises(InvestigationCancelled) as raised:
            await investigator.investigate(
                session, ctx, investigation_id=investigation_id, question="Why did signups fall?"
            )

        assert raised.value.steps == 1
        assert raised.value.evidence == 1
        rows = await session.scalar(
            select(func.count()).select_from(Evidence).where(Evidence.tenant_id == ctx.tenant_id)
        )
        assert rows == 1, "the observation survives the cancellation"

    async def test_a_failing_check_does_not_cancel_anything(self, session: AsyncSession) -> None:
        """A database blip must not cancel an investigation nobody asked to cancel. The next
        step asks again."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        tool = _ProbeTool()
        llm = RecordedLLM(completions=[_tool_turn(), _final_turn()])
        registry = ToolRegistry()
        registry.register(tool)

        async def _broken() -> bool:
            raise RuntimeError("database unreachable")

        investigator = Investigator(
            llm=llm,
            registry=registry,
            executor=ToolExecutor(registry),
            employee=_employee(),
            cancelled=_broken,
        )

        original = llm.structured

        async def _structured(**kwargs):  # type: ignore[no-untyped-def]
            rows = (
                (
                    await session.execute(
                        select(Evidence.id).where(Evidence.investigation_id == investigation_id)
                    )
                )
                .scalars()
                .all()
            )
            llm._structured = [_draft(list(rows))]  # type: ignore[attr-defined]
            return await original(**kwargs)

        llm.structured = _structured  # type: ignore[method-assign]

        result = await investigator.investigate(
            session, ctx, investigation_id=investigation_id, question="Why did signups fall?"
        )
        assert result.report is not None

    async def test_no_checker_means_no_cancellation(self, session: AsyncSession) -> None:
        """The default. A caller that does not want cancellation is unaffected, and pays for
        no extra query per step."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        registry = ToolRegistry()
        registry.register(_ProbeTool())
        investigator = Investigator(
            llm=RecordedLLM(completions=[_tool_turn(), _final_turn()]),
            registry=registry,
            executor=ToolExecutor(registry),
            employee=_employee(),
        )
        assert investigator._cancelled is None
        del investigator, investigation_id


class TestADegenerateDraftIsRetriedNotDelivered:
    """A draft that satisfies the schema and is not a report.

    Measured: two attempts in fifteen returned a report whose only summary claim was the literal
    text "placeholder", with no findings, no hypotheses and no risks -- 95 and 112 output tokens
    against 1,400 to 4,700 on healthy drafts. The loop had run normally and gathered evidence in
    both cases; the drafting call collapsed.

    Nothing noticed. A placeholder claim was *valid* -- `Claim.text` required one character -- so
    the repair retry never fired, and the collapse surfaced three phases later as a citation gate
    rejection that discarded the whole investigation. The most expensive possible place to find it.

    That one-character floor is gone: `Claim.text` now refuses a single word, so the literal
    "placeholder" cannot be constructed at all. This check still earns its place, because a
    two-word stub validates fine and is still a stub -- which is why the case below uses one.
    """

    def test_a_stub_with_evidence_is_degenerate(self) -> None:
        from cortex.agents.investigator import _is_degenerate

        evidence = [uuid.uuid4()]
        stub = InvestigationReport(
            question="Why did signups fall?",
            # Two words, so it clears the claim-level floor and reaches this check. The
            # one-word version of exactly this stub is what the loop actually produced.
            executive_summary=[Claim(text="placeholder text", evidence_ids=evidence)],
            confidence=Confidence.MEDIUM,
        )
        assert _is_degenerate(stub, evidence)

    def test_the_one_word_version_cannot_be_built_any_more(self) -> None:
        """The stub the loop really emitted, now stopped a layer earlier."""
        with pytest.raises(ValidationError, match="one word cannot"):
            Claim(text="placeholder", evidence_ids=[uuid.uuid4()])

    def test_the_test_is_structural_not_a_word_search(self) -> None:
        """Matching "placeholder" would be the fifth keyword list standing in for meaning in
        this codebase. A stub that says something plausible is still a stub."""
        from cortex.agents.investigator import _is_degenerate

        evidence = [uuid.uuid4()]
        plausible_stub = InvestigationReport(
            question="Why did signups fall?",
            executive_summary=[
                Claim(text="Signups declined over the period.", evidence_ids=evidence)
            ],
            confidence=Confidence.MEDIUM,
        )
        assert _is_degenerate(plausible_stub, evidence)

    def test_a_false_premise_report_with_no_hypotheses_is_fine(self) -> None:
        """`GUIDANCE[Shape.FACTUAL]` asks for empty hypotheses when nothing needs explaining, and
        across thirteen healthy drafts the false-premise reports had exactly that -- with
        findings. Requiring hypotheses alone would fail the reports doing it right."""
        from cortex.agents.investigator import _is_degenerate

        evidence = [uuid.uuid4()]
        factual = InvestigationReport(
            question="Did signups fall last month?",
            executive_summary=[Claim(text="No, that is a partial month.", evidence_ids=evidence)],
            findings=[
                Finding(
                    title="August covers 12 days",
                    claims=[Claim(text="Through the 12th only.", evidence_ids=evidence)],
                    confidence=Confidence.HIGH,
                )
            ],
            confidence=Confidence.HIGH,
        )
        assert not _is_degenerate(factual, evidence)

    def test_a_report_with_hypotheses_and_no_findings_is_fine(self) -> None:
        """Either section is enough. Demanding both would fail a report that tested candidates
        and had nothing separate to add as a finding."""
        from cortex.agents.investigator import _is_degenerate

        evidence = [uuid.uuid4()]
        report = InvestigationReport(
            question="Why did signups fall?",
            executive_summary=[Claim(text="Unclear claim.", evidence_ids=evidence)],
            hypotheses=[
                Hypothesis(
                    statement="A deploy did it.",
                    verdict=Verdict.CONTRADICTED,
                    contradicting_evidence_ids=evidence,
                )
            ],
            confidence=Confidence.LOW,
        )
        assert not _is_degenerate(report, evidence)

    def test_gathering_nothing_is_not_a_degenerate_draft(self) -> None:
        """An investigation with no evidence has nothing to write a finding from, and failing it
        here would report the wrong problem -- the absence of evidence, which the loop already
        reports."""
        from cortex.agents.investigator import _is_degenerate

        stub = InvestigationReport(
            question="Why did signups fall?",
            executive_summary=[Claim(text="placeholder text", evidence_ids=[uuid.uuid4()])],
            confidence=Confidence.MEDIUM,
        )
        assert not _is_degenerate(stub, [])


class TestATruncatedDraftDoesNotDiscardTheInvestigation:
    """Twice now, a live run has died with "output truncated at max_tokens".

    The repeat is what makes it structural rather than unlucky. `max_tokens` bounds thinking
    *plus* output, so the drafting call carries two demands on one budget: it thinks the
    longest of any call in the run, and its output is all-or-nothing because partial JSON
    parses to nothing. It happens at the point where every tool call has already been made
    and paid for, so the cost of giving up is the entire investigation.

    Raising the ceiling was the response the first time and it recurred, so the ceiling is
    not the fix -- being able to try again with less to write is.
    """

    async def _run(
        self,
        session: AsyncSession,
        *,
        failures: list[Exception | dict],
    ):
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        llm = RecordedLLM(
            completions=[_tool_turn(), _final_turn()],
            structured_outputs=[{}],
        )
        investigator = _investigator(llm, _ProbeTool())
        original = llm.structured
        pending = list(failures)

        async def _structured(**kwargs):  # type: ignore[no-untyped-def]
            if pending:
                nxt = pending.pop(0)
                # A dict is returned rather than raised, because that is how an invalid draft
                # really arrives: the provider call succeeds and validation rejects the
                # payload. Raising here would test a path the provider cannot produce.
                if isinstance(nxt, dict):
                    return nxt, Usage()
                raise nxt
            rows = (
                (
                    await session.execute(
                        select(Evidence.id).where(Evidence.investigation_id == investigation_id)
                    )
                )
                .scalars()
                .all()
            )
            llm._structured = [_draft(list(rows))]  # type: ignore[attr-defined]
            return await original(**kwargs)

        llm.structured = _structured  # type: ignore[method-assign]
        return await investigator.investigate(
            session, ctx, investigation_id=investigation_id, question="Why did signups fall?"
        )

    async def test_one_truncation_is_survived(self, session: AsyncSession) -> None:
        """The evidence is already paid for; a second attempt is cheaper than losing it."""
        result = await self._run(
            session, failures=[LLMOutputTruncated("recorded: output truncated")]
        )
        assert result.report.executive_summary[0].text == "Signups fell 18%."

    async def test_the_reader_is_told_the_report_was_shortened(self, session: AsyncSession) -> None:
        """The one event that changes what the report says with nothing in the report saying so.

        It goes in `steps` rather than a private counter because `steps` is what the timeline
        renders: a silent recovery is how a degraded answer gets read as a complete one.
        """
        result = await self._run(
            session, failures=[LLMOutputTruncated("recorded: output truncated")]
        )
        shortened = [s for s in result.steps if "shorter" in s.note]
        assert len(shortened) == 1
        assert "truncated" in shortened[0].errors[0]

    async def test_a_second_truncation_gives_up(self, session: AsyncSession) -> None:
        """At 64k, a report that cannot fit twice is not long -- it is looping."""
        with pytest.raises(InvestigationFailed, match="truncated"):
            await self._run(
                session,
                failures=[
                    LLMOutputTruncated("recorded: output truncated"),
                    LLMOutputTruncated("recorded: output truncated again"),
                ],
            )

    async def test_a_refusal_is_not_retried(self, session: AsyncSession) -> None:
        """The opposite response to truncation: a model that will not answer will not answer.

        Both used to raise `LLMRefusedStructure`, so no caller could tell them apart.
        """
        with pytest.raises(InvestigationFailed, match="declined"):
            await self._run(
                session, failures=[LLMRefusedStructure("recorded: declined to produce")]
            )

    async def test_truncation_and_an_invalid_repair_both_get_their_attempt(
        self, session: AsyncSession
    ) -> None:
        """Why the two budgets are separate rather than one shared count.

        A run that comes back short *and then* invalid is the likeliest place to need both,
        since either symptom means the model was struggling. Sharing one budget would leave
        exactly that case with nothing left to fix it.
        """
        result = await self._run(
            session,
            failures=[
                LLMOutputTruncated("recorded: output truncated"),
                # Schema-valid and not a report: a summary claim with no findings and no
                # hypotheses, from a run that gathered evidence. `_is_degenerate` rejects it
                # into the repair loop -- the same path a real invalid draft takes.
                {
                    "question": "Why did signups fall?",
                    "executive_summary": [
                        {"text": "Signups fell.", "evidence_ids": [str(uuid.uuid4())]}
                    ],
                    "confidence": "medium",
                },
            ],
        )
        assert result.report.findings

    def test_the_draft_gets_the_whole_model_ceiling(self) -> None:
        """Nothing is reserved because nothing follows: drafting is the last provider call."""
        from cortex.agents.anthropic_llm import DEFAULT_MODEL
        from cortex.agents.investigator import DRAFT_MAX_TOKENS
        from cortex.agents.models import card_for

        assert DRAFT_MAX_TOKENS == card_for(DEFAULT_MODEL).max_output_tokens
