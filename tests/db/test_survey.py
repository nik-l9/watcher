"""Establishing what exists before investigating anything.

Asked which pull request shipped most recently to the company website, the analyst searched
`acme/acme` — the only repository name it had ever seen — and honestly reported that it
could not confirm an answer. `acme/company-website` exists, and the ingest already knew about
it. Nothing in the tool surface could have told the analyst, because all eight GitHub capabilities
take a repository name as a parameter.

Adding a discovery capability alone would have been half a fix, and the reason is measured
elsewhere: *Agents Explore but Agents Ignore* (arXiv 2604.17609) found a model observing
documentation that named the solution in **97.54%** of attempts and calling it in **0.53%**. Its
tool-availability finding cuts the other way too — richer tooling made agents investigate *less*.
So the survey is deterministic: every capability declaring `discovery=True` runs once, before the
first step.

What these tests pin down is the part that could quietly stop working:

  - the survey actually runs, and its results reach the prompt with citable ids
  - it is derived from declarations rather than a hardcoded list, so a new discovery capability
    is included without anyone remembering to add it here
  - it is bounded — one call per capability, and it cannot grow with the question
  - a broken connector does not stop the investigation, but is *reported* rather than swallowed
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from cortex.agents.employee import gtm_data_analyst
from cortex.agents.investigator import Investigator, _opening
from cortex.agents.llm import RecordedLLM
from cortex.agents.progress import Phase, ProgressEvent
from cortex.db.models import Investigation, Tenant
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.tenancy.context import TenantContext
from cortex.tools.base import (
    Capability,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    UpstreamError,
)
from cortex.tools.executor import ToolExecutor


class _Catalogue(Tool):
    """A connector with one discovery capability and one ordinary one."""

    name = "catalogue"
    provider = None

    def __init__(self, *, fail: bool = False) -> None:
        # Set before super().__init__, which calls capabilities().
        self._fail = fail
        super().__init__()

    def capabilities(self) -> list[Capability]:
        return [
            Capability(
                name="list_things",
                description="What exists.",
                params_schema={"type": "object", "additionalProperties": False, "properties": {}},
                handler=self.list_things,
                result_key="things",
                discovery=True,
            ),
            Capability(
                name="measure",
                description="Measure one thing.",
                params_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"thing": {"type": "string"}},
                },
                handler=self.measure,
                result_key="values",
            ),
        ]

    async def list_things(self, ctx: ToolContext) -> ToolResult:
        if self._fail:
            raise UpstreamError("catalogue: 503 from GET /things")
        return ToolResult(
            payload={"things": ["org/company-website", "org/product"]},
            source_ref="catalogue://things",
        )

    async def measure(self, ctx: ToolContext, *, thing: str = "") -> ToolResult:
        return ToolResult(payload={"values": [1, 2, 3]})


async def _tenant(session: AsyncSession, slug: str) -> tuple[TenantContext, uuid.UUID]:
    tenant_id = uuid.uuid4()
    graph_name = graph_name_for_new_tenant(slug, tenant_id)
    session.add(Tenant(id=tenant_id, slug=slug, name=slug, graph_name=graph_name))
    await session.flush()
    row = Investigation(tenant_id=tenant_id, question="Which PR shipped to the website?")
    session.add(row)
    await session.flush()
    return (
        TenantContext(tenant_id=tenant_id, tenant_slug=slug, graph_name=graph_name),
        row.id,
    )


def _investigator(
    registry: ToolRegistry, events: list[ProgressEvent] | None = None
) -> Investigator:
    return Investigator(
        llm=RecordedLLM([]),
        registry=registry,
        executor=ToolExecutor(registry),
        employee=gtm_data_analyst(),
        progress=events.append if events is not None else None,
    )


def _registry(*, fail: bool = False) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_Catalogue(fail=fail))
    return registry


class TestTheSurveyRuns:
    async def test_it_calls_every_discovery_capability(self, session: AsyncSession) -> None:
        tenant, investigation_id = await _tenant(session, "survey-runs")
        survey, evidence_ids, calls = await _investigator(_registry())._survey(
            session, tenant, investigation_id
        )

        assert calls == 1
        assert len(evidence_ids) == 1
        assert survey is not None
        assert "org/company-website" in survey

    async def test_it_does_not_call_ordinary_capabilities(self, session: AsyncSession) -> None:
        """The budget is fixed at one call per *discovery* capability. A survey that called
        everything would spend the investigation's tool budget before it began."""
        tenant, investigation_id = await _tenant(session, "survey-bounded")
        survey, _, calls = await _investigator(_registry())._survey(
            session, tenant, investigation_id
        )
        assert calls == 1
        assert survey is not None
        assert "values" not in survey

    async def test_the_results_are_citable(self, session: AsyncSession) -> None:
        """The survey goes through the executor, so its results are real evidence rows. "These
        repositories exist" is an observation, and a report naming one should be able to cite
        where that came from."""
        tenant, investigation_id = await _tenant(session, "survey-citable")
        survey, evidence_ids, _ = await _investigator(_registry())._survey(
            session, tenant, investigation_id
        )
        assert survey is not None
        assert str(evidence_ids[0]) in survey
        assert "evidence_id:" in survey

    async def test_it_is_derived_from_declarations(self) -> None:
        """Not a hardcoded list. A connector that adds a discovery capability is surveyed
        without anyone remembering to edit the investigator."""
        from cortex.tools.registry import gtm_analyst_registry

        declared = gtm_analyst_registry().discovery_capabilities()
        assert "github__list_repositories" in declared
        assert "posthog__list_projects" in declared
        # An ordinary measuring capability must not creep in: the survey runs before the
        # question is considered, so a measurement here would be a metric fetched blind.
        assert "posthog__event_trend" not in declared
        assert "github__commits" not in declared

    async def test_an_empty_registry_surveys_nothing(self, session: AsyncSession) -> None:
        tenant, investigation_id = await _tenant(session, "survey-none")
        survey, evidence_ids, calls = await _investigator(ToolRegistry())._survey(
            session, tenant, investigation_id
        )
        assert survey is None
        assert evidence_ids == []
        assert calls == 0

    async def test_progress_is_reported(self, session: AsyncSession) -> None:
        """Surveying happens before the first step, so without an event the run appears to
        hang for however long enumeration takes."""
        tenant, investigation_id = await _tenant(session, "survey-progress")
        events: list[ProgressEvent] = []
        await _investigator(_registry(), events)._survey(session, tenant, investigation_id)
        assert any(event.phase is Phase.SURVEYING for event in events)


class TestItDoesNotImplyAbsence:
    """The regression the first version of the survey caused.

    Its header named only the *resources* it had found. GA4 has no discovery capability, so it
    never appeared — and "anything not on that list, you have not observed" read as "GA4 is not
    part of this estate". Two of six eval attempts then stopped calling GA4 entirely:
    `tool_selection` 0.67 and 0.00, both missing only GA4 capabilities, on a suite where it had
    been 1.00 across the board.

    An enumeration of some sources is not a statement about the others, and the block has to say
    so, because a strongly worded prompt is read as exhaustive whether it meant to be or not.
    """

    async def test_every_available_connector_is_named(self, session: AsyncSession) -> None:
        """Including ones with nothing to enumerate. `_Catalogue` here stands for the enumerable
        connector and `_Measurer` for the GA4-shaped one that has no discovery capability."""

        class _Measurer(Tool):
            name = "measurer"
            provider = None

            def capabilities(self) -> list[Capability]:
                return [
                    Capability(
                        name="compare_periods",
                        description="Compare two periods.",
                        params_schema={
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {},
                        },
                        handler=self._compare,
                        result_key="rows",
                    )
                ]

            async def _compare(self, ctx: ToolContext) -> ToolResult:
                return ToolResult(payload={"rows": [1]})

        registry = ToolRegistry()
        registry.register(_Catalogue())
        registry.register(_Measurer())

        tenant, investigation_id = await _tenant(session, "survey-not-absent")
        survey, _, calls = await _investigator(registry)._survey(session, tenant, investigation_id)

        assert survey is not None
        # Only the enumerable connector is surveyed...
        assert calls == 1
        # ...but the one with nothing to list is still named as available.
        assert "measurer" in survey
        assert "catalogue" in survey

    async def test_it_says_an_unenumerated_source_is_not_absent(
        self, session: AsyncSession
    ) -> None:
        """The sentence that does the work. Without it the block is read as exhaustive."""
        tenant, investigation_id = await _tenant(session, "survey-says-so")
        survey, _, _ = await _investigator(_registry())._survey(session, tenant, investigation_id)
        assert survey is not None
        assert "says nothing about it" in survey
        assert "query it normally" in survey


class TestABrokenConnector:
    async def test_it_does_not_stop_the_investigation(self, session: AsyncSession) -> None:
        """A connector that will not answer its own enumeration is one the analyst will
        discover is broken when it calls it for real. Refusing to investigate over it would be
        worse than proceeding."""
        tenant, investigation_id = await _tenant(session, "survey-broken")
        survey, evidence_ids, calls = await _investigator(_registry(fail=True))._survey(
            session, tenant, investigation_id
        )
        assert calls == 1
        assert evidence_ids == []
        assert survey is not None

    async def test_the_failure_is_reported_rather_than_swallowed(
        self, session: AsyncSession
    ) -> None:
        """ "I could not list your repositories" is information. Swallowed, the analyst would
        attribute the absence to the data — which is the empty-result bug in a new place."""
        tenant, investigation_id = await _tenant(session, "survey-reported")
        survey, _, _ = await _investigator(_registry(fail=True))._survey(
            session, tenant, investigation_id
        )
        assert survey is not None
        assert "unavailable" in survey
        assert "catalogue__list_things" in survey


class TestTheOpeningPrompt:
    def test_the_survey_precedes_the_question(self) -> None:
        """It constrains what everything else can mean: a repository named nowhere in it does
        not exist, and "the website" has to be resolved against the list first."""
        opening = _opening("Which PR shipped to the website?", survey="WHAT EXISTS\norg/site")
        assert opening.index("WHAT EXISTS") < opening.index("Question:")

    def test_it_tells_the_analyst_not_to_invent_names(self) -> None:
        opening = _opening("Which PR shipped?", survey="WHAT EXISTS\norg/site")
        # The instruction lives in the survey block itself, which the loop builds -- so what
        # this asserts is that the block reaches the prompt intact rather than being summarised.
        assert "WHAT EXISTS" in opening
        assert "org/site" in opening

    def test_an_investigation_without_a_survey_is_unchanged(self) -> None:
        """Nothing is added when there is nothing to add — the shape every existing test and
        every eval scenario already measures."""
        assert "WHAT EXISTS" not in _opening("Why did signups fall?")


class TestWidthPrompting:
    """Asking for breadth, from W&D (arXiv 2602.07359).

    They swept 1/2/3/5/8 tool calls per turn and found three best: on BrowseComp, 66% → 68%
    accuracy with wall clock 1522.6s → 904.2s and turns 45.7 → 23.8. Prompting only.

    Our reason is our own measurement: `loop_model` is 40–44% of wall clock across 3–5 calls at
    ~19 seconds each, and we are output-bound. Fewer turns is the only lever that touches it.

    The tests here are about the *escape hatch*, not the instruction. W&D explicitly does not
    address tools whose arguments depend on a previous tool's output, and we have exactly those
    chains — so an instruction that demanded a quota would push the analyst to invent independent
    work or guess a parameter it should have looked up.
    """

    def test_it_asks_for_breadth(self) -> None:
        opening = _opening("Why did signups fall?")
        assert "SAME message" in opening
        assert "independent of each other" in opening

    def test_the_opening_asks_for_the_widest_step(self) -> None:
        from cortex.agents.investigator import tool_calls_for_step

        assert str(tool_calls_for_step(0)) in _opening("Why did signups fall?")

    def test_the_schedule_descends(self) -> None:
        """The correction. W&D Table 3 compares *schedules*, not only constant widths, on
        BrowseComp/GPT-5-Medium:

            Constant 1   66%   45.7 turns
            Constant 3   68%   23.8 turns   <- what this used to be
            Ascending    63%   36.5 turns
            Descending   74%   23.5 turns   <- this
            Automatic    72%   26.6 turns

        +6 points of accuracy at an identical turn count, so free on the budget that binds."""
        from cortex.agents.investigator import tool_calls_for_step

        widths = [tool_calls_for_step(i) for i in range(6)]
        assert widths == sorted(widths, reverse=True), widths
        assert widths[0] > widths[-1]

    def test_it_holds_at_the_tail_rather_than_running_out(self) -> None:
        """A long investigation keeps the tail width per turn instead of falling off the end of the
        schedule."""
        from cortex.agents.investigator import TOOL_CALL_SCHEDULE, tool_calls_for_step

        assert tool_calls_for_step(50) == TOOL_CALL_SCHEDULE[-1]
        assert tool_calls_for_step(0) == tool_calls_for_step(-1)

    def test_the_tail_leaves_room_for_a_late_discriminating_call(self) -> None:
        """Measured, not inherited. A tail of 1 cost `insufficient_evidence` its discriminating
        GA4 call in both attempts of eval 17 — front-loading five means the analyst picks its five,
        gathers enough to decline, and a tail of one never revisits a source it did not think of at
        step 1. The wide opening is the latency win; the narrow tail was the cost."""
        from cortex.agents.investigator import TOOL_CALL_SCHEDULE

        assert TOOL_CALL_SCHEDULE[-1] >= 2

    def test_every_turn_forbids_padding(self) -> None:
        """W&D's own limitation: they do not address tools whose arguments depend on a previous
        tool's output, and we have exactly those chains. A number with no exemption would be met
        by guessing a repository name rather than looking it up."""
        from cortex.agents.investigator import width_instruction

        for index in (0, 1, 5, 20):
            instruction = width_instruction(index)
            assert "Never invent calls" in instruction or "Do not pad" in instruction

    def test_an_early_turn_states_a_number_rather_than_advice(self) -> None:
        """The paper measured the alternative: their Automatic arm, where the model chose its own
        width, scored 72% at 26.6 turns against Descending's 74% at 23.5, and they conclude the
        LLM "cannot determine the optimal number of tool calls in each iteration"."""
        from cortex.agents.investigator import width_instruction

        early = width_instruction(0)
        assert "Request about 5 tool calls" in early
        assert "Never invent calls" in early

    def test_it_permits_fewer_calls_on_a_dependency(self) -> None:
        """The condition W&D's limitation section names. A repository we have not identified
        cannot be queried in the same turn that discovers it."""
        opening = _opening("Why did signups fall?")
        assert "Ask for fewer" in opening
        assert "know what to ask next" in opening

    def test_it_forbids_padding_to_reach_the_count(self) -> None:
        """A quota would be met by inventing work. The instruction has to say so, because the
        cheapest way to satisfy "make five calls" is five calls that answer nothing."""
        opening = _opening("Why did signups fall?")
        assert "Never invent calls to reach the number" in opening
        assert "never guess a parameter" in opening
