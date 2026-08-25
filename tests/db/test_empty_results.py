"""Empty results, and the three layers that keep them from becoming false absences.

One bug has now shipped four times in two days:

| # | What happened | What the analyst reported |
|---|---|---|
| 1 | Slack query too narrow | "nobody discussed the wins" |
| 2 | GitHub `environment=production` does not exist | "nothing was deployed" |
| 3 | PostHog queried the wrong project | "that event never fired" |
| 4 | Slack `text` empty because apps post Block Kit | "no complaints found" |

Each was fixed individually. Each fix was correct. None of them stopped the next one,
because the class was never closed — and this is the class that the grounding machinery is
structurally unable to catch: the citation is real, the hash matches, the row belongs to the
investigation, and the result genuinely was empty. Every check passes and the conclusion is
wrong.

Three layers, tested here:

1. **A capability must declare where its results live.** You cannot add one without deciding
   what "empty" means for it.
2. **The loop labels an empty observation as empty**, in the text the analyst reads, with the
   filters that produced it and whatever the source can say about it.
3. **The gate discloses** when a delivered report's own citations point at observations that
   found nothing.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.agents.investigator import _empty_warning
from cortex.db.models import Evidence
from cortex.reports.gate import _empty_evidence_notes, _looks_empty
from cortex.tools.base import NEVER_EMPTY, Capability, ToolResult
from cortex.tools.executor import ExecutedTool
from cortex.tools.registry import gtm_analyst_registry

SCHEMA = {"type": "object", "additionalProperties": False, "properties": {}}


async def _handler(ctx: object, **kwargs: object) -> ToolResult:
    del ctx, kwargs
    return ToolResult(payload={})


def _capability(**kwargs: object) -> Capability:
    return Capability(
        name="probe",
        description="probe",
        params_schema=dict(SCHEMA),
        handler=_handler,
        **kwargs,  # type: ignore[arg-type]
    )


# ------------------------------------------------------ layer 1: the declaration


class TestEveryCapabilityDeclaresWhatEmptyMeans:
    def test_a_capability_without_a_result_key_cannot_be_constructed(self) -> None:
        """The author of a capability is the only person who knows which key holds its
        results. Asking them once is cheaper than the analyst reporting "nothing happened"
        about data it never saw."""
        with pytest.raises(ValueError, match="must declare result_key"):
            _capability()

    def test_never_empty_is_an_accepted_answer_but_needs_a_reason(self) -> None:
        """Some results genuinely cannot be empty. That is a decision, and it should be
        visible in review rather than being an omission that looks like an oversight."""
        capability = _capability(result_key=f"{NEVER_EMPTY}: the allowlist is compiled in")
        assert capability.is_empty({"anything": []}) is False

    def test_every_shipped_capability_declares_one(self) -> None:
        """The guard that makes this hold for the next connector, not just today's."""
        registry = gtm_analyst_registry()
        for tool_name in registry.tool_names:
            tool = registry.get(tool_name)
            for name in tool.capability_names:
                capability = tool.capability(name)
                assert capability.result_key, f"{tool_name}.{name}"

    def test_a_never_empty_declaration_explains_itself(self) -> None:
        """`NEVER_EMPTY` alone would be a way to opt out silently."""
        registry = gtm_analyst_registry()
        for tool_name in registry.tool_names:
            tool = registry.get(tool_name)
            for name in tool.capability_names:
                key = tool.capability(name).result_key
                if key.startswith(NEVER_EMPTY):
                    assert len(key) > len(NEVER_EMPTY) + 2, f"{tool_name}.{name} gives no reason"


class TestDetectingEmptiness:
    def test_an_empty_list_under_the_declared_key_is_empty(self) -> None:
        assert _capability(result_key="messages").is_empty({"messages": [], "count": 0}) is True

    def test_a_populated_list_is_not_empty(self) -> None:
        assert _capability(result_key="messages").is_empty({"messages": [{"ts": "1"}]}) is False

    def test_a_missing_key_is_treated_as_empty(self) -> None:
        """A result that does not contain the thing it promised has not found it. Returning
        False here would be exactly the hole this mechanism exists to close."""
        assert _capability(result_key="messages").is_empty({"count": 0}) is True

    def test_the_hint_collects_what_the_source_said(self) -> None:
        """This is the line that turns "nothing found" into "you asked for an environment
        that does not exist" — bug #2, made visible."""
        hint = _capability(result_key="deployments").empty_hint(
            {
                "deployments": [],
                "note": "no deployment matched environment='production'",
                "environments_available": ["dev-deploy", "staging - docs"],
            }
        )
        assert "production" in hint
        assert "dev-deploy" in hint

    def test_a_total_that_disagrees_with_the_rows_is_surfaced(self) -> None:
        """Slack reporting 40 matches while returning none is a permission or paging
        problem, not an absence — bug #1 and #4's shape."""
        hint = _capability(result_key="messages").empty_hint({"messages": [], "total_matching": 40})
        assert "total_matching=40" in hint


# ------------------------------------------------------ layer 2: what the analyst reads


def _executed(*, is_empty: bool, hint: str = "") -> ExecutedTool:
    from cortex.tools.base import Freshness

    return ExecutedTool(
        evidence_id=uuid.uuid4(),
        tool_name="github",
        capability="deployment_history",
        payload={"deployments": []},
        source_ref="https://github.com/acme/web/deployments",
        freshness=Freshness.LIVE,
        payload_hash="x",
        duration_ms=10,
        is_empty=is_empty,
        empty_hint=hint,
    )


class TestTheAnalystCannotMissAnEmptyResult:
    def test_a_populated_result_gets_no_warning(self) -> None:
        """A warning on every observation would be noise, and noise is ignored."""
        assert _empty_warning(_executed(is_empty=False)) == ""

    def test_an_empty_result_says_empty_is_not_absent(self) -> None:
        warning = _empty_warning(_executed(is_empty=True))
        assert "THIS RESULT IS EMPTY" in warning
        assert "not evidence that nothing happened" in warning
        # And it says what to do about it, because a warning with no next step gets
        # acknowledged and then ignored.
        assert "widen or remove one filter" in warning

    def test_the_sources_own_explanation_is_included(self) -> None:
        warning = _empty_warning(
            _executed(
                is_empty=True,
                hint="environments_available=['dev-deploy', 'staging - docs']",
            )
        )
        assert "dev-deploy" in warning


# ------------------------------------------------------ layer 3: the report discloses


def _evidence(tenant_id: uuid.UUID, investigation_id: uuid.UUID, payload: dict) -> Evidence:
    return Evidence(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        investigation_id=investigation_id,
        tool_name="slack",
        capability="search_messages",
        params={"query": "onboarding"},
        payload=payload,
        payload_hash="x",
    )


class TestTheReportDisclosesConclusionsBuiltOnSilence:
    """The citation gate cannot catch this. The row is real; it just says nothing."""

    def test_a_report_citing_only_empty_observations_is_disclosed(self) -> None:
        tenant_id, investigation_id = uuid.uuid4(), uuid.uuid4()
        rows = {
            "a": _evidence(tenant_id, investigation_id, {"messages": [], "count": 0}),
            "b": _evidence(tenant_id, investigation_id, {"messages": [{"ts": "1"}]}),
        }
        by_id = {row.id: row for row in rows.values()}
        notes = _empty_evidence_notes(by_id, set(by_id))

        assert len(notes) == 1
        assert "1 of the 2 cited observation(s) returned no results" in notes[0].note
        assert "not the same as what does not exist" in notes[0].note
        assert notes[0].evidence_ids == [rows["a"].id]

    def test_nothing_is_said_when_every_citation_carried_results(self) -> None:
        tenant_id, investigation_id = uuid.uuid4(), uuid.uuid4()
        row = _evidence(tenant_id, investigation_id, {"messages": [{"ts": "1"}]})
        assert _empty_evidence_notes({row.id: row}, {row.id}) == []

    def test_uncited_empty_observations_are_not_mentioned(self) -> None:
        """Exploring a hypothesis that turned out to be unsupported is good practice. Only
        the observations the conclusions actually rest on are disclosed."""
        tenant_id, investigation_id = uuid.uuid4(), uuid.uuid4()
        cited = _evidence(tenant_id, investigation_id, {"messages": [{"ts": "1"}]})
        explored = _evidence(tenant_id, investigation_id, {"messages": []})
        by_id = {cited.id: cited, explored.id: explored}
        assert _empty_evidence_notes(by_id, {cited.id}) == []

    def test_a_scalar_observation_is_not_called_empty(self) -> None:
        """A single PR's state has no result container. There is nothing here to have been
        empty, and a false disclosure is a wasted line that teaches readers to skim."""
        assert _looks_empty({"number": 913, "state": "closed", "merged_at": "..."}) is False

    def test_the_four_real_bugs_would_now_be_disclosed(self) -> None:
        """The regression test for the class rather than for one instance."""
        assert _looks_empty({"messages": [], "total_matching": 0}) is True  # Slack, narrow query
        assert _looks_empty({"deployments": [], "count": 0}) is True  # GitHub, bad environment
        assert _looks_empty({"series": [], "total": 0}) is True  # PostHog, wrong project
        assert _looks_empty({"messages": []}) is True  # Slack, Block Kit text


class TestEndToEndThroughTheGate:
    async def test_a_delivered_report_carries_the_disclosure(self, session: AsyncSession) -> None:
        """Proof it reaches a reader, not just that the helper works."""
        from cortex.db.models import Investigation, Tenant
        from cortex.memory.naming import graph_name_for_new_tenant
        from cortex.reports.gate import GroundingGate
        from cortex.reports.schema import InvestigationReport
        from cortex.tenancy.context import TenantContext
        from cortex.tools.executor import canonical_hash

        tenant_id = uuid.uuid4()
        slug = "empty-test"
        session.add(
            Tenant(
                id=tenant_id,
                slug=slug,
                name=slug,
                graph_name=graph_name_for_new_tenant(slug, tenant_id),
            )
        )
        await session.flush()
        ctx = TenantContext(
            tenant_id=tenant_id,
            tenant_slug=slug,
            graph_name=graph_name_for_new_tenant(slug, tenant_id),
        )
        investigation = Investigation(tenant_id=tenant_id, question="did anyone complain?")
        session.add(investigation)
        await session.flush()

        payload = {"messages": [], "count": 0}
        row = Evidence(
            tenant_id=tenant_id,
            investigation_id=investigation.id,
            tool_name="slack",
            capability="search_messages",
            params={"query": "broken"},
            payload=payload,
            payload_hash=canonical_hash(payload),
        )
        session.add(row)
        await session.flush()

        report = InvestigationReport(
            question="did anyone complain?",
            executive_summary=[
                {"text": "Nobody reported a problem.", "evidence_ids": [str(row.id)]}
            ],
            confidence="high",
        )
        result = await GroundingGate().apply(
            session, ctx, investigation_id=investigation.id, report=report
        )

        disclosures = [n.note for n in result.report.data_quality]
        assert any("returned no results" in note for note in disclosures), disclosures
