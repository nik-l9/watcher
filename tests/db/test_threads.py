"""Follow-up questions, and the grounding boundary they widen.

A follow-up may cite evidence its parent already paid for. That is the feature — and it means
`GroundingGate`, whose whole job is to refuse a citation it cannot resolve *within this
investigation*, now resolves within a thread. Widening a grounding boundary is exactly the kind
of change that quietly undoes a guarantee, so most of this file is written as the attacks it must
refuse rather than as the feature it enables:

  - a sibling's evidence (same tenant, same parent) must stay uncitable
  - an unrelated investigation's evidence must stay uncitable
  - a parent belonging to another tenant must be refused at creation (F-01's shape)
  - a parent must not be able to cite its child's evidence — the parent's report was gated
    before the child existed, and evidence gathered afterwards cannot be what established it
  - a cycle in `parent_id`, which a self-referencing foreign key permits, must not hang a walk

The one test that matters most is the *unchanged* case: an investigation with no parent must
resolve exactly as it did before, because that is what makes the widening safe to ship at all.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import Evidence, Investigation, Report, Tenant
from cortex.db.threads import (
    MAX_THREAD_DEPTH,
    ParentNotFound,
    ThreadTooDeep,
    ancestor_ids,
    check_parent,
    citable_investigation_ids,
    prior_context,
)
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.tenancy.context import TenantContext


async def _tenant(session: AsyncSession, slug: str) -> TenantContext:
    tenant_id = uuid.uuid4()
    graph_name = graph_name_for_new_tenant(slug, tenant_id)
    session.add(Tenant(id=tenant_id, slug=slug, name=slug, graph_name=graph_name))
    await session.flush()
    return TenantContext(tenant_id=tenant_id, tenant_slug=slug, graph_name=graph_name)


async def _investigation(
    session: AsyncSession,
    tenant: TenantContext,
    *,
    question: str = "Why did signups fall?",
    parent_id: uuid.UUID | None = None,
) -> Investigation:
    row = Investigation(tenant_id=tenant.tenant_id, question=question, parent_id=parent_id)
    session.add(row)
    await session.flush()
    return row


async def _evidence(
    session: AsyncSession,
    tenant: TenantContext,
    investigation: Investigation,
    *,
    capability: str = "event_trend",
) -> Evidence:
    row = Evidence(
        tenant_id=tenant.tenant_id,
        investigation_id=investigation.id,
        tool_name="posthog",
        capability=capability,
        params={"event": "signup"},
        payload={"data": [{"day": "2026-07-14", "unique_sessions": 88}]},
        payload_hash="deadbeef" * 8,
    )
    session.add(row)
    await session.flush()
    return row


class TestTheChain:
    async def test_an_investigation_with_no_parent_has_no_ancestors(
        self, session: AsyncSession
    ) -> None:
        tenant = await _tenant(session, "thread-solo")
        row = await _investigation(session, tenant)
        assert await ancestor_ids(session, tenant, row.id) == []

    async def test_only_itself_is_citable_without_a_parent(self, session: AsyncSession) -> None:
        """The property that makes this safe to ship: for every investigation that is not a
        follow-up, the gate resolves exactly the set it resolved before."""
        tenant = await _tenant(session, "thread-unchanged")
        row = await _investigation(session, tenant)
        assert await citable_investigation_ids(session, tenant, row.id) == {row.id}

    async def test_ancestors_are_nearest_first(self, session: AsyncSession) -> None:
        tenant = await _tenant(session, "thread-order")
        grandparent = await _investigation(session, tenant)
        parent = await _investigation(session, tenant, parent_id=grandparent.id)
        child = await _investigation(session, tenant, parent_id=parent.id)

        assert await ancestor_ids(session, tenant, child.id) == [parent.id, grandparent.id]

    async def test_a_follow_up_can_cite_its_ancestors(self, session: AsyncSession) -> None:
        tenant = await _tenant(session, "thread-cite")
        parent = await _investigation(session, tenant)
        child = await _investigation(session, tenant, parent_id=parent.id)

        citable = await citable_investigation_ids(session, tenant, child.id)
        assert citable == {child.id, parent.id}


class TestTheBoundaryItMustNotCross:
    async def test_a_sibling_is_not_citable(self, session: AsyncSession) -> None:
        """Two follow-ups to one report, asked by different people about different segments, do
        not become one pool of citable facts because they share a parent."""
        tenant = await _tenant(session, "thread-siblings")
        parent = await _investigation(session, tenant)
        first = await _investigation(session, tenant, parent_id=parent.id)
        second = await _investigation(session, tenant, parent_id=parent.id)

        assert second.id not in await citable_investigation_ids(session, tenant, first.id)

    async def test_an_unrelated_investigation_is_not_citable(self, session: AsyncSession) -> None:
        """The predicate the gate had before, preserved: tenant alone was never enough."""
        tenant = await _tenant(session, "thread-unrelated")
        mine = await _investigation(session, tenant)
        other = await _investigation(session, tenant, question="Why did churn rise?")

        assert other.id not in await citable_investigation_ids(session, tenant, mine.id)

    async def test_a_parent_cannot_cite_its_child(self, session: AsyncSession) -> None:
        """Ancestors, never descendants. The parent's report was gated before the child
        existed, so evidence gathered afterwards cannot be what established it — and a report
        that could acquire new grounds after delivery would make the gate meaningless."""
        tenant = await _tenant(session, "thread-descendant")
        parent = await _investigation(session, tenant)
        child = await _investigation(session, tenant, parent_id=parent.id)

        assert await citable_investigation_ids(session, tenant, parent.id) == {parent.id}
        assert child.id not in await citable_investigation_ids(session, tenant, parent.id)

    async def test_a_foreign_ancestor_is_not_citable(self, session: AsyncSession) -> None:
        """The chain is walked with a tenant predicate at every hop. `parent_id` is a foreign
        key, which proves a row exists and says nothing about who owns it — F-01 exactly."""
        mine = await _tenant(session, "thread-mine")
        theirs = await _tenant(session, "thread-theirs")
        foreign_parent = await _investigation(session, theirs)
        # Written directly, bypassing `check_parent`, because the question here is whether the
        # *read* path holds even if a bad row exists.
        child = await _investigation(session, mine, parent_id=foreign_parent.id)

        citable = await citable_investigation_ids(session, mine, child.id)
        assert citable == {child.id}
        assert foreign_parent.id not in citable

    async def test_a_cycle_does_not_hang_the_walk(self, session: AsyncSession) -> None:
        """A self-referencing foreign key permits a cycle at the database level, and an
        unbounded walk over one would hang whichever request is rendering a report."""
        tenant = await _tenant(session, "thread-cycle")
        first = await _investigation(session, tenant)
        second = await _investigation(session, tenant, parent_id=first.id)
        first.parent_id = second.id
        await session.flush()

        chain = await ancestor_ids(session, tenant, second.id)
        assert len(chain) <= MAX_THREAD_DEPTH
        assert len(set(chain)) == len(chain)

    async def test_the_chain_is_depth_bounded(self, session: AsyncSession) -> None:
        """A long chain must not make a report render slowly, and a tenth follow-up citing the
        first investigation's evidence is citing an observation from a different day about a
        different question."""
        tenant = await _tenant(session, "thread-deep")
        current = await _investigation(session, tenant)
        for _ in range(MAX_THREAD_DEPTH + 4):
            current = await _investigation(session, tenant, parent_id=current.id)

        assert len(await ancestor_ids(session, tenant, current.id)) <= MAX_THREAD_DEPTH


class TestValidatingAParentFromARequest:
    async def test_a_parent_from_another_tenant_is_refused(self, session: AsyncSession) -> None:
        mine = await _tenant(session, "check-mine")
        theirs = await _tenant(session, "check-theirs")
        foreign = await _investigation(session, theirs)

        with pytest.raises(ParentNotFound):
            await check_parent(session, mine, foreign.id)

    async def test_an_absent_parent_is_refused_the_same_way(self, session: AsyncSession) -> None:
        """One error for both, so an id cannot be probed for existence."""
        tenant = await _tenant(session, "check-absent")
        with pytest.raises(ParentNotFound):
            await check_parent(session, tenant, uuid.uuid4())

    async def test_a_valid_parent_passes(self, session: AsyncSession) -> None:
        tenant = await _tenant(session, "check-ok")
        parent = await _investigation(session, tenant)
        await check_parent(session, tenant, parent.id)

    async def test_a_chain_at_the_limit_is_refused_rather_than_truncated(
        self, session: AsyncSession
    ) -> None:
        """Refused loudly at creation. A follow-up that silently could not see its
        grandparent's evidence would lose findings for a reason nobody could observe."""
        tenant = await _tenant(session, "check-deep")
        current = await _investigation(session, tenant)
        for _ in range(MAX_THREAD_DEPTH):
            current = await _investigation(session, tenant, parent_id=current.id)

        with pytest.raises(ThreadTooDeep):
            await check_parent(session, tenant, current.id)


class TestWhatTheAnalystIsTold:
    async def test_the_parents_conclusion_is_labelled_as_a_conclusion(
        self, session: AsyncSession
    ) -> None:
        """It was gated and verified when delivered, but it is still a claim. A follow-up that
        cited it as an observation would be laundering prose into evidence."""
        tenant = await _tenant(session, "ctx-conclusion")
        parent = await _investigation(session, tenant)
        session.add(
            Report(
                tenant_id=tenant.tenant_id,
                investigation_id=parent.id,
                body={
                    "question": "Why did signups fall?",
                    "executive_summary": [
                        {"text": "Signups fell 12% after the 14 July deploy.", "evidence_ids": []}
                    ],
                },
                confidence=0.6,
            )
        )
        await session.flush()

        context = await prior_context(session, tenant, parent.id)
        assert "Signups fell 12%" in context
        assert "not evidence" in context

    async def test_the_evidence_inventory_carries_ids(self, session: AsyncSession) -> None:
        """The point of the feature: an id the analyst can cite instead of a call it would
        otherwise pay for."""
        tenant = await _tenant(session, "ctx-inventory")
        parent = await _investigation(session, tenant)
        row = await _evidence(session, tenant, parent)

        context = await prior_context(session, tenant, parent.id)
        assert str(row.id) in context
        assert "posthog__event_trend" in context

    async def test_a_payload_preview_is_included(self, session: AsyncSession) -> None:
        """This test asserted the opposite until a live follow-up proved it wrong.

        The inventory originally listed ids and parameters only, on the reasoning that the row
        is already stored and reprinting payloads would cost more than the call it saves. Asked
        "using only what you already observed, on which day was conversation_created highest?",
        the analyst concluded in one step having called nothing -- it could see that the trend
        existed and could not see what was in it. An invitation to cite an observation whose
        contents are withheld is not an invitation to reuse it.
        """
        tenant = await _tenant(session, "ctx-preview")
        parent = await _investigation(session, tenant)
        await _evidence(session, tenant, parent)

        context = await prior_context(session, tenant, parent.id)
        assert "2026-07-14" in context
        assert "unique_sessions" in context

    async def test_an_oversized_payload_is_withheld_whole_rather_than_truncated(
        self, session: AsyncSession
    ) -> None:
        """The second live failure, and the more interesting one.

        The first version truncated at 400 characters. Asked which day was highest, the analyst
        read a `conversation_created` series cut off at 16 July, took the maximum of the visible
        prefix, and stated it as the maximum of the range. The verifier read the full payload and
        refused every claim — the system working, over an error the design had invited.

        A fragment reads as the whole, and an ellipsis is not enough of a warning: the numbers
        before it are real, so a model reasoning over them is not obviously wrong. Nothing reads
        as nothing; half a series reads as a series."""
        tenant = await _tenant(session, "ctx-truncated")
        parent = await _investigation(session, tenant)
        row = Evidence(
            tenant_id=tenant.tenant_id,
            investigation_id=parent.id,
            tool_name="posthog",
            capability="event_trend",
            params={"event": "signup"},
            payload={"data": [{"day": f"2026-07-{d:02d}", "n": d} for d in range(1, 29)] * 20},
            payload_hash="deadbeef" * 8,
        )
        session.add(row)
        await session.flush()

        context = await prior_context(session, tenant, parent.id, max_preview_chars=200)
        assert "not shown" in context
        # No fragment of the payload appears: not the first day, not the last, nothing to
        # mistake for the series.
        assert "2026-07-01" not in context
        assert '"n":' not in context
        assert len(context) < 2000

    async def test_the_total_preview_budget_is_enforced(self, session: AsyncSession) -> None:
        """Budgeted across the inventory rather than per row, so many observations stay
        readable while one enormous one cannot crowd out the rest."""
        tenant = await _tenant(session, "ctx-budget")
        parent = await _investigation(session, tenant)
        for index in range(20):
            await _evidence(session, tenant, parent, capability=f"trend_{index}")

        context = await prior_context(
            session, tenant, parent.id, max_preview_chars=200, max_total_preview_chars=300
        )
        # Every row is still listed with its id and parameters, because those are citable facts;
        # only the payloads past the budget go unshown.
        assert "not shown" in context
        assert context.count("posthog__trend_") == 20

    async def test_another_tenants_evidence_is_not_listed(self, session: AsyncSession) -> None:
        """The inventory names ids the analyst is invited to cite, so it is the last place a
        foreign row should be able to appear."""
        mine = await _tenant(session, "ctx-mine")
        theirs = await _tenant(session, "ctx-theirs")
        parent = await _investigation(session, mine)
        foreign = await _investigation(session, theirs)
        foreign_row = await _evidence(session, theirs, foreign)

        context = await prior_context(session, mine, parent.id)
        assert str(foreign_row.id) not in context

    async def test_a_parent_with_no_report_still_produces_context(
        self, session: AsyncSession
    ) -> None:
        """A follow-up to a failed or cancelled investigation. Its evidence is still real and
        still citable, so it must not raise for want of a conclusion."""
        tenant = await _tenant(session, "ctx-noreport")
        parent = await _investigation(session, tenant)
        await _evidence(session, tenant, parent)

        context = await prior_context(session, tenant, parent.id)
        assert "PRIOR INVESTIGATION" in context
