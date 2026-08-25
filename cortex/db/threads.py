"""Follow-up questions, and what a follow-up may cite.

A GTM analyst is asked follow-ups. *"Why did signups fall?"* is answered, and the next thing a
human says is *"and what about mobile?"* — which is not a new question, it is the same
investigation continued. Cortex had no way to express that: every question started an
investigation that had never seen the previous one, re-gathered the same evidence, could reach a
slightly different answer from the same data, and billed for both.

The gap was named by τ-bench, which scores multi-turn tool use on final state rather than on the
transcript. It cannot be run against Cortex — not for want of a harness, but because Cortex had
no conversation to score. See `docs/benchmarks.md`.

## The part that needs care

`GroundingGate` resolves a cited evidence id with **two** predicates: the tenant *and* the
investigation. Its docstring says why both are load-bearing — tenant alone would let a report
cite evidence from the same tenant's unrelated investigation, and investigation alone would
trust a foreign key that F-01 showed can be attacker-supplied.

A follow-up that reuses its parent's evidence has to widen that scope, and widening a grounding
boundary is exactly the kind of change that quietly undoes a guarantee. So the widening is
constrained in three ways:

  - **The scope is the thread, not the tenant.** An investigation may cite evidence from itself
    and from its ancestors. A sibling, a cousin, and an unrelated investigation of the same
    tenant all remain uncitable.
  - **The thread is computed from the database, never from the request.** A caller supplies a
    `parent_id` once, at creation, where it is checked against the tenant. Everything afterwards
    walks stored rows. A report cannot widen the set of evidence it is allowed to cite, because
    nothing it produces is an input to this function.
  - **Ancestors only, never descendants.** A follow-up may cite what came before it. The parent
    must not be able to cite its child's evidence: the parent's report is already written and
    already gated, and evidence gathered after it was delivered cannot be what established it.

## Why ancestors rather than the whole tree

A thread can branch — two different follow-ups to one report. Each branch can see the trunk and
neither can see the other, which is both the intuitive reading and the safe one: two follow-ups
asked by different people, about different segments, do not become one pool of citable facts
because they share a parent.
"""

from __future__ import annotations

import json
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import Evidence, Investigation
from cortex.tenancy.context import TenantContext

#: How many ancestors a follow-up chain may reach back through.
#:
#: Bounded for two reasons. A thread is a conversation, and a tenth follow-up citing the first
#: investigation's evidence is citing an observation from a different day about a different
#: question — staleness the gate cannot detect, because the row resolves perfectly. And an
#: unbounded walk over a self-referencing table is one bad row away from a loop; the visited set
#: below stops a cycle regardless, but a depth limit means a long chain cannot make a report
#: render slowly either.
MAX_THREAD_DEPTH = 8


class ThreadTooDeep(ValueError):
    """A follow-up chain longer than `MAX_THREAD_DEPTH`.

    Raised at creation rather than silently truncating the citable scope. A follow-up that
    quietly could not see its grandparent's evidence would produce a report missing findings for
    a reason nobody could observe.
    """


class ParentNotFound(ValueError):
    """A parent that does not exist, or belongs to another tenant.

    One error for both, so an id cannot be probed for existence — the same rule the endpoints
    follow. This is the F-01 check: `parent_id` arrives from a request and must never be trusted
    as proof of ownership.
    """


async def ancestor_ids(
    session: AsyncSession,
    tenant: TenantContext,
    investigation_id: uuid.UUID,
) -> list[uuid.UUID]:
    """The chain above this investigation, nearest parent first.

    Every row is confirmed to belong to `tenant`. A chain that leaves the tenant stops there
    rather than raising: the boundary holds either way, and a follow-up should not become
    unreadable because an ancestor was reassigned.
    """
    chain: list[uuid.UUID] = []
    # Guards against a cycle, which a self-referencing foreign key permits at the database level.
    # An unbounded walk over one would hang the request that renders a report.
    seen: set[uuid.UUID] = {investigation_id}
    current = investigation_id

    for _ in range(MAX_THREAD_DEPTH):
        parent_id = await session.scalar(
            select(Investigation.parent_id).where(
                Investigation.id == current,
                Investigation.tenant_id == tenant.tenant_id,
            )
        )
        if parent_id is None or parent_id in seen:
            break
        # Confirmed to exist *and* to be this tenant's before it joins the citable scope. The
        # column is a foreign key, which proves the row exists and says nothing about who owns
        # it.
        owned = await session.scalar(
            select(Investigation.id).where(
                Investigation.id == parent_id,
                Investigation.tenant_id == tenant.tenant_id,
            )
        )
        if owned is None:
            break
        chain.append(parent_id)
        seen.add(parent_id)
        current = parent_id

    return chain


async def citable_investigation_ids(
    session: AsyncSession,
    tenant: TenantContext,
    investigation_id: uuid.UUID,
) -> set[uuid.UUID]:
    """Whose evidence this investigation's report may cite: itself plus its ancestors.

    This is the set `GroundingGate` resolves against. For an investigation with no parent it is
    a one-element set, so the gate's behaviour is unchanged for every investigation that is not
    a follow-up — which is the property that makes this safe to add.
    """
    return {investigation_id, *await ancestor_ids(session, tenant, investigation_id)}


async def check_parent(
    session: AsyncSession,
    tenant: TenantContext,
    parent_id: uuid.UUID,
) -> None:
    """Validate a `parent_id` arriving from a request.

    Called at creation, which is the only place an untrusted parent enters the system. Raises
    `ParentNotFound` for an absent or foreign parent, and `ThreadTooDeep` when attaching to it
    would exceed the depth limit.
    """
    owned = await session.scalar(
        select(Investigation.id).where(
            Investigation.id == parent_id,
            Investigation.tenant_id == tenant.tenant_id,
        )
    )
    if owned is None:
        raise ParentNotFound(f"investigation {parent_id} is not available to this tenant")

    # The new investigation sits one level below the parent, so the parent's own chain may be
    # at most MAX_THREAD_DEPTH - 1.
    depth = len(await ancestor_ids(session, tenant, parent_id))
    if depth + 1 >= MAX_THREAD_DEPTH:
        raise ThreadTooDeep(
            f"this follow-up would be {depth + 2} deep; the limit is {MAX_THREAD_DEPTH}. "
            "Ask it as a new investigation instead."
        )


async def prior_context(
    session: AsyncSession,
    tenant: TenantContext,
    parent_id: uuid.UUID,
    *,
    max_evidence: int = 40,
    #: Per-row allowance. Sized above a typical metric series (~1,400 characters for a
    #: three-week daily trend) so the common case is included whole rather than dropped.
    max_preview_chars: int = 2_500,
    max_total_preview_chars: int = 12_000,
) -> str:
    """What the analyst is told about the investigation this one follows.

    Two things, and the split matters.

    **The parent's answer**, as prose, so a question like "and what about mobile?" is
    interpretable at all. It is labelled as a prior conclusion rather than as evidence: it was
    gated and verified when it was delivered, but it is still a *claim*, and a follow-up that
    cited it as though it were an observation would be laundering prose into evidence.

    **An inventory of the parent's evidence**, with ids and a bounded preview of each payload,
    so the analyst can cite an observation it already has instead of paying to fetch it again.

    The preview is the part that took a live failure to get right. The first version listed ids
    and parameters only, reasoning that the row is already stored and that reprinting payloads
    would cost more than the call it saves. A follow-up -- *"using only what you already
    observed, on which day was conversation_created highest?"* -- then concluded in one step
    having called nothing: it could see that a `conversation_created` trend existed and could not
    see what was in it. It had been invited to cite an observation whose contents were withheld.

    So each row carries its payload -- and **whole or not at all**, which took a second live
    failure to learn. The first attempt truncated at 400 characters. Asked which day was highest,
    the analyst read a `conversation_created` series cut off at 16 July, took the maximum of the
    visible prefix, and stated it as the maximum of the range. The adversarial verifier then read
    the full payload and refused every claim, which is the system working exactly as designed --
    but the design had invited the error.

    A fragment reads as the whole. An ellipsis is not enough of a warning, because the numbers
    before it are real and a model reasoning over them is not obviously wrong. So a payload that
    does not fit its allowance is replaced by a note naming its size, which is unambiguous: there
    is nothing here to reason from, call the capability. Nothing reads as nothing; half a series
    reads as a series.

    The budget is a real cost -- a few thousand tokens on a follow-up's first turn, cached
    thereafter -- bought against vendor calls that cost seconds of wall clock and can fail.
    """
    from cortex.db.models import Report  # local import: avoids a cycle via reports.gate

    report = (
        await session.execute(
            select(Report)
            .where(Report.investigation_id == parent_id, Report.tenant_id == tenant.tenant_id)
            .order_by(Report.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    question = await session.scalar(
        select(Investigation.question).where(
            Investigation.id == parent_id, Investigation.tenant_id == tenant.tenant_id
        )
    )

    rows = (
        (
            await session.execute(
                select(Evidence)
                .where(
                    Evidence.investigation_id == parent_id,
                    Evidence.tenant_id == tenant.tenant_id,
                )
                .order_by(Evidence.observed_at.asc())
                .limit(max_evidence)
            )
        )
        .scalars()
        .all()
    )

    lines = [
        "PRIOR INVESTIGATION (this question is a follow-up to it)",
        f"Earlier question: {question or 'unknown'}",
    ]

    if report is not None and isinstance(report.body, dict):
        summary = [
            claim.get("text", "")
            for claim in report.body.get("executive_summary", [])
            if isinstance(claim, dict)
        ]
        if summary:
            lines.append("What was concluded then (a prior CONCLUSION, not evidence):")
            lines += [f"  - {text}" for text in summary if text]

    if rows:
        lines.append(
            "Observations already gathered for that investigation. These are real evidence "
            "rows and you MAY cite these ids directly -- if one of them already answers part "
            "of this question, cite it rather than calling the tool again:"
        )
        spent = 0
        for row in rows:
            params = ", ".join(
                f"{key}={str(value)[:40]}" for key, value in (row.params or {}).items()
            )
            lines.append(f"  - {row.id}  {row.tool_name}__{row.capability}({params})")

            # The preview. Budgeted across the whole inventory rather than per row, so forty
            # small observations stay readable while one enormous one cannot crowd out the rest
            # -- and the ellipsis is load-bearing: it tells the analyst there is more, so a
            # claim needing the remainder becomes a call rather than an assumption.
            if spent >= max_total_preview_chars:
                lines.append("      (payload not shown -- call the capability to read it)")
                continue
            preview = json.dumps(row.payload, default=str, separators=(",", ":"))
            allowance = min(max_preview_chars, max_total_preview_chars - spent)
            if len(preview) > allowance:
                # Whole or nothing. A truncated series is the worst of both: the numbers in it
                # are real, so a model reasoning over them is not obviously wrong, and it will
                # state a maximum drawn from the visible prefix as the maximum of the range.
                # That happened, and the verifier had to catch it.
                lines.append(
                    f"      (payload is {len(preview):,} characters -- not shown. Call the "
                    "capability to read it; do not infer its contents.)"
                )
                continue
            spent += len(preview)
            lines.append(f"      {preview}")
        lines.append(
            "Anything not on that list, you have not observed. Gather it or say you could not. "
            "Where a payload is not shown, you have not seen its contents: cite that observation "
            "only for what its parameters establish, or call the capability to read it."
        )

    return "\n".join(lines)
