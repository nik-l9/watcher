"""Executing a step's tool calls concurrently.

Measured motive: eight tool calls in one investigation took 23.7 seconds serially at a mean of
3.0 seconds each — 17% of a 143-second wall clock spent waiting on one request at a time.

**The hazard this file exists for.** `execute` interleaves database work with the network call:
the F-01 ownership check before it, the F-05 savepoint that writes `Evidence` after it, and an
audit row on either path. `AsyncSession` is not safe for concurrent use, and concurrent
`begin_nested()` savepoints on one session would interleave and corrupt each other. So
`execute_many` separates the phases — serial preflight, concurrent network, serial record —
rather than wrapping the whole method in `gather`.

That makes the interesting tests the ones about what must **not** have changed: every guarantee
`execute` gives per call, `execute_many` must give per batch. Concurrency is the easy part; not
losing an invariant while adding it is the hard part.
"""

from __future__ import annotations

import asyncio
import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import Evidence, Investigation, Tenant, ToolCall
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.tenancy.context import TenantContext
from cortex.tenancy.limits import Decision, RateLimiter
from cortex.tools.base import (
    Capability,
    CredentialProvider,
    InvalidParams,
    RateLimited,
    Tool,
    ToolContext,
    ToolError,
    ToolRegistry,
    ToolResult,
    UpstreamError,
)
from cortex.tools.executor import ExecutedTool, ToolExecutor

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"tag": {"type": "string"}},
}


class _Slow(Tool):
    """A connector whose calls sleep, so serial and concurrent are distinguishable."""

    name = "slow"
    provider = None

    def __init__(self, *, delay: float = 0.20, fail_on: set[str] | None = None) -> None:
        self.delay = delay
        self.fail_on = fail_on or set()
        self.concurrent = 0
        self.peak_concurrent = 0
        self.invocations: list[str] = []
        super().__init__()

    def capabilities(self) -> list[Capability]:
        return [
            Capability(
                name="fetch",
                description="Sleep, then return rows.",
                params_schema=SCHEMA,
                handler=self.fetch,
                result_key="rows",
            )
        ]

    async def fetch(self, ctx: ToolContext, *, tag: str = "") -> ToolResult:
        self.invocations.append(tag)
        self.concurrent += 1
        self.peak_concurrent = max(self.peak_concurrent, self.concurrent)
        try:
            await asyncio.sleep(self.delay)
            if tag in self.fail_on:
                raise UpstreamError(f"slow: upstream refused {tag}")
            return ToolResult(payload={"rows": [{"tag": tag}]}, source_ref=f"slow://{tag}")
        finally:
            self.concurrent -= 1


async def _setup(session: AsyncSession, slug: str) -> tuple[TenantContext, uuid.UUID]:
    tenant_id = uuid.uuid4()
    graph_name = graph_name_for_new_tenant(slug, tenant_id)
    session.add(Tenant(id=tenant_id, slug=slug, name=slug, graph_name=graph_name))
    await session.flush()
    row = Investigation(tenant_id=tenant_id, question="Why did signups fall?")
    session.add(row)
    await session.flush()
    return (
        TenantContext(tenant_id=tenant_id, tenant_slug=slug, graph_name=graph_name),
        row.id,
    )


def _executor(tool: _Slow, limiter: RateLimiter | None = None) -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(tool)
    return ToolExecutor(registry, limiter=limiter)


def _calls(*tags: str) -> list[tuple[str, dict]]:
    return [("slow__fetch", {"tag": tag}) for tag in tags]


class TestItIsActuallyConcurrent:
    async def test_the_network_calls_overlap(self, session: AsyncSession) -> None:
        """The whole point. Four 200ms calls serially are 800ms; overlapped they are ~200ms."""
        tenant, investigation_id = await _setup(session, "many-overlap")
        tool = _Slow(delay=0.20)

        started = asyncio.get_running_loop().time()
        results = await _executor(tool).execute_many(
            session,
            tenant,
            investigation_id=investigation_id,
            calls=_calls("a", "b", "c", "d"),
        )
        elapsed = asyncio.get_running_loop().time() - started

        assert len(results) == 4
        assert all(isinstance(r, ExecutedTool) for r in results)
        assert tool.peak_concurrent == 4, "calls did not overlap"
        assert elapsed < 0.60, f"{elapsed:.2f}s suggests serial execution"

    async def test_results_come_back_in_call_order(self, session: AsyncSession) -> None:
        """Positional, because the loop zips them against `response.tool_requests` to pair each
        result with its `tool_use_id`. A reordered batch would attach observations to the wrong
        calls — every citation still resolving, every one describing the wrong request."""
        tenant, investigation_id = await _setup(session, "many-order")
        # Descending delays, so completion order is the reverse of call order.
        tool = _Slow(delay=0.0)

        results = await _executor(tool).execute_many(
            session,
            tenant,
            investigation_id=investigation_id,
            calls=_calls("first", "second", "third"),
        )

        tags = [r.payload["rows"][0]["tag"] for r in results if isinstance(r, ExecutedTool)]
        assert tags == ["first", "second", "third"]

    async def test_an_empty_batch_does_nothing(self, session: AsyncSession) -> None:
        tenant, investigation_id = await _setup(session, "many-empty")
        tool = _Slow()
        assert (
            await _executor(tool).execute_many(
                session, tenant, investigation_id=investigation_id, calls=[]
            )
            == []
        )
        assert tool.invocations == []


class TestOneFailureDoesNotTakeTheBatch:
    async def test_a_failing_call_returns_an_error_value(self, session: AsyncSession) -> None:
        """Failures are values, not raises. A batch is several independent attempts, and one
        rate limit must not discard the observations that succeeded beside it."""
        tenant, investigation_id = await _setup(session, "many-partial")
        tool = _Slow(delay=0.0, fail_on={"bad"})

        results = await _executor(tool).execute_many(
            session,
            tenant,
            investigation_id=investigation_id,
            calls=_calls("good", "bad", "alsogood"),
        )

        assert isinstance(results[0], ExecutedTool)
        assert isinstance(results[1], ToolError)
        assert isinstance(results[2], ExecutedTool)

    async def test_a_failure_does_not_cancel_calls_in_flight(self, session: AsyncSession) -> None:
        """`gather` without `return_exceptions` cancels its siblings on the first exception. The
        sibling calls here are slow, so an unguarded gather would lose them."""
        tenant, investigation_id = await _setup(session, "many-nocancel")
        tool = _Slow(delay=0.15, fail_on={"bad"})

        results = await _executor(tool).execute_many(
            session,
            tenant,
            investigation_id=investigation_id,
            calls=_calls("bad", "slow1", "slow2"),
        )

        assert isinstance(results[0], ToolError)
        assert isinstance(results[1], ExecutedTool)
        assert isinstance(results[2], ExecutedTool)

    async def test_an_unresolvable_capability_is_an_error_not_a_crash(
        self, session: AsyncSession
    ) -> None:
        """The model can invent a tool name. That is a correctable mistake to feed back, not a
        reason to lose the rest of the batch."""
        tenant, investigation_id = await _setup(session, "many-unknown")
        tool = _Slow(delay=0.0)

        results = await _executor(tool).execute_many(
            session,
            tenant,
            investigation_id=investigation_id,
            calls=[("slow__fetch", {"tag": "ok"}), ("nosuch__thing", {})],
        )

        assert isinstance(results[0], ExecutedTool)
        assert isinstance(results[1], ToolError)

    async def test_invalid_params_never_reach_the_network(self, session: AsyncSession) -> None:
        """Schema validation stays in the serial preflight, so a bad call is rejected before a
        request is made — the same ordering `execute` has."""
        tenant, investigation_id = await _setup(session, "many-badparams")
        tool = _Slow(delay=0.0)

        results = await _executor(tool).execute_many(
            session,
            tenant,
            investigation_id=investigation_id,
            calls=[("slow__fetch", {"invented": True})],
        )

        assert isinstance(results[0], ToolError)
        assert tool.invocations == [], "an invalid call reached the connector"


class TestTheInvariantsThatMustNotChange:
    async def test_every_success_writes_exactly_one_evidence_row(
        self, session: AsyncSession
    ) -> None:
        """Concurrent savepoints on one session would interleave. The record phase is serial
        precisely so this count is exact."""
        tenant, investigation_id = await _setup(session, "many-evidence")
        tool = _Slow(delay=0.05)

        results = await _executor(tool).execute_many(
            session,
            tenant,
            investigation_id=investigation_id,
            calls=_calls("a", "b", "c", "d", "e"),
        )
        await session.flush()

        count = await session.scalar(
            select(func.count())
            .select_from(Evidence)
            .where(Evidence.investigation_id == investigation_id)
        )
        assert count == 5
        ids = {r.evidence_id for r in results if isinstance(r, ExecutedTool)}
        assert len(ids) == 5, "evidence ids were not distinct"

    async def test_every_call_writes_an_audit_row_including_failures(
        self, session: AsyncSession
    ) -> None:
        """A failed or empty call produces no evidence and must still be auditable — the reason
        `tool_calls` is a separate table."""
        tenant, investigation_id = await _setup(session, "many-audit")
        tool = _Slow(delay=0.0, fail_on={"bad"})

        await _executor(tool).execute_many(
            session,
            tenant,
            investigation_id=investigation_id,
            calls=_calls("good", "bad"),
        )
        await session.flush()

        rows = (
            (
                await session.execute(
                    select(ToolCall).where(ToolCall.investigation_id == investigation_id)
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 2
        assert {row.succeeded for row in rows} == {True, False}

    async def test_a_foreign_investigation_is_refused_for_the_whole_batch(
        self, session: AsyncSession
    ) -> None:
        """F-01. The ownership check moved from per-call to once-per-batch, which is the same
        query on the same investigation — but if it had been dropped, evidence could be written
        onto another tenant's investigation and become citable by their report."""
        mine = await _setup(session, "many-mine")
        theirs_tenant, theirs_investigation = await _setup(session, "many-theirs")
        tool = _Slow(delay=0.0)

        results = await _executor(tool).execute_many(
            session,
            mine[0],
            # Another tenant's investigation id, supplied by the caller.
            investigation_id=theirs_investigation,
            calls=_calls("a", "b"),
        )

        assert all(isinstance(r, ToolError) for r in results)
        assert tool.invocations == [], "a foreign investigation reached the network"
        # Nothing was written onto the victim's investigation.
        count = await session.scalar(
            select(func.count())
            .select_from(Evidence)
            .where(Evidence.investigation_id == theirs_investigation)
        )
        assert count == 0

    async def test_the_rate_limiter_is_consulted_per_call(self, session: AsyncSession) -> None:
        """Serially, in preflight, so a batch cannot slip past a per-tenant ceiling by asking
        several questions at once. A limiter consulted once per *batch* would let eight calls
        through on one decision."""
        tenant, investigation_id = await _setup(session, "many-limiter")

        class _CountingLimiter:
            def __init__(self) -> None:
                self.checks = 0

            async def check(
                self, tenant_id: uuid.UUID, provider: CredentialProvider | None
            ) -> Decision:
                self.checks += 1
                # Allows the first two, refuses the rest.
                allowed = self.checks <= 2
                # `reason` is a derived property, not a field -- it is composed from the
                # window and limit, so those are what a refusal has to carry.
                return Decision(
                    allowed=allowed,
                    limited=True,
                    window_seconds=None if allowed else 60,
                    limit=None if allowed else 2,
                    retry_after_seconds=None if allowed else 30.0,
                )

        limiter = _CountingLimiter()
        tool = _Slow(delay=0.0)
        results = await _executor(tool, limiter).execute_many(  # type: ignore[arg-type]
            session,
            tenant,
            investigation_id=investigation_id,
            calls=_calls("a", "b", "c", "d"),
        )

        assert limiter.checks == 4, "the limiter was not asked about every call"
        assert sum(isinstance(r, ExecutedTool) for r in results) == 2
        refused = [r for r in results if isinstance(r, RateLimited)]
        assert len(refused) == 2
        # The refused calls never reached the connector.
        assert len(tool.invocations) == 2

    async def test_empty_results_are_still_labelled_empty(self, session: AsyncSession) -> None:
        """The `result_key` mechanism (F-24) must survive the batching. An empty observation
        that reached the analyst unlabelled is the bug that shipped four times."""
        tenant, investigation_id = await _setup(session, "many-empty-result")

        class _Nothing(_Slow):
            name = "nothing"

            async def fetch(self, ctx: ToolContext, *, tag: str = "") -> ToolResult:
                self.invocations.append(tag)
                return ToolResult(payload={"rows": []}, source_ref="nothing://")

        tool = _Nothing(delay=0.0)
        registry = ToolRegistry()
        registry.register(tool)
        results = await ToolExecutor(registry).execute_many(
            session,
            tenant,
            investigation_id=investigation_id,
            calls=[("nothing__fetch", {"tag": "x"})],
        )

        assert isinstance(results[0], ExecutedTool)
        assert results[0].is_empty is True

    async def test_the_payload_hash_matches_what_was_stored(self, session: AsyncSession) -> None:
        """The gate re-hashes each payload and compares. If batching wrote a payload that did
        not match its hash, every claim citing it would be silently deleted."""
        tenant, investigation_id = await _setup(session, "many-hash")
        tool = _Slow(delay=0.0)

        results = await _executor(tool).execute_many(
            session,
            tenant,
            investigation_id=investigation_id,
            calls=_calls("a", "b"),
        )
        await session.flush()

        for result in results:
            assert isinstance(result, ExecutedTool)
            row = (
                await session.execute(select(Evidence).where(Evidence.id == result.evidence_id))
            ).scalar_one()
            assert row.payload_hash == result.payload_hash
            assert row.payload == result.payload


class _Varying(Tool):
    """A connector whose calls take deliberately different amounts of time.

    `_Slow` sleeps for a fixed interval, so it cannot distinguish a per-call duration from a
    per-batch one -- every row would carry the same number either way.
    """

    name = "varying"
    provider = None

    def __init__(self, delays: dict[str, float]) -> None:
        self.delays = delays
        super().__init__()

    def capabilities(self) -> list[Capability]:
        return [
            Capability(
                name="fetch",
                description="Sleep for this tag's own interval, then return rows.",
                params_schema=SCHEMA,
                handler=self.fetch,
                result_key="rows",
            )
        ]

    async def fetch(self, ctx: ToolContext, *, tag: str = "") -> ToolResult:
        await asyncio.sleep(self.delays.get(tag, 0.0))
        return ToolResult(payload={"rows": [{"tag": tag}]}, source_ref=f"varying://{tag}")


class TestEachCallIsTimedSeparately:
    """The batch's elapsed time used to be stamped on every row in it.

    On a real investigation that produced three parallel calls each recorded at 8,003 ms, when
    8,003 ms was the total for all three. It makes the slowest call in a batch
    indistinguishable from the fastest, which is the one thing a latency trace exists to show --
    and it is the audit row, not just the display, so the record was wrong too.
    """

    async def test_a_fast_call_is_not_charged_for_a_slow_sibling(
        self, session: AsyncSession
    ) -> None:
        tenant, investigation_id = await _setup(session, "many-timing")
        tool = _Varying({"quick": 0.0, "slow": 0.30})

        results = await _executor(tool).execute_many(
            session,
            tenant,
            investigation_id=investigation_id,
            calls=[("varying__fetch", {"tag": "quick"}), ("varying__fetch", {"tag": "slow"})],
        )
        quick, slow = results
        assert isinstance(quick, ExecutedTool) and isinstance(slow, ExecutedTool)

        assert slow.duration_ms >= 280, slow.duration_ms
        # The batch took at least 300ms; the quick call must not be charged for it.
        assert quick.duration_ms < 150, quick.duration_ms

    async def test_the_audit_row_carries_the_call_duration(self, session: AsyncSession) -> None:
        """Not only the returned object. The trace on the report page reads the row."""
        tenant, investigation_id = await _setup(session, "many-timing-row")
        tool = _Varying({"quick": 0.0, "slow": 0.30})

        await _executor(tool).execute_many(
            session,
            tenant,
            investigation_id=investigation_id,
            calls=[("varying__fetch", {"tag": "quick"}), ("varying__fetch", {"tag": "slow"})],
        )
        await session.flush()

        rows = (
            (
                await session.execute(
                    select(ToolCall)
                    .where(ToolCall.investigation_id == investigation_id)
                    .order_by(ToolCall.created_at)
                )
            )
            .scalars()
            .all()
        )
        durations = sorted(row.duration_ms or 0 for row in rows)
        assert len(durations) == 2
        assert durations[0] < 150 and durations[1] >= 280
        assert durations[0] != durations[1], "the batch duration was stamped on both rows"

    async def test_a_call_that_never_ran_falls_back_to_the_batch(
        self, session: AsyncSession
    ) -> None:
        """A call rejected in preflight never reached the network, so it has no duration of its
        own. It still gets an audit row, and 0 there would read as "instant" rather than "never
        ran" -- so it falls back to the batch's elapsed time."""
        tenant, investigation_id = await _setup(session, "many-timing-preflight")
        tool = _Varying({"slow": 0.20})

        results = await _executor(tool).execute_many(
            session,
            tenant,
            investigation_id=investigation_id,
            # The second call violates the schema, so it is rejected before the network and is
            # returned as an error rather than as an ExecutedTool.
            calls=[("varying__fetch", {"tag": "slow"}), ("varying__fetch", {"tag": 12345})],
        )
        assert isinstance(results[0], ExecutedTool)
        assert isinstance(results[1], InvalidParams)
        await session.flush()

        rejected = (
            (
                await session.execute(
                    select(ToolCall).where(
                        ToolCall.investigation_id == investigation_id,
                        ToolCall.succeeded.is_(False),
                    )
                )
            )
            .scalars()
            .one()
        )
        # Not zero, and not silently absent: the row says the batch took this long, which is
        # the honest thing to say about a call that never started.
        assert (rejected.duration_ms or 0) >= 180
