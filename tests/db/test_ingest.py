"""The nightly ingest.

What is worth testing here is not that data moves — it is the failure behaviour, because
every interesting bug in a sync is silent. A watermark that advances past a window that was
never written loses data and leaves nothing to point at it. A metric point written twice
doubles a baseline. A stream that fails and takes its siblings down loses the data that did
arrive.

So these tests are mostly about what happens when something goes wrong, and they use a
stub syncer: the connectors are already covered against recorded payloads in `tests/tools`,
and what needs proving here is the runner's contract with any syncer at all.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import (
    Credential,
    CredentialProvider,
    MetricPoint,
    SyncState,
    SyncStatus,
    Tenant,
)
from cortex.ingest.base import StreamResult, Syncer
from cortex.ingest.dispatch import SYNCERS, nightly_targets, syncer_for
from cortex.ingest.metrics import Point, write_points
from cortex.ingest.runner import IngestRunner
from cortex.ingest.state import (
    FIRST_SYNC_WINDOW,
    OVERLAP,
    STALE_AFTER,
    Window,
    is_stale,
    read_state,
    record_failure,
    record_success,
    window_for,
)
from cortex.memory.entities import Edge, Node, NodeLabel, RelType
from cortex.memory.graph_store import GraphStore
from cortex.memory.vector_store import Document
from cortex.security.vault import encrypt_credential
from cortex.tenancy.context import TenantContext

NOW = datetime(2026, 7, 30, 2, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- stubs


class _FakeGraph(GraphStore):
    """Records what it was asked to write, and in what order.

    Order matters: an edge whose endpoints do not exist yet is silently dropped by the
    MERGE, which loses the relationship while keeping the entities — a graph that looks
    populated and cannot be walked.
    """

    def __init__(self, fail_on_edges: bool = False) -> None:
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self.calls: list[str] = []
        self._fail_on_edges = fail_on_edges

    async def provision(self, ctx: TenantContext) -> None: ...
    async def drop(self, ctx: TenantContext) -> None: ...

    async def upsert_nodes(self, ctx: TenantContext, nodes) -> int:  # type: ignore[no-untyped-def]
        self.calls.append("nodes")
        self.nodes.extend(nodes)
        return len(nodes)

    async def upsert_edges(self, ctx: TenantContext, edges) -> int:  # type: ignore[no-untyped-def]
        self.calls.append("edges")
        if self._fail_on_edges:
            raise RuntimeError("falkordb unavailable")
        self.edges.extend(edges)
        return len(edges)

    async def get_node(self, ctx, label, key):  # type: ignore[no-untyped-def]
        return None

    async def neighbors(self, ctx, label, key, **kwargs):  # type: ignore[no-untyped-def]
        return []

    async def find_paths(self, ctx, *args, **kwargs):  # type: ignore[no-untyped-def]
        return []

    async def nodes_by_label(self, ctx, label, **kwargs):  # type: ignore[no-untyped-def]
        return [n.props | {"key": n.key} for n in self.nodes if n.label == label]

    async def stats(self, ctx: TenantContext) -> dict[str, int]:
        return {"nodes": len(self.nodes), "relationships": len(self.edges)}


class _FakeVectors:
    def __init__(self, error: Exception | None = None) -> None:
        self.documents: list[Document] = []
        self._error = error

    async def provision(self, ctx: TenantContext) -> None: ...
    async def drop(self, ctx: TenantContext) -> None: ...

    async def upsert(self, ctx: TenantContext, documents) -> int:  # type: ignore[no-untyped-def]
        if self._error is not None:
            raise self._error
        self.documents.extend(documents)
        return len(documents)

    async def search(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return []

    async def stats(self, ctx: TenantContext) -> dict[str, int]:
        return {}


class _StubSyncer(Syncer):
    """A syncer whose behaviour per stream a test dictates."""

    provider = CredentialProvider.GITHUB
    tool_name = "github"

    def __init__(self, results: dict[str, StreamResult | Exception]) -> None:
        self._results = results
        self.windows: dict[str, Window] = {}

    @property
    def streams(self) -> tuple[str, ...]:
        return tuple(self._results)

    async def sync_stream(  # type: ignore[override]
        self, session, tenant, ctx, *, stream, window, graph
    ):
        self.windows[stream] = window
        outcome = self._results[stream]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


# ------------------------------------------------------------------------ fixtures


async def _tenant(session: AsyncSession, slug: str = "ingest-test") -> TenantContext:
    tenant_id = uuid.uuid4()
    graph_name = f"cortex_g_{slug}_{tenant_id.hex}"
    session.add(Tenant(id=tenant_id, slug=slug, name=slug, graph_name=graph_name))
    await session.flush()
    return TenantContext(tenant_id=tenant_id, tenant_slug=slug, graph_name=graph_name)


async def _connect(session: AsyncSession, tenant: TenantContext, **metadata: str) -> None:
    wrapped, ciphertext = encrypt_credential(
        tenant.tenant_id, CredentialProvider.GITHUB.value, "ghp_fake", label="default"
    )
    session.add(
        Credential(
            tenant_id=tenant.tenant_id,
            provider=CredentialProvider.GITHUB,
            label="default",
            wrapped_data_key=wrapped,
            ciphertext=ciphertext,
            metadata_=dict(metadata),
        )
    )
    await session.flush()


# -------------------------------------------------------------------------- windows


class TestTheWindow:
    def test_a_first_sync_pulls_a_bounded_history(self) -> None:
        """Enough to establish a baseline, bounded so a first sync against a busy
        repository is not an hour of API calls."""
        window = window_for(None, now=NOW)
        assert window.first_sync is True
        assert window.since == NOW - FIRST_SYNC_WINDOW

    def test_an_incremental_sync_re_reads_an_overlap(self) -> None:
        """Upstreams backfill late-arriving data, so a window starting exactly at the
        previous watermark misses anything that landed behind it."""
        state = SyncState(watermark=NOW - timedelta(days=1))
        window = window_for(state, now=NOW)
        assert window.first_sync is False
        assert window.since == NOW - timedelta(days=1) - OVERLAP

    def test_dates_are_rendered_as_the_connectors_expect(self) -> None:
        window = window_for(None, now=NOW)
        assert window.until_date == "2026-07-30"


# ------------------------------------------------------------------------ watermarks


class TestWatermarks:
    async def test_a_watermark_advances_only_on_success(self, session: AsyncSession) -> None:
        """The defect this prevents is silent: a cursor that moves past a window nothing
        wrote leaves a hole with nothing to point at it."""
        tenant = await _tenant(session)
        await record_success(
            session,
            tenant,
            provider=CredentialProvider.GITHUB,
            label="default",
            stream="commits",
            watermark=NOW,
            items_written=5,
            now=NOW,
        )
        await record_failure(
            session,
            tenant,
            provider=CredentialProvider.GITHUB,
            label="default",
            stream="commits",
            detail="upstream 500",
            now=NOW + timedelta(days=1),
        )

        state = await read_state(
            session,
            tenant,
            provider=CredentialProvider.GITHUB,
            label="default",
            stream="commits",
        )
        assert state is not None
        assert state.watermark == NOW, "a failure must not advance the cursor"
        assert state.status is SyncStatus.FAILED
        assert state.consecutive_failures == 1

    async def test_a_watermark_never_moves_backwards(self, session: AsyncSession) -> None:
        """A retry handed an older window than a run that already succeeded must not
        rewind the cursor and cause the next run to re-read weeks."""
        tenant = await _tenant(session)
        for watermark in (NOW, NOW - timedelta(days=7)):
            await record_success(
                session,
                tenant,
                provider=CredentialProvider.GITHUB,
                label="default",
                stream="commits",
                watermark=watermark,
                items_written=1,
                now=NOW,
            )
        state = await read_state(
            session,
            tenant,
            provider=CredentialProvider.GITHUB,
            label="default",
            stream="commits",
        )
        assert state is not None and state.watermark == NOW

    async def test_a_success_clears_the_failure_count(self, session: AsyncSession) -> None:
        tenant = await _tenant(session)
        for _ in range(3):
            await record_failure(
                session,
                tenant,
                provider=CredentialProvider.GITHUB,
                label="default",
                stream="commits",
                detail="flaky",
                now=NOW,
            )
        state = await record_success(
            session,
            tenant,
            provider=CredentialProvider.GITHUB,
            label="default",
            stream="commits",
            watermark=NOW,
            items_written=2,
            now=NOW,
        )
        assert state.consecutive_failures == 0
        assert state.status is SyncStatus.OK

    async def test_a_partial_pull_advances_and_records_the_gap(self, session: AsyncSession) -> None:
        """The data that arrived is real, so refusing to advance would re-read it every
        night forever; pretending it was complete would let a report imply a whole
        history."""
        tenant = await _tenant(session)
        state = await record_success(
            session,
            tenant,
            provider=CredentialProvider.GITHUB,
            label="default",
            stream="commits",
            watermark=NOW,
            items_written=100,
            now=NOW,
            detail="capped at 100",
        )
        assert state.watermark == NOW
        assert state.status is SyncStatus.PARTIAL
        assert state.detail == "capped at 100"

    async def test_streams_carry_independent_watermarks(self, session: AsyncSession) -> None:
        """With one cursor per provider, a failed issues pull would rewind commits, or a
        successful commits pull would mark issues current when it never ran."""
        tenant = await _tenant(session)
        await record_success(
            session,
            tenant,
            provider=CredentialProvider.GITHUB,
            label="default",
            stream="commits",
            watermark=NOW,
            items_written=1,
            now=NOW,
        )
        issues = await read_state(
            session,
            tenant,
            provider=CredentialProvider.GITHUB,
            label="default",
            stream="issues",
        )
        assert issues is None, "one stream's success must say nothing about another's"


class TestStaleness:
    def test_staleness_is_measured_from_the_last_success(self) -> None:
        """A connector failing every night for a week still has a recent `last_run_at`.
        Reporting that as health would be a lie of omission."""
        state = SyncState(
            last_run_at=NOW,
            last_success_at=NOW - STALE_AFTER - timedelta(hours=1),
        )
        assert is_stale(state, now=NOW) is True

    def test_a_never_successful_stream_is_stale(self) -> None:
        assert is_stale(SyncState(last_run_at=NOW, last_success_at=None), now=NOW) is True

    def test_a_recent_success_is_not_stale(self) -> None:
        assert is_stale(SyncState(last_success_at=NOW - timedelta(hours=1)), now=NOW) is False


# ---------------------------------------------------------------------- metric points


class TestMetricPoints:
    async def test_re_reading_the_same_window_does_not_duplicate(
        self, session: AsyncSession
    ) -> None:
        """The overlap means the same observation arrives more than once by design. If
        that duplicated rows, every baseline computed from the series would be wrong."""
        tenant = await _tenant(session)
        points = [
            Point(metric="signups", observed_at=NOW, value=100.0, segment={"project": "saas"})
        ]
        await write_points(session, tenant, provider=CredentialProvider.POSTHOG, points=points)
        await write_points(session, tenant, provider=CredentialProvider.POSTHOG, points=points)

        total = await session.scalar(select(func.count()).select_from(MetricPoint))
        assert total == 1

    async def test_a_revised_value_replaces_the_earlier_read(self, session: AsyncSession) -> None:
        """An analytics upstream revises a figure as late events land. Keeping the first
        read would freeze a number the source itself no longer agrees with."""
        tenant = await _tenant(session)
        await write_points(
            session,
            tenant,
            provider=CredentialProvider.POSTHOG,
            points=[Point(metric="signups", observed_at=NOW, value=100.0)],
        )
        await write_points(
            session,
            tenant,
            provider=CredentialProvider.POSTHOG,
            points=[Point(metric="signups", observed_at=NOW, value=140.0)],
        )
        value = await session.scalar(select(MetricPoint.value))
        assert value == 140.0

    async def test_segments_are_distinct_series(self, session: AsyncSession) -> None:
        tenant = await _tenant(session)
        await write_points(
            session,
            tenant,
            provider=CredentialProvider.POSTHOG,
            points=[
                Point(metric="signups", observed_at=NOW, value=10.0, segment={"project": "a"}),
                Point(metric="signups", observed_at=NOW, value=20.0, segment={"project": "b"}),
            ],
        )
        total = await session.scalar(select(func.count()).select_from(MetricPoint))
        assert total == 2

    def test_a_segment_key_is_order_independent(self) -> None:
        """Unsorted, the same segment could be inserted twice under two spellings and a
        total over the series would double-count it."""
        first = Point(metric="m", observed_at=NOW, value=1, segment={"a": "1", "b": "2"})
        second = Point(metric="m", observed_at=NOW, value=1, segment={"b": "2", "a": "1"})
        assert first.segment_key == second.segment_key

    async def test_no_points_is_not_an_error(self, session: AsyncSession) -> None:
        tenant = await _tenant(session)
        assert (
            await write_points(session, tenant, provider=CredentialProvider.POSTHOG, points=[]) == 0
        )


# ---------------------------------------------------------------------------- runner


class TestTheRunner:
    async def test_it_writes_nodes_before_edges(self, session: AsyncSession) -> None:
        """An edge whose endpoints do not exist is silently dropped by the MERGE, leaving
        a graph that looks populated but cannot be walked."""
        tenant = await _tenant(session)
        await _connect(session, tenant)
        graph = _FakeGraph()
        result = StreamResult(
            nodes=[Node(NodeLabel.PR, "913", {})],
            edges=[Edge(RelType.AUTHORED, NodeLabel.PERSON, "dana", NodeLabel.PR, "913")],
        )

        await IngestRunner(graph).run(session, tenant, _StubSyncer({"commits": result}), now=NOW)
        assert graph.calls == ["nodes", "edges"]

    async def test_one_failing_stream_does_not_fail_its_siblings(
        self, session: AsyncSession
    ) -> None:
        """ "GitHub is down" and "GitHub's issues endpoint is down" call for different
        responses, and collapsing them throws away the commits that did arrive."""
        tenant = await _tenant(session)
        await _connect(session, tenant)
        graph = _FakeGraph()
        syncer = _StubSyncer(
            {
                "commits": StreamResult(nodes=[Node(NodeLabel.PR, "913", {})]),
                "issues": RuntimeError("upstream 503"),
            }
        )

        outcome = await IngestRunner(graph).run(session, tenant, syncer, now=NOW)

        assert outcome.succeeded is False
        by_stream = {s.stream: s for s in outcome.streams}
        assert by_stream["commits"].succeeded is True
        assert by_stream["issues"].succeeded is False
        assert "upstream 503" in (by_stream["issues"].detail or "")
        # The commits data survived.
        assert len(graph.nodes) == 1

    async def test_a_failed_stream_records_its_reason(self, session: AsyncSession) -> None:
        tenant = await _tenant(session)
        await _connect(session, tenant)
        await IngestRunner(_FakeGraph()).run(
            session,
            tenant,
            _StubSyncer({"commits": RuntimeError("token expired")}),
            now=NOW,
        )
        state = await read_state(
            session,
            tenant,
            provider=CredentialProvider.GITHUB,
            label="default",
            stream="commits",
        )
        assert state is not None
        assert state.status is SyncStatus.FAILED
        assert "token expired" in (state.detail or "")
        assert state.watermark is None

    async def test_a_graph_write_failure_is_recorded_not_raised(
        self, session: AsyncSession
    ) -> None:
        """A write failure must leave the watermark alone, or the next run skips a window
        that was never written."""
        tenant = await _tenant(session)
        await _connect(session, tenant)
        graph = _FakeGraph(fail_on_edges=True)
        result = StreamResult(
            nodes=[Node(NodeLabel.PR, "913", {})],
            edges=[Edge(RelType.AUTHORED, NodeLabel.PERSON, "dana", NodeLabel.PR, "913")],
        )

        outcome = await IngestRunner(graph).run(
            session, tenant, _StubSyncer({"commits": result}), now=NOW
        )

        assert outcome.succeeded is False
        assert "write failed" in (outcome.streams[0].detail or "")
        state = await read_state(
            session,
            tenant,
            provider=CredentialProvider.GITHUB,
            label="default",
            stream="commits",
        )
        assert state is not None and state.watermark is None

    async def test_a_vector_store_outage_does_not_lose_the_stream(
        self, session: AsyncSession
    ) -> None:
        """Found by running it: the first real ingest lost a whole sync to a transport
        error from a managed Qdrant cluster, after the graph writes had succeeded.
        Semantic memory is additive; the graph is the core."""
        tenant = await _tenant(session)
        await _connect(session, tenant)
        graph = _FakeGraph()
        vectors = _FakeVectors(error=RuntimeError("ResponseHandlingException"))
        result = StreamResult(
            nodes=[Node(NodeLabel.PR, "913", {})],
            documents=[Document(kind="docs", source_id="x", text="a commit")],
        )

        outcome = await IngestRunner(graph, vectors).run(  # type: ignore[arg-type]
            session, tenant, _StubSyncer({"commits": result}), now=NOW
        )

        assert outcome.succeeded is True
        assert "documents not stored" in (outcome.streams[0].detail or "")
        assert len(graph.nodes) == 1
        state = await read_state(
            session,
            tenant,
            provider=CredentialProvider.GITHUB,
            label="default",
            stream="commits",
        )
        # PARTIAL, not OK: the data is usable and the gap is disclosed.
        assert state is not None and state.status is SyncStatus.PARTIAL

    async def test_a_partial_pull_is_carried_through_to_the_state(
        self, session: AsyncSession
    ) -> None:
        tenant = await _tenant(session)
        await _connect(session, tenant)
        result = StreamResult(nodes=[Node(NodeLabel.PR, "1", {})], partial="capped at 100")
        await IngestRunner(_FakeGraph()).run(
            session, tenant, _StubSyncer({"commits": result}), now=NOW
        )
        state = await read_state(
            session,
            tenant,
            provider=CredentialProvider.GITHUB,
            label="default",
            stream="commits",
        )
        assert state is not None
        assert state.status is SyncStatus.PARTIAL
        assert "capped at 100" in (state.detail or "")

    async def test_an_explicit_window_overrides_the_watermark(self, session: AsyncSession) -> None:
        """A backfill must be requestable without hand-editing a cursor."""
        tenant = await _tenant(session)
        await _connect(session, tenant)
        syncer = _StubSyncer({"commits": StreamResult()})
        override = Window(since=datetime(2026, 1, 1, tzinfo=UTC), until=NOW, first_sync=True)

        await IngestRunner(_FakeGraph()).run(session, tenant, syncer, now=NOW, window=override)
        assert syncer.windows["commits"].since == datetime(2026, 1, 1, tzinfo=UTC)

    async def test_a_missing_credential_fails_before_any_upstream_call(
        self, session: AsyncSession
    ) -> None:
        """A tenant that never connected GitHub should produce a clear failure rather than
        a sync that quietly writes nothing."""
        from cortex.tools.base import CredentialMissing

        tenant = await _tenant(session)
        with pytest.raises(CredentialMissing):
            await IngestRunner(_FakeGraph()).run(
                session, tenant, _StubSyncer({"commits": StreamResult()}), now=NOW
            )


# -------------------------------------------------------------------------- dispatch


class TestDispatch:
    async def test_only_connected_providers_with_a_syncer_are_dispatched(
        self, session: AsyncSession
    ) -> None:
        """A provider nobody connected should produce no message, rather than one that
        fails with "not connected" every night."""
        tenant = await _tenant(session)
        await _connect(session, tenant)

        targets = await nightly_targets(session)
        assert [t.provider for t in targets] == [CredentialProvider.GITHUB]
        assert targets[0].tenant.graph_name == tenant.graph_name

    async def test_a_tenant_with_no_credentials_produces_no_work(
        self, session: AsyncSession
    ) -> None:
        await _tenant(session)
        assert await nightly_targets(session) == []

    def test_a_provider_without_a_syncer_says_so(self) -> None:
        """GA4, HubSpot and Slack are read live during an investigation. The error names
        what does have a syncer, so the absence is a fact rather than a mystery."""
        with pytest.raises(KeyError, match="no ingest syncer"):
            syncer_for(CredentialProvider.HUBSPOT)

    def test_every_registered_syncer_declares_matching_metadata(self) -> None:
        """A syncer whose provider does not match its registry key would load the wrong
        tenant's credential for the right-looking provider."""
        for provider, factory in SYNCERS.items():
            syncer = factory()
            assert syncer.provider is provider
            assert syncer.streams, provider
            assert syncer.tool_name


# ---------------------------------------------------------------------------- health


class TestHealthDisclosure:
    """A stale connector must reach the report, because the analyst cannot know it is
    stale. This is the plan's requirement, enforced in code rather than in a prompt."""

    async def test_a_healthy_sync_says_nothing(self, session: AsyncSession) -> None:
        """A report that discloses every connector's status on every question buries the
        one disclosure that matters."""
        from cortex.ingest.health import health_notes

        tenant = await _tenant(session)
        await record_success(
            session,
            tenant,
            provider=CredentialProvider.GITHUB,
            label="default",
            stream="commits",
            watermark=NOW,
            items_written=10,
            now=NOW,
        )
        assert await health_notes(session, tenant, now=NOW) == []

    async def test_a_failing_sync_says_absence_is_not_evidence(self, session: AsyncSession) -> None:
        """The exact failure mode this exists for: an investigation reading three-day-old
        data would honestly report "no deploys that week"."""
        from cortex.ingest.health import health_notes

        tenant = await _tenant(session)
        await record_failure(
            session,
            tenant,
            provider=CredentialProvider.GITHUB,
            label="default",
            stream="deployments",
            detail="401 bad credentials",
            now=NOW,
        )
        notes = await health_notes(session, tenant, now=NOW)
        assert len(notes) == 1
        assert "failing" in notes[0].note
        assert "401 bad credentials" in notes[0].note
        assert "not evidence that nothing happened" in notes[0].note

    async def test_a_stale_sync_reports_its_age(self, session: AsyncSession) -> None:
        from cortex.ingest.health import health_notes

        tenant = await _tenant(session)
        await record_success(
            session,
            tenant,
            provider=CredentialProvider.GITHUB,
            label="default",
            stream="commits",
            watermark=NOW - timedelta(days=4),
            items_written=1,
            now=NOW - timedelta(days=4),
        )
        notes = await health_notes(session, tenant, now=NOW)
        assert len(notes) == 1
        assert "4 days ago" in notes[0].note

    async def test_a_partial_sync_is_disclosed_as_a_floor(self, session: AsyncSession) -> None:
        """Current but truncated. A count drawn from it is a floor, not a total, and a
        report implying otherwise would overstate what the data supports."""
        from cortex.ingest.health import health_notes

        tenant = await _tenant(session)
        await record_success(
            session,
            tenant,
            provider=CredentialProvider.GITHUB,
            label="default",
            stream="commits",
            watermark=NOW,
            items_written=100,
            now=NOW,
            detail="capped at 100",
        )
        notes = await health_notes(session, tenant, now=NOW)
        assert "incomplete" in notes[0].note
        assert "floor rather than a total" in notes[0].note

    async def test_health_notes_do_not_cross_tenants(self, session: AsyncSession) -> None:
        from cortex.ingest.health import health_notes

        first = await _tenant(session, slug="health-a")
        second = await _tenant(session, slug="health-b")
        await record_failure(
            session,
            first,
            provider=CredentialProvider.GITHUB,
            label="default",
            stream="commits",
            detail="broken",
            now=NOW,
        )
        assert await health_notes(session, second, now=NOW) == []

    async def test_a_real_emptiness_is_not_reported_as_incomplete_data(
        self, session: AsyncSession
    ) -> None:
        """All five real PostHog projects have zero annotations. Calling that "incomplete —
        treat counts as a floor" tells a reader their change log might be missing entries
        when the truth is that nobody writes any, and a disclosure that cries wolf stops
        being read."""
        from cortex.ingest.health import health_notes

        tenant = await _tenant(session)
        await _connect(session, tenant)
        result = StreamResult(note="saas: no annotations; oss: no annotations")

        outcome = await IngestRunner(_FakeGraph()).run(
            session, tenant, _StubSyncer({"annotations": result}), now=NOW
        )

        assert outcome.succeeded is True
        state = await read_state(
            session,
            tenant,
            provider=CredentialProvider.GITHUB,
            label="default",
            stream="annotations",
        )
        assert state is not None
        assert state.status is SyncStatus.OK, "a real emptiness is not a gap in collection"
        # Still recorded, so the fact is available for diagnosis.
        assert "no annotations" in (state.detail or "")
        # And it produces no disclosure, because there is nothing to disclose.
        assert await health_notes(session, tenant, now=NOW) == []

    async def test_a_vendors_internals_do_not_reach_a_report(self, session: AsyncSession) -> None:
        """A real run put Voyage's rate-limit body — including a link to its billing
        dashboard and an explanation of its free-tier token allowance — on course for a
        customer-facing report about someone's own funnel."""
        from cortex.ingest.health import health_notes

        tenant = await _tenant(session)
        await record_success(
            session,
            tenant,
            provider=CredentialProvider.GITHUB,
            label="default",
            stream="issues",
            watermark=NOW,
            items_written=1,
            now=NOW,
            detail=(
                'documents not embedded: voyage returned 429: {"detail":"You have not yet '
                "added your payment method in the billing page and will have reduced rate "
                'limits of 3 RPM. See https://dashboard.voyageai.com/ for details."}'
            ),
        )
        note = (await health_notes(session, tenant, now=NOW))[0].note

        assert "voyage returned 429" in note
        assert "dashboard.voyageai.com" not in note
        assert "payment method" not in note
        # The full text stays in sync_state, where diagnosis needs it.
        state = await read_state(
            session,
            tenant,
            provider=CredentialProvider.GITHUB,
            label="default",
            stream="issues",
        )
        assert state is not None and "billing page" in (state.detail or "")


class TestAnOperatorIsToldWhyMemoryIsIncomplete:
    """Both halves of one afternoon's real syncs, where the notes said nothing usable.

    The `documents not stored` note existed and was empty of reason:
    `ResponseHandlingException: `, twice, from a managed Qdrant cluster. And a sync whose
    vector store was unreachable at *provision* time skipped the write block entirely, so every
    stream reported `ok` with `docs=0` and `--status` showed green while the tenant had no
    semantic memory at all.

    `IngestRunner`'s own contract is that "a stream that succeeded with a note is the
    interesting case — it means the data is real but incomplete, which is exactly what a report
    has to disclose and what a silent success would hide". These are the two ways it was
    hiding one.
    """

    def test_a_wrapper_with_no_message_still_names_its_cause(self) -> None:
        """qdrant-client's wrapper sets neither `__cause__` nor `__context__` — it takes the
        underlying exception as a constructor argument and keeps it on an attribute, so
        following only the standard chain finds nothing and prints the bare class name."""
        import httpx
        from qdrant_client.http.exceptions import ResponseHandlingException

        from cortex.ingest.runner import _describe

        described = _describe(
            ResponseHandlingException(source=httpx.ReadTimeout("timed out reaching the cluster"))
        )
        assert described == (
            "ResponseHandlingException <- ReadTimeout: timed out reaching the cluster"
        )

    def test_a_plain_exception_is_described_as_itself(self) -> None:
        from cortex.ingest.runner import _describe

        assert _describe(ValueError("collection width mismatch")) == (
            "ValueError: collection width mismatch"
        )

    def test_a_cause_with_no_message_is_still_named(self) -> None:
        """The floor: a class name with no message is the one thing this must not produce
        *twice over*. An empty inner message still leaves the reader the inner class."""
        import httpx
        from qdrant_client.http.exceptions import ResponseHandlingException

        from cortex.ingest.runner import _describe

        assert _describe(ResponseHandlingException(source=httpx.ConnectError(""))) == (
            "ResponseHandlingException <- ConnectError"
        )

    async def test_a_stream_whose_documents_were_never_offered_says_so(
        self, session: AsyncSession
    ) -> None:
        """Semantic memory off for the whole run, not failing per write.

        This is the silent one. The header line said "semantic memory unavailable" once, the
        streams said nothing, and `--status` — which reads the stream rows, not the console —
        reported `ok`.
        """
        tenant = await _tenant(session)
        await _connect(session, tenant)
        graph = _FakeGraph()
        result = StreamResult(
            nodes=[Node(NodeLabel.PR, "913", {})],
            documents=[
                Document(kind="docs", source_id="x", text="a commit"),
                Document(kind="docs", source_id="y", text="another commit"),
            ],
        )

        outcome = await IngestRunner(graph, None).run(  # type: ignore[arg-type]
            session, tenant, _StubSyncer({"commits": result}), now=NOW
        )

        assert outcome.succeeded is True
        detail = outcome.streams[0].detail or ""
        assert "2 document(s) not stored" in detail, detail
        assert "semantic memory was unavailable" in detail, detail
        # The graph is the core and must still be written.
        assert len(graph.nodes) == 1

    async def test_a_stream_with_no_documents_needs_no_note(self, session: AsyncSession) -> None:
        """A note that appears whenever semantic memory is off, on streams that produced
        nothing to store, is a note nobody reads on the sync that needed it."""
        tenant = await _tenant(session)
        await _connect(session, tenant)
        graph = _FakeGraph()
        result = StreamResult(nodes=[Node(NodeLabel.PR, "913", {})])

        outcome = await IngestRunner(graph, None).run(  # type: ignore[arg-type]
            session, tenant, _StubSyncer({"commits": result}), now=NOW
        )

        assert "not stored" not in (outcome.streams[0].detail or "")
