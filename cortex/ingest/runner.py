"""Running one connector's sync, and recording honestly what happened.

The runner owns every write. A syncer returns data; this decides what reaches the graph,
semantic memory and the metric series, and it records the watermark and health in the same
transaction as the data. That ordering is the whole design:

  - **Data first, watermark second, in one transaction.** If the watermark committed first
    and the write failed, the next run would skip a window that was never ingested — a
    silent hole with nothing to point at it. If the data committed and the watermark did
    not, the next run re-reads and writes the same rows again, which every write path here
    is built to absorb. One of those failure modes loses data; the other costs a round trip.
  - **A stream failing does not fail its siblings.** "GitHub is down" and "GitHub's issues
    endpoint is down" call for different responses, and collapsing them throws away the
    commits that did arrive.
  - **A partial pull advances the watermark and says so.** The data that arrived is real.
    Refusing to advance would re-read it every night forever; pretending it was complete
    would let a report imply a whole history. So it advances, and the gap is recorded where
    a report's data-quality section can read it.

Embedding is the one step allowed to be skipped without failing the stream. A tenant with
no Voyage key, or a provider rate-limiting us, still gets its graph and its metrics — and
the missing documents are reported as partial rather than losing the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import CredentialProvider
from cortex.ingest.base import Syncer
from cortex.ingest.metrics import write_points
from cortex.ingest.state import (
    Window,
    read_state,
    record_failure,
    record_success,
    window_for,
)
from cortex.memory.embeddings import EmbeddingError
from cortex.memory.graph_store import GraphStore
from cortex.memory.vector_store import VectorStore
from cortex.tenancy.context import TenantContext
from cortex.tools.executor import load_tool_context


@dataclass(slots=True)
class StreamOutcome:
    stream: str
    succeeded: bool
    items: int = 0
    nodes: int = 0
    edges: int = 0
    documents: int = 0
    points: int = 0
    detail: str | None = None
    watermark: datetime | None = None


@dataclass(slots=True)
class SyncOutcome:
    provider: str
    label: str
    streams: list[StreamOutcome] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        """True when every stream succeeded.

        Deliberately not "any stream succeeded": a caller deciding whether to alert needs
        to know that something is broken, and a provider with one dead stream is broken
        even though most of it worked.
        """
        return bool(self.streams) and all(s.succeeded for s in self.streams)

    @property
    def items_written(self) -> int:
        return sum(s.items for s in self.streams)

    def summary(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "label": self.label,
            "succeeded": self.succeeded,
            "items_written": self.items_written,
            "streams": {
                s.stream: {
                    "ok": s.succeeded,
                    "items": s.items,
                    "detail": s.detail,
                }
                for s in self.streams
            },
        }


def _describe(exc: BaseException) -> str:
    """An exception rendered so the note names the reason, not only the class.

    `ResponseHandlingException` is what qdrant-client raises when a request never got an
    answer, and its `str()` is empty -- so the note read `documents not stored:
    ResponseHandlingException: ` and told an operator nothing. Twice in one afternoon, on a
    managed cluster, with the real reason sitting one link down the cause chain.

    The chain is walked rather than just `repr`'d because that is where the reason lives: an
    `httpx.ReadTimeout` or `ConnectError` under the qdrant wrapper. A class name with no
    message is the one thing this must never produce.

    `.source` is walked alongside `__cause__` and `__context__` because qdrant-client's
    wrapper sets neither -- it takes the underlying exception as a constructor argument and
    stores it on an attribute, so `raise ... from ...` never happens and the standard chain
    is empty. Following only the standard links found nothing and printed the bare class name,
    which is the defect this function exists to fix.
    """
    parts: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).strip()
        parts.append(f"{type(current).__name__}: {message}" if message else type(current).__name__)
        nested = getattr(current, "source", None)
        current = (
            nested
            if isinstance(nested, BaseException)
            else current.__cause__ or current.__context__
        )
    return " <- ".join(parts[:3])


class IngestRunner:
    def __init__(
        self,
        graph: GraphStore,
        vectors: VectorStore | None = None,
    ) -> None:
        self._graph = graph
        self._vectors = vectors

    async def run(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        syncer: Syncer,
        *,
        label: str = "default",
        now: datetime | None = None,
        window: Window | None = None,
    ) -> SyncOutcome:
        """Sync every stream this syncer declares.

        `now` is injectable so a run is reproducible and a test does not depend on the
        clock; `window` overrides the watermark entirely, which is how a backfill is
        requested without rewriting the cursor by hand.
        """
        moment = now or datetime.now(UTC)
        outcome = SyncOutcome(provider=syncer.provider.value, label=label)

        # Resolved once for every stream: it decrypts a credential, and doing that per
        # stream would triple the exposure window for no benefit.
        ctx = await load_tool_context(session, tenant, _tool_stub(syncer), label)

        for stream in syncer.streams:
            outcome.streams.append(
                await self._run_stream(
                    session,
                    tenant,
                    syncer,
                    ctx=ctx,
                    stream=stream,
                    label=label,
                    now=moment,
                    override=window,
                )
            )
        return outcome

    async def _run_stream(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        syncer: Syncer,
        *,
        ctx: object,
        stream: str,
        label: str,
        now: datetime,
        override: Window | None,
    ) -> StreamOutcome:
        state = await read_state(
            session, tenant, provider=syncer.provider, label=label, stream=stream
        )
        window = override or window_for(state, now=now)

        try:
            result = await syncer.sync_stream(
                session,
                tenant,
                ctx,  # type: ignore[arg-type]
                stream=stream,
                window=window,
                graph=self._graph,
            )
        except Exception as exc:  # noqa: BLE001 - see below
            # Every failure is recorded rather than raised: one broken stream must not
            # abort the rest of the provider, and a stream that failed with its reason
            # recorded is diagnosable where a traceback in a worker log is not.
            await record_failure(
                session,
                tenant,
                provider=syncer.provider,
                label=label,
                stream=stream,
                detail=f"{type(exc).__name__}: {exc}",
                now=now,
            )
            return StreamOutcome(
                stream=stream, succeeded=False, detail=f"{type(exc).__name__}: {exc}"
            )

        # Kept apart: `partial` claims the data is incomplete, `note` records something
        # true about it that is not a gap. Only the first may set PARTIAL status.
        gaps: list[str] = [result.partial] if result.partial else []
        notes: list[str] = [result.note] if result.note else []
        nodes = edges = documents = points = 0

        try:
            if result.nodes:
                nodes = await self._graph.upsert_nodes(tenant, result.nodes)
            if result.edges:
                # After nodes, always: an edge whose endpoints do not exist yet is
                # silently dropped by the MERGE, which would lose the relationship
                # without losing the entities and leave a graph that looks populated but
                # cannot be walked.
                edges = await self._graph.upsert_edges(tenant, result.edges)

            if result.documents and self._vectors is None:
                # Recorded per stream, not only in the header line the caller printed once.
                # A sync whose vector store was unreachable at provision time skipped this
                # block entirely and every stream reported `ok` with `docs=0` -- so
                # `--status` showed green while the tenant had no semantic memory at all.
                # "A stream that succeeded with a note is the interesting case"; this one
                # succeeded with no note.
                gaps.append(
                    f"{len(result.documents)} document(s) not stored: semantic memory was "
                    "unavailable for this run"
                )
            elif result.documents and self._vectors is not None:
                try:
                    documents = await self._vectors.upsert(tenant, result.documents)
                except EmbeddingError as exc:
                    # Embedding is allowed to fail without failing the stream. The graph
                    # and the metrics are already correct, and Voyage's free tier
                    # rate-limits an ingest of any size — losing the whole run to that
                    # would leave a tenant with no memory rather than partial memory.
                    gaps.append(f"documents not embedded: {exc}")
                except Exception as exc:  # noqa: BLE001 - recorded, never silent
                    # A vector-store outage gets the same treatment, for the same reason
                    # and after seeing it happen: the first real ingest run lost an entire
                    # sync to a `ResponseHandlingException` from a managed Qdrant cluster
                    # that was briefly unreachable, after the graph writes had already
                    # succeeded. Semantic memory is additive; the graph is the core. The
                    # note is what keeps this from being silent — a report reading
                    # sync_state can say memory is incomplete.
                    gaps.append(f"documents not stored: {_describe(exc)}")

            if result.points:
                points = await write_points(
                    session, tenant, provider=syncer.provider, points=result.points
                )

            watermark = min(window.until, now)
            await record_success(
                session,
                tenant,
                provider=syncer.provider,
                label=label,
                stream=stream,
                watermark=watermark,
                items_written=result.item_count,
                now=now,
                detail="; ".join(gaps) if gaps else None,
                note="; ".join(notes) if notes else None,
            )
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            await record_failure(
                session,
                tenant,
                provider=syncer.provider,
                label=label,
                stream=stream,
                detail=f"write failed: {type(exc).__name__}: {exc}",
                now=now,
            )
            return StreamOutcome(
                stream=stream,
                succeeded=False,
                detail=f"write failed: {type(exc).__name__}: {exc}",
            )

        return StreamOutcome(
            stream=stream,
            succeeded=True,
            items=result.item_count,
            nodes=nodes,
            edges=edges,
            documents=documents,
            points=points,
            detail="; ".join(gaps + notes) or None,
            watermark=watermark,
        )


class _ToolStub:
    """The minimum `load_tool_context` needs: a name and a provider.

    A stub rather than the real tool, because the credential loader only reads those two
    fields and constructing a connector here would tie the runner to every connector's
    constructor.
    """

    def __init__(self, name: str, provider: CredentialProvider) -> None:
        self.name = name
        self.provider = provider


def _tool_stub(syncer: Syncer) -> _ToolStub:
    return _ToolStub(syncer.tool_name, syncer.provider)
