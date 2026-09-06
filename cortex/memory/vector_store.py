"""VectorStore — tenant-isolated semantic memory.

Same isolation model as the graph, for the same reason: **a collection per tenant**.
Qdrant supports filtering a shared collection by a payload field, and that was rejected
here on the same grounds as a `tenant_id` predicate in Cypher — its failure mode is one
forgotten filter leaking another tenant's Slack threads, and no test catches the day
someone adds a query without it. Isolation is instead a property of *which collection
you open*, and the collection name comes from `naming.collection_name` rather than from
any caller.

What lives here: unstructured text the graph cannot represent — Slack threads, meeting
notes, ticket bodies, documents. The graph holds entities and their relationships; this
holds prose, and `recall.py` joins the two.

Two deliberate constraints:

  - **Ids are derived, not random.** A point id is a UUID5 of (kind, source id), so a
    nightly re-sync of an edited Slack thread overwrites its point instead of adding a
    near-duplicate. Random ids would make memory grow monotonically and rank the same
    thread three times.
  - **The vector width is checked against the collection.** A collection built for a
    1024-wide model and written to with a 256-wide one fails at the boundary here,
    rather than becoming a corrupt collection that returns nonsense neighbours.
"""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from cortex.config.settings import get_settings
from cortex.memory.embeddings import Embeddings
from cortex.memory.naming import collection_name
from cortex.tenancy.context import TenantContext

#: The content kinds that get their own collection per tenant.
#:
#: Separate collections rather than one with a `kind` payload filter, so a recall for
#: meeting notes cannot accidentally rank support tickets, and so a kind can be dropped
#: (a retention policy on ticket bodies, say) without rewriting anything else.
KINDS = ("slack", "notes", "tickets", "docs")

#: The namespace for derived point ids. Fixed forever: changing it would orphan every
#: point already written, since the same document would then hash to a new id.
_ID_NAMESPACE = uuid.UUID("6f3a2e14-9c4b-5d7a-8e1f-2b3c4d5e6f70")

_MAX_LIMIT = 100


class InvalidKind(ValueError):
    pass


class FutureCutoff(ValueError):
    """A retention cutoff ahead of the clock, which would delete unexpired points."""


#: How far ahead of the real clock a cutoff may be before it is refused. Workers disagree
#: about the time by seconds, never by days.
_MAX_CUTOFF_SKEW = timedelta(hours=1)


def _check_kind(kind: str) -> str:
    if kind not in KINDS:
        raise InvalidKind(f"unknown kind {kind!r}; expected one of {', '.join(KINDS)}")
    return kind


def point_id(kind: str, source_id: str) -> str:
    """The stable id for one document.

    Derived so that re-ingesting an edited thread replaces its point. Random ids would
    let memory accumulate near-duplicates of the same conversation, each competing for
    the same recall slot.
    """
    return str(uuid.uuid5(_ID_NAMESPACE, f"{_check_kind(kind)}:{source_id}"))


@dataclass(frozen=True, slots=True)
class Document:
    """One piece of text to remember, with the provenance to cite it."""

    kind: str
    #: The id in the source system — a Slack thread ts, a ticket id.
    source_id: str
    text: str
    #: Where a reader can go to see the original. Carried through recall so a recalled
    #: fact can be cited rather than asserted.
    source_ref: str | None = None
    #: Filterable, non-secret facts: channel, author, date. Never credentials.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Hit:
    kind: str
    source_id: str
    text: str
    score: float
    source_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """Semantic memory, one collection per (tenant, kind)."""

    @abstractmethod
    async def provision(self, ctx: TenantContext) -> None:
        """Create the tenant's collections. Idempotent."""

    @abstractmethod
    async def drop(self, ctx: TenantContext) -> None:
        """Delete every collection belonging to this tenant. Offboarding must be total."""

    @abstractmethod
    async def upsert(self, ctx: TenantContext, documents: Sequence[Document]) -> int: ...

    @abstractmethod
    async def search(
        self,
        ctx: TenantContext,
        query: str,
        *,
        kinds: Sequence[str] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[Hit]: ...

    @abstractmethod
    async def delete_older_than(
        self, ctx: TenantContext, *, cutoff: datetime, kinds: Sequence[str] | None = None
    ) -> int:
        """Delete points ingested before the cutoff. Returns how many were removed."""

    @abstractmethod
    async def stats(self, ctx: TenantContext) -> dict[str, int]:
        """Point count per kind. Used for connector health and for isolation tests."""


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        embeddings: Embeddings,
        *,
        url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._embeddings = embeddings
        # Resolved once and remembered, so a client rebuilt for a new event loop is
        # rebuilt against the *same* cluster. An earlier version took an injected client
        # and rebuilt from settings on rebind, which would have silently moved a test
        # pointed at local Qdrant onto whatever `.env` named.
        settings = get_settings()
        self._url = url or settings.qdrant_url
        self._api_key = api_key if url else settings.qdrant_api_key
        self._client: AsyncQdrantClient | None = None
        # Which event loop the cached client belongs to. See `_get_client`.
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_client(self) -> AsyncQdrantClient:
        """The client for the *current* event loop, created if needed.

        An httpx async client caches connection state bound to the loop that created it,
        so a cached client used from a second loop fails with "is bound to a different
        event loop" — the exact failure `runtime.resources` was written to avoid, arriving
        here through a different door. A Celery worker that runs one loop per task, and a
        test session that runs one loop per test, both hit it.

        Rebinding rather than forbidding: the alternative is that every caller must
        construct a store per loop, which is a rule that holds until someone forgets. The
        stale client is dropped rather than closed, because awaiting a close on a dead
        loop is itself an error.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if self._client is not None and self._loop is not None and loop is not self._loop:
            self._client = None

        if self._client is None:
            self._client = AsyncQdrantClient(url=self._url, api_key=self._api_key)
            self._loop = loop
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._loop = None

    async def ping(self) -> bool:
        await self._get_client().get_collections()
        return True

    def _collection(self, ctx: TenantContext, kind: str) -> str:
        """The one place a collection is addressed.

        Reads the graph name from the context rather than accepting a name, so no caller
        can name another tenant's collection — the same reason `_graph()` exists in the
        FalkorDB store.
        """
        return collection_name(ctx.graph_name, _check_kind(kind))

    # ------------------------------------------------------------ lifecycle

    async def provision(self, ctx: TenantContext) -> None:
        client = self._get_client()
        for kind in KINDS:
            name = self._collection(ctx, kind)
            if await client.collection_exists(name):
                continue
            await client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=self._embeddings.dimensions,
                    # Cosine, because the embeddings are normalised and the question is
                    # "how similar is this text", not "how far apart in absolute terms".
                    distance=models.Distance.COSINE,
                ),
            )

    async def drop(self, ctx: TenantContext) -> None:
        client = self._get_client()
        for kind in KINDS:
            name = self._collection(ctx, kind)
            # Offboarding must be retryable, so an already-absent collection is success.
            if await client.collection_exists(name):
                await client.delete_collection(name)

    # ------------------------------------------------------------ writes

    async def upsert(self, ctx: TenantContext, documents: Sequence[Document]) -> int:
        if not documents:
            return 0
        for document in documents:
            _check_kind(document.kind)

        client = self._get_client()
        vectors = await self._embeddings.embed([d.text for d in documents])
        if len(vectors) != len(documents):
            raise ValueError(
                f"embedded {len(vectors)} vectors for {len(documents)} documents; "
                "refusing to write, since the mapping between text and vector is lost"
            )

        by_kind: dict[str, list[models.PointStruct]] = {}
        for document, vector in zip(documents, vectors, strict=True):
            if len(vector) != self._embeddings.dimensions:
                raise ValueError(
                    f"vector width {len(vector)} does not match the collection's "
                    f"{self._embeddings.dimensions}"
                )
            by_kind.setdefault(document.kind, []).append(
                models.PointStruct(
                    id=point_id(document.kind, document.source_id),
                    vector=vector,
                    payload={
                        "kind": document.kind,
                        "source_id": document.source_id,
                        "text": document.text,
                        "source_ref": document.source_ref,
                        # Written so retention can find this point later. Qdrant has no
                        # server-side creation time, so a point with no stamp is a point no
                        # age-based policy can ever delete -- the stamp is what makes the
                        # retention promise enforceable rather than aspirational.
                        "ingested_at": _now_iso(),
                        # Recorded so a collection written by one embedding model is
                        # recognisable after a model change, instead of quietly mixing
                        # two vector spaces in one index.
                        "embedding_model": self._embeddings.model,
                        **document.metadata,
                    },
                )
            )

        written = 0
        for kind, points in by_kind.items():
            await self._write(client, self._collection(ctx, kind), points)
            written += len(points)
        return written

    #: How many times a write is retried, and how long it waits between attempts.
    #:
    #: The managed cluster drops a request now and then -- twice in one afternoon, both times
    #: with a `ResponseHandlingException` carrying no message at all. The vectors are already
    #: computed by the time the write runs, so a retry costs one HTTP round trip and no
    #: embedding quota, which is the whole argument: the alternative was a stream reporting
    #: `ok` with its documents silently dropped, and re-embedding them on the next sync.
    _WRITE_ATTEMPTS = 3
    _WRITE_BACKOFF = 1.0

    async def _write(
        self, client: AsyncQdrantClient, collection: str, points: list[models.PointStruct]
    ) -> None:
        """One collection's points, retried on a transport failure.

        Only `ResponseHandlingException`, which is what qdrant-client raises when the request
        never got an answer. An `UnexpectedResponse` means the cluster answered and refused --
        a missing collection, a bad key, a malformed point -- and none of those become true on
        a second attempt.
        """
        for attempt in range(self._WRITE_ATTEMPTS):
            try:
                await client.upsert(collection_name=collection, points=points)
                return
            except ResponseHandlingException:
                if attempt == self._WRITE_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(self._WRITE_BACKOFF * (attempt + 1))

    # ------------------------------------------------------------ reads

    async def search(
        self,
        ctx: TenantContext,
        query: str,
        *,
        kinds: Sequence[str] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[Hit]:
        limit = max(1, min(int(limit), _MAX_LIMIT))
        wanted = [_check_kind(k) for k in (kinds or KINDS)]
        client = self._get_client()
        # `query=True`: Voyage embeds a question differently from a document, and using
        # the document setting for both measurably degrades retrieval.
        vector = (await self._embeddings.embed([query], query=True))[0]

        hits: list[Hit] = []
        for kind in wanted:
            name = self._collection(ctx, kind)
            if not await client.collection_exists(name):
                # A tenant that never ingested this kind has nothing to say about it.
                # Distinguished from an error deliberately: "no meeting notes" is a real
                # observation, and turning it into an exception would make an
                # investigation fail for asking a reasonable question.
                continue
            found = await client.query_points(
                collection_name=name,
                query=vector,
                limit=limit,
                score_threshold=min_score or None,
                with_payload=True,
            )
            for point in found.points:
                payload = dict(point.payload or {})
                hits.append(
                    Hit(
                        kind=kind,
                        source_id=str(payload.pop("source_id", "")),
                        text=str(payload.pop("text", "")),
                        score=float(point.score),
                        source_ref=payload.pop("source_ref", None),
                        metadata={
                            k: v for k, v in payload.items() if k not in ("kind", "embedding_model")
                        },
                    )
                )

        # Merged across kinds and re-ranked, then trimmed: asking for 10 must return the
        # best 10 overall, not the best 10 from each of four collections.
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    async def delete_older_than(
        self, ctx: TenantContext, *, cutoff: datetime, kinds: Sequence[str] | None = None
    ) -> int:
        """Delete points ingested before the cutoff.

        Filtered on the `ingested_at` stamp written at upsert time rather than on anything the
        source said: a Slack message from 2019 re-ingested last night is data we have held for
        one day, and expiring it on the message's own date would delete what we just fetched
        while keeping what we fetched years ago.

        Points written before the stamp existed carry none, and a range filter cannot match a
        missing field. Those are deleted too, by a second pass on `is_null` -- otherwise the
        oldest points in the system would be precisely the ones retention could never reach.

        A `cutoff` in the future is refused. `apply_retention` guards its own `now` for the
        same reason, and the guard is repeated here because this is the code that actually
        deletes: a caller passing a future cutoff turns "expire old points" into "empty the
        collection", and the recovery is a re-embed of everything at whatever the provider
        charges. The caller's guard protects the nightly job; this one protects every other
        caller, including the next script somebody writes in a hurry.
        """
        if cutoff > datetime.now(UTC) + _MAX_CUTOFF_SKEW:
            raise FutureCutoff(
                f"cutoff={cutoff.isoformat()} is in the future, which would delete points "
                f"that are not expired. Deletion here is not reversible."
            )
        client = self._get_client()
        removed = 0
        for kind in [_check_kind(k) for k in (kinds or KINDS)]:
            name = self._collection(ctx, kind)
            if not await client.collection_exists(name):
                continue
            before = int((await client.count(collection_name=name)).count)
            await client.delete(
                collection_name=name,
                points_selector=models.Filter(
                    should=[
                        models.FieldCondition(
                            key="ingested_at",
                            range=models.DatetimeRange(lt=cutoff),
                        ),
                        # Both, because Qdrant distinguishes them and only one describes a
                        # point written before the stamp existed: `is_null` matches a key
                        # present *with* a null value, `is_empty` matches a key that is
                        # missing entirely. The first version here used only `is_null`, which
                        # left the oldest points in the system permanently unreachable by
                        # retention — the exact opposite of what an expiry policy is for.
                        # Verified against the live server: `is_null` matched 0, `is_empty`
                        # matched 1.
                        models.IsNullCondition(is_null=models.PayloadField(key="ingested_at")),
                        models.IsEmptyCondition(is_empty=models.PayloadField(key="ingested_at")),
                    ]
                ),
                wait=True,
            )
            after = int((await client.count(collection_name=name)).count)
            removed += max(0, before - after)
        return removed

    async def stats(self, ctx: TenantContext) -> dict[str, int]:
        client = self._get_client()
        out: dict[str, int] = {}
        for kind in KINDS:
            name = self._collection(ctx, kind)
            if not await client.collection_exists(name):
                out[kind] = 0
                continue
            out[kind] = int((await client.count(collection_name=name)).count)
        return out


def _now_iso() -> str:
    """The ingestion timestamp, in the form Qdrant's datetime range filter understands."""
    return datetime.now(UTC).isoformat()
