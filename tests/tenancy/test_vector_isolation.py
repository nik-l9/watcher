"""Cross-tenant isolation for semantic memory.

The graph suite's counterpart, and a gate on the same terms: it must pass before a real
credential is entered. What it demonstrates is that a tenant's Slack threads are
unreachable from another tenant's search because they are in a **different collection**,
not because a payload filter was remembered.

Run against real Qdrant with fake embeddings. Real engine, because collection isolation
is the engine's behaviour and a stub would prove nothing; fake embeddings, because
nothing here depends on semantic quality and a paid API call per test would make the gate
expensive to run — which is how a gate stops being run.
"""

from __future__ import annotations

import pytest

from cortex.memory.naming import COLLECTION_PREFIX, InvalidTenantSlug, collection_name
from cortex.memory.vector_store import (
    KINDS,
    Document,
    InvalidKind,
    QdrantVectorStore,
    point_id,
)
from cortex.tenancy.context import TenantContext


def _docs(marker: str) -> list[Document]:
    return [
        Document(
            kind="slack",
            source_id=f"{marker}-thread-1",
            text=f"pausing the {marker} campaign today, budget exhausted",
            source_ref=f"https://slack.example/{marker}/1",
            metadata={"owner": marker, "channel": "marketing"},
        ),
        Document(
            kind="tickets",
            source_id=f"{marker}-ticket-9",
            text=f"customer cannot finish signup on mobile ({marker})",
            source_ref=f"https://help.example/{marker}/9",
            metadata={"owner": marker},
        ),
    ]


async def test_collections_are_named_per_tenant(
    vector_tenant_a: TenantContext, vector_tenant_b: TenantContext
) -> None:
    a = collection_name(vector_tenant_a.graph_name, "slack")
    b = collection_name(vector_tenant_b.graph_name, "slack")
    assert a != b
    assert a.startswith(COLLECTION_PREFIX)


async def test_search_never_returns_another_tenants_text(
    vectors: QdrantVectorStore,
    vector_tenant_a: TenantContext,
    vector_tenant_b: TenantContext,
) -> None:
    await vectors.upsert(vector_tenant_a, _docs("aaa"))
    await vectors.upsert(vector_tenant_b, _docs("bbb"))

    # The exact text tenant B stored, searched from tenant A. A shared collection with a
    # forgotten filter would return it; separate collections cannot.
    hits = await vectors.search(vector_tenant_a, "pausing the bbb campaign today", limit=20)
    assert hits, "tenant A should still see its own documents"
    assert all(hit.metadata.get("owner") == "aaa" for hit in hits), hits


async def test_a_tenant_with_no_memory_gets_nothing_rather_than_an_error(
    vectors: QdrantVectorStore, vector_tenant_a: TenantContext
) -> None:
    """ "Nothing recalled" is a real observation and must not surface as a failure."""
    assert await vectors.search(vector_tenant_a, "anything at all") == []


async def test_stats_count_only_the_tenants_own_points(
    vectors: QdrantVectorStore,
    vector_tenant_a: TenantContext,
    vector_tenant_b: TenantContext,
) -> None:
    await vectors.upsert(vector_tenant_a, _docs("aaa"))
    await vectors.upsert(vector_tenant_b, _docs("bbb") + _docs("bbb2"))

    assert await vectors.stats(vector_tenant_a) == {"slack": 1, "notes": 0, "tickets": 1, "docs": 0}


async def test_reingesting_the_same_document_replaces_it(
    vectors: QdrantVectorStore, vector_tenant_a: TenantContext
) -> None:
    """A nightly re-sync must not accumulate near-duplicates of one thread.

    Random point ids would add a second copy of an edited Slack message, and both copies
    would then compete for the same recall slot — so the same conversation would be
    presented to the analyst twice, as if it were two pieces of corroborating context.
    """
    await vectors.upsert(vector_tenant_a, _docs("aaa"))
    edited = Document(
        kind="slack",
        source_id="aaa-thread-1",
        text="actually the campaign is paused until August",
        metadata={"owner": "aaa"},
    )
    await vectors.upsert(vector_tenant_a, [edited])

    assert (await vectors.stats(vector_tenant_a))["slack"] == 1
    # Queried with the *exact* new text. The deterministic embedder is a hash, so it can
    # only match text identical to what was stored — it has no notion that "paused until
    # August" is related to "budget exhausted". Asserting a semantic neighbour here would
    # be asserting something the fake cannot do, and the property under test is
    # replacement, not ranking quality.
    hits = await vectors.search(vector_tenant_a, edited.text, kinds=["slack"], limit=5)
    assert [hit.text for hit in hits] == [edited.text]


async def test_dropping_a_tenant_removes_every_kind(
    vectors: QdrantVectorStore, vector_tenant_a: TenantContext
) -> None:
    """Offboarding must be total: a leftover collection is retained customer data."""
    await vectors.upsert(vector_tenant_a, _docs("aaa"))
    await vectors.drop(vector_tenant_a)

    assert await vectors.stats(vector_tenant_a) == dict.fromkeys(KINDS, 0)
    # Idempotent, so an interrupted offboarding can be retried.
    await vectors.drop(vector_tenant_a)


async def test_provision_is_idempotent(
    vectors: QdrantVectorStore, vector_tenant_a: TenantContext
) -> None:
    await vectors.upsert(vector_tenant_a, _docs("aaa"))
    await vectors.provision(vector_tenant_a)
    # Re-provisioning must not recreate (and so empty) an existing collection.
    assert (await vectors.stats(vector_tenant_a))["slack"] == 1


class TestNamingCannotEscapeATenant:
    """The whole isolation argument rests on names being unforgeable."""

    def test_a_foreign_collection_cannot_be_addressed(self) -> None:
        with pytest.raises(InvalidTenantSlug):
            collection_name("someone_elses_collection", "slack")

    def test_an_unknown_kind_is_refused(self) -> None:
        with pytest.raises(InvalidKind):
            point_id("../../etc/passwd", "1")

    def test_point_ids_are_stable_and_kind_scoped(self) -> None:
        assert point_id("slack", "t-1") == point_id("slack", "t-1")
        # The same source id in two kinds is two documents, not one.
        assert point_id("slack", "t-1") != point_id("tickets", "t-1")


class TestRetentionCannotEmptyACollectionByAccident:
    """The same guard as `apply_retention`, one level down.

    Repeated on the code that actually deletes rather than only on its caller: the caller's
    guard protects the nightly job, this one protects every other caller — including the next
    script somebody writes in a hurry. Which is how the incident recorded in
    `security-findings.md` happened.
    """

    async def test_old_points_are_deleted_and_recent_ones_kept(
        self, vectors: QdrantVectorStore, vector_tenant_a: TenantContext
    ) -> None:
        from datetime import UTC, datetime, timedelta

        await vectors.upsert(vector_tenant_a, _docs("aaa"))
        # Everything was ingested just now, so a cutoff an hour ago matches nothing.
        removed = await vectors.delete_older_than(
            vector_tenant_a, cutoff=datetime.now(UTC) - timedelta(hours=1)
        )
        assert removed == 0
        assert (await vectors.stats(vector_tenant_a))["slack"] == 1

    async def test_a_future_cutoff_is_refused(
        self, vectors: QdrantVectorStore, vector_tenant_a: TenantContext
    ) -> None:
        from datetime import UTC, datetime, timedelta

        from cortex.memory.vector_store import FutureCutoff

        await vectors.upsert(vector_tenant_a, _docs("aaa"))

        with pytest.raises(FutureCutoff, match="in the future"):
            await vectors.delete_older_than(
                vector_tenant_a, cutoff=datetime.now(UTC) + timedelta(days=500)
            )

        # And nothing was deleted on the way to the refusal.
        assert (await vectors.stats(vector_tenant_a))["slack"] == 1

    async def test_points_with_no_stamp_are_reachable(
        self, vectors: QdrantVectorStore, vector_tenant_a: TenantContext
    ) -> None:
        """Points written before `ingested_at` existed carry none, and a range filter cannot
        match a missing field. Without the `is_null` pass the oldest points in the system
        would be exactly the ones retention could never reach."""
        from datetime import UTC, datetime, timedelta

        from qdrant_client import models

        from cortex.memory.naming import collection_name

        await vectors.upsert(vector_tenant_a, _docs("aaa"))
        name = collection_name(vector_tenant_a.graph_name, "slack")
        client = vectors._get_client()
        # Strip the stamp, reproducing a point written before the field existed.
        await client.delete_payload(
            collection_name=name,
            keys=["ingested_at"],
            points=models.Filter(must=[]),
            wait=True,
        )

        removed = await vectors.delete_older_than(
            vector_tenant_a, cutoff=datetime.now(UTC) - timedelta(days=1), kinds=["slack"]
        )
        assert removed == 1


class TestATransportFailureIsRetriedRatherThanLosingTheDocuments:
    """The write is retried; the embedding is not repeated. That asymmetry is the point.

    A managed cluster dropped two writes in one afternoon, both with a
    `ResponseHandlingException` carrying no message. Each cost a stream its documents, which
    the next sync would have had to re-embed — and on a rate-limited embedding key that is the
    expensive half. By the time the write runs the vectors exist, so a retry costs one HTTP
    round trip and no quota.

    Run against a fake client rather than real Qdrant: the behaviour under test is what happens
    when the transport fails, and there is no way to ask a healthy cluster for that.
    """

    class _Flaky:
        """Fails `fail_times` writes with a transport error, then succeeds."""

        def __init__(self, fail_times: int) -> None:
            self.remaining = fail_times
            self.attempts = 0

        async def upsert(self, *, collection_name: str, points: list) -> None:
            del collection_name, points
            self.attempts += 1
            if self.remaining > 0:
                self.remaining -= 1
                import httpx
                from qdrant_client.http.exceptions import ResponseHandlingException

                raise ResponseHandlingException(source=httpx.ReadTimeout("dropped"))

    def _store(self) -> QdrantVectorStore:
        class _Embeddings:
            model = "fake"
            dimensions = 4

            async def embed(self, texts: list[str]) -> list[list[float]]:
                return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        store = QdrantVectorStore(_Embeddings(), url="http://unused", api_key=None)  # type: ignore[arg-type]
        store._WRITE_BACKOFF = 0.0  # type: ignore[misc]
        return store

    async def test_one_dropped_write_is_retried_and_succeeds(self) -> None:
        store = self._store()
        client = self._Flaky(fail_times=1)

        await store._write(client, "c_acme_1_docs", [])  # type: ignore[arg-type]

        assert client.attempts == 2, "the write was not retried"

    async def test_a_cluster_that_never_answers_still_raises(self) -> None:
        """Bounded. A retry loop that never gives up turns a dead cluster into a hung sync,
        and the note the runner writes is what makes the failure visible."""
        from qdrant_client.http.exceptions import ResponseHandlingException

        store = self._store()
        client = self._Flaky(fail_times=99)

        with pytest.raises(ResponseHandlingException):
            await store._write(client, "c_acme_1_docs", [])  # type: ignore[arg-type]

        assert client.attempts == store._WRITE_ATTEMPTS

    async def test_a_refusal_is_not_retried(self) -> None:
        """An `UnexpectedResponse` means the cluster answered and said no — a missing
        collection, a bad key, a malformed point. None of those become true on a second
        attempt, and retrying them turns a clear error into a slow one."""
        from qdrant_client.http.exceptions import UnexpectedResponse

        class _Refuses:
            def __init__(self) -> None:
                self.attempts = 0

            async def upsert(self, *, collection_name: str, points: list) -> None:
                del collection_name, points
                self.attempts += 1
                raise UnexpectedResponse(404, "Not Found", b"no such collection", None)

        store = self._store()
        client = _Refuses()

        with pytest.raises(UnexpectedResponse):
            await store._write(client, "c_acme_1_docs", [])  # type: ignore[arg-type]

        assert client.attempts == 1
