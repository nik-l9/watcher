"""The embedding providers.

The Voyage path is tested against a stubbed transport rather than the real API: the free
tier allows three requests a minute, so a suite that called it for real would either be
rate-limited into flakiness or cost money to run. What is worth testing here is not
Voyage's model — it is our handling of Voyage's failures, which is where the bugs live.
"""

from __future__ import annotations

import httpx
import pytest

from cortex.memory.embeddings import (
    _CHARS_PER_TOKEN,
    _MAX_BATCH,
    _MAX_BATCH_TOKENS,
    VOYAGE_DIMENSIONS,
    DeterministicEmbeddings,
    EmbeddingError,
    VoyageEmbeddings,
    _batches,
)


def _client(handler: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def _stub_transport(monkeypatch: pytest.MonkeyPatch, handler: object) -> None:
    """Make `embed()` talk to a stub, so its own checks are what gets tested.

    `embed` builds its own client, which is right for production and awkward for a test.
    Patching the class rather than reimplementing the checks in the test: a test that
    re-derives the assertion it is verifying passes even when the code is wrong.
    """
    real = httpx.AsyncClient

    def _factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs.pop("timeout", None)
        return real(transport=httpx.MockTransport(handler), **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("cortex.memory.embeddings.httpx.AsyncClient", _factory)


def _ok(count: int, dimensions: int = VOYAGE_DIMENSIONS) -> httpx.Response:
    return httpx.Response(
        200,
        json={"data": [{"index": i, "embedding": [0.1] * dimensions} for i in range(count)]},
    )


class TestDeterministicEmbeddings:
    async def test_identical_text_embeds_identically(self) -> None:
        e = DeterministicEmbeddings()
        first, second = await e.embed(["signups fell", "signups fell"])
        assert first == second

    async def test_different_text_embeds_differently(self) -> None:
        e = DeterministicEmbeddings()
        first, second = await e.embed(["signups fell", "deploys are fine"])
        assert first != second

    async def test_vectors_are_the_declared_width_and_normalised(self) -> None:
        e = DeterministicEmbeddings(dimensions=64)
        (vector,) = await e.embed(["anything"])
        assert len(vector) == 64 == e.dimensions
        # Unit length, so cosine distance compares documents of different length fairly.
        assert abs(sum(v * v for v in vector) ** 0.5 - 1.0) < 1e-9

    async def test_the_model_name_marks_it_as_a_fake(self) -> None:
        """A collection accidentally built with the fake must be obvious in Qdrant
        rather than passing for a real one."""
        assert "deterministic" in DeterministicEmbeddings().model


class TestVoyageEmbeddings:
    async def test_an_absent_key_fails_loudly(self) -> None:
        """Returning nothing would be indistinguishable from a tenant with no memory,
        and an investigation would report absent history rather than a broken config."""
        provider = VoyageEmbeddings(api_key="")
        with pytest.raises(EmbeddingError, match="VOYAGE_API_KEY"):
            await provider.embed(["anything"])

    async def test_an_empty_batch_makes_no_request(self) -> None:
        assert await VoyageEmbeddings(api_key="k").embed([]) == []

    async def test_a_rate_limit_is_waited_out_rather_than_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Voyage's free tier is 3 requests a minute. An ingest that trips it must wait,
        not abort half way through and leave memory partly populated."""
        slept: list[float] = []

        async def _no_sleep(seconds: float) -> None:
            slept.append(seconds)

        monkeypatch.setattr("cortex.memory.embeddings.asyncio.sleep", _no_sleep)

        calls = 0

        def _handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(429, json={"detail": "rate limited"})
            return _ok(1)

        provider = VoyageEmbeddings(api_key="k")
        async with _client(_handler) as client:
            response = await provider._with_retry(client, ["text"], query=False)

        assert response.status_code == 200
        assert calls == 2
        assert slept, "it should have waited before retrying"

    async def test_retry_after_is_honoured_when_the_server_sends_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The server knows its own window; the doubling fallback is only a guess."""
        slept: list[float] = []

        async def _no_sleep(seconds: float) -> None:
            slept.append(seconds)

        monkeypatch.setattr("cortex.memory.embeddings.asyncio.sleep", _no_sleep)

        calls = 0

        def _handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(429, headers={"retry-after": "7"}, json={})
            return _ok(1)

        async with _client(_handler) as client:
            await VoyageEmbeddings(api_key="k")._with_retry(client, ["t"], query=False)

        assert slept == [7.0]

    async def test_a_bad_request_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 400 fails identically however long we wait, so retrying only delays the
        error — and hides it behind a minute of apparent progress."""

        async def _no_sleep(seconds: float) -> None:  # pragma: no cover
            raise AssertionError("a 400 must not be retried")

        monkeypatch.setattr("cortex.memory.embeddings.asyncio.sleep", _no_sleep)

        calls = 0

        def _handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(400, text="unknown model")

        provider = VoyageEmbeddings(api_key="k")
        async with _client(_handler) as client:
            with pytest.raises(EmbeddingError, match="unknown model"):
                await provider._with_retry(client, ["t"], query=False)

        assert calls == 1

    async def test_a_short_response_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Count and order carry the mapping from vector back to source text. Accepting
        a short response would attach the wrong vector to the wrong document, which
        nothing downstream could detect."""
        _stub_transport(monkeypatch, lambda request: _ok(1))

        with pytest.raises(EmbeddingError, match="1 embeddings for 2 inputs"):
            await VoyageEmbeddings(api_key="k").embed(["a", "b"])

    async def test_a_width_mismatch_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A collection built for 1024 dimensions and written with 256 becomes a corrupt
        index that returns plausible nonsense, so the mismatch fails at the boundary."""
        _stub_transport(monkeypatch, lambda request: _ok(1, dimensions=256))

        with pytest.raises(EmbeddingError, match="256 dimensions"):
            await VoyageEmbeddings(api_key="k").embed(["t"])

    async def test_a_good_response_is_returned_in_input_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Voyage may return items out of order; they are re-sorted by index, because the
        position in the list is the only thing tying a vector to its text."""

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [0.2] * VOYAGE_DIMENSIONS},
                        {"index": 0, "embedding": [0.1] * VOYAGE_DIMENSIONS},
                    ]
                },
            )

        _stub_transport(monkeypatch, _handler)
        first, second = await VoyageEmbeddings(api_key="k").embed(["a", "b"])
        assert first[0] == pytest.approx(0.1)
        assert second[0] == pytest.approx(0.2)

    async def test_a_query_is_embedded_as_a_query(self) -> None:
        """Voyage embeds a question differently from a document, and using the document
        setting for both measurably degrades retrieval."""
        seen: list[str] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            import json

            seen.append(json.loads(request.content)["input_type"])
            return _ok(1)

        provider = VoyageEmbeddings(api_key="k")
        async with _client(_handler) as client:
            await provider._with_retry(client, ["t"], query=True)
            await provider._with_retry(client, ["t"], query=False)

        assert seen == ["query", "document"]


class TestBatchingRespectsBothCaps:
    """A count cap alone cannot keep a request inside a token-per-minute limit.

    A GitHub issues sync of 88 documents exhausted every retry and failed with Voyage's own
    explanation: *"reduced rate limits of 3 RPM and 10K TPM"*. Eighty-eight issue bodies is one
    request by count and far more than ten thousand tokens — and no retry policy can fix that,
    because waiting clears a request-per-minute limit while the batch still does not fit the token
    budget on the next attempt, or ever.

    Commits succeeded in the same run. A commit subject is a line and an issue body is a page,
    which is why a count-only cap looked adequate for months.
    """

    async def test_a_batch_of_long_texts_is_split_by_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Eighty-eight documents of ~1,000 tokens each. One request by count, eleven by tokens."""
        sizes: list[int] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            import json

            batch = json.loads(request.content)["input"]
            sizes.append(len(batch))
            return _ok(len(batch))

        _stub_transport(monkeypatch, _handler)
        long_text = "x" * (1000 * _CHARS_PER_TOKEN)
        vectors = await VoyageEmbeddings(api_key="k").embed([long_text] * 88)

        assert len(vectors) == 88, "every document must still be embedded"
        assert len(sizes) > 1, "a count-only cap would have sent this as one request"
        # No request may exceed the token budget, which is the whole point.
        assert max(sizes) * 1000 <= _MAX_BATCH_TOKENS
        assert sum(sizes) == 88

    async def test_short_texts_still_batch_by_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The token cap must not shrink the ordinary case. 300 commit subjects are nowhere near
        the token budget and should go out in ceil(300/128) = 3 requests, as before."""
        sizes: list[int] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            import json

            batch = json.loads(request.content)["input"]
            sizes.append(len(batch))
            return _ok(len(batch))

        _stub_transport(monkeypatch, _handler)
        await VoyageEmbeddings(api_key="k").embed(["fix: a short commit subject"] * 300)

        assert sizes == [_MAX_BATCH, _MAX_BATCH, 300 - 2 * _MAX_BATCH]

    async def test_order_survives_batching(self) -> None:
        """Load-bearing, and the failure would be silent. `embed` maps the response back onto the
        input by position, so a batching change that reordered anything would attach the wrong
        vector to the wrong document — and every vector would still be individually valid."""
        texts = [f"{i}:{'x' * (i * 400)}" for i in range(40)]
        assert [t for batch in _batches(texts) for t in batch] == texts

    async def test_one_oversized_text_is_sent_alone_rather_than_dropped(self) -> None:
        """Splitting a document changes what it means, so it goes out on its own and Voyage
        decides. A silent drop would make recall quietly incomplete, which is the failure this
        module's error type exists to prevent."""
        huge = "x" * (_MAX_BATCH_TOKENS * _CHARS_PER_TOKEN * 3)
        batches = list(_batches(["short", huge, "short"]))

        assert [huge] in batches
        assert sum(len(b) for b in batches) == 3
