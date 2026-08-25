"""Embedding providers.

Recall needs text turned into vectors, and the choice of model is a deployment
decision rather than a design one — so it sits behind an interface, like the graph
store and the LLM. Voyage is the V1 implementation because `voyage-3` is strong on
retrieval and cheap enough to embed a nightly sync.

Two properties this module is built around:

  - **The dimension is declared, not discovered.** A collection is created with a fixed
    vector size, and a provider that silently returns a different width would either
    error on every write or, worse, be pointed at a collection built for another model.
    `dimensions` is part of the interface and asserted against what comes back.
  - **Tests never need a key or a network.** `DeterministicEmbeddings` hashes text into
    a unit vector, so identical text embeds identically and unrelated text lands far
    apart. That is enough to test isolation, upserts, ranking order and recall plumbing
    — everything except semantic quality, which a fake could not test honestly anyway.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import struct
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

import httpx

from cortex.config.settings import get_settings

#: Voyage's general-purpose retrieval model and its native width.
VOYAGE_MODEL = "voyage-3"
VOYAGE_DIMENSIONS = 1024

#: Voyage rejects oversized batches, and a nightly sync will happily hand over
#: thousands of Slack messages at once.
_MAX_BATCH = 128

#: A second cap, on **tokens** rather than count, because a count cap alone cannot keep a
#: batch inside a token-per-minute limit.
#:
#: Found by hitting it. A GitHub issues sync of 88 documents exhausted every retry and failed
#: with Voyage's own explanation: *"reduced rate limits of 3 RPM and 10K TPM"*. Eighty-eight
#: issue bodies is one request by count and far more than ten thousand tokens, and that is a
#: failure no retry policy can fix — waiting clears a request-per-minute limit, but the batch
#: still does not fit the token budget on the next attempt, or ever. Commits succeeded in the
#: same run because a commit subject is a line and an issue body is a page, which is why the
#: count cap looked adequate for months.
#:
#: Set below the observed 10,000 so a batch leaves room for the request that follows it inside
#: the same minute.
_MAX_BATCH_TOKENS = 8_000

#: Tokens are estimated from character length rather than tokenised.
#:
#: Four characters per token is the usual English approximation. A real tokeniser would mean
#: importing one to serve a *safety margin*, and being wrong by 20% here costs one extra
#: request; being wrong about which tokeniser Voyage uses would cost correctness. The estimate
#: is deliberately crude and the budget deliberately conservative.
_CHARS_PER_TOKEN = 4


_TIMEOUT_SECONDS = 60.0

#: Retry policy for a rate-limited or briefly failing provider.
#:
#: Sized against the free tier's 3 requests per minute: 20s, then 40s, then 80s covers a
#: full window twice over, so an ingest that trips the limit waits rather than dies.
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 20.0
_MAX_RETRY_WAIT_SECONDS = 120.0
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _batches(texts: Sequence[str]) -> Iterator[list[str]]:
    """Split into batches inside both the count and the token cap, preserving order.

    Order is load-bearing: `embed` maps the response back onto the input by position, so a
    batching change that reordered anything would attach the wrong vector to the wrong
    document — a corruption no later test detects, because every vector is individually valid.

    A single text over the whole budget is still yielded alone. Splitting it would change what
    it means, and a document that large is a different problem: `voyage-3` accepts 32k tokens per
    input, so it either fits alone or Voyage says so in a 400 that names the real cause.
    """
    batch: list[str] = []
    tokens = 0
    for text in texts:
        estimate = max(1, len(text) // _CHARS_PER_TOKEN)
        if batch and (len(batch) >= _MAX_BATCH or tokens + estimate > _MAX_BATCH_TOKENS):
            yield batch
            batch, tokens = [], 0
        batch.append(text)
        tokens += estimate
    if batch:
        yield batch


class EmbeddingError(RuntimeError):
    """The provider could not embed. Never swallowed: an unembedded document is
    invisible to recall, and silently skipping it makes memory quietly incomplete."""


class Embeddings(ABC):
    """Turns text into vectors of a fixed, declared width."""

    @property
    @abstractmethod
    def dimensions(self) -> int: ...

    @property
    @abstractmethod
    def model(self) -> str:
        """Recorded on every point, so a collection built with one model can be told
        apart from one built with another."""

    @abstractmethod
    async def embed(self, texts: Sequence[str], *, query: bool = False) -> list[list[float]]:
        """Embed a batch, in order.

        `query` selects the asymmetric input type where the provider supports one:
        Voyage embeds a question and a document differently, and using the document
        setting for both measurably degrades retrieval.
        """


class VoyageEmbeddings(Embeddings):
    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = VOYAGE_MODEL,
        dimensions: int = VOYAGE_DIMENSIONS,
    ) -> None:
        # `None` means "take it from settings"; an explicit empty string means "there is
        # no key". `or` conflated the two, so a caller deliberately constructing an
        # unconfigured provider silently picked up the ambient one — which is exactly the
        # kind of accidental credential use this codebase avoids elsewhere.
        self._api_key = get_settings().voyage_api_key if api_key is None else api_key
        self._model = model
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, texts: Sequence[str], *, query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        if not self._api_key:
            raise EmbeddingError(
                "VOYAGE_API_KEY is not set, so text cannot be embedded. Recall would "
                "return nothing, which is indistinguishable from a tenant having no "
                "memory — so this fails instead."
            )

        out: list[list[float]] = []
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            for batch in _batches(texts):
                response = await self._with_retry(client, batch, query=query)
                data = response.json().get("data") or []
                if len(data) != len(batch):
                    # Order and count carry the mapping back to the source text. A short
                    # response would silently attach the wrong vector to the wrong
                    # document, which no later test could detect.
                    raise EmbeddingError(
                        f"voyage returned {len(data)} embeddings for {len(batch)} inputs"
                    )
                for item in sorted(data, key=lambda d: d.get("index", 0)):
                    vector = [float(v) for v in item.get("embedding") or []]
                    if len(vector) != self._dimensions:
                        raise EmbeddingError(
                            f"voyage returned {len(vector)} dimensions, but this store "
                            f"was built for {self._dimensions}"
                        )
                    out.append(vector)
        return out

    async def _with_retry(
        self, client: httpx.AsyncClient, batch: list[str], *, query: bool
    ) -> httpx.Response:
        """POST one batch, waiting out a rate limit rather than failing on it.

        Voyage's free tier is **3 requests per minute**, which a nightly ingest of a few
        thousand Slack messages exceeds in its first second. Discovered by hitting it: a
        smoke test embedding four short strings got a 429 on the fourth. Raising there
        would abort the whole sync partway, leaving memory half-populated with no record
        of which half — and a partly-embedded corpus produces confidently incomplete
        recall, which is worse than none.

        Only 429 and 5xx are retried. A 400 (batch too long, unknown model) will fail
        identically however long we wait, and retrying it just delays the error.
        """
        delay = _RETRY_BACKOFF_SECONDS
        last: httpx.Response | None = None
        for attempt in range(_RETRY_ATTEMPTS + 1):
            last = await client.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "input": batch,
                    "input_type": "query" if query else "document",
                },
            )
            if last.status_code < 400:
                return last
            if last.status_code not in _RETRYABLE_STATUS or attempt == _RETRY_ATTEMPTS:
                break
            # `Retry-After` when the server sends one, since it knows the window; the
            # doubling fallback otherwise.
            wait = _retry_after(last) or delay
            await asyncio.sleep(wait)
            delay *= 2

        assert last is not None  # the loop always assigns before breaking
        # The body, not just the status: Voyage explains a rejected batch (too long, bad
        # model name, unpaid account) in it, and the status alone turns a five-second fix
        # into a guess.
        raise EmbeddingError(f"voyage returned {last.status_code}: {last.text[:400]}")


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    try:
        return min(float(raw), _MAX_RETRY_WAIT_SECONDS) if raw else None
    except ValueError:
        # A date-formatted Retry-After is legal but rare here; the fallback covers it.
        return None


class DeterministicEmbeddings(Embeddings):
    """A hash-based stand-in, for tests and offline development.

    Not a semantic model and not presented as one: it cannot tell that "signups fell"
    and "registrations dropped" are related. What it *can* do is make identical text
    embed identically and unrelated text embed far apart, deterministically and with no
    network — which is what the isolation, upsert and ranking tests actually need.
    Using the real provider for those would make the test suite cost money and fail
    offline, while testing nothing extra.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model(self) -> str:
        # Named so a collection accidentally built with the fake is obvious in Qdrant
        # rather than looking like a real one.
        return f"deterministic-{self._dimensions}"

    async def embed(self, texts: Sequence[str], *, query: bool = False) -> list[list[float]]:
        del query  # Symmetric: a fake has no asymmetric input types to honour.
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        # Expanded from a counter-salted digest so the whole width is filled from the
        # text rather than padded with zeros, then normalised so cosine distance is
        # comparable across documents of different length.
        raw = bytearray()
        counter = 0
        seed = text.encode("utf-8")
        while len(raw) < self._dimensions * 4:
            raw += hashlib.sha256(seed + struct.pack(">I", counter)).digest()
            counter += 1
        floats = [
            struct.unpack(">I", bytes(raw[i * 4 : i * 4 + 4]))[0] / 2**32 - 0.5
            for i in range(self._dimensions)
        ]
        norm = math.sqrt(sum(f * f for f in floats)) or 1.0
        return [f / norm for f in floats]
