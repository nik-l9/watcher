"""Per-tenant call ceilings.

Run against real Redis, on the same reasoning as the graph and vector isolation suites: the
behaviour under test is the store's — sorted-set semantics, expiry, atomicity of a pipeline —
and a fake would happily confirm a guarantee Redis does not provide.

What is worth proving is not that a counter counts. It is that:

  - one tenant's burst cannot spend another tenant's allowance;
  - a rejected call does not keep its own slot, or a tenant at its ceiling would be punished
    for retrying rather than merely bounded;
  - a store outage permits the call **and says nothing was checked**, so "we rejected
    nothing" stays distinguishable from "we could not tell".
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
import redis.asyncio as redis

from cortex.config.settings import get_settings
from cortex.db.models import CredentialProvider
from cortex.tenancy.limits import NullRateLimiter, RedisRateLimiter

PROVIDER = CredentialProvider.GITHUB


@pytest_asyncio.fixture
async def client() -> AsyncIterator[redis.Redis]:
    connection = redis.from_url(get_settings().redis_url, decode_responses=True)
    try:
        await connection.ping()
    except Exception as exc:  # pragma: no cover - the compose stack must be up
        pytest.skip(f"redis unavailable: {exc}")
    try:
        yield connection
    finally:
        await connection.aclose()


@pytest_asyncio.fixture
async def keys(client: redis.Redis) -> AsyncIterator[list[str]]:
    """Tracks the keys a test created, so nothing leaks into the next one."""
    created: list[str] = []
    try:
        yield created
    finally:
        if created:
            await client.delete(*created)


async def _limiter(
    client: redis.Redis, *, per_minute: int = 3, clock: object | None = None
) -> RedisRateLimiter:
    return RedisRateLimiter(
        client,
        limits=((60, per_minute),),
        **({"clock": clock} if clock else {}),  # type: ignore[arg-type]
    )


class TestTheCeilingBinds:
    async def test_calls_are_permitted_up_to_the_limit_then_refused(
        self, client: redis.Redis, keys: list[str]
    ) -> None:
        tenant = uuid.uuid4()
        keys.append(f"cortex:rl:{tenant}:{PROVIDER.value}:60")
        limiter = await _limiter(client, per_minute=3)

        verdicts = [(await limiter.check(tenant, PROVIDER)).allowed for _ in range(5)]

        assert verdicts == [True, True, True, False, False]

    async def test_a_refusal_explains_itself_and_says_how_long_to_wait(
        self, client: redis.Redis, keys: list[str]
    ) -> None:
        tenant = uuid.uuid4()
        keys.append(f"cortex:rl:{tenant}:{PROVIDER.value}:60")
        limiter = await _limiter(client, per_minute=1)

        await limiter.check(tenant, PROVIDER)
        decision = await limiter.check(tenant, PROVIDER)

        assert decision.allowed is False
        assert decision.limit == 1
        assert decision.window_seconds == 60
        # The window, not a shorter guess: the oldest call in it ages out first, and a
        # smaller hint would invite an immediate retry that also fails.
        assert decision.retry_after_seconds == 60.0
        assert "rate limit reached" in decision.reason

    async def test_a_refused_call_does_not_keep_its_slot(
        self, client: redis.Redis, keys: list[str]
    ) -> None:
        """Otherwise a tenant at its ceiling keeps its own window full by retrying — a
        limiter that punishes retries rather than bounding work."""
        tenant = uuid.uuid4()
        key = f"cortex:rl:{tenant}:{PROVIDER.value}:60"
        keys.append(key)
        limiter = await _limiter(client, per_minute=2)

        for _ in range(2):
            assert (await limiter.check(tenant, PROVIDER)).allowed
        for _ in range(5):
            assert not (await limiter.check(tenant, PROVIDER)).allowed

        # Two admitted calls, and none of the five refusals left a member behind.
        assert await client.zcard(key) == 2


class TestTenantsAndProvidersAreSeparate:
    async def test_one_tenants_burst_does_not_spend_anothers_allowance(
        self, client: redis.Redis, keys: list[str]
    ) -> None:
        """The noisy-neighbour case this exists for."""
        loud, quiet = uuid.uuid4(), uuid.uuid4()
        keys.extend(
            [
                f"cortex:rl:{loud}:{PROVIDER.value}:60",
                f"cortex:rl:{quiet}:{PROVIDER.value}:60",
            ]
        )
        limiter = await _limiter(client, per_minute=2)

        for _ in range(6):
            await limiter.check(loud, PROVIDER)

        assert (await limiter.check(quiet, PROVIDER)).allowed is True

    async def test_providers_are_counted_separately(
        self, client: redis.Redis, keys: list[str]
    ) -> None:
        """Exhausting a GitHub allowance must not stop a Slack call: the quotas being
        protected are the upstreams' own, and they are unrelated."""
        tenant = uuid.uuid4()
        keys.extend(
            [
                f"cortex:rl:{tenant}:{PROVIDER.value}:60",
                f"cortex:rl:{tenant}:{CredentialProvider.SLACK.value}:60",
            ]
        )
        limiter = await _limiter(client, per_minute=1)

        await limiter.check(tenant, PROVIDER)
        assert not (await limiter.check(tenant, PROVIDER)).allowed
        assert (await limiter.check(tenant, CredentialProvider.SLACK)).allowed

    async def test_a_credential_free_tool_has_its_own_scope(
        self, client: redis.Redis, keys: list[str]
    ) -> None:
        """The SQL tool has no provider. It still gets a ceiling — a runaway loop over local
        files is still a runaway loop — but not one shared with a real upstream."""
        tenant = uuid.uuid4()
        keys.append(f"cortex:rl:{tenant}:local:60")
        limiter = await _limiter(client, per_minute=1)

        assert (await limiter.check(tenant, None)).allowed
        assert not (await limiter.check(tenant, None)).allowed


class TestTheWindowSlides:
    async def test_old_calls_age_out(self, client: redis.Redis, keys: list[str]) -> None:
        """A fixed window would let a tenant spend its whole minute in the last second of
        one window and again in the first second of the next — twice the limit at exactly
        the moment a runaway loop produces it."""
        tenant = uuid.uuid4()
        keys.append(f"cortex:rl:{tenant}:{PROVIDER.value}:60")
        # An injected clock, so the test does not sleep for a minute.
        ticks = iter([1000.0, 1000.1, 1000.2, 1200.0])
        limiter = await _limiter(client, per_minute=2, clock=lambda: next(ticks))

        assert (await limiter.check(tenant, PROVIDER)).allowed
        assert (await limiter.check(tenant, PROVIDER)).allowed
        assert not (await limiter.check(tenant, PROVIDER)).allowed
        # 200 seconds later the earlier calls are outside the window.
        assert (await limiter.check(tenant, PROVIDER)).allowed

    async def test_the_key_expires_so_idle_tenants_cost_nothing(
        self, client: redis.Redis, keys: list[str]
    ) -> None:
        tenant = uuid.uuid4()
        key = f"cortex:rl:{tenant}:{PROVIDER.value}:60"
        keys.append(key)
        limiter = await _limiter(client, per_minute=5)

        await limiter.check(tenant, PROVIDER)
        ttl = await client.ttl(key)

        assert 0 < ttl <= 70


class TestOutages:
    async def test_a_store_failure_permits_the_call_and_admits_it_checked_nothing(
        self,
    ) -> None:
        """An advisory limiter must not be the thing that stops work. But `allowed=True`
        alone would let a caller believe a limit was enforced, which is the same
        absence-of-access-reads-as-absence-of-evidence confusion in a new place."""

        class _Broken:
            def pipeline(self) -> object:
                raise RuntimeError("redis unreachable")

        limiter = RedisRateLimiter(_Broken(), limits=((60, 1),))
        decision = await limiter.check(uuid.uuid4(), PROVIDER)

        assert decision.allowed is True
        assert decision.limited is False

    async def test_the_null_limiter_reports_that_it_did_not_limit(self) -> None:
        decision = await NullRateLimiter().check(uuid.uuid4(), PROVIDER)
        assert decision.allowed is True
        assert decision.limited is False
