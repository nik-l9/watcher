"""Per-tenant call ceilings.

Per-investigation budgets already exist (`Employee.budget`): steps, tool calls, tokens,
wall clock. They bound *one* investigation. Nothing bounded a tenant across investigations,
which leaves three real gaps:

  - **A runaway loop.** A bug that enqueues investigations, or a user hammering the button,
    multiplies a bounded budget by an unbounded number of runs.
  - **A tenant DoS-ing their own upstream.** Credentials are bring-your-own, so a tenant's
    HubSpot quota is theirs to exhaust — but when they exhaust it *through us*, every
    subsequent investigation fails with a 429 and it looks like our outage.
  - **Noisy-neighbour latency.** One tenant's burst fills the worker pool. The queue is
    fair; the upstreams are not.

**This is a cost and fairness control, not a security boundary.** Tenant isolation is
enforced structurally — a separate graph, a separate collection, a scoped credential lookup.
Nothing here is load-bearing for that, which is why it may fail open.

**Failing open is deliberate, and recorded.** Redis is the Celery broker; if it is down, no
investigation is running anyway. But an advisory limiter that cannot reach its store must not
be the thing that stops work — so a store failure permits the call and says so, and the
`limited` flag on the decision is what stops "we never rejected anything" and "we could not
tell" from being indistinguishable. That distinction is the same one this codebase keeps
having to make explicit.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from cortex.db.models import CredentialProvider

#: Windows enforced, longest last. Two rather than one because they answer different
#: questions: the minute window catches a runaway loop within seconds, and the hour window
#: catches a slow drip that no per-minute limit would ever notice.
DEFAULT_LIMITS: tuple[tuple[int, int], ...] = (
    (60, 120),  # 120 calls per minute per provider
    (3600, 2000),  # 2000 calls per hour per provider
)

#: How long a window's key outlives the window itself, so a clock skew between workers
#: cannot expire a window that is still being counted.
_KEY_GRACE_SECONDS = 10


@dataclass(frozen=True, slots=True)
class Decision:
    """Whether a call may proceed, and whether we actually know."""

    allowed: bool
    #: False when the limiter could not reach its store. The call is permitted, but nothing
    #: was counted — so a caller reading `allowed` alone would believe a limit was checked.
    limited: bool = True
    #: Which window was exceeded, for the message the analyst reads.
    window_seconds: int | None = None
    limit: int | None = None
    retry_after_seconds: float | None = None

    @property
    def reason(self) -> str:
        if self.allowed:
            return ""
        return (
            f"tenant rate limit reached: more than {self.limit} calls in "
            f"{self.window_seconds}s for this provider"
        )


class RateLimiter(Protocol):
    async def check(
        self, tenant_id: uuid.UUID, provider: CredentialProvider | None
    ) -> Decision: ...


class NullRateLimiter:
    """Permits everything, and says it did not limit.

    The default, so a test or a script needs no Redis. `limited=False` matters: it is what
    keeps "nothing was rejected" distinguishable from "nothing was checked".
    """

    async def check(self, tenant_id: uuid.UUID, provider: CredentialProvider | None) -> Decision:
        del tenant_id, provider
        return Decision(allowed=True, limited=False)


class RedisRateLimiter:
    """A sliding-window limiter over Redis sorted sets.

    Sliding rather than fixed windows: a fixed window lets a tenant spend its whole minute's
    allowance in the last second of one window and again in the first second of the next,
    which is twice the limit at exactly the moment a runaway loop produces it.

    **Adds first, then checks.** The alternative — count, then add if under — is a race two
    workers can both win, admitting more than the limit. Adding first can over-*reject* under
    contention (both requests see the inflated count and both withdraw), which is the safe
    direction to be wrong in: a rejected call is retried or reported as a gap, an
    over-admitted one is spend we did not intend.
    """

    def __init__(
        self,
        client: Any,
        *,
        limits: tuple[tuple[int, int], ...] = DEFAULT_LIMITS,
        clock: Any = time.time,
    ) -> None:
        self._client = client
        self._limits = limits
        self._clock = clock

    async def check(self, tenant_id: uuid.UUID, provider: CredentialProvider | None) -> Decision:
        now = self._clock()
        # One member per call, unique so two calls in the same millisecond both count. A
        # timestamp alone would collapse them and undercount a burst.
        member = f"{now:.6f}:{uuid.uuid4().hex[:8]}"
        scope = provider.value if provider is not None else "local"

        for window, limit in self._limits:
            key = f"cortex:rl:{tenant_id}:{scope}:{window}"
            try:
                count = await self._record(key, member, now, window)
            except Exception:  # noqa: BLE001 - see the module docstring
                # The store is unreachable. Permit, and say nothing was counted.
                return Decision(allowed=True, limited=False)

            if count > limit:
                await self._withdraw(key, member)
                return Decision(
                    allowed=False,
                    window_seconds=window,
                    limit=limit,
                    # The window is the honest wait: the oldest call in it ages out first,
                    # and a shorter hint would invite an immediate retry that also fails.
                    retry_after_seconds=float(window),
                )
        return Decision(allowed=True)

    async def _record(self, key: str, member: str, now: float, window: int) -> int:
        pipeline = self._client.pipeline()
        # Drop everything older than the window before counting, so the set never grows past
        # what one window can hold.
        pipeline.zremrangebyscore(key, 0, now - window)
        pipeline.zadd(key, {member: now})
        pipeline.zcard(key)
        pipeline.expire(key, window + _KEY_GRACE_SECONDS)
        results = await pipeline.execute()
        return int(results[2])

    async def _withdraw(self, key: str, member: str) -> None:
        """Remove our own member after a rejection.

        Without this a rejected call still occupies a slot, so a tenant at its ceiling would
        keep its own window full by retrying — a limiter that punishes retries rather than
        bounding work.
        """
        try:
            await self._client.zrem(key, member)
        except Exception:  # noqa: BLE001 - a leaked member expires with the key
            return
