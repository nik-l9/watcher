"""What a tenant has cost.

Usage was recorded per investigation and never aggregated, so there was no answer to "what is
this tenant costing us" — the question every multi-tenant product is eventually asked, usually
by someone deciding a price.

Adapted in intent from OpenHands' `ConversationStats` / `MetricsSnapshot`, which track spend per
conversation. Ours differs in one way that matters: their unit is a conversation and ours is a
*tenant*, because ours is the thing that gets an invoice.

**The reason this needed a migration first.** `investigations.tokens_used` was a single total,
and a total cannot be priced. Input and output differ by 5x on every model we run, and a cache
read costs a tenth of a fresh input token — against a measured 98% cache-hit rate, a bill
computed from the total overstates it by roughly an order of magnitude. `Usage`'s own docstring
said exactly that, and the row discarded the fields anyway.

**Rates are dollars per million tokens, and the fallback is deliberately expensive.** An
unpriced model bills at the highest rate we know rather than at zero: a spend report that
silently under-counts is worse than one that visibly over-counts, because nobody investigates a
number that looks fine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import Investigation, InvestigationStatus, Tenant

#: Dollars per million tokens: (input, output). Cache rates are derived below.
#:
#: Duplicated from `cortex.bench.budget` on purpose rather than imported. The benchmark's
#: table exists to stop a benchmark run overspending; this one prices a customer's bill. They
#: happen to agree today, and coupling them would mean a benchmark tweak silently repricing
#: history.
RATES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5": (10.00, 50.00),
}

#: The most expensive thing we know of. Used when a model is unrecognised or absent.
_FALLBACK = (10.00, 50.00)

#: A cache read costs a tenth of a fresh input token; a cache write costs a quarter more than
#: one. Ratios rather than absolute rates, so they stay right when a model's price changes.
_CACHE_READ_SHARE = 0.10
_CACHE_WRITE_SHARE = 1.25


@dataclass(frozen=True, slots=True)
class Spend:
    """One tenant's usage over a window."""

    tenant_slug: str
    investigations: int
    completed: int
    failed: int
    cancelled: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    usd: float
    #: True when at least one investigation carried no model name, so its cost was priced at
    #: the fallback rate. Surfaced rather than hidden: an estimate presented as a measurement
    #: is the failure this codebase keeps guarding against.
    estimated: bool = False

    @property
    def usd_per_investigation(self) -> float:
        return self.usd / self.investigations if self.investigations else 0.0

    @property
    def cache_hit_rate(self) -> float:
        considered = self.input_tokens + self.cache_read_tokens
        return self.cache_read_tokens / considered if considered else 0.0


def cost_usd(
    model: str | None,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """What one investigation cost.

    Cache reads are billed at a fraction and cache writes at a premium, because that is how
    the provider bills them — and with caching on, the fresh-input count is a small minority
    of what was actually sent.
    """
    input_rate, output_rate = RATES.get((model or "").strip().lower(), _FALLBACK)
    dollars = (
        input_tokens * input_rate
        + output_tokens * output_rate
        + cache_read_tokens * input_rate * _CACHE_READ_SHARE
        + cache_write_tokens * input_rate * _CACHE_WRITE_SHARE
    )
    return dollars / 1_000_000


async def spend_for_tenant(
    session: AsyncSession,
    *,
    tenant_id: object,
    days: int = 30,
    now: datetime | None = None,
) -> Spend | None:
    """One tenant's spend over the trailing window, or None if the tenant is unknown."""
    moment = now or datetime.now(UTC)
    since = moment - timedelta(days=days)

    slug = await session.scalar(select(Tenant.slug).where(Tenant.id == tenant_id))
    if slug is None:
        return None

    rows = (
        await session.execute(
            select(
                Investigation.model,
                Investigation.status,
                func.count().label("n"),
                func.sum(Investigation.input_tokens),
                func.sum(Investigation.output_tokens),
                func.sum(Investigation.cache_read_tokens),
                func.sum(Investigation.cache_write_tokens),
            )
            .where(
                Investigation.tenant_id == tenant_id,
                Investigation.created_at >= since,
            )
            # Grouped by model because pricing is per model: summing tokens across models and
            # applying one rate would be wrong the moment a second model is ever used, and
            # wrong silently.
            .group_by(Investigation.model, Investigation.status)
        )
    ).all()

    return _fold(slug, rows)


async def spend_by_tenant(
    session: AsyncSession, *, days: int = 30, now: datetime | None = None
) -> list[Spend]:
    """Every active tenant's spend, most expensive first.

    The operator's view. Ordered by cost rather than by name, because the reason to open this
    is to find out who is expensive.
    """
    moment = now or datetime.now(UTC)
    since = moment - timedelta(days=days)

    rows = (
        await session.execute(
            select(
                Tenant.slug,
                Investigation.model,
                Investigation.status,
                func.count().label("n"),
                func.sum(Investigation.input_tokens),
                func.sum(Investigation.output_tokens),
                func.sum(Investigation.cache_read_tokens),
                func.sum(Investigation.cache_write_tokens),
            )
            .join(Investigation, Investigation.tenant_id == Tenant.id)
            .where(Tenant.is_active.is_(True), Investigation.created_at >= since)
            .group_by(Tenant.slug, Investigation.model, Investigation.status)
        )
    ).all()

    by_slug: dict[str, list[tuple]] = {}
    for slug, *rest in rows:
        by_slug.setdefault(slug, []).append(tuple(rest))

    spends = [_fold(slug, grouped) for slug, grouped in by_slug.items()]
    return sorted(spends, key=lambda spend: spend.usd, reverse=True)


def _fold(slug: str, rows: list[tuple]) -> Spend:
    """Collapse (model, status) groups into one tenant's figure."""
    totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    counts = {"all": 0, "completed": 0, "failed": 0, "cancelled": 0}
    usd = 0.0
    estimated = False

    for model, status, count, inp, out, cread, cwrite in rows:
        inp, out, cread, cwrite = (int(v or 0) for v in (inp, out, cread, cwrite))
        totals["input"] += inp
        totals["output"] += out
        totals["cache_read"] += cread
        totals["cache_write"] += cwrite
        counts["all"] += int(count)
        if status is InvestigationStatus.COMPLETED:
            counts["completed"] += int(count)
        elif status is InvestigationStatus.FAILED:
            counts["failed"] += int(count)
        elif status is InvestigationStatus.CANCELLED:
            counts["cancelled"] += int(count)

        # Priced per group, then summed. Pricing the summed totals once would apply one
        # model's rates to another model's tokens.
        usd += cost_usd(
            model,
            input_tokens=inp,
            output_tokens=out,
            cache_read_tokens=cread,
            cache_write_tokens=cwrite,
        )
        if not model or model.strip().lower() not in RATES:
            # Only counts as an estimate if the group actually spent something. A zero-token
            # group with no model name would otherwise flag every tenant's report.
            estimated = estimated or bool(inp or out or cread or cwrite)

    return Spend(
        tenant_slug=slug,
        investigations=counts["all"],
        completed=counts["completed"],
        failed=counts["failed"],
        cancelled=counts["cancelled"],
        input_tokens=totals["input"],
        output_tokens=totals["output"],
        cache_read_tokens=totals["cache_read"],
        cache_write_tokens=totals["cache_write"],
        usd=round(usd, 4),
        estimated=estimated,
    )
