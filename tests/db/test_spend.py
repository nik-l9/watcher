"""What a tenant has cost.

The interesting tests are not "does it add up". They are about the two ways a spend figure goes
wrong quietly:

  - **Pricing the wrong thing.** `tokens_used` was a single total, and input and output differ
    by 5x on every model we run while a cache read costs a tenth of a fresh input token.
    Against a measured 98% cache-hit rate, a bill computed from the total overstates it by
    roughly an order of magnitude. This is why the breakdown needed a migration before the
    aggregation could exist at all.
  - **Under-counting invisibly.** An unpriced model, a path that never records usage, another
    tenant's rows leaking in. Each produces a number that looks plausible, and nobody
    investigates a number that looks fine.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import Investigation, InvestigationStatus, Tenant
from cortex.db.spend import RATES, cost_usd, spend_by_tenant, spend_for_tenant
from cortex.memory.naming import graph_name_for_new_tenant

NOW = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)


async def _tenant(session: AsyncSession, slug: str, *, active: bool = True) -> uuid.UUID:
    tenant_id = uuid.uuid4()
    session.add(
        Tenant(
            id=tenant_id,
            slug=slug,
            name=slug,
            graph_name=graph_name_for_new_tenant(slug, tenant_id),
            is_active=active,
        )
    )
    await session.flush()
    return tenant_id


async def _investigation(
    session: AsyncSession,
    tenant_id: uuid.UUID,
    *,
    model: str | None = "claude-sonnet-5",
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
    status: InvestigationStatus = InvestigationStatus.COMPLETED,
    age_days: int = 1,
) -> None:
    session.add(
        Investigation(
            tenant_id=tenant_id,
            question="why did signups fall?",
            status=status,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            tokens_used=input_tokens + output_tokens,
            created_at=NOW - timedelta(days=age_days),
        )
    )
    await session.flush()


class TestPricing:
    def test_input_and_output_are_priced_differently(self) -> None:
        """5x apart on every model we run. A bill from the total is wrong by a multiple that
        depends on the mix."""
        input_rate, output_rate = RATES["claude-sonnet-5"]
        assert output_rate == input_rate * 5

        all_input = cost_usd("claude-sonnet-5", input_tokens=1_000_000, output_tokens=0)
        all_output = cost_usd("claude-sonnet-5", input_tokens=0, output_tokens=1_000_000)
        assert all_input == 3.0
        assert all_output == 15.0

    def test_a_cache_read_costs_a_fraction_of_a_fresh_input_token(self) -> None:
        """The reason this matters here specifically: we measured a 98% cache-hit rate, so
        cached reads are the overwhelming majority of what is sent. Charging them at the
        input rate would overstate the bill by nearly an order of magnitude."""
        fresh = cost_usd("claude-sonnet-5", input_tokens=1_000_000, output_tokens=0)
        cached = cost_usd(
            "claude-sonnet-5", input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000
        )
        assert cached < fresh
        assert abs(cached - fresh * 0.10) < 1e-9

    def test_a_cache_write_costs_a_premium(self) -> None:
        fresh = cost_usd("claude-sonnet-5", input_tokens=1_000_000, output_tokens=0)
        written = cost_usd(
            "claude-sonnet-5", input_tokens=0, output_tokens=0, cache_write_tokens=1_000_000
        )
        assert written > fresh

    def test_an_unknown_model_bills_at_the_highest_rate_we_know(self) -> None:
        """Over-counting is the deliberate direction. A spend report that silently
        under-counts is worse than one that visibly over-counts, because nobody investigates
        a number that looks fine."""
        unknown = cost_usd("some-new-model", input_tokens=1_000_000, output_tokens=0)
        most_expensive = max(rate for rate, _ in RATES.values())
        assert unknown == most_expensive

    def test_a_missing_model_is_priced_rather_than_skipped(self) -> None:
        """A row with no model must not cost zero: zero is indistinguishable from a free
        investigation, and it is the one value that hides the problem."""
        assert cost_usd(None, input_tokens=1_000_000, output_tokens=0) > 0


class TestAggregation:
    async def test_a_tenants_spend_is_summed_and_counted_by_status(
        self, session: AsyncSession
    ) -> None:
        tenant_id = await _tenant(session, "spend-one")
        await _investigation(session, tenant_id, input_tokens=1_000_000, output_tokens=200_000)
        await _investigation(
            session, tenant_id, output_tokens=100_000, status=InvestigationStatus.FAILED
        )
        await _investigation(
            session, tenant_id, output_tokens=50_000, status=InvestigationStatus.CANCELLED
        )
        await session.commit()

        spend = await spend_for_tenant(session, tenant_id=tenant_id, days=30, now=NOW)

        assert spend is not None
        assert spend.investigations == 3
        assert spend.completed == 1
        assert spend.failed == 1
        assert spend.cancelled == 1
        # 3.00 + 3.00 (output 200k at 15) + 1.50 + 0.75
        assert spend.usd == 8.25
        assert spend.usd_per_investigation == 2.75

    async def test_a_failed_investigation_still_costs(self, session: AsyncSession) -> None:
        """It spent the tokens. A spend view that only counted successes would understate the
        bill by exactly the amount nobody wants to pay for."""
        tenant_id = await _tenant(session, "spend-failed")
        await _investigation(
            session, tenant_id, output_tokens=1_000_000, status=InvestigationStatus.FAILED
        )
        await session.commit()

        spend = await spend_for_tenant(session, tenant_id=tenant_id, days=30, now=NOW)
        assert spend is not None and spend.usd == 15.0

    async def test_another_tenants_spend_is_not_included(self, session: AsyncSession) -> None:
        mine = await _tenant(session, "spend-mine")
        theirs = await _tenant(session, "spend-theirs")
        await _investigation(session, mine, output_tokens=100_000)
        await _investigation(session, theirs, output_tokens=900_000)
        await session.commit()

        spend = await spend_for_tenant(session, tenant_id=mine, days=30, now=NOW)
        assert spend is not None and spend.usd == 1.5

    async def test_two_models_are_priced_separately(self, session: AsyncSession) -> None:
        """Summing tokens across models and applying one rate would be wrong the moment a
        second model is used, and wrong silently."""
        tenant_id = await _tenant(session, "spend-models")
        await _investigation(session, tenant_id, model="claude-sonnet-5", output_tokens=1_000_000)
        await _investigation(session, tenant_id, model="claude-haiku-4-5", output_tokens=1_000_000)
        await session.commit()

        spend = await spend_for_tenant(session, tenant_id=tenant_id, days=30, now=NOW)
        # 15.00 at sonnet plus 5.00 at haiku, not 2 million tokens at either rate.
        assert spend is not None and spend.usd == 20.0

    async def test_the_window_excludes_older_investigations(self, session: AsyncSession) -> None:
        tenant_id = await _tenant(session, "spend-window")
        await _investigation(session, tenant_id, output_tokens=1_000_000, age_days=2)
        await _investigation(session, tenant_id, output_tokens=1_000_000, age_days=60)
        await session.commit()

        spend = await spend_for_tenant(session, tenant_id=tenant_id, days=30, now=NOW)
        assert spend is not None and spend.usd == 15.0

    async def test_an_unknown_tenant_is_none_not_zero(self, session: AsyncSession) -> None:
        """Zero would read as a tenant that costs nothing, which is a different fact from a
        tenant that does not exist."""
        assert await spend_for_tenant(session, tenant_id=uuid.uuid4(), days=30, now=NOW) is None


class TestTheEstimateFlag:
    async def test_a_missing_model_marks_the_figure_as_estimated(
        self, session: AsyncSession
    ) -> None:
        """An estimate presented as a measurement is the failure this codebase keeps
        guarding against."""
        tenant_id = await _tenant(session, "spend-est")
        await _investigation(session, tenant_id, model=None, output_tokens=100_000)
        await session.commit()

        spend = await spend_for_tenant(session, tenant_id=tenant_id, days=30, now=NOW)
        assert spend is not None and spend.estimated is True

    async def test_a_zero_token_row_with_no_model_does_not_flag_the_report(
        self, session: AsyncSession
    ) -> None:
        """Historical rows predate the breakdown columns and carry zeros. Flagging every
        tenant's report because of them would make the marker meaningless."""
        tenant_id = await _tenant(session, "spend-zero")
        await _investigation(session, tenant_id, model=None)
        await _investigation(session, tenant_id, model="claude-sonnet-5", output_tokens=100_000)
        await session.commit()

        spend = await spend_for_tenant(session, tenant_id=tenant_id, days=30, now=NOW)
        assert spend is not None and spend.estimated is False

    async def test_known_models_are_not_marked(self, session: AsyncSession) -> None:
        tenant_id = await _tenant(session, "spend-known")
        await _investigation(session, tenant_id, output_tokens=100_000)
        await session.commit()

        spend = await spend_for_tenant(session, tenant_id=tenant_id, days=30, now=NOW)
        assert spend is not None and spend.estimated is False


class TestTheOperatorView:
    async def test_tenants_are_ordered_by_cost(self, session: AsyncSession) -> None:
        """The reason to open this is to find out who is expensive."""
        cheap = await _tenant(session, "spend-cheap")
        pricey = await _tenant(session, "spend-pricey")
        await _investigation(session, cheap, output_tokens=100_000)
        await _investigation(session, pricey, output_tokens=900_000)
        await session.commit()

        spends = await spend_by_tenant(session, days=30, now=NOW)
        slugs = [spend.tenant_slug for spend in spends]
        assert slugs.index("spend-pricey") < slugs.index("spend-cheap")

    async def test_a_suspended_tenant_is_excluded(self, session: AsyncSession) -> None:
        suspended = await _tenant(session, "spend-suspended", active=False)
        await _investigation(session, suspended, output_tokens=1_000_000)
        await session.commit()

        slugs = {spend.tenant_slug for spend in await spend_by_tenant(session, days=30, now=NOW)}
        assert "spend-suspended" not in slugs

    async def test_the_cache_hit_rate_is_reported(self, session: AsyncSession) -> None:
        """The number that explains why the bill is lower than the token count suggests."""
        tenant_id = await _tenant(session, "spend-cache")
        await _investigation(
            session, tenant_id, input_tokens=2_000, cache_read=98_000, output_tokens=1_000
        )
        await session.commit()

        spend = await spend_for_tenant(session, tenant_id=tenant_id, days=30, now=NOW)
        assert spend is not None
        assert abs(spend.cache_hit_rate - 0.98) < 0.001
