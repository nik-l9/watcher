"""`python -m cortex.spend` — what tenants are costing.

The operator's cross-tenant view. Deliberately a CLI rather than an endpoint: the HTTP `/spend`
route is scoped to the caller's own tenant, so there is no version of it that can read across
tenants. A cross-tenant figure is an operator question answered against the database, and
keeping it off the API means no authorisation mistake can expose one tenant's spend to another.

Runs in a **read-only transaction** (`cortex.runtime.readonly`), so a reporting script cannot
write even by accident. That is not theoretical caution: an earlier probe script against this
database deleted 2,231 metric points because it called something that commits per batch.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from sqlalchemy import select

from cortex.db.models import Tenant
from cortex.db.spend import spend_by_tenant, spend_for_tenant
from cortex.runtime.readonly import read_only_session
from cortex.runtime.resources import open_resources


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="watcher spend", description=__doc__)
    parser.add_argument(
        "--tenant",
        help="One tenant's slug. Omit for every active tenant, most expensive first.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Trailing window in days (default 30).",
    )
    return parser.parse_args(argv)


async def _main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if not 1 <= args.days <= 365:
        sys.stderr.write("--days must be between 1 and 365.\n")
        return 2

    async with open_resources() as resources:
        async with read_only_session(resources) as session:
            if args.tenant:
                tenant_id = await session.scalar(
                    select(Tenant.id).where(Tenant.slug == args.tenant)
                )
                if tenant_id is None:
                    sys.stderr.write(f"No tenant named {args.tenant!r}.\n")
                    return 2
                spends = [await spend_for_tenant(session, tenant_id=tenant_id, days=args.days)]
                spends = [spend for spend in spends if spend is not None]
            else:
                spends = await spend_by_tenant(session, days=args.days)

    if not spends:
        print(f"No investigations in the last {args.days} day(s).")
        return 0

    print(f"Spend over the last {args.days} day(s):\n")
    header = (
        f"  {'tenant':20} {'runs':>5} {'ok':>4} {'fail':>5} {'cxl':>4} "
        f"{'USD':>9} {'$/run':>8}  cache"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    total = 0.0
    for spend in spends:
        total += spend.usd
        # The estimate marker is printed on the row rather than in a footnote, because a
        # footnote is read after the number has already been believed.
        mark = " (est)" if spend.estimated else ""
        print(
            f"  {spend.tenant_slug[:20]:20} {spend.investigations:>5} {spend.completed:>4} "
            f"{spend.failed:>5} {spend.cancelled:>4} {spend.usd:>9.4f} "
            f"{spend.usd_per_investigation:>8.4f}  {spend.cache_hit_rate:>5.0%}{mark}"
        )
    if len(spends) > 1:
        print(f"\n  {'total':20} {'':5} {'':4} {'':5} {'':4} {total:>9.4f}")

    if any(spend.estimated for spend in spends):
        print(
            "\n  (est) at least one investigation carried no model name and was priced at the "
            "highest rate we know.\n        Over-counting is the deliberate direction: nobody "
            "investigates a number that looks fine."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
