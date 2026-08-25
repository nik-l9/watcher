"""`python -m cortex.ingest` — run one tenant's sync now, and show what it wrote.

The nightly job runs this through Celery at 02:00. This is the same code path invoked
directly, which matters for two reasons: a first sync should be run and *looked at* before
it is left to a scheduler, and a partial or failed stream is much easier to diagnose from a
terminal than from a worker log.

Prints what each stream wrote and what it could not. A stream that succeeded with a note is
the interesting case — it means the data is real but incomplete, which is exactly what a
report has to disclose and what a silent success would hide.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from cortex.db.models import CredentialProvider, SyncState, Tenant
from cortex.ingest.dispatch import nightly_targets, supported_providers, syncer_for
from cortex.ingest.runner import IngestRunner
from cortex.ingest.state import Window, is_stale
from cortex.runtime.resources import open_resources
from cortex.tenancy.context import TenantContext


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="cortex.ingest", description=__doc__)
    parser.add_argument("--tenant", help="Tenant slug. Omit with --targets.")
    parser.add_argument(
        "--provider",
        choices=supported_providers(),
        help="Which connector to sync. Only providers with a syncer are listed; the "
        "others are read live during an investigation.",
    )
    parser.add_argument("--label", default="default")
    parser.add_argument(
        "--since",
        help="Backfill from this date (YYYY-MM-DD), ignoring the stored watermark. Use "
        "when a stream needs re-reading after a mapping change.",
    )
    parser.add_argument(
        "--targets",
        action="store_true",
        help="Show what tonight's run would dispatch, and exit without syncing.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show each stream's watermark, health and staleness, and exit.",
    )
    return parser.parse_args(argv)


async def _main(argv: list[str] | None = None) -> int:
    args = _parse(argv)

    async with open_resources() as resources:
        maker = async_sessionmaker(resources.engine, expire_on_commit=False)

        if args.targets:
            async with maker() as session:
                for target in await nightly_targets(session):
                    print(f"{target.tenant_slug:20} {target.provider.value:10} {target.label}")
            return 0

        if not args.tenant:
            sys.stderr.write("--tenant is required unless --targets is given.\n")
            return 2

        async with maker() as session:
            tenant = await _tenant(session, args.tenant)
            if tenant is None:
                sys.stderr.write(f"No tenant named {args.tenant!r}.\n")
                return 2

            if args.status:
                await _print_status(session, tenant)
                return 0

            if not args.provider:
                sys.stderr.write("--provider is required to sync. Try --status first.\n")
                return 2

            provider = CredentialProvider(args.provider)
            now = datetime.now(UTC)
            window = None
            if args.since:
                window = Window(
                    since=datetime.fromisoformat(args.since).replace(tzinfo=UTC),
                    until=now,
                    first_sync=True,
                )
                print(f"Backfilling from {args.since}, ignoring the stored watermark.")

            # Provisioned before writing rather than assumed: a tenant created before the
            # vector store existed has a graph and no collections, and an upsert into a
            # collection that does not exist fails the whole stream.
            await resources.graph.provision(tenant)

            # Semantic memory is optional for a sync; the graph is not. The first real run
            # against a managed Qdrant cluster aborted here on a transport error, losing a
            # sync whose graph writes would have succeeded — so an unreachable vector store
            # now degrades the run instead of ending it, and says so.
            vectors = resources.vectors
            try:
                await vectors.provision(tenant)
            except Exception as exc:  # noqa: BLE001 - reported below, not swallowed
                print(
                    f"Semantic memory unavailable ({type(exc).__name__}: {exc}); "
                    "syncing the graph and metrics only."
                )
                vectors = None

            runner = IngestRunner(resources.graph, vectors)
            outcome = await runner.run(
                session,
                tenant,
                syncer_for(provider),
                label=args.label,
                now=now,
                window=window,
            )
            await session.commit()

        _render(outcome)
        # Zero even when a stream failed: the failure is recorded and printed, and a
        # non-zero exit would make a partially successful nightly run look like an outage.
        return 0


async def _tenant(session: object, slug: str) -> TenantContext | None:
    row = (
        await session.execute(select(Tenant).where(Tenant.slug == slug))  # type: ignore[attr-defined]
    ).scalar_one_or_none()
    if row is None:
        return None
    return TenantContext(tenant_id=row.id, tenant_slug=row.slug, graph_name=row.graph_name)


async def _print_status(session: object, tenant: TenantContext) -> None:
    rows = (
        (
            await session.execute(  # type: ignore[attr-defined]
                select(SyncState)
                .where(SyncState.tenant_id == tenant.tenant_id)
                .order_by(SyncState.provider, SyncState.stream)
            )
        )
        .scalars()
        .all()
    )
    if not rows:
        print(f"Tenant {tenant.tenant_slug} has never been synced.")
        return

    print(f"Tenant {tenant.tenant_slug}:")
    print(f"  {'provider':10} {'stream':14} {'status':8} {'items':>6}  watermark")
    for row in rows:
        stale = " STALE" if is_stale(row) else ""
        watermark = row.watermark.isoformat(timespec="minutes") if row.watermark else "-"
        print(
            f"  {row.provider.value:10} {row.stream:14} {row.status.value:8} "
            f"{row.items_written:>6}  {watermark}{stale}"
        )
        if row.detail:
            print(f"      {row.detail[:160]}")


def _render(outcome: object) -> None:
    print(f"\n{outcome.provider} ({outcome.label}) — {'ok' if outcome.succeeded else 'PROBLEMS'}")  # type: ignore[attr-defined]
    for stream in outcome.streams:  # type: ignore[attr-defined]
        mark = " " if stream.succeeded else "!"
        print(
            f" {mark} {stream.stream:14} nodes={stream.nodes:<5} edges={stream.edges:<5} "
            f"docs={stream.documents:<5} points={stream.points:<5}"
        )
        if stream.detail:
            print(f"      {stream.detail[:300]}")


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
