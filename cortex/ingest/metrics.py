"""Writing metric points idempotently.

A nightly sync re-reads an overlapping window on purpose, so the same observation arrives
more than once and the write path has to be indifferent to that. `ON CONFLICT DO UPDATE`
rather than `DO NOTHING`, because the second arrival is often the *better* one: an
analytics upstream revises a figure as late events land, and keeping the first read would
freeze a number the source itself no longer agrees with.

The revision is not silent — `metric_points` is append-only in spirit but a point's
identity is (tenant, provider, metric, segment, timestamp), and a changed value for the
same identity is a correction, not a new observation. Storing both would make "sessions on
14 July" ambiguous, and every aggregate over it wrong.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import CredentialProvider, MetricPoint
from cortex.tenancy.context import TenantContext

#: Rows per statement. Large enough that a 90-day daily series is one round trip, small
#: enough that a parameter list stays inside Postgres's limit.
_CHUNK = 500


@dataclass(frozen=True, slots=True)
class Point:
    """One observation, before it is written."""

    metric: str
    observed_at: datetime
    value: float
    segment: dict[str, Any] = field(default_factory=dict)

    @property
    def segment_key(self) -> str:
        """The segment flattened to a stable string.

        Sorted, so `{"device": "mobile", "channel": "paid"}` and the same pairs in the
        other order produce one key. Without sorting, the uniqueness constraint would let
        the same segment be inserted twice under two spellings, and a total over the
        series would double-count it.
        """
        return ",".join(f"{k}={v}" for k, v in sorted(self.segment.items()))


async def write_points(
    session: AsyncSession,
    tenant: TenantContext,
    *,
    provider: CredentialProvider,
    points: Sequence[Point],
) -> int:
    """Upsert points, returning how many rows were written or corrected."""
    if not points:
        return 0

    written = 0
    for chunk in _chunks(points, _CHUNK):
        rows = [
            {
                "tenant_id": tenant.tenant_id,
                "provider": provider,
                "metric": point.metric[:128],
                "segment_key": point.segment_key[:256],
                "segment": point.segment,
                "observed_at": point.observed_at,
                "value": float(point.value),
            }
            for point in chunk
        ]
        statement = insert(MetricPoint).values(rows)
        statement = statement.on_conflict_do_update(
            constraint="uq_metric_points_identity",
            # Only the value: the identity columns are what conflicted, and the segment
            # dict is derived from the key that is already part of it.
            set_={"value": statement.excluded.value},
        )
        result = await session.execute(statement)
        written += result.rowcount or 0
    return written


def _chunks(items: Sequence[Point], size: int) -> Iterable[Sequence[Point]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
