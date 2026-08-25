"""PostHog ingest — the metric baselines, and the change record beside them.

This is the syncer that makes "is this week unusual" answerable. A live API call can tell
you last week's signups; only a stored series can tell you that last week was two standard
deviations below the previous eight. Baselines, anomaly detection and the weekly brief all
rest on this table, and none of them are possible from live calls alone.

Two streams:

  - `events` — a daily count per event per project, written as metric points.
  - `annotations` — PostHog's own change log, which is the one place a deploy marker and a
    human note about a metric live on the same timeline as the metric.

**Which events, and why it is bounded.** The event list is discovered rather than declared,
because an operator should not have to enumerate a product's telemetry to get baselines,
and a new event should appear in the series without a config change. But it is capped:
PostHog's `event_definitions` returns hundreds of events for a mature project, one trend
query each, and a nightly job that issues 300 HogQL queries per project per tenant is a
cost incident rather than a sync. `$autocapture` and friends are excluded for the same
reason — a pageview count is not a GTM metric and would dominate the table.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import CredentialProvider
from cortex.ingest.base import StreamResult, Syncer
from cortex.ingest.metrics import Point
from cortex.ingest.state import Window
from cortex.memory.entities import Node, NodeLabel
from cortex.memory.graph_store import GraphStore
from cortex.memory.vector_store import Document
from cortex.tenancy.context import TenantContext
from cortex.tools.base import ToolContext, ToolError
from cortex.tools.posthog import PostHogTool

#: Events per project per night.
#:
#: Twelve, chosen against a real deployment's three projects, which defined 14, 36 and 58
#: custom events. Twelve covers the funnel of the smallest and the top of the others, at twelve
#: HogQL queries per project rather than sixty.
_MAX_EVENTS_PER_PROJECT = 12

#: PostHog reserves the `$` prefix for its own events, so all of them are excluded.
#:
#: This started as a hand-written list of prefixes and that was the wrong shape. The first
#: real run spent three of twelve slots per project on `$set`, `$identify` and
#: `$groupidentify` — person-property bookkeeping, not metrics — which crowded out real
#: events under the cap. Enumerating system events means the list is always one PostHog
#: release behind; the prefix is the actual rule, and a custom event never starts with `$`.
#:
#: What this deliberately excludes along with the noise is `$pageview`. It is a real metric,
#: and `ga4` already answers page-level questions better; a pageview series would also
#: outnumber every meaningful metric here by two orders of magnitude.
_RESERVED_PREFIX = "$"


class PostHogSyncer(Syncer):
    provider = CredentialProvider.POSTHOG
    tool_name = "posthog"

    def __init__(self, tool: PostHogTool | None = None) -> None:
        self._tool = tool or PostHogTool()

    @property
    def streams(self) -> tuple[str, ...]:
        return ("events", "annotations")

    async def sync_stream(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        ctx: ToolContext,
        *,
        stream: str,
        window: Window,
        graph: GraphStore,
    ) -> StreamResult:
        del session, tenant, graph
        result = StreamResult()
        gaps: list[str] = []
        notes: list[str] = []

        projects = (await self._tool.list_projects(ctx)).payload.get("projects", [])
        for project in projects:
            project_id = str(project.get("id"))
            label = str(project.get("label") or project_id)
            try:
                if stream == "events":
                    note = await self._events(ctx, project_id, label, window, result)
                elif stream == "annotations":
                    note = await self._annotations(ctx, project_id, label, result)
                else:  # pragma: no cover - the runner only passes declared streams
                    raise ValueError(f"posthog syncer has no stream {stream!r}")
                if note:
                    # An emptiness that is real, not a gap in our collection: every one of
                    # the five real projects has zero annotations, and calling that
                    # "incomplete" would tell a reader their change log might be missing
                    # entries when nobody writes any.
                    notes.append(note)
            except ToolError as exc:
                # One project failing must not lose the others. A tenant with three
                # products should still get baselines for two of them, and the third's
                # absence is a genuine gap.
                gaps.append(f"{label}: {type(exc).__name__}: {exc}")

        result.partial = "; ".join(gaps) or None
        result.note = "; ".join(notes) or None
        return result

    # ------------------------------------------------------------------ streams

    async def _events(
        self,
        ctx: ToolContext,
        project_id: str,
        label: str,
        window: Window,
        out: StreamResult,
    ) -> str | None:
        names = await self._event_names(ctx, project_id)
        if not names:
            # A project with no custom events is a real state — Staging and the CLI project
            # both look like this — and saying so beats an empty series that reads as a
            # product nobody uses.
            return f"{label}: no custom events defined"

        for name in names:
            trend = (
                await self._tool.event_trend(
                    ctx,
                    event=name,
                    start_date=window.since_date,
                    end_date=window.until_date,
                    project=project_id,
                    interval="day",
                )
            ).payload
            for row in trend.get("series", []):
                observed = _timestamp(row.get("bucket"))
                if observed is None:
                    continue
                out.points.append(
                    Point(
                        metric=name,
                        observed_at=observed,
                        value=float(row.get("value") or 0),
                        # The project is part of the segment, not the metric name: the same
                        # event exists in two products with different volumes, and merging
                        # them into one series would produce a baseline that describes
                        # neither.
                        segment={"project": label},
                    )
                )
        return None

    async def _event_names(self, ctx: ToolContext, project_id: str) -> list[str]:
        """The events worth a nightly series, most-recently-seen first.

        Ordered by recency rather than alphabetically so the cap keeps what is live: an
        event nobody has fired since March is not what a baseline is for.
        """
        payload = (await self._tool.list_events(ctx, project=project_id, limit=200)).payload
        candidates = [
            row
            for row in payload.get("events", [])
            if isinstance(row.get("name"), str)
            and not str(row["name"]).startswith(_RESERVED_PREFIX)
        ]
        candidates.sort(key=lambda row: str(row.get("last_seen_at") or ""), reverse=True)
        return [str(row["name"]) for row in candidates[:_MAX_EVENTS_PER_PROJECT]]

    async def _annotations(
        self, ctx: ToolContext, project_id: str, label: str, out: StreamResult
    ) -> str | None:
        payload = (await self._tool.annotations(ctx, project=project_id, limit=100)).payload
        items = payload.get("annotations", [])
        if not items:
            # Verified against all five real projects: they are empty. Recorded rather
            # than left implicit, so the absence is a known fact about the tenant instead
            # of a suspected gap in the sync.
            return f"{label}: no annotations"

        for item in items:
            marker = str(item.get("date_marker") or "")
            content = str(item.get("content") or "").strip()
            if not (marker and content):
                continue
            key = f"{label}:{marker}"
            # A Decision, because that is what an annotation is: somebody recording that a
            # thing happened and mattered. A deployment marker is the automated form of the
            # same statement, and `source` keeps the two distinguishable — one is testimony
            # and the other is a record of a ship.
            out.nodes.append(
                Node(
                    NodeLabel.DECISION,
                    key,
                    {
                        "content": content[:500],
                        "at": marker,
                        "source": item.get("source"),
                        "project": label,
                        "created_by": item.get("created_by"),
                    },
                )
            )
            out.documents.append(
                Document(
                    kind="notes",
                    source_id=key,
                    text=content[:2000],
                    source_ref=f"posthog://projects/{project_id}/annotations",
                    metadata={
                        "project": label,
                        "at": marker,
                        "entity_label": NodeLabel.DECISION.value,
                        "entity_key": key,
                    },
                )
            )
        return None


def _timestamp(value: Any) -> datetime | None:
    """Parse a HogQL bucket into an aware datetime.

    Returns None rather than raising: one unparseable bucket should cost one point, not the
    whole stream, and a series with a gap is more useful than no series.
    """
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
