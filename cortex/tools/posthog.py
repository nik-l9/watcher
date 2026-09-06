"""PostHog connector — product analytics, and the change record that sits beside it.

PostHog is the most causally useful source Cortex has, for a reason none of the others
share: the metrics and the record of *what changed* live in the same system, on the same
timeline. GA4 can tell you conversion fell but knows nothing about why; establishing a
cause there means joining GitHub deploys and Slack chatter by timestamp and hoping the
clocks agree. Here, annotations, feature-flag rollouts and experiments are first-class
objects next to the events they affected.

Three capabilities exist purely to answer "what changed":

  - `annotations` — PostHog's own change log. `creation_type` distinguishes a human note
    (`USR`) from an automated deployment marker (`GIT`), so a deploy timeline is available
    without leaving the product.
  - `feature_flags` — a rollout is a change event. "Conversion fell for users in this
    flag" is a root cause, and it is invisible to every other connector.
  - `experiments` — a running A/B test explains a metric move and is otherwise
    indistinguishable from an unexplained regression.

**No agent-authored SQL.** PostHog's query endpoint takes HogQL, which is arbitrary SQL
over production event data. The plan forbids agent-written SQL against production
warehouses, and the reason applies with equal force here: a generated query can be
expensive, can scan far more than intended, and cannot be reviewed before it runs. So
HogQL is *composed here* from typed, validated parameters — the model chooses an event, a
date range and a breakdown, and this module writes the SQL. Event and property names are
checked against the project's own definitions before interpolation, which is also what
makes an invented event name fail with a correctable message instead of an opaque 400.
"""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from cortex.analysis.coverage import gap_disclosure
from cortex.db.models import CredentialProvider
from cortex.tools.base import (
    NEVER_EMPTY,
    Capability,
    Freshness,
    InvalidParams,
    ToolContext,
    ToolError,
    ToolResult,
)
from cortex.tools.base import Tool as BaseTool
from cortex.tools.disclosures import grade_series, series_disclosures
from cortex.tools.http import DEFAULT_TIMEOUT, request_json

#: The qualified capability name the shared assembler dispatches on.
_EVENT_TREND = "posthog__event_trend"

#: PostHog is regionally partitioned and the regions do not share data. A key issued in
#: the EU returns 401 against the US host, which looks exactly like a bad key — so the
#: host is configuration, and an unknown one is rejected rather than guessed at.
KNOWN_HOSTS = (
    "https://us.posthog.com",
    "https://eu.posthog.com",
    "https://app.posthog.com",
)

_MAX_ROWS = 500
_DEFAULT_ROWS = 100

_DATE = {
    "type": "string",
    "pattern": r"^\d{4}-\d{2}-\d{2}$",
    "description": "Date as YYYY-MM-DD.",
}

#: Intervals HogQL's date truncation accepts. Allowlisted because this value is
#: interpolated into SQL.
INTERVALS = ("day", "week", "month")

#: Aggregations the trend capability can compute. Allowlisted for the same reason.
AGGREGATIONS = {
    "total_events": "count()",
    "unique_users": "count(distinct person_id)",
    "unique_sessions": "count(distinct $session_id)",
}

#: A conservative identifier pattern for anything interpolated into HogQL. Event and
#: property names come from the project's own definitions, but they are still checked:
#: a definitions endpoint is not a promise about quoting, and one apostrophe in an event
#: name would otherwise change the query's meaning.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_$ .:\-/]{1,200}$")

#: Optional project selector, on every capability.
#:
#: An organisation's analytics are split across projects that do not share events, so a
#: question about one product answered from another's data is silently wrong. Omitting it
#: keeps the previous behaviour: the first declared project.
_PROJECT = {
    "type": "string",
    "description": (
        "Which PostHog project to read, when the tenant has more than one (a marketing "
        "site, a SaaS product, an open-source client). Call list_projects first; omit to "
        "use the default. Events do not exist across projects, so an empty result may "
        "mean you are querying the wrong one."
    ),
}


def _reject_unsafe(value: str, *, field: str) -> str:
    if not _SAFE_NAME.match(value):
        raise InvalidParams(
            f"posthog: {field} {value!r} contains characters that cannot be used in a "
            "query. Call list_events to see the exact names this project defines."
        )
    return value


class PostHogTool(BaseTool):
    name = "posthog"
    provider = CredentialProvider.POSTHOG

    def capabilities(self) -> list[Capability]:
        return [
            Capability(
                name="list_projects",
                description=(
                    "The PostHog projects this tenant can read. Call this first when a "
                    "question might concern a different product: an organisation splits "
                    "analytics across projects — a marketing site, a SaaS product, an "
                    "open-source client — and events do not exist across them, so "
                    "querying the wrong project returns an empty result that looks "
                    "exactly like a real absence."
                ),
                params_schema={"type": "object", "additionalProperties": False, "properties": {}},
                handler=self.list_projects,
                result_key=(
                    f"{NEVER_EMPTY}: the tenant's project allowlist is validated at\n"
                    "connect time, and an empty one raises rather than returning nothing."
                ),
                # Enumerates what exists. Run before the first step -- see Capability.discovery.
                discovery=True,
            ),
            Capability(
                name="list_events",
                description=(
                    "The events this project actually defines, newest activity first. "
                    "Call this before any trend or funnel: event names are project-"
                    "specific, and a guessed name returns an empty series that looks "
                    "like a real drop to zero."
                ),
                params_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "search": {
                            "type": "string",
                            "description": "Filter by name fragment, e.g. 'signup'.",
                        },
                        "project": _PROJECT,
                        "include_stale": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "Include events not ingested in the last 30 days. If a "
                                "search returns nothing, retry with this true — the "
                                "event may exist but have stopped firing, which is "
                                "itself a finding."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": _MAX_ROWS,
                            "default": _DEFAULT_ROWS,
                        },
                    },
                },
                handler=self.list_events,
                result_key="events",
                # Enumerates what exists. Run before the first step -- see Capability.discovery.
                discovery=True,
            ),
            Capability(
                name="event_trend",
                description=(
                    "A time series for one event, optionally broken down by a property. "
                    "Use this to confirm a change is real, measure it, and find which "
                    "segment moved."
                ),
                params_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["event", "start_date", "end_date"],
                    "properties": {
                        "project": _PROJECT,
                        "event": {
                            "type": "string",
                            "description": "Event name, exactly as list_events reports it.",
                        },
                        "start_date": _DATE,
                        "end_date": _DATE,
                        "interval": {
                            "type": "string",
                            "enum": list(INTERVALS),
                            "default": "day",
                        },
                        "measure": {
                            "type": "string",
                            "enum": sorted(AGGREGATIONS),
                            "default": "total_events",
                        },
                        "breakdown_property": {
                            "type": "string",
                            "description": (
                                "Event property to split by, e.g. '$browser', "
                                "'$current_url', 'plan'. Omit for a single series."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": _MAX_ROWS,
                            "default": _DEFAULT_ROWS,
                        },
                    },
                },
                handler=self.event_trend,
                result_key="series",
            ),
            Capability(
                name="funnel",
                description=(
                    "Per-step conversion for an ordered sequence of events within a "
                    "window. Use this to locate where users drop out rather than "
                    "inferring it from separate totals."
                ),
                params_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["steps", "start_date", "end_date"],
                    "properties": {
                        "project": _PROJECT,
                        "steps": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 6,
                            "description": "Event names in the order users pass through.",
                        },
                        "start_date": _DATE,
                        "end_date": _DATE,
                        "conversion_window_days": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 30,
                            "default": 7,
                            "description": (
                                "How long a user may take to complete the sequence. A "
                                "window shorter than the real sales or onboarding cycle "
                                "reports a drop-off that is only impatience."
                            ),
                        },
                    },
                },
                handler=self.funnel,
                result_key="steps",
            ),
            Capability(
                name="annotations",
                description=(
                    "PostHog's change log: releases, incidents and notes placed on the "
                    "timeline, with who created each and whether it came from a human "
                    "or a deployment bot. This is the fastest way to find what changed "
                    "on the day a metric moved."
                ),
                params_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "project": _PROJECT,
                        "search": {"type": "string", "description": "Filter by content."},
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": _MAX_ROWS,
                            "default": 50,
                        },
                    },
                },
                handler=self.annotations,
                result_key="annotations",
            ),
            Capability(
                name="feature_flags",
                description=(
                    "Feature flags and their rollout state. A flag going live is a "
                    "change event: if a metric moved for one cohort and not another, "
                    "the flag serving them is a candidate cause no other source can see."
                ),
                params_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "project": _PROJECT,
                        "search": {
                            "type": "string",
                            "description": "Filter by flag key or name.",
                        },
                        "active_only": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "Only currently-active flags. Leave false when "
                                "investigating a past change — a flag switched off "
                                "after an incident is exactly what you are looking for."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": _MAX_ROWS,
                            "default": 50,
                        },
                    },
                },
                handler=self.feature_flags,
                result_key="flags",
            ),
            Capability(
                name="experiments",
                description=(
                    "Running and completed experiments. An in-flight A/B test moves "
                    "aggregate metrics by design, and mistaking that for a regression "
                    "is a common false positive."
                ),
                params_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "project": _PROJECT,
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": _MAX_ROWS,
                            "default": 50,
                        },
                    },
                },
                handler=self.experiments,
                result_key="experiments",
            ),
        ]

    # ----------------------------------------------------------------- credentials

    @staticmethod
    def _projects(ctx: ToolContext) -> dict[str, str]:
        """Projects this credential may read, as id -> label.

        An organisation splits its analytics across projects — a marketing site, a SaaS
        product, an open-source client — and they do not share events. Pinning one meant a
        question about a different product was answered from the wrong data without anyone
        noticing, which is worse than failing.

        Declared as an allowlist rather than "whatever the key can reach", so widening
        access is a deliberate act at connect time and a model cannot wander into a
        project nobody meant to expose.
        """
        raw = str(ctx.credential_metadata.get("projects", "")).strip()
        mapping: dict[str, str] = {}
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            # `id:label` for readability in a tool listing; a bare id is also accepted.
            pid, _, label = entry.partition(":")
            if pid.strip().isdigit():
                mapping[pid.strip()] = label.strip() or pid.strip()

        legacy = str(ctx.credential_metadata.get("project_id", "")).strip()
        if legacy.isdigit():
            mapping.setdefault(legacy, "default")

        if not mapping:
            raise InvalidParams(
                "posthog: this tenant's credential lists no projects. Reconnect with "
                "--meta projects=100001:web,100002:oss-client (ids from the PostHog "
                "project URL)."
            )
        return mapping

    def _project_id(self, ctx: ToolContext, requested: str | None = None) -> str:
        projects = self._projects(ctx)
        if requested is None:
            # The first declared project is the default, so a single-project tenant and
            # every existing call behave exactly as before.
            return next(iter(projects))
        wanted = str(requested).strip()
        if wanted not in projects:
            raise InvalidParams(
                f"posthog: project {wanted!r} is not available to this tenant. "
                f"Available: {', '.join(f'{k} ({v})' for k, v in projects.items())}"
            )
        return wanted

    @staticmethod
    def _host(ctx: ToolContext) -> str:
        raw = str(ctx.credential_metadata.get("host", "")).strip().rstrip("/")
        if not raw:
            # US cloud is the default region, and stating the assumption beats guessing
            # silently — a wrong region returns 401, which reads as a bad key.
            return KNOWN_HOSTS[0]
        if raw not in KNOWN_HOSTS and not raw.startswith("https://"):
            raise InvalidParams(
                f"posthog: host {raw!r} is not a recognised PostHog host and is not "
                "https. Use one of: " + ", ".join(KNOWN_HOSTS)
            )
        return raw

    async def _client(self, ctx: ToolContext) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._host(ctx),
            headers={
                "Authorization": f"Bearer {ctx.credential}",
                "Content-Type": "application/json",
            },
            timeout=DEFAULT_TIMEOUT,
        )

    # ---------------------------------------------------------------------- queries

    async def _query(
        self, ctx: ToolContext, hogql: str, project: str | None = None
    ) -> dict[str, Any]:
        """Run one HogQL statement composed by this module, never by the model."""
        async with await self._client(ctx) as client:
            return await request_json(
                client,
                "POST",
                f"/api/projects/{self._project_id(ctx, project)}/query/",
                tool=self.name,
                json_body={"query": {"kind": "HogQLQuery", "query": hogql}},
            )

    async def _get(
        self,
        ctx: ToolContext,
        path: str,
        params: dict[str, Any] | None = None,
        project: str | None = None,
    ) -> dict[str, Any]:
        async with await self._client(ctx) as client:
            return await request_json(
                client,
                "GET",
                f"/api/projects/{self._project_id(ctx, project)}/{path}",
                tool=self.name,
                params=params,
            )

    # ------------------------------------------------------------------ capabilities

    async def list_projects(self, ctx: ToolContext) -> ToolResult:
        projects = self._projects(ctx)
        default = next(iter(projects))
        return ToolResult(
            payload={
                "project_count": len(projects),
                "default_project": default,
                "projects": [
                    {"id": pid, "label": label, "is_default": pid == default}
                    for pid, label in projects.items()
                ],
            },
            source_ref="posthog://projects",
            meta={"freshness": Freshness.LIVE},
        )

    async def list_events(
        self,
        ctx: ToolContext,
        *,
        project: str | None = None,
        search: str | None = None,
        include_stale: bool = False,
        limit: int = _DEFAULT_ROWS,
    ) -> ToolResult:
        params: dict[str, Any] = {"limit": limit, "exclude_stale": not include_stale}
        if search:
            params["search"] = search
        raw = await self._get(ctx, "event_definitions/", params, project=project)
        events = [
            {
                "name": row.get("name"),
                "description": row.get("description"),
                "last_seen_at": row.get("last_seen_at"),
                "verified": bool(row.get("verified")),
            }
            for row in raw.get("results", [])
        ]
        return ToolResult(
            payload={
                "event_count": len(events),
                "total_available": raw.get("count"),
                "excluded_stale": not include_stale,
                "events": events,
            },
            source_ref=f"posthog://projects/{self._project_id(ctx, project)}/event_definitions",
            meta={"freshness": Freshness.LIVE},
        )

    async def _sibling_context(
        self,
        ctx: ToolContext,
        project: str | None,
        *,
        event: str,
        gap: dict[str, Any] | None,
        moved: bool,
    ) -> dict[str, Any]:
        """What the project's other event definitions say about this series.

        Two disclosures from one request, each fetched only when it could matter:

        **`blast_radius`**, when the series ended early — did this event stop alone, or did a
        group stop together? `include_stale` is off-by-default elsewhere and forced on here: an
        event that stopped six weeks ago is precisely what this is looking for.

        **`related_events`**, when the series *moved* — is there another event measuring the
        same thing under a different name? Five attempts at one real question produced "volume
        did not fall", "it fell 78%", "it rose 18x", and twice "the premise does not hold",
        because `conversation_created` and `agent_server.conversation_created` both exist and
        whether the analyst noticed depended on whether it happened to query both. The one
        attempt that read a single series reported *"a confirmed level shift of about 78%"* --
        every figure real, correctly cited, and wrong. No grounding mechanism can see that.

        Fetched here rather than left to the investigation for the same reason `blast_radius`
        is: the analyst has to hold the movement and the sibling at the same time to interpret
        either, and in production it did not go back for the second call.

        Failure is swallowed. These enrich a disclosure; neither may be the reason a trend
        that was fetched successfully fails to return.
        """
        last_bucket = _last_seen((gap or {}).get("series_ends_early", {}).get("last_bucket"))
        if last_bucket is None and not moved:
            return {}
        try:
            raw = await self._get(
                ctx,
                "event_definitions/",
                {"limit": _DEFAULT_ROWS, "exclude_stale": False},
                project=project,
            )
        except (ToolError, httpx.HTTPError):
            return {}
        definitions = list(raw.get("results") or [])
        found: dict[str, Any] = {}
        if last_bucket is not None:
            found.update(_blast_radius(definitions, last_bucket) or {})
        if moved:
            found.update(_related_events(definitions, event) or {})
        return found

    async def event_trend(
        self,
        ctx: ToolContext,
        *,
        event: str,
        start_date: str,
        end_date: str,
        project: str | None = None,
        interval: str = "day",
        measure: str = "total_events",
        breakdown_property: str | None = None,
        limit: int = _DEFAULT_ROWS,
    ) -> ToolResult:
        _check_range(start_date, end_date)
        if interval not in INTERVALS:
            raise InvalidParams(f"posthog: interval must be one of {', '.join(INTERVALS)}")
        if measure not in AGGREGATIONS:
            raise InvalidParams(
                f"posthog: measure must be one of {', '.join(sorted(AGGREGATIONS))}"
            )
        _reject_unsafe(event, field="event")
        aggregate = AGGREGATIONS[measure]

        if breakdown_property:
            _reject_unsafe(breakdown_property, field="breakdown_property")
            select = (
                f"SELECT toStartOf{interval.title()}(timestamp) AS bucket, "
                f"properties.{_prop(breakdown_property)} AS segment, {aggregate} AS value"
            )
            group = "GROUP BY bucket, segment ORDER BY bucket, value DESC"
        else:
            select = (
                f"SELECT toStartOf{interval.title()}(timestamp) AS bucket, {aggregate} AS value"
            )
            group = "GROUP BY bucket ORDER BY bucket"

        hogql = (
            f"{select} FROM events "
            f"WHERE event = '{event}' "
            f"AND timestamp >= toDateTime('{start_date} 00:00:00') "
            f"AND timestamp <= toDateTime('{end_date} 23:59:59') "
            f"{group} LIMIT {int(limit)}"
        )
        raw = await self._query(ctx, hogql, project=project)
        rows = _rows(raw)
        gap = _series_gap(rows, end_date, interval)
        payload: dict[str, Any] = {
            "event": event,
            "measure": measure,
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date,
            "breakdown_property": breakdown_property,
            "row_count": len(rows),
            "series": rows,
            # Returned so an empty series can be told apart from a real drop to
            # zero: no rows for an event that exists is a different finding from no
            # rows because the name was wrong.
            "total": sum(_number(r.get("value")) for r in rows),
        }

        # Short buckets, a series that stops early, and where it moved -- all computable from
        # the payload and the request, so all assembled by the shared function the eval harness
        # also calls. Before that, the evaluation suite exercised none of them: it swaps in a
        # handler returning a canned payload, so this method never ran and two 10/10 runs said
        # nothing about any disclosure here.
        request = {
            "start_date": start_date,
            "end_date": end_date,
            "interval": interval,
            "breakdown_property": breakdown_property,
        }
        payload.update(series_disclosures(_EVENT_TREND, payload, request))

        # The exception, and it stays here: when the series stops early the sibling events say
        # what kind of stop it was, and that needs a second query rather than a reading of this
        # payload. Fetched now rather than left to the investigation because the analyst has to
        # hold the warning and the resolving evidence at the same time to use either, and in
        # production it did not go back for the second call. A fixture supplies `blast_radius`
        # directly, which is what this query would have returned.
        payload.update(
            await self._sibling_context(
                ctx, project, event=event, gap=gap, moved=_worth_comparing(payload)
            )
        )

        # The data-trust gate, last, because it reads the disclosures above rather than the
        # rows. ADR 0005 decision 1: a series whose events stopped together with others on the
        # same emitter cannot answer a business question, and saying so here -- in the payload,
        # beside the numbers -- is what stops the analyst reaching for the nearest falling line
        # instead. Only `broken` and `degraded` are reported; an ordinary series carries nothing.
        payload.update(grade_series(_EVENT_TREND, payload))

        return ToolResult(
            payload=payload,
            source_ref=(
                f"posthog://projects/{self._project_id(ctx, project)}/trend"
                f"?event={event}&from={start_date}&to={end_date}"
            ),
            meta={"freshness": Freshness.LIVE, "hogql": raw.get("hogql")},
        )

    async def funnel(
        self,
        ctx: ToolContext,
        *,
        steps: list[str],
        start_date: str,
        end_date: str,
        project: str | None = None,
        conversion_window_days: int = 7,
    ) -> ToolResult:
        _check_range(start_date, end_date)
        if not steps:
            raise InvalidParams("posthog: funnel needs at least one step")
        for step in steps:
            _reject_unsafe(step, field="steps")

        # One aggregate per step over the same person set, which is what makes the
        # per-step counts comparable. A separate query per step would count different
        # populations and produce conversion rates that do not compose.
        counts = ", ".join(
            f"count(distinct if(event = '{step}', person_id, NULL)) AS step_{index}"
            for index, step in enumerate(steps)
        )
        events = ", ".join(f"'{step}'" for step in steps)
        hogql = (
            f"SELECT {counts} FROM events "
            f"WHERE event IN ({events}) "
            f"AND timestamp >= toDateTime('{start_date} 00:00:00') "
            f"AND timestamp <= toDateTime('{end_date} 23:59:59')"
        )
        raw = await self._query(ctx, hogql, project=project)
        rows = _rows(raw)
        first = rows[0] if rows else {}

        entered = _number(first.get("step_0"))
        breakdown = []
        previous = entered
        for index, step in enumerate(steps):
            reached = _number(first.get(f"step_{index}"))
            breakdown.append(
                {
                    "step": index + 1,
                    "event": step,
                    "users": reached,
                    "conversion_from_previous": (
                        round(reached / previous, 4) if previous else None
                    ),
                    "conversion_from_first": round(reached / entered, 4) if entered else None,
                }
            )
            previous = reached

        return ToolResult(
            payload={
                "steps": breakdown,
                "start_date": start_date,
                "end_date": end_date,
                "conversion_window_days": conversion_window_days,
                # Disclosed rather than implied: this counts users who did each step
                # inside the window, not users who did them *in order*. Ordering needs
                # PostHog's funnel query type, and presenting an unordered count as an
                # ordered funnel would overstate what the number means.
                "ordering_enforced": False,
            },
            source_ref=(
                f"posthog://projects/{self._project_id(ctx, project)}/funnel"
                f"?steps={'>'.join(steps)}&from={start_date}&to={end_date}"
            ),
            meta={"freshness": Freshness.LIVE, "hogql": raw.get("hogql")},
        )

    async def annotations(
        self,
        ctx: ToolContext,
        *,
        project: str | None = None,
        search: str | None = None,
        limit: int = 50,
    ) -> ToolResult:
        params: dict[str, Any] = {"limit": limit}
        if search:
            params["search"] = search
        raw = await self._get(ctx, "annotations/", params, project=project)
        items = [
            {
                "content": row.get("content"),
                "date_marker": row.get("date_marker"),
                # USR is a person writing a note; GIT is a deployment marker. The
                # distinction matters: one is testimony, the other is a record of a ship.
                "source": "deployment" if row.get("creation_type") == "GIT" else "person",
                "created_by": (row.get("created_by") or {}).get("email"),
                "created_at": row.get("created_at"),
            }
            for row in raw.get("results", [])
        ]
        return ToolResult(
            payload={
                "annotation_count": len(items),
                "total_available": raw.get("count"),
                "annotations": items,
            },
            source_ref=f"posthog://projects/{self._project_id(ctx, project)}/annotations",
            meta={"freshness": Freshness.LIVE},
        )

    async def feature_flags(
        self,
        ctx: ToolContext,
        *,
        project: str | None = None,
        search: str | None = None,
        active_only: bool = False,
        limit: int = 50,
    ) -> ToolResult:
        params: dict[str, Any] = {"limit": limit}
        if search:
            params["search"] = search
        if active_only:
            params["active"] = "true"
        raw = await self._get(ctx, "feature_flags/", params, project=project)
        flags = [
            {
                "key": row.get("key"),
                "name": row.get("name"),
                "active": bool(row.get("active")),
                "type": row.get("filters", {}).get("multivariate") and "multivariate" or "boolean",
                "rollout_percentage": _rollout(row),
                "created_at": row.get("created_at"),
                "created_by": (row.get("created_by") or {}).get("email"),
            }
            for row in raw.get("results", [])
        ]
        return ToolResult(
            payload={
                "flag_count": len(flags),
                "total_available": raw.get("count"),
                "active_only": active_only,
                "flags": flags,
            },
            source_ref=f"posthog://projects/{self._project_id(ctx, project)}/feature_flags",
            meta={"freshness": Freshness.LIVE},
        )

    async def experiments(
        self, ctx: ToolContext, *, project: str | None = None, limit: int = 50
    ) -> ToolResult:
        raw = await self._get(ctx, "experiments/", {"limit": limit}, project=project)
        items = [
            {
                "name": row.get("name"),
                "feature_flag_key": row.get("feature_flag_key"),
                "start_date": row.get("start_date"),
                "end_date": row.get("end_date"),
                "archived": bool(row.get("archived")),
            }
            for row in raw.get("results", [])
        ]
        return ToolResult(
            payload={
                "experiment_count": len(items),
                "total_available": raw.get("count"),
                "experiments": items,
            },
            source_ref=f"posthog://projects/{self._project_id(ctx, project)}/experiments",
            meta={"freshness": Freshness.LIVE},
        )


def _prop(name: str) -> str:
    """Render a property access for HogQL.

    Properties beginning with `$` are PostHog's own and need quoting; a bare `$browser`
    is a syntax error.
    """
    return f'"{name}"' if name.startswith("$") or " " in name else name


def _rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Zip the query endpoint's positional results against its column names.

    The API returns `results` as arrays and `columns` separately. Returning the arrays
    as-is would hand the model tuples whose meaning depends on order — the same mistake
    that made the report schema reject positional chart points.
    """
    columns = [str(c) for c in raw.get("columns", [])]
    out: list[dict[str, Any]] = []
    for row in raw.get("results", []):
        if isinstance(row, list | tuple):
            out.append(
                {
                    columns[i] if i < len(columns) else f"col_{i}": value
                    for i, value in enumerate(row)
                }
            )
        elif isinstance(row, dict):
            out.append(row)
    return out


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _rollout(flag: dict[str, Any]) -> Any:
    groups = (flag.get("filters") or {}).get("groups") or []
    for group in groups:
        if isinstance(group, dict) and group.get("rollout_percentage") is not None:
            return group["rollout_percentage"]
    return None


def _series_gap(
    rows: list[dict[str, Any]], end_date: str, interval: str, *, as_of: date | None = None
) -> dict[str, Any] | None:
    """Disclose that the data stops before the requested range does.

    The logic moved to `cortex.analysis.coverage`, which every connector can reach. It was
    unreachable from anywhere but here, while the same hole existed in each of the others --
    three quarters of captured GA4 session calls ask for a daily series and could stop early
    without saying so. What stays here is the part only PostHog knows: how to find a bucket
    date in a PostHog row, and which explanations are candidates for a PostHog event.

    **The failure it exists for.** Asked whether signups had fallen, the analyst was handed a
    `user signed up` series running 1-3 August against a range ending on the 15th, computed a
    healthy 206/day from those three days, described it as "the August run rate", and concluded
    there was no real decline. Signup tracking had stopped recording twelve days earlier, which
    is either a catastrophic drop or a broken pipeline and is urgent under either reading.
    """
    if not rows:
        # Already covered: an empty series is reported through `total` and `row_count`, and
        # `result_key` makes the executor mark it empty. Repeating it here would add a second
        # voice saying the same thing.
        return None
    try:
        window_end = date.fromisoformat(end_date)
    except ValueError:
        return None

    buckets: list[date] = []
    for row in rows:
        raw = row.get("bucket")
        if not isinstance(raw, str):
            continue
        try:
            buckets.append(date.fromisoformat(raw[:10]))
        except ValueError:
            continue

    return gap_disclosure(
        buckets,
        window_end=window_end,
        interval=interval,
        # An event is the one thing here that can be *renamed*, which is the explanation most
        # easily mistaken for a real decline: the metric goes to zero and nothing is broken.
        as_of=as_of,
        alternatives=(
            "the event may have stopped firing, tracking may be broken, or the event may "
            "have been renamed"
        ),
    )


def _last_seen(value: Any) -> date | None:
    """The date part of a PostHog `last_seen_at`, or None if it is missing or malformed.

    Also accepts a bare `YYYY-MM-DD`, so the same parser reads both an event definition's
    timestamp and a bucket label. A naive datetime is treated as UTC rather than local: the
    timestamps here are all UTC, and letting the host's timezone move a date by a day would
    silently change which side of the threshold an event falls on.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        if len(value) == 10:
            return date.fromisoformat(value)
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).date()


#: How many name-siblings are worth naming, and how long a shared word has to be to count.
#:
#: Five, because the point is to make the analyst check one other series, not to hand it a
#: research project -- and a project with `$pageview_1` through `$pageview_84` would otherwise
#: bury the disclosure it is supposed to be. Six characters, because shorter shared words
#: ("user", "api", "new") relate almost anything to anything.
_MAX_SIBLINGS = 5
_MIN_SHARED_WORD = 6

#: Lifecycle verbs, excluded from the overlap. Almost every product event ends in one, so
#: matching on them relates almost anything to anything: `conversation_created` came back
#: "related" to `api key created` on the strength of the word "created" alone. What makes two
#: events candidates for measuring one concept is a shared *subject*, not a shared verb.
_LIFECYCLE_WORDS = frozenset(
    {
        "created",
        "updated",
        "deleted",
        "removed",
        "started",
        "stopped",
        "finished",
        "completed",
        "changed",
        "clicked",
        "viewed",
        "opened",
        "closed",
        "saved",
        "failed",
        "succeeded",
        "submitted",
        "received",
        "enabled",
        "disabled",
    }
)


def _words(name: str) -> set[str]:
    return {
        word
        for word in re.split(r"[^0-9a-z]+", name.lower())
        if len(word) >= _MIN_SHARED_WORD and word not in _LIFECYCLE_WORDS
    }


def _worth_comparing(payload: dict[str, Any]) -> bool:
    """Whether this series is one a sibling event could change the reading of.

    A confirmed level shift, and nothing else. A series that did not move has nothing in it to
    misread, and that keeps the cost discipline `blast_radius` set: one extra request on the
    calls that need it, none on the calls that do not.

    **Known blind spot, chosen rather than missed.** `describe_movement` needs
    `min_segment_days` (14) on each side and returns no verdict at all below that, so a
    request for a window under a month carries no `movement` block whether or not anything
    happened -- and an analyst reading a three-week series is exactly as exposed to a rename as
    one reading a longer series. Treating an absent verdict as grounds to look was implemented
    and reverted: "last two weeks" and "last month" are the commonest questions asked, so it
    put an extra request on nearly every trend call, which is the cost this gate exists to
    avoid. Closing it properly means making the shift test work on short windows, not paying
    for a lookup on every call.
    """
    if not payload.get("series"):
        return False
    return bool((payload.get("movement") or {}).get("level_shifts"))


def _related_events(events: list[dict[str, Any]], event: str) -> dict[str, Any] | None:
    """Other events in this project that may measure the same thing under a different name.

    **Why a connector says this rather than the analyst working it out.** A rename or a
    re-emitter migration leaves two events for one concept, and a series read without its
    sibling is a series that can move for reasons that have nothing to do with behaviour. Five
    attempts at one real question -- *did conversation volume change in August?* -- returned
    "it did not fall", "it fell 78%", "it rose 18x", and twice "the premise does not hold",
    entirely according to whether the attempt happened to query both
    `conversation_created` and `agent_server.conversation_created`.

    Name overlap, not semantics: a shared word of six characters or more, in either direction,
    which is what a rename and a re-prefixing both look like. The connector is entitled to say
    *these names overlap and both have data*; whether they measure the same thing is the
    investigation's judgement and the note says so.

    Stale siblings are kept and their `last_seen_at` reported, because a handover is exactly
    the case this exists for: the old event dies as the new one starts, and a filter that
    dropped dead events would drop the more informative half of it.
    """
    mine = _words(event)
    if not mine or not events:
        return None
    siblings: list[dict[str, Any]] = []
    for row in events:
        name = row.get("name")
        if not isinstance(name, str) or name == event:
            continue
        seen = _last_seen(row.get("last_seen_at"))
        if seen is None or not (mine & _words(name)):
            continue
        siblings.append({"name": name, "last_seen_at": seen.isoformat()})
    if not siblings:
        return None
    siblings.sort(key=lambda entry: entry["last_seen_at"], reverse=True)
    shown = siblings[:_MAX_SIBLINGS]
    return {
        "related_events": {
            "matched_on": sorted(mine),
            "count": len(siblings),
            "events": shown,
            "note": (
                f"{len(siblings)} other event(s) in this project share a name with "
                f"{event!r} and have data of their own"
                + (f", the {len(shown)} most recent shown" if len(siblings) > len(shown) else "")
                + ". A rename or a change of emitter leaves two events for one concept, so a "
                "movement in this series may be a movement in what is being recorded rather "
                "than in what users did. Compare before reading it as a change in behaviour."
            ),
        }
    }


def _blast_radius(events: list[dict[str, Any]], last_bucket: date) -> dict[str, Any] | None:
    """Which *other* events stopped when this one did, and which kept flowing.

    **The discriminating fact, and the reason this is not a guess.** `_series_gap` correctly
    refuses to say why a series ended, listing three possibilities -- the event stopped firing,
    tracking broke, the event was renamed. But it left the analyst holding a warning with no
    way to resolve it, and a warning that cannot be resolved gets resolved by whatever is
    nearest to hand. In production the analyst reached for a second metric that had also
    fallen, and asserted a common cause across two failures that were seven days apart.

    The sibling events settle it, and settling it needs no interpretation at all -- only the
    observation that these events stopped at the same moment and those did not:

      - **this event alone stopped** while everything else kept recording, so no shared
        pipeline can be responsible. Instrumentation or a rename.
      - **a group stopped together** while others continued, so no change in user behaviour
        can be responsible -- demand does not stop unrelated events within the same hour.
      - **everything stopped**, so nothing about this event is distinctive.

    Each of those is an *elimination*, which is the part a connector is entitled to do. What
    caused the surviving possibility is still the investigation's job.
    """
    if not events:
        return None

    # A day of slack on either side. Events that share a pipeline stop within minutes of each
    # other, but they are recorded against wall-clock timestamps that can straddle midnight.
    threshold = last_bucket + timedelta(days=1)
    stopped: list[dict[str, Any]] = []
    live: list[dict[str, Any]] = []
    for row in events:
        seen = _last_seen(row.get("last_seen_at"))
        name = row.get("name")
        if seen is None or not isinstance(name, str):
            continue
        entry = {"name": name, "last_seen_at": seen.isoformat()}
        (stopped if seen <= threshold else live).append(entry)

    if not stopped and not live:
        return None

    # Everything the query itself already covers is excluded from the counts below, so
    # "stopped with it" never counts the event under investigation as its own corroboration.
    if live and len(stopped) > 1:
        scope = "shared_with_other_events"
        note = (
            f"{len(stopped)} events stopped recording on or before "
            f"{threshold.isoformat()}, while {len(live)} others are still recording. Events "
            "that share no product surface do not stop within a day of each other because "
            "user behaviour changed -- that pattern is what a common collection failure looks "
            "like. Which events are in each group localises it: compare what the stopped ones "
            "have in common against what the live ones have in common, and note that a second "
            "metric falling on a *different* date is a separate incident, not corroboration "
            "for this one."
        )
    elif live and len(stopped) <= 1:
        scope = "this_event_only"
        note = (
            f"Every other event ({len(live)}) is still recording after "
            f"{threshold.isoformat()}. A shared pipeline or collection failure would have "
            "stopped them too, so this gap is specific to this event -- it was renamed, its "
            "instrumentation was removed or broken, or it genuinely stopped happening. "
            "Distinguish a rename by looking for a new event whose history starts where this "
            "one ends."
        )
    else:
        scope = "project_wide"
        note = (
            f"No event in this project has recorded anything after {threshold.isoformat()}. "
            "Nothing about this event is distinctive; the whole project stopped receiving "
            "data, which is an ingestion or credential failure rather than anything about "
            "the metric asked for."
        )

    return {
        "blast_radius": {
            "scope": scope,
            "stopped_count": len(stopped),
            "still_recording_count": len(live),
            # Capped: this is orientation, not a catalogue, and an unbounded list would put
            # several hundred event names into every gapped trend.
            "stopped_with_it": sorted(stopped, key=lambda e: e["last_seen_at"])[:25],
            "still_recording": sorted(live, key=lambda e: e["last_seen_at"])[:25],
        },
        "blast_radius_note": note,
    }


def _bucket_coverage(
    rows: list[dict[str, Any]], start_date: str, end_date: str, interval: str
) -> dict[str, Any] | None:
    """Which returned buckets cover less time than a full bucket, or None if all are whole.

    **The production defect this exists for.** Asked whether signups had fallen from last
    month, the analyst received two monthly buckets — 4,849 for July and 1,884 for the first
    twelve days of August — and nothing in the payload said the second was a third of a month
    long. A total against a total is the obvious comparison and it was meaningless. The answer it
    produced was grounded in real numbers and wrong, which is the failure no citation gate can
    see.

    So the tool discloses it. A short bucket is a fact about the data, not an inference the
    reader should be left to make, and the same rule already applies elsewhere in this file:
    `event_trend` returns `total` so an empty series can be told from a real drop to zero.

    **Why the bucket starts are read from the rows rather than computed.** `toStartOfWeek` follows
    ClickHouse's default mode, which begins the week on Sunday — a convention this module would
    otherwise have to assume and would eventually assume wrongly. Each bucket's *full* length is
    computable from its own start without knowing that convention (a week is seven days, a month
    is whatever `monthrange` says), so the data supplies the boundaries and the calendar supplies
    the lengths.
    """
    if interval == "day":
        # A day bucket is a day. There is nothing to disclose.
        return None
    try:
        window_start = date.fromisoformat(start_date)
        window_end = date.fromisoformat(end_date)
    except ValueError:
        return None

    starts: list[date] = []
    for row in rows:
        raw = row.get("bucket")
        if not isinstance(raw, str):
            continue
        try:
            parsed = date.fromisoformat(raw[:10])
        except ValueError:
            continue
        if parsed not in starts:
            starts.append(parsed)
    if not starts:
        return None
    starts.sort()

    partial: list[dict[str, Any]] = []
    for bucket_start in starts:
        if interval == "week":
            full_days = 7
        else:
            full_days = monthrange(bucket_start.year, bucket_start.month)[1]
        bucket_end = bucket_start + timedelta(days=full_days - 1)
        covered_from = max(bucket_start, window_start)
        covered_to = min(bucket_end, window_end)
        covered = (covered_to - covered_from).days + 1
        if covered < full_days:
            partial.append(
                {
                    "bucket": bucket_start.isoformat(),
                    "days_covered": covered,
                    "days_in_full_bucket": full_days,
                }
            )
    if not partial:
        return None
    return {
        "partial_buckets": partial,
        "note": (
            "These buckets cover less than a full "
            + interval
            + " because the requested range starts or ends mid-"
            + interval
            + ". Their totals are not comparable with the whole buckets in this series; "
            "compare rates per day, or request whole buckets."
        ),
    }


def _check_range(start_date: str, end_date: str) -> None:
    if start_date > end_date:
        raise InvalidParams(f"posthog: start_date {start_date} is after end_date {end_date}")
