"""Amplitude connector — the third product-analytics source.

Same purpose as `mixpanel.py`: a company on Amplitude should be able to paste an API key pair and
get the answers a PostHog tenant gets. The shape mirrors the other two on purpose, because the
investigation loop reasons about *capabilities*, and three analytics sources that each spell
"count this event over these days" differently would push that variation into the prompt.

**Two things differ from Mixpanel and are worth knowing before reading the code.**

Amplitude's Dashboard REST API takes dates as `YYYYMMDD` and its interval as a number of days —
`1`, `7` or `30` — rather than a named unit. Both are translated here from the same vocabulary
the other connectors use, so the model writes one kind of date and one kind of bucket everywhere.
A connector that leaks its vendor's spelling into the tool schema makes the analyst's job harder
for no benefit.

The event definition is a **JSON object in a query parameter** (`e={"event_type":"..."}`), which
means a name reaches the API through a nested encoding. It is validated against the same
conservative pattern the other connectors use and then serialised with `json.dumps` rather than
concatenated, so quoting is the serialiser's problem rather than this module's.

**Authentication** is HTTP Basic with the project's API key and secret key, stored as one
`api_key:secret_key` string — the same one-secret-per-provider rule as Mixpanel, and for the same
reason: the halves are useless apart.
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

import httpx

from cortex.db.models import CredentialProvider
from cortex.tools.base import (
    Capability,
    Freshness,
    InvalidParams,
    ToolContext,
    ToolResult,
)
from cortex.tools.base import Tool as BaseTool
from cortex.tools.http import DEFAULT_TIMEOUT, request_json

#: Amplitude's residency hosts. Like Mixpanel's, these do not share data, so an unknown region
#: is refused rather than defaulted — a key issued for the EU fails against the US host in a way
#: that reads as a bad key.
REGIONS = {
    "us": "https://amplitude.com",
    "eu": "https://analytics.eu.amplitude.com",
}

#: Metric names the segmentation endpoint accepts. Allowlisted because the value is sent verbatim.
#:
#: `totals` and `uniques` are the pair that matters: "signups fell" is a totals question and
#: "fewer people signed up" is a uniques one, and answering one with the other is a wrong answer
#: that looks right.
MEASURES = ("totals", "uniques", "pct_dau", "average")

#: Bucket widths, named here and translated to Amplitude's day counts on the way out. The
#: vocabulary matches `posthog.event_trend` and `mixpanel.event_trend` so the analyst writes the
#: same thing whichever source a tenant happens to run.
INTERVALS = {"day": "1", "week": "7", "month": "30"}

_DATE = {
    "type": "string",
    "pattern": r"^\d{4}-\d{2}-\d{2}$",
    "description": "Date as YYYY-MM-DD, inclusive.",
}

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_$ .:\-/]{1,255}$")


class AmplitudeTool(BaseTool):
    name = "amplitude"
    provider = CredentialProvider.AMPLITUDE

    def capabilities(self) -> list[Capability]:
        return [
            Capability(
                name="list_events",
                description=(
                    "The event types this project defines. Call this before any trend: an "
                    "event name that does not exist returns an empty series, which is "
                    "indistinguishable from a real drop to zero."
                ),
                params_schema={"type": "object", "additionalProperties": False, "properties": {}},
                handler=self.list_events,
                result_key="events",
                discovery=True,
            ),
            Capability(
                name="event_trend",
                description=(
                    "A time series for one event, optionally segmented by a property. The "
                    "main measurement capability: use it to establish whether a metric moved."
                ),
                params_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["event", "start_date", "end_date"],
                    "properties": {
                        "event": {
                            "type": "string",
                            "description": "Event type, exactly as list_events reports it.",
                        },
                        "start_date": _DATE,
                        "end_date": _DATE,
                        "interval": {
                            "type": "string",
                            "enum": sorted(INTERVALS),
                            "default": "day",
                            "description": (
                                "Bucket width. Prefer 'day' when checking whether something "
                                "changed: Amplitude's 'month' is a rolling 30 days, so a "
                                "bucket does not line up with a calendar month."
                            ),
                        },
                        "measure": {
                            "type": "string",
                            "enum": list(MEASURES),
                            "default": "totals",
                            "description": (
                                "'totals' counts occurrences, 'uniques' counts distinct "
                                "users. These answer different questions."
                            ),
                        },
                        "group_by": {
                            "type": "string",
                            "description": "Event property to segment by, e.g. 'platform'.",
                        },
                    },
                },
                handler=self.event_trend,
                result_key="series",
            ),
        ]

    async def list_events(self, ctx: ToolContext) -> ToolResult:
        raw = await self._get(ctx, "/api/2/events/list", {})
        rows = raw.get("data")
        events = (
            [
                {
                    "name": row.get("value"),
                    "display_name": row.get("display"),
                    # Amplitude marks events it has stopped seeing. A name that exists but is no
                    # longer firing is a finding in itself, and hiding it would turn "this stopped"
                    # into "this never existed".
                    "non_active": bool(row.get("non_active")),
                    "deleted": bool(row.get("deleted")),
                }
                for row in rows
                if isinstance(row, dict)
            ]
            if isinstance(rows, list)
            else []
        )
        return ToolResult(
            payload={"event_count": len(events), "events": events},
            source_ref="amplitude://events/list",
            meta={"freshness": Freshness.LIVE},
        )

    async def event_trend(
        self,
        ctx: ToolContext,
        *,
        event: str,
        start_date: str,
        end_date: str,
        interval: str = "day",
        measure: str = "totals",
        group_by: str | None = None,
    ) -> ToolResult:
        if start_date > end_date:
            raise InvalidParams(f"amplitude: start_date {start_date} is after end_date {end_date}")
        if interval not in INTERVALS:
            raise InvalidParams(
                f"amplitude: interval must be one of {', '.join(sorted(INTERVALS))}"
            )
        if measure not in MEASURES:
            raise InvalidParams(f"amplitude: measure must be one of {', '.join(MEASURES)}")
        _reject_unsafe(event, field="event")

        definition: dict[str, Any] = {"event_type": event}
        if group_by:
            _reject_unsafe(group_by, field="group_by")
            # Serialised, never concatenated: the quoting rules are json's problem, and a
            # hand-built string is where an injected quote would change the query's meaning.
            definition["group_by"] = [{"type": "event", "value": group_by}]

        params = {
            "e": json.dumps(definition, separators=(",", ":")),
            "start": _compact(start_date, field="start_date"),
            "end": _compact(end_date, field="end_date"),
            "m": measure,
            "i": INTERVALS[interval],
        }
        raw = await self._get(ctx, "/api/2/events/segmentation", params)
        series = _flatten(raw)

        return ToolResult(
            payload={
                "event": event,
                "measure": measure,
                "interval": interval,
                "start_date": start_date,
                "end_date": end_date,
                "group_by": group_by,
                "row_count": len(series),
                "series": series,
                # As elsewhere: an empty series and a real zero are different findings.
                "total": sum(row["value"] for row in series),
            },
            source_ref=(
                f"amplitude://events/segmentation?event={event}&start={start_date}&end={end_date}"
            ),
            meta={"freshness": Freshness.LIVE},
        )

    # -- plumbing ---------------------------------------------------------------------

    def _host(self, ctx: ToolContext) -> str:
        region = str(ctx.credential_metadata.get("region") or "us").strip().lower()
        host = REGIONS.get(region)
        if host is None:
            raise InvalidParams(
                f"amplitude: unknown region {region!r}; expected one of "
                f"{', '.join(sorted(REGIONS))}. The regions do not share data."
            )
        return host

    def _auth(self, ctx: ToolContext) -> tuple[str, str]:
        """The API key and secret key, stored as one `api_key:secret_key` string.

        Partitioned on the first colon only, so a secret containing one survives intact rather
        than being corrupted into a credential that fails for an invisible reason.
        """
        raw = ctx.credential or ""
        api_key, sep, secret = raw.partition(":")
        if not sep or not api_key or not secret:
            raise InvalidParams(
                "amplitude: the credential must be 'api_key:secret_key' from the project's "
                "settings. Both halves are required — the Dashboard REST API authenticates "
                "with HTTP Basic over the pair."
            )
        return api_key, secret

    def _client(self, ctx: ToolContext) -> httpx.AsyncClient:
        """The configured client. Separate from `_get` so tests can exercise the real headers,
        base URL and auth while replacing only the socket."""
        api_key, secret = self._auth(ctx)
        return httpx.AsyncClient(
            base_url=self._host(ctx),
            auth=httpx.BasicAuth(api_key, secret),
            headers={"Accept": "application/json"},
            timeout=DEFAULT_TIMEOUT,
        )

    async def _get(self, ctx: ToolContext, path: str, params: dict[str, Any]) -> dict[str, Any]:
        async with self._client(ctx) as client:
            return await request_json(client, "GET", path, tool=self.name, params=params)


def _reject_unsafe(value: str, *, field: str) -> str:
    if not _SAFE_NAME.match(value):
        raise InvalidParams(
            f"amplitude: {field} {value!r} contains characters that cannot be used in a "
            "query. Call list_events to see the exact names this project defines."
        )
    return value


def _compact(value: str, *, field: str) -> str:
    """`2026-08-12` to `20260812`, which is the only date format this API accepts.

    Parsed rather than stripped of dashes: `date.fromisoformat` rejects `2026-13-45`, and a
    silently malformed date would come back as an empty series that reads as a real absence.
    """
    try:
        return date.fromisoformat(value).strftime("%Y%m%d")
    except ValueError as exc:
        raise InvalidParams(f"amplitude: {field} {value!r} is not a valid YYYY-MM-DD date") from exc


def _flatten(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn Amplitude's parallel arrays into flat, ordered rows.

    The response carries `xValues` (the dates), `series` (one array of numbers per segment) and
    `seriesLabels` (what each of those arrays is). The value's meaning therefore depends on the
    position of its array *and* the position within it — the same positional-data problem the
    report schema rejects for chart points, and the same fix: name the fields.

    A short or missing label list is tolerated rather than raising. Amplitude omits labels for an
    unsegmented query, and an analyst that gets rows without a segment name has still learned
    what it asked; one that gets an exception has learned nothing.
    """
    data = raw.get("data")
    if not isinstance(data, dict):
        return []
    dates = [d for d in (data.get("xValues") or []) if isinstance(d, str)]
    series = data.get("series")
    if not isinstance(series, list):
        return []
    labels = data.get("seriesLabels") or []

    rows: list[dict[str, Any]] = []
    for index, values in enumerate(series):
        if not isinstance(values, list):
            continue
        label = labels[index] if index < len(labels) else None
        for position, value in enumerate(values):
            if position >= len(dates):
                break
            row: dict[str, Any] = {
                "bucket": dates[position],
                "value": value if isinstance(value, int | float) else 0,
            }
            if label is not None and len(series) > 1:
                # A label on a single-series response names the event, not a segment, and
                # echoing it as `segment` would invent a breakdown nobody asked for.
                row["segment"] = label if isinstance(label, str) else str(label)
            rows.append(row)
    return rows
