"""GA4 connector — traffic, conversion and funnel data.

This answers the "what changed" half of an investigation. It deliberately does not
try to answer "why": attributing a conversion drop to a deploy requires GitHub, and
to a campaign requires HubSpot. Each capability returns dimensioned rows so the
investigation loop can find the largest contributor itself rather than trusting a
single aggregate.

GA4 sampling and thresholding are surfaced in `meta` rather than hidden. A report
that cites a sampled figure as though it were exact is a grounding failure even
when the number came from a real API.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import httpx

from cortex.analysis.coverage import gap_disclosure
from cortex.db.models import CredentialProvider
from cortex.tools.base import Capability, Freshness, InvalidParams, ToolContext, ToolResult
from cortex.tools.base import Tool as BaseTool
from cortex.tools.disclosures import grade_series, series_disclosures
from cortex.tools.google_auth import GA4_SCOPES, access_token
from cortex.tools.http import DEFAULT_TIMEOUT, request_json

#: The qualified capability name the shared assembler dispatches on.
_GET_SESSIONS = "ga4__get_sessions"

API_ROOT = "https://analyticsdata.googleapis.com/v1beta"

_DATE = {
    "type": "string",
    "pattern": r"^\d{4}-\d{2}-\d{2}$",
    "description": "Date as YYYY-MM-DD.",
}

_MAX_ROWS = 250

# Allowlisted so a hallucinated metric name fails validation with a clear message
# instead of an opaque 400 from Google.
METRICS = (
    "sessions",
    "activeUsers",
    "newUsers",
    "screenPageViews",
    "conversions",
    "userEngagementDuration",
    "bounceRate",
    "engagementRate",
    "sessionConversionRate",
    "totalRevenue",
)

DIMENSIONS = (
    "date",
    "deviceCategory",
    "sessionDefaultChannelGroup",
    "sessionSource",
    "sessionMedium",
    "sessionCampaignName",
    "country",
    "pagePath",
    "landingPage",
    "browser",
    "operatingSystem",
)


def _base_schema(_required: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    """Schema head for a date-ranged capability.

    `_required` adds to the mandatory set. Every parameter a handler cannot default
    must appear there, or a model omitting it gets a TypeError instead of a
    correctable validation error. Enforced by
    tests/tools/test_framework.py::TestSchemaMatchesHandler (F-06).
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["start_date", "end_date", *(_required or [])],
        "properties": {"start_date": _DATE, "end_date": _DATE, **extra},
    }


class GA4Tool(BaseTool):
    name = "ga4"
    provider = CredentialProvider.GA4

    def capabilities(self) -> list[Capability]:
        return [
            Capability(
                name="get_sessions",
                description=(
                    "Session and user totals for a date range, optionally broken down "
                    "by dimension. Start here to confirm whether a reported change is "
                    "real and to find which segment drove it."
                ),
                params_schema=_base_schema(
                    dimensions={
                        "type": "array",
                        "items": {"type": "string", "enum": list(DIMENSIONS)},
                        "maxItems": 3,
                        "description": "Break the totals down by these dimensions.",
                    },
                    limit={
                        "type": "integer",
                        "minimum": 1,
                        "maximum": _MAX_ROWS,
                        "default": 50,
                    },
                ),
                handler=self.get_sessions,
                result_key="rows",
            ),
            Capability(
                name="compare_periods",
                description=(
                    "Compare metrics between two date ranges and return the absolute "
                    "and percentage change per row. Use this instead of two separate "
                    "calls so the comparison is computed on identical row keys."
                ),
                params_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "current_start",
                        "current_end",
                        "previous_start",
                        "previous_end",
                    ],
                    "properties": {
                        "current_start": _DATE,
                        "current_end": _DATE,
                        "previous_start": _DATE,
                        "previous_end": _DATE,
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(METRICS)},
                            "minItems": 1,
                            "maxItems": 5,
                            "default": ["sessions", "conversions"],
                        },
                        "dimensions": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(DIMENSIONS)},
                            "maxItems": 2,
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": _MAX_ROWS,
                            "default": 50,
                        },
                    },
                },
                handler=self.compare_periods,
                result_key="comparison",
            ),
            Capability(
                name="get_funnel",
                description=(
                    "Conversion rate by step for a date range, broken down by "
                    "dimension. Use this to locate where in onboarding users drop."
                ),
                params_schema=_base_schema(
                    dimension={
                        "type": "string",
                        "enum": list(DIMENSIONS),
                        "default": "deviceCategory",
                    },
                    limit={
                        "type": "integer",
                        "minimum": 1,
                        "maximum": _MAX_ROWS,
                        "default": 50,
                    },
                ),
                handler=self.get_funnel,
                result_key="rows",
            ),
            Capability(
                name="top_pages",
                description="Highest-traffic pages for a date range, with engagement.",
                params_schema=_base_schema(
                    limit={
                        "type": "integer",
                        "minimum": 1,
                        "maximum": _MAX_ROWS,
                        "default": 25,
                    },
                ),
                handler=self.top_pages,
                result_key="rows",
            ),
            Capability(
                name="run_report",
                description=(
                    "Arbitrary GA4 report over allowlisted metrics and dimensions. "
                    "Use only when no more specific capability fits."
                ),
                params_schema=_base_schema(
                    _required=["metrics"],
                    metrics={
                        "type": "array",
                        "items": {"type": "string", "enum": list(METRICS)},
                        "minItems": 1,
                        "maxItems": 8,
                    },
                    dimensions={
                        "type": "array",
                        "items": {"type": "string", "enum": list(DIMENSIONS)},
                        "maxItems": 4,
                    },
                    limit={
                        "type": "integer",
                        "minimum": 1,
                        "maximum": _MAX_ROWS,
                        "default": 50,
                    },
                ),
                handler=self.run_report,
                result_key="rows",
            ),
        ]

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _property_id(ctx: ToolContext) -> str:
        """The GA4 property, from non-secret connector metadata.

        Not a tool parameter: the property is a property of the tenant's
        connection, and letting the model supply it would let a hallucinated id
        reach Google.
        """
        raw = str(ctx.credential_metadata.get("property_id", "")).strip()
        if not raw:
            raise InvalidParams(
                "GA4 credential has no property_id in its metadata; reconnect the "
                "integration with the GA4 property id"
            )
        digits = raw.removeprefix("properties/")
        if not re.fullmatch(r"\d{6,20}", digits):
            raise InvalidParams(f"GA4 property_id must be numeric, got {raw!r}")
        return digits

    async def _client(self, ctx: ToolContext) -> httpx.AsyncClient:
        # Token minting is awaited rather than done synchronously: a blocking
        # refresh would stall the event loop on every tool call.
        token = await access_token(str(ctx.credential), GA4_SCOPES)
        return httpx.AsyncClient(
            base_url=API_ROOT,
            timeout=DEFAULT_TIMEOUT,
            headers={"Authorization": f"Bearer {token}"},
        )

    async def _run(
        self,
        ctx: ToolContext,
        *,
        metrics: list[str],
        dimensions: list[str],
        ranges: list[dict[str, str]],
        limit: int,
    ) -> dict[str, Any]:
        property_id = self._property_id(ctx)
        body = {
            "dateRanges": ranges,
            "metrics": [{"name": m} for m in metrics],
            "dimensions": [{"name": d} for d in dimensions],
            "limit": str(limit),
        }
        async with await self._client(ctx) as client:
            return await request_json(
                client,
                "POST",
                f"/properties/{property_id}:runReport",
                tool=self.name,
                json_body=body,
            )

    # ------------------------------------------------------------------ capabilities

    async def get_sessions(
        self,
        ctx: ToolContext,
        *,
        start_date: str,
        end_date: str,
        dimensions: list[str] | None = None,
        limit: int = 50,
    ) -> ToolResult:
        _check_range(start_date, end_date)
        metrics = ["sessions", "activeUsers", "newUsers"]
        raw = await self._run(
            ctx,
            metrics=metrics,
            dimensions=dimensions or [],
            ranges=[{"startDate": start_date, "endDate": end_date}],
            limit=limit,
        )
        rows = _parse_rows(raw)
        payload: dict[str, Any] = {
            "property_id": self._property_id(ctx),
            "start_date": start_date,
            "end_date": end_date,
            "dimensions": dimensions or [],
            "metrics": metrics,
            "totals": _parse_totals(raw, metrics),
            "row_count": len(rows),
            "rows": rows,
        }
        # Only when the caller asked for a series. Without a `date` dimension these rows are a
        # breakdown, not a timeline, and there is no "stops early" to report -- 31 of 42 captured
        # calls to this capability did ask for one, so this is the common case rather than a
        # corner of it.
        # Through the shared assembler, which the eval harness also calls. Two paths computing
        # the same disclosures would drift, and drift between what production discloses and what
        # the suite measures is exactly what this indirection prevents.
        payload.update(series_disclosures(_GET_SESSIONS, payload, {"end_date": end_date}))
        payload.update(grade_series(_GET_SESSIONS, payload))
        return ToolResult(
            payload=payload,
            source_ref=_source_ref(self._property_id(ctx), start_date, end_date),
            meta=_quality(raw),
        )

    async def compare_periods(
        self,
        ctx: ToolContext,
        *,
        current_start: str,
        current_end: str,
        previous_start: str,
        previous_end: str,
        metrics: list[str] | None = None,
        dimensions: list[str] | None = None,
        limit: int = 50,
    ) -> ToolResult:
        _check_range(current_start, current_end)
        _check_range(previous_start, previous_end)
        metrics = metrics or ["sessions", "conversions"]
        dimensions = dimensions or []

        raw = await self._run(
            ctx,
            metrics=metrics,
            dimensions=dimensions,
            ranges=[
                {"startDate": current_start, "endDate": current_end, "name": "current"},
                {"startDate": previous_start, "endDate": previous_end, "name": "previous"},
            ],
            limit=limit,
        )

        # GA4 returns one row per (dimension tuple, date range), distinguished by a
        # synthetic dateRange dimension appended to the row key. Rows are rejoined
        # here so the comparison is computed on identical keys — the reason this is
        # one capability rather than two calls the model would have to align.
        current, previous = _split_by_range(raw, dimensions, metrics)
        comparison = []
        for key in sorted(set(current) | set(previous)):
            cur_row = current.get(key, {})
            prev_row = previous.get(key, {})
            entry: dict[str, Any] = {
                "dimensions": dict(zip(dimensions, key, strict=False)) if dimensions else {},
            }
            for metric in metrics:
                cur = cur_row.get(metric)
                prev = prev_row.get(metric)
                entry[metric] = {
                    "current": cur,
                    "previous": prev,
                    "absolute_change": _absolute(cur, prev),
                    "percent_change": _percent(cur, prev),
                }
            comparison.append(entry)

        return ToolResult(
            payload={
                "property_id": self._property_id(ctx),
                "current_period": {"start": current_start, "end": current_end},
                "previous_period": {"start": previous_start, "end": previous_end},
                "metrics": metrics,
                "dimensions": dimensions,
                "row_count": len(comparison),
                "comparison": comparison,
            },
            source_ref=_source_ref(self._property_id(ctx), previous_start, current_end),
            meta=_quality(raw),
        )

    async def get_funnel(
        self,
        ctx: ToolContext,
        *,
        start_date: str,
        end_date: str,
        dimension: str = "deviceCategory",
        limit: int = 50,
    ) -> ToolResult:
        _check_range(start_date, end_date)
        metrics = ["sessions", "conversions", "sessionConversionRate", "engagementRate"]
        raw = await self._run(
            ctx,
            metrics=metrics,
            dimensions=[dimension],
            ranges=[{"startDate": start_date, "endDate": end_date}],
            limit=limit,
        )
        rows = _parse_rows(raw)
        for row in rows:
            # Recomputed rather than trusted: sessionConversionRate is unavailable
            # for some property configurations, and a missing rate would otherwise
            # read as zero.
            sessions = row["metrics"].get("sessions")
            conversions = row["metrics"].get("conversions")
            row["derived_conversion_rate"] = (
                round(conversions / sessions, 6) if sessions and conversions is not None else None
            )
        return ToolResult(
            payload={
                "property_id": self._property_id(ctx),
                "start_date": start_date,
                "end_date": end_date,
                "dimension": dimension,
                "metrics": metrics,
                "totals": _parse_totals(raw, metrics),
                "row_count": len(rows),
                "rows": rows,
            },
            source_ref=_source_ref(self._property_id(ctx), start_date, end_date),
            meta=_quality(raw),
        )

    async def top_pages(
        self, ctx: ToolContext, *, start_date: str, end_date: str, limit: int = 25
    ) -> ToolResult:
        _check_range(start_date, end_date)
        metrics = ["screenPageViews", "sessions", "engagementRate"]
        raw = await self._run(
            ctx,
            metrics=metrics,
            dimensions=["pagePath"],
            ranges=[{"startDate": start_date, "endDate": end_date}],
            limit=limit,
        )
        rows = _parse_rows(raw)
        return ToolResult(
            payload={
                "property_id": self._property_id(ctx),
                "start_date": start_date,
                "end_date": end_date,
                "metrics": metrics,
                "row_count": len(rows),
                "rows": rows,
            },
            source_ref=_source_ref(self._property_id(ctx), start_date, end_date),
            meta=_quality(raw),
        )

    async def run_report(
        self,
        ctx: ToolContext,
        *,
        start_date: str,
        end_date: str,
        metrics: list[str],
        dimensions: list[str] | None = None,
        limit: int = 50,
    ) -> ToolResult:
        _check_range(start_date, end_date)
        raw = await self._run(
            ctx,
            metrics=metrics,
            dimensions=dimensions or [],
            ranges=[{"startDate": start_date, "endDate": end_date}],
            limit=limit,
        )
        rows = _parse_rows(raw)
        return ToolResult(
            payload={
                "property_id": self._property_id(ctx),
                "start_date": start_date,
                "end_date": end_date,
                "metrics": metrics,
                "dimensions": dimensions or [],
                "totals": _parse_totals(raw, metrics),
                "row_count": len(rows),
                "rows": rows,
            },
            source_ref=_source_ref(self._property_id(ctx), start_date, end_date),
            meta=_quality(raw),
            freshness=Freshness.LIVE,
        )


# ---------------------------------------------------------------------- parsing


def _check_range(start: str, end: str) -> None:
    if start > end:
        raise InvalidParams(f"start_date {start} is after end_date {end}")


def _headers(raw: dict[str, Any]) -> tuple[list[str], list[str]]:
    dims = [h.get("name", "") for h in raw.get("dimensionHeaders", [])]
    mets = [h.get("name", "") for h in raw.get("metricHeaders", [])]
    return dims, mets


def _coerce(value: str | None) -> float | int | None:
    """GA4 returns every metric as a string; typed numbers are what charts need."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def _parse_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    dims, mets = _headers(raw)
    rows = []
    for row in raw.get("rows", []):
        dim_values = [d.get("value") for d in row.get("dimensionValues", [])]
        met_values = [_coerce(m.get("value")) for m in row.get("metricValues", [])]
        rows.append(
            {
                "dimensions": dict(zip(dims, dim_values, strict=False)),
                "metrics": dict(zip(mets, met_values, strict=False)),
            }
        )
    return rows


def _parse_totals(raw: dict[str, Any], metrics: list[str]) -> dict[str, Any]:
    totals = raw.get("totals") or []
    if not totals:
        return {}
    values = [_coerce(m.get("value")) for m in totals[0].get("metricValues", [])]
    return dict(zip(metrics, values, strict=False))


def _split_by_range(
    raw: dict[str, Any], dimensions: list[str], metrics: list[str]
) -> tuple[dict[tuple[str, ...], dict[str, Any]], dict[tuple[str, ...], dict[str, Any]]]:
    """Split a two-range report into current and previous, keyed by dimension tuple."""
    dims, mets = _headers(raw)
    try:
        range_index = dims.index("dateRange")
    except ValueError:
        range_index = -1

    current: dict[tuple[str, ...], dict[str, Any]] = {}
    previous: dict[tuple[str, ...], dict[str, Any]] = {}

    for row in raw.get("rows", []):
        dim_values = [d.get("value", "") for d in row.get("dimensionValues", [])]
        met_values = [_coerce(m.get("value")) for m in row.get("metricValues", [])]
        metrics_map = dict(zip(mets, met_values, strict=False))

        if range_index >= 0 and range_index < len(dim_values):
            range_name = dim_values[range_index]
            key = tuple(v for i, v in enumerate(dim_values) if i != range_index)
        else:
            range_name, key = "current", tuple(dim_values)

        target = previous if range_name == "previous" else current
        target[key] = {m: metrics_map.get(m) for m in metrics}

    del dimensions  # keys are positional; names are attached by the caller
    return current, previous


def _absolute(current: float | int | None, previous: float | int | None) -> float | int | None:
    if current is None or previous is None:
        return None
    delta = current - previous
    return int(delta) if isinstance(delta, float) and delta.is_integer() else delta


def _percent(current: float | int | None, previous: float | int | None) -> float | None:
    """Percentage change, or None when it is undefined.

    Returns None rather than 0 or infinity when the previous value is zero: a
    metric going from 0 to 500 has no meaningful percentage, and reporting one
    would be a fabricated figure in a grounded report.
    """
    if current is None or previous is None or previous == 0:
        return None
    return round((current - previous) / previous * 100, 2)


def _quality(raw: dict[str, Any]) -> dict[str, Any]:
    """Sampling and thresholding notices.

    Surfaced so the report's confidence section can disclose them. A sampled figure
    presented as exact is a grounding failure even when the API call was real.
    """
    meta = raw.get("metadata") or {}
    return {
        "sampled": bool(raw.get("propertyQuota", {}).get("tokensPerDay"))
        if "propertyQuota" in raw
        else False,
        "data_loss_from_other_row": bool(meta.get("dataLossFromOtherRow")),
        "currency_code": meta.get("currencyCode"),
        "time_zone": meta.get("timeZone"),
        "row_count_total": raw.get("rowCount"),
    }


def _source_ref(property_id: str, start: str, end: str) -> str:
    return f"ga4://properties/{property_id}/runReport?{start}..{end}"


def _gap(
    rows: list[dict[str, Any]], end_date: str, *, as_of: date | None = None
) -> dict[str, Any] | None:
    """Whether a date-dimensioned series stops before the requested range does.

    The same disclosure PostHog has carried since the "206/day is the August run rate" answer,
    reaching GA4 for the first time. It was PostHog-only while three quarters of captured GA4
    `get_sessions` calls asked for a date-dimensioned series -- the identical defect on a
    different source, with nothing to stop it.

    Returns None for a non-series request. A breakdown by country has no last bucket, and
    inventing one from row order would disclose a fact about the sort, not about the data.
    """
    try:
        window_end = date.fromisoformat(end_date)
    except ValueError:
        return None

    buckets: list[date] = []
    for row in rows:
        raw = (row.get("dimensions") or {}).get("date")
        if not isinstance(raw, str):
            continue
        try:
            # GA4 returns `YYYYMMDD` with no separators, which `fromisoformat` accepts.
            buckets.append(date.fromisoformat(raw))
        except ValueError:
            continue
    if not buckets:
        return None

    return gap_disclosure(
        buckets,
        window_end=window_end,
        # A `date` dimension is one row per day by construction.
        interval="day",
        # No rename case, unlike a PostHog event: a GA4 metric name is fixed by Google. What
        # replaces it is the tag -- the property keeps existing while nothing reports to it.
        as_of=as_of,
        alternatives=(
            "the property may have stopped receiving hits, the measurement tag may have been "
            "removed from the site, or the sessions may genuinely have stopped"
        ),
    )
