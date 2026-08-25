"""Mixpanel connector — product analytics for tenants that do not run PostHog.

The second product-analytics source, and the reason it exists is portability rather than
capability. Cortex's value is the investigation, not the vendor: a company on Mixpanel should be
able to paste a service-account credential and get the same grounded answers a PostHog tenant
gets. So this deliberately mirrors `posthog.py` — the same three questions, the same result
shapes — rather than exposing Mixpanel's full surface.

**What it deliberately does not have.** PostHog's causal advantage is that annotations, feature
flags and experiments live beside the events, so "what changed" is answerable from one source.
Mixpanel has no equivalent, and pretending otherwise would be worse than the gap: an analyst told
a change-log exists will go looking for one. A Mixpanel tenant establishes "what changed" from
GitHub and Slack, which is the same route a GA4 tenant takes.

**Authentication is a service account, not a project token.** HTTP Basic, username and secret,
with `project_id` on every request — verified against Mixpanel's own docs rather than assumed.
The two halves are stored as one `username:secret` string in the vault, because they are useless
apart and the vault's job is to hold one secret per provider. The non-secret half of the
configuration — project id and region — lives in `credential_metadata` alongside it.

**Region is configuration, not a guess.** Mixpanel partitions data between `mixpanel.com`,
`eu.mixpanel.com` and `in.mixpanel.com`, and the regions do not share data. A key issued in the
EU returns an error against the US host that reads exactly like a bad key, which is the same trap
PostHog's `KNOWN_HOSTS` exists to close.
"""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import date
from typing import Any

import httpx

from cortex.analysis.coverage import gap_disclosure
from cortex.db.models import CredentialProvider
from cortex.tools.base import (
    Capability,
    Freshness,
    InvalidParams,
    ToolContext,
    ToolResult,
)
from cortex.tools.base import Tool as BaseTool
from cortex.tools.disclosures import grade_series, series_disclosures
from cortex.tools.http import DEFAULT_TIMEOUT, request_json

#: The qualified capability name the shared assembler dispatches on.
_EVENT_TREND = "mixpanel__event_trend"

#: Mixpanel's data-residency hosts. The regions do not share data, so this is an allowlist
#: rather than a hint: an unknown region is refused instead of guessed at.
REGIONS = {
    "us": "https://mixpanel.com",
    "eu": "https://eu.mixpanel.com",
    "in": "https://in.mixpanel.com",
}

#: Bucket widths the segmentation query accepts, allowlisted because the value is sent verbatim.
UNITS = ("minute", "hour", "day", "month")

#: How a count is computed. `general` totals every occurrence, `unique` counts distinct users.
#: Both are needed: "signups fell" is a `general` question and "fewer people signed up" is a
#: `unique` one, and conflating them has produced wrong answers elsewhere in this codebase.
TYPES = ("general", "unique", "average")

_MAX_ROWS = 500
_DEFAULT_ROWS = 100

_DATE = {
    "type": "string",
    "pattern": r"^\d{4}-\d{2}-\d{2}$",
    "description": "Date as YYYY-MM-DD, inclusive.",
}

#: Conservative pattern for an event name before it is placed in a query string. Mixpanel
#: accepts almost anything as an event name, and this module builds URLs from them.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_$ .:\-/]{1,255}$")


class MixpanelTool(BaseTool):
    name = "mixpanel"
    provider = CredentialProvider.MIXPANEL

    def capabilities(self) -> list[Capability]:
        return [
            Capability(
                name="list_events",
                description=(
                    "The event names this project actually defines. Call this before any "
                    "trend: event names are project-specific and a guessed name returns an "
                    "empty series that looks exactly like a real drop to zero."
                ),
                params_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": _MAX_ROWS,
                            "default": _DEFAULT_ROWS,
                        }
                    },
                },
                handler=self.list_events,
                result_key="events",
                # Enumerates what exists, so the investigator runs it before the first step.
                discovery=True,
            ),
            Capability(
                name="event_trend",
                description=(
                    "A time series for one event, optionally broken down by a property. "
                    "The main measurement capability: use it to establish whether a metric "
                    "moved, and over what period."
                ),
                params_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["event", "from_date", "to_date"],
                    "properties": {
                        "event": {
                            "type": "string",
                            "description": ("Event name, exactly as list_events reports it."),
                        },
                        "from_date": _DATE,
                        "to_date": _DATE,
                        "unit": {
                            "type": "string",
                            "enum": list(UNITS),
                            "default": "day",
                            "description": (
                                "Bucket width. Prefer 'day' when checking whether something "
                                "changed: a 'month' bucket covering a partial month is not "
                                "comparable with the whole ones beside it."
                            ),
                        },
                        "measure": {
                            "type": "string",
                            "enum": list(TYPES),
                            "default": "general",
                            "description": (
                                "'general' counts occurrences, 'unique' counts distinct "
                                "users. 'Signups fell' and 'fewer people signed up' are "
                                "different questions."
                            ),
                        },
                        "breakdown_property": {
                            "type": "string",
                            "description": "Property to segment by, e.g. '$os' or 'plan'.",
                        },
                    },
                },
                handler=self.event_trend,
                result_key="series",
            ),
        ]

    async def list_events(self, ctx: ToolContext, *, limit: int = _DEFAULT_ROWS) -> ToolResult:
        raw = await self._get(
            ctx, "/api/query/events/names", {"type": "general", "limit": int(limit)}
        )
        # This endpoint answers with a bare JSON array rather than an object. `request_json`
        # wraps a non-dict body under `data` precisely so a capability always has a dict to
        # build a payload from, so the array arrives there.
        names = raw.get("data")
        events = [n for n in names if isinstance(n, str)] if isinstance(names, list) else []
        return ToolResult(
            payload={"event_count": len(events), "events": events},
            source_ref=f"mixpanel://projects/{self._project_id(ctx)}/events/names",
            meta={"freshness": Freshness.LIVE},
        )

    async def event_trend(
        self,
        ctx: ToolContext,
        *,
        event: str,
        from_date: str,
        to_date: str,
        unit: str = "day",
        measure: str = "general",
        breakdown_property: str | None = None,
    ) -> ToolResult:
        _check_range(from_date, to_date)
        if unit not in UNITS:
            raise InvalidParams(f"mixpanel: unit must be one of {', '.join(UNITS)}")
        if measure not in TYPES:
            raise InvalidParams(f"mixpanel: measure must be one of {', '.join(TYPES)}")
        _reject_unsafe(event, field="event")

        params: dict[str, Any] = {
            "event": event,
            "from_date": from_date,
            "to_date": to_date,
            "unit": unit,
            "type": measure,
        }
        if breakdown_property:
            _reject_unsafe(breakdown_property, field="breakdown_property")
            # Mixpanel's segmentation expression grammar. Composed here from a validated
            # property name rather than accepted as free text from the model, for the same
            # reason `posthog.py` composes HogQL: an expression is a small language, and a
            # model-authored one is unreviewable before it runs.
            params["on"] = f'properties["{breakdown_property}"]'

        raw = await self._get(ctx, "/api/query/segmentation", params)
        series = _flatten(raw)

        payload: dict[str, Any] = {
            "event": event,
            "measure": measure,
            "unit": unit,
            "from_date": from_date,
            "to_date": to_date,
            "breakdown_property": breakdown_property,
            "row_count": len(series),
            "series": series,
            # Returned so an empty series can be told apart from a real drop to zero — the
            # same disclosure `posthog.event_trend` makes, for the same reason.
            "total": sum(row["value"] for row in series),
        }
        # Through the shared assembler, which the eval harness also calls, so the suite measures
        # the disclosures production actually makes. Mixpanel had `partial_buckets` and no gap
        # and no grade at all: it could stop collecting mid-range while the payload said only
        # that the trailing month was short -- the two invite opposite conclusions, and it was
        # disclosing the reassuring one.
        request = {"from_date": from_date, "to_date": to_date, "unit": unit}
        payload.update(series_disclosures(_EVENT_TREND, payload, request))
        payload.update(grade_series(_EVENT_TREND, payload))
        return ToolResult(
            payload=payload,
            source_ref=(
                f"mixpanel://projects/{self._project_id(ctx)}/segmentation"
                f"?event={event}&from={from_date}&to={to_date}"
            ),
            meta={"freshness": Freshness.LIVE},
        )

    # -- plumbing ---------------------------------------------------------------------

    def _project_id(self, ctx: ToolContext) -> str:
        project = str(ctx.credential_metadata.get("project_id") or "").strip()
        if not project:
            raise InvalidParams(
                "mixpanel: no project_id on this credential. Reconnect with "
                "--meta project_id=<id>; the Query API rejects service-account requests "
                "without one."
            )
        return project

    def _host(self, ctx: ToolContext) -> str:
        region = str(ctx.credential_metadata.get("region") or "us").strip().lower()
        host = REGIONS.get(region)
        if host is None:
            raise InvalidParams(
                f"mixpanel: unknown region {region!r}. Mixpanel partitions data between "
                f"{', '.join(sorted(REGIONS))} and they do not share it — a key issued in "
                "one region fails against another in a way that looks like a bad key."
            )
        return host

    def _auth(self, ctx: ToolContext) -> tuple[str, str]:
        """The service account username and secret, stored as one `username:secret` string.

        Split on the first colon only: a secret may legitimately contain one, and splitting on
        every colon would corrupt it into something that fails authentication for a reason
        nobody could see.
        """
        raw = ctx.credential or ""
        username, sep, secret = raw.partition(":")
        if not sep or not username or not secret:
            raise InvalidParams(
                "mixpanel: the credential must be 'username:secret' from a Mixpanel service "
                "account. A project token will not work — the Query API authenticates with "
                "service accounts."
            )
        return username, secret

    def _client(self, ctx: ToolContext) -> httpx.AsyncClient:
        """The configured client. Separate from `_get` so tests can exercise the real headers,
        base URL and auth while replacing only the socket."""
        username, secret = self._auth(ctx)
        return httpx.AsyncClient(
            base_url=self._host(ctx),
            auth=httpx.BasicAuth(username, secret),
            headers={"Accept": "application/json"},
            timeout=DEFAULT_TIMEOUT,
        )

    async def _get(self, ctx: ToolContext, path: str, params: dict[str, Any]) -> Any:
        async with self._client(ctx) as client:
            return await request_json(
                client,
                "GET",
                path,
                tool=self.name,
                params={**params, "project_id": self._project_id(ctx)},
            )


def _reject_unsafe(value: str, *, field: str) -> str:
    if not _SAFE_NAME.match(value):
        raise InvalidParams(
            f"mixpanel: {field} {value!r} contains characters that cannot be used in a "
            "query. Call list_events to see the exact names this project defines."
        )
    return value


def _check_range(from_date: str, to_date: str) -> None:
    if from_date > to_date:
        raise InvalidParams(f"mixpanel: from_date {from_date} is after to_date {to_date}")


def _flatten(raw: Any) -> list[dict[str, Any]]:
    """Turn Mixpanel's nested `{series, values}` into flat, ordered rows.

    Mixpanel answers with a list of dates and a mapping of segment to date to count. Handing
    the model that shape would make the meaning of a number depend on how deep it sits, which
    is the same mistake positional rows are: it reads fine until one lookup goes to the wrong
    level and the resulting claim is confidently wrong.
    """
    data = raw.get("data") if isinstance(raw, dict) else None
    if not isinstance(data, dict):
        return []
    dates = [d for d in (data.get("series") or []) if isinstance(d, str)]
    values = data.get("values")
    if not isinstance(values, dict):
        return []

    rows: list[dict[str, Any]] = []
    # Segments sorted for determinism; dates in the order Mixpanel returned them, which is
    # chronological and is the order a reader expects a series in.
    for segment in sorted(values):
        counts = values[segment]
        if not isinstance(counts, dict):
            continue
        for bucket in dates:
            value = counts.get(bucket, 0)
            row: dict[str, Any] = {
                "bucket": bucket,
                "value": value if isinstance(value, int) else 0,
            }
            # Only present when there really is a breakdown. A `segment` equal to the event
            # name is Mixpanel's unsegmented answer, not a real dimension, and echoing it as
            # one would invent a breakdown the analyst never asked for.
            if len(values) > 1 or segment not in ("", "undefined"):
                row["segment"] = segment
            rows.append(row)
    return rows


def _partial_bucket_note(from_date: str, to_date: str, unit: str) -> dict[str, Any] | None:
    """Disclose a month bucket that the requested range only partly covers.

    The same failure `posthog._bucket_coverage` exists for, and it is not vendor-specific: a
    total for twelve days of a month sitting beside totals for whole months invites exactly one
    comparison, and that comparison is meaningless. Mixpanel returns no such warning, so this
    is computed from the range the caller asked for.

    Narrower than the PostHog version on purpose: that one reads bucket starts back from the
    response, which Mixpanel's flattened shape does not carry as reliably, so this reports only
    the case it can be certain of — a month unit whose endpoints are not month boundaries.
    """
    if unit != "month":
        return None

    try:
        start = date.fromisoformat(from_date)
        end = date.fromisoformat(to_date)
    except ValueError:
        return None

    partial = []
    if start.day != 1:
        partial.append({"bucket": start.replace(day=1).isoformat(), "reason": "starts mid-month"})
    if end.day != monthrange(end.year, end.month)[1]:
        partial.append(
            {
                "bucket": end.replace(day=1).isoformat(),
                "reason": "ends mid-month",
                "days_covered": end.day,
                "days_in_full_bucket": monthrange(end.year, end.month)[1],
            }
        )
    if not partial:
        return None
    return {
        "partial_buckets": partial,
        "note": (
            "These month buckets are only partly covered by the requested range. Their totals "
            "are not comparable with the whole months beside them; compare rates per day, or "
            "request whole months."
        ),
    }


def _gap(
    series: list[dict[str, Any]], to_date: str, unit: str, *, as_of: date | None = None
) -> dict[str, Any] | None:
    """Whether the series stops before the requested range does.

    Mixpanel computed `partial_buckets` and nothing else, which discloses "the last month is
    young" while staying silent on "collection stopped". Those invite opposite conclusions from
    the same shape of gap -- calm versus escalate -- so disclosing only the first is worse than
    disclosing neither.
    """
    if not series:
        return None
    try:
        window_end = date.fromisoformat(to_date)
    except ValueError:
        return None

    buckets: list[date] = []
    for row in series:
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
        interval=unit,
        # Mixpanel events can be renamed like PostHog's, and additionally hidden from the
        # project's schema while still being ingested -- which looks identical to a stop.
        as_of=as_of,
        alternatives=(
            "the event may have stopped firing, tracking may be broken, or the event may have "
            "been renamed or hidden from the project schema"
        ),
    )
