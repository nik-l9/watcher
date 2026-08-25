"""HubSpot connector — pipeline, revenue and account context.

This supplies the commercial half of an investigation: whether a traffic change
actually reached revenue, which segments moved, and what customers were saying at
the time.

Uses HubSpot's CRM v3 search endpoints so filtering happens upstream rather than by
pulling every record and filtering locally — the difference between one request and
several hundred for a real portal.

Read-only: a private-app token with read scopes only. No capability here creates,
updates or deletes a record.
"""

from __future__ import annotations

from datetime import UTC
from typing import Any

import httpx

from cortex.db.models import CredentialProvider
from cortex.tools.base import Capability, InvalidParams, ToolContext, ToolError, ToolResult
from cortex.tools.base import Tool as BaseTool
from cortex.tools.http import DEFAULT_TIMEOUT, SEARCH_TIMEOUT, request_json

API_ROOT = "https://api.hubapi.com"

# HubSpot caps search results at 100 per page.
_MAX_LIMIT = 100

#: Ceiling on pages followed for one search. A bound is needed because a broad filter can
#: match an entire portal, and each page is a round trip against a deadline. Set so the
#: default limit of 50 and the maximum of 100 both complete in one or two pages, while a
#: pathological filter cannot walk a hundred thousand records.
_MAX_PAGES = 10

_DEAL_PROPERTIES = [
    "dealname",
    "amount",
    "dealstage",
    "pipeline",
    "closedate",
    "createdate",
    "hs_lastmodifieddate",
    "hs_deal_stage_probability",
    "hubspot_owner_id",
    # Added after a live investigation reported "closed-won records contain no deal
    # owner, lead source, region, or channel fields, so segmentation by GTM segment was
    # not possible" — it could measure the change but not decompose it, which is half of
    # what an investigation is for. `hubspot_owner_id` was already requested but is an
    # opaque numeric id, so it is resolved to a name below.
    "dealtype",
    "hs_analytics_source",
    "hs_analytics_source_data_1",
    "num_associated_contacts",
]

_CONTACT_PROPERTIES = [
    "email",
    "firstname",
    "lastname",
    "company",
    "jobtitle",
    "lifecyclestage",
    "hs_lead_status",
    "createdate",
    "lastmodifieddate",
]

_COMPANY_PROPERTIES = [
    "name",
    "domain",
    "industry",
    "numberofemployees",
    "annualrevenue",
    "lifecyclestage",
    "createdate",
    "country",
]

_ISO_DATE = {
    "type": "string",
    "pattern": r"^\d{4}-\d{2}-\d{2}$",
    "description": "Date as YYYY-MM-DD.",
}

_LIMIT = {"type": "integer", "minimum": 1, "maximum": _MAX_LIMIT, "default": 50}


class HubSpotTool(BaseTool):
    name = "hubspot"
    provider = CredentialProvider.HUBSPOT

    def capabilities(self) -> list[Capability]:
        return [
            Capability(
                name="pipeline",
                description=(
                    "Open deals with stage, amount and expected close date. Use this "
                    "to check whether a funnel change has reached pipeline yet."
                ),
                params_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "pipeline_id": {
                            "type": "string",
                            "description": "Restrict to one pipeline. Omit for all.",
                        },
                        "min_amount": {"type": "number", "minimum": 0},
                        "limit": _LIMIT,
                    },
                },
                handler=self.pipeline,
                result_key="stages",
            ),
            Capability(
                name="closed_won",
                description=(
                    "Deals closed won within a date range, with amounts. This is the "
                    "revenue ground truth an investigation should land on."
                ),
                params_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["start_date", "end_date"],
                    "properties": {
                        "start_date": _ISO_DATE,
                        "end_date": _ISO_DATE,
                        "limit": _LIMIT,
                    },
                },
                handler=self.closed_won,
                result_key="deals",
            ),
            Capability(
                name="contacts",
                description=(
                    "Contacts created or modified in a date range, with lifecycle "
                    "stage. Use this to confirm whether a signup change appears in CRM."
                ),
                params_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["start_date", "end_date"],
                    "properties": {
                        "start_date": _ISO_DATE,
                        "end_date": _ISO_DATE,
                        "lifecycle_stage": {
                            "type": "string",
                            "description": "e.g. 'lead', 'marketingqualifiedlead', 'customer'.",
                        },
                        "limit": _LIMIT,
                    },
                },
                handler=self.contacts,
                result_key="contacts",
            ),
            Capability(
                name="companies",
                description="Search companies by name or domain, with firmographics.",
                params_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "query": {
                            "type": "string",
                            "minLength": 2,
                            "description": "Name or domain fragment.",
                        },
                        "limit": _LIMIT,
                    },
                },
                handler=self.companies,
                result_key="companies",
            ),
            Capability(
                name="activities",
                description=(
                    "Recent engagements (calls, emails, meetings, notes) for one "
                    "company or contact, to establish what was actually said."
                ),
                params_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["object_type", "object_id"],
                    "properties": {
                        "object_type": {"type": "string", "enum": ["companies", "contacts"]},
                        "object_id": {"type": "string", "pattern": r"^\d+$"},
                        "limit": _LIMIT,
                    },
                },
                handler=self.activities,
                result_key="activities",
            ),
        ]

    # ------------------------------------------------------------------ helpers

    def _client(self, ctx: ToolContext) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=API_ROOT,
            timeout=DEFAULT_TIMEOUT,
            headers={
                "Authorization": f"Bearer {ctx.credential}",
                "Content-Type": "application/json",
            },
        )

    async def _owners(self, ctx: ToolContext) -> dict[str, str]:
        """Map owner id to a human name, so deals can be segmented by rep.

        One extra request per deal-returning call. Worth it: `hubspot_owner_id` is an
        opaque number, and a report that groups by it produces rows a reader cannot act
        on. A failure here is swallowed deliberately — an unresolvable owner should
        degrade the segmentation, not fail the investigation.
        """
        try:
            async with self._client(ctx) as client:
                raw = await request_json(
                    client, "GET", "/crm/v3/owners", tool=self.name, params={"limit": 500}
                )
        except ToolError:
            return {}

        out: dict[str, str] = {}
        for row in raw.get("results", []):
            name = " ".join(
                part for part in (row.get("firstName"), row.get("lastName")) if part
            ).strip()
            identifier = row.get("id")
            if identifier is not None:
                out[str(identifier)] = name or row.get("email") or str(identifier)
        return out

    async def _search(
        self,
        ctx: ToolContext,
        object_type: str,
        *,
        filters: list[dict[str, Any]],
        properties: list[str],
        sorts: list[dict[str, str]] | None = None,
        limit: int,
    ) -> dict[str, Any]:
        """Search, following pagination up to `limit` results.

        HubSpot caps a page at 100 and returns `paging.next.after`. Returning the first
        page only was silently lossy: a live investigation asked for the open pipeline,
        received 50 of 68 matching deals, and could only report the gap because it
        happened to compare the returned count against the reported total. A connector
        that quietly returns a prefix produces a confident answer about a subset, which
        is worse than an error.
        """
        collected: list[dict[str, Any]] = []
        after: str | None = None
        total: int | None = None
        pages = 0

        while len(collected) < limit and pages < _MAX_PAGES:
            body: dict[str, Any] = {
                "filterGroups": [{"filters": filters}] if filters else [],
                "properties": properties,
                "limit": min(limit - len(collected), _MAX_LIMIT),
            }
            if sorts:
                body["sorts"] = sorts
            if after:
                body["after"] = after

            async with self._client(ctx) as client:
                page = await request_json(
                    client,
                    "POST",
                    f"/crm/v3/objects/{object_type}/search",
                    tool=self.name,
                    json_body=body,
                    # Portal-wide search, not a keyed read: observed timing out twice in
                    # one live investigation at the 30-second default.
                    timeout=SEARCH_TIMEOUT,
                )

            pages += 1
            collected.extend(page.get("results", []))
            if total is None:
                total = page.get("total")
            after = ((page.get("paging") or {}).get("next") or {}).get("after")
            if not after:
                break

        return {
            "results": collected[:limit],
            "total": total,
            # Stated rather than implied, so a report can disclose its own coverage. The
            # analyst cannot know it saw a subset unless the connector says so.
            "returned": len(collected[:limit]),
            "truncated": bool(after) or (total is not None and len(collected) < total),
            "pages_fetched": pages,
        }

    # ------------------------------------------------------------------ capabilities

    async def pipeline(
        self,
        ctx: ToolContext,
        *,
        pipeline_id: str | None = None,
        min_amount: float | None = None,
        limit: int = 50,
    ) -> ToolResult:
        filters: list[dict[str, Any]] = [
            # Open deals only: HubSpot marks won/lost via hs_is_closed.
            {"propertyName": "hs_is_closed", "operator": "EQ", "value": "false"}
        ]
        if pipeline_id:
            filters.append({"propertyName": "pipeline", "operator": "EQ", "value": pipeline_id})
        if min_amount is not None:
            filters.append({"propertyName": "amount", "operator": "GTE", "value": str(min_amount)})

        raw = await self._search(
            ctx,
            "deals",
            filters=filters,
            properties=_DEAL_PROPERTIES,
            sorts=[{"propertyName": "amount", "direction": "DESCENDING"}],
            limit=limit,
        )
        owners = await self._owners(ctx)
        deals = [_deal(record, owners) for record in _results(raw)]
        return ToolResult(
            payload={
                "pipeline_id": pipeline_id,
                "total_matching": raw.get("total"),
                "count": len(deals),
                "total_amount": _sum_amounts(deals),
                "deals": deals,
            },
            source_ref="hubspot://crm/v3/objects/deals/search?open=true",
        )

    async def closed_won(
        self, ctx: ToolContext, *, start_date: str, end_date: str, limit: int = 50
    ) -> ToolResult:
        _check_range(start_date, end_date)
        raw = await self._search(
            ctx,
            "deals",
            filters=[
                {"propertyName": "hs_is_closed_won", "operator": "EQ", "value": "true"},
                {
                    "propertyName": "closedate",
                    "operator": "BETWEEN",
                    "value": _epoch_ms(start_date),
                    "highValue": _epoch_ms(end_date, end_of_day=True),
                },
            ],
            properties=_DEAL_PROPERTIES,
            sorts=[{"propertyName": "closedate", "direction": "DESCENDING"}],
            limit=limit,
        )
        owners = await self._owners(ctx)
        deals = [_deal(record, owners) for record in _results(raw)]
        return ToolResult(
            payload={
                "start_date": start_date,
                "end_date": end_date,
                "total_matching": raw.get("total"),
                "count": len(deals),
                "total_amount": _sum_amounts(deals),
                "deals": deals,
            },
            source_ref=(
                f"hubspot://crm/v3/objects/deals/search?closed_won={start_date}..{end_date}"
            ),
        )

    async def contacts(
        self,
        ctx: ToolContext,
        *,
        start_date: str,
        end_date: str,
        lifecycle_stage: str | None = None,
        limit: int = 50,
    ) -> ToolResult:
        _check_range(start_date, end_date)
        filters: list[dict[str, Any]] = [
            {
                "propertyName": "createdate",
                "operator": "BETWEEN",
                "value": _epoch_ms(start_date),
                "highValue": _epoch_ms(end_date, end_of_day=True),
            }
        ]
        if lifecycle_stage:
            filters.append(
                {"propertyName": "lifecyclestage", "operator": "EQ", "value": lifecycle_stage}
            )

        raw = await self._search(
            ctx,
            "contacts",
            filters=filters,
            properties=_CONTACT_PROPERTIES,
            sorts=[{"propertyName": "createdate", "direction": "DESCENDING"}],
            limit=limit,
        )
        records = [_contact(record) for record in _results(raw)]
        by_stage: dict[str, int] = {}
        for record in records:
            stage = record.get("lifecycle_stage") or "unknown"
            by_stage[stage] = by_stage.get(stage, 0) + 1

        return ToolResult(
            payload={
                "start_date": start_date,
                "end_date": end_date,
                "lifecycle_stage": lifecycle_stage,
                "total_matching": raw.get("total"),
                "count": len(records),
                "by_lifecycle_stage": by_stage,
                "contacts": records,
            },
            source_ref=(
                f"hubspot://crm/v3/objects/contacts/search?created={start_date}..{end_date}"
            ),
        )

    async def companies(
        self, ctx: ToolContext, *, query: str | None = None, limit: int = 50
    ) -> ToolResult:
        body: dict[str, Any] = {
            "properties": _COMPANY_PROPERTIES,
            "limit": min(limit, _MAX_LIMIT),
        }
        if query:
            # `query` is HubSpot's own full-text search across the object, which
            # matches name and domain without needing to guess which field applies.
            body["query"] = query

        async with self._client(ctx) as client:
            raw = await request_json(
                client,
                "POST",
                "/crm/v3/objects/companies/search",
                tool=self.name,
                json_body=body,
            )
        records = [_company(record) for record in _results(raw)]
        return ToolResult(
            payload={
                "query": query,
                "total_matching": raw.get("total"),
                "count": len(records),
                "companies": records,
            },
            source_ref=f"hubspot://crm/v3/objects/companies/search?q={query or ''}",
        )

    async def activities(
        self, ctx: ToolContext, *, object_type: str, object_id: str, limit: int = 50
    ) -> ToolResult:
        engagement_types = ("notes", "calls", "emails", "meetings")
        activities: list[dict[str, Any]] = []

        async with self._client(ctx) as client:
            for engagement in engagement_types:
                # Associations first, then a batch read: HubSpot has no single
                # endpoint returning every engagement for an object.
                assoc = await request_json(
                    client,
                    "GET",
                    f"/crm/v4/objects/{object_type}/{object_id}/associations/{engagement}",
                    tool=self.name,
                    params={"limit": min(limit, _MAX_LIMIT)},
                )
                ids = [
                    str(row.get("toObjectId"))
                    for row in _as_list(assoc.get("results"))
                    if row.get("toObjectId") is not None
                ]
                if not ids:
                    continue

                detail = await request_json(
                    client,
                    "POST",
                    f"/crm/v3/objects/{engagement}/batch/read",
                    tool=self.name,
                    json_body={
                        "properties": _ENGAGEMENT_PROPERTIES.get(engagement, ["hs_timestamp"]),
                        "inputs": [{"id": i} for i in ids[:limit]],
                    },
                )
                for record in _results(detail):
                    props = record.get("properties") or {}
                    activities.append(
                        {
                            "type": engagement,
                            "id": record.get("id"),
                            "timestamp": props.get("hs_timestamp") or props.get("hs_createdate"),
                            "summary": _engagement_summary(engagement, props),
                        }
                    )

        # Newest first, with undated records last rather than sorted as epoch zero.
        # The first key element is `is not None` because reverse=True sorts True
        # before False — with `is None` the undated records would come first.
        activities.sort(
            key=lambda a: (a["timestamp"] is not None, a["timestamp"] or ""), reverse=True
        )
        return ToolResult(
            payload={
                "object_type": object_type,
                "object_id": object_id,
                "count": len(activities),
                "activities": activities[:limit],
            },
            source_ref=f"hubspot://crm/v3/objects/{object_type}/{object_id}/engagements",
        )


_ENGAGEMENT_PROPERTIES = {
    "notes": ["hs_timestamp", "hs_note_body"],
    "calls": ["hs_timestamp", "hs_call_title", "hs_call_body", "hs_call_duration"],
    "emails": ["hs_timestamp", "hs_email_subject", "hs_email_text"],
    "meetings": ["hs_timestamp", "hs_meeting_title", "hs_meeting_body"],
}


# ---------------------------------------------------------------------- parsing


def _check_range(start: str, end: str) -> None:
    if start > end:
        raise InvalidParams(f"start_date {start} is after end_date {end}")


def _epoch_ms(date: str, *, end_of_day: bool = False) -> str:
    """HubSpot date filters take epoch milliseconds.

    end_of_day makes BETWEEN inclusive of the final day; without it a range ending
    today would silently exclude everything that happened today.
    """
    from datetime import datetime, time

    year, month, day = (int(part) for part in date.split("-"))
    moment = datetime.combine(
        datetime(year, month, day, tzinfo=UTC).date(),
        time.max if end_of_day else time.min,
        tzinfo=UTC,
    )
    return str(int(moment.timestamp() * 1000))


def _results(raw: dict[str, Any]) -> list[dict[str, Any]]:
    return _as_list(raw.get("results"))


def _as_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    return []


def _number(value: Any) -> float | int | None:
    """HubSpot returns numerics as strings, and empty string for unset."""
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def _deal(record: dict[str, Any], owners: dict[str, str] | None = None) -> dict[str, Any]:
    props = record.get("properties") or {}
    owner_id = props.get("hubspot_owner_id")
    return {
        "id": record.get("id"),
        "name": props.get("dealname"),
        "amount": _number(props.get("amount")),
        "stage": props.get("dealstage"),
        "pipeline": props.get("pipeline"),
        "close_date": props.get("closedate"),
        "created_at": props.get("createdate"),
        "probability": _number(props.get("hs_deal_stage_probability")),
        # The segmentation fields. Without an owner and a source, an investigation can
        # measure that revenue moved but cannot say which rep or channel moved it, which
        # is the half of the answer that is actionable.
        "owner": (owners or {}).get(str(owner_id)) or None,
        "owner_id": owner_id,
        "deal_type": props.get("dealtype"),
        "source": props.get("hs_analytics_source"),
        "source_detail": props.get("hs_analytics_source_data_1"),
        "associated_contacts": _number(props.get("num_associated_contacts")),
    }


def _contact(record: dict[str, Any]) -> dict[str, Any]:
    props = record.get("properties") or {}
    return {
        "id": record.get("id"),
        "email": props.get("email"),
        "name": " ".join(part for part in (props.get("firstname"), props.get("lastname")) if part)
        or None,
        "company": props.get("company"),
        "job_title": props.get("jobtitle"),
        "lifecycle_stage": props.get("lifecyclestage"),
        "lead_status": props.get("hs_lead_status"),
        "created_at": props.get("createdate"),
    }


def _company(record: dict[str, Any]) -> dict[str, Any]:
    props = record.get("properties") or {}
    return {
        "id": record.get("id"),
        "name": props.get("name"),
        "domain": props.get("domain"),
        "industry": props.get("industry"),
        "employees": _number(props.get("numberofemployees")),
        "annual_revenue": _number(props.get("annualrevenue")),
        "lifecycle_stage": props.get("lifecyclestage"),
        "country": props.get("country"),
        "created_at": props.get("createdate"),
    }


def _sum_amounts(deals: list[dict[str, Any]]) -> float | int | None:
    """Total of the returned deals.

    None when no deal carries an amount, rather than 0: an absent amount and a
    genuine zero are different facts, and reporting 0 would be a fabricated figure.
    """
    amounts = [d["amount"] for d in deals if d.get("amount") is not None]
    if not amounts:
        return None
    total = sum(amounts)
    return int(total) if float(total).is_integer() else round(total, 2)


def _engagement_summary(engagement: str, props: dict[str, Any]) -> str | None:
    """A short, loggable description. Bodies are truncated — they are often long
    and frequently contain customer PII."""
    candidates = {
        "notes": props.get("hs_note_body"),
        "calls": props.get("hs_call_title") or props.get("hs_call_body"),
        "emails": props.get("hs_email_subject"),
        "meetings": props.get("hs_meeting_title") or props.get("hs_meeting_body"),
    }
    text = candidates.get(engagement)
    if not text:
        return None
    collapsed = " ".join(str(text).split())
    return collapsed if len(collapsed) <= 500 else collapsed[:500] + "…"
