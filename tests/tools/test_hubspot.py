"""HubSpot connector.

Two things matter most: HubSpot returns numerics as strings and unset values as
empty strings, so typing has to be deliberate; and its date filters take epoch
milliseconds, where an inclusive upper bound is easy to get wrong in a way that
silently drops the most recent day of data.
"""

from __future__ import annotations

import json
from datetime import UTC

import httpx
import pytest

from cortex.tools.base import InvalidParams, ToolContext
from cortex.tools.http import AuthRejected
from cortex.tools.hubspot import HubSpotTool, _epoch_ms, _sum_amounts

DEAL = {
    "id": "701",
    "properties": {
        "dealname": "Acme Enterprise",
        "amount": "48000",
        "dealstage": "contractsent",
        "pipeline": "default",
        "closedate": "2026-07-25T00:00:00Z",
        "createdate": "2026-06-01T00:00:00Z",
        "hs_deal_stage_probability": "0.8",
    },
}

CONTACT = {
    "id": "9001",
    "properties": {
        "email": "buyer@acme.com",
        "firstname": "Dana",
        "lastname": "Reed",
        "company": "Acme",
        "jobtitle": "Head of Growth",
        "lifecyclestage": "marketingqualifiedlead",
        "createdate": "2026-07-20T09:00:00Z",
    },
}


@pytest.fixture
def tool() -> HubSpotTool:
    return HubSpotTool()


@pytest.fixture
def ctx(tenant: object) -> ToolContext:
    return ToolContext(tenant=tenant, credential="pat-na1-fake")  # type: ignore[arg-type]


class TestPipeline:
    async def test_returns_open_deals_with_totals(
        self, tool: HubSpotTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(tool, {"/deals/search": {"total": 12, "results": [DEAL]}})
        result = await tool.pipeline(ctx)

        assert result.payload["total_matching"] == 12
        assert result.payload["total_amount"] == 48000
        deal = result.payload["deals"][0]
        assert deal["amount"] == 48000 and isinstance(deal["amount"], int)
        assert deal["probability"] == 0.8

    async def test_filters_to_open_deals(
        self, tool: HubSpotTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Without this filter "pipeline" would include closed-lost deals."""
        transport = patch_client(tool, {"/deals/search": {"total": 0, "results": []}})
        await tool.pipeline(ctx)
        filters = transport.request_bodies()[0]["filterGroups"][0]["filters"]
        assert {"propertyName": "hs_is_closed", "operator": "EQ", "value": "false"} in filters

    async def test_optional_filters_are_forwarded(
        self, tool: HubSpotTool, ctx: ToolContext, patch_client: object
    ) -> None:
        transport = patch_client(tool, {"/deals/search": {"total": 0, "results": []}})
        await tool.pipeline(ctx, pipeline_id="enterprise", min_amount=10000)
        filters = transport.request_bodies()[0]["filterGroups"][0]["filters"]
        names = {f["propertyName"] for f in filters}
        assert {"pipeline", "amount", "hs_is_closed"} <= names

    async def test_filtering_happens_upstream(
        self, tool: HubSpotTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """One search request, not a full-portal scan filtered locally.

        Counted as searches rather than as total requests: deals are now accompanied by
        one owner lookup, which resolves `hubspot_owner_id` to a name so results can be
        segmented by rep. The property under test is that the *filtering* happens
        upstream, and that is unaffected.
        """
        transport = patch_client(tool, {"/deals/search": {"total": 0, "results": []}})
        await tool.pipeline(ctx)
        searches = [r for r in transport.requests if "search" in str(r.url)]
        assert len(searches) == 1
        assert all(r.method == "GET" for r in transport.requests if r not in searches)


class TestPagination:
    """HubSpot caps a page at 100 and returns `paging.next.after`.

    Returning the first page only was silently lossy: a live investigation asked for the
    open pipeline and received 50 of 68 matching deals. It reported the gap only because
    it compared the returned count against the reported total — a connector that quietly
    returns a prefix produces a confident answer about a subset.
    """

    async def test_pages_are_followed_until_the_limit_is_met(
        self, tool: HubSpotTool, ctx: ToolContext, patch_client: object
    ) -> None:
        pages = [
            {
                "total": 3,
                "results": [{**DEAL, "id": "1"}, {**DEAL, "id": "2"}],
                "paging": {"next": {"after": "cursor-1"}},
            },
            {"total": 3, "results": [{**DEAL, "id": "3"}]},
        ]
        sent: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if "owners" in str(request.url):
                return httpx.Response(200, json={"results": []})
            sent.append(request)
            return httpx.Response(200, json=pages[min(len(sent) - 1, len(pages) - 1)])

        patch_client(tool, handler)
        result = await tool.closed_won(
            ctx, start_date="2026-07-01", end_date="2026-07-31", limit=10
        )

        assert len(sent) == 2, "the second page should have been requested"
        assert json.loads(sent[1].content)["after"] == "cursor-1"
        assert result.payload["count"] == 3
        assert result.payload["deals"][2]["id"] == "3"

    async def test_truncation_is_disclosed_rather_than_hidden(
        self, tool: HubSpotTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """The analyst cannot report its own coverage gap unless the connector says so."""

        def handler(request: httpx.Request) -> httpx.Response:
            if "owners" in str(request.url):
                return httpx.Response(200, json={"results": []})
            return httpx.Response(
                200,
                json={
                    "total": 68,
                    "results": [DEAL],
                    "paging": {"next": {"after": "more"}},
                },
            )

        patch_client(tool, handler)
        result = await tool.closed_won(ctx, start_date="2026-07-01", end_date="2026-07-31", limit=1)
        assert result.payload["total_matching"] == 68
        assert result.payload["count"] == 1

    async def test_a_single_page_does_not_request_a_second(
        self, tool: HubSpotTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """No `paging.next` means done. Requesting again would double every call."""
        sent: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if "owners" in str(request.url):
                return httpx.Response(200, json={"results": []})
            sent.append(request)
            return httpx.Response(200, json={"total": 1, "results": [DEAL]})

        patch_client(tool, handler)
        await tool.closed_won(ctx, start_date="2026-07-01", end_date="2026-07-31", limit=50)
        assert len(sent) == 1


class TestClosedWon:
    async def test_uses_an_inclusive_date_window(
        self, tool: HubSpotTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Without an end-of-day upper bound, a range ending today silently excludes
        everything that happened today."""
        transport = patch_client(tool, {"/deals/search": {"total": 1, "results": [DEAL]}})
        await tool.closed_won(ctx, start_date="2026-07-01", end_date="2026-07-31")

        filters = transport.request_bodies()[0]["filterGroups"][0]["filters"]
        window = next(f for f in filters if f["propertyName"] == "closedate")
        assert window["operator"] == "BETWEEN"
        assert int(window["highValue"]) > int(_epoch_ms("2026-07-31"))

    async def test_filters_to_won_deals(
        self, tool: HubSpotTool, ctx: ToolContext, patch_client: object
    ) -> None:
        transport = patch_client(tool, {"/deals/search": {"total": 0, "results": []}})
        await tool.closed_won(ctx, start_date="2026-07-01", end_date="2026-07-31")
        filters = transport.request_bodies()[0]["filterGroups"][0]["filters"]
        assert any(f["propertyName"] == "hs_is_closed_won" for f in filters)

    async def test_reversed_range_is_rejected(self, tool: HubSpotTool, ctx: ToolContext) -> None:
        with pytest.raises(InvalidParams, match="after end_date"):
            await tool.closed_won(ctx, start_date="2026-07-31", end_date="2026-07-01")


class TestContacts:
    async def test_summarises_by_lifecycle_stage(
        self, tool: HubSpotTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """The breakdown is what makes a signup change interpretable."""
        second = {
            "id": "9002",
            "properties": {"email": "b@acme.com", "lifecyclestage": "lead"},
        }
        patch_client(tool, {"/contacts/search": {"total": 2, "results": [CONTACT, second]}})
        result = await tool.contacts(ctx, start_date="2026-07-01", end_date="2026-07-31")

        assert result.payload["by_lifecycle_stage"] == {
            "marketingqualifiedlead": 1,
            "lead": 1,
        }

    async def test_missing_stage_is_labelled_unknown(
        self, tool: HubSpotTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Not silently folded into an existing bucket."""
        patch_client(
            tool, {"/contacts/search": {"total": 1, "results": [{"id": "1", "properties": {}}]}}
        )
        result = await tool.contacts(ctx, start_date="2026-07-01", end_date="2026-07-31")
        assert result.payload["by_lifecycle_stage"] == {"unknown": 1}

    async def test_builds_a_display_name(
        self, tool: HubSpotTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(tool, {"/contacts/search": {"total": 1, "results": [CONTACT]}})
        result = await tool.contacts(ctx, start_date="2026-07-01", end_date="2026-07-31")
        assert result.payload["contacts"][0]["name"] == "Dana Reed"

    async def test_absent_name_is_none_not_empty_string(
        self, tool: HubSpotTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(
            tool,
            {"/contacts/search": {"total": 1, "results": [{"id": "1", "properties": {}}]}},
        )
        result = await tool.contacts(ctx, start_date="2026-07-01", end_date="2026-07-31")
        assert result.payload["contacts"][0]["name"] is None


class TestCompanies:
    async def test_uses_full_text_search(
        self, tool: HubSpotTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """HubSpot's own `query` matches name and domain, so no field guessing."""
        transport = patch_client(
            tool,
            {
                "/companies/search": {
                    "total": 1,
                    "results": [
                        {
                            "id": "3001",
                            "properties": {
                                "name": "Acme",
                                "domain": "acme.com",
                                "numberofemployees": "450",
                                "annualrevenue": "12000000",
                            },
                        }
                    ],
                }
            },
        )
        result = await tool.companies(ctx, query="acme")
        assert transport.request_bodies()[0]["query"] == "acme"
        company = result.payload["companies"][0]
        assert company["employees"] == 450
        assert company["annual_revenue"] == 12000000

    async def test_query_is_omitted_when_absent(
        self, tool: HubSpotTool, ctx: ToolContext, patch_client: object
    ) -> None:
        transport = patch_client(tool, {"/companies/search": {"total": 0, "results": []}})
        await tool.companies(ctx)
        assert "query" not in transport.request_bodies()[0]


class TestActivities:
    async def test_collects_engagements_across_types(
        self, tool: HubSpotTool, ctx: ToolContext, patch_client: object
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "/associations/notes" in url:
                return httpx.Response(200, json={"results": [{"toObjectId": 11}]})
            if "/associations/" in url:
                return httpx.Response(200, json={"results": []})
            if "/notes/batch/read" in url:
                return httpx.Response(
                    200,
                    json={
                        "results": [
                            {
                                "id": "11",
                                "properties": {
                                    "hs_timestamp": "2026-07-21T10:00:00Z",
                                    "hs_note_body": "Customer flagged the new onboarding flow",
                                },
                            }
                        ]
                    },
                )
            return httpx.Response(200, json={"results": []})

        patch_client(tool, handler)
        result = await tool.activities(ctx, object_type="companies", object_id="3001")

        assert result.payload["count"] == 1
        activity = result.payload["activities"][0]
        assert activity["type"] == "notes"
        assert "onboarding" in activity["summary"]

    async def test_skips_batch_read_when_no_associations(
        self, tool: HubSpotTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """No wasted requests: four association lookups, no batch reads."""
        transport = patch_client(tool, lambda r: httpx.Response(200, json={"results": []}))
        result = await tool.activities(ctx, object_type="contacts", object_id="9001")
        assert result.payload["count"] == 0
        assert all("batch/read" not in str(r.url) for r in transport.requests)

    async def test_undated_activities_sort_last(
        self, tool: HubSpotTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """An activity with no timestamp must not sort as though it were epoch zero
        or, worse, as the newest."""

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "/associations/notes" in url:
                return httpx.Response(200, json={"results": [{"toObjectId": 1}, {"toObjectId": 2}]})
            if "/associations/" in url:
                return httpx.Response(200, json={"results": []})
            if "/notes/batch/read" in url:
                return httpx.Response(
                    200,
                    json={
                        "results": [
                            {"id": "1", "properties": {"hs_note_body": "undated"}},
                            {
                                "id": "2",
                                "properties": {
                                    "hs_timestamp": "2026-07-21T10:00:00Z",
                                    "hs_note_body": "dated",
                                },
                            },
                        ]
                    },
                )
            return httpx.Response(200, json={"results": []})

        patch_client(tool, handler)
        result = await tool.activities(ctx, object_type="companies", object_id="3001")
        summaries = [a["summary"] for a in result.payload["activities"]]
        assert summaries == ["dated", "undated"]

    async def test_long_bodies_are_truncated(
        self, tool: HubSpotTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Engagement bodies are long and frequently contain customer PII."""

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "/associations/notes" in url:
                return httpx.Response(200, json={"results": [{"toObjectId": 1}]})
            if "/associations/" in url:
                return httpx.Response(200, json={"results": []})
            if "/notes/batch/read" in url:
                return httpx.Response(
                    200,
                    json={"results": [{"id": "1", "properties": {"hs_note_body": "y" * 4000}}]},
                )
            return httpx.Response(200, json={"results": []})

        patch_client(tool, handler)
        result = await tool.activities(ctx, object_type="companies", object_id="3001")
        assert len(result.payload["activities"][0]["summary"]) < 4000

    def test_object_id_must_be_numeric(self, tool: HubSpotTool) -> None:
        """object_id is interpolated into the URL path."""
        import re

        pattern = re.compile(
            tool.capability("activities").params_schema["properties"]["object_id"]["pattern"]
        )
        assert pattern.match("3001")
        for bad in ("../deals", "3001/x", "abc", ""):
            assert not pattern.match(bad), bad

    def test_object_type_is_a_closed_enum(self, tool: HubSpotTool) -> None:
        schema = tool.capability("activities").params_schema
        assert schema["properties"]["object_type"]["enum"] == ["companies", "contacts"]


class TestNumberHandling:
    """HubSpot returns numerics as strings and unset values as empty strings."""

    async def test_empty_amount_is_none_not_zero(
        self, tool: HubSpotTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(
            tool,
            {
                "/deals/search": {
                    "total": 1,
                    "results": [{"id": "1", "properties": {"dealname": "x", "amount": ""}}],
                }
            },
        )
        result = await tool.pipeline(ctx)
        assert result.payload["deals"][0]["amount"] is None

    def test_total_is_none_when_no_deal_has_an_amount(self) -> None:
        """An absent amount and a genuine zero are different facts; reporting 0
        would be a fabricated figure."""
        assert _sum_amounts([{"amount": None}, {"amount": None}]) is None

    def test_total_ignores_missing_amounts(self) -> None:
        assert _sum_amounts([{"amount": 100}, {"amount": None}, {"amount": 50}]) == 150

    def test_total_rounds_floats_to_currency_precision(self) -> None:
        assert _sum_amounts([{"amount": 10.25}, {"amount": 1.10}]) == 11.35

    def test_integral_total_stays_an_int(self) -> None:
        """Currency displayed as 48000.0 reads as a bug in a report."""
        total = _sum_amounts([{"amount": 48000.0}])
        assert total == 48000 and isinstance(total, int)

    def test_epoch_ms_is_utc_midnight(self) -> None:
        """Computed rather than hardcoded, so the test states the property instead
        of restating whatever the implementation happens to produce."""
        from datetime import datetime

        expected = int(datetime(2026, 7, 1, 0, 0, 0, tzinfo=UTC).timestamp() * 1000)
        assert _epoch_ms("2026-07-01") == str(expected)

    def test_epoch_ms_is_timezone_independent(self) -> None:
        """A tenant's local timezone must not shift the query window."""
        assert _epoch_ms("2026-07-01").endswith("000")
        assert len(_epoch_ms("2026-07-01")) == 13

    def test_epoch_ms_end_of_day_is_later(self) -> None:
        assert int(_epoch_ms("2026-07-01", end_of_day=True)) > int(_epoch_ms("2026-07-01"))


class TestFailures:
    async def test_bad_token_is_auth_rejected(
        self, tool: HubSpotTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(tool, {"/deals/search": httpx.Response(401, json={"message": "bad"})})
        with pytest.raises(AuthRejected):
            await tool.pipeline(ctx)

    async def test_credential_never_appears_in_the_payload(
        self, tool: HubSpotTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(tool, {"/deals/search": {"total": 1, "results": [DEAL]}})
        result = await tool.pipeline(ctx)
        assert "pat-na1-fake" not in str(result.payload)
