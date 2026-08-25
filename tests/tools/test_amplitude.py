"""Amplitude connector.

Amplitude's Dashboard REST API is the most translated of the three analytics sources — dates as
`YYYYMMDD`, intervals as day counts, the event definition as JSON inside a query parameter — so
the tests concentrate on the translation. A connector that leaks its vendor's spelling into the
tool schema makes the analyst's job harder for no benefit, and a translation that is wrong in one
direction produces an empty series that reads as a real absence.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from cortex.tools.amplitude import INTERVALS, REGIONS, AmplitudeTool
from cortex.tools.base import InvalidParams, ToolContext


@pytest.fixture
def tool() -> AmplitudeTool:
    return AmplitudeTool()


def _ctx(tenant: object, **metadata: str) -> ToolContext:
    return ToolContext(
        tenant=tenant,  # type: ignore[arg-type]
        credential="apikey123:secret456",
        credential_metadata=dict(metadata),
    )


@pytest.fixture
def ctx(tenant: object) -> ToolContext:
    return _ctx(tenant)


def _segmentation(
    x_values: list[str], series: list[list[int]], labels: list[str] | None = None
) -> dict[str, Any]:
    """Amplitude's parallel-array response shape."""
    data: dict[str, Any] = {"xValues": x_values, "series": series}
    if labels is not None:
        data["seriesLabels"] = labels
    return {"data": data}


class TestTheCredentialIsAKeyPair:
    async def test_the_key_and_secret_split_on_the_first_colon(
        self, tool: AmplitudeTool, ctx: ToolContext
    ) -> None:
        assert tool._auth(ctx) == ("apikey123", "secret456")

    async def test_a_secret_containing_a_colon_survives(
        self, tool: AmplitudeTool, tenant: object
    ) -> None:
        ctx = ToolContext(
            tenant=tenant,  # type: ignore[arg-type]
            credential="key:a:b",
            credential_metadata={},
        )
        assert tool._auth(ctx) == ("key", "a:b")

    @pytest.mark.parametrize("credential", ["", "onlykey", ":secret", "key:"])
    async def test_a_malformed_credential_says_both_halves_are_needed(
        self, tool: AmplitudeTool, tenant: object, credential: str
    ) -> None:
        ctx = ToolContext(
            tenant=tenant,  # type: ignore[arg-type]
            credential=credential,
            credential_metadata={},
        )
        with pytest.raises(InvalidParams, match="api_key:secret_key"):
            tool._auth(ctx)

    async def test_the_credential_reaches_the_request_as_basic_auth(
        self, tool: AmplitudeTool, ctx: ToolContext
    ) -> None:
        """On a real request rather than the client object: auth that is configured but never
        applied would pass a test that only inspects configuration."""
        import base64

        seen: list[httpx.Request] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={})

        client = tool._client(ctx)
        client._transport = httpx.MockTransport(_handler)
        async with client:
            await client.get("/api/2/events/list")

        header = seen[0].headers["authorization"]
        assert base64.b64decode(header.split(" ", 1)[1]).decode() == "apikey123:secret456"


class TestRegion:
    async def test_each_known_region_maps_to_its_own_host(
        self, tool: AmplitudeTool, tenant: object
    ) -> None:
        for region, host in REGIONS.items():
            assert tool._host(_ctx(tenant, region=region)) == host

    async def test_an_unknown_region_is_refused(self, tool: AmplitudeTool, tenant: object) -> None:
        with pytest.raises(InvalidParams, match="unknown region"):
            tool._host(_ctx(tenant, region="apac"))


class TestTheVendorsSpellingStaysInTheConnector:
    """The analyst writes one kind of date and one kind of bucket for every source."""

    async def test_iso_dates_become_the_compact_form_amplitude_requires(
        self, tool: AmplitudeTool, ctx: ToolContext, patch_client: object
    ) -> None:
        transport = patch_client(tool, {"/segmentation": _segmentation([], [])})
        await tool.event_trend(
            ctx, event="user signed up", start_date="2026-08-01", end_date="2026-08-12"
        )
        params = transport.requests[0].url.params
        assert params["start"] == "20260801"
        assert params["end"] == "20260812"

    async def test_a_date_that_looks_valid_but_is_not_is_refused(
        self, tool: AmplitudeTool, ctx: ToolContext
    ) -> None:
        """Parsed rather than stripped of dashes. `2026-13-45` matches the schema pattern and
        is not a date; sending it would return an empty series that reads as a real absence."""
        with pytest.raises(InvalidParams, match="not a valid"):
            await tool.event_trend(
                ctx, event="user signed up", start_date="2026-13-45", end_date="2026-13-46"
            )

    @pytest.mark.parametrize("named,days", sorted(INTERVALS.items()))
    async def test_named_intervals_become_day_counts(
        self, tool: AmplitudeTool, ctx: ToolContext, patch_client: object, named: str, days: str
    ) -> None:
        transport = patch_client(tool, {"/segmentation": _segmentation([], [])})
        await tool.event_trend(
            ctx,
            event="user signed up",
            start_date="2026-08-01",
            end_date="2026-08-12",
            interval=named,
        )
        assert transport.requests[0].url.params["i"] == days

    async def test_the_event_definition_is_serialised_not_concatenated(
        self, tool: AmplitudeTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """The name travels inside JSON inside a query parameter. Building that string by hand
        is where an embedded quote would change the request's meaning, so `json.dumps` owns the
        quoting."""
        transport = patch_client(tool, {"/segmentation": _segmentation([], [])})
        await tool.event_trend(
            ctx,
            event="user signed up",
            start_date="2026-08-01",
            end_date="2026-08-12",
            group_by="platform",
        )
        definition = json.loads(transport.requests[0].url.params["e"])
        assert definition == {
            "event_type": "user signed up",
            "group_by": [{"type": "event", "value": "platform"}],
        }


class TestEventTrend:
    async def test_parallel_arrays_become_flat_named_rows(
        self, tool: AmplitudeTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """The value's meaning depends on the position of its array *and* the position within
        it — the positional-data problem the report schema rejects for chart points."""
        patch_client(
            tool,
            {
                "/segmentation": _segmentation(
                    ["2026-08-01", "2026-08-02"], [[150, 161]], ["user signed up"]
                )
            },
        )
        result = await tool.event_trend(
            ctx, event="user signed up", start_date="2026-08-01", end_date="2026-08-02"
        )
        assert result.payload["series"] == [
            {"bucket": "2026-08-01", "value": 150},
            {"bucket": "2026-08-02", "value": 161},
        ]
        assert result.payload["total"] == 311

    async def test_a_single_series_label_is_not_reported_as_a_segment(
        self, tool: AmplitudeTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """On an unsegmented query the label names the event, not a dimension. Echoing it as a
        segment would invent a breakdown nobody asked for, and a report can be built on that."""
        patch_client(
            tool, {"/segmentation": _segmentation(["2026-08-01"], [[150]], ["user signed up"])}
        )
        result = await tool.event_trend(
            ctx, event="user signed up", start_date="2026-08-01", end_date="2026-08-01"
        )
        assert "segment" not in result.payload["series"][0]

    async def test_a_real_breakdown_keeps_its_segment_names(
        self, tool: AmplitudeTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(
            tool,
            {"/segmentation": _segmentation(["2026-08-01"], [[90], [60]], ["ios", "android"])},
        )
        result = await tool.event_trend(
            ctx,
            event="user signed up",
            start_date="2026-08-01",
            end_date="2026-08-01",
            group_by="platform",
        )
        assert result.payload["series"] == [
            {"bucket": "2026-08-01", "value": 90, "segment": "ios"},
            {"bucket": "2026-08-01", "value": 60, "segment": "android"},
        ]

    async def test_missing_labels_are_tolerated_rather_than_raising(
        self, tool: AmplitudeTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Amplitude omits labels for some queries. An analyst that gets rows without a segment
        name has still learned what it asked; one that gets an exception has learned nothing."""
        patch_client(tool, {"/segmentation": _segmentation(["2026-08-01"], [[150]])})
        result = await tool.event_trend(
            ctx, event="user signed up", start_date="2026-08-01", end_date="2026-08-01"
        )
        assert result.payload["series"] == [{"bucket": "2026-08-01", "value": 150}]

    async def test_more_values_than_dates_does_not_invent_buckets(
        self, tool: AmplitudeTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """A row needs a date to mean anything. Pairing a value with a missing bucket would
        produce a number attached to nothing, which is worse than dropping it."""
        patch_client(tool, {"/segmentation": _segmentation(["2026-08-01"], [[150, 161, 172]])})
        result = await tool.event_trend(
            ctx, event="user signed up", start_date="2026-08-01", end_date="2026-08-01"
        )
        assert result.payload["row_count"] == 1

    async def test_an_empty_series_is_reported_as_empty_not_as_zero(
        self, tool: AmplitudeTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(tool, {"/segmentation": _segmentation([], [])})
        result = await tool.event_trend(
            ctx, event="never fired", start_date="2026-08-01", end_date="2026-08-02"
        )
        assert result.payload["row_count"] == 0
        assert result.payload["total"] == 0

    @pytest.mark.parametrize("hostile", ['x"}]}&e={"event_type":"other', "a\nb", "x" * 300])
    async def test_a_hostile_event_name_is_refused(
        self, tool: AmplitudeTool, ctx: ToolContext, hostile: str
    ) -> None:
        with pytest.raises(InvalidParams, match="cannot be used in a query"):
            await tool.event_trend(
                ctx, event=hostile, start_date="2026-08-01", end_date="2026-08-02"
            )

    async def test_an_inverted_range_is_refused(
        self, tool: AmplitudeTool, ctx: ToolContext
    ) -> None:
        with pytest.raises(InvalidParams):
            await tool.event_trend(
                ctx, event="user signed up", start_date="2026-08-12", end_date="2026-08-01"
            )

    @pytest.mark.parametrize("field,value", [("interval", "fortnight"), ("measure", "median")])
    async def test_an_unallowlisted_value_is_refused(
        self, tool: AmplitudeTool, ctx: ToolContext, field: str, value: str
    ) -> None:
        with pytest.raises(InvalidParams, match=field):
            await tool.event_trend(
                ctx,
                event="user signed up",
                start_date="2026-08-01",
                end_date="2026-08-02",
                **{field: value},
            )


class TestListEvents:
    async def test_an_event_that_stopped_firing_is_still_reported(
        self, tool: AmplitudeTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Amplitude marks these `non_active`. Hiding them would turn "this stopped" into "this
        never existed", which is the class of confusion `result_key` exists to prevent."""
        patch_client(
            tool,
            {
                "/events/list": {
                    "data": [
                        {"value": "user signed up", "display": "Signed Up", "non_active": False},
                        {"value": "legacy checkout", "display": "Legacy", "non_active": True},
                    ]
                }
            },
        )
        result = await tool.list_events(ctx)

        assert result.payload["event_count"] == 2
        stale = next(e for e in result.payload["events"] if e["name"] == "legacy checkout")
        assert stale["non_active"] is True

    async def test_it_is_a_discovery_capability(self, tool: AmplitudeTool) -> None:
        listing = next(c for c in tool.capabilities() if c.name == "list_events")
        assert listing.discovery is True


class TestEveryCapabilityIsReadOnly:
    def test_no_capability_can_write(self, tool: AmplitudeTool) -> None:
        assert all(c.read_only for c in tool.capabilities())

    def test_every_capability_declares_where_its_results_live(self, tool: AmplitudeTool) -> None:
        assert all(c.result_key for c in tool.capabilities())
