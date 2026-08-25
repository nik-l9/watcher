"""Mixpanel connector.

The connector exists so Cortex is not a PostHog product, so the tests that matter are the ones
proving it behaves like the other analytics sources rather than like Mixpanel: flat rows instead
of nested maps, a disclosed partial bucket, and an event name that cannot reshape the query.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from cortex.tools.base import InvalidParams, ToolContext
from cortex.tools.mixpanel import REGIONS, MixpanelTool


@pytest.fixture
def tool() -> MixpanelTool:
    return MixpanelTool()


def _ctx(tenant: object, **metadata: str) -> ToolContext:
    return ToolContext(
        tenant=tenant,  # type: ignore[arg-type]
        credential="svc.1234.abcdef:s3cr3t",
        credential_metadata={"project_id": "2195012", **metadata},
    )


@pytest.fixture
def ctx(tenant: object) -> ToolContext:
    return _ctx(tenant)


def _segmentation(values: dict[str, dict[str, int]], series: list[str]) -> dict[str, Any]:
    """The shape Mixpanel's segmentation endpoint returns: dates plus a nested map."""
    return {"data": {"series": series, "values": values}, "legend_size": len(values)}


class TestTheCredentialIsAServiceAccountPair:
    async def test_a_username_and_secret_are_split_on_the_first_colon(
        self, tool: MixpanelTool, ctx: ToolContext
    ) -> None:
        assert tool._auth(ctx) == ("svc.1234.abcdef", "s3cr3t")

    async def test_a_secret_containing_a_colon_survives(
        self, tool: MixpanelTool, tenant: object
    ) -> None:
        """Splitting on every colon would corrupt the secret into one that fails
        authentication, and the resulting 401 says nothing about why."""
        ctx = ToolContext(
            tenant=tenant,  # type: ignore[arg-type]
            credential="user:part1:part2",
            credential_metadata={"project_id": "1"},
        )
        assert tool._auth(ctx) == ("user", "part1:part2")

    @pytest.mark.parametrize("credential", ["", "no-colon-here", ":secret", "user:"])
    async def test_a_malformed_credential_says_what_is_needed(
        self, tool: MixpanelTool, tenant: object, credential: str
    ) -> None:
        """Pasting a project token is the likely mistake, and Mixpanel's own error for it
        arrives much later and much less clearly."""
        ctx = ToolContext(
            tenant=tenant,  # type: ignore[arg-type]
            credential=credential,
            credential_metadata={"project_id": "1"},
        )
        with pytest.raises(InvalidParams, match="username:secret"):
            tool._auth(ctx)

    async def test_the_credential_reaches_the_request_as_basic_auth(
        self, tool: MixpanelTool, ctx: ToolContext
    ) -> None:
        """Asserted on a real request rather than on the client object: an auth flow that is
        configured but never applied would pass any test that only inspects configuration."""
        seen: list[httpx.Request] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=[])

        client = tool._client(ctx)
        client._transport = httpx.MockTransport(_handler)
        async with client:
            await client.get("/api/query/events/names")

        import base64

        header = seen[0].headers["authorization"]
        assert header.startswith("Basic ")
        assert base64.b64decode(header.split(" ", 1)[1]).decode() == "svc.1234.abcdef:s3cr3t"


class TestRegionIsConfigurationNotAGuess:
    async def test_each_known_region_maps_to_its_own_host(
        self, tool: MixpanelTool, tenant: object
    ) -> None:
        for region, host in REGIONS.items():
            assert tool._host(_ctx(tenant, region=region)) == host

    async def test_an_unknown_region_is_refused_rather_than_defaulted(
        self, tool: MixpanelTool, tenant: object
    ) -> None:
        """The regions do not share data, so guessing produces an authentication error that
        reads exactly like a bad key — the trap PostHog's host allowlist already closes."""
        with pytest.raises(InvalidParams, match="unknown region"):
            tool._host(_ctx(tenant, region="apac"))

    async def test_the_default_is_us(self, tool: MixpanelTool, ctx: ToolContext) -> None:
        assert tool._host(ctx) == REGIONS["us"]


class TestProjectIdIsRequired:
    async def test_a_credential_without_one_says_how_to_fix_it(
        self, tool: MixpanelTool, tenant: object
    ) -> None:
        ctx = ToolContext(
            tenant=tenant,  # type: ignore[arg-type]
            credential="u:s",
            credential_metadata={},
        )
        with pytest.raises(InvalidParams, match="--meta project_id"):
            tool._project_id(ctx)

    async def test_it_is_sent_on_every_request(
        self, tool: MixpanelTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """The Query API rejects a service-account request without it, and the rejection does
        not name the missing parameter."""
        transport = patch_client(tool, {"/events/names": []})
        await tool.list_events(ctx)
        assert transport.requests[0].url.params["project_id"] == "2195012"


class TestEventTrend:
    async def test_a_nested_response_becomes_flat_ordered_rows(
        self, tool: MixpanelTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Mixpanel answers with a map of segment to date to count. Handing the model that
        shape makes a number's meaning depend on how deep it sits, which is the same defect as
        positional rows: it reads fine until one lookup lands a level off."""
        patch_client(
            tool,
            {
                "/segmentation": _segmentation(
                    {"user signed up": {"2026-08-01": 150, "2026-08-02": 161}},
                    ["2026-08-01", "2026-08-02"],
                )
            },
        )
        result = await tool.event_trend(
            ctx, event="user signed up", from_date="2026-08-01", to_date="2026-08-02"
        )

        assert result.payload["series"] == [
            {"bucket": "2026-08-01", "value": 150, "segment": "user signed up"},
            {"bucket": "2026-08-02", "value": 161, "segment": "user signed up"},
        ]
        # As in every other connector: an empty series and a real zero are different findings.
        assert result.payload["total"] == 311
        assert result.payload["row_count"] == 2

    async def test_an_empty_series_is_reported_as_empty_not_as_zero(
        self, tool: MixpanelTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(tool, {"/segmentation": _segmentation({}, [])})
        result = await tool.event_trend(
            ctx, event="never fired", from_date="2026-08-01", to_date="2026-08-02"
        )
        assert result.payload["row_count"] == 0
        assert result.payload["total"] == 0
        assert result.payload["event"] == "never fired"

    async def test_a_breakdown_is_composed_not_accepted_as_an_expression(
        self, tool: MixpanelTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Mixpanel's `on` parameter is a small expression language. It is built here from a
        validated property name for the same reason HogQL is composed in `posthog.py`: a
        model-authored expression cannot be reviewed before it runs."""
        transport = patch_client(tool, {"/segmentation": _segmentation({}, [])})
        await tool.event_trend(
            ctx,
            event="user signed up",
            from_date="2026-08-01",
            to_date="2026-08-02",
            breakdown_property="plan",
        )
        assert transport.requests[0].url.params["on"] == 'properties["plan"]'

    @pytest.mark.parametrize(
        "hostile",
        ['signed up" or 1==1', 'x"]; properties["secret', "a\nb"],
    )
    async def test_a_hostile_event_name_is_refused(
        self, tool: MixpanelTool, ctx: ToolContext, hostile: str
    ) -> None:
        with pytest.raises(InvalidParams, match="cannot be used in a query"):
            await tool.event_trend(ctx, event=hostile, from_date="2026-08-01", to_date="2026-08-02")

    async def test_a_hostile_breakdown_property_is_refused(
        self, tool: MixpanelTool, ctx: ToolContext
    ) -> None:
        with pytest.raises(InvalidParams, match="cannot be used in a query"):
            await tool.event_trend(
                ctx,
                event="user signed up",
                from_date="2026-08-01",
                to_date="2026-08-02",
                breakdown_property='plan"] or properties["email',
            )

    async def test_an_inverted_range_is_refused(self, tool: MixpanelTool, ctx: ToolContext) -> None:
        with pytest.raises(InvalidParams):
            await tool.event_trend(
                ctx, event="user signed up", from_date="2026-08-12", to_date="2026-08-01"
            )

    @pytest.mark.parametrize("field,value", [("unit", "fortnight"), ("measure", "median")])
    async def test_an_unallowlisted_value_is_refused(
        self, tool: MixpanelTool, ctx: ToolContext, field: str, value: str
    ) -> None:
        """Both are interpolated into the query string, so neither can be free text."""
        with pytest.raises(InvalidParams, match=field):
            await tool.event_trend(
                ctx,
                event="user signed up",
                from_date="2026-08-01",
                to_date="2026-08-02",
                **{field: value},
            )


class TestAPartialMonthIsDisclosed:
    """The failure that produced a wrong answer on real data, in a second vendor.

    A total for twelve days of August sitting beside a total for all of July invites exactly one
    comparison, and it is meaningless. Mixpanel returns no warning, so the connector computes it
    from the range the caller asked for.
    """

    async def test_a_trailing_partial_month_is_named_with_its_day_count(
        self, tool: MixpanelTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(
            tool,
            {
                "/segmentation": _segmentation(
                    {"user signed up": {"2026-07-01": 4849, "2026-08-01": 1884}},
                    ["2026-07-01", "2026-08-01"],
                )
            },
        )
        result = await tool.event_trend(
            ctx,
            event="user signed up",
            from_date="2026-07-01",
            to_date="2026-08-12",
            unit="month",
        )
        partial = result.payload["partial_buckets"]
        assert {
            "bucket": "2026-08-01",
            "reason": "ends mid-month",
            "days_covered": 12,
            "days_in_full_bucket": 31,
        } in partial
        assert "not comparable" in result.payload["note"]

    async def test_whole_months_carry_no_note(
        self, tool: MixpanelTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """An ordinary series must look ordinary, or the field becomes noise a reader learns to
        skip."""
        patch_client(tool, {"/segmentation": _segmentation({}, [])})
        result = await tool.event_trend(
            ctx,
            event="user signed up",
            from_date="2026-06-01",
            to_date="2026-07-31",
            unit="month",
        )
        assert "partial_buckets" not in result.payload
        assert "note" not in result.payload

    async def test_a_daily_series_is_never_flagged(
        self, tool: MixpanelTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(tool, {"/segmentation": _segmentation({}, [])})
        result = await tool.event_trend(
            ctx, event="user signed up", from_date="2026-08-01", to_date="2026-08-12"
        )
        assert "partial_buckets" not in result.payload


class TestListEvents:
    async def test_a_bare_array_response_is_read_correctly(
        self, tool: MixpanelTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """This endpoint answers with a top-level JSON array rather than an object, which
        `request_json` wraps under `data` so a capability always has a dict to work from."""
        patch_client(tool, {"/events/names": ["user signed up", "trial started"]})
        result = await tool.list_events(ctx)

        assert result.payload["events"] == ["user signed up", "trial started"]
        assert result.payload["event_count"] == 2

    async def test_it_is_a_discovery_capability(self, tool: MixpanelTool) -> None:
        """The investigator runs discovery capabilities before the first step. Without it the
        analyst guesses event names, and a guessed name returns an empty series that looks like
        a real drop to zero."""
        listing = next(c for c in tool.capabilities() if c.name == "list_events")
        assert listing.discovery is True


class TestEveryCapabilityIsReadOnly:
    def test_no_capability_can_write(self, tool: MixpanelTool) -> None:
        """V1 ships no write capability anywhere, so "humans approve" is enforced by the
        absence of destructive tools rather than by a prompt."""
        assert all(c.read_only for c in tool.capabilities())

    def test_every_capability_declares_where_its_results_live(self, tool: MixpanelTool) -> None:
        """`result_key` is what lets the executor mark an empty observation as empty — the bug
        this codebase has shipped four times."""
        assert all(c.result_key for c in tool.capabilities())


class TestASeriesThatStopsEarly:
    """Mixpanel computed `partial_buckets` and was never graded at all.

    That combination is worse than disclosing nothing. `partial_buckets` says "the last month is
    young", the gap says "collection stopped", and the two invite opposite conclusions from the
    same shape of missing data -- calm versus escalate. Mixpanel disclosed only the reassuring
    one, and the trust gate never ran over its payloads, so nothing downstream could enforce it.
    """

    async def test_a_stopped_series_is_disclosed_and_graded(
        self, tool: MixpanelTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(
            tool,
            {
                "/segmentation": _segmentation(
                    {"user signed up": {"2026-08-01": 150, "2026-08-02": 161}},
                    ["2026-08-01", "2026-08-02"],
                )
            },
        )
        result = await tool.event_trend(
            ctx, event="user signed up", from_date="2026-08-01", to_date="2026-08-15"
        )

        assert result.payload["series_ends_early"] == {
            "last_bucket": "2026-08-02",
            "requested_end": "2026-08-15",
            "days_missing": 13,
        }
        assert result.payload["data_trust"]["state"] == "degraded"
        assert [t["check"] for t in result.payload["data_trust"]["tripped"]] == [
            "gate4_single_cessation"
        ]

    async def test_the_alternatives_include_being_hidden_from_the_schema(
        self, tool: MixpanelTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Mixpanel-specific: an event can be hidden from the project schema while still being
        ingested, which looks identical to a stop and is not a case PostHog has."""
        patch_client(
            tool,
            {"/segmentation": _segmentation({"e": {"2026-08-01": 5}}, ["2026-08-01"])},
        )
        result = await tool.event_trend(
            ctx, event="e", from_date="2026-08-01", to_date="2026-08-15"
        )
        assert "hidden from the project schema" in result.payload["series_gap_note"]

    async def test_a_complete_series_carries_neither(
        self, tool: MixpanelTool, ctx: ToolContext, patch_client: object
    ) -> None:
        days = [f"2026-08-{d:02d}" for d in range(1, 16)]
        patch_client(
            tool,
            {"/segmentation": _segmentation({"e": dict.fromkeys(days, 10)}, days)},
        )
        result = await tool.event_trend(
            ctx, event="e", from_date="2026-08-01", to_date="2026-08-15"
        )
        assert "series_ends_early" not in result.payload
        assert "data_trust" not in result.payload

    async def test_a_partial_month_now_reaches_the_gate(
        self, tool: MixpanelTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """The disclosure it already computed, now with a verdict attached to it.

        `partial_buckets` was in the payload and graded by nothing, so a downstream gate that
        withholds causal claims over a broken series had no verdict to act on.
        """
        patch_client(
            tool,
            {
                "/segmentation": _segmentation(
                    {"e": {"2026-08-01": 900}},
                    ["2026-08-01"],
                )
            },
        )
        result = await tool.event_trend(
            ctx, event="e", from_date="2026-07-01", to_date="2026-08-12", unit="month"
        )
        assert result.payload["partial_buckets"]
        tripped = {t["check"] for t in result.payload["data_trust"]["tripped"]}
        assert "gate2_trailing_bucket" in tripped
