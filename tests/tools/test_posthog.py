"""PostHog connector.

Two things here carry more weight than the parsing, and both are why this file exists.

**The project allowlist is an access-control boundary.** An organisation's analytics are
split across projects that do not share events, so a question about one product answered
from another's data is silently wrong — and a key that can technically reach a fourth
project must not be able to. That the allowlist is honoured is checked here, including the
case where the *credential itself* could read more than the tenant declared.

**HogQL is composed here, never authored by the model.** The query endpoint takes arbitrary
SQL over production event data. Every value interpolated into it — event name, breakdown
property, interval, aggregation — is either allowlisted or pattern-checked, and the tests
that matter most are the ones proving a hostile name is refused rather than quoted.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Any

import httpx
import pytest

from cortex.tools.base import InvalidParams, ToolContext
from cortex.tools.posthog import AGGREGATIONS, INTERVALS, KNOWN_HOSTS, PostHogTool

THREE_PROJECTS = "100001:web-app,100002:oss-client,100003:mobile-app"


def _query_response(columns: list[str], results: list[list[Any]]) -> dict[str, Any]:
    """The shape PostHog's query endpoint returns: positional rows plus column names."""
    return {"columns": columns, "results": results, "hogql": "SELECT ..."}


@pytest.fixture
def tool() -> PostHogTool:
    return PostHogTool()


def _ctx(tenant: object, **metadata: str) -> ToolContext:
    return ToolContext(
        tenant=tenant,  # type: ignore[arg-type]
        credential="phx_fake_key",
        credential_metadata={"projects": THREE_PROJECTS, **metadata},
    )


@pytest.fixture
def ctx(tenant: object) -> ToolContext:
    return _ctx(tenant)


class TestTheProjectAllowlist:
    """Which projects a tenant may read is declared, not discovered."""

    async def test_lists_the_declared_projects_with_a_default(
        self, tool: PostHogTool, ctx: ToolContext
    ) -> None:
        result = await tool.list_projects(ctx)

        assert result.payload["project_count"] == 3
        assert result.payload["default_project"] == "100001"
        labels = {p["id"]: p["label"] for p in result.payload["projects"]}
        assert labels["100002"] == "oss-client"
        assert [p["is_default"] for p in result.payload["projects"]] == [True, False, False]

    async def test_a_project_outside_the_allowlist_is_refused(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """The key can technically reach the staging project. The tenant did not declare
        it, so the connector must not read it — widening access is a deliberate act at
        connect time, not something a model can wander into."""
        patch_client(
            tool, {"/query/": _query_response(["bucket", "value"], [])}, is_async_factory=True
        )

        with pytest.raises(InvalidParams, match="not available to this tenant"):
            await tool.event_trend(
                ctx,
                event="user signed up",
                start_date="2026-07-01",
                end_date="2026-07-07",
                project="163845",
            )

    async def test_the_refusal_names_what_is_available(
        self, tool: PostHogTool, ctx: ToolContext
    ) -> None:
        """An empty result and a wrong project are indistinguishable to the analyst, so
        the error has to say what it *could* have read."""
        with pytest.raises(InvalidParams) as raised:
            tool._project_id(ctx, "163845")
        message = str(raised.value)
        assert "web-app" in message and "oss-client" in message

    async def test_a_declared_project_is_used_in_the_request_path(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        transport = patch_client(
            tool, {"/query/": _query_response(["bucket", "value"], [])}, is_async_factory=True
        )
        await tool.event_trend(
            ctx,
            event="agent_task_completed",
            start_date="2026-07-01",
            end_date="2026-07-07",
            project="100002",
        )
        assert "/api/projects/100002/query/" in str(transport.requests[0].url)

    async def test_the_legacy_single_project_form_still_works(
        self, tool: PostHogTool, tenant: object
    ) -> None:
        """Credentials stored before multi-project support carry `project_id`. Breaking
        them would take a connected tenant offline on deploy."""
        ctx = ToolContext(
            tenant=tenant,  # type: ignore[arg-type]
            credential="phx_fake_key",
            credential_metadata={"project_id": "100001"},
        )
        assert tool._project_id(ctx) == "100001"

    async def test_a_credential_with_no_projects_says_how_to_fix_it(
        self, tool: PostHogTool, tenant: object
    ) -> None:
        ctx = ToolContext(
            tenant=tenant,  # type: ignore[arg-type]
            credential="phx_fake_key",
            credential_metadata={},
        )
        with pytest.raises(InvalidParams, match="--meta projects="):
            tool._project_id(ctx)


class TestHogqlIsComposedNotAuthored:
    """Every value reaching the SQL is allowlisted or pattern-checked."""

    @pytest.mark.parametrize(
        "hostile",
        [
            "signed up' OR 1=1 --",
            "x'); DROP TABLE events; --",
            "a\nb",
        ],
    )
    async def test_a_hostile_event_name_is_refused(
        self, tool: PostHogTool, ctx: ToolContext, hostile: str
    ) -> None:
        with pytest.raises(InvalidParams, match="cannot be used in a query"):
            await tool.event_trend(
                ctx, event=hostile, start_date="2026-07-01", end_date="2026-07-07"
            )

    async def test_a_hostile_breakdown_property_is_refused(
        self, tool: PostHogTool, ctx: ToolContext
    ) -> None:
        with pytest.raises(InvalidParams, match="cannot be used in a query"):
            await tool.event_trend(
                ctx,
                event="user signed up",
                start_date="2026-07-01",
                end_date="2026-07-07",
                breakdown_property="device' OR '1'='1",
            )

    async def test_an_unknown_interval_is_refused(
        self, tool: PostHogTool, ctx: ToolContext
    ) -> None:
        """Interpolated into the SQL as `toStartOfDay`, so it cannot be free text."""
        with pytest.raises(InvalidParams, match="interval"):
            await tool.event_trend(
                ctx,
                event="user signed up",
                start_date="2026-07-01",
                end_date="2026-07-07",
                interval="'; DROP TABLE events; --",
            )

    async def test_an_unknown_measure_is_refused(self, tool: PostHogTool, ctx: ToolContext) -> None:
        with pytest.raises(InvalidParams, match="measure"):
            await tool.event_trend(
                ctx,
                event="user signed up",
                start_date="2026-07-01",
                end_date="2026-07-07",
                measure="count(*) FROM secrets --",
            )

    async def test_the_allowlists_are_the_only_source_of_sql_fragments(self) -> None:
        """A regression guard on the design, not on one call: if either allowlist ever
        holds something that is not a fixed fragment, the composition is no longer safe."""
        assert set(INTERVALS) == {"day", "week", "month"}
        assert all(value.endswith(")") for value in AGGREGATIONS.values())

    async def test_the_composed_query_carries_the_typed_parameters(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        transport = patch_client(
            tool,
            {"/query/": _query_response(["bucket", "value"], [["2026-07-01", 12]])},
            is_async_factory=True,
        )
        await tool.event_trend(
            ctx,
            event="user signed up",
            start_date="2026-07-01",
            end_date="2026-07-07",
            interval="week",
            measure="unique_users",
        )
        sent = transport.request_bodies()[0]["query"]["query"]
        assert "toStartOfWeek" in sent
        assert "count(distinct person_id)" in sent
        assert "event = 'user signed up'" in sent
        # Both boundaries are explicit, so the last day is not silently excluded.
        assert "'2026-07-01 00:00:00'" in sent and "'2026-07-07 23:59:59'" in sent


class TestEventTrend:
    async def test_positional_rows_become_named_fields(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """PostHog returns rows as arrays with the column names alongside. Handing the
        model tuples whose meaning depends on order is the same mistake that made the
        report schema reject positional chart points."""
        patch_client(
            tool,
            {
                "/query/": _query_response(
                    ["bucket", "segment", "value"],
                    [["2026-07-15", "Mobile", 39], ["2026-07-15", "Desktop", 26]],
                )
            },
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx,
            event="onboarding completed",
            start_date="2026-07-15",
            end_date="2026-07-16",
            breakdown_property="$device_type",
        )

        assert result.payload["row_count"] == 2
        assert result.payload["series"][0] == {
            "bucket": "2026-07-15",
            "segment": "Mobile",
            "value": 39,
        }
        # The total is returned so an empty series can be told apart from a real drop to
        # zero: no rows for an event that exists is a different finding from no rows
        # because the name was wrong.
        assert result.payload["total"] == 65

    async def test_an_empty_series_is_reported_as_empty_not_as_zero(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(
            tool, {"/query/": _query_response(["bucket", "value"], [])}, is_async_factory=True
        )
        result = await tool.event_trend(
            ctx, event="never fired", start_date="2026-07-01", end_date="2026-07-07"
        )
        assert result.payload["row_count"] == 0
        assert result.payload["total"] == 0
        assert result.payload["event"] == "never fired"

    async def test_an_inverted_date_range_is_refused(
        self, tool: PostHogTool, ctx: ToolContext
    ) -> None:
        with pytest.raises(InvalidParams):
            await tool.event_trend(
                ctx, event="user signed up", start_date="2026-07-07", end_date="2026-07-01"
            )


class TestAShortBucketIsDisclosed:
    """The production defect, and the one no grounding mechanism could have caught.

    Asked whether signups had fallen from last month, the analyst received two monthly buckets --
    4,849 for July, 1,884 for the first twelve days of August -- and nothing said the second was a
    third of a month long. Comparing totals is the obvious thing to do with two totals, and it
    was meaningless. Every claim in the answer was cited and every citation resolved.

    So the truncation is returned as data. Same principle as `total`: a fact about the shape of
    the result belongs in the result, not in the reader's head.
    """

    async def test_a_partial_trailing_month_is_named_with_its_day_count(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(
            tool,
            {
                "/query/": _query_response(
                    ["bucket", "value"],
                    [
                        ["2026-07-01T00:00:00", 4849],
                        ["2026-08-01T00:00:00", 1884],
                    ],
                )
            },
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx,
            event="user signed up",
            start_date="2026-07-01",
            end_date="2026-08-12",
            interval="month",
        )

        partial = result.payload["partial_buckets"]
        assert partial == [{"bucket": "2026-08-01", "days_covered": 12, "days_in_full_bucket": 31}]
        # The day counts are what make the comparison recoverable rather than merely suspect:
        # 1,884/12 against 4,849/31 is the calculation the answer needed.
        assert "not comparable" in result.payload["note"]

    async def test_whole_buckets_carry_no_extra_field(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """An ordinary series must look ordinary. A `partial_buckets: []` on every response
        trains a reader to ignore the field, which is the same as not having it."""
        patch_client(
            tool,
            {
                "/query/": _query_response(
                    ["bucket", "value"],
                    [["2026-06-01T00:00:00", 4692], ["2026-07-01T00:00:00", 4849]],
                )
            },
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx,
            event="user signed up",
            start_date="2026-06-01",
            end_date="2026-07-31",
            interval="month",
        )
        assert "partial_buckets" not in result.payload
        assert "note" not in result.payload

    async def test_a_leading_partial_bucket_is_disclosed_too(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Both ends, not just the trailing one. A range starting mid-month understates its
        first bucket exactly as badly, and reading it as a rise is the same error mirrored."""
        patch_client(
            tool,
            {
                "/query/": _query_response(
                    ["bucket", "value"],
                    [["2026-06-01T00:00:00", 900], ["2026-07-01T00:00:00", 4849]],
                )
            },
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx,
            event="user signed up",
            start_date="2026-06-25",
            end_date="2026-07-31",
            interval="month",
        )
        assert result.payload["partial_buckets"] == [
            {"bucket": "2026-06-01", "days_covered": 6, "days_in_full_bucket": 30}
        ]

    async def test_a_week_is_seven_days_whatever_day_it_starts_on(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """`toStartOfWeek` follows ClickHouse's default mode, which begins the week on Sunday.
        Rather than encode that assumption, the bucket starts are read from the rows and only
        their *length* comes from the calendar — so a change of mode cannot make this wrong."""
        patch_client(
            tool,
            {
                "/query/": _query_response(
                    ["bucket", "value"],
                    [["2026-06-14T00:00:00", 400], ["2026-06-21T00:00:00", 1100]],
                )
            },
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx,
            event="user signed up",
            start_date="2026-06-15",
            end_date="2026-06-27",
            interval="week",
        )
        assert result.payload["partial_buckets"] == [
            {"bucket": "2026-06-14", "days_covered": 6, "days_in_full_bucket": 7}
        ]

    async def test_a_daily_series_is_never_flagged(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """A day bucket is a day. Flagging one would be noise on the most common call there is."""
        patch_client(
            tool,
            {"/query/": _query_response(["bucket", "value"], [["2026-08-12T00:00:00", 51]])},
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx, event="user signed up", start_date="2026-08-12", end_date="2026-08-12"
        )
        assert "partial_buckets" not in result.payload


class TestFunnel:
    async def test_steps_compose_into_conversion_rates(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(
            tool,
            {"/query/": _query_response(["step_0", "step_1", "step_2"], [[1000, 400, 100]])},
            is_async_factory=True,
        )
        result = await tool.funnel(
            ctx,
            steps=["$pageview", "user signed up", "onboarding completed"],
            start_date="2026-07-01",
            end_date="2026-07-07",
        )

        steps = result.payload["steps"]
        assert [s["users"] for s in steps] == [1000, 400, 100]
        assert steps[1]["conversion_from_previous"] == 0.4
        assert steps[2]["conversion_from_first"] == 0.1

    async def test_it_discloses_that_ordering_is_not_enforced(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """This counts users who did each step inside the window, not users who did them
        in order. Presenting an unordered count as an ordered funnel would overstate what
        the number means — so the payload says which it is."""
        patch_client(
            tool,
            {"/query/": _query_response(["step_0", "step_1"], [[10, 5]])},
            is_async_factory=True,
        )
        result = await tool.funnel(
            ctx, steps=["a", "b"], start_date="2026-07-01", end_date="2026-07-07"
        )
        assert result.payload["ordering_enforced"] is False

    async def test_no_steps_is_refused(self, tool: PostHogTool, ctx: ToolContext) -> None:
        with pytest.raises(InvalidParams, match="at least one step"):
            await tool.funnel(ctx, steps=[], start_date="2026-07-01", end_date="2026-07-07")

    async def test_an_empty_result_does_not_divide_by_zero(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(
            tool, {"/query/": _query_response(["step_0", "step_1"], [])}, is_async_factory=True
        )
        result = await tool.funnel(
            ctx, steps=["a", "b"], start_date="2026-07-01", end_date="2026-07-07"
        )
        assert result.payload["steps"][0]["conversion_from_first"] is None


class TestTheChangeRecord:
    """The three capabilities that answer "what changed" — PostHog's real advantage."""

    async def test_annotations_separate_a_human_note_from_a_deploy_marker(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """One is testimony, the other is a record of a ship. Conflating them would let a
        person's guess be cited as a deployment fact."""
        patch_client(
            tool,
            {
                "/annotations/": {
                    "count": 2,
                    "results": [
                        {
                            "content": "shipped onboarding rework",
                            "date_marker": "2026-07-14T11:04:00Z",
                            "creation_type": "GIT",
                            "created_by": {"email": "ci@acme.com"},
                        },
                        {
                            "content": "mobile signups look off",
                            "date_marker": "2026-07-15T09:00:00Z",
                            "creation_type": "USR",
                            "created_by": {"email": "dana@acme.com"},
                        },
                    ],
                }
            },
            is_async_factory=True,
        )
        result = await tool.annotations(ctx)

        sources = [a["source"] for a in result.payload["annotations"]]
        assert sources == ["deployment", "person"]

    async def test_no_annotations_is_reported_with_the_total(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Verified against all five real projects: they are empty. An analyst must be
        able to tell "nobody annotates" from "this call failed"."""
        patch_client(tool, {"/annotations/": {"count": 0, "results": []}}, is_async_factory=True)
        result = await tool.annotations(ctx)
        assert result.payload["annotation_count"] == 0
        assert result.payload["total_available"] == 0

    async def test_feature_flags_report_rollout_and_activity(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """A rollout is a change event, and "conversion fell for users in this flag" is a
        root cause invisible to every other connector."""
        patch_client(
            tool,
            {
                "/feature_flags/": {
                    "count": 1,
                    "results": [
                        {
                            "key": "new-onboarding",
                            "name": "New onboarding sheet",
                            "active": True,
                            "filters": {"groups": [{"rollout_percentage": 50}]},
                            "created_by": {"email": "dana@acme.com"},
                        }
                    ],
                }
            },
            is_async_factory=True,
        )
        result = await tool.feature_flags(ctx, active_only=True)

        flag = result.payload["flags"][0]
        assert flag["key"] == "new-onboarding"
        assert flag["active"] is True
        assert flag["rollout_percentage"] == 50

    async def test_active_only_is_forwarded(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        transport = patch_client(
            tool, {"/feature_flags/": {"count": 0, "results": []}}, is_async_factory=True
        )
        await tool.feature_flags(ctx, active_only=True, search="onboarding")

        params = transport.requests[0].url.params
        assert params["active"] == "true"
        assert params["search"] == "onboarding"

    async def test_experiments_are_listed(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """A running A/B test explains a metric move and is otherwise indistinguishable
        from an unexplained regression."""
        patch_client(
            tool,
            {
                "/experiments/": {
                    "count": 1,
                    "results": [
                        {
                            "name": "Onboarding sheet vs modal",
                            "feature_flag_key": "new-onboarding",
                            "start_date": "2026-07-14T00:00:00Z",
                            "end_date": None,
                        }
                    ],
                }
            },
            is_async_factory=True,
        )
        result = await tool.experiments(ctx)
        assert result.payload["experiments"][0]["feature_flag_key"] == "new-onboarding"


class TestRegionAndCredentialHandling:
    def test_the_default_host_is_stated_rather_than_guessed(
        self, tool: PostHogTool, ctx: ToolContext
    ) -> None:
        """A key issued in the EU returns 401 against the US host, which looks exactly
        like a bad key."""
        assert tool._host(ctx) == KNOWN_HOSTS[0]

    def test_a_declared_region_is_honoured(self, tool: PostHogTool, tenant: object) -> None:
        ctx = _ctx(tenant, host="https://eu.posthog.com")
        assert tool._host(ctx) == "https://eu.posthog.com"

    def test_a_plaintext_host_is_refused(self, tool: PostHogTool, tenant: object) -> None:
        ctx = _ctx(tenant, host="http://evil.example.com")
        with pytest.raises(InvalidParams, match="not https"):
            tool._host(ctx)

    async def test_the_credential_never_reaches_the_payload(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(
            tool,
            {"/query/": _query_response(["bucket", "value"], [["2026-07-01", 1]])},
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx, event="user signed up", start_date="2026-07-01", end_date="2026-07-07"
        )
        assert "phx_fake_key" not in str(result.payload)

    async def test_the_credential_is_sent_as_a_bearer_token(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        transport = patch_client(
            tool, {"/query/": _query_response(["bucket", "value"], [])}, is_async_factory=True
        )
        await tool.event_trend(
            ctx, event="user signed up", start_date="2026-07-01", end_date="2026-07-07"
        )
        assert transport.requests[0].headers["authorization"] == "Bearer phx_fake_key"


class TestEveryCapabilityIsReadOnly:
    def test_no_capability_declares_a_write(self, tool: PostHogTool) -> None:
        """V1 ships no write capability anywhere, so "humans approve" is enforced by the
        absence of destructive tools rather than by a prompt."""
        for name in tool.capability_names:
            assert tool.capability(name).read_only, name

    def test_every_capability_accepts_a_project(self, tool: PostHogTool) -> None:
        """A capability that cannot be pointed at a project can only read the default,
        which is how "why did OSS engagement drop" came to be answered from SaaS data."""
        for name in tool.capability_names:
            if name == "list_projects":
                continue
            assert "project" in tool.capability(name).params_schema["properties"], name


class TestUpstreamFailures:
    async def test_a_rejected_key_is_surfaced_not_swallowed(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """An unusable credential must not read as an absence of data."""
        patch_client(
            tool,
            lambda request: httpx.Response(401, json={"detail": "Invalid personal API key"}),
            is_async_factory=True,
        )
        with pytest.raises(Exception) as raised:
            await tool.event_trend(
                ctx, event="user signed up", start_date="2026-07-01", end_date="2026-07-07"
            )
        assert "401" in str(raised.value) or "auth" in str(raised.value).lower()


class TestASeriesThatStopsEarlyIsDisclosed:
    """The failure that produced a confidently wrong answer on real data.

    Asked whether signups had fallen, the analyst received a `user signed up` series running
    1-3 August against a range ending on the 15th, computed 206/day from those three days,
    described it as "the August run rate", and concluded there was no real decline. Signup
    tracking had stopped recording twelve days earlier.

    Distinct from `partial_buckets`, and the distinction is the point: a short bucket means the
    month is young and is a reason to be calm; a series that stops means collection may be broken
    and is a reason to escalate. A connector disclosing only the first actively encourages the
    wrong conclusion from the second.
    """

    async def test_a_series_ending_before_the_range_is_flagged_with_its_gap(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(
            tool,
            {
                "/query/": _query_response(
                    ["bucket", "value"],
                    [
                        ["2026-08-01T00:00:00", 174],
                        ["2026-08-02T00:00:00", 250],
                        ["2026-08-03T00:00:00", 195],
                    ],
                )
            },
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx, event="user signed up", start_date="2026-08-01", end_date="2026-08-15"
        )

        assert result.payload["series_ends_early"] == {
            "last_bucket": "2026-08-03",
            "requested_end": "2026-08-15",
            "days_missing": 12,
        }
        # The instruction matters as much as the fact: the wrong answer came from averaging
        # three days and calling it the month.
        assert (
            "Do not compute a rate for the whole requested period"
            in (result.payload["series_gap_note"])
        )

    async def test_it_does_not_interpret_the_gap(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """A series can stop because an event was deprecated, because a deploy broke the SDK, or
        because nobody signed up. The connector cannot tell which and must not guess — naming a
        cause here would put an unevidenced claim into the analyst's context."""
        patch_client(
            tool,
            {"/query/": _query_response(["bucket", "value"], [["2026-08-01T00:00:00", 174]])},
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx, event="user signed up", start_date="2026-08-01", end_date="2026-08-15"
        )
        note = result.payload["series_gap_note"]
        assert "may have stopped firing" in note and "may be broken" in note
        assert "without establishing which" in note

    async def test_a_complete_series_carries_no_gap(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Noise on every ordinary call is a field readers learn to skip, which would cost the
        disclosure its value on the one call that needs it."""
        rows = [[f"2026-08-{day:02d}T00:00:00", 100] for day in range(1, 16)]
        patch_client(
            tool, {"/query/": _query_response(["bucket", "value"], rows)}, is_async_factory=True
        )
        result = await tool.event_trend(
            ctx, event="user signed up", start_date="2026-08-01", end_date="2026-08-15"
        )
        assert "series_ends_early" not in result.payload

    async def test_an_empty_series_is_not_double_reported(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """An empty result is already reported by `total`, `row_count` and the executor's
        empty-observation marking. A second voice saying the same thing adds nothing."""
        patch_client(
            tool, {"/query/": _query_response(["bucket", "value"], [])}, is_async_factory=True
        )
        result = await tool.event_trend(
            ctx, event="never fired", start_date="2026-08-01", end_date="2026-08-15"
        )
        assert "series_ends_early" not in result.payload
        assert result.payload["total"] == 0


def _definitions(events: dict[str, str]) -> dict[str, Any]:
    """An `event_definitions/` page, as `name -> last_seen_at`."""
    return {"results": [{"name": name, "last_seen_at": seen} for name, seen in events.items()]}


class TestTheBlastRadiusOfAGap:
    """Which *other* events stopped when this one did.

    `series_ends_early` was the right disclosure and only half the fix. It correctly refuses to
    say why a series ended, listing three possibilities -- stopped firing, tracking broke, event
    renamed -- and then leaves the analyst holding a warning with nothing to resolve it against.

    On real data the analyst resolved it with whatever was nearest: it found a second metric that
    had also fallen, and asserted a common cause for two failures seven days apart. The sibling
    events settle it without interpretation, because a group of unrelated events falling silent
    within the same day is not something changing user behaviour can produce.
    """

    async def test_a_group_stopping_together_localises_the_failure(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """The real incident. Every server-side product event ceased on 2026-08-03 while every
        browser-autocapture event kept flowing -- which is one emitter dying, not demand."""
        patch_client(
            tool,
            {
                "/query/": _query_response(
                    ["bucket", "value"],
                    [
                        ["2026-08-01T00:00:00", 174],
                        ["2026-08-02T00:00:00", 250],
                        ["2026-08-03T00:00:00", 195],
                    ],
                ),
                "event_definitions/": _definitions(
                    {
                        "user signed up": "2026-08-03T20:36:02.469722Z",
                        "user logged in": "2026-08-03T20:33:46.513354Z",
                        "conversation created": "2026-08-03T20:33:52.648161Z",
                        "credit purchased": "2026-08-03T20:17:37.363805Z",
                        "onboarding completed": "2026-08-03T20:34:03.924923Z",
                        "$pageview": "2026-08-15T12:37:17.261719Z",
                        "$autocapture": "2026-08-15T11:03:07.497747Z",
                        "$pageleave": "2026-08-15T13:24:12.447316Z",
                    }
                ),
            },
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx, event="user signed up", start_date="2026-08-01", end_date="2026-08-15"
        )

        radius = result.payload["blast_radius"]
        assert radius["scope"] == "shared_with_other_events"
        assert radius["stopped_count"] == 5
        assert radius["still_recording_count"] == 3
        assert {entry["name"] for entry in radius["stopped_with_it"]} == {
            "user signed up",
            "user logged in",
            "conversation created",
            "credit purchased",
            "onboarding completed",
        }
        # The names are what localises it -- an analyst compares what the stopped group shares
        # against what the live group shares. A count alone would not have caught this one.
        assert {entry["name"] for entry in radius["still_recording"]} == {
            "$pageview",
            "$autocapture",
            "$pageleave",
        }

    async def test_it_warns_against_the_mistake_that_was_actually_made(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """The wrong answer cited a second metric that collapsed a week later as corroboration.
        Nothing in the payload said that a different date makes it a different incident."""
        patch_client(
            tool,
            {
                "/query/": _query_response(["bucket", "value"], [["2026-08-03T00:00:00", 195]]),
                "event_definitions/": _definitions(
                    {
                        "user signed up": "2026-08-03T20:36:02Z",
                        "user logged in": "2026-08-03T20:33:46Z",
                        "$pageview": "2026-08-15T12:37:17Z",
                    }
                ),
            },
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx, event="user signed up", start_date="2026-08-01", end_date="2026-08-15"
        )
        note = result.payload["blast_radius_note"]
        assert "a separate incident, not corroboration" in note
        assert "user behaviour changed" in note

    async def test_one_event_stopping_alone_points_at_instrumentation(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Everything else still recording rules out a shared pipeline, which leaves a rename,
        broken instrumentation, or a genuine stop -- and says so rather than picking one."""
        patch_client(
            tool,
            {
                "/query/": _query_response(["bucket", "value"], [["2026-08-03T00:00:00", 195]]),
                "event_definitions/": _definitions(
                    {
                        "user signed up": "2026-08-03T20:36:02Z",
                        "$pageview": "2026-08-15T12:37:17Z",
                        "user logged in": "2026-08-15T11:00:00Z",
                        "credit purchased": "2026-08-14T09:00:00Z",
                    }
                ),
            },
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx, event="user signed up", start_date="2026-08-01", end_date="2026-08-15"
        )
        radius = result.payload["blast_radius"]
        assert radius["scope"] == "this_event_only"
        assert radius["stopped_count"] == 1
        note = result.payload["blast_radius_note"]
        assert "specific to this event" in note
        # A rename is the cheapest of the three to check and the easiest to miss.
        assert "new event whose history starts where this one ends" in note

    async def test_everything_stopping_makes_the_event_unremarkable(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """A dead project is an ingestion or credential failure. Nothing about the metric asked
        for is distinctive, and an investigation that reports on the metric has missed it."""
        patch_client(
            tool,
            {
                "/query/": _query_response(["bucket", "value"], [["2026-08-03T00:00:00", 195]]),
                "event_definitions/": _definitions(
                    {
                        "user signed up": "2026-08-03T20:36:02Z",
                        "$pageview": "2026-08-03T21:00:00Z",
                        "user logged in": "2026-08-02T11:00:00Z",
                    }
                ),
            },
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx, event="user signed up", start_date="2026-08-01", end_date="2026-08-15"
        )
        assert result.payload["blast_radius"]["scope"] == "project_wide"
        assert "whole project stopped receiving" in result.payload["blast_radius_note"]

    async def test_a_complete_series_pays_nothing_for_this(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """No gap, no second request. The check is worth a round trip only on the calls whose
        disclosure is otherwise unresolvable."""
        rows = [[f"2026-08-{day:02d}T00:00:00", 100] for day in range(1, 16)]
        transport = patch_client(
            tool, {"/query/": _query_response(["bucket", "value"], rows)}, is_async_factory=True
        )
        result = await tool.event_trend(
            ctx, event="user signed up", start_date="2026-08-01", end_date="2026-08-15"
        )
        assert "blast_radius" not in result.payload
        assert not any("event_definitions" in str(r.url) for r in transport.requests)

    async def test_it_asks_for_stale_definitions(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """An event that stopped six weeks ago is exactly what this is looking for. PostHog hides
        those by default, so excluding them would filter out the evidence."""
        transport = patch_client(
            tool,
            {
                "/query/": _query_response(["bucket", "value"], [["2026-08-03T00:00:00", 195]]),
                "event_definitions/": _definitions({"user signed up": "2026-08-03T20:36:02Z"}),
            },
            is_async_factory=True,
        )
        await tool.event_trend(
            ctx, event="user signed up", start_date="2026-08-01", end_date="2026-08-15"
        )
        definitions = [r for r in transport.requests if "event_definitions" in str(r.url)]
        assert definitions, "the gap should have triggered a definitions lookup"
        assert "exclude_stale=false" in str(definitions[0].url).lower()

    async def test_a_failed_lookup_does_not_take_the_trend_with_it(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """This enriches a disclosure. A trend that was fetched successfully must still return."""
        patch_client(
            tool,
            {"/query/": _query_response(["bucket", "value"], [["2026-08-03T00:00:00", 195]])},
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx, event="user signed up", start_date="2026-08-01", end_date="2026-08-15"
        )
        assert result.payload["series_ends_early"]["days_missing"] == 12
        assert "blast_radius" not in result.payload


class TestTheMovementTravelsWithTheSeries:
    """When the series moved, computed on the rows just fetched.

    Same reason as `blast_radius`: seven fixes in this codebase have had the shape "the
    connector disclosed a problem, supplied no resolution, and the analyst resolved it with
    whatever was nearest". A fact requiring a second call is a fact that does not get obtained.
    """

    @staticmethod
    def _daily(levels: list[tuple[int, float]]) -> list[list[Any]]:
        """`(count, value)` runs, as HogQL rows from 2026-05-06."""
        generator = random.Random(4)
        rows: list[list[Any]] = []
        day = date(2026, 5, 6)
        for count, value in levels:
            for _ in range(count):
                rows.append([f"{day.isoformat()}T00:00:00", value + generator.gauss(0, 6)])
                day += timedelta(days=1)
        return rows

    async def test_a_level_shift_is_reported_with_its_significance(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """A long run before the break and a shorter one after -- the configuration the
        permutation test has power in. See `ConformalResult.resolvable`."""
        patch_client(
            tool,
            {
                "/query/": _query_response(
                    ["bucket", "value"], self._daily([(65, 200.0), (25, 90.0)])
                )
            },
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx, event="user signed up", start_date="2026-05-06", end_date="2026-08-03"
        )
        shifts = result.payload["movement"]["level_shifts"]
        assert len(shifts) == 1
        assert shifts[0]["change_per_day"] < -100
        assert shifts[0]["established"] is True
        assert shifts[0]["p_value"] <= 0.05

    async def test_an_untestable_break_is_null_with_a_reason_not_false(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """The power cliff, surfaced honestly. With the post period no shorter than the pre
        period the test cannot reject whatever the data show, so reporting `established: false`
        would say "we checked and it is not real" -- the opposite of what happened."""
        patch_client(
            tool,
            {
                "/query/": _query_response(
                    ["bucket", "value"], self._daily([(30, 200.0), (60, 90.0)])
                )
            },
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx, event="user signed up", start_date="2026-05-06", end_date="2026-08-03"
        )
        shift = result.payload["movement"]["level_shifts"][0]
        assert shift["established"] is None
        assert "not shorter than" in shift["not_established_because"]
        assert "p_value" not in shift
        note = result.payload["movement_note"]
        assert "not the same as being tested and found unreal" in note
        assert "movements of unknown significance, not as noise" in note

    async def test_the_payload_forbids_reading_a_cause_out_of_it(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """The note is where the analyst is reading, rather than in a system prompt from
        thousands of tokens earlier."""
        patch_client(
            tool,
            {
                "/query/": _query_response(
                    ["bucket", "value"], self._daily([(45, 200.0), (45, 90.0)])
                )
            },
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx, event="user signed up", start_date="2026-05-06", end_date="2026-08-03"
        )
        note = result.payload["movement_note"]
        assert "nothing about why" in note
        assert "precedes a candidate cause's own date rules that cause out" in note

    async def test_a_weekly_series_carries_no_movement_field(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """The method is calibrated on daily buckets against a weekly cycle. Running it on
        weekly data would be a confident answer from an uncalibrated method."""
        patch_client(
            tool,
            {
                "/query/": _query_response(
                    ["bucket", "value"], self._daily([(45, 200.0), (45, 90.0)])
                )
            },
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx,
            event="user signed up",
            start_date="2026-05-06",
            end_date="2026-08-03",
            interval="week",
        )
        assert "movement" not in result.payload

    async def test_a_short_series_carries_no_movement_field(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Fewer than two minimum segments has nothing to split, and an ordinary short query
        should not pay a disclosure for it."""
        patch_client(
            tool,
            {"/query/": _query_response(["bucket", "value"], self._daily([(10, 200.0)]))},
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx, event="user signed up", start_date="2026-05-06", end_date="2026-05-15"
        )
        assert "movement" not in result.payload


class TestTheDataTrustStateTravelsWithTheSeries:
    """ADR 0005 decision 1, in the payload beside the numbers.

    Saying it here is what stops the analyst reaching for the nearest falling line instead: the
    original wrong answer explained a signup cessation with a pageview collapse that began seven
    days later, and every disclosure it needed was already on the page in pieces.
    """

    async def test_a_correlated_cessation_marks_the_series_broken(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(
            tool,
            {
                "/query/": _query_response(
                    ["bucket", "value"],
                    [
                        ["2026-08-01T00:00:00", 174],
                        ["2026-08-02T00:00:00", 250],
                        ["2026-08-03T00:00:00", 195],
                    ],
                ),
                "event_definitions/": _definitions(
                    {
                        "user signed up": "2026-08-03T20:36:02Z",
                        "user logged in": "2026-08-03T20:33:46Z",
                        "credit purchased": "2026-08-03T20:17:37Z",
                        "$pageview": "2026-08-15T12:37:17Z",
                        "$autocapture": "2026-08-15T11:03:07Z",
                    }
                ),
            },
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx, event="user signed up", start_date="2026-08-01", end_date="2026-08-15"
        )
        trust = result.payload["data_trust"]
        assert trust["state"] == "broken"
        assert trust["may_answer_the_business_question"] is False
        note = result.payload["data_trust_note"]
        assert "only an absence of measurement" in note

    async def test_it_names_the_checks_it_could_not_run(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Five of the gate's eleven rows need metadata this project does not collect. Showing
        only the runnable checks would make the gate look complete."""
        patch_client(
            tool,
            {"/query/": _query_response(["bucket", "value"], [["2026-08-03T00:00:00", 195]])},
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx, event="user signed up", start_date="2026-08-01", end_date="2026-08-15"
        )
        assert len(result.payload["data_trust"]["not_evaluated"]) == 5

    async def test_an_ordinary_series_carries_no_trust_field(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """A disclosure on every call is one a reader learns to skip, which would cost it its
        value on the call that needed it."""
        rows = [[f"2026-08-{day:02d}T00:00:00", 100] for day in range(1, 16)]
        patch_client(
            tool, {"/query/": _query_response(["bucket", "value"], rows)}, is_async_factory=True
        )
        result = await tool.event_trend(
            ctx, event="user signed up", start_date="2026-08-01", end_date="2026-08-15"
        )
        assert "data_trust" not in result.payload
        assert "data_trust_note" not in result.payload


class TestARenamedEventIsNamedBesideTheMovement:
    """The failure five live attempts at one question produced, and none of them hallucinated.

    Asked *did conversation volume change in August 2026*, against real PostHog data, the
    analyst returned "it did not fall", "it fell 78%", "it rose 18x", and twice "the premise
    does not hold" — entirely according to whether that attempt happened to query both
    `conversation_created` and `agent_server.conversation_created`. The one attempt that read a
    single series reported *"a confirmed level shift of about 78%"*: every figure real, every
    citation resolving, and wrong.

    No grounding mechanism can see that, because nothing is ungrounded. So the connector says
    it, next to the movement, for the same reason `blast_radius` is fetched next to the gap —
    the analyst has to hold both to interpret either, and in production it did not go back for
    the second call.
    """

    #: A series with a level shift the movement check can actually confirm.
    #:
    #: Noisy on purpose, and that is not decoration: `describe_movement` estimates a noise
    #: scale from the series to test a candidate shift against, so a *perfectly* clean step
    #: from 180 to 40 has zero estimated variance and is reported as no shift at all. Every
    #: split tried -- 15/15, 20/10, 30/14 -- found nothing until the values carried noise.
    #: The real series that exposed this whole defect had noise, which is why it registered.
    COLLAPSE = [
        ["2026-08-01T00:00:00", 175],
        ["2026-08-02T00:00:00", 170],
        ["2026-08-03T00:00:00", 184],
        ["2026-08-04T00:00:00", 168],
        ["2026-08-05T00:00:00", 181],
        ["2026-08-06T00:00:00", 176],
        ["2026-08-07T00:00:00", 167],
        ["2026-08-08T00:00:00", 180],
        ["2026-08-09T00:00:00", 167],
        ["2026-08-10T00:00:00", 178],
        ["2026-08-11T00:00:00", 168],
        ["2026-08-12T00:00:00", 168],
        ["2026-08-13T00:00:00", 178],
        ["2026-08-14T00:00:00", 189],
        ["2026-08-15T00:00:00", 169],
        ["2026-08-16T00:00:00", 38],
        ["2026-08-17T00:00:00", 41],
        ["2026-08-18T00:00:00", 43],
        ["2026-08-19T00:00:00", 40],
        ["2026-08-20T00:00:00", 39],
        ["2026-08-21T00:00:00", 43],
        ["2026-08-22T00:00:00", 37],
        ["2026-08-23T00:00:00", 42],
        ["2026-08-24T00:00:00", 39],
        ["2026-08-25T00:00:00", 38],
        ["2026-08-26T00:00:00", 38],
        ["2026-08-27T00:00:00", 39],
        ["2026-08-28T00:00:00", 42],
        ["2026-08-29T00:00:00", 38],
        ["2026-08-30T00:00:00", 41],
    ]

    async def test_the_sibling_is_named_when_the_series_moves(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: Any
    ) -> None:
        patch_client(
            tool,
            {
                "/query/": _query_response(["bucket", "value"], self.COLLAPSE),
                "event_definitions/": _definitions(
                    {
                        "conversation_created": "2026-08-28T10:00:00Z",
                        "agent_server.conversation_created": "2026-09-06T10:00:00Z",
                        "settings saved": "2026-09-06T10:00:00Z",
                    }
                ),
            },
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx, event="conversation_created", start_date="2026-08-01", end_date="2026-08-30"
        )

        related = result.payload["related_events"]
        assert [entry["name"] for entry in related["events"]] == [
            "agent_server.conversation_created"
        ]
        assert related["matched_on"] == ["conversation"]
        assert "may be a movement in what is being recorded" in related["note"]

    async def test_a_shared_lifecycle_verb_is_not_a_relation(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: Any
    ) -> None:
        """`api key created` came back "related" to `conversation_created` on the strength of
        the word "created" alone. Almost every product event ends in a lifecycle verb, so
        matching on one relates almost anything to anything — what makes two events candidates
        for measuring one concept is a shared subject."""
        patch_client(
            tool,
            {
                "/query/": _query_response(["bucket", "value"], self.COLLAPSE),
                "event_definitions/": _definitions(
                    {
                        "api key created": "2026-09-06T10:00:00Z",
                        "billing portal opened": "2026-09-06T10:00:00Z",
                        "workspace renamed": "2026-09-06T10:00:00Z",
                    }
                ),
            },
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx, event="conversation_created", start_date="2026-08-01", end_date="2026-08-28"
        )

        assert "related_events" not in result.payload

    async def test_a_flat_series_costs_no_extra_request(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: Any
    ) -> None:
        """The cost discipline `blast_radius` established: one extra request on the calls that
        need it and nothing on the calls that do not. A series that did not move needs no
        sibling to interpret it."""
        patch_client(
            tool,
            {
                "/query/": _query_response(
                    ["bucket", "value"],
                    [[f"2026-08-{day:02d}T00:00:00", 180] for day in range(1, 31)],
                ),
                "event_definitions/": _definitions(
                    {"agent_server.conversation_created": "2026-09-06T10:00:00Z"}
                ),
            },
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx, event="conversation_created", start_date="2026-08-01", end_date="2026-08-30"
        )

        assert not result.payload["movement"]["level_shifts"]
        assert "related_events" not in result.payload

    async def test_a_dead_sibling_is_kept_because_a_handover_is_the_point(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: Any
    ) -> None:
        """The old event dies as the new one starts. A filter that dropped stale siblings would
        drop the more informative half of a migration."""
        patch_client(
            tool,
            {
                "/query/": _query_response(["bucket", "value"], self.COLLAPSE),
                "event_definitions/": _definitions(
                    {"conversation_started": "2026-08-11T10:00:00Z"}
                ),
            },
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx, event="conversation_created", start_date="2026-08-01", end_date="2026-08-30"
        )

        events = result.payload["related_events"]["events"]
        assert events == [{"name": "conversation_started", "last_seen_at": "2026-08-11"}]

    async def test_a_noisy_project_is_capped_rather_than_dumped(
        self, tool: PostHogTool, ctx: ToolContext, patch_client: Any
    ) -> None:
        """A project with `$pageview_1` through `$pageview_84` would bury the disclosure it is
        supposed to be. Capped, most recent first, with the true count in the note."""
        patch_client(
            tool,
            {
                "/query/": _query_response(["bucket", "value"], self.COLLAPSE),
                "event_definitions/": _definitions(
                    {
                        f"conversation_variant_{n}": f"2026-08-{n:02d}T10:00:00Z"
                        for n in range(1, 21)
                    }
                ),
            },
            is_async_factory=True,
        )
        result = await tool.event_trend(
            ctx, event="conversation_created", start_date="2026-08-01", end_date="2026-08-30"
        )

        related = result.payload["related_events"]
        assert related["count"] == 20
        assert len(related["events"]) == 5
        assert related["events"][0]["last_seen_at"] == "2026-08-20"
        assert "the 5 most recent shown" in related["note"]
