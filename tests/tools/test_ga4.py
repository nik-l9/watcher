"""GA4 connector.

The load-bearing behaviour here is compare_periods: it rejoins GA4's two-range
response on identical row keys and computes the deltas. If that join is wrong, every
"conversion fell from 4.2% to 2.9%" claim in a report is wrong while looking
perfectly plausible.

Percentage change against a zero baseline is the other case worth pinning — the
answer must be None, never a fabricated number.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from cortex.tools.base import InvalidParams, ToolContext
from cortex.tools.ga4 import GA4Tool, _percent


def _report(
    dimension_headers: list[str],
    metric_headers: list[str],
    rows: list[tuple[list[str], list[str]]],
    totals: list[str] | None = None,
) -> dict:
    body: dict = {
        "dimensionHeaders": [{"name": n} for n in dimension_headers],
        "metricHeaders": [{"name": n} for n in metric_headers],
        "rows": [
            {
                "dimensionValues": [{"value": v} for v in dims],
                "metricValues": [{"value": v} for v in mets],
            }
            for dims, mets in rows
        ],
        "metadata": {"currencyCode": "USD", "timeZone": "UTC"},
        "rowCount": len(rows),
    }
    if totals:
        body["totals"] = [{"metricValues": [{"value": v} for v in totals]}]
    return body


@pytest.fixture
def tool() -> GA4Tool:
    return GA4Tool()


@pytest.fixture
def ctx(tenant: object) -> ToolContext:
    return ToolContext(
        tenant=tenant,  # type: ignore[arg-type]
        credential="{}",
        credential_metadata={"property_id": "properties/123456789"},
    )


class TestPropertyResolution:
    async def test_property_comes_from_connector_metadata(
        self, tool: GA4Tool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Not a tool parameter: the property belongs to the tenant's connection,
        and a model-supplied id could reach Google."""
        transport = patch_client(
            tool,
            {":runReport": _report(["date"], ["sessions"], [(["20260720"], ["100"])])},
            is_async_factory=True,
        )
        await tool.get_sessions(ctx, start_date="2026-07-01", end_date="2026-07-07")
        assert "/properties/123456789:runReport" in str(transport.requests[0].url)

    async def test_missing_property_is_a_clear_error(self, tool: GA4Tool, tenant: object) -> None:
        bare = ToolContext(tenant=tenant, credential="{}", credential_metadata={})  # type: ignore[arg-type]
        with pytest.raises(InvalidParams, match="no property_id"):
            await tool.get_sessions(bare, start_date="2026-07-01", end_date="2026-07-07")

    @pytest.mark.parametrize(
        "bad", ["abc", "properties/abc", "123", "1234567890123456789012", "12345;DROP"]
    )
    async def test_non_numeric_property_is_rejected(
        self, tool: GA4Tool, tenant: object, bad: str
    ) -> None:
        ctx = ToolContext(
            tenant=tenant,  # type: ignore[arg-type]
            credential="{}",
            credential_metadata={"property_id": bad},
        )
        with pytest.raises(InvalidParams, match="numeric"):
            await tool.get_sessions(ctx, start_date="2026-07-01", end_date="2026-07-07")

    async def test_bare_numeric_property_is_accepted(
        self, tool: GA4Tool, tenant: object, patch_client: object
    ) -> None:
        """Users paste the id both with and without the 'properties/' prefix."""
        ctx = ToolContext(
            tenant=tenant,  # type: ignore[arg-type]
            credential="{}",
            credential_metadata={"property_id": "123456789"},
        )
        patch_client(tool, {":runReport": _report([], ["sessions"], [])}, is_async_factory=True)
        result = await tool.get_sessions(ctx, start_date="2026-07-01", end_date="2026-07-07")
        assert result.payload["property_id"] == "123456789"


class TestGetSessions:
    async def test_parses_and_types_metrics(
        self, tool: GA4Tool, ctx: ToolContext, patch_client: object
    ) -> None:
        """GA4 returns every metric as a string; charts need typed numbers."""
        patch_client(
            tool,
            {
                ":runReport": _report(
                    ["deviceCategory"],
                    ["sessions", "activeUsers", "newUsers"],
                    [(["mobile"], ["1200", "980", "310"]), (["desktop"], ["800", "700", "120"])],
                    totals=["2000", "1680", "430"],
                )
            },
            is_async_factory=True,
        )
        result = await tool.get_sessions(
            ctx, start_date="2026-07-01", end_date="2026-07-07", dimensions=["deviceCategory"]
        )

        assert result.payload["row_count"] == 2
        mobile = result.payload["rows"][0]
        assert mobile["dimensions"]["deviceCategory"] == "mobile"
        assert mobile["metrics"]["sessions"] == 1200
        assert isinstance(mobile["metrics"]["sessions"], int)
        assert result.payload["totals"]["sessions"] == 2000

    async def test_reversed_date_range_is_rejected(self, tool: GA4Tool, ctx: ToolContext) -> None:
        with pytest.raises(InvalidParams, match="after end_date"):
            await tool.get_sessions(ctx, start_date="2026-07-31", end_date="2026-07-01")

    async def test_non_numeric_metric_becomes_none(
        self, tool: GA4Tool, ctx: ToolContext, patch_client: object
    ) -> None:
        """An unparseable value must not silently become zero."""
        patch_client(
            tool,
            {":runReport": _report([], ["sessions"], [([], ["not-a-number"])])},
            is_async_factory=True,
        )
        result = await tool.get_sessions(ctx, start_date="2026-07-01", end_date="2026-07-07")
        assert result.payload["rows"][0]["metrics"]["sessions"] is None


class TestComparePeriods:
    RESPONSE = _report(
        ["deviceCategory", "dateRange"],
        ["sessions", "conversions"],
        [
            (["mobile", "current"], ["900", "26"]),
            (["mobile", "previous"], ["1200", "50"]),
            (["desktop", "current"], ["800", "40"]),
            (["desktop", "previous"], ["780", "38"]),
        ],
    )

    async def test_rejoins_rows_and_computes_deltas(
        self, tool: GA4Tool, ctx: ToolContext, patch_client: object
    ) -> None:
        """GA4 returns one row per (dimensions, range); they must be rejoined on
        identical keys or every delta in the report is wrong."""
        patch_client(tool, {":runReport": self.RESPONSE}, is_async_factory=True)
        result = await tool.compare_periods(
            ctx,
            current_start="2026-07-08",
            current_end="2026-07-14",
            previous_start="2026-07-01",
            previous_end="2026-07-07",
            dimensions=["deviceCategory"],
        )

        by_device = {
            row["dimensions"]["deviceCategory"]: row for row in result.payload["comparison"]
        }
        mobile = by_device["mobile"]["sessions"]
        assert mobile["current"] == 900
        assert mobile["previous"] == 1200
        assert mobile["absolute_change"] == -300
        assert mobile["percent_change"] == -25.0

        desktop = by_device["desktop"]["sessions"]
        assert desktop["absolute_change"] == 20

    async def test_issues_a_single_request(
        self, tool: GA4Tool, ctx: ToolContext, patch_client: object
    ) -> None:
        """One request with two dateRanges, not two requests the model must align."""
        transport = patch_client(tool, {":runReport": self.RESPONSE}, is_async_factory=True)
        await tool.compare_periods(
            ctx,
            current_start="2026-07-08",
            current_end="2026-07-14",
            previous_start="2026-07-01",
            previous_end="2026-07-07",
        )
        assert len(transport.requests) == 1
        body = transport.request_bodies()[0]
        assert [r["name"] for r in body["dateRanges"]] == ["current", "previous"]

    async def test_row_present_in_only_one_period(
        self, tool: GA4Tool, ctx: ToolContext, patch_client: object
    ) -> None:
        """A segment that appeared or vanished must still be reported, with the
        missing side as None rather than zero."""
        patch_client(
            tool,
            {
                ":runReport": _report(
                    ["deviceCategory", "dateRange"],
                    ["sessions"],
                    [(["tablet", "current"], ["40"])],
                )
            },
            is_async_factory=True,
        )
        result = await tool.compare_periods(
            ctx,
            current_start="2026-07-08",
            current_end="2026-07-14",
            previous_start="2026-07-01",
            previous_end="2026-07-07",
            dimensions=["deviceCategory"],
        )
        row = result.payload["comparison"][0]["sessions"]
        assert row["current"] == 40
        assert row["previous"] is None
        assert row["absolute_change"] is None
        assert row["percent_change"] is None

    async def test_works_without_dimensions(
        self, tool: GA4Tool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(
            tool,
            {
                ":runReport": _report(
                    ["dateRange"],
                    ["sessions"],
                    [(["current"], ["900"]), (["previous"], ["1000"])],
                )
            },
            is_async_factory=True,
        )
        result = await tool.compare_periods(
            ctx,
            current_start="2026-07-08",
            current_end="2026-07-14",
            previous_start="2026-07-01",
            previous_end="2026-07-07",
        )
        assert result.payload["row_count"] == 1
        assert result.payload["comparison"][0]["sessions"]["percent_change"] == -10.0


class TestPercentChange:
    """A fabricated percentage in a grounded report is a hallucination even though
    it came from arithmetic."""

    def test_zero_baseline_has_no_percentage(self) -> None:
        assert _percent(500, 0) is None

    def test_missing_values_have_no_percentage(self) -> None:
        assert _percent(None, 100) is None
        assert _percent(100, None) is None

    def test_both_zero(self) -> None:
        assert _percent(0, 0) is None

    @pytest.mark.parametrize(
        ("current", "previous", "expected"),
        [(1200, 1000, 20.0), (800, 1000, -20.0), (1000, 1000, 0.0), (1, 3, -66.67)],
    )
    def test_computes_and_rounds(self, current: int, previous: int, expected: float) -> None:
        assert _percent(current, previous) == expected


class TestGetFunnel:
    async def test_recomputes_conversion_rate(
        self, tool: GA4Tool, ctx: ToolContext, patch_client: object
    ) -> None:
        """sessionConversionRate is unavailable on some properties; a missing rate
        would otherwise read as zero."""
        patch_client(
            tool,
            {
                ":runReport": _report(
                    ["deviceCategory"],
                    ["sessions", "conversions", "sessionConversionRate", "engagementRate"],
                    [(["mobile"], ["1000", "29", "", "0.55"])],
                )
            },
            is_async_factory=True,
        )
        result = await tool.get_funnel(ctx, start_date="2026-07-01", end_date="2026-07-07")
        row = result.payload["rows"][0]
        assert row["metrics"]["sessionConversionRate"] is None
        assert row["derived_conversion_rate"] == 0.029

    async def test_zero_sessions_yields_no_rate(
        self, tool: GA4Tool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(
            tool,
            {
                ":runReport": _report(
                    ["deviceCategory"],
                    ["sessions", "conversions", "sessionConversionRate", "engagementRate"],
                    [(["tablet"], ["0", "0", "0", "0"])],
                )
            },
            is_async_factory=True,
        )
        result = await tool.get_funnel(ctx, start_date="2026-07-01", end_date="2026-07-07")
        assert result.payload["rows"][0]["derived_conversion_rate"] is None


class TestDataQuality:
    async def test_surfaces_thresholding_and_timezone(
        self, tool: GA4Tool, ctx: ToolContext, patch_client: object
    ) -> None:
        """A sampled or truncated figure presented as exact is a grounding failure
        even when the call was real."""
        body = _report([], ["sessions"], [([], ["100"])])
        body["metadata"]["dataLossFromOtherRow"] = True
        patch_client(tool, {":runReport": body}, is_async_factory=True)

        result = await tool.get_sessions(ctx, start_date="2026-07-01", end_date="2026-07-07")
        assert result.meta["data_loss_from_other_row"] is True
        assert result.meta["time_zone"] == "UTC"


class TestAllowlists:
    def test_metric_enum_is_closed(self, tool: GA4Tool) -> None:
        """A hallucinated metric must fail validation, not produce an opaque 400."""
        schema = tool.capability("run_report").params_schema
        assert "sessions" in schema["properties"]["metrics"]["items"]["enum"]
        assert "madeUpMetric" not in schema["properties"]["metrics"]["items"]["enum"]

    def test_dimension_enum_is_closed(self, tool: GA4Tool) -> None:
        schema = tool.capability("run_report").params_schema
        assert "deviceCategory" in schema["properties"]["dimensions"]["items"]["enum"]

    def test_row_limits_are_bounded(self, tool: GA4Tool) -> None:
        """No single call may pull an unbounded page into an LLM context."""
        for name in tool.capability_names:
            properties = tool.capability(name).params_schema["properties"]
            if "limit" in properties:
                assert properties["limit"]["maximum"] <= 250, name


class TestFailures:
    async def test_google_error_is_surfaced(
        self, tool: GA4Tool, ctx: ToolContext, patch_client: object
    ) -> None:
        from cortex.tools.http import AuthRejected

        patch_client(
            tool,
            {":runReport": httpx.Response(403, json={"error": {"message": "denied"}})},
            is_async_factory=True,
        )
        with pytest.raises(AuthRejected):
            await tool.get_sessions(ctx, start_date="2026-07-01", end_date="2026-07-07")


class TestTopPages:
    async def test_returns_pages_with_engagement(
        self, tool: GA4Tool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(
            tool,
            {
                ":runReport": _report(
                    ["pagePath"],
                    ["screenPageViews", "sessions", "engagementRate"],
                    [
                        (["/signup"], ["4200", "3100", "0.62"]),
                        (["/pricing"], ["1800", "1500", "0.71"]),
                    ],
                )
            },
            is_async_factory=True,
        )
        result = await tool.top_pages(ctx, start_date="2026-07-01", end_date="2026-07-07")

        assert result.payload["row_count"] == 2
        first = result.payload["rows"][0]
        assert first["dimensions"]["pagePath"] == "/signup"
        assert first["metrics"]["screenPageViews"] == 4200
        assert result.source_ref
        assert result.meta["time_zone"] == "UTC"

    async def test_limit_is_forwarded(
        self, tool: GA4Tool, ctx: ToolContext, patch_client: object
    ) -> None:
        transport = patch_client(
            tool, {":runReport": _report(["pagePath"], ["sessions"], [])}, is_async_factory=True
        )
        await tool.top_pages(ctx, start_date="2026-07-01", end_date="2026-07-07", limit=5)
        assert transport.request_bodies()[0]["limit"] == "5"


class TestRunReport:
    async def test_returns_arbitrary_allowlisted_metrics(
        self, tool: GA4Tool, ctx: ToolContext, patch_client: object
    ) -> None:
        """The escape hatch for questions no specific capability covers. Still bounded
        by the metric and dimension allowlists."""
        patch_client(
            tool,
            {
                ":runReport": _report(
                    ["country"],
                    ["totalRevenue", "activeUsers"],
                    [(["United States"], ["48000", "1200"])],
                    totals=["48000", "1200"],
                )
            },
            is_async_factory=True,
        )
        result = await tool.run_report(
            ctx,
            start_date="2026-07-01",
            end_date="2026-07-07",
            metrics=["totalRevenue", "activeUsers"],
            dimensions=["country"],
        )

        assert result.payload["metrics"] == ["totalRevenue", "activeUsers"]
        assert result.payload["totals"]["totalRevenue"] == 48000
        assert result.payload["rows"][0]["dimensions"]["country"] == "United States"
        assert result.freshness.value == "live"

    async def test_works_without_dimensions(
        self, tool: GA4Tool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(
            tool,
            {":runReport": _report([], ["sessions"], [([], ["9000"])], totals=["9000"])},
            is_async_factory=True,
        )
        result = await tool.run_report(
            ctx, start_date="2026-07-01", end_date="2026-07-07", metrics=["sessions"]
        )
        assert result.payload["dimensions"] == []
        assert result.payload["totals"]["sessions"] == 9000

    async def test_reversed_range_is_rejected(self, tool: GA4Tool, ctx: ToolContext) -> None:
        with pytest.raises(InvalidParams, match="after end_date"):
            await tool.run_report(
                ctx, start_date="2026-07-31", end_date="2026-07-01", metrics=["sessions"]
            )


class TestASeriesThatStopsEarly:
    """The disclosure that stopped the worst answer this project has recorded, reaching GA4.

    A `user signed up` series ran 1-3 August against a range ending on the 15th; the analyst
    averaged the three days into "the August run rate" and concluded there was no real decline,
    when collection had stopped twelve days earlier. That fix landed in PostHog only, while 31 of
    42 captured GA4 `get_sessions` calls asked for a date-dimensioned series -- the same defect
    on a different source, with nothing to stop it.
    """

    async def test_a_stopped_series_is_disclosed(
        self, tool: GA4Tool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(
            tool,
            {
                ":runReport": _report(
                    ["date"],
                    ["sessions", "activeUsers", "newUsers"],
                    [
                        (["20260801"], ["210", "180", "40"]),
                        (["20260802"], ["198", "171", "36"]),
                        (["20260803"], ["206", "175", "38"]),
                    ],
                )
            },
            is_async_factory=True,
        )
        result = await tool.get_sessions(
            ctx, start_date="2026-08-01", end_date="2026-08-15", dimensions=["date"]
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

    async def test_the_alternatives_are_the_ones_ga4_actually_has(
        self, tool: GA4Tool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Not PostHog's list. A GA4 metric name is fixed by Google and cannot be renamed, so
        offering "it may have been renamed" would name a cause that cannot happen here and omit
        the one that does -- the tag being removed while the property keeps existing."""
        patch_client(
            tool,
            {":runReport": _report(["date"], ["sessions"], [(["20260801"], ["210"])])},
            is_async_factory=True,
        )
        result = await tool.get_sessions(
            ctx, start_date="2026-08-01", end_date="2026-08-15", dimensions=["date"]
        )
        note = result.payload["series_gap_note"]
        assert "measurement tag may have been removed" in note
        assert "renamed" not in note

    async def test_it_degrades_but_cannot_block(
        self, tool: GA4Tool, ctx: ToolContext, patch_client: object
    ) -> None:
        """The honest ceiling for GA4, not a gap to close.

        Blocking is reserved for a correlated cessation -- a group of series falling silent while
        a sibling keeps recording -- and GA4 exposes nothing to correlate against. A gate that
        blocked on a single stopped series would be blocking on evidence that cannot rule out a
        genuine drop to zero.
        """
        patch_client(
            tool,
            {":runReport": _report(["date"], ["sessions"], [(["20260801"], ["210"])])},
            is_async_factory=True,
        )
        result = await tool.get_sessions(
            ctx, start_date="2026-08-01", end_date="2026-08-15", dimensions=["date"]
        )
        trust = result.payload["data_trust"]
        assert trust["state"] == "degraded"
        assert trust["may_answer_the_business_question"] is True
        assert [t["check"] for t in trust["tripped"]] == ["gate4_single_cessation"]

    async def test_a_complete_series_carries_nothing(
        self, tool: GA4Tool, ctx: ToolContext, patch_client: object
    ) -> None:
        """A disclosure that fires on every ordinary call is one a reader learns to skip."""
        patch_client(
            tool,
            {
                ":runReport": _report(
                    ["date"],
                    ["sessions"],
                    [([f"202608{day:02d}"], ["210"]) for day in range(1, 16)],
                )
            },
            is_async_factory=True,
        )
        result = await tool.get_sessions(
            ctx, start_date="2026-08-01", end_date="2026-08-15", dimensions=["date"]
        )
        assert "series_ends_early" not in result.payload
        assert "data_trust" not in result.payload

    async def test_a_breakdown_is_not_a_series(
        self, tool: GA4Tool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Without a `date` dimension these rows are a breakdown, and have no last bucket.

        Inventing one from row order would disclose a fact about the sort, not about the data.
        """
        patch_client(
            tool,
            {
                ":runReport": _report(
                    ["country"],
                    ["sessions"],
                    [(["India"], ["900"]), (["United States"], ["620"])],
                )
            },
            is_async_factory=True,
        )
        result = await tool.get_sessions(
            ctx, start_date="2026-08-01", end_date="2026-08-15", dimensions=["country"]
        )
        assert "series_ends_early" not in result.payload
        assert "data_trust" not in result.payload


class TestAComparisonSaysWhetherItsWindowsAreComparable:
    """The defect that produced a confidently wrong headline three separate times.

    Asked to compare August against July, an analyst asks for 2026-08-01..08-31 against
    07-01..07-31 — two windows of equal *length*. When the property's data stops on 12 August,
    GA4 answers with twelve days of sessions against thirty-one and calls it a 61.8% decline,
    which is 12/31 restated. Seen in eval run 36, in a five-attempt repeat, and again in a
    ten-attempt repeat; each time the report led with the phantom figure.

    Nothing in the comparison payload can show it. The comparison is grouped by dimension, not
    by date, so twelve days and thirty-one days of sessions are the same single number — which
    is why this costs one extra dated request, and why it is worth it.
    """

    @classmethod
    def _handler(cls, days: list[str]) -> Any:
        """Dispatch on the request body, because the two calls differ by what they ask for.

        The comparison asks for two `dateRanges`; the coverage query asks for one range broken
        down by `date`. A response map keyed on the URL cannot tell them apart -- both are
        `:runReport` -- so the body is what distinguishes them.
        """
        import json as _json

        def _respond(request: httpx.Request) -> httpx.Response:
            body = _json.loads(request.content)
            wants_dates = [d.get("name") for d in body.get("dimensions") or []] == ["date"]
            return httpx.Response(200, json=cls._dated(days) if wants_dates else cls._comparison())

        return _respond

    @staticmethod
    def _dated(days: list[str]) -> dict[str, Any]:
        return {
            "dimensionHeaders": [{"name": "date"}],
            "metricHeaders": [{"name": "sessions"}],
            "rows": [
                {"dimensionValues": [{"value": day}], "metricValues": [{"value": "100"}]}
                for day in days
            ],
        }

    @staticmethod
    def _comparison() -> dict[str, Any]:
        return {
            "dimensionHeaders": [{"name": "dateRange"}],
            "metricHeaders": [{"name": "sessions"}],
            "rows": [
                {
                    "dimensionValues": [{"value": "date_range_0"}],
                    "metricValues": [{"value": "1200"}],
                },
                {
                    "dimensionValues": [{"value": "date_range_1"}],
                    "metricValues": [{"value": "3100"}],
                },
            ],
        }

    #: July complete, August stopping on the 12th. The shape that goes wrong.
    TRUNCATED = [f"2026-07-{d:02d}" for d in range(1, 32)] + [
        f"2026-08-{d:02d}" for d in range(1, 13)
    ]

    async def test_unequal_coverage_is_disclosed_with_the_reason(
        self, tool: GA4Tool, ctx: ToolContext, patch_client: Any
    ) -> None:
        patch_client(tool, self._handler(self.TRUNCATED), is_async_factory=True)
        result = await tool.compare_periods(
            ctx,
            current_start="2026-08-01",
            current_end="2026-08-31",
            previous_start="2026-07-01",
            previous_end="2026-07-31",
        )

        coverage = result.payload["window_coverage"]
        assert coverage["comparable"] is False
        assert coverage["current"] == {
            "requested_days": 31,
            "days_with_data": 12,
            "last_day_with_data": "2026-08-12",
        }
        assert coverage["previous"]["days_with_data"] == 31
        assert "is not a change in the metric" in coverage["note"]

    async def test_equal_coverage_says_nothing(
        self, tool: GA4Tool, ctx: ToolContext, patch_client: Any
    ) -> None:
        """A fair comparison needs no note, and a note on every month-scale call is a note
        nobody reads on the one that needed it."""
        complete = [f"2026-07-{d:02d}" for d in range(1, 32)] + [
            f"2026-08-{d:02d}" for d in range(1, 32)
        ]
        patch_client(tool, self._handler(complete), is_async_factory=True)
        result = await tool.compare_periods(
            ctx,
            current_start="2026-08-01",
            current_end="2026-08-31",
            previous_start="2026-07-01",
            previous_end="2026-07-31",
        )

        assert "window_coverage" not in result.payload

    async def test_a_short_comparison_pays_nothing_for_this(
        self, tool: GA4Tool, ctx: ToolContext, patch_client: Any
    ) -> None:
        """Under a fortnight the shortfall is visible in the figures themselves, so the extra
        request is not worth making — the same discipline `blast_radius` set."""
        transport = patch_client(tool, self._handler([]), is_async_factory=True)
        await tool.compare_periods(
            ctx,
            current_start="2026-08-08",
            current_end="2026-08-14",
            previous_start="2026-08-01",
            previous_end="2026-08-07",
        )

        assert len(transport.requests) == 1
