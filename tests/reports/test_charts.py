"""Charts.

Pydantic proves a chart's shape. These tests are about the things it cannot prove, which are
the ones that mislead a reader: a deploy marker at a position the data does not cover, an
empty panel under a confident title, a trend drawn from one point.

The renderer is tested too, and not for cosmetics. A chart nobody can see is a chart nobody
checks — and the first version of the renderer drew the lowest value in a chart as blank,
which is what a *gap* draws as. A flat series at the chart minimum appeared as no data at
all. Rendering is part of the honesty surface, so it gets tests.
"""

from __future__ import annotations

import uuid

from cortex.reports.charts import render_chart, validate_chart
from cortex.reports.schema import (
    ChartAnnotation,
    ChartPoint,
    ChartSeries,
    ChartSpec,
    ChartType,
)

EVIDENCE = uuid.uuid4()


def _chart(
    *,
    points: list[tuple[str, float | None]] | None = None,
    series: list[ChartSeries] | None = None,
    annotations: list[ChartAnnotation] | None = None,
    chart_type: ChartType = ChartType.LINE,
    title: str = "Daily sessions",
) -> ChartSpec:
    built = series or [
        ChartSeries(
            name="mobile",
            points=[ChartPoint(x=x, y=y) for x, y in (points or [("07-01", 10.0), ("07-02", 8.0)])],
        )
    ]
    return ChartSpec(
        type=chart_type,
        title=title,
        y_label="sessions",
        series=built,
        annotations=annotations or [],
        evidence_ids=[EVIDENCE],
    )


class TestAnnotationsMustLandOnTheData:
    """An annotation is a causal statement, which is the whole reason charts are here."""

    def test_a_marker_on_the_axis_is_fine(self) -> None:
        chart = _chart(
            points=[("07-01", 10.0), ("07-02", 8.0)],
            annotations=[ChartAnnotation(x="07-02", label="deploy 91c3e4a", evidence_id=EVIDENCE)],
        )
        assert validate_chart(chart) == []

    def test_a_floating_marker_is_a_problem(self) -> None:
        """Rendered, it lands at an arbitrary position and shows the reader a coincidence
        that the data does not contain."""
        chart = _chart(
            points=[("07-01", 10.0), ("07-02", 8.0)],
            annotations=[ChartAnnotation(x="07-09", label="deploy 91c3e4a", evidence_id=EVIDENCE)],
        )
        problems = validate_chart(chart)
        assert len(problems) == 1
        assert "invents a coincidence" in problems[0].problem
        assert "07-09" in problems[0].problem

    def test_a_marker_may_sit_on_any_series(self) -> None:
        """Two series, one axis. A marker belonging to the second is still on the chart."""
        chart = _chart(
            series=[
                ChartSeries(name="mobile", points=[ChartPoint(x="07-01", y=1.0)]),
                ChartSeries(
                    name="desktop",
                    points=[ChartPoint(x="07-02", y=2.0), ChartPoint(x="07-03", y=3.0)],
                ),
            ],
            annotations=[ChartAnnotation(x="07-03", label="deploy", evidence_id=EVIDENCE)],
            chart_type=ChartType.BAR,
        )
        assert [p.problem for p in validate_chart(chart) if "coincidence" in p.problem] == []


class TestASeriesMustSaySomething:
    def test_a_series_of_only_gaps_is_a_problem(self) -> None:
        """One null point satisfies `min_length=1`. Drawn, it is an empty panel under a
        confident title — the same absence-looks-like-data failure as an empty tool
        result, arriving through the picture instead of the prose."""
        chart = _chart(points=[("07-01", None), ("07-02", None)])
        problems = validate_chart(chart)
        assert any("no values at all" in p.problem for p in problems)

    def test_a_single_point_line_is_a_problem(self) -> None:
        chart = _chart(points=[("07-01", 10.0)])
        problems = validate_chart(chart)
        assert any("one point" in p.problem for p in problems)

    def test_a_single_bar_is_fine(self) -> None:
        """A bar chart of one category is a legitimate picture; a line of one point is not."""
        chart = _chart(points=[("Paid Search", 10.0)], chart_type=ChartType.BAR)
        assert validate_chart(chart) == []

    def test_two_points_at_the_same_x_is_a_problem(self) -> None:
        """Which value gets drawn would depend on ordering, so the picture is not
        determined by the data."""
        chart = _chart(points=[("07-01", 10.0), ("07-01", 40.0), ("07-02", 8.0)])
        problems = validate_chart(chart)
        assert any("more than one point at" in p.problem for p in problems)

    def test_a_partial_series_is_allowed(self) -> None:
        """A gap in the middle is a real observation about the data, not a defect."""
        chart = _chart(points=[("07-01", 10.0), ("07-02", None), ("07-03", 9.0)])
        assert validate_chart(chart) == []


class TestRendering:
    def test_a_gap_and_the_minimum_look_different(self) -> None:
        """The bug this test exists for. The first renderer began its block characters with a
        space, so the lowest value drew as blank — and blank is a gap. A flat series at the
        chart minimum rendered as an empty panel."""
        chart = _chart(
            series=[
                ChartSeries(
                    name="mobile",
                    points=[ChartPoint(x="a", y=100.0), ChartPoint(x="b", y=None)],
                ),
                ChartSeries(
                    name="desktop",
                    points=[ChartPoint(x="a", y=10.0), ChartPoint(x="b", y=10.0)],
                ),
            ]
        )
        output = render_chart(chart)
        desktop = next(line for line in output.splitlines() if "desktop" in line)
        mobile = next(line for line in output.splitlines() if "mobile" in line)

        bars = desktop.split("|")[1]
        assert bars.strip() != "", "a real value must never render as a gap"
        # And the mobile row's null must render as a gap.
        assert " " in mobile.split("|")[1]

    def test_the_annotation_sits_under_its_column(self) -> None:
        chart = _chart(
            points=[("07-01", 10.0), ("07-02", 9.0), ("07-03", 3.0)],
            annotations=[ChartAnnotation(x="07-03", label="deploy", evidence_id=EVIDENCE)],
        )
        lines = render_chart(chart).splitlines()
        marker_row = next(line for line in lines if "^" in line and "|" in line)
        assert marker_row.split("|")[1].index("^") == 2

    def test_a_flat_series_does_not_draw_at_the_floor(self) -> None:
        """Flat is a finding — it is how the campaign scenario is distinguished from a funnel
        regression. Drawing it at zero would read as collapsed."""
        from cortex.reports.charts import _BLOCKS, _GAP

        chart = _chart(points=[("a", 50.0), ("b", 50.0), ("c", 50.0)])
        bars = render_chart(chart).splitlines()[2].split("|")[1]

        # Asserted as a property rather than an exact glyph: what matters is that a flat
        # series draws flat, somewhere in the middle, and never at the floor or as a gap.
        assert len(set(bars)) == 1, bars
        drawn = bars[0]
        assert drawn not in (_GAP, _BLOCKS[0], _BLOCKS[-1]), bars
        assert _BLOCKS.index(drawn) == len(_BLOCKS) // 2, bars

    def test_the_axis_keeps_the_datas_own_order(self) -> None:
        """An x may be a category, and alphabetising would silently reorder a funnel."""
        chart = _chart(
            points=[("Paid Search", 3.0), ("Organic", 8.0), ("Direct", 5.0)],
            chart_type=ChartType.BAR,
        )
        assert "Paid Search … Direct" in render_chart(chart)

    def test_the_value_range_is_printed(self) -> None:
        """The blocks show shape; the numbers are what a claim can be checked against."""
        chart = _chart(points=[("a", 1930.0), ("b", 2450.0)])
        assert "range 1,930–2,450" in render_chart(chart)

    def test_a_series_with_no_values_says_so_rather_than_drawing_zero(self) -> None:
        chart = _chart(points=[("a", None), ("b", None)])
        assert "no data" in render_chart(chart)


# --------------------------------------------------------------- derived from evidence


class _Row:
    """The parts of an `Evidence` row the extractor reads."""

    def __init__(self, payload: dict) -> None:
        self.id = uuid.uuid4()
        self.payload = payload


def _ga4_series(days: int = 14, drop_from: int | None = None) -> _Row:
    values = [2400 + (index % 3) * 20 for index in range(days)]
    if drop_from is not None:
        values = [v if i < drop_from else 1930 for i, v in enumerate(values)]
    return _Row(
        {
            "rows": [
                {
                    "dimensions": {"date": f"2026-07-{index + 1:02d}"},
                    "metrics": {"sessions": value},
                }
                for index, value in enumerate(values)
            ]
        }
    )


class TestChartsAreDerivedNotAuthored:
    """The same rule the Sources section follows, for the same reason: a model asked to
    write a series inline can write a number nobody observed, and a chart is the part of a
    report a reader trusts without checking the citation."""

    def test_a_time_series_becomes_a_chart(self) -> None:
        from cortex.reports.charts import charts_from_evidence

        row = _ga4_series()
        charts = charts_from_evidence([row])

        assert len(charts) == 1
        assert charts[0].series[0].name == "sessions"
        assert len(charts[0].series[0].points) == 14
        # Cites the row it was built from, so the chart is as checkable as a sentence.
        assert charts[0].evidence_ids == [row.id]

    def test_a_deploy_lands_on_the_axis_as_an_annotation(self) -> None:
        """This is what makes a chart causal rather than decorative."""
        from cortex.reports.charts import charts_from_evidence

        deploys = _Row(
            {
                "deployments": [
                    {"sha": "91c3e4a7bd2f", "created_at": "2026-07-13T11:04:00Z"},
                ]
            }
        )
        charts = charts_from_evidence([_ga4_series(drop_from=12), deploys])

        assert charts[0].annotations
        annotation = charts[0].annotations[0]
        assert annotation.x == "2026-07-13"
        assert "91c3e4a" in annotation.label
        # And it cites the evidence that establishes the deploy, not the metric row.
        assert annotation.evidence_id == deploys.id

    def test_a_change_outside_the_series_window_is_not_marked(self) -> None:
        """A marker at a position the data does not cover invents the coincidence."""
        from cortex.reports.charts import charts_from_evidence

        deploys = _Row({"deployments": [{"sha": "aaa", "created_at": "2026-09-01T00:00:00Z"}]})
        charts = charts_from_evidence([_ga4_series(), deploys])
        assert charts[0].annotations == []

    def test_a_posthog_trend_is_titled_with_its_event(self) -> None:
        from cortex.reports.charts import charts_from_evidence

        row = _Row(
            {
                "event": "user signed up",
                "series": [
                    {"bucket": f"2026-07-{index + 1:02d}T00:00:00", "value": 40 + index}
                    for index in range(5)
                ],
            }
        )
        charts = charts_from_evidence([row])
        assert "user signed up" in charts[0].title

    def test_a_breakdown_becomes_one_series_per_segment(self) -> None:
        from cortex.reports.charts import charts_from_evidence

        row = _Row(
            {
                "event": "onboarding completed",
                "series": [
                    {"bucket": f"2026-07-{d:02d}T00:00:00", "segment": segment, "value": value}
                    for d in range(1, 5)
                    for segment, value in (("Mobile", 60), ("Desktop", 24))
                ],
            }
        )
        charts = charts_from_evidence([row])
        names = {series.name for chart in charts for series in chart.series}
        assert any("Mobile" in name for name in names)

    def test_a_short_series_is_not_charted(self) -> None:
        """Two points make a line but no shape worth drawing; the title would be doing all
        the work."""
        from cortex.reports.charts import charts_from_evidence

        row = _Row(
            {
                "rows": [
                    {"dimensions": {"date": "2026-07-01"}, "metrics": {"sessions": 10}},
                    {"dimensions": {"date": "2026-07-02"}, "metrics": {"sessions": 12}},
                ]
            }
        )
        assert charts_from_evidence([row]) == []

    def test_a_category_axis_is_not_mistaken_for_a_time_series(self) -> None:
        """`dimensions` can hold a device or a channel name. A permissive date parser would
        happily draw a category on a time axis."""
        from cortex.reports.charts import charts_from_evidence

        row = _Row(
            {
                "rows": [
                    {"dimensions": {"deviceCategory": "mobile"}, "metrics": {"sessions": 9800}},
                    {"dimensions": {"deviceCategory": "desktop"}, "metrics": {"sessions": 4100}},
                    {"dimensions": {"deviceCategory": "tablet"}, "metrics": {"sessions": 300}},
                ]
            }
        )
        assert charts_from_evidence([row]) == []

    def test_bookkeeping_fields_are_not_charted(self) -> None:
        """A generic "find the numbers" extractor was the first attempt, and it drew
        plausible-looking lines out of page counts and result totals."""
        from cortex.reports.charts import charts_from_evidence

        row = _Row({"total_matching": 40, "count": 0, "messages": []})
        assert charts_from_evidence([row]) == []

    def test_the_number_of_charts_is_capped(self) -> None:
        """A report is read, not browsed. A page of panels gets skimmed rather than
        checked, which defeats the point of drawing the data."""
        from cortex.reports.charts import charts_from_evidence

        rows = [_ga4_series() for _ in range(6)]
        assert len(charts_from_evidence(rows)) <= 3

    def test_every_derived_chart_passes_its_own_validation(self) -> None:
        """The extractor is code and can be wrong. A chart that fails its own checks must
        never reach the gate, which would record a rejection against the analyst for
        something the extractor did."""
        from cortex.reports.charts import charts_from_evidence

        deploys = _Row({"deployments": [{"sha": "aaa", "created_at": "2026-07-05T00:00:00Z"}]})
        for chart in charts_from_evidence([_ga4_series(), deploys]):
            assert validate_chart(chart) == []

    def test_the_legend_is_numbered_in_axis_order(self) -> None:
        """Unsorted, a chart with two deploys printed "^1" under the later one and "^2"
        under the earlier, so the legend read right-to-left against the picture."""
        chart = _chart(
            points=[("07-01", 10.0), ("07-02", 9.0), ("07-03", 3.0)],
            annotations=[
                ChartAnnotation(x="07-03", label="later deploy", evidence_id=EVIDENCE),
                ChartAnnotation(x="07-02", label="earlier deploy", evidence_id=EVIDENCE),
            ],
        )
        lines = render_chart(chart).splitlines()
        legend = [line.strip() for line in lines if line.strip().startswith("^")]
        assert legend[0].startswith("^1 earlier deploy")
        assert legend[1].startswith("^2 later deploy")


class TestTheAxisLabelIsDroppedWhenItRepeatsTheSeries:
    """A PostHog trend rendered "value" above a row also labelled "value".

    A single-series trend takes its name from the payload's value column, and the y-axis
    label is derived from the same place, so the chart printed the same word twice. It reads
    as an unfinished interface rather than an axis, and it was visible in a published
    recording before anyone noticed.
    """

    def _spec(self, y_label: str, series_name: str) -> ChartSpec:
        points = [ChartPoint(x=f"2026-06-0{n}", y=float(n * 10)) for n in range(1, 6)]
        return ChartSpec(
            title="user signed up over time",
            type="line",
            y_label=y_label,
            evidence_ids=[uuid.uuid4()],
            series=[ChartSeries(name=series_name, points=points)],
        )

    def test_a_repeated_label_is_printed_once(self) -> None:
        rendered = render_chart(self._spec("value", "value"))
        assert rendered.count("value") == 1

    def test_a_label_that_says_something_else_survives(self) -> None:
        rendered = render_chart(self._spec("sessions", "mobile"))
        assert "sessions" in rendered
        assert "mobile" in rendered

    def test_the_title_is_never_dropped(self) -> None:
        assert "user signed up over time" in render_chart(self._spec("value", "value"))


class TestTheRangeIsNotWrittenAsAMovement:
    """`459 → 1,110` was the series minimum and maximum, not its start and end.

    An arrow between two numbers is read as a change over the period. So a report whose
    conclusion was that signups had *not* fallen -- flat at about 157 a day -- rendered a
    chart directly beneath it that appeared to say they had more than doubled. Both numbers
    were correct. The notation was the whole error, and it contradicted the answer on the
    same screen.
    """

    def _chart(self, values: list[float]) -> ChartSpec:
        points = [ChartPoint(x=f"2026-06-{n:02d}", y=v) for n, v in enumerate(values, start=1)]
        return ChartSpec(
            title="user signed up over time",
            type="line",
            y_label="signups",
            evidence_ids=[uuid.uuid4()],
            series=[ChartSeries(name="value", points=points)],
        )

    def test_it_is_labelled_a_range(self) -> None:
        rendered = render_chart(self._chart([459, 900, 1110, 300]))
        assert "range 300–1,110" in rendered

    def test_no_arrow_suggests_a_movement(self) -> None:
        assert "→" not in render_chart(self._chart([459, 900, 1110, 300]))

    def test_a_flat_series_does_not_look_like_a_climb(self) -> None:
        # The case that produced the contradiction: values within a few percent of each
        # other must not render as one number arrowing into a larger one.
        rendered = render_chart(self._chart([156, 157, 155, 158]))
        assert "range 155–158" in rendered
