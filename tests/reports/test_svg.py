"""A chart spec as SVG.

The tests are the same three questions `test_charts.py` asks of the ASCII renderer, because
they are properties of the *picture* rather than of either renderer: does a gap read as a gap,
does an annotation land where it says, and can a title become markup.

The first is the one that has already gone wrong once. The ASCII renderer's block ramp began
with a space, so the lowest value in a series drew as blank — and blank is what a gap draws as,
so a flat series at its minimum appeared as no data at all. The equivalent mistake here is
joining a polyline across a `None`, which invents a trend through data nobody has.
"""

from __future__ import annotations

import re
import uuid

import pytest

from cortex.reports.schema import (
    ChartAnnotation,
    ChartPoint,
    ChartSeries,
    ChartSpec,
    ChartType,
)
from cortex.reports.svg import render_svg

EVIDENCE_ID = uuid.uuid4()


def _chart(
    points: list[tuple[str, float | None]],
    *,
    kind: ChartType = ChartType.LINE,
    title: str = "GA4 sessions",
    annotations: list[ChartAnnotation] | None = None,
    extra: list[tuple[str, list[tuple[str, float | None]]]] | None = None,
) -> ChartSpec:
    series = [ChartSeries(name="sessions", points=[ChartPoint(x=x, y=y) for x, y in points])]
    for name, more in extra or []:
        series.append(ChartSeries(name=name, points=[ChartPoint(x=x, y=y) for x, y in more]))
    return ChartSpec(
        type=kind,
        title=title,
        y_label="sessions",
        series=series,
        annotations=annotations or [],
        evidence_ids=[EVIDENCE_ID],
    )


class TestGapsStayGaps:
    def test_a_missing_value_breaks_the_line(self) -> None:
        """A None y is a gap. Drawing through it asserts a value nobody observed — and it is
        the *shape* of the line that a reader takes the trend from, not the points.

        Two points either side of the gap, not one: a single-point run is drawn as a dot,
        which is correct behaviour and would make this assertion count zero polylines while
        proving nothing about joining."""
        svg = render_svg(
            _chart([("d1", 10.0), ("d2", 12.0), ("d3", None), ("d4", 30.0), ("d5", 31.0)]),
        )
        assert svg.count("<polyline") == 2

    def test_an_unbroken_series_is_one_line(self) -> None:
        svg = render_svg(_chart([("d1", 10.0), ("d2", 20.0), ("d3", 30.0)]))
        assert svg.count("<polyline") == 1

    def test_a_single_point_is_drawn_as_a_dot(self) -> None:
        """A one-point polyline renders as nothing at all, which turns a present observation
        into apparently missing data."""
        svg = render_svg(_chart([("d1", 10.0)]))
        assert "<circle" in svg

    def test_a_series_of_only_gaps_draws_no_line(self) -> None:
        svg = render_svg(_chart([("d1", None), ("d2", None)]))
        assert "<polyline" not in svg
        assert "<circle" not in svg


class TestAnnotationsLandWhereTheySay:
    def test_a_marker_is_drawn_at_its_own_x(self) -> None:
        chart = _chart(
            [("d1", 10.0), ("d2", 20.0), ("d3", 30.0)],
            annotations=[ChartAnnotation(x="d2", label="deploy 91c3e4a", evidence_id=EVIDENCE_ID)],
        )
        svg = render_svg(chart)
        assert "deploy 91c3e4a" in svg

        # The dashed line's x must be the middle position, not an edge. Reading it back from
        # the output rather than trusting the arithmetic: a marker at the wrong x points at a
        # day the data does not cover, which is worse than no marker at all.
        dashed = re.search(r'<line x1="([\d.]+)"[^>]*stroke-dasharray', svg)
        assert dashed is not None
        x = float(dashed.group(1))
        assert 300 < x < 420, x

    def test_a_marker_outside_the_axis_is_not_drawn(self) -> None:
        """`validate_chart` already rejects a floating annotation, so this is the second line
        of defence. Drawn at an edge it would silently claim a date the chart does not show."""
        chart = _chart(
            [("d1", 10.0), ("d2", 20.0)],
            annotations=[
                ChartAnnotation(x="d9", label="deploy elsewhere", evidence_id=EVIDENCE_ID)
            ],
        )
        assert "deploy elsewhere" not in render_svg(chart)


class TestItCannotBecomeMarkup:
    @pytest.mark.parametrize(
        "hostile",
        [
            "</text><script>alert(1)</script>",
            'sessions" onload="alert(1)',
            "<img src=x onerror=alert(1)>",
        ],
    )
    def test_a_hostile_title_is_escaped(self, hostile: str) -> None:
        """A chart title is model output. It reaches both an attribute and an element, so
        both quoting and angle brackets matter."""
        svg = render_svg(_chart([("d1", 1.0)], title=hostile))
        # What matters is that no *tag* or attribute boundary survives. The words themselves
        # may appear -- they are the title, and a title reading "<img onerror=...>" as visible
        # text is the correct outcome, while asserting their absence would only be asserting
        # that the payload was mangled.
        assert "<script" not in svg
        assert "<img" not in svg
        # No new attribute can be opened: the quote that would close the aria-label is gone.
        aria = svg.split('aria-label="', 1)[1].split('"', 1)[0]
        assert "onload" not in aria or "&quot;" in svg
        assert "&lt;" in svg or "&quot;" in svg

    def test_a_hostile_series_name_is_escaped(self) -> None:
        chart = _chart(
            [("d1", 1.0)],
            extra=[("</text><script>x</script>", [("d1", 2.0)])],
        )
        assert "<script" not in render_svg(chart)

    def test_a_hostile_x_label_is_escaped(self) -> None:
        assert "<script" not in render_svg(_chart([("<script>x</script>", 1.0)]))


class TestTheAxes:
    def test_a_bar_chart_includes_zero(self) -> None:
        """A bar chart with a truncated baseline exaggerates every difference in it, which is
        the most common way an honest number becomes a misleading picture."""
        svg = render_svg(_chart([("d1", 100.0), ("d2", 104.0)], kind=ChartType.BAR))
        ticks = re.findall(r'chart-tick-y">([\d.,-]+)<', svg)
        assert "0" in ticks

    def test_a_line_chart_does_not_flatten_a_real_movement(self) -> None:
        """The converse: forcing zero onto a line chart of 1000-1100 draws a flat line and
        hides the movement the chart exists to show."""
        svg = render_svg(_chart([("d1", 1000.0), ("d2", 1100.0)]))
        ticks = re.findall(r'chart-tick-y">([\d.,-]+)<', svg)
        assert "0" not in ticks

    def test_a_flat_series_is_not_drawn_along_an_edge(self) -> None:
        """A zero-height range would put the line on the axis, where it reads as the axis."""
        svg = render_svg(_chart([("d1", 50.0), ("d2", 50.0)]))
        ys = {y for y in re.findall(r'<polyline points="[\d.]+,([\d.]+)', svg)}
        assert ys
        assert all(30 < float(y) < 250 for y in ys)

    def test_x_labels_are_thinned_rather_than_crowded(self) -> None:
        """Forty labels on a 720-unit axis is an unreadable smear. The first and last are
        always kept, because they are the range."""
        points = [(f"2026-07-{day:02d}", float(day)) for day in range(1, 31)]
        svg = render_svg(_chart(points))
        labels = re.findall(r'chart-tick-x">([^<]+)<', svg)
        assert len(labels) < len(points)
        assert labels[0] == "2026-07-01"
        assert labels[-1] == "2026-07-30"


class TestItNeverRaises:
    def test_a_legend_appears_only_with_more_than_one_series(self) -> None:
        one = render_svg(_chart([("d1", 1.0)]))
        two = render_svg(_chart([("d1", 1.0)], extra=[("signups", [("d1", 2.0)])]))
        assert "chart-legend" not in one
        assert "chart-legend" in two

    @pytest.mark.parametrize("kind", list(ChartType))
    def test_every_chart_type_renders(self, kind: ChartType) -> None:
        """A renderer that raises takes the report page down at the last step, after
        everything expensive has already succeeded."""
        svg = render_svg(_chart([("d1", 1.0), ("d2", None), ("d3", 3.0)], kind=kind))
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")

    def test_a_negative_series_renders(self) -> None:
        svg = render_svg(_chart([("d1", -10.0), ("d2", -30.0)], kind=ChartType.BAR))
        assert "<rect" in svg
