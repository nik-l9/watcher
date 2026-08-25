"""A chart spec as PNG.

The third renderer over one validated spec. The tests are the same properties `test_svg.py`
asserts, because they are properties of the *picture* rather than of any renderer — a gap must
read as a gap, an annotation must land where it says, and nothing may raise.

`render_png` returning bytes on every path is load-bearing rather than defensive: a caller
posting to Slack has already committed to sending something, and a missing attachment is harder
to diagnose than a picture that explains itself.
"""

from __future__ import annotations

import struct
import uuid

import pytest

from cortex.reports.png import render_png
from cortex.reports.schema import (
    ChartAnnotation,
    ChartPoint,
    ChartSeries,
    ChartSpec,
    ChartType,
)

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
        x_label="day",
        y_label="sessions",
        series=series,
        annotations=annotations or [],
        evidence_ids=[EVIDENCE_ID],
    )


def _dimensions(png: bytes) -> tuple[int, int]:
    """Width and height from the IHDR chunk, which starts at byte 16."""
    width, height = struct.unpack(">II", png[16:24])
    return width, height


class TestItProducesAValidImage:
    def test_the_bytes_are_a_png(self) -> None:
        png = render_png(_chart([("d1", 10.0), ("d2", 20.0)]))
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_it_is_large_enough_to_read_in_a_message(self) -> None:
        """Sized for Slack: a chart nobody can read without clicking is a chart nobody reads."""
        width, height = _dimensions(render_png(_chart([("d1", 1.0), ("d2", 2.0)])))
        assert width >= 800
        assert height >= 300

    @pytest.mark.parametrize("kind", list(ChartType))
    def test_every_chart_type_renders(self, kind: ChartType) -> None:
        """Including a gap, because a renderer that raises takes the report down at the last
        step, after everything expensive has already succeeded."""
        png = render_png(_chart([("d1", 1.0), ("d2", None), ("d3", 3.0)], kind=kind))
        assert png[:4] == b"\x89PNG"

    def test_a_single_point_renders(self) -> None:
        assert render_png(_chart([("d1", 10.0)]))[:4] == b"\x89PNG"

    def test_a_series_of_only_gaps_renders(self) -> None:
        assert render_png(_chart([("d1", None), ("d2", None)]))[:4] == b"\x89PNG"

    def test_a_negative_series_renders(self) -> None:
        png = render_png(_chart([("d1", -10.0), ("d2", -30.0)], kind=ChartType.BAR))
        assert png[:4] == b"\x89PNG"

    def test_many_points_render(self) -> None:
        """Thirty dates on one axis. The tick thinning is what keeps this legible; without it
        the labels overlap into a smear."""
        points = [(f"2026-07-{day:02d}", float(day)) for day in range(1, 31)]
        assert render_png(_chart(points))[:4] == b"\x89PNG"

    def test_a_hostile_title_does_not_break_it(self) -> None:
        """A chart title is model output. matplotlib interprets `$...$` as mathtext and `\\` as
        an escape, so a title is a rendering hazard even though it cannot be markup here."""
        for hostile in ("$\\frac{1}{0}$", "100% \\up {unclosed", "</text><script>x"):
            assert render_png(_chart([("d1", 1.0)], title=hostile))[:4] == b"\x89PNG"


class TestGapsAndAnnotations:
    def test_a_gap_changes_the_picture(self) -> None:
        """The property, asserted the only way a raster image allows: the same series with and
        without a gap must not produce identical bytes. If the gap were filled or the point
        silently dropped, the two would match."""
        unbroken = render_png(_chart([("d1", 10.0), ("d2", 20.0), ("d3", 30.0)]))
        gapped = render_png(_chart([("d1", 10.0), ("d2", None), ("d3", 30.0)]))
        assert unbroken != gapped

    def test_an_annotation_changes_the_picture(self) -> None:
        without = render_png(_chart([("d1", 10.0), ("d2", 20.0)]))
        with_marker = render_png(
            _chart(
                [("d1", 10.0), ("d2", 20.0)],
                annotations=[
                    ChartAnnotation(x="d2", label="deploy 91c3e4a", evidence_id=EVIDENCE_ID)
                ],
            )
        )
        assert without != with_marker

    def test_an_annotation_outside_the_axis_is_not_drawn(self) -> None:
        """`validate_chart` already rejects a floating annotation, so this is the second line of
        defence: drawn at an edge it would claim a date the chart does not show. Asserted by the
        image being byte-identical to one with no annotation at all."""
        without = render_png(_chart([("d1", 10.0), ("d2", 20.0)]))
        floating = render_png(
            _chart(
                [("d1", 10.0), ("d2", 20.0)],
                annotations=[
                    ChartAnnotation(x="d9", label="deploy elsewhere", evidence_id=EVIDENCE_ID)
                ],
            )
        )
        assert without == floating


class TestItDoesNotLeak:
    def test_repeated_renders_do_not_accumulate_figures(self) -> None:
        """pyplot keeps a global registry of open figures, so a worker rendering one chart per
        investigation would leak memory until it was restarted."""
        import matplotlib.pyplot as plt

        for _ in range(25):
            render_png(_chart([("d1", 1.0), ("d2", 2.0)]))
        assert plt.get_fignums() == []
