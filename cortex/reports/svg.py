"""A `ChartSpec` as inline SVG.

The plan's M5 said chart specs are "rendered by the frontend", and there was no frontend, so
the only renderer was the terminal's ASCII one — useful for checking a chart by eye during
development and not something to show anyone. This is the second renderer over the same
validated spec.

**Inline SVG, no chart library.** Not a preference for hand-rolling: a self-contained `<svg>`
element needs no build step, no JavaScript, no CDN, and no second deploy target, and it renders
in an email or a saved page. The specs are small — a handful of series over a few dozen points
— which is the size at which a charting library costs more than it saves.

**The honesty rules from `charts.py` carry over, because they are properties of the picture
rather than of the data.** A `None` y-value is a *gap* and must be drawn as a break in the
line: joining across it invents a trend through data nobody has. A deploy annotation must land
at the x position it names. Both are the difference between a chart that shows a causal story
and a chart that asserts one.

Everything that reaches the output is escaped. A chart title is model output, and a title
containing `</text><script>` would otherwise be markup rather than a title.
"""

from __future__ import annotations

from html import escape

from cortex.reports.schema import ChartSeries, ChartSpec, ChartType

#: Canvas, in user units. Rendered with a viewBox so the element scales to its container
#: rather than to these numbers.
_WIDTH = 720
_HEIGHT = 300
_PAD_LEFT = 56
_PAD_RIGHT = 16
_PAD_TOP = 28
_PAD_BOTTOM = 46

#: Series colours, in order. Chosen to stay distinguishable in greyscale and to the most
#: common forms of colour blindness, because a chart whose two lines are indistinguishable to
#: the reader has no legend at all.
_COLOURS = ("#2563eb", "#d97706", "#059669", "#9333ea", "#dc2626")

#: Annotations are drawn in one colour, deliberately not a series colour: an annotation is a
#: different kind of statement from a data point and should not read as one.
_ANNOTATION = "#6b7280"

#: Horizontal gridlines. Enough to read a value off, few enough not to become the picture.
_GRID_LINES = 4


def render_svg(chart: ChartSpec) -> str:
    """One chart as a self-contained `<svg>` element.

    Never raises on a chart that validated. A renderer that can fail is a renderer that takes
    a report down at the last step, after everything expensive has already succeeded.
    """
    xs = _x_labels(chart)
    if not xs:
        # Cannot happen for a validated spec (every series carries at least one point), and
        # returning an empty panel beats dividing by zero if it ever does.
        return _empty_panel(chart.title)

    lo, hi = _y_range(chart)
    body = [
        _grid(lo, hi, chart.y_label),
        _x_axis(xs),
    ]
    for index, series in enumerate(chart.series):
        colour = _COLOURS[index % len(_COLOURS)]
        body.append(_series(series, xs, lo, hi, colour, chart.type))
    for annotation in chart.annotations:
        body.append(_annotation(annotation.x, annotation.label, xs))
    body.append(_legend(chart))

    return (
        f'<svg viewBox="0 0 {_WIDTH} {_HEIGHT}" role="img" '
        f'aria-label="{escape(chart.title)}" class="chart">'
        f"<title>{escape(chart.title)}</title>"
        f'<text x="{_PAD_LEFT}" y="18" class="chart-title">{escape(chart.title)}</text>'
        + "".join(body)
        + "</svg>"
    )


def _empty_panel(title: str) -> str:
    return (
        f'<svg viewBox="0 0 {_WIDTH} 80" role="img" aria-label="{escape(title)}" class="chart">'
        f'<text x="{_PAD_LEFT}" y="40" class="chart-empty">'
        f"{escape(title)} — no points to draw</text></svg>"
    )


def _x_labels(chart: ChartSpec) -> list[str]:
    """Every x value, in the order the series present them.

    Ordered by first appearance rather than sorted. The x axis carries dates as strings, and
    sorting them lexically is right for ISO dates and wrong for everything else — while the
    series order is what the evidence produced.
    """
    seen: list[str] = []
    known: set[str] = set()
    for series in chart.series:
        for point in series.points:
            if point.x not in known:
                known.add(point.x)
                seen.append(point.x)
    return seen


def _y_range(chart: ChartSpec) -> tuple[float, float]:
    """The value range to draw, always including zero for bars.

    A bar chart with a non-zero baseline exaggerates every difference in it, which is the
    single most common way an honest number becomes a misleading picture. Lines keep a fitted
    range, where a baseline of zero would flatten a real movement into a straight line.
    """
    values = [point.y for series in chart.series for point in series.points if point.y is not None]
    if not values:
        return 0.0, 1.0

    lo, hi = min(values), max(values)
    if chart.type in (ChartType.BAR, ChartType.STACKED_BAR, ChartType.AREA):
        lo = min(lo, 0.0)
    if hi == lo:
        # A flat series. Padding rather than a zero-height range, so the line is drawn
        # mid-panel instead of along an edge where it reads as an axis.
        hi = lo + (abs(lo) or 1.0) * 0.1
        lo = lo - (abs(lo) or 1.0) * 0.1
    return lo, hi


def _x_at(index: int, count: int) -> float:
    inner = _WIDTH - _PAD_LEFT - _PAD_RIGHT
    if count == 1:
        return _PAD_LEFT + inner / 2
    return _PAD_LEFT + inner * index / (count - 1)


def _y_at(value: float, lo: float, hi: float) -> float:
    inner = _HEIGHT - _PAD_TOP - _PAD_BOTTOM
    share = (value - lo) / (hi - lo) if hi > lo else 0.5
    # Inverted: SVG's y grows downward, and a chart drawn upside down is a chart that says
    # the opposite of the data.
    return _PAD_TOP + inner * (1.0 - share)


def _grid(lo: float, hi: float, y_label: str) -> str:
    parts = []
    for step in range(_GRID_LINES + 1):
        value = lo + (hi - lo) * step / _GRID_LINES
        y = _y_at(value, lo, hi)
        parts.append(
            f'<line x1="{_PAD_LEFT}" y1="{y:.1f}" x2="{_WIDTH - _PAD_RIGHT}" y2="{y:.1f}" '
            f'class="chart-grid"/>'
            f'<text x="{_PAD_LEFT - 8}" y="{y + 4:.1f}" class="chart-tick chart-tick-y">'
            f"{_number(value)}</text>"
        )
    if y_label:
        parts.append(
            f'<text x="4" y="{_PAD_TOP - 12}" class="chart-axis-label">{escape(y_label)}</text>'
        )
    return "".join(parts)


def _x_axis(xs: list[str]) -> str:
    """X labels, thinned so they do not overlap.

    Thinning rather than rotating: a chart with every label drawn is unreadable, and rotated
    text is worse. The first and last are always kept, because they are the range.
    """
    keep = max(1, len(xs) // 8)
    parts = []
    for index, label in enumerate(xs):
        if index % keep and index != len(xs) - 1:
            continue
        x = _x_at(index, len(xs))
        parts.append(
            f'<text x="{x:.1f}" y="{_HEIGHT - _PAD_BOTTOM + 18:.1f}" '
            f'class="chart-tick chart-tick-x">{escape(label)}</text>'
        )
    return "".join(parts)


def _series(
    series: ChartSeries,
    xs: list[str],
    lo: float,
    hi: float,
    colour: str,
    kind: ChartType,
) -> str:
    positions = {label: index for index, label in enumerate(xs)}
    if kind in (ChartType.BAR, ChartType.STACKED_BAR):
        return _bars(series, positions, len(xs), lo, hi, colour)
    if kind is ChartType.SCATTER:
        return _dots(series, positions, len(xs), lo, hi, colour)
    return _line(series, positions, len(xs), lo, hi, colour, filled=kind is ChartType.AREA)


def _segments(
    series: ChartSeries, positions: dict[str, int], count: int, lo: float, hi: float
) -> list[list[tuple[float, float]]]:
    """Contiguous runs of present values.

    The load-bearing function. A `None` y is a gap, and a line drawn across a gap invents a
    trend through data nobody has — the same failure the ASCII renderer had when it drew the
    lowest value as blank, where blank means gap. Splitting into segments makes the break
    visible instead.
    """
    runs: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for point in series.points:
        if point.y is None or point.x not in positions:
            if current:
                runs.append(current)
                current = []
            continue
        current.append((_x_at(positions[point.x], count), _y_at(point.y, lo, hi)))
    if current:
        runs.append(current)
    return runs


def _line(
    series: ChartSeries,
    positions: dict[str, int],
    count: int,
    lo: float,
    hi: float,
    colour: str,
    *,
    filled: bool,
) -> str:
    parts = []
    baseline = _y_at(max(lo, 0.0), lo, hi)
    for run in _segments(series, positions, count, lo, hi):
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in run)
        if filled and len(run) > 1:
            area = f"{run[0][0]:.1f},{baseline:.1f} {points} {run[-1][0]:.1f},{baseline:.1f}"
            parts.append(f'<polygon points="{area}" fill="{colour}" fill-opacity="0.12"/>')
        if len(run) == 1:
            # A single point is drawn as a dot. A one-point polyline renders as nothing at
            # all, which would make a present observation look like missing data.
            x, y = run[0]
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{colour}"/>')
        else:
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="{colour}" '
                f'stroke-width="2" stroke-linejoin="round"/>'
            )
    return "".join(parts)


def _dots(
    series: ChartSeries, positions: dict[str, int], count: int, lo: float, hi: float, colour: str
) -> str:
    return "".join(
        f'<circle cx="{_x_at(positions[point.x], count):.1f}" '
        f'cy="{_y_at(point.y, lo, hi):.1f}" r="3.5" fill="{colour}" fill-opacity="0.8"/>'
        for point in series.points
        if point.y is not None and point.x in positions
    )


def _bars(
    series: ChartSeries, positions: dict[str, int], count: int, lo: float, hi: float, colour: str
) -> str:
    inner = _WIDTH - _PAD_LEFT - _PAD_RIGHT
    width = max(2.0, min(28.0, inner / max(count, 1) * 0.6))
    baseline = _y_at(max(lo, 0.0), lo, hi)
    parts = []
    for point in series.points:
        if point.y is None or point.x not in positions:
            continue
        x = _x_at(positions[point.x], count)
        y = _y_at(point.y, lo, hi)
        top, height = min(y, baseline), abs(baseline - y)
        parts.append(
            f'<rect x="{x - width / 2:.1f}" y="{top:.1f}" width="{width:.1f}" '
            f'height="{max(height, 1.0):.1f}" fill="{colour}" fill-opacity="0.85"/>'
        )
    return "".join(parts)


def _annotation(x_value: str, label: str, xs: list[str]) -> str:
    """A dated marker — a deploy, a release, a campaign start.

    Drawn only where the axis actually covers the date. `validate_chart` already rejects a
    floating annotation, so this is the second line of defence rather than the first: a marker
    silently drawn at the edge would point at a day the data does not include, which is worse
    than no marker.
    """
    if x_value not in xs:
        return ""
    x = _x_at(xs.index(x_value), len(xs))
    return (
        f'<line x1="{x:.1f}" y1="{_PAD_TOP}" x2="{x:.1f}" y2="{_HEIGHT - _PAD_BOTTOM}" '
        f'stroke="{_ANNOTATION}" stroke-width="1" stroke-dasharray="4 3"/>'
        f'<text x="{x + 4:.1f}" y="{_PAD_TOP + 10}" class="chart-annotation">'
        f"{escape(label)}</text>"
    )


def _legend(chart: ChartSpec) -> str:
    """Series names, in the order they were drawn.

    Omitted for a single series: its name is the chart's subject and the title already carries
    it, so a one-entry legend is furniture.
    """
    if len(chart.series) < 2:
        return ""
    parts = []
    x = _PAD_LEFT
    y = _HEIGHT - 10
    for index, series in enumerate(chart.series):
        colour = _COLOURS[index % len(_COLOURS)]
        parts.append(
            f'<rect x="{x}" y="{y - 8}" width="9" height="9" fill="{colour}"/>'
            f'<text x="{x + 14}" y="{y}" class="chart-legend">{escape(series.name)}</text>'
        )
        # Advanced by the label's rendered width, approximated from its length. Exact text
        # metrics need a font engine; overlapping legend entries need only an estimate.
        x += 30 + int(len(series.name) * 6.2)
    return "".join(parts)


def _number(value: float) -> str:
    """A value a reader can scan.

    Thousands separated and decimals dropped above 100: an axis reading 12,431.6 costs three
    characters to say nothing, and the point of a tick is to be read at a glance.
    """
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:,.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".") or "0"
