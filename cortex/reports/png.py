"""A `ChartSpec` as a PNG.

The third renderer over the same validated spec, after the terminal's ASCII one and the report
view's inline SVG. It exists for the places SVG cannot go: Slack, email, and anything that
accepts an image and not markup — which is most of where a GTM team actually reads things.

## Why this is not the plan's escape hatch, and what that traded

The plan specified something different: *"agent writes matplotlib inside the Agent Server
sandbox, PNG to S3, still carrying evidence_ids"*, for charts the spec cannot express.

That version trades away the guarantee the product is built on. A `ChartSpec` is **checked**:
`charts_from_evidence` derives every series point from the payload of a cited evidence row, so a
model cannot plot a number nobody observed, and `validate_chart` rejects a deploy marker at a date
the data does not cover. A PNG produced by agent-authored code is **opaque** — the code can draw
any number it likes, the `evidence_ids` attached to it still resolve, and both the citation gate
and the adversarial verifier are blind to the pixels. The chart would be the one part of a report
nothing checks, in a product whose claim is that every part is checked.

So this renderer takes the same validated spec and draws it deterministically. Expressiveness is
bounded by `ChartType`, which is the point: a chart type we cannot check is a chart type we cannot
ship. Adding heatmaps or dual axes means extending the schema and the validator, which is more work
than letting a model write code and considerably more defensible.

If agent-authored charts are ever genuinely needed, the missing pieces are a sandbox with no
network and no credentials (`agent-server` is already in `docker-compose.yml` under the `sandbox`
profile, pinned, with no credentials passed), object storage for the output, and — the hard part —
an answer to "what does a citation mean on an image nobody can verify".

## Honesty rules, carried across from the SVG renderer

They are properties of the picture rather than of either renderer, so they are reimplemented here
rather than assumed:

  - A `None` y is a **gap** and must break the line. Joining across it invents a trend through
    data nobody has. matplotlib does this natively for NaN, which is why the values are converted
    rather than filtered.
  - An annotation is drawn **only where the axis covers its x**. `validate_chart` already refuses
    a floating annotation; this is the second line of defence, because a marker silently drawn at
    the edge would point at a day the data does not include.
  - A bar chart's baseline **includes zero**. A truncated baseline exaggerates every difference in
    it, which is the commonest way an honest number becomes a misleading picture. A line chart
    keeps a fitted range, where forcing zero would flatten a real movement.
"""

from __future__ import annotations

import io
import math

import matplotlib

# Set before pyplot is imported. Agg is a headless raster backend: the default would try to
# find a display, which a worker process does not have and must not need.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - must follow matplotlib.use

from cortex.reports.schema import ChartSpec, ChartType  # noqa: E402

#: Figure size in inches and dots per inch. 9.6 x 4.2 at 110 dpi is 1056 x 462 pixels — wide
#: enough for a date axis to stay legible in a Slack message without needing a click.
_FIGSIZE = (9.6, 4.2)
_DPI = 110

#: Series colours, matching `svg.py` so the same chart does not change identity between the report
#: page and a Slack post. Chosen to stay distinguishable in greyscale and to the commonest forms of
#: colour blindness.
_COLOURS = ("#2563eb", "#d97706", "#059669", "#9333ea", "#dc2626")

#: Annotations get one deliberately non-series colour: an annotation is a different kind of
#: statement from a data point and should not read as one.
_ANNOTATION = "#6b7280"

#: Above this many x values, only every nth tick is labelled. Forty dates on one axis is an
#: unreadable smear, and the first and last are what carry the range.
_MAX_TICKS = 12


def render_png(chart: ChartSpec) -> bytes:
    """One chart as PNG bytes.

    Never raises on a spec that validated. A renderer that can fail is a renderer that takes a
    report down at the last step, after everything expensive has already succeeded — so the
    figure is always closed and any failure returns a small placeholder rather than propagating.
    """
    figure = None
    try:
        figure, axes = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
        xs = _x_labels(chart)
        if not xs:
            # Cannot happen for a validated spec: every series carries at least one point.
            return _placeholder(chart.title)

        positions = {label: index for index, label in enumerate(xs)}
        _draw_series(axes, chart, xs, positions)
        _draw_annotations(axes, chart, positions)
        _style(axes, chart, xs)
        return _to_bytes(figure)
    except Exception:  # noqa: BLE001 - see the docstring
        return _placeholder(chart.title)
    finally:
        if figure is not None:
            # Closed explicitly rather than left to garbage collection. pyplot keeps a global
            # registry of open figures, so a worker rendering a chart per investigation would
            # leak memory until it was restarted.
            plt.close(figure)


def _x_labels(chart: ChartSpec) -> list[str]:
    """Every x value, in the order the series present them.

    By first appearance rather than sorted: the axis carries dates as strings, and sorting them
    lexically is right for ISO dates and wrong for everything else, while the series order is
    what the evidence produced.
    """
    seen: list[str] = []
    known: set[str] = set()
    for series in chart.series:
        for point in series.points:
            if point.x not in known:
                known.add(point.x)
                seen.append(point.x)
    return seen


def _draw_series(axes, chart: ChartSpec, xs: list[str], positions: dict[str, int]) -> None:
    bar_width = 0.8 / max(len(chart.series), 1)

    for index, series in enumerate(chart.series):
        colour = _COLOURS[index % len(_COLOURS)]
        # A dict rather than two parallel lists, so a series that skips an x value produces a
        # gap at that position instead of shifting every later point one place left.
        by_position = {
            positions[point.x]: point.y for point in series.points if point.x in positions
        }
        # NaN rather than a dropped point: matplotlib breaks a line at NaN, which is exactly the
        # "a gap is a gap" rule. Filtering the point out would join across it silently.
        values = [
            by_position.get(position) if by_position.get(position) is not None else float("nan")
            for position in range(len(xs))
        ]

        if chart.type in (ChartType.BAR, ChartType.STACKED_BAR):
            offsets = [
                position + (index - (len(chart.series) - 1) / 2) * bar_width
                for position in range(len(xs))
            ]
            axes.bar(offsets, values, width=bar_width, color=colour, label=series.name, alpha=0.85)
        elif chart.type is ChartType.SCATTER:
            axes.scatter(range(len(xs)), values, color=colour, label=series.name, alpha=0.8, s=22)
        else:
            axes.plot(
                range(len(xs)),
                values,
                color=colour,
                label=series.name,
                linewidth=2,
                # Present observations either side of a gap would otherwise be invisible: a
                # single point between two NaNs draws no line segment at all.
                marker="o",
                markersize=3,
            )
            if chart.type is ChartType.AREA:
                axes.fill_between(range(len(xs)), values, color=colour, alpha=0.12)


def _draw_annotations(axes, chart: ChartSpec, positions: dict[str, int]) -> None:
    """Deploy and release markers, drawn only where the axis covers them.

    `validate_chart` already rejects a floating annotation, so this is the second line of
    defence. A marker quietly drawn at the edge would claim a date the chart does not show.
    """
    for annotation in chart.annotations:
        if annotation.x not in positions:
            continue
        axes.axvline(
            positions[annotation.x], color=_ANNOTATION, linestyle="--", linewidth=1, alpha=0.9
        )
        axes.annotate(
            annotation.label,
            xy=(positions[annotation.x], 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=8,
            color=_ANNOTATION,
        )


def _style(axes, chart: ChartSpec, xs: list[str]) -> None:
    axes.set_title(chart.title, fontsize=12, loc="left")
    if chart.x_label:
        axes.set_xlabel(chart.x_label, fontsize=9)
    if chart.y_label:
        axes.set_ylabel(chart.y_label, fontsize=9)

    step = max(1, math.ceil(len(xs) / _MAX_TICKS))
    ticks = list(range(0, len(xs), step))
    # The last value is always labelled: with the first, it is what states the range.
    if ticks and ticks[-1] != len(xs) - 1:
        ticks.append(len(xs) - 1)
    axes.set_xticks(ticks)
    axes.set_xticklabels([xs[position] for position in ticks], fontsize=8, rotation=0)
    axes.tick_params(axis="y", labelsize=8)

    if chart.type in (ChartType.BAR, ChartType.STACKED_BAR, ChartType.AREA):
        # Zero baseline for anything whose area encodes magnitude. Truncating it exaggerates
        # every difference in the chart.
        axes.set_ylim(bottom=min(0.0, axes.get_ylim()[0]))

    axes.grid(axis="y", alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    # A legend only for more than one series: a single series' name is the chart's subject and
    # the title already carries it, so a one-entry legend is furniture.
    if len(chart.series) > 1:
        axes.legend(fontsize=8, frameon=False, loc="best")


def _to_bytes(figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    return buffer.getvalue()


def _placeholder(title: str) -> bytes:
    """A small image saying the chart could not be drawn.

    Bytes rather than an exception, and an image rather than nothing: a caller posting to Slack
    has already committed to sending something, and a missing attachment is harder to diagnose
    than a picture that explains itself.
    """
    figure = None
    try:
        figure, axes = plt.subplots(figsize=(6.0, 1.2), dpi=_DPI)
        axes.axis("off")
        axes.text(0.0, 0.5, f"{title} — could not be rendered", fontsize=10, color=_ANNOTATION)
        return _to_bytes(figure)
    finally:
        if figure is not None:
            plt.close(figure)
