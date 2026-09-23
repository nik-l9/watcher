"""Charts: the semantic checks Pydantic cannot do, and a terminal renderer.

A `ChartSpec` that validates is not yet a chart worth showing. Pydantic proves the shape —
a title, at least one series, at least one point, an evidence id. It cannot prove the thing
that makes a chart honest, which is that the picture and the claim agree.

Three defects a well-formed spec can still carry, all of which mislead:

  - **A floating annotation.** The whole point of a deploy marker is that the reader sees the
    line bend at the deploy. An annotation whose `x` appears on no series has nothing to
    mark; rendered, it lands at an arbitrary position and *invents* a coincidence.
  - **A series that is all gaps.** One point with `y=None` satisfies `min_length=1`. Drawn,
    it is an empty panel under a confident title — the same "absence looks like data"
    failure as an empty tool result, arriving through the picture instead of the prose.
  - **A trend asserted from one point.** Two points make a line; one makes a dot. A title
    saying "signups fell" over a single observation is a claim the chart does not support.

**Terminal rendering exists because an invisible chart is unverifiable.** The frontend is
deferred, and a chart nobody can see is a chart nobody checks — so the CLI draws it, badly
but truthfully, and a wrong axis or an off-by-one annotation becomes obvious at a glance
rather than after a frontend exists to reveal it.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from cortex.reports.schema import (
    ChartAnnotation,
    ChartPoint,
    ChartSeries,
    ChartSpec,
    ChartType,
)

#: Terminal-safe blocks, ascending. Chosen over a sparkline font because these render
#: identically in a pipe, a log file and a terminal with no unusual font.
#:
#: **No leading space.** The first draft started this string with one, so the lowest value in
#: a chart rendered as blank — and blank is what a *gap* renders as. A flat series sitting at
#: the chart minimum drew as an empty panel, which is the same "absence looks like data"
#: failure the empty-result work just closed, arriving through the picture instead of the
#: prose. Caught by rendering a two-series chart and looking at it.
_BLOCKS = "▁▂▃▄▅▆▇█"

#: What a missing observation draws as, and nothing else may draw as.
_GAP = " "

#: How wide a rendered chart may be. Narrower than a terminal on purpose: report output is
#: read in a pane beside other things.
_WIDTH = 72


@dataclass(frozen=True, slots=True)
class ChartProblem:
    """One reason a chart should not be shown as it stands."""

    chart_title: str
    problem: str

    def __str__(self) -> str:
        return f"{self.chart_title}: {self.problem}"


def validate_chart(chart: ChartSpec) -> list[ChartProblem]:
    """Semantic problems with a chart, or an empty list.

    Returns problems rather than raising: a report with one bad chart should lose the chart,
    not the report. The gate decides what to do with them, which keeps the policy in one
    place.
    """
    problems: list[ChartProblem] = []
    axis = {point.x for series in chart.series for point in series.points}

    for annotation in chart.annotations:
        if annotation.x not in axis:
            # The reason this matters: an annotation is a *causal* statement. A deploy marker
            # that lands where no data exists tells the reader the line bent at a moment the
            # chart has no observation for.
            problems.append(
                ChartProblem(
                    chart.title,
                    f"annotation {annotation.label!r} marks x={annotation.x!r}, which appears "
                    f"on no series; a marker at a position the data does not cover invents a "
                    f"coincidence",
                )
            )

    for series in chart.series:
        if all(point.y is None for point in series.points):
            problems.append(
                ChartProblem(
                    chart.title,
                    f"series {series.name!r} has no values at all, so the chart would draw an "
                    f"empty panel under a confident title",
                )
            )
        elif len(series.points) == 1 and chart.type in (ChartType.LINE, ChartType.AREA):
            problems.append(
                ChartProblem(
                    chart.title,
                    f"series {series.name!r} is a {chart.type.value} chart with one point; two "
                    f"points make a line, one makes a dot, and a trend cannot be read from it",
                )
            )

        seen: set[str] = set()
        duplicates = {p.x for p in series.points if p.x in seen or seen.add(p.x)}  # type: ignore[func-returns-value]
        if duplicates:
            problems.append(
                ChartProblem(
                    chart.title,
                    f"series {series.name!r} has more than one point at "
                    f"{sorted(duplicates)[:3]}; which value is drawn depends on ordering",
                )
            )

    return problems


def render_chart(chart: ChartSpec, width: int = _WIDTH) -> str:
    """Draw a chart in text, honestly.

    Gaps render as a space rather than a floor, for the same reason `ChartPoint.y` is
    `None` rather than `0`: a missing observation drawn at zero reads as a collapse.
    """
    lines = [f"  {chart.title}"]
    # The axis label is dropped when it says nothing the series row does not already say.
    # A single-series trend takes its name from the payload's value column, so a PostHog
    # event series rendered as "value" above a row also labelled "value" -- the same word
    # twice, which reads as an unfinished interface rather than an axis.
    names = {series.name for series in chart.series}
    if chart.y_label and names != {chart.y_label}:
        lines.append(f"  {chart.y_label}")

    axis = _axis(chart)
    scale = _scale(chart)

    for series in chart.series:
        values = {point.x: point.y for point in series.points}
        cells = [_cell(values.get(x), scale) for x in axis]
        rendered = "".join(cells)[:width]
        low, high = _range(series)
        # "range 459-1,110", not "459 → 1,110". These are the series minimum and maximum,
        # and an arrow between them is read by everyone as a movement from the first to the
        # second. A report concluding that signups were *flat* rendered a chart beside it
        # that appeared to say they had more than doubled -- same screen, opposite stories,
        # and the chart had never made that claim. The values were right; the notation lied.
        lines.append(f"    {series.name[:18]:18} |{rendered}|  range {low}–{high}")

    if chart.annotations:
        # Markers on their own row, aligned under the column they annotate. This is the row
        # that makes the causal story visible instead of merely stated.
        marks = [" "] * len(axis)
        legend: list[str] = []
        # Numbered in axis order, not in the order the annotations happen to be listed.
        # Unsorted, a chart with two deploys printed "^1" under the later one and "^2" under
        # the earlier, so the legend read right-to-left against the picture.
        on_axis = sorted(
            (a for a in chart.annotations if a.x in axis), key=lambda a: axis.index(a.x)
        )
        for index, annotation in enumerate(on_axis):
            marks[axis.index(annotation.x)] = "^"
            legend.append(f"^{index + 1} {annotation.label} ({annotation.x})")
        lines.append(f"    {'':18} |{''.join(marks)[:width]}|")
        lines.extend(f"      {entry}" for entry in legend)

    if axis:
        lines.append(f"    {'':18}  {axis[0]} … {axis[-1]}")
    return "\n".join(lines)


def _axis(chart: ChartSpec) -> list[str]:
    """Every x value across all series, in first-seen order.

    Ordering is taken from the data rather than sorted, because an x may be a label ("Paid
    Search") rather than a date, and alphabetising a category axis would silently reorder a
    funnel.
    """
    axis: list[str] = []
    for series in chart.series:
        for point in series.points:
            if point.x not in axis:
                axis.append(point.x)
    return axis


def _scale(chart: ChartSpec) -> tuple[float, float]:
    values = [p.y for s in chart.series for p in s.points if p.y is not None]
    if not values:
        return (0.0, 1.0)
    low, high = min(values), max(values)
    # A flat series would divide by zero. Widened rather than special-cased, so a genuinely
    # flat line draws at mid-height instead of at the floor — flat is a finding, and drawing
    # it at zero would read as collapsed.
    return (low, high) if high > low else (low - 1.0, high + 1.0)


def _cell(value: float | None, scale: tuple[float, float]) -> str:
    if value is None:
        return _GAP
    low, high = scale
    fraction = (value - low) / (high - low)
    index = round(fraction * (len(_BLOCKS) - 1))
    return _BLOCKS[max(0, min(index, len(_BLOCKS) - 1))]


def _range(series: object) -> tuple[str, str]:
    values = [p.y for p in series.points if p.y is not None]  # type: ignore[attr-defined]
    if not values:
        return ("no data", "no data")
    return (_number(min(values)), _number(max(values)))


def _number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if value == int(value):
        return str(int(value))
    return f"{value:.4g}"


# --------------------------------------------------------------------------- extraction

#: How many charts one report may carry.
#:
#: Three. A report is read, not browsed, and a page of panels is skimmed rather than
#: checked — which defeats the purpose of drawing the data in the first place.
_MAX_CHARTS = 3

#: Minimum points before a series is worth drawing. Two make a line; below that there is no
#: shape to see and the title would be doing all the work.
_MIN_POINTS = 3

#: Payload keys that hold a time series, and how to read one row of it.
#:
#: Kept as a small table rather than sniffed generically. A generic "find the numbers"
#: extractor was the first attempt and it charted things like `total_matching` and page
#: counts — plausible-looking lines built from bookkeeping fields, which is a worse failure
#: than drawing nothing.
_SERIES_SHAPES = (
    # GA4: {"rows": [{"dimensions": {"date": "..."}, "metrics": {"sessions": 2400}}]}
    ("rows", "dimensions", "metrics"),
    # PostHog: {"series": [{"bucket": "...", "segment": "Mobile", "value": 42}]}
    ("series", None, None),
)


def charts_from_evidence(rows: Sequence[Any], *, limit: int = _MAX_CHARTS) -> list[ChartSpec]:
    """Build charts from observations, never from prose.

    This is the same rule the Sources section follows, for the same reason: a model asked to
    write a series inline can write a number that was never observed, and a chart is the one
    part of a report a reader trusts without reading the citation. Derived from the stored
    payload, a chart cannot say anything the tool did not return.

    Annotations come from change records in the *same* investigation — a deploy, a commit, a
    PostHog annotation — and only when their date lands on the axis. A marker at a position
    the series does not cover would invent the coincidence the reader is being shown.
    """
    markers = _markers(rows)
    charts: list[ChartSpec] = []

    for row in rows:
        if len(charts) >= limit:
            break
        payload = getattr(row, "payload", None)
        if not isinstance(payload, dict):
            continue
        for series in _series_from(payload):
            if len(series.points) < _MIN_POINTS:
                continue
            axis = {point.x for point in series.points}
            annotations = [
                ChartAnnotation(x=x, label=label, evidence_id=evidence_id)
                for x, label, evidence_id in markers
                if x in axis
            ][:3]
            chart = ChartSpec(
                type=ChartType.LINE,
                title=f"{_metric_name(payload, series.name)} over time",
                x_label="date",
                y_label=series.name,
                series=[series],
                annotations=annotations,
                evidence_ids=[row.id],
            )
            # Validated before it is offered. The extractor is code and can be wrong, and a
            # chart that fails its own checks should never reach the gate to be rejected
            # there — that would report a rejection against the analyst for something the
            # extractor did.
            if not validate_chart(chart):
                charts.append(chart)
            if len(charts) >= limit:
                break
    return charts


def _series_from(payload: dict[str, Any]) -> list[ChartSeries]:
    """Every time series in one payload, split by segment where it has one."""
    out: list[ChartSeries] = []
    for key, dimension_key, metric_key in _SERIES_SHAPES:
        items = payload.get(key)
        if not isinstance(items, list) or not items:
            continue
        buckets: dict[str, list[ChartPoint]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            when = _when(item, dimension_key)
            value, label = _value(item, metric_key)
            if when is None or value is None:
                continue
            segment = str(item.get("segment") or "") if metric_key is None else ""
            name = f"{label} · {segment}" if segment else label
            buckets.setdefault(name, []).append(ChartPoint(x=when, y=value))
        for name, points in buckets.items():
            # Deduplicated on x: two points at one date make the drawn value depend on
            # ordering, which `validate_chart` rejects — and it is the extractor's job not
            # to produce it.
            unique: dict[str, ChartPoint] = {}
            for point in points:
                unique.setdefault(point.x, point)
            out.append(ChartSeries(name=name[:120], points=list(unique.values())))
    return out


def _when(item: dict[str, Any], dimension_key: str | None) -> str | None:
    """The row's date, as a plain YYYY-MM-DD string."""
    candidates: list[Any] = []
    if dimension_key:
        dimensions = item.get(dimension_key)
        if isinstance(dimensions, dict):
            candidates.extend(dimensions.values())
    candidates.extend(item.get(key) for key in ("bucket", "date", "day", "timestamp"))
    for candidate in candidates:
        normalised = _as_date(candidate)
        if normalised:
            return normalised
    return None


def _value(item: dict[str, Any], metric_key: str | None) -> tuple[float | None, str]:
    """The row's number and what to call it."""
    if metric_key:
        metrics = item.get(metric_key)
        if isinstance(metrics, dict):
            for name, raw in metrics.items():
                number = _as_number(raw)
                if number is not None:
                    return number, str(name)
        return None, ""
    number = _as_number(item.get("value"))
    return (number, "value") if number is not None else (None, "")


def _markers(rows: Sequence[Any]) -> list[tuple[str, str, uuid.UUID]]:
    """Change records in this investigation, as (date, label, evidence id).

    Only changes — a deploy, a merged commit, a PostHog annotation. This is what makes a
    chart causal rather than decorative: the reader sees the line bend at the deploy instead
    of being told that it did.
    """
    markers: list[tuple[str, str, uuid.UUID]] = []
    for row in rows:
        payload = getattr(row, "payload", None)
        if not isinstance(payload, dict):
            continue
        for deployment in payload.get("deployments") or []:
            when = _as_date(deployment.get("created_at")) if isinstance(deployment, dict) else None
            if when:
                sha = str(deployment.get("sha") or "")[:7]
                markers.append((when, f"deploy {sha}".strip(), row.id))
        for commit in payload.get("commits") or []:
            when = _as_date(commit.get("date")) if isinstance(commit, dict) else None
            if when:
                subject = str(commit.get("subject") or "")[:40]
                markers.append((when, subject or f"commit {commit.get('short_sha')}", row.id))
        for annotation in payload.get("annotations") or []:
            when = _as_date(annotation.get("date_marker")) if isinstance(annotation, dict) else None
            if when:
                markers.append((when, str(annotation.get("content") or "")[:40], row.id))
    return markers


def _metric_name(payload: dict[str, Any], series_name: str) -> str:
    """What to call the chart.

    Prefers the source's own name for the thing — PostHog returns the event name, so the
    title reads "user signed up over time" rather than "value over time". Falls back to the
    series name, which for GA4 is already the metric.
    """
    for key in ("event", "metric", "name"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:120]
    return series_name[:120]


def _as_date(value: Any) -> str | None:
    """A YYYY-MM-DD string, or None if this is not a date.

    Strict about the prefix rather than parsing loosely: `dimensions` can hold
    `deviceCategory` or a channel name, and a permissive parser would happily treat one as a
    date and draw a chart with a category on a time axis.
    """
    if not isinstance(value, str) or len(value) < 10:
        return None
    head = value[:10]
    if head[4] != "-" or head[7] != "-":
        return None
    try:
        datetime.strptime(head, "%Y-%m-%d")  # noqa: DTZ007 - a date, not an instant
    except ValueError:
        return None
    return head


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
