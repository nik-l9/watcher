"""The report view — M7.

**Server-rendered HTML from the gateway, not a Next.js app, and that is a deliberate
departure from the plan.** The plan said "Next.js: single report page". One page does not
justify a build toolchain, a `node_modules` tree, a second deploy target and a second place
where a claim can be rendered without its citation. The JSON endpoints remain the contract, so
a Next.js app can be built against them later without changing anything here — what would be
hard to undo is the coupling, and there is none.

What the page has to earn is stated in the plan and does not change with the technology:
**every claim visibly clickable to its evidence.** That is the product's whole argument. So a
citation renders as a link to the source that established it, an uncited claim cannot appear
(the gate removed it before this code ever sees the report), and the trace of what the analyst
did is on the page rather than behind a developer tool.

**Everything interpolated is escaped.** Claim text, chart titles and tool errors are all model
or vendor output, and a finding whose title is `<img onerror=...>` would otherwise be markup.
There is a test for it, because "we escape" is the kind of claim that is true until someone
adds a field.

No JavaScript. A page that needs a script to show a report cannot be saved, printed, emailed,
or read with scripting off — and a progress view that polls is a `<meta refresh>`, which is
five characters against a framework.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime, timedelta
from html import escape
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import Investigation, InvestigationStatus, Report, ToolCall
from cortex.reports.schema import ChartSpec
from cortex.reports.svg import render_svg
from cortex.tenancy.context import TenantContext
from services.gateway.deps import get_session, require_tenant, tenant_for_investigation

router = APIRouter(tags=["ui"], include_in_schema=False)

#: How often a running investigation's page reloads itself, in seconds.
#:
#: Matched to the loop's own pace: steps are roughly ten seconds apart, so a faster refresh
#: costs requests to show the same trace again. A `<meta refresh>` rather than a poll in
#: JavaScript — it is one tag, it stops when the investigation finishes, and it works on a page
#: with scripting disabled.
_REFRESH_SECONDS = 5

#: Statuses at which the page stops refreshing itself, because nothing further will change.
_TERMINAL = frozenset(
    {
        InvestigationStatus.COMPLETED,
        InvestigationStatus.FAILED,
        InvestigationStatus.CANCELLED,
    }
)

#: After this long, a non-terminal investigation is almost certainly abandoned rather than
#: working.
#:
#: Found by looking at real rows: eleven of twelve investigations in this database sit at
#: `queued` permanently, because a `POST /investigations` enqueues work that no running worker
#: consumed, and nothing ever revisits the row. On a page, "queued" and "queued since Tuesday"
#: render identically — so a list of mostly-abandoned rows reads as a product that is broken
#: rather than one whose worker is not running.
#:
#: Twenty minutes, against a longest observed run of 123 seconds and a 300-second loop budget.
#: Set well above the budget on purpose: calling a live investigation abandoned is a worse error
#: than being slow to call an abandoned one abandoned.
#:
#: Disclosed rather than corrected. Marking the row FAILED from a page render would mean a GET
#: mutating state, and the gateway deliberately does not decide that a worker is dead.
_LIKELY_ABANDONED = timedelta(minutes=20)

_STYLE = """
:root { color-scheme: light dark; --fg:#111827; --muted:#6b7280; --line:#e5e7eb;
        --bg:#ffffff; --card:#f9fafb; --link:#1d4ed8; --warn:#b45309; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e5e7eb; --muted:#9ca3af; --line:#374151; --bg:#0b0f19; --card:#111827;
          --link:#93b4ff; --warn:#fbbf24; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.6 ui-sans-serif,
       -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 860px; margin:0 auto; padding: 32px 20px 96px; }
a { color:var(--link); }
h1 { font-size:22px; line-height:1.35; margin:0 0 4px; font-weight:650; }
h2 { font-size:13px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted);
     margin:36px 0 10px; font-weight:650; }
h3 { font-size:16px; margin:20px 0 6px; font-weight:600; }
p  { margin:0 0 10px; }
.meta { color:var(--muted); font-size:13px; margin-bottom:6px; }
.answer { font-size:17px; }
.cites { font-size:12px; color:var(--muted); margin-left:2px; white-space:nowrap; }
.cites a { text-decoration:none; border-bottom:1px dotted currentColor; }
.claim { margin:0 0 12px; }
ul { margin:0 0 10px; padding-left:20px; }
li { margin:0 0 6px; }
.pill { display:inline-block; font-size:12px; padding:1px 8px; border-radius:999px;
        border:1px solid var(--line); color:var(--muted); }
.verdict-supported { color:#047857; border-color:currentColor; }
.verdict-contradicted { color:#b91c1c; border-color:currentColor; }
.verdict-inconclusive { color:var(--muted); }
table { border-collapse:collapse; width:100%; font-size:13px; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line);
         vertical-align:top; }
th { color:var(--muted); font-weight:600; }
td.failed { color:#b91c1c; }
.chart { width:100%; height:auto; margin:8px 0 18px; }
.chart-title { font-size:13px; font-weight:600; fill:var(--fg); }
.chart-grid { stroke:var(--line); stroke-width:1; }
.chart-tick { font-size:10px; fill:var(--muted); }
.chart-tick-y { text-anchor:end; }
.chart-tick-x { text-anchor:middle; }
.chart-axis-label, .chart-legend, .chart-annotation { font-size:10px; fill:var(--muted); }
.chart-empty { font-size:12px; fill:var(--muted); }
.notice { background:var(--card); border-left:3px solid var(--warn); padding:10px 14px;
          margin:12px 0; font-size:14px; }
.rows { list-style:none; padding:0; margin:0; }
.rows li { border-bottom:1px solid var(--line); padding:10px 0; margin:0; }
.rows .title { font-weight:550; }
footer { margin-top:48px; color:var(--muted); font-size:12px; }
code { font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; }

/* The trace as a vertical timeline. A table answered "what ran"; the question a reader
   actually has is "how did this get decided", which is an ordering with things happening
   along it. The rail is one pseudo-element and the nodes are another, so this costs no
   markup a reader without CSS would trip over -- it degrades to a plain list. */
.tl { list-style:none; margin:0; padding:0 0 0 26px; position:relative; }
.tl::before { content:""; position:absolute; left:5px; top:8px; bottom:8px; width:2px;
              background:var(--line); }
.tl > li { position:relative; padding:0 0 16px; border:0; margin:0; }
.tl > li::before { content:""; position:absolute; left:-26px; top:5px; width:10px; height:10px;
                   border-radius:50%; background:var(--bg); border:2px solid var(--muted);
                   box-sizing:border-box; }
.tl .step-ok::before { border-color:#047857; }
.tl .step-empty::before { border-color:var(--warn); }
.tl .step-failed::before { border-color:#b91c1c; }
/* A verdict is not a step. Square, filled, and offset so the eye reads it as something that
   happened *to* the investigation rather than another call it made. */
.tl .step-verdict::before { border-radius:2px; width:9px; height:9px; left:-25px;
                            background:var(--muted); border-color:var(--muted); }
.tl .step-verdict.v-supported::before { background:#047857; border-color:#047857; }
.tl .step-verdict.v-contradicted::before { background:#b91c1c; border-color:#b91c1c; }
.step-head { font-size:14px; }
.step-when { color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums; }
.step-detail { color:var(--muted); font-size:12px; margin-top:2px; word-break:break-word; }
.struck { text-decoration:line-through; text-decoration-thickness:1px; color:var(--muted); }
details { margin:6px 0 14px; }
summary { cursor:pointer; color:var(--link); font-size:13px; }
summary::marker { color:var(--muted); }
.removed { border-left:2px solid var(--line); padding:8px 0 2px 12px; margin:10px 0 0; }
.removed li { list-style:none; margin:0 0 12px; }
"""


def _page(title: str, body: str, *, refresh: bool = False) -> HTMLResponse:
    head = f'<meta http-equiv="refresh" content="{_REFRESH_SECONDS}">' if refresh else ""
    return HTMLResponse(
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>{escape(title)}</title>{head}<style>{_STYLE}</style></head>"
        f"<body><main>{body}</main></body></html>"
    )


@router.get("/ui/investigations", response_class=HTMLResponse)
async def investigation_list(
    limit: int = 25,
    tenant: TenantContext = Depends(require_tenant),
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    """This tenant's investigations. Titles, not questions."""
    limit = max(1, min(limit, 100))
    rows = (
        (
            await session.execute(
                select(Investigation, Report.id)
                .outerjoin(
                    Report,
                    (Report.investigation_id == Investigation.id)
                    & (Report.tenant_id == tenant.tenant_id),
                )
                .where(Investigation.tenant_id == tenant.tenant_id)
                .order_by(Investigation.created_at.desc(), Investigation.id.desc())
                .limit(limit)
            )
        )
        .tuples()
        .all()
    )

    if not rows:
        items = "<p class=meta>No investigations yet.</p>"
    else:
        items = (
            "<ul class=rows>"
            + "".join(
                f"<li><div class=title>"
                f'<a href="/ui/investigations/{row.id}">'
                f"{escape(row.title or row.question[:72])}</a></div>"
                f"<div class=meta>{escape(row.status.value)} · {_when(row.created_at)}"
                f"{' · report ready' if report_id else ''}"
                f"{' · likely abandoned' if _looks_abandoned(row) else ''}</div></li>"
                for row, report_id in rows
            )
            + "</ul>"
        )

    return _page(
        f"Investigations — {tenant.tenant_slug}",
        f"<h1>Investigations</h1><p class=meta>{escape(tenant.tenant_slug)}</p>{items}",
    )


@router.get("/ui/investigations/{investigation_id}", response_class=HTMLResponse)
async def investigation_page(
    investigation_id: uuid.UUID,
    tenant: TenantContext = Depends(tenant_for_investigation),
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    """One investigation: the report if it has one, the trace either way.

    The trace is on the page while the investigation is still running, which is the answer to
    the complaint that started this work — a 63-second wait that printed nothing was
    indistinguishable from a hang.
    """
    investigation = (
        await session.execute(
            select(Investigation).where(
                Investigation.id == investigation_id,
                Investigation.tenant_id == tenant.tenant_id,
            )
        )
    ).scalar_one_or_none()
    if investigation is None:
        # One response for absent and foreign, as everywhere else, so ids cannot be
        # enumerated through the UI either.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown investigation")

    report_row = (
        await session.execute(
            select(Report).where(
                Report.investigation_id == investigation_id,
                Report.tenant_id == tenant.tenant_id,
            )
        )
    ).scalar_one_or_none()

    calls = (
        (
            await session.execute(
                select(ToolCall)
                .where(
                    ToolCall.investigation_id == investigation_id,
                    ToolCall.tenant_id == tenant.tenant_id,
                )
                .order_by(ToolCall.created_at.asc(), ToolCall.id.asc())
            )
        )
        .scalars()
        .all()
    )

    # The thread, if this is a follow-up. Without it a follow-up reads as an oddly specific
    # standalone question -- "on which single day was conversation_created highest" makes sense
    # only next to what it follows, and its citations may point at evidence gathered there.
    parent = None
    if investigation.parent_id is not None:
        parent = (
            await session.execute(
                select(Investigation).where(
                    Investigation.id == investigation.parent_id,
                    Investigation.tenant_id == tenant.tenant_id,
                )
            )
        ).scalar_one_or_none()

    children = (
        (
            await session.execute(
                select(Investigation)
                .where(
                    Investigation.parent_id == investigation_id,
                    Investigation.tenant_id == tenant.tenant_id,
                )
                .order_by(Investigation.created_at.asc())
            )
        )
        .scalars()
        .all()
    )

    title = investigation.title or investigation.question[:72]
    parts = [
        # The list route has no investigation id to resolve a tenant from, so this link carries
        # one. Without it, following "all investigations" from a working report page produced
        # the same 422 the report page used to -- a dead end one click deeper.
        '<p class=meta><a href="/ui/investigations?tenant='
        f'{quote(tenant.tenant_slug)}">← all investigations</a></p>',
        f"<h1>{escape(title)}</h1>",
        f"<p class=meta>{escape(investigation.question)}</p>",
        f"<p><span class=pill>{escape(investigation.status.value)}</span> "
        f"<span class=meta>{investigation.steps_used} steps · "
        f"{investigation.tokens_used:,} tokens · {_when(investigation.created_at)}</span></p>",
    ]

    stale = _looks_abandoned(investigation)
    if stale:
        parts.append(
            "<div class=notice>This investigation has been "
            f"{escape(investigation.status.value)} since {_when(investigation.created_at)} and "
            "has almost certainly been abandoned -- no worker appears to have finished it. "
            "The row is left as it is rather than marked failed here: reading a page must not "
            "change what it reports.</div>"
        )
    elif investigation.status not in _TERMINAL:
        parts.append(
            "<div class=notice>Still working. This page refreshes itself; the trace below "
            "shows what the analyst has done so far.</div>"
        )
    if investigation.error:
        parts.append(f"<div class=notice>{escape(investigation.error)}</div>")

    if parent is not None:
        parts.append(
            '<p class=meta>Follow-up to <a href="/ui/investigations/'
            f'{parent.id}">{escape(parent.title or parent.question[:72])}</a>. '
            "Claims below may cite observations gathered there.</p>"
        )
    elif investigation.parent_id is not None:
        # The parent was deleted, or reassigned. Said rather than hidden: a citation pointing at
        # evidence the reader cannot open is otherwise unexplained.
        parts.append("<p class=meta>Follow-up to an investigation that is no longer available.</p>")

    if children:
        parts.append("<p class=meta>Followed up by: ")
        parts.append(
            ", ".join(
                f'<a href="/ui/investigations/{child.id}">'
                f"{escape(child.title or child.question[:60])}</a>"
                for child in children
            )
        )
        parts.append("</p>")

    body = report_row.body if report_row is not None and isinstance(report_row.body, dict) else {}

    # The trace goes *above* the answer, and folded shut once there is an answer to read.
    #
    # Two readers, and the order has to serve both. Someone waiting on a running investigation
    # wants the trace and nothing else, because there is no report yet. Someone reading a
    # finished one wants the answer first and the trace only when they doubt it. At the bottom,
    # "how was this decided" became the last thing on a long page -- the one question the page
    # exists to answer. Left open, it was the first four screens.
    #
    # Hypotheses are rendered on the trace rather than in the report body, so a verdict appears
    # at the step whose evidence produced it. See `_trace_html`.
    trace = _trace_html(calls, list(body.get("hypotheses") or []))
    if report_row is not None:
        parts.append(
            f"<details><summary>How this was investigated — {len(calls)} step(s)</summary>"
            f"{trace}</details>"
        )
        parts.append(_report_html(report_row))
    else:
        # No report yet, so the trace is the whole page. It carries its own heading rather than
        # a disclosure, because there is nothing to fold it away in favour of.
        parts.append("<h2>How this was investigated</h2>")
        parts.append(trace)

    return _page(
        title,
        "".join(parts),
        # An abandoned row is not refreshed. Reloading every five seconds forever to show the
        # same dead row is the one behaviour that turns a disclosure into a cost.
        refresh=investigation.status not in _TERMINAL and not stale,
    )


def _looks_abandoned(investigation: Investigation) -> bool:
    """Whether a non-terminal investigation has stopped being plausible.

    Advisory, and computed rather than stored: nothing writes this, so it cannot become a
    second piece of state that disagrees with the row.
    """
    if investigation.status in _TERMINAL or investigation.created_at is None:
        return False
    started = investigation.created_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    return datetime.now(UTC) - started > _LIKELY_ABANDONED


#: How many items of a long list are shown before the rest is folded away.
#:
#: Three, because the page's job is to be read. A real report carried nine data-quality notes,
#: five risks and twenty trace steps all fully expanded, and the effect of showing everything at
#: once is that nothing is read -- the important disclosure and the sixth restatement of "this
#: sync is a week old" have identical visual weight.
_SHOWN_BEFORE_FOLD = 3


def _folded_list(items: list[str], *, more_label: str) -> str:
    """A list showing the first few items, with the remainder behind a disclosure.

    `<details>` rather than a script, as everywhere else on this page. The count is in the
    summary so a reader knows what they are choosing not to open.
    """
    if not items:
        return ""
    head = "".join(f"<li>{item}</li>" for item in items[:_SHOWN_BEFORE_FOLD])
    if len(items) <= _SHOWN_BEFORE_FOLD:
        return f"<ul>{head}</ul>"
    rest = "".join(f"<li>{item}</li>" for item in items[_SHOWN_BEFORE_FOLD:])
    return (
        f"<ul>{head}</ul>"
        f"<details><summary>{len(items) - _SHOWN_BEFORE_FOLD} more {escape(more_label)}"
        f"</summary><ul>{rest}</ul></details>"
    )


#: Notes that differ only by which connector they name.
#:
#: A report listed six of these -- github commits, deployments, issues and links, posthog
#: annotations and events -- each a separate bullet saying the same thing about a different
#: stream. Six bullets is not six findings, and printing them separately buried three notes that
#: were about this investigation's own evidence.
_STALE_SYNC = re.compile(r"^The (?P<what>.+?) sync last succeeded (?P<age>.+?) ago\.")


def _fold_stale_syncs(notes: list[str]) -> list[str]:
    """Collapse the repeated sync-staleness notes into one, keeping every other note intact."""
    stale: list[str] = []
    kept: list[str] = []
    age = ""
    for note in notes:
        match = _STALE_SYNC.match(note)
        if match:
            stale.append(match.group("what"))
            age = age or match.group("age")
        else:
            kept.append(note)
    if not stale:
        return kept
    if len(stale) == 1:
        return [*kept, notes[[bool(_STALE_SYNC.match(n)) for n in notes].index(True)]]
    return [
        *kept,
        f"{len(stale)} syncs last succeeded {age} ago ({', '.join(stale)}). Stored history for "
        "them is that old, and a recent change may not appear.",
    ]


def _report_html(row: Report) -> str:
    """The stored report body, rendered.

    Read from the stored body rather than re-derived. That body is what the gate and the
    verifier produced; re-deriving it here would create a second path by which a claim could
    reach a reader without passing through them.
    """
    body: dict[str, Any] = row.body if isinstance(row.body, dict) else {}
    sources = {
        str(source.get("evidence_id")): source
        for source in body.get("sources", [])
        if isinstance(source, dict)
    }
    parts: list[str] = []

    # Surfaced, and shown rather than counted. A reader is entitled to know that claims were
    # removed before this was shown to them — it is the difference between a report that was
    # checked and one that merely looks tidy — and the claims themselves are the evidence for
    # which of the two this is.
    parts.append(_removed_html(row.gate_rejections, row.verifier_rejections))

    # Above the answer, because when a question's premise is false that is the most important
    # thing on the page and burying it is the failure the premise field exists to prevent -- a
    # reader who stops after one line has been misinformed by a report that was accurate
    # throughout. Absent on the ordinary report, where the question asserted nothing checkable.
    parts.append(_premise_html(body))

    parts.append("<h2>Answer</h2>")
    for claim in body.get("executive_summary", []):
        parts.append(f"<p class='claim answer'>{_claim_html(claim, sources)}</p>")

    findings = body.get("findings") or []
    if findings:
        parts.append("<h2>Findings</h2>")
        rendered = [
            f"<h3>{escape(str(finding.get('title', '')))} "
            f"<span class=pill>{escape(str(finding.get('confidence', '')))}</span></h3>"
            + "".join(
                f"<p class=claim>{_claim_html(claim, sources)}</p>"
                for claim in finding.get("claims", [])
            )
            for finding in findings
        ]
        parts.append("".join(rendered[:_SHOWN_BEFORE_FOLD]))
        if len(rendered) > _SHOWN_BEFORE_FOLD:
            parts.append(
                f"<details><summary>{len(rendered) - _SHOWN_BEFORE_FOLD} more finding(s)"
                f"</summary>{''.join(rendered[_SHOWN_BEFORE_FOLD:])}</details>"
            )

    charts = body.get("charts") or []
    if charts:
        parts.append("<h2>Charts</h2>")
        for chart in charts:
            parts.append(_chart_html(chart))

    recommendations = body.get("recommendations") or []
    if recommendations:
        parts.append("<h2>Recommendations</h2><ul>")
        for recommendation in sorted(
            recommendations, key=lambda r: r.get("priority", 2) if isinstance(r, dict) else 2
        ):
            parts.append(
                f"<li><strong>P{escape(str(recommendation.get('priority', 2)))}</strong> "
                f"{escape(str(recommendation.get('action', '')))}"
                f"<div class=meta>{escape(str(recommendation.get('rationale', '')))} "
                f"{_cites_html(recommendation.get('evidence_ids', []), sources)}</div></li>"
            )
        parts.append("</ul>")

    for heading, key, field, label in (
        ("Risks and caveats", "risks", "description", "risk(s)"),
        ("Data quality", "data_quality", "note", "note(s)"),
    ):
        entries = body.get(key) or []
        if not entries:
            continue
        notes = [str(entry.get(field, "")) for entry in entries if isinstance(entry, dict)]
        if key == "data_quality":
            notes = _fold_stale_syncs(notes)
        parts.append(f"<h2>{heading}</h2>")
        parts.append(_folded_list([escape(note) for note in notes], more_label=label))

    if sources:
        # Open by default, unlike the other folds. Every citation on the page is an anchor into
        # this table, and a `<details>` that is closed swallows the jump -- the reason the page
        # exists is that a claim can be followed to its source in one click.
        rows = "".join(
            f'<tr id="e-{escape(evidence_id)}"><td><code>{escape(evidence_id[:8])}</code>'
            f"<td>{escape(str(source.get('tool_name', '')))}."
            f"{escape(str(source.get('capability', '')))}"
            f"{' <span class=pill>cached</span>' if source.get('from_cache') else ''}"
            f"<td>{escape(str(source.get('source_ref') or ''))}</tr>"
            for evidence_id, source in sources.items()
        )
        parts.append(
            f"<h2>Sources</h2><details open><summary>{len(sources)} observation(s) cited"
            "</summary><table><tr><th>id<th>source<th>reference</tr>"
            f"{rows}</table></details>"
        )

    return "".join(parts)


#: How each premise verdict reads on the page, and whether it is worth interrupting for.
#:
#: Only `false` gets the notice treatment. "The question's assertion holds" is what a reader
#: already assumes, so announcing it would make the one case that matters harder to spot.
_PREMISE_NOTICES = {
    "false": (
        "notice",
        "The question assumes something the evidence contradicts. The answer below corrects "
        "it rather than answering as asked.",
    ),
    "unverifiable": (
        "meta",
        "The question assumes something this evidence can neither confirm nor contradict.",
    ),
}


def _premise_html(body: dict[str, Any]) -> str:
    """What the report concluded about the question's own assertion, or nothing."""
    verdict = str(body.get("premise") or "none_asserted")
    if verdict not in _PREMISE_NOTICES:
        return ""
    css, sentence = _PREMISE_NOTICES[verdict]
    checked = str(body.get("premise_checked") or "")
    tail = f" Checked: {escape(checked)}" if checked else ""
    return f"<div class={css}>{escape(sentence)}{tail}</div>"


def _claim_html(claim: object, sources: dict[str, dict]) -> str:
    if not isinstance(claim, dict):
        return ""
    return escape(str(claim.get("text", ""))) + _cites_html(claim.get("evidence_ids", []), sources)


def _cites_html(ids: object, sources: dict[str, dict]) -> str:
    """Citations as links to the source row that established the claim.

    This is the page's reason to exist. A citation rendered as a bare uuid is technically
    grounded and practically unreadable, so each one shows the capability it came from and
    jumps to its row in Sources.
    """
    if not isinstance(ids, list) or not ids:
        return ""
    links = []
    for raw in ids:
        evidence_id = str(raw)
        source = sources.get(evidence_id)
        label = (
            f"{source.get('tool_name', '')}.{source.get('capability', '')}"
            if source
            else evidence_id[:8]
        )
        links.append(
            f'<a href="#e-{escape(evidence_id)}" title="{escape(evidence_id)}">{escape(label)}</a>'
        )
    return f" <span class=cites>[{', '.join(links)}]</span>"


def _chart_html(chart: object) -> str:
    """One chart, or nothing.

    Validated back into a `ChartSpec` before rendering. The stored body is JSON, and a
    renderer that trusted its shape would raise on the report page — the last step, after
    everything expensive has already succeeded. A chart that cannot be parsed is dropped with
    a note rather than taking the report with it.
    """
    try:
        spec = ChartSpec.model_validate(chart)
    except Exception:  # noqa: BLE001 - see the docstring
        return "<p class=meta>A chart in this report could not be rendered.</p>"
    return render_svg(spec)


def _trace_html(calls: list[ToolCall], hypotheses: list[dict]) -> str:
    """What the analyst did, in order, with the moment each hypothesis was settled on it.

    This was a table. A table answers "what ran", which nobody asks; the question a reader has
    is "how was this decided", and that is an ordering with events along it. So: a vertical
    rail, one node per call, and a squared-off node wherever a hypothesis was supported or
    contradicted by the evidence that step returned.

    Placing verdicts on the trace rather than in a list of their own is the part that earns
    trust. "Four hypotheses were tested" is a claim about diligence. "This one died here, on
    that call, against that evidence" is a thing the reader can check, and it is checkable
    precisely because the citation already resolves to a row further down the page.

    Params are shown and responses are not, which is the line the audit row itself draws: a
    response body belongs in the evidence store, and putting it on a page would double the
    surface on which customer data is displayed.
    """
    if not calls:
        return "<p class=meta>No tool calls yet.</p>"

    # Which step settled which hypothesis. A hypothesis is placed at the *last* step that
    # contributed evidence to it, because that is the one after which its verdict was
    # available -- placing it at the first would claim the analyst knew sooner than it did.
    step_of_evidence = {
        str(call.evidence_id): index for index, call in enumerate(calls) if call.evidence_id
    }
    settled_at: dict[int, list[dict]] = {}
    unsettled: list[dict] = []
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, dict):
            continue
        cited = [
            str(raw)
            for key in ("supporting_evidence_ids", "contradicting_evidence_ids")
            for raw in (hypothesis.get(key) or [])
        ]
        steps = [step_of_evidence[e] for e in cited if e in step_of_evidence]
        if steps:
            settled_at.setdefault(max(steps), []).append(hypothesis)
        else:
            unsettled.append(hypothesis)

    rows = []
    for index, call in enumerate(calls):
        outcome = (
            "ok"
            if call.succeeded and call.evidence_id
            else ("empty" if call.succeeded else "failed")
        )
        detail = escape(call.error or "") if call.error else _params_html(call.params)
        took = f" · {call.duration_ms} ms" if call.duration_ms else ""
        rows.append(
            f"<li class=step-{outcome}>"
            f"<div class=step-head>{escape(call.tool_name)}.{escape(call.capability)} "
            f"<span class=pill>{outcome}</span></div>"
            f"<div class=step-when>{_time(call.created_at)}{escape(took)}</div>"
            + (f"<div class=step-detail>{detail}</div>" if detail else "")
            + "</li>"
        )
        for hypothesis in settled_at.get(index, []):
            rows.append(_verdict_step_html(hypothesis))

    empty = sum(1 for call in calls if not call.succeeded or call.evidence_id is None)
    note = (
        f"<p class=meta>{empty} of {len(calls)} call(s) returned nothing. "
        "A conclusion resting on those describes what was not found, which is not the same "
        "as what does not exist.</p>"
        if empty
        else ""
    )
    # No heading of its own: the caller wraps this in a disclosure whose summary already names
    # it, and a heading inside that reads as a second section.
    return f"<ul class=tl>{''.join(rows)}</ul>" + note + _unsettled_html(unsettled)


def _verdict_step_html(hypothesis: dict) -> str:
    """A hypothesis resolving, rendered as a moment on the trace rather than a row in a list.

    An elimination on dates is labelled *impossible* rather than *contradicted*, and the two
    are worth distinguishing on the page. "Contradicted" invites a reader to weigh the
    reasoning; "impossible" says a cause post-dated its effect, which is arithmetic and not a
    judgement anyone can disagree with. It is also the strongest thing this system can say
    about a candidate, so hiding it inside the same label as every other rejection wastes it.
    """
    verdict = str(hypothesis.get("verdict", "inconclusive"))
    reasoning = str(hypothesis.get("reasoning") or "")
    cause_at = str(hypothesis.get("cause_at") or "")
    onset = str(hypothesis.get("effect_onset") or "")
    eliminated = bool(cause_at and onset and cause_at > onset)

    label = "impossible" if eliminated else verdict
    dates = (
        f"<div class=step-detail>Cause dated {escape(cause_at)}; the movement began "
        f"{escape(onset)}. A cause cannot post-date its effect.</div>"
        if eliminated
        else ""
    )
    return (
        f"<li class='step-verdict v-{escape(verdict)}'>"
        f"<div class=step-head><span class='pill verdict-{escape(verdict)}'>"
        f"{escape(label)}</span> "
        f"<span class={'struck' if verdict == 'contradicted' else ''}>"
        f"{escape(str(hypothesis.get('statement', '')))}</span></div>"
        + dates
        + (f"<div class=step-detail>{escape(reasoning)}</div>" if reasoning else "")
        + "</li>"
    )


def _unsettled_html(hypotheses: list[dict]) -> str:
    """Hypotheses the investigation raised and could not decide, kept where a reader sees them.

    These have no place on the trace, because nothing settled them — which is exactly why they
    have to be somewhere. An investigation that considered a cause and could not rule it in or
    out has told the reader something real about the limits of the answer, and dropping it
    because it produced no verdict would make the report look more decided than it is.
    """
    if not hypotheses:
        return ""
    items = []
    for hypothesis in hypotheses:
        reasoning = str(hypothesis.get("reasoning") or "")
        items.append(
            f"<li><span class=pill>{escape(str(hypothesis.get('verdict', 'inconclusive')))}"
            f"</span> {escape(str(hypothesis.get('statement', '')))}"
            + (f"<div class=step-detail>{escape(reasoning)}</div>" if reasoning else "")
            + "</li>"
        )
    return (
        "<h2>Raised, not settled</h2>"
        "<p class=meta>Considered during the investigation and left undecided. Listed because "
        "an answer's limits are part of the answer.</p>"
        f'<ul class="rows">{"".join(items)}</ul>'
    )


def _removed_html(gate: list, verifier: list) -> str:
    """The claims that were cut, shown rather than counted.

    The count was already on the page and was the wrong unit. "9 claims were removed" reads as
    either an accusation or a boast depending on the reader's mood; the claims themselves read
    as a system that checks its work, and let a reader judge whether the right ones went.

    Collapsed by default, in a `<details>`, because on a report where nothing was cut this
    should take no space and on the one where nine were it should not bury the answer. That is
    an HTML element, not a script -- the page still has no JavaScript.

    Struck through rather than reworded: the exact text that was drafted is the point. A
    paraphrase of a rejected claim is a new claim, and nothing checked it.
    """
    entries = [("citation gate", r) for r in gate or []] + [
        ("verification", r) for r in verifier or []
    ]
    if not entries:
        return ""
    items = []
    for stage, rejection in entries:
        if not isinstance(rejection, dict):
            continue
        text = str(rejection.get("text") or "").strip()
        detail = str(rejection.get("detail") or rejection.get("reason") or "")
        location = str(rejection.get("location") or "")
        items.append(
            f"<li><span class=struck>{escape(text) or '(no text recorded)'}</span>"
            f"<div class=step-detail>removed by {escape(stage)}"
            + (f" at {escape(location)}" if location else "")
            + (f" — {escape(detail)}" if detail else "")
            + "</div></li>"
        )
    if not items:
        return ""
    return (
        "<details><summary>"
        f"{len(items)} claim(s) were removed before this report was shown — see what they said"
        "</summary>"
        f"<ul class=removed>{''.join(items)}</ul></details>"
    )


def _params_html(params: object) -> str:
    if not isinstance(params, dict) or not params:
        return ""
    # Truncated per value. A parameter can carry a long query string, and a trace table whose
    # rows wrap over four lines each stops being scannable, which is the only thing it is for.
    shown = ", ".join(f"{key}={str(value)[:40]}" for key, value in list(params.items())[:4])
    return f"<code>{escape(shown)}</code>"


def _when(moment: datetime | None) -> str:
    return moment.strftime("%Y-%m-%d %H:%M") if moment else ""


def _time(moment: datetime | None) -> str:
    return moment.strftime("%H:%M:%S") if moment else ""
