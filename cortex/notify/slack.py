"""Delivering an answer back into Slack.

A question asked in Slack has to be answered in Slack. Nobody returns to a web page to collect a
result they asked for in a thread — which is the whole reason the Slack entry point exists rather
than the report view being enough.

## What gets posted, and what deliberately does not

The **executive summary, the confidence, and a link** to the full report. Not the findings, not
the hypotheses, not the risks. A Slack message is read in a scroll, and a report that was already
too long for a page is unreadable in a channel — the shape work in `cortex/reports/shape.py`
exists because a yes/no question came back as a page.

**Citations survive**, rendered as the capability that produced each observation, because a
grounded answer whose grounding is stripped for the channel is just an assertion. What cannot
survive is clickability: Slack has nowhere to jump to, so the link to the report page is how a
reader reaches the evidence.

**Charts go as PNG**, which is why `cortex/reports/png.py` exists — Slack cannot render inline
SVG. One chart, the first, because a thread reply with four images is a wall.

## Delivery never fails an investigation

Every failure here is swallowed and reported as a return value. The investigation is complete,
stored and correct whether or not anybody was told about it; raising would mark a finished
investigation failed because a chat API was briefly unavailable, which is a worse outcome than a
missing message. The audit trail is the durable record, and the report page always has the answer.
"""

from __future__ import annotations

import uuid
from typing import Any
from urllib.parse import quote

import httpx

#: Slack's own ceiling on a `text` field is 40,000 characters, but the practical limit is a
#: reader's patience. An executive summary is one to three claims; this truncates rather than
#: letting a pathological report fill a channel.
MAX_TEXT_CHARS = 2800

#: Chart images to attach. One: a thread reply carrying four PNGs is a wall rather than an answer,
#: and the first chart is the one the drafting pass considered most relevant.
MAX_CHARTS = 1

_POST_MESSAGE = "https://slack.com/api/chat.postMessage"
_UPLOAD_URL = "https://slack.com/api/files.getUploadURLExternal"
_UPLOAD_COMPLETE = "https://slack.com/api/files.completeUploadExternal"

#: Short, because a chat post sits on the critical path of nothing. If Slack is slow the answer is
#: still safe in Postgres, so waiting is pure downside.
_TIMEOUT = httpx.Timeout(connect=10.0, read=20.0, write=20.0, pool=5.0)


def render_for_slack(
    *,
    question: str,
    report: dict[str, Any],
    report_url: str | None = None,
    seconds: float | None = None,
) -> str:
    """The message body: the answer, its citations, and where to read the rest.

    Takes the stored report body as a dict rather than an `InvestigationReport`, because the
    worker has already serialised it and re-parsing it here would create a second place where the
    delivered text could disagree with the stored one.
    """
    lines = [f"*{_escape(question.strip())}*"]

    sources = {
        str(source.get("evidence_id")): source
        for source in report.get("sources", [])
        if isinstance(source, dict)
    }

    summary = [claim for claim in report.get("executive_summary", []) if isinstance(claim, dict)]
    if not summary:
        # Cannot happen for a delivered report: the gate rejects one with an empty summary. Said
        # plainly rather than posting an empty message if it ever does.
        lines.append("_The report contains no summary._")
    for claim in summary:
        text = _escape(str(claim.get("text", "")).strip())
        lines.append(f"• {text}{_cites(claim.get('evidence_ids', []), sources)}")

    footer: list[str] = []
    confidence = report.get("confidence")
    if isinstance(confidence, str):
        footer.append(f"confidence: {confidence.replace('_', ' ')}")
    if sources:
        footer.append(f"{len(sources)} observation(s)")
    if seconds is not None:
        footer.append(f"{seconds:.0f}s")
    if footer:
        lines.append("")
        lines.append("_" + " · ".join(footer) + "_")

    # The link is how a reader reaches the evidence, since Slack cannot make a citation
    # clickable. Without it the citations name a source nobody can open.
    if report_url:
        lines.append(f"<{report_url}|Full report, charts and sources>")

    body = "\n".join(lines)
    return body if len(body) <= MAX_TEXT_CHARS else body[: MAX_TEXT_CHARS - 1] + "…"


def _cites(ids: object, sources: dict[str, dict]) -> str:
    """Citations as the capability that produced each observation.

    A grounded answer whose grounding is stripped for the channel is an assertion, so the
    capability names travel even though Slack cannot link to the rows.
    """
    if not isinstance(ids, list) or not ids:
        return ""
    labels: list[str] = []
    for raw in ids:
        source = sources.get(str(raw))
        if source:
            labels.append(f"{source.get('tool_name', '')}.{source.get('capability', '')}")
        else:
            labels.append(str(raw)[:8])
    # Deduplicated: three claims citing the same capability twice each would otherwise read as
    # six sources.
    unique = list(dict.fromkeys(labels))
    return "  _[" + ", ".join(unique) + "]_"


def _escape(text: str) -> str:
    """Slack's three mandatory escapes, and only those.

    `&`, `<` and `>` are what Slack's own documentation requires escaping in message text.
    Escaping more would put backslashes in front of characters an analyst's prose legitimately
    uses — a percentage or an underscore in an event name.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


async def post_answer(
    *,
    token: str,
    channel: str,
    thread_ts: str | None,
    text: str,
    chart_png: bytes | None = None,
    chart_title: str = "chart",
    client: httpx.AsyncClient | None = None,
) -> tuple[bool, str]:
    """Post one answer. Returns `(delivered, detail)` and never raises.

    `detail` carries the reason on failure so the worker can log it. Slack reports application
    errors inside an HTTP 200 with `ok: false`, so the status code alone is not the outcome —
    the same shape as the provider's SSE errors, and the same trap.
    """
    owned = client is None
    http = client or httpx.AsyncClient(timeout=_TIMEOUT)
    try:
        payload: dict[str, Any] = {"channel": channel, "text": text}
        if thread_ts:
            payload["thread_ts"] = thread_ts
        response = await http.post(
            _POST_MESSAGE,
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        body = _body(response)
        if not body.get("ok"):
            # The token is never included in the reason, and Slack's own error strings do not
            # echo it. "invalid_auth" is actionable without quoting a secret.
            return False, f"chat.postMessage: {body.get('error', response.status_code)}"

        if chart_png:
            uploaded, detail = await _upload_chart(
                http,
                token=token,
                channel=channel,
                thread_ts=thread_ts or body.get("ts"),
                png=chart_png,
                title=chart_title,
            )
            if not uploaded:
                # The answer is already posted, so a failed chart is a partial success rather
                # than a failure. Reported, not raised.
                return True, f"posted without chart: {detail}"
        return True, "posted"
    except httpx.HTTPError as exc:
        return False, f"transport: {type(exc).__name__}"
    finally:
        if owned:
            await http.aclose()


async def _upload_chart(
    http: httpx.AsyncClient,
    *,
    token: str,
    channel: str,
    thread_ts: str | None,
    png: bytes,
    title: str,
) -> tuple[bool, str]:
    """Slack's three-step external upload: get a URL, PUT the bytes, complete.

    `files.upload` was the one-step version and is deprecated; this is the replacement. Written
    out rather than hidden behind a helper because each step fails differently and a partial
    upload leaves a file with no message attached to it.
    """
    filename = "chart.png"
    reserve = await http.post(
        _UPLOAD_URL,
        data={"filename": filename, "length": str(len(png))},
        headers={"Authorization": f"Bearer {token}"},
    )
    body = _body(reserve)
    if not body.get("ok") or not body.get("upload_url") or not body.get("file_id"):
        return False, f"getUploadURLExternal: {body.get('error', reserve.status_code)}"

    put = await http.put(
        str(body["upload_url"]),
        content=png,
        headers={"Content-Type": "image/png"},
    )
    if put.status_code >= 400:
        return False, f"upload: HTTP {put.status_code}"

    files = [{"id": body["file_id"], "title": title}]
    payload: dict[str, Any] = {"files": files, "channel_id": channel}
    if thread_ts:
        payload["thread_ts"] = thread_ts
    complete = await http.post(
        _UPLOAD_COMPLETE,
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )
    done = _body(complete)
    if not done.get("ok"):
        return False, f"completeUploadExternal: {done.get('error', complete.status_code)}"
    return True, "uploaded"


def _body(response: httpx.Response) -> dict[str, Any]:
    """Slack's JSON body, defensively.

    Slack reports application errors inside an HTTP 200 with `ok: false`, so the body is the
    outcome rather than the status. A non-JSON body — an HTML error page from a proxy — must not
    raise inside a delivery path that is not allowed to fail.
    """
    try:
        parsed = response.json()
    except ValueError:
        return {"ok": False, "error": f"non-JSON response (HTTP {response.status_code})"}
    return parsed if isinstance(parsed, dict) else {"ok": False, "error": "unexpected body"}


def report_url(
    base_url: str | None, investigation_id: uuid.UUID, tenant_slug: str | None = None
) -> str | None:
    """Where to read the full report, or None when no public base URL is configured.

    None rather than a localhost link: a message telling a colleague to open `localhost:8000`
    is worse than a message with no link, because it looks like a broken feature rather than an
    unconfigured one.

    No query string. The report page resolves its own tenant from the investigation id, because
    that id is the only secret in the link and requiring the tenant beside it protected nothing
    while breaking every link that lost its query string on the way -- pasted, forwarded, or
    truncated by a chat client. `tenant_slug` is accepted and appended only when a caller asks
    for it explicitly, which the Slack path does not.
    """
    if not base_url:
        return None
    url = f"{base_url.rstrip('/')}/ui/investigations/{investigation_id}"
    return f"{url}?tenant={quote(tenant_slug)}" if tenant_slug else url
