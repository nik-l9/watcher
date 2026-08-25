"""Delivering a finished report to wherever it was asked for.

The worker calls this once, after the report is stored. Everything about it is shaped by one rule:
**delivery must never fail an investigation.**

The report is already committed, gated, verified and readable on its own page. If Slack is down,
or the bot token was revoked, or the channel was archived between question and answer, the right
outcome is a logged failure and a completed investigation — not a completed investigation marked
failed because a chat API had a bad minute. So every path here returns a string and raises
nothing, and the caller treats that string as a log line rather than a control-flow signal.

The credential is read from the vault per tenant, like every other credential. A Slack *bot*
token is a different secret from the *user* token the search capability uses, so it lives under
the same provider with the label `bot`: `Credential(provider=SLACK, label="bot")`. Nothing here
logs it, and the failure reasons Slack returns ("invalid_auth", "channel_not_found") are
actionable without quoting it.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.config.settings import get_settings
from cortex.db.models import Credential, CredentialProvider, Investigation, Report
from cortex.notify.slack import post_answer, render_for_slack, report_url
from cortex.reports.schema import ChartSpec
from cortex.security.vault import decrypt_credential
from cortex.tenancy.context import TenantContext

#: The credential label a Slack *bot* token is stored under, distinct from the `default` label the
#: user token for search uses. Posting and searching are different scopes, and in the deployment
#: this was built against they were different tokens.
BOT_LABEL = "bot"


async def deliver_report(
    session: AsyncSession,
    tenant: TenantContext,
    *,
    investigation_id: uuid.UUID,
) -> str:
    """Deliver one finished report, and report what happened as a string.

    Returns a short reason in every case — delivered, nothing to do, or why not. Never raises.
    """
    try:
        return await _deliver(session, tenant, investigation_id=investigation_id)
    except Exception as exc:  # noqa: BLE001 - the module docstring is this except clause
        # The type leads and the message is truncated: a delivery bug must be visible in a log
        # without a vendor's stack trace masquerading as a Cortex failure.
        return f"delivery failed: {type(exc).__name__}: {str(exc)[:200]}"


async def _deliver(
    session: AsyncSession,
    tenant: TenantContext,
    *,
    investigation_id: uuid.UUID,
) -> str:
    row = (
        await session.execute(
            select(Investigation).where(
                Investigation.id == investigation_id,
                # Tenant-scoped like every other read of this table, even though the worker
                # resolved the tenant already. One predicate is not a burden; a missing one is
                # F-01.
                Investigation.tenant_id == tenant.tenant_id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        return "no such investigation"

    target = _slack_target(row.notify)
    if target is None:
        # The common case: a question asked through the API or the CLI has nowhere to be
        # delivered, and that is not a defect.
        return "nothing to deliver to"
    channel, thread_ts = target

    report = (
        await session.execute(
            select(Report)
            .where(
                Report.investigation_id == investigation_id,
                Report.tenant_id == tenant.tenant_id,
            )
            .order_by(Report.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if report is None or not isinstance(report.body, dict):
        return "no report to deliver"

    token = await _bot_token(session, tenant)
    if token is None:
        # Named specifically, because this is the one failure an operator can fix and the one
        # most likely on a first deployment.
        return f"no Slack bot token for tenant {tenant.tenant_slug!r} (label {BOT_LABEL!r})"

    settings = get_settings()
    seconds = None
    if row.completed_at and row.created_at:
        seconds = (row.completed_at - row.created_at).total_seconds()

    text = render_for_slack(
        question=row.question,
        report=report.body,
        report_url=report_url(settings.public_base_url, investigation_id),
        seconds=seconds,
    )

    chart_png, chart_title = _first_chart(report.body)
    delivered, detail = await post_answer(
        token=token,
        channel=channel,
        thread_ts=thread_ts,
        text=text,
        chart_png=chart_png,
        chart_title=chart_title,
    )
    return detail if delivered else f"not delivered: {detail}"


def _slack_target(notify: Any) -> tuple[str, str | None] | None:
    """The channel and thread to answer in, or None.

    Read here and in the endpoint's own helper; both are three lines, and a shared import would
    make the gateway a dependency of the worker for the sake of a dict lookup.
    """
    if not isinstance(notify, dict) or notify.get("kind") != "slack":
        return None
    channel = notify.get("channel")
    if not isinstance(channel, str) or not channel:
        return None
    thread_ts = notify.get("thread_ts")
    return channel, thread_ts if isinstance(thread_ts, str) and thread_ts else None


async def _bot_token(session: AsyncSession, tenant: TenantContext) -> str | None:
    """The tenant's Slack bot token, decrypted, or None.

    Tenant-scoped by the same rule as `ToolExecutor._load_context`: a credential is never
    resolvable across tenants, and this is a second decryption path — which is exactly the shape
    F-01 warned about, so it carries the same predicate.
    """
    row = (
        await session.execute(
            select(Credential).where(
                Credential.tenant_id == tenant.tenant_id,
                Credential.provider == CredentialProvider.SLACK,
                Credential.label == BOT_LABEL,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        return None
    # `label` is part of the AAD, so it must match what sealed the credential. Omitting it
    # defaulted to "default" and every decryption failed with "credential failed authentication --
    # wrong master key, wrong tenant, or tampering", which is the vault telling the truth: the
    # authenticated data really did differ. Found on the first live Slack question, where the
    # investigation completed and the answer never left the building.
    return decrypt_credential(
        tenant.tenant_id,
        CredentialProvider.SLACK.value,
        row.wrapped_data_key,
        row.ciphertext,
        label=BOT_LABEL,
    )


def _first_chart(body: dict[str, Any]) -> tuple[bytes | None, str]:
    """The first chart as PNG, or nothing.

    One chart: a thread reply carrying four images is a wall rather than an answer. Rendering is
    attempted rather than assumed to work — `render_png` already returns a placeholder rather than
    raising, but a chart that fails to *validate* back into a `ChartSpec` is dropped silently
    here, because a missing picture is a far better outcome than an undelivered answer.
    """
    charts = body.get("charts")
    if not isinstance(charts, list) or not charts:
        return None, "chart"
    try:
        spec = ChartSpec.model_validate(charts[0])
    except Exception:  # noqa: BLE001 - see the docstring
        return None, "chart"

    # Imported here rather than at module scope: matplotlib pulls in a large dependency tree and
    # selects a backend on import, and a gateway process that never renders a chart should not pay
    # for either.
    from cortex.reports.png import render_png

    return render_png(spec), spec.title
