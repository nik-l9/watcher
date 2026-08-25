"""Turning a Slack mention into a queued investigation.

Shared by both transports, and that is the point of the module existing.

Slack offers two ways to receive events and an app may use exactly one: an HTTP **Request URL**,
which needs a public HTTPS endpoint and authenticates each delivery by HMAC signature; or **Socket
Mode**, an outbound WebSocket authenticated once by an app-level token. Cortex supports both —
`services/gateway/slack_events.py` and `services/slack_socket/` — because the trade is real: a
hosted deployment already has a public URL and should use it, while somebody who has cloned this
repo to run against their own API keys should not have to acquire one.

**What must not differ between them is the decision.** Which tenant a message reads, what counts
as a question, where the answer goes, and what gets enqueued are all decided here, once. A second
copy of that logic is a second chance to answer into the wrong channel or read the wrong tenant's
data, and the two copies would drift in precisely the way that is hardest to notice — a fix
applied to the transport somebody was testing that day.

**Authentication is *not* here, because it genuinely differs.** The HTTP path verifies an HMAC
over the raw bytes before anything parses them; Socket Mode has no signature at all, because the
socket itself was authenticated when it was opened and Slack is the only party that can write to
it. Putting a signature check in this module would either be dead code on one path or a false
reassurance on the other. Each transport authenticates in its own way, and only a payload that
has already been authenticated reaches this function.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import Investigation, Tenant
from cortex.db.titles import title_for

#: Strips the leading `<@U123>` a mention carries. Without it every question would begin with a
#: user id, which would reach the title, the drafting prompt and the report's `question` field.
_MENTION = re.compile(r"<@[A-Z0-9]+>")

#: Shortest question worth investigating. Matches `AskRequest.question`'s own floor so the entry
#: points cannot disagree about what counts as a question.
_MIN_QUESTION = 8


class Outcome:
    """Why a payload did or did not become an investigation.

    Returned rather than raised, and stringly-typed on purpose: every caller's response to every
    outcome is the same — acknowledge Slack and move on. An exception would tempt a transport into
    a 500, and Slack disables a subscription that keeps erroring, so a configuration problem would
    escalate itself into an outage.
    """

    QUEUED = "queued"
    NOT_A_MENTION = "ignored: not an app_mention"
    FROM_A_BOT = "ignored: posted by a bot"
    UNKNOWN_WORKSPACE = "ignored: no active tenant claims this workspace"
    TOO_SHORT = "ignored: no question in the mention"
    ALREADY_SEEN = "ignored: this event was already investigated"


async def enqueue_mention(
    session: AsyncSession,
    payload: dict[str, Any],
    *,
    send_task: Any,
) -> tuple[str, uuid.UUID | None]:
    """Decide what an already-authenticated Slack payload means, and queue it if it is a question.

    Returns the outcome and the investigation id when one was created. `send_task` is injected
    rather than imported so this stays testable without a broker and so neither service has to
    reach into the other's Celery producer.
    """
    event = payload.get("event")
    if not isinstance(event, dict) or event.get("type") != "app_mention":
        # Message edits, reactions, joins. Acknowledged and ignored: subscribing narrowly is the
        # alternative, but an unexpected event type must never raise.
        return Outcome.NOT_A_MENTION, None

    # A bot's own message, including ours. Without this an answer that happened to mention the
    # app would start another investigation, and that loop bills for itself.
    if event.get("bot_id") or event.get("subtype") == "bot_message":
        return Outcome.FROM_A_BOT, None

    question = _MENTION.sub("", str(event.get("text", ""))).strip()
    channel = str(event.get("channel", ""))
    # Answered in the thread it was asked in: `thread_ts` when the mention is already inside a
    # thread, otherwise the message's own `ts`, which starts one -- so an answer never lands as a
    # loose channel message beside an unrelated conversation.
    thread_ts = str(event.get("thread_ts") or event.get("ts") or "")

    # **The only thing that decides whose data is read.** Taken from the envelope, never from the
    # message text: a user typing a tenant slug into a question changes nothing. F-01's lesson,
    # applied before F-01 can happen here.
    team_id = str(payload.get("team_id") or event.get("team") or "")
    tenant_id = await session.scalar(
        select(Tenant.id).where(
            Tenant.slack_team_id == team_id,
            # A suspended tenant stops answering. Checked here rather than left to the worker,
            # which would spend the tokens first.
            Tenant.is_active.is_(True),
        )
    )
    if not team_id or tenant_id is None:
        return Outcome.UNKNOWN_WORKSPACE, None

    if len(question) < _MIN_QUESTION or not channel:
        return Outcome.TOO_SHORT, None

    row = Investigation(
        tenant_id=tenant_id,
        question=question,
        title=title_for(question),
        # Where the answer goes. Taken from the event, so even a forged one could not redirect an
        # answer somewhere the forger can read.
        notify={"kind": "slack", "channel": channel, "thread_ts": thread_ts},
        # Slack's own id for this event, stable across redeliveries. See below.
        source_event_id=str(payload.get("event_id") or "") or None,
    )
    session.add(row)
    try:
        # Flushed before enqueueing, so the row exists by the time a worker picks the message up
        # -- and so a duplicate is refused *here*, by the unique constraint, rather than after a
        # second investigation has already been billed.
        await session.flush()
    except IntegrityError:
        # **Decided by the database, not by a prior read.** A select-then-insert races itself:
        # two deliveries arriving together both see no existing row and both insert. Letting the
        # constraint decide makes the second one lose, always.
        #
        # One Slack mention produced two investigations this way, and because the model is not
        # deterministic the two answers disagreed with each other in the same thread. Duplicate
        # work is a cost; two contradictory public answers to one question is the product
        # failing at the thing it promises.
        await session.rollback()
        return Outcome.ALREADY_SEEN, None

    send_task(tenant_id=tenant_id, investigation_id=row.id)
    return Outcome.QUEUED, row.id
