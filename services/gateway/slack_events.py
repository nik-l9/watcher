"""The Slack entry point.

A GTM team asks questions in Slack. Until this existed, using Cortex meant a terminal or a
localhost page with a tenant header — a demo, not a trial. This is the first surface a colleague
can use without being taught anything.

## It is unauthenticated, and that shapes everything here

Every other route is behind Clerk or a tenant header. This one is open to the internet, because
Slack has to reach it. Three consequences, enforced below rather than assumed:

  - **The signature is the authentication.** `verify_slack_request` runs on the raw body before
    anything parses it, and a failure is a 401 with no detail. Nothing downstream runs first.
  - **Nothing in the message body decides whose data is read.** The tenant comes from `team_id`
    matched against `tenants.slack_team_id`, and an unclaimed workspace is refused. A user typing
    a tenant slug into a message changes nothing — F-01's lesson, applied before F-01 can happen
    here.
  - **The reply target comes from the event, not the text.** A question asked in a channel is
    answered in that channel, so even a forged event could not redirect an answer somewhere the
    forger can read.

## Three seconds

Slack retries anything it does not see acknowledged within three seconds; an investigation takes
ninety. So this endpoint does what `POST /investigations` does — write the row, enqueue, return —
and the worker delivers into the thread when the report exists. Slack's own retries are ignored
by `X-Slack-Retry-Num`, because without that one flaky delivery becomes several identical
investigations and several bills.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, Header, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.config.settings import get_settings
from cortex.contracts.messages import QUEUE_INVESTIGATION, RunInvestigation
from cortex.inbound.slack import enqueue_mention
from cortex.security.slack_signing import SlackSignatureError, verify_slack_request
from services.gateway.deps import get_session
from services.gateway.investigations import _producer

router = APIRouter(tags=["slack"], include_in_schema=False)


def _enqueue(*, tenant_id: uuid.UUID, investigation_id: uuid.UUID) -> None:
    """Hand the work to the investigation queue.

    Injected into `enqueue_mention` rather than imported by it, so the shared decision logic does
    not depend on this service's Celery producer and can be tested without a broker.
    """
    _producer.send_task(
        "cortex.investigation.run",
        args=[
            RunInvestigation(
                tenant_id=tenant_id,
                investigation_id=investigation_id,
                requested_by_user_id=None,
            ).model_dump(mode="json")
        ],
        queue=QUEUE_INVESTIGATION,
    )


@router.post("/slack/events")
async def slack_events(
    request: Request,
    x_slack_signature: str | None = Header(default=None),
    x_slack_request_timestamp: str | None = Header(default=None),
    x_slack_retry_num: str | None = Header(default=None),
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Slack's Events API webhook: verify, resolve the tenant, enqueue, acknowledge."""
    settings = get_settings()

    # The raw bytes, before anything parses them. A re-serialised body will not verify, which is
    # the intended behaviour rather than a limitation -- the signature is over bytes.
    body = await request.body()
    try:
        verify_slack_request(
            body=body,
            timestamp=x_slack_request_timestamp,
            signature=x_slack_signature,
            signing_secret=settings.slack_signing_secret or "",
        )
    except SlackSignatureError:
        # No detail. The caller is unauthenticated, and naming which half of the scheme failed
        # tells an attacker which half to work on.
        return Response(status_code=status.HTTP_401_UNAUTHORIZED)

    payload = _parse(body)

    # Slack's handshake when the URL is first configured. Answered only after verification, so
    # the endpoint cannot be used as an unauthenticated echo.
    if payload.get("type") == "url_verification":
        return Response(content=str(payload.get("challenge", "")), media_type="text/plain")

    # A retry means Slack did not see our earlier acknowledgement. The work is already queued, so
    # running it again would spend a second investigation's tokens to produce the same answer
    # twice in the same thread.
    if x_slack_retry_num:
        return Response(status_code=status.HTTP_200_OK)

    # Everything after authentication is decided in one place, shared with the Socket Mode
    # transport. A second copy of this logic would be a second chance to answer into the wrong
    # channel or read the wrong tenant's data.
    await enqueue_mention(session, payload, send_task=_enqueue)
    return Response(status_code=status.HTTP_200_OK)


def _parse(body: bytes) -> dict[str, Any]:
    """Slack's JSON body, or an empty dict.

    Defensive because this runs on an internet-facing endpoint: a malformed body must be a
    quiet 200 rather than a 500 that teaches Slack to disable the subscription.
    """
    import json

    try:
        parsed = json.loads(body or b"{}")
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def notify_target(notify: dict[str, Any] | None) -> tuple[str, str | None] | None:
    """The Slack channel and thread to answer in, or None.

    Shared with the worker so the shape of the `notify` dict is read in exactly one place. A
    second reader would be a second chance to post an answer into the wrong channel.
    """
    if not isinstance(notify, dict) or notify.get("kind") != "slack":
        return None
    channel = notify.get("channel")
    if not isinstance(channel, str) or not channel:
        return None
    thread_ts = notify.get("thread_ts")
    return channel, thread_ts if isinstance(thread_ts, str) and thread_ts else None
