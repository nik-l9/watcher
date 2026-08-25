"""Slack Socket Mode service — the inbound path that needs no public URL.

Thin on purpose, like the workers: the decision about what a mention means lives in
`cortex.inbound.slack` and the connection policy lives in `cortex.inbound.socket_mode`. What
belongs here is the wiring between them, and nothing else.

**One replica, like the scheduler.** More than one is not *wrong* — Slack delivers each event to
exactly one open connection, so duplicates do not arise — but a second replica buys nothing and
doubles the number of sockets an operator has to reason about when something is not arriving.

**It is optional.** A deployment with a public URL should use the webhook, which is one fewer
moving part and authenticates every delivery independently. This exists for the deployment that
has no public URL, which includes anybody who has cloned the repo to run it against their own
keys. Without `CORTEX_SLACK_APP_TOKEN` the service exits cleanly rather than crash-looping, so it
can sit in the compose file unused.
"""

from __future__ import annotations

import asyncio
import signal
import uuid
from typing import Any

import structlog

from cortex.config.settings import get_settings
from cortex.contracts.messages import QUEUE_INVESTIGATION, RunInvestigation
from cortex.inbound.slack import Outcome, enqueue_mention
from cortex.inbound.socket_mode import run_socket_mode
from cortex.runtime.celery_app import make_celery
from cortex.runtime.resources import open_resources

SERVICE_NAME = "slack-socket"

log = structlog.get_logger(SERVICE_NAME)

#: A producer, not a worker: this service enqueues and never consumes.
_producer = make_celery(SERVICE_NAME)


async def main() -> int:
    settings = get_settings()
    token = settings.slack_app_token
    if not token:
        # Not an error. The webhook transport may be the one in use, and a service that
        # crash-loops on an absent optional setting is worse than one that says why and stops.
        log.info("slack.socket.not_configured", detail="CORTEX_SLACK_APP_TOKEN is unset")
        return 0

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        # Handled so a shutdown closes the socket rather than being killed mid-frame, and so a
        # backoff sleep does not hold the container open for a minute.
        loop.add_signal_handler(sig, stop.set)

    async with open_resources() as resources:

        async def _handle(payload: dict[str, Any]) -> None:
            # A session per event, opened here rather than held for the life of the socket: a
            # connection kept open for hours is a connection that dies quietly, and the first
            # symptom would be a dropped question.
            async with resources.session() as session:
                outcome, investigation_id = await enqueue_mention(
                    session, payload, send_task=_enqueue
                )
                if outcome == Outcome.QUEUED:
                    await session.commit()
                    log.info("slack.socket.queued", investigation_id=str(investigation_id))
                else:
                    # Nothing was written, but roll back explicitly rather than relying on the
                    # context manager's default: an implicit commit of a half-decided event is
                    # the kind of thing that only shows up under load.
                    await session.rollback()
                    log.info("slack.socket.ignored", reason=outcome)

        log.info("slack.socket.starting")
        await run_socket_mode(token, handle=_handle, stop=stop)

    log.info("slack.socket.stopped")
    return 0


def _enqueue(*, tenant_id: uuid.UUID, investigation_id: uuid.UUID) -> None:
    """Hand the work to the investigation queue — the same message the HTTP path sends."""
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


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
