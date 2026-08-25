"""Celery application shared by the worker services.

Each worker process consumes exactly one queue, so investigation load and ingest
load scale and fail independently — an ingest backlog must never delay a user
waiting on an answer.

JSON-only serialization: a pickled payload on the queue would let a compromised
producer execute code in a worker, and it would also couple the two services'
Python versions.
"""

from __future__ import annotations

from celery import Celery

from cortex.config.settings import get_settings
from cortex.contracts.messages import QUEUE_EVENTS, QUEUE_INGEST, QUEUE_INVESTIGATION


def make_celery(name: str) -> Celery:
    settings = get_settings()
    app = Celery(name, broker=settings.redis_url, backend=settings.redis_url)
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        # Acknowledge only after completion so a worker killed mid-investigation
        # results in a redelivery rather than a silently dropped request.
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_reject_on_worker_lost=True,
        # A hung third-party API must not pin a worker slot forever.
        task_soft_time_limit=settings.max_investigation_seconds,
        task_time_limit=settings.max_investigation_seconds + 60,
        task_routes={
            "cortex.investigation.*": {"queue": QUEUE_INVESTIGATION},
            "cortex.ingest.*": {"queue": QUEUE_INGEST},
            # Declared even though no worker consumes this queue yet: the UI is its
            # consumer, and routing belongs beside the other two rather than only in the
            # producer's call site.
            "cortex.events.*": {"queue": QUEUE_EVENTS},
        },
    )
    return app
