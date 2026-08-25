"""Inter-service contracts.

These schemas *are* the service boundary. Services never import each other — the
gateway and the workers communicate only by exchanging these messages over the
queue, so this module is the one place a breaking change between services can
happen, and the one place to look when diagnosing one.

Rules:
  - Every message carries tenant_id. A worker must never infer tenancy from
    ambient state.
  - Every message carries a schema_version. Workers reject versions they do not
    understand rather than guessing, so a half-deployed fleet fails loudly.
  - Payloads are plain JSON. No pickled Python crosses the queue.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1


class UnsupportedSchemaVersion(Exception):
    """Raised when a consumer receives a message version it cannot handle."""


class _Message(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = SCHEMA_VERSION
    tenant_id: uuid.UUID
    # Propagated across every hop so one investigation can be traced end to end
    # through gateway logs, worker logs and tool-call audit rows.
    correlation_id: uuid.UUID = Field(default_factory=uuid.uuid4)

    def require_supported(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise UnsupportedSchemaVersion(
                f"{type(self).__name__} v{self.schema_version}; "
                f"this service speaks v{SCHEMA_VERSION}"
            )


# --------------------------------------------------- gateway -> investigation worker


class RunInvestigation(_Message):
    """Ask the investigation worker to run one investigation.

    Deliberately minimal: the worker re-reads the investigation row rather than
    trusting a denormalized copy, so a redelivered message cannot resurrect stale
    parameters.
    """

    investigation_id: uuid.UUID
    requested_by_user_id: uuid.UUID | None = None


class CancelInvestigation(_Message):
    investigation_id: uuid.UUID
    reason: str = "cancelled by user"


# --------------------------------------------------- investigation worker -> gateway


class InvestigationProgress(_Message):
    """Streamed to the gateway so the UI can show the loop thinking.

    Progress is advisory. Losing one of these must never corrupt an
    investigation — the authoritative state is the Postgres row.
    """

    investigation_id: uuid.UUID
    status: str
    step: int
    note: str | None = None
    at: datetime | None = None

    #: Position in this investigation's progress stream, starting at 1.
    #:
    #: Added before anything consumed these, on a finding from indexing the OpenHands
    #: frontend (ADR 0002, round two). Their conversation store dedups incoming events by id
    #: and skips side-effects for ones it has already seen, with a comment citing their own
    #: issue: *"a reconnect replays the backlog from a stale anchor"*.
    #:
    #: The original design here was for a terminal, which prints each line as it arrives and
    #: never reconnects. A UI is a different consumer: it drops out, comes back, and receives
    #: some frames twice and misses others entirely. Without a sequence it cannot tell those
    #: apart — two identical "step 5 calling github__commits" lines are indistinguishable
    #: from one call reported twice, and a gap is invisible.
    #:
    #: Monotonic per investigation and never reused, so a consumer can deduplicate on
    #: `(investigation_id, sequence)` and detect a gap by arithmetic rather than by guessing.
    sequence: int = 0


# --------------------------------------------------- scheduler -> ingest worker


class SyncScope(enum.StrEnum):
    """How much history to pull.

    INCREMENTAL is the nightly default; BACKFILL is for onboarding a new tenant
    or recovering from a schema change.
    """

    INCREMENTAL = "incremental"
    BACKFILL = "backfill"


class SyncConnector(_Message):
    """Nightly sync of one connector for one tenant.

    One connector per message rather than one tenant per message: a HubSpot
    outage should not block the GA4 sync, and retries stay narrow.
    """

    provider: str
    credential_label: str = "default"
    scope: SyncScope = SyncScope.INCREMENTAL
    # Absolute, never relative. A retry hours later must cover the same window as
    # the original attempt.
    since: datetime | None = None
    until: datetime | None = None


class SyncResult(_Message):
    provider: str
    credential_label: str = "default"
    succeeded: bool
    nodes_written: int = 0
    edges_written: int = 0
    metric_points_written: int = 0
    error: str | None = None
    watermark: datetime | None = None


# --------------------------------------------------- queue routing

# Queue names are declared here, with the contracts, so a producer and its
# consumer cannot drift apart silently.
QUEUE_INVESTIGATION = "cortex.investigation"
QUEUE_INGEST = "cortex.ingest"
QUEUE_EVENTS = "cortex.events"
