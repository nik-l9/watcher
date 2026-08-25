"""Postgres schema.

Holds tenant/app data, the immutable evidence store, and the audit trail. The
knowledge graph lives in FalkorDB and embeddings in Qdrant; this database is the
system of record for everything that must be transactional or auditable.

Every tenant-scoped table carries tenant_id as the first column of its indexes so
that scoping is cheap and obvious in query plans.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    type_annotation_map = {dict: JSON}


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


#: Wall clock at statement execution, for rows whose order within one transaction is a fact
#: somebody reads.
#:
#: `now()` in Postgres is the *transaction's* start time — verified against this database, not
#: recalled — so every row a transaction writes carries the same value to the microsecond. The
#: investigation loop writes a `tool_calls` and an `evidence` row per capability call and
#: flushes rather than commits, so with `now()` an entire investigation's audit trail claims
#: the calls happened simultaneously while `duration_ms` says each took a different length of
#: time.
#:
#: Used only where that matters. A tenant or a report row keeps `now()`: for those, transaction
#: time is the right answer, because the row *is* the transaction.
_CLOCK = text("clock_timestamp()")


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


# ---------------------------------------------------------------- tenancy


class Tenant(Base, TimestampMixin):
    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = _uuid_pk()
    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Physical graph name in FalkorDB and collection prefix in Qdrant. Assigned
    # once at creation and never derived ad hoc by callers — see
    # cortex.memory.naming for the single resolver.
    graph_name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    #: The Slack workspace this tenant answers questions for, if any.
    #:
    #: Unique, and it is the *only* thing that maps an inbound Slack event to a tenant. A Slack
    #: request is unauthenticated apart from its signature, so nothing in the message body may
    #: decide whose data gets read -- the workspace id is checked against this column and a
    #: workspace nobody has claimed is refused. A shared column would let one workspace address
    #: another tenant's evidence, which is F-01 arriving through a new door.
    slack_team_id: Mapped[str | None] = mapped_column(String(32), unique=True)

    # passive_deletes defers to the DB's ON DELETE CASCADE. Without it SQLAlchemy
    # tries to NULL out users.tenant_id first, which violates NOT NULL — the ORM
    # would fight the schema on every tenant offboarding.
    users: Mapped[list[User]] = relationship(
        back_populates="tenant", passive_deletes=True, cascade="all, delete"
    )


class User(Base, TimestampMixin):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("tenant_id", "email", name="uq_users_tenant_email"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    # Clerk subject id.
    external_id: Mapped[str | None] = mapped_column(String(255), unique=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    role: Mapped[str] = mapped_column(String(32), default="member", nullable=False)

    tenant: Mapped[Tenant] = relationship(back_populates="users")


# ---------------------------------------------------------------- credentials


class CredentialProvider(enum.StrEnum):
    GA4 = "ga4"
    BIGQUERY = "bigquery"
    HUBSPOT = "hubspot"
    GITHUB = "github"
    SLACK = "slack"
    POSTHOG = "posthog"
    MIXPANEL = "mixpanel"
    AMPLITUDE = "amplitude"
    #: An MCP server, which is a transport rather than a vendor. One provider covers all of them;
    #: which server a credential belongs to is the credential's `label`.
    MCP = "mcp"


class Credential(Base, TimestampMixin):
    """Encrypted third-party credentials.

    V1 is bring-your-own: an admin pastes a service-account JSON or private-app
    token. Envelope encrypted — the plaintext never touches this table, and the
    per-tenant data key is itself wrapped by the vault master key.
    """

    __tablename__ = "credentials"
    __table_args__ = (
        UniqueConstraint("tenant_id", "provider", "label", name="uq_credentials_tenant_provider"),
        Index("ix_credentials_tenant", "tenant_id", "provider"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[CredentialProvider] = mapped_column(
        Enum(CredentialProvider, name="credential_provider"), nullable=False
    )
    label: Mapped[str] = mapped_column(String(64), default="default", nullable=False)

    wrapped_data_key: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    # Non-secret connector metadata: GA4 property id, BigQuery project, Slack
    # workspace id. Safe to display and to log.
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)

    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_sync_error: Mapped[str | None] = mapped_column(Text)


# ---------------------------------------------------------------- investigations


class InvestigationStatus(enum.StrEnum):
    QUEUED = "queued"
    PLANNING = "planning"
    INVESTIGATING = "investigating"
    SYNTHESIZING = "synthesizing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Investigation(Base, TimestampMixin):
    __tablename__ = "investigations"
    __table_args__ = (
        Index("ix_investigations_tenant_created", "tenant_id", "created_at"),
        # Every read of parent_id is tenant-scoped: the column proves a row exists and says
        # nothing about who owns it (F-01).
        Index("ix_investigations_tenant_parent", "tenant_id", "parent_id"),
        # One investigation per inbound event, enforced by the database rather than by a check
        # that races itself. Two deliveries of one Slack mention produced two full
        # investigations, two bills and two answers in the same thread -- and, because the model
        # is not deterministic, two answers that *disagreed*.
        UniqueConstraint("tenant_id", "source_event_id", name="uq_investigations_source_event"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )

    # Which specialist ran this. V1 is always "gtm_data_analyst".
    employee: Mapped[str] = mapped_column(String(64), default="gtm_data_analyst", nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    #: The investigation this one follows up on, if any.
    #:
    #: A GTM analyst is asked follow-ups: "why did signups fall?" is answered and the next
    #: thing said is "and what about mobile?". Without this, that second question started an
    #: investigation that had never seen the first -- re-gathering the same evidence, able to
    #: reach a different answer from the same data, and billing for both.
    #:
    #: Load-bearing for grounding, not just for display: it defines which evidence a report may
    #: cite (see cortex/db/threads.py). SET NULL rather than CASCADE, because deleting a parent
    #: should orphan the thread rather than delete a follow-up somebody may still be reading --
    #: and the parent's evidence goes with the parent either way, so a broken link narrows the
    #: citable scope rather than widening it.
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="SET NULL")
    )
    #: A short label for lists and tabs. Derived from the question at creation rather than
    #: generated, and stored rather than computed on read so it stays stable if the
    #: derivation ever changes — a list whose rows rename themselves on deploy is worse
    #: than one with slightly worse titles.
    title: Mapped[str | None] = mapped_column(String(96))
    status: Mapped[InvestigationStatus] = mapped_column(
        Enum(InvestigationStatus, name="investigation_status"),
        default=InvestigationStatus.QUEUED,
        nullable=False,
    )

    #: Where to deliver the answer when it is ready, if anywhere.
    #:
    #: `{"kind": "slack", "channel": "C…", "thread_ts": "…"}` for a question asked in Slack. The
    #: worker reads it after the report is stored, so a question asked in a thread is answered in
    #: that thread rather than requiring anyone to come back to a web page.
    #:
    #: A dict rather than columns because the second channel is a matter of when, not if, and
    #: because the reply target belongs to whoever asked rather than to the investigation -- the
    #: investigation is complete and correct whether or not anyone is told about it. Delivery
    #: failure is therefore never allowed to fail the investigation.
    notify: Mapped[dict | None] = mapped_column(JSONB)

    #: The upstream event that caused this investigation, when there was one.
    #:
    #: Slack's `event_id` is stable across redeliveries, which is what makes it usable as an
    #: idempotency key. The webhook transport suppressed duplicates with the `X-Slack-Retry-Num`
    #: header; Socket Mode has no equivalent, because a redelivery there is simply the same
    #: envelope arriving again -- so the first Socket Mode deployment reintroduced a bug the
    #: webhook had solved, in a form the webhook's fix could not catch.
    #:
    #: Nullable, because a question asked through the API or the CLI has no upstream event. The
    #: unique constraint is therefore on `(tenant_id, source_event_id)`, and Postgres treats
    #: NULLs as distinct, so CLI questions never collide with each other.
    source_event_id: Mapped[str | None] = mapped_column(String(128))

    # Working state of the hypothesis-driven loop: each entry records the
    # hypothesis, its supporting and contradicting evidence ids, and a verdict.
    hypotheses: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)

    steps_used: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # The breakdown, because the total cannot be priced. Input and output differ by 5x on
    # every model we run, and a cache read costs a fraction of a fresh input token — so a
    # bill computed from `tokens_used` alone is wrong by an unknown multiple that depends on
    # the mix. `Usage`'s own docstring said as much; the row discarded the fields anyway.
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cache_write_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    #: Which model produced it. Pricing is per model, so a spend figure without this is a
    #: guess — and it stops a model change silently repricing history.
    model: Mapped[str | None] = mapped_column(String(64))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)


class Evidence(Base):
    """Immutable record of one observation returned by one tool call.

    This table is append-only. Reports cite evidence by id, and the grounding
    gate resolves those ids against this table before anything renders — which is
    what makes "never hallucinate" a structural property rather than a prompt.
    """

    __tablename__ = "evidence"
    __table_args__ = (
        Index("ix_evidence_tenant_investigation", "tenant_id", "investigation_id"),
        Index("ix_evidence_tenant_hash", "tenant_id", "payload_hash"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False
    )

    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    capability: Mapped[str] = mapped_column(String(128), nullable=False)
    params: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # sha256 of the canonicalized payload. Lets the verifier prove a citation
    # refers to data that was actually observed and not since mutated.
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # Human-facing pointer back to the source: a permalink, a SQL string, a
    # commit sha. Rendered in the report's Sources section.
    source_ref: Mapped[str | None] = mapped_column(Text)

    # True when the underlying data came from a nightly sync rather than a live
    # call. Reports must disclose staleness in their confidence section.
    from_cache: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # clock_timestamp(), not now(): now() is the *transaction's* start time in Postgres, so
    # every observation an investigation gathers would be stamped with the instant its
    # transaction opened. That is wrong beyond display -- retention deletes by this column,
    # and a report cites it as when the observation was made.
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=_CLOCK, nullable=False
    )


class ToolCall(Base):
    """Audit row for every tool invocation, successful or not.

    Separate from Evidence on purpose: a failed or empty call produces no
    evidence but must still be auditable.
    """

    __tablename__ = "tool_calls"
    __table_args__ = (Index("ix_tool_calls_tenant_created", "tenant_id", "created_at"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    investigation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="SET NULL")
    )
    evidence_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("evidence.id", ondelete="SET NULL")
    )

    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    capability: Mapped[str] = mapped_column(String(128), nullable=False)
    read_only: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    # Params only. Never the response body — that belongs in Evidence, and
    # duplicating it here would double the PII surface.
    params: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    succeeded: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    # clock_timestamp(), for the same reason as Evidence.observed_at above: the loop writes
    # every call in one transaction, and with now() the whole audit trail claims the calls
    # happened simultaneously -- while duration_ms says each took a different length of time.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=_CLOCK, nullable=False
    )


class Report(Base, TimestampMixin):
    """Rendered investigation output, post-gate and post-verifier."""

    __tablename__ = "reports"
    __table_args__ = (Index("ix_reports_tenant_created", "tenant_id", "created_at"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False
    )

    # Serialized cortex.reports.schema.InvestigationReport.
    body: Mapped[dict] = mapped_column(JSONB, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)

    # Claims the grounding gate removed and claims the verifier judged
    # unsupported. Kept for the eval suite: hallucination count must be zero.
    gate_rejections: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    verifier_rejections: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)


# ---------------------------------------------------------------- ingest


class SyncStatus(enum.StrEnum):
    OK = "ok"
    FAILED = "failed"
    #: Ran, but the upstream refused part of what was asked for. Distinguished from
    #: FAILED because partial data is usable and total absence is not, and from OK
    #: because a report drawing on it must disclose the gap.
    PARTIAL = "partial"


class SyncState(Base, TimestampMixin):
    """Where a connector's nightly sync got to, and whether it worked.

    One row per (tenant, provider, label, stream). The **stream** is what makes the
    watermark correct: a GitHub sync pulls commits, deployments and issues, and they
    advance independently — one watermark for the whole provider would let a failed
    issues pull rewind the commits cursor, or a successful commits pull mark issues as
    current when it never ran.

    This table is deliberately both the watermark *and* the health record. They were
    separate in an earlier sketch, and that is a bug waiting to happen: a watermark that
    advances while the health row says "failed" describes a sync that both did and did not
    happen, and there is no way to tell afterwards which is true. Written together in one
    transaction, they cannot disagree.

    `Credential.last_sync_at` predates this and stays as a coarse, per-credential summary
    for the connector-health UI. This table is the authoritative per-stream cursor.
    """

    __tablename__ = "sync_state"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "provider", "label", "stream", name="uq_sync_state_tenant_stream"
        ),
        Index("ix_sync_state_tenant", "tenant_id", "provider"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[CredentialProvider] = mapped_column(
        Enum(CredentialProvider, name="credential_provider"), nullable=False
    )
    label: Mapped[str] = mapped_column(String(64), default="default", nullable=False)
    #: Which stream within the provider — "commits", "deployments", "messages".
    stream: Mapped[str] = mapped_column(String(64), nullable=False)

    #: The high-water mark: everything up to here is ingested. Advanced only on a
    #: successful pull, so a failure re-reads rather than skipping a window.
    watermark: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    status: Mapped[SyncStatus] = mapped_column(
        Enum(SyncStatus, name="sync_status"), default=SyncStatus.OK, nullable=False
    )
    #: Why it failed, kept because "the sync failed" is not actionable on its own.
    detail: Mapped[str | None] = mapped_column(Text)

    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    #: Last run that actually succeeded. The gap between this and `last_run_at` is what
    #: makes staleness visible: a connector failing every night for a week still has a
    #: recent `last_run_at`, and reporting that as health would be a lie of omission.
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    items_written: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class MetricPoint(Base):
    """One observation of one metric at one time, for one segment.

    Why Postgres and not only the graph: baselines are arithmetic over long series, and
    "is this week unusual" is a query over thousands of points. A graph traversal is the
    wrong shape for that, and the plan calls for these to be mirrored to DuckDB/Parquet
    for aggregation during an investigation.

    **Immutable, and keyed for idempotency.** A nightly sync re-reads an overlapping
    window on purpose (an upstream backfills late-arriving data), so the same point
    arrives more than once. The unique constraint makes the re-read a no-op rather than a
    duplicate, which is what keeps a re-run from doubling a metric.
    """

    __tablename__ = "metric_points"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "provider",
            "metric",
            "segment_key",
            "observed_at",
            name="uq_metric_points_identity",
        ),
        Index("ix_metric_points_lookup", "tenant_id", "metric", "observed_at"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[CredentialProvider] = mapped_column(
        Enum(CredentialProvider, name="credential_provider"), nullable=False
    )
    #: The metric's name as the source calls it: "sessions", "user signed up".
    metric: Mapped[str] = mapped_column(String(128), nullable=False)
    #: The segment this point describes, flattened to a stable string — "device=mobile",
    #: or "" for the total. Flattened rather than JSONB so it can participate in the
    #: uniqueness constraint: two JSONB objects with the same pairs in a different order
    #: are not equal, which would let the same segment be inserted twice.
    segment_key: Mapped[str] = mapped_column(String(256), default="", nullable=False)
    #: The same segment as structured data, for reading.
    segment: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class AuditLog(Base):
    """Tenant-visible security audit trail."""

    __tablename__ = "audit_log"
    __table_args__ = (Index("ix_audit_tenant_created", "tenant_id", "created_at"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    resource_type: Mapped[str | None] = mapped_column(String(64))
    resource_id: Mapped[str | None] = mapped_column(String(128))
    detail: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
