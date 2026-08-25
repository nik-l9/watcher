"""Add sync_state and metric_points for the nightly ingest.

Two tables, and one shared decision worth recording: both are keyed so that re-running a
sync is a no-op rather than a duplication. A nightly ingest deliberately re-reads an
overlapping window — upstreams backfill late-arriving data — so "the same row arrives
twice" is the normal case, not an error case.

`sync_state` holds the watermark *and* the health record together, in one row. Splitting
them was considered and rejected: a watermark that advances while the health row says
"failed" describes a sync that both did and did not happen, and afterwards there is no way
to tell which is true.

Revision ID: 3c8d51a7e0f2
Revises: 7a1c2f9b4d31
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "3c8d51a7e0f2"
down_revision = "7a1c2f9b4d31"
branch_labels = None
depends_on = None

#: Upper-cased, because SQLAlchemy persists a Python enum by **name**. Learned from the
#: previous migration, where lowercase members would have failed on the first insert
#: against a table whose existing rows read `GITHUB`.
_SYNC_STATUS = postgresql.ENUM("OK", "FAILED", "PARTIAL", name="sync_status", create_type=False)

#: Referenced, never created: `credential_provider` already exists. `create_type=False`
#: stops Alembic emitting a second CREATE TYPE, which fails with "type already exists".
_PROVIDER = postgresql.ENUM(name="credential_provider", create_type=False)


def upgrade() -> None:
    _SYNC_STATUS.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "sync_state",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("provider", _PROVIDER, nullable=False),
        sa.Column("label", sa.String(64), nullable=False, server_default="default"),
        sa.Column("stream", sa.String(64), nullable=False),
        sa.Column("watermark", sa.DateTime(timezone=True)),
        sa.Column("status", _SYNC_STATUS, nullable=False, server_default="OK"),
        sa.Column("detail", sa.Text()),
        sa.Column("last_run_at", sa.DateTime(timezone=True)),
        sa.Column("last_success_at", sa.DateTime(timezone=True)),
        sa.Column("consecutive_failures", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("items_written", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "tenant_id", "provider", "label", "stream", name="uq_sync_state_tenant_stream"
        ),
    )
    op.create_index("ix_sync_state_tenant", "sync_state", ["tenant_id", "provider"])

    op.create_table(
        "metric_points",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("provider", _PROVIDER, nullable=False),
        sa.Column("metric", sa.String(128), nullable=False),
        # Flattened to a string rather than JSONB so it can take part in the uniqueness
        # constraint: two JSONB objects holding the same pairs in a different order are
        # not equal, which would let one segment be inserted twice.
        sa.Column("segment_key", sa.String(256), nullable=False, server_default=""),
        sa.Column(
            "segment", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "provider",
            "metric",
            "segment_key",
            "observed_at",
            name="uq_metric_points_identity",
        ),
    )
    op.create_index(
        "ix_metric_points_lookup", "metric_points", ["tenant_id", "metric", "observed_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_metric_points_lookup", table_name="metric_points")
    op.drop_table("metric_points")
    op.drop_index("ix_sync_state_tenant", table_name="sync_state")
    op.drop_table("sync_state")
    _SYNC_STATUS.drop(op.get_bind(), checkfirst=True)
