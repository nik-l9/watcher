"""Thread investigations to the one they follow up on.

A follow-up question was indistinguishable from a new one, so "and what about mobile?" started an
investigation that had never seen the report it was following. This column is what relates them.

It is load-bearing for grounding rather than only for display: `cortex/db/threads.py` derives the
set of investigations whose evidence a report may cite from this chain, so a follow-up can cite an
observation its parent already paid for instead of re-fetching it.

Indexed on (tenant_id, parent_id): every read of this column is tenant-scoped, because a
`parent_id` arriving in a request proves the row exists and says nothing about who owns it (F-01).

ON DELETE SET NULL rather than CASCADE. Deleting a parent should orphan the thread rather than
delete a follow-up somebody may still be reading -- and the parent's evidence cascades away with
the parent regardless, so a broken link narrows what a report can cite rather than widening it.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "b3e91c47d5a2"
down_revision = "9a2d5e7c1b40"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "investigations",
        sa.Column("parent_id", sa.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_investigations_parent",
        "investigations",
        "investigations",
        ["parent_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_investigations_tenant_parent",
        "investigations",
        ["tenant_id", "parent_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_investigations_tenant_parent", table_name="investigations")
    op.drop_constraint("fk_investigations_parent", "investigations", type_="foreignkey")
    op.drop_column("investigations", "parent_id")
