"""Timestamp audit rows with clock_timestamp(), not now().

Found by a test that passed alone and failed in the suite. The trace endpoint orders tool
calls by `created_at` and tie-breaks by id; two calls written in one transaction came back in
an arbitrary order, because **`now()` in Postgres is the transaction's start time, not the
current time.** Every row a transaction writes gets the same value, to the microsecond.

Verified against this database rather than recalled:

    now() same across statements:      True
    clock_timestamp() advances:        True

That makes it a defect in the audit trail itself, not only in one endpoint's ORDER BY. The
investigation loop writes a `tool_calls` row and an `evidence` row per capability call and
flushes rather than commits, so an entire investigation's audit history is stamped with the
instant its transaction opened. Two consequences:

  - **The order of what the analyst did is not recoverable.** `duration_ms` says each call
    took a different length of time while every `created_at` says they all happened at once,
    which is self-contradictory on its face.
  - **`observed_at` on evidence is wrong in the same way**, and that column is load-bearing
    beyond display: retention deletes by it, and a report cites it as when an observation was
    made. "Observed at the moment the transaction opened" is not what a reader is being told.

`clock_timestamp()` is the wall clock at statement execution, so rows written in one
transaction get distinct, increasing values in insertion order.

Scoped to the two tables where within-transaction ordering carries meaning. `TimestampMixin`
elsewhere keeps `now()`: for a tenant or a report row, transaction time is both correct and
preferable, because those rows *are* the transaction.

Existing rows are not rewritten. The information to correct them does not exist — that is
precisely what was lost — and a backfill would invent an ordering that looks measured.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "9a2d5e7c1b40"
down_revision = "7c4e0d8b3a15"
branch_labels = None
depends_on = None

#: (table, column) pairs whose ordering within one transaction has to be real.
_COLUMNS = (("tool_calls", "created_at"), ("evidence", "observed_at"))


def upgrade() -> None:
    for table, column in _COLUMNS:
        op.alter_column(
            table,
            column,
            server_default=sa.text("clock_timestamp()"),
            existing_type=sa.DateTime(timezone=True),
            existing_nullable=False,
        )


def downgrade() -> None:
    for table, column in _COLUMNS:
        op.alter_column(
            table,
            column,
            server_default=sa.text("now()"),
            existing_type=sa.DateTime(timezone=True),
            existing_nullable=False,
        )
