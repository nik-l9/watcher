"""Record the token breakdown on an investigation, so spend can be computed.

`tokens_used` is a single total, and a total cannot be priced: input and output differ by 5x
on every model we run, and a cache read costs a fraction of a fresh input token. A bill
computed from the total is wrong by a multiple that depends on the mix — and with a measured
98% cache-hit rate, ignoring cache overstates it badly.

`tokens_used` is kept rather than replaced. It is what the existing API returns, and dropping
a column to add four is a breaking change for no gain.

Revision ID: 5b1f7c2a94d6
Revises: 3c8d51a7e0f2
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "5b1f7c2a94d6"
down_revision = "3c8d51a7e0f2"
branch_labels = None
depends_on = None

_COLUMNS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")


def upgrade() -> None:
    for name in _COLUMNS:
        # Defaulted at the server so existing rows read 0 rather than NULL. A NULL here would
        # make every historical investigation cost `None`, and a spend total that silently
        # skips rows is worse than one that reports them as zero — zero is visibly wrong.
        op.add_column(
            "investigations",
            sa.Column(name, sa.Integer(), nullable=False, server_default="0"),
        )
    op.add_column("investigations", sa.Column("model", sa.String(64)))


def downgrade() -> None:
    op.drop_column("investigations", "model")
    for name in _COLUMNS:
        op.drop_column("investigations", name)
