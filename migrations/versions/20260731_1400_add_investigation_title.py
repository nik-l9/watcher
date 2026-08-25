"""Add a short title to investigations.

A list rendered from raw question text is unusable, and without a stored title any UI would
have to invent one in the frontend — where it could not be searched, could not be kept stable,
and would differ between views.

Backfilled in SQL rather than left NULL for existing rows: a list is built from this column, and
a mix of titled and untitled rows looks like a bug in the list. The backfill is a plain
truncation, which is what `title_for` does for a long question anyway.

Revision ID: 7c4e0d8b3a15
Revises: 5b1f7c2a94d6
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "7c4e0d8b3a15"
down_revision = "5b1f7c2a94d6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("investigations", sa.Column("title", sa.String(96)))
    # Left-trim, drop a trailing question mark, cut to 72 characters. Deliberately simpler
    # than `title_for`: a migration should not import application code, because the code will
    # change and the migration must keep producing what it produced on the day it ran.
    op.execute(
        """
        UPDATE investigations
        SET title = left(trim(trailing '?' from btrim(question)), 72)
        WHERE title IS NULL
        """
    )


def downgrade() -> None:
    op.drop_column("investigations", "title")
