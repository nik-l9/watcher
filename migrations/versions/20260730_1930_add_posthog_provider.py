"""Add posthog to the credential_provider enum.

`CredentialProvider` is a Postgres enum type, so adding a member is DDL rather than a
Python-only change: without this migration the application accepts `posthog` and the
insert fails with `invalid input value for enum credential_provider`.

`ALTER TYPE ... ADD VALUE` cannot run inside a transaction block on older Postgres and
is not reversible, which is why the downgrade rebuilds the type rather than pretending
to remove a value.

Revision ID: 7a1c2f9b4d31
Revises: 25e90186497b
"""

from __future__ import annotations

from alembic import op

revision = "7a1c2f9b4d31"
down_revision = "25e90186497b"
branch_labels = None
depends_on = None

#: Upper-cased because SQLAlchemy's `Enum` persists a Python enum by **name**, not by
#: value. The existing rows read `GITHUB`, `HUBSPOT`, `SLACK`, so a lowercase member is
#: never written and never matched — the first attempt to store one fails with
#: `invalid input value for enum credential_provider`. Verified against the live table
#: rather than assumed, after adding the wrong casing first.
_VALUES_BEFORE = ("GA4", "BIGQUERY", "HUBSPOT", "GITHUB", "SLACK")


def upgrade() -> None:
    # IF NOT EXISTS so re-running against a database that already has the value is a
    # no-op rather than an error — the case when a developer created the enum from
    # models.py metadata before this migration existed.
    op.execute("ALTER TYPE credential_provider ADD VALUE IF NOT EXISTS 'POSTHOG'")


def downgrade() -> None:
    """Rebuild the type without `posthog`.

    Postgres cannot drop an enum value, so the only honest downgrade is to recreate the
    type and re-point the column at it. Any credential row still using `posthog` would
    block the cast, which is correct: silently deleting a tenant's stored credential to
    make a downgrade succeed would be worse than failing.
    """
    values = ", ".join(f"'{value}'" for value in _VALUES_BEFORE)
    op.execute(f"CREATE TYPE credential_provider_old AS ENUM ({values})")
    op.execute(
        "ALTER TABLE credentials ALTER COLUMN provider TYPE credential_provider_old "
        "USING provider::text::credential_provider_old"
    )
    op.execute("DROP TYPE credential_provider")
    op.execute("ALTER TYPE credential_provider_old RENAME TO credential_provider")
