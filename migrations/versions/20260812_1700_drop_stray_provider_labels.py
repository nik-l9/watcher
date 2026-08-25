"""Remove two enum labels that nothing can ever use.

Repair for a mistake in `d8c05b3e7f19`, which briefly added `mixpanel` and `amplitude` in
lowercase. SQLAlchemy persists a Python enum by `.name`, so the labels the application writes are
`MIXPANEL` and `AMPLITUDE`; the lowercase pair are unreachable — no code path can produce them and
no row holds one.

**Why bother, given they are harmless.** They are harmless *now*. What they are is misleading:
the next person to read `enum_range(credential_provider)` sees ten labels for eight providers and
has to work out which four are real, and the obvious guess — that the lowercase ones are current
and the uppercase ones legacy — is exactly backwards. A schema that documents a mistake is a trap
laid for whoever debugs the next connector.

**Postgres cannot drop an enum value**, so the type is rebuilt: create the correct type, move
every column onto it, drop the old one, rename. Three columns use it — `credentials.provider`,
`sync_state.provider` and `metric_points.provider` — and all three are NOT NULL with no default,
which is what makes this a straightforward `ALTER ... TYPE ... USING` rather than a dance around
defaults.

**It refuses rather than destroys.** If any row somehow holds a stray label the cast would fail
mid-migration, so the migration checks first and raises with the offending rows named. Rebuilding
a type is the one operation here that could lose data, and it should stop rather than guess.

Idempotent: a database that never applied the bad version already has exactly these labels, and
the rebuild leaves it identical.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "e1a7c9d40b62"
down_revision = "d8c05b3e7f19"
branch_labels = None
depends_on = None

#: The labels the type should hold: the `.name` of every `CredentialProvider` member.
#:
#: Written out rather than imported from the application. A migration describes the schema at one
#: point in history; importing the live enum would make this file's behaviour change as the code
#: changes, so a replay against an old database would produce a type from the future.
CORRECT_LABELS = (
    "GA4",
    "BIGQUERY",
    "HUBSPOT",
    "GITHUB",
    "SLACK",
    "POSTHOG",
    "MIXPANEL",
    "AMPLITUDE",
)

#: Every column carrying the type. Missing one would leave it pointing at the dropped type and
#: fail the migration, which is noisy rather than silent — but the list is verified against
#: `information_schema` in the check below so it cannot rot quietly.
COLUMNS = (
    ("credentials", "provider"),
    ("sync_state", "provider"),
    ("metric_points", "provider"),
)

TYPE = "credential_provider"
STRAYS = ("mixpanel", "amplitude")


def upgrade() -> None:
    bind = op.get_bind()

    # 1. Refuse if any row holds a label about to disappear. The cast would fail anyway; failing
    #    here names the table and the value instead of surfacing a bare cast error.
    for table, column in COLUMNS:
        held = bind.execute(
            sa.text(
                f"SELECT {column}::text AS label, count(*) AS n FROM {table} "
                f"WHERE {column}::text = ANY(:strays) GROUP BY 1"
            ),
            {"strays": list(STRAYS)},
        ).all()
        if held:
            raise RuntimeError(
                f"{table}.{column} holds rows with labels this migration removes: {held}. "
                "Rewrite them to the uppercase form before running this."
            )

    # 2. Confirm the column list is complete, so a table added later cannot be silently orphaned.
    actual = {
        (row.table_name, row.column_name)
        for row in bind.execute(
            sa.text(
                "SELECT table_name, column_name FROM information_schema.columns "
                "WHERE udt_name = :type"
            ),
            {"type": TYPE},
        ).all()
    }
    if actual != set(COLUMNS):
        raise RuntimeError(
            f"columns using {TYPE} have changed: expected {sorted(COLUMNS)}, found "
            f"{sorted(actual)}. Update COLUMNS in this migration."
        )

    # 3. Rebuild.
    labels = ", ".join(f"'{label}'" for label in CORRECT_LABELS)
    op.execute(f"CREATE TYPE {TYPE}_new AS ENUM ({labels})")
    for table, column in COLUMNS:
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN {column} "
            f"TYPE {TYPE}_new USING {column}::text::{TYPE}_new"
        )
    op.execute(f"DROP TYPE {TYPE}")
    op.execute(f"ALTER TYPE {TYPE}_new RENAME TO {TYPE}")


def downgrade() -> None:
    """Puts the stray labels back, so the migration is reversible in the only sense that matters.

    Additive, so it cannot fail on data. Nothing will write them — that is the whole point of
    removing them — but a downgrade should restore the schema it found.
    """
    op.execute("COMMIT")
    for stray in STRAYS:
        op.execute(f"ALTER TYPE {TYPE} ADD VALUE IF NOT EXISTS '{stray}'")
