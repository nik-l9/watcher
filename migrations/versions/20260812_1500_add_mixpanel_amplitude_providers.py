"""Let a tenant connect Mixpanel or Amplitude.

Two new values on the `credentialprovider` enum. The enum is the reason this needs a migration at
all: adding a member to the Python `StrEnum` is free, but Postgres will reject an insert naming a
label its type does not have, so a tenant connecting Mixpanel would fail at the vault write with
an error about an enum rather than about a connector.

**`ALTER TYPE ... ADD VALUE` cannot run inside a transaction block** on Postgres before 12, and
even on 12+ a value added in a transaction cannot be *used* in that same transaction. Alembic
wraps a migration in one by default, so this commits first and adds the values outside it.

**There is no downgrade.** Postgres cannot remove a value from an enum type; the documented route
is to create a replacement type, rewrite every column that uses it, and drop the original. That is
a destructive rewrite of the credentials table to undo an additive change that breaks nothing, so
the downgrade is deliberately a no-op with this comment as its reason. Rows naming the new
providers would also have nowhere to go.
"""

from __future__ import annotations

from alembic import op

revision = "d8c05b3e7f19"
down_revision = "c7f4a91d2e60"
branch_labels = None
depends_on = None

#: The **names** of the new `cortex.db.models.CredentialProvider` members, not their values.
#:
#: This is the trap, and it is worth being explicit about because the wrong version of this list
#: applies cleanly and breaks nothing until a tenant connects the provider. SQLAlchemy's `Enum`
#: persists a Python enum by `.name` unless told otherwise, so the labels in Postgres are
#: `GA4`, `SLACK`, `POSTHOG` -- uppercase -- while the Python values are lowercase. A migration
#: adding `mixpanel` therefore adds a label nothing will ever write, and the insert of
#: `CredentialProvider.MIXPANEL` still fails on the missing `MIXPANEL`.
#:
#: `tests/db/test_schema.py::TestTheProviderEnumMatchesTheDatabase` compares `.name` against
#: `enum_range` and exercises a real insert, which is how this was caught.
NEW_PROVIDERS = ("MIXPANEL", "AMPLITUDE")

#: The Postgres type name, which is **not** the Python class name lowercased.
#:
#: SQLAlchemy names an enum type from the `name=` given to `sa.Enum(...)`, and this schema uses
#: snake_case. Guessing `credentialprovider` fails with `type ... does not exist`, which is a
#: confusing error for an additive migration -- verified against `pg_type` rather than assumed.
ENUM_TYPE = "credential_provider"


def upgrade() -> None:
    # Commit the implicit transaction Alembic opened, so ADD VALUE runs on its own.
    op.execute("COMMIT")
    for provider in NEW_PROVIDERS:
        # IF NOT EXISTS so re-running against a database that already has the value is a no-op
        # rather than a failure -- the state this leaves behind is what matters, not whether
        # this particular run was the one that created it.
        op.execute(f"ALTER TYPE {ENUM_TYPE} ADD VALUE IF NOT EXISTS '{provider}'")


def downgrade() -> None:
    """Deliberately empty. See the module docstring: removing an enum value in Postgres means
    rewriting every column that uses the type, which is a destructive undo of an additive
    change."""
