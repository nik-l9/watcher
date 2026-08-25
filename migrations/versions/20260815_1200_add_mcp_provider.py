"""Let a tenant connect an MCP server.

One new label on `credential_provider`. Unlike the other providers, `MCP` names a *transport*
rather than a vendor: a tenant may connect several MCP servers, and which one a credential belongs
to is carried by the credential's `label`, not by a provider of its own. Adding a provider per
server would mean a migration every time somebody connected one, which is the opposite of the point.

Same two traps as `d8c05b3e7f19`, and they are why this file exists rather than an enum edit: the
Postgres type is `credential_provider` in snake_case, and SQLAlchemy persists a Python enum by
`.name`, so the label is uppercase `MCP` rather than the lowercase value. Getting either wrong
applies cleanly and fails only when a tenant first connects.
"""

from __future__ import annotations

from alembic import op

revision = "a4d719e0c853"
down_revision = "f2b6d81c0a37"
branch_labels = None
depends_on = None

ENUM_TYPE = "credential_provider"


def upgrade() -> None:
    # Committed first: ADD VALUE cannot be used in the transaction that created it.
    op.execute("COMMIT")
    op.execute(f"ALTER TYPE {ENUM_TYPE} ADD VALUE IF NOT EXISTS 'MCP'")


def downgrade() -> None:
    """Empty. Postgres cannot remove an enum value without rewriting every column that uses the
    type, which is a destructive undo of an additive change -- see `e1a7c9d40b62`, which had to do
    exactly that to clean up two unreachable labels."""
