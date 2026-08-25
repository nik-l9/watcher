"""Let a question arrive from Slack and its answer go back there.

Two columns, and each carries a rule.

`tenants.slack_team_id` is the *only* thing that maps an inbound Slack event to a tenant. A Slack
request is unauthenticated apart from its HMAC signature, so nothing in the message body may decide
whose data is read. Unique, because two tenants claiming one workspace would make the mapping
ambiguous exactly where ambiguity means reading the wrong customer's evidence -- F-01 through a new
door.

`investigations.notify` records where to deliver the answer:
`{"kind": "slack", "channel": "C…", "thread_ts": "…"}`. Nullable, because a question asked through
the API or the CLI has nowhere to be delivered and that is not a defect. A dict rather than
columns: the second channel is a matter of when rather than if, and the reply target belongs to
whoever asked rather than to the investigation, which is complete and correct whether or not
anybody is told about it.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "c7f4a91d2e60"
down_revision = "b3e91c47d5a2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tenants", sa.Column("slack_team_id", sa.String(length=32), nullable=True))
    op.create_unique_constraint("uq_tenants_slack_team", "tenants", ["slack_team_id"])
    op.add_column(
        "investigations",
        sa.Column("notify", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("investigations", "notify")
    op.drop_constraint("uq_tenants_slack_team", "tenants", type_="unique")
    op.drop_column("tenants", "slack_team_id")
