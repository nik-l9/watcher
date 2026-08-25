"""One investigation per inbound event.

A single Slack mention produced two investigations, two bills, and two answers in the same
thread — and because the model is not deterministic, the two answers **disagreed** on whether
signups had fallen. That is the worst shape this failure can take: a product whose promise is a
grounded answer gave two grounded answers that contradicted each other, in public, to one
question.

The webhook transport already suppressed duplicates using Slack's `X-Slack-Retry-Num` header.
Socket Mode has no equivalent — a redelivery there is the same envelope arriving again — so the
first Socket Mode deployment reintroduced a bug the webhook had solved, in a form the webhook's
fix could not catch. Hence a constraint in the database rather than a second header check: it
holds for every transport, including the next one, and it cannot race itself the way a
select-then-insert can.

`source_event_id` holds Slack's `event_id`, which is stable across redeliveries. Nullable,
because a question asked through the API or the CLI has no upstream event, and Postgres treats
NULLs as distinct under a unique constraint — so CLI questions never collide with one another
while two deliveries of one mention do.

Scoped by tenant like every other constraint here. Two workspaces could in principle mint the
same identifier, and a global unique index would let one tenant's traffic suppress another's
question — F-01 through an index.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "f2b6d81c0a37"
down_revision = "e1a7c9d40b62"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "investigations",
        sa.Column("source_event_id", sa.String(length=128), nullable=True),
    )
    # Existing rows keep NULL, which is correct: none of them came from a deduplicated event, and
    # backfilling a synthetic value would invent a claim about where they came from.
    op.create_unique_constraint(
        "uq_investigations_source_event",
        "investigations",
        ["tenant_id", "source_event_id"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_investigations_source_event", "investigations", type_="unique")
    op.drop_column("investigations", "source_event_id")
