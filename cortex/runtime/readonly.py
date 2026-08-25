"""A session that cannot write, for scripts that only mean to look.

This exists because of an incident. A probe script — commented *"rolled back, so nothing is
actually lost"* — called `apply_retention` with a cutoff 500 days in the future against the live
database. Retention commits per batch by design, so there was no transaction to roll back, and
it deleted 2,231 metric points, 58 evidence rows and 72 audit rows. The metric points were
re-ingestible. The others were not.

Guards were added at both places that delete. But the *class* of mistake was not "retention
lacks a guard" — it was that ad-hoc scripts get full write access to production data by
default, and the only thing standing between a typo and a deletion is whatever the script's
author was thinking at the time.

**So the protection is Postgres', not ours.** `SET TRANSACTION READ ONLY` makes the database
refuse every INSERT, UPDATE, DELETE and DDL statement in the transaction. A script using this
cannot destroy data even if it calls a function designed to. That is a different kind of
assurance from a comment or a keyword argument: it does not depend on the caller understanding
the code they are calling.

Use it for anything exploratory:

    async with read_only_session(resources) as session:
        rows = await session.execute(select(MetricPoint).limit(10))

A write attempt raises `InternalError: cannot execute DELETE in a read-only transaction`,
which is a much better outcome than a successful one.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.runtime.resources import Resources


@asynccontextmanager
async def read_only_session(resources: Resources) -> AsyncIterator[AsyncSession]:
    """A session whose transaction the database will not let write.

    Rolls back on exit rather than committing, but that is the *second* line of defence and
    not the one that matters: the incident this exists for happened because the code being
    called committed on its own. The read-only transaction is what makes that impossible.
    """
    async with resources.sessionmaker() as session:
        # Issued before anything else on the connection. Postgres only accepts this at the
        # start of a transaction, so a session that has already written cannot be demoted —
        # which is correct: demoting one would imply the earlier writes were reviewed.
        await session.execute(text("SET TRANSACTION READ ONLY"))
        try:
            yield session
        finally:
            # Always. A read-only transaction left open holds a snapshot, and a long-running
            # probe would pin the database's cleanup horizon behind it.
            await session.rollback()
