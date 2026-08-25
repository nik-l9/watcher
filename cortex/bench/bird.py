"""BIRD — text-to-SQL over large, messy, real databases.

This is the benchmark that directly measures the capability built this evening: the analyst
writing its own SQL against a schema it has never seen. BIRD's databases are deliberately
unpleasant — inconsistent naming, columns with spaces and backticks, values that need
domain knowledge to interpret — which is what makes it a better test of SQL authoring than
a clean synthetic schema.

**Graded by execution, not by string comparison.** BIRD ships gold SQL rather than gold
answers, and two correct queries for the same question rarely look alike: different join
order, different aliases, `COUNT(*)` against `COUNT(column)`. So both queries run and their
result sets are compared. Comparing SQL text would score a correct answer wrong for
cosmetic reasons and tell us nothing.

**The agent explores, then commits.** The loop may run trial queries through the `sql` tool
— checking the schema, sampling a column, discovering that a "date" is stored as text — and
then produces one final statement. That mirrors how an agentic system is normally run
against BIRD, and it is the honest shape: a human analyst also looks before writing the
query that counts.

**Order-insensitive comparison, deliberately.** Row order is only meaningful when the
question asks for it, and a query that returns the right rows in a different order has
answered the question. Comparing as sorted multisets keeps duplicate rows significant,
which matters: `[a, a, b]` and `[a, b]` are different answers.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import duckdb
from sqlalchemy.ext.asyncio import async_sessionmaker

from cortex.agents.anthropic_llm import DEFAULT_EFFORT, DEFAULT_MODEL, AnthropicLLM
from cortex.agents.employee import gtm_data_analyst
from cortex.agents.investigator import Investigator
from cortex.agents.llm import Message, Usage
from cortex.bench.harness import Task, TaskResult
from cortex.db.models import Investigation as InvestigationRow
from cortex.db.models import Tenant
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.runtime.resources import Resources
from cortex.tenancy.context import TenantContext
from cortex.tools.base import ToolRegistry
from cortex.tools.executor import ToolExecutor
from cortex.tools.sql import SQLTool

QUESTIONS_URL = (
    "https://huggingface.co/datasets/birdsql/bird_sql_dev_20251106/"
    "resolve/main/data/dev_20251106-00000-of-00001.json"
)

#: Rows compared before giving up on a result set.
#:
#: A BIRD question can legitimately return thousands of rows, and comparing all of them is
#: both slow and pointless: if the first few thousand match, the queries agree. Set high
#: enough that no realistic answer is truncated into a false match.
_MAX_COMPARE_ROWS = 20_000

#: Above the product's 120 seconds, for the same reason as the DABstep adapter: a query
#: over a 543MB database is a different shape of call from a GTM loop turn.
_BENCH_TIMEOUT_SECONDS = 300.0

_SQL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["sql"],
    "properties": {
        "sql": {
            "type": "string",
            "description": (
                "The single final SELECT statement that answers the question. No prose, "
                "no markdown fences, no explanation. It will be executed exactly as given."
            ),
        }
    },
}


class BirdAdapter:
    """Runs Cortex against BIRD dev, scoring by execution accuracy."""

    name = "bird-dev"

    def __init__(self, resources: Resources, workspace: Path) -> None:
        self._resources = resources
        self._workspace = Path(workspace)
        self._databases = self._workspace / "db"
        self._llm = AnthropicLLM(
            model=DEFAULT_MODEL, effort=DEFAULT_EFFORT, timeout=_BENCH_TIMEOUT_SECONDS
        )

    # --------------------------------------------------------------------- loading

    def load(self) -> list[Task]:
        cache = self._workspace / "dev.json"
        if not cache.is_file():
            raise FileNotFoundError(
                f"BIRD questions not found at {cache}. Fetch with:\n"
                f"  curl -sL {QUESTIONS_URL} -o {cache}"
            )
        rows = json.loads(cache.read_text())

        available = _readable_databases(self._databases)
        if not available:
            raise FileNotFoundError(
                f"No .duckdb databases in {self._databases}. See docs/benchmarks.md."
            )

        # Tasks whose database is missing are dropped rather than errored. An absent
        # download is a setup gap, not an agent failure, and counting it against the score
        # would understate the agent while looking like a real result.
        return [
            Task(
                task_id=str(row["question_id"]),
                question=row["question"],
                payload={
                    "db_id": row["db_id"],
                    "evidence": row.get("evidence", ""),
                    "gold_sql": row["SQL"],
                    "difficulty": row.get("difficulty", ""),
                },
            )
            for row in rows
            if row["db_id"] in available
        ]

    # --------------------------------------------------------------------- solving

    async def solve(self, task: Task) -> TaskResult:
        db_id = str(task.payload["db_id"])
        database = self._databases / f"{db_id}.duckdb"

        registry = ToolRegistry()
        registry.register(SQLTool(workspace=self._databases, attach=database))
        investigator = Investigator(
            llm=self._llm,
            registry=registry,
            executor=ToolExecutor(registry),
            employee=gtm_data_analyst(),
        )

        maker = async_sessionmaker(self._resources.engine, expire_on_commit=False)
        async with maker() as session:
            tenant, investigation_id = await _provision(session, task)
            try:
                investigation = await investigator.investigate(
                    session,
                    tenant,
                    investigation_id=investigation_id,
                    question=_prompt(task),
                )
            finally:
                await session.rollback()

        predicted, extract_usage = await self._final_sql(task, investigation.report)
        correct, detail = _execution_matches(database, predicted, str(task.payload["gold_sql"]))

        return TaskResult(
            task_id=task.task_id,
            correct=correct,
            answer=predicted[:400],
            expected=detail,
            # A gold query that will not run against the data we have cannot grade
            # anything. 17% of the dev set is in this state because the DuckDB conversion
            # dropped tables named `order` and `trans` — both reserved words — so counting
            # these as agent failures would understate the agent by an unknown amount.
            unscoreable=detail.startswith("GOLD QUERY FAILED"),
            usage=investigation.usage + extract_usage,
        )

    async def _final_sql(self, task: Task, report: Any) -> tuple[str, Usage]:
        """Ask for the one statement that answers the question.

        Taken from a dedicated call rather than scraped from the report's prose: a report
        mentions several queries as it reasons, and picking "the last one that looks like
        SQL" would grade whichever trial query happened to come last.
        """
        summary = " ".join(claim.text for claim in report.executive_summary)
        findings = " ".join(claim.text for finding in report.findings for claim in finding.claims)
        payload, usage = await self._llm.structured(
            system=(
                "You produce the final SQL for a question, based on an analyst's notes. "
                "Return exactly one SELECT statement, valid DuckDB SQL, and nothing else.\n"
                # Two graded failures had the right rows and the wrong shape: one returned
                # a single column where three were wanted, another returned four where one
                # was wanted. Execution accuracy compares result sets, so a correct answer
                # with a different projection scores zero — and the fix is an instruction,
                # not more reasoning.
                "Project exactly the columns the question asks for, in the order it asks "
                "for them: no id columns it did not request, no extra context columns, "
                "and no fewer than it names. If it asks for a name, return the name alone; "
                "if it asks which and when, return both. Do not use SELECT * unless the "
                "question genuinely asks for whole rows."
            ),
            messages=[
                Message(
                    role="user",
                    content=(
                        f"QUESTION: {task.question}\n\n"
                        f"DOMAIN NOTES: {task.payload.get('evidence', '')}\n\n"
                        f"ANALYST NOTES:\n{summary}\n{findings}"[:20000]
                    ),
                )
            ],
            schema=_SQL_SCHEMA,
            max_tokens=2048,
        )
        return _strip_fences(str(payload.get("sql", ""))), usage


def _readable_databases(directory: Path) -> set[str]:
    """Databases that actually open.

    Checked rather than assumed: a 543MB download in this set was truncated, and every
    task against it failed with an IO error that looked like an agent problem in the
    scorecard. A database that cannot be opened is a setup fault, and its tasks are
    dropped before they are ever attempted.
    """
    readable: set[str] = set()
    for path in sorted(directory.glob("*.duckdb")):
        try:
            connection = duckdb.connect(str(path), read_only=True)
        except duckdb.Error:
            continue
        try:
            connection.execute("SHOW TABLES").fetchall()
            readable.add(path.stem)
        except duckdb.Error:
            continue
        finally:
            connection.close()
    return readable


def _provision(session: Any, task: Task) -> Any:
    async def _run() -> tuple[TenantContext, uuid.UUID]:
        tenant_id = uuid.uuid4()
        slug = f"bench-bird-{tenant_id.hex[:8]}"
        graph_name = graph_name_for_new_tenant(slug, tenant_id)
        session.add(Tenant(id=tenant_id, slug=slug, name=slug, graph_name=graph_name))
        await session.flush()
        row = InvestigationRow(tenant_id=tenant_id, question=task.question[:2000])
        session.add(row)
        await session.flush()
        return (
            TenantContext(tenant_id=tenant_id, tenant_slug=slug, graph_name=graph_name),
            row.id,
        )

    return _run()


def _prompt(task: Task) -> str:
    evidence = task.payload.get("evidence", "")
    return (
        f"{task.question}\n\n"
        + (
            f"DOMAIN NOTES (definitions you need; they cannot be inferred from the "
            f"schema): {evidence}\n\n"
            if evidence
            else ""
        )
        + "The database is available through the `sql` tool. Call `sql__list_tables` first "
        "to see tables and columns — the schema is unfamiliar and names are irregular, so "
        "do not guess them. Explore with trial queries as needed, then state the single "
        "query that answers the question and what it returns."
    )


def _strip_fences(sql: str) -> str:
    """Remove markdown fencing a model may add despite being told not to."""
    text = sql.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines)
    return text.strip().rstrip(";").strip()


def _execution_matches(database: Path, predicted: str, gold: str) -> tuple[bool, str]:
    """Run both queries and compare result sets.

    Returns the verdict and a short description of what happened, because "wrong" and
    "would not execute" are different failures needing different fixes, and a score that
    conflates them hides which one we have.
    """
    if not predicted:
        return False, "no SQL produced"

    connection = duckdb.connect(str(database), read_only=True)
    try:
        try:
            gold_rows = _fetch(connection, gold)
        except duckdb.Error as exc:
            # The gold query failing is a harness problem, not an agent failure, and must
            # not be silently scored against the agent.
            return False, f"GOLD QUERY FAILED: {_first_line(exc)}"

        try:
            predicted_rows = _fetch(connection, predicted)
        except duckdb.Error as exc:
            return False, f"predicted SQL error: {_first_line(exc)}"
    finally:
        connection.close()

    if predicted_rows == gold_rows:
        return True, f"{len(gold_rows)} row(s) matched"
    return (
        False,
        f"got {len(predicted_rows)} row(s), expected {len(gold_rows)}: {str(gold_rows[:3])[:120]}",
    )


def _fetch(connection: duckdb.DuckDBPyConnection, sql: str) -> list[tuple[Any, ...]]:
    """Execute and normalise a result set for comparison.

    Sorted so row order does not matter — order is only meaningful when the question asks
    for it, and a query returning the right rows in another order has answered it.
    Duplicates are preserved, because `[a, a, b]` and `[a, b]` are different answers.
    """
    rows = connection.execute(sql).fetchmany(_MAX_COMPARE_ROWS)
    return sorted(tuple(_comparable(value) for value in row) for row in rows)


def _comparable(value: Any) -> Any:
    """Make a value comparable across the two queries.

    Floats are rounded: two correct queries can differ in the last bits through a
    different order of summation, and scoring that as wrong would measure floating-point
    associativity rather than SQL. Everything else compares as its string form, so a
    Decimal and an int holding the same number agree.
    """
    if isinstance(value, float):
        return round(value, 6)
    if value is None:
        return None
    return str(value)


def _first_line(exc: Exception) -> str:
    return str(exc).splitlines()[0][:200]
