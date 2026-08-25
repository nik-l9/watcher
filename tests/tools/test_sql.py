"""SQL over local data.

Mostly containment tests. SQL is the one capability where a mistake reads something it
should not, and the tests are organised by *which layer* stops each attack — because the
reason there are two layers is that neither is sufficient alone.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from cortex.tools.base import InvalidParams, ToolContext
from cortex.tools.sql import MAX_ROWS, SQLTool, _table_name


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    with (tmp_path / "payments.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "amount", "plan"])
        for i in range(1, 11):
            writer.writerow([i, i * 100, "pro" if i % 2 else "free"])

    with (tmp_path / "2026 signups (final).csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["day", "count"])
        writer.writerow(["2026-06-01", 50])

    (tmp_path / "notes.txt").write_text("ignored: not a data file")
    return tmp_path


@pytest.fixture
def tool(workspace: Path) -> SQLTool:
    return SQLTool(workspace=workspace)


@pytest.fixture
def ctx() -> ToolContext:
    return ToolContext(tenant=None, credential=None)  # type: ignore[arg-type]


class TestDiscovery:
    async def test_data_files_become_tables(self, tool: SQLTool, ctx: ToolContext) -> None:
        result = await tool.list_tables(ctx)
        names = {t["table"] for t in result.payload["tables"]}
        assert "payments" in names

    async def test_an_awkward_filename_becomes_a_usable_identifier(
        self, tool: SQLTool, ctx: ToolContext
    ) -> None:
        result = await tool.list_tables(ctx)
        assert "2026_signups_final" in {t["table"] for t in result.payload["tables"]}

    async def test_non_data_files_are_ignored(self, tool: SQLTool, ctx: ToolContext) -> None:
        result = await tool.list_tables(ctx)
        assert "notes" not in {t["table"] for t in result.payload["tables"]}

    async def test_columns_and_counts_are_reported(self, tool: SQLTool, ctx: ToolContext) -> None:
        """The model must not have to guess column names: a guessed name returns an error
        or, worse, an empty result that reads as a real absence."""
        result = await tool.list_tables(ctx)
        payments = next(t for t in result.payload["tables"] if t["table"] == "payments")
        assert payments["columns"] == ["id", "amount", "plan"]
        assert payments["row_count"] == 10
        assert payments["sample_rows"]

    async def test_a_missing_workspace_is_a_clear_message(self, ctx: ToolContext) -> None:
        with pytest.raises(InvalidParams, match="does not exist"):
            await SQLTool(workspace="/nonexistent/path").list_tables(ctx)

    async def test_no_workspace_configured(self, ctx: ToolContext) -> None:
        with pytest.raises(InvalidParams, match="no data workspace"):
            await SQLTool().list_tables(ctx)


class TestQueries:
    async def test_an_aggregate_query_returns_rows(self, tool: SQLTool, ctx: ToolContext) -> None:
        result = await tool.query(
            ctx, sql="SELECT plan, sum(amount) AS total FROM payments GROUP BY plan"
        )
        assert result.payload["columns"] == ["plan", "total"]
        assert dict(result.payload["rows"]) == {"free": 3000, "pro": 2500}

    async def test_the_executed_statement_is_recorded(
        self, tool: SQLTool, ctx: ToolContext
    ) -> None:
        """The guard may add a limit. A reader checking the evidence has to see what ran,
        not what was submitted."""
        result = await tool.query(ctx, sql="SELECT * FROM payments")
        assert "LIMIT" in result.payload["sql"].upper()

    async def test_the_purpose_is_carried_into_the_evidence(
        self, tool: SQLTool, ctx: ToolContext
    ) -> None:
        result = await tool.query(
            ctx, sql="SELECT count(*) FROM payments", purpose="confirm the row count"
        )
        assert result.payload["purpose"] == "confirm the row count"

    async def test_tables_read_are_recorded(self, tool: SQLTool, ctx: ToolContext) -> None:
        result = await tool.query(ctx, sql="SELECT count(*) FROM payments")
        assert result.payload["tables_read"] == ["payments"]

    async def test_truncation_is_disclosed(self, tool: SQLTool, ctx: ToolContext) -> None:
        """A silently truncated result would let a report cite a partial answer as
        complete."""
        result = await tool.query(
            ctx, sql=f"SELECT * FROM range({MAX_ROWS + 50}) CROSS JOIN payments"
        )
        assert result.payload["row_count"] == MAX_ROWS
        assert result.payload["truncated"] is True

    async def test_a_broken_query_is_a_correctable_message(
        self, tool: SQLTool, ctx: ToolContext
    ) -> None:
        """Tool errors are fed back to the model, so the message has to be actionable."""
        with pytest.raises(InvalidParams):
            await tool.query(ctx, sql="SELECT nonexistent_column FROM payments")


class TestEngineLayerContainment:
    """Blocked by DuckDB itself, after `enable_external_access = false`.

    This layer exists because the parser cannot cover it: `read_csv_auto('/etc/passwd')`
    parses with an **empty** table name, so an allowlist built from table names never sees
    it. A guard that looked correct would have passed it straight through.
    """

    async def test_a_file_reading_function_is_refused(
        self, tool: SQLTool, ctx: ToolContext
    ) -> None:
        with pytest.raises(InvalidParams):
            await tool.query(ctx, sql="SELECT * FROM read_csv_auto('/etc/passwd')")

    async def test_a_remote_url_is_refused(self, tool: SQLTool, ctx: ToolContext) -> None:
        with pytest.raises(InvalidParams):
            await tool.query(
                ctx, sql="SELECT * FROM read_parquet('https://example.invalid/x.parquet')"
            )

    async def test_the_workspace_tables_still_work_after_lockdown(
        self, tool: SQLTool, ctx: ToolContext
    ) -> None:
        """Containment must not cost us the data. The files are materialised before the
        door shuts, so they remain readable."""
        result = await tool.query(ctx, sql="SELECT sum(amount) AS t FROM payments")
        assert result.payload["rows"][0][0] == 5500


class TestStatementLayerContainment:
    """Blocked by `sqlguard`, which catches what the engine cannot see."""

    @pytest.mark.parametrize(
        "sql",
        [
            "DROP TABLE payments",
            "DELETE FROM payments",
            "UPDATE payments SET amount = 0",
            "CREATE TABLE t AS SELECT 1",
            "INSERT INTO payments SELECT * FROM payments",
        ],
    )
    async def test_writes_are_refused(self, tool: SQLTool, ctx: ToolContext, sql: str) -> None:
        with pytest.raises(InvalidParams):
            await tool.query(ctx, sql=sql)

    async def test_stacked_statements_are_refused(self, tool: SQLTool, ctx: ToolContext) -> None:
        with pytest.raises(InvalidParams, match="statements"):
            await tool.query(ctx, sql="SELECT 1 FROM payments; DROP TABLE payments")

    async def test_an_unlisted_table_is_refused_and_the_options_are_named(
        self, tool: SQLTool, ctx: ToolContext
    ) -> None:
        with pytest.raises(InvalidParams) as caught:
            await tool.query(ctx, sql="SELECT * FROM secrets")
        assert "payments" in str(caught.value)

    async def test_a_raw_path_is_refused_as_an_unknown_table(
        self, tool: SQLTool, ctx: ToolContext
    ) -> None:
        with pytest.raises(InvalidParams, match="not available"):
            await tool.query(ctx, sql="SELECT * FROM '/etc/hosts'")


class TestTableNaming:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("payments.csv", "payments"),
            ("2026 signups (final).csv", "2026_signups_final"),
            ("Weird---Name!!.parquet", "weird_name"),
            ("....csv", "data"),
        ],
    )
    def test_names_are_normalised(self, filename: str, expected: str) -> None:
        assert _table_name(Path(filename)) == expected


class TestQueryTimeout:
    """A row limit bounds the result, not the work.

    `SELECT ... FROM big CROSS JOIN big LIMIT 200` returns two hundred rows after computing
    a cartesian product. A benchmark run was observed alive for 65 minutes having used five
    seconds of CPU, which is the signature of a process waiting rather than working.
    """

    async def test_a_runaway_query_is_cancelled(self, tool: SQLTool, ctx: ToolContext) -> None:
        import time

        from cortex.tools.sql import _execute_bounded

        connection = tool._connect()
        connection.execute("CREATE TABLE wide AS SELECT i FROM range(2000000) t(i)")
        started = time.monotonic()
        with pytest.raises(InvalidParams, match="cancelled"):
            _execute_bounded(connection, "SELECT count(*) FROM wide a, wide b, wide c", seconds=2.0)
        # Cancelled near the deadline rather than merely eventually.
        assert time.monotonic() - started < 15.0

    async def test_the_message_tells_the_model_what_to_do(
        self, tool: SQLTool, ctx: ToolContext
    ) -> None:
        """Tool errors are fed back, so "query cancelled" has to be actionable. The raw
        interrupt message reads like an internal fault the model cannot act on."""
        from cortex.tools.sql import _execute_bounded

        connection = tool._connect()
        connection.execute("CREATE TABLE wide2 AS SELECT i FROM range(2000000) t(i)")
        with pytest.raises(InvalidParams) as caught:
            _execute_bounded(
                connection, "SELECT count(*) FROM wide2 a, wide2 b, wide2 c", seconds=2.0
            )
        message = str(caught.value)
        assert "LIMIT alone does not help" in message

    async def test_a_fast_query_is_unaffected(self, tool: SQLTool, ctx: ToolContext) -> None:
        result = await tool.query(ctx, sql="SELECT count(*) AS n FROM payments")
        assert result.payload["rows"][0][0] == 10
