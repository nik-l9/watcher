"""SQL over local data — the analyst writes the query.

A fixed set of named queries cannot answer questions nobody anticipated, which is most of
the interesting ones. "Which segment drove the drop, broken down by plan and cohort" is a
query, not a capability someone thought to build. So the analyst authors SQL, and the job
of this module is to make that safe enough to point at real data.

DuckDB is the engine because it reads CSV, parquet, JSON and SQLite directly, which means
one tool covers a customer's exports, a warehouse extract and a benchmark's fixtures
without a separate code path for each.

**Two independent layers of containment**, because SQL is the one place where a mistake
reads something it should not:

  1. **The engine door is shut.** Files are materialised into native tables during setup,
     by this module, from paths this module chose. Then `enable_external_access = false`
     is set, after which DuckDB itself refuses every path, URL and `read_*` function.
     Verified rather than assumed: `read_csv_auto('/etc/passwd')` raises
     `PermissionException`, and `FROM '/etc/hosts'` raises `CatalogException`.
  2. **The statement is parsed.** `sqlguard` enforces one statement, reads only, and a
     table allowlist. This catches what the engine cannot: a write, a stacked statement,
     or a query touching a table that exists but is not this tenant's to read.

Layer 2 alone would not have been enough, and finding out why is the reason both exist.
`read_csv_auto('/etc/passwd')` parses with an **empty** table name, so an allowlist built
from table names never sees it. A guard that looked correct would have passed the query
straight through.
"""

from __future__ import annotations

import datetime as dt
import decimal
import json
import threading
from pathlib import Path
from typing import Any

import duckdb

from cortex.tools.base import Capability, Freshness, InvalidParams, ToolContext, ToolResult
from cortex.tools.base import Tool as BaseTool
from cortex.tools.sqlguard import UnsafeSQL, check

#: File types DuckDB can read directly, mapped to the reader that materialises them.
#: Anything else in the workspace is ignored rather than guessed at.
_READERS = {
    ".csv": "read_csv_auto",
    ".tsv": "read_csv_auto",
    ".parquet": "read_parquet",
    ".json": "read_json_auto",
    ".jsonl": "read_json_auto",
    ".ndjson": "read_json_auto",
}

#: Rows returned to the model for one query. An observation becomes prompt input, so an
#: unbounded result costs tokens on a real invoice as well as memory.
MAX_ROWS = 200

#: Rows shown per table when describing the workspace. Enough to see the shape of a
#: column — a date format, an id convention, whether nulls are empty strings — without
#: turning schema discovery into a data dump.
_SAMPLE_ROWS = 3

#: Wall-clock ceiling on one query.
#:
#: A row limit bounds the *result*, not the *work*: `SELECT ... FROM big CROSS JOIN big
#: LIMIT 200` returns two hundred rows after scanning a cartesian product. Against a 543MB
#: benchmark database that runs for as long as it is allowed to, and a benchmark run was
#: observed alive for 65 minutes having used 5 seconds of CPU — the shape of a process
#: waiting rather than working.
#:
#: Enforced by DuckDB's own interrupt from a watchdog thread rather than by a signal, so it
#: works off the main thread and inside async code, where `signal.alarm` does not.
QUERY_TIMEOUT_SECONDS = 60.0


class SQLTool(BaseTool):
    """Read-only SQL over a directory of data files.

    `provider` is None: the data is configured, not credentialed. A warehouse-backed
    variant would carry a credential, but pointing this at a local workspace must not
    require one, or every benchmark and every customer export needs a fake credential row.
    """

    name = "sql"
    provider = None

    def __init__(
        self,
        workspace: Path | str | None = None,
        attach: Path | str | None = None,
    ) -> None:
        self._workspace = Path(workspace) if workspace else None
        # An existing DuckDB database to query instead of loose files. Opened read-only,
        # which is a third containment layer under the engine lockdown and the statement
        # parser: even a write that somehow passed both cannot reach the file.
        self._attach = Path(attach) if attach else None
        self._connection: duckdb.DuckDBPyConnection | None = None
        self._tables: dict[str, list[str]] = {}
        super().__init__()

    def capabilities(self) -> list[Capability]:
        return [
            Capability(
                name="list_tables",
                description=(
                    "The tables available to query, with their columns, row counts and a "
                    "few sample rows. Call this first: table and column names are "
                    "dataset-specific, and a guessed name produces an error or, worse, an "
                    "empty result that looks like a real absence."
                ),
                params_schema={"type": "object", "additionalProperties": False, "properties": {}},
                handler=self.list_tables,
                result_key="tables",
            ),
            Capability(
                name="query",
                description=(
                    "Run one read-only SQL query (DuckDB dialect) and return the rows. "
                    "Use this to measure a change, break it down by segment, or test an "
                    "explanation — anything the fixed capabilities do not cover.\n"
                    "SELECT only. A row limit is applied automatically. Table names must "
                    "come from list_tables; file paths and URLs are not readable."
                ),
                params_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["sql"],
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": (
                                "One SELECT statement. Prefer explicit column lists and "
                                "aggregates over SELECT * on a large table."
                            ),
                        },
                        "purpose": {
                            "type": "string",
                            "description": (
                                "What this query is meant to establish, in one line. "
                                "Recorded with the evidence so a reader can tell what the "
                                "numbers were gathered to test."
                            ),
                        },
                    },
                },
                handler=self.query,
                result_key="rows",
            ),
        ]

    # ---------------------------------------------------------------- the sandbox

    def _connect(self) -> duckdb.DuckDBPyConnection:
        """Materialise the workspace, then shut the engine's door.

        Done once and cached: re-materialising per query would re-read every file, and
        the lockdown is not reversible within a connection, which is the point.
        """
        if self._connection is not None:
            return self._connection

        if self._attach is not None:
            if not self._attach.is_file():
                raise InvalidParams(f"sql: database {self._attach} does not exist")
            connection = duckdb.connect(str(self._attach), read_only=True)
            self._tables = {
                str(row[0]): [
                    str(column[0])
                    for column in connection.execute(f'DESCRIBE "{row[0]}"').fetchall()
                ]
                for row in connection.execute("SHOW TABLES").fetchall()
            }
            # Same lockdown as the file path: the attached tables stay readable while every
            # path, URL and read_* function is refused from here on.
            connection.execute("SET enable_external_access = false")
            self._connection = connection
            return connection

        if self._workspace is None:
            raise InvalidParams(
                "sql: no data workspace is configured for this tenant, so there is "
                "nothing to query."
            )
        if not self._workspace.is_dir():
            raise InvalidParams(f"sql: data workspace {self._workspace} does not exist")

        connection = duckdb.connect(":memory:")

        loaded: dict[str, list[str]] = {}
        for path in sorted(self._workspace.iterdir()):
            reader = _READERS.get(path.suffix.lower())
            if reader is None or not path.is_file():
                continue
            table = _table_name(path)
            try:
                # The path is interpolated from a directory *we* chose, not from anything
                # the model supplied — which is why this is safe here and why the model is
                # never given a capability that takes a path.
                connection.execute(
                    f'CREATE TABLE "{table}" AS SELECT * FROM {reader}(?)', [str(path)]
                )
            except duckdb.Error:
                # A malformed file is skipped rather than failing the whole workspace: one
                # unreadable export should not make the other twelve unavailable.
                continue
            loaded[table] = [
                str(row[0]) for row in connection.execute(f'DESCRIBE "{table}"').fetchall()
            ]

        # Everything the agent will run happens after this line. From here DuckDB refuses
        # every file, URL and read_* function, so a query cannot reach outside the tables
        # materialised above even if the statement parser were fooled.
        connection.execute("SET enable_external_access = false")

        self._connection = connection
        self._tables = loaded
        return connection

    def _source_name(self) -> str:
        """What the evidence cites, so a citation names the data it came from."""
        if self._attach is not None:
            return f"database/{self._attach.stem}"
        if self._workspace is not None:
            return f"workspace/{self._workspace.name}"
        return "none"

    # ------------------------------------------------------------------ capabilities

    async def list_tables(self, ctx: ToolContext) -> ToolResult:
        del ctx
        connection = self._connect()
        tables: list[dict[str, Any]] = []
        for name, columns in self._tables.items():
            count = connection.execute(f'SELECT count(*) FROM "{name}"').fetchone()
            sample = connection.execute(f'SELECT * FROM "{name}" LIMIT {_SAMPLE_ROWS}').fetchall()
            tables.append(
                {
                    "table": name,
                    "columns": columns,
                    "row_count": int(count[0]) if count else 0,
                    "sample_rows": [[_jsonable(v) for v in row] for row in sample],
                }
            )
        return ToolResult(
            payload={"table_count": len(tables), "tables": tables},
            source_ref=f"sql://{self._source_name()}",
            meta={"freshness": Freshness.LIVE},
        )

    async def query(self, ctx: ToolContext, *, sql: str, purpose: str = "") -> ToolResult:
        del ctx
        connection = self._connect()

        try:
            checked = check(
                sql,
                allowed_tables=set(self._tables),
                dialect="duckdb",
                row_limit=MAX_ROWS,
            )
        except UnsafeSQL as exc:
            # Surfaced as a correctable message, not a crash. The loop feeds tool errors
            # back to the model, and "table not available: orders. You may read: payments,
            # deals" gets a fixed query, where "invalid SQL" gets the same one again.
            raise InvalidParams(f"sql: {exc}") from exc

        try:
            cursor = _execute_bounded(connection, checked.sql)
            columns = [d[0] for d in cursor.description or []]
            rows = cursor.fetchall()
        except duckdb.Error as exc:
            raise InvalidParams(f"sql: query failed: {_first_line(exc)}") from exc

        return ToolResult(
            payload={
                # The executed statement, not the submitted one: the guard may have added
                # a limit, and a reader checking the evidence must see what actually ran.
                "sql": checked.sql,
                "purpose": purpose,
                "tables_read": sorted(checked.tables),
                "columns": columns,
                "row_count": len(rows),
                "truncated": len(rows) >= MAX_ROWS,
                "rows": [[_jsonable(v) for v in row] for row in rows],
            },
            source_ref=f"sql://query/{_digest(checked.sql)}",
            meta={"freshness": Freshness.LIVE},
        )


def _execute_bounded(
    connection: duckdb.DuckDBPyConnection, sql: str, seconds: float = QUERY_TIMEOUT_SECONDS
) -> Any:
    """Run a query, interrupting it if it exceeds `seconds`.

    DuckDB's `interrupt()` is the mechanism rather than a signal alarm, because alarms only
    fire on the main thread and this runs inside async code — the case where a signal-based
    timeout silently never triggers. The timer is cancelled on the normal path, so a fast
    query pays nothing but the cost of arming it.
    """
    timer = threading.Timer(seconds, connection.interrupt)
    timer.daemon = True
    timer.start()
    try:
        return connection.execute(sql)
    except duckdb.Error as exc:
        # An interrupt surfaces as a DuckDB error, so it is relabelled: "query cancelled
        # after 60s" tells the model to write a cheaper query, where the raw interrupt
        # message reads like an internal fault it cannot act on.
        if "interrupt" in str(exc).lower():
            raise InvalidParams(
                f"sql: query cancelled after {seconds:.0f}s. It scanned too much data — "
                "add a filter, aggregate earlier, or narrow the date range. A LIMIT alone "
                "does not help, because the rows are still computed before being cut."
            ) from exc
        raise
    finally:
        timer.cancel()


def _table_name(path: Path) -> str:
    """A SQL-safe table name from a filename.

    Kept close to the original so the model can guess it from a directory listing, but
    normalised: `2026 payments (final).csv` is not a usable identifier.
    """
    stem = path.stem.lower()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "data"


def _jsonable(value: Any) -> Any:
    """Make a DuckDB value safe to serialise into an evidence row.

    Dates and decimals are common in real exports and neither is JSON-serialisable;
    storing them raw would fail at the point the observation is persisted, which is after
    the query has already been paid for.
    """
    if isinstance(value, dt.datetime | dt.date | dt.time):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict | list):
        return json.loads(json.dumps(value, default=str))
    return value


def _digest(sql: str) -> str:
    import hashlib

    return hashlib.sha256(sql.encode()).hexdigest()[:12]


def _first_line(exc: Exception) -> str:
    return str(exc).splitlines()[0][:300]
