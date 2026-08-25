"""Validation for model-authored SQL.

The product needs the analyst to write SQL — a fixed set of named queries cannot answer
questions nobody anticipated, which is most of them. What it must never do is let generated
SQL mutate anything or read outside what the tenant is allowed to see.

Doing that with string matching does not work, and the ways it fails are not exotic:
`SELECT` appears in `INSERT INTO t SELECT ...`; a comment can hide a keyword
(`/*x*/DELETE`); `;` splits a statement someone forgot to check for; and a CTE can name a
table the allowlist never sees. So the statement is **parsed**, and the guarantees are
asserted against the syntax tree.

`sqlglot` does the parsing — MIT, pure Python, no runtime dependencies, and it understands
the dialects we care about. What it gives us is the ability to say "this expression tree
contains no node that writes" rather than "this string looks like a read".

Three properties are enforced, and each one is a real attack or accident:

  - **One statement.** Multiple statements is how a read becomes a read *and* a write.
  - **Reads only.** Any DDL or DML node anywhere in the tree rejects the whole statement,
    including inside a subquery or CTE.
  - **Allowlisted tables.** Every table referenced must be one the caller permitted. This
    is what stops a query reading another tenant's table in a shared database.

A row limit is applied by rewriting the tree rather than by appending text, because
appending ` LIMIT 100` to a statement that already ends in a comment, or that is a UNION,
produces either a syntax error or the wrong limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

#: Node types that read. Anything else — INSERT, UPDATE, DELETE, MERGE, CREATE, DROP,
#: ALTER, TRUNCATE, GRANT, a COPY, a call to a procedure — is refused. Named as an
#: allowlist rather than a denylist of writes: a denylist silently permits whatever the
#: next sqlglot version learns to parse.
_READ_ONLY_ROOTS = (exp.Select, exp.Union, exp.Except, exp.Intersect, exp.Subquery)

#: Nodes that mutate or change structure. Checked across the whole tree, so a write hidden
#: inside a CTE or subquery is caught even when the root looks like a SELECT.
_FORBIDDEN = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Grant,
    exp.Command,  # sqlglot's catch-all for statements it does not model — refuse those
)

DEFAULT_ROW_LIMIT = 1000


class UnsafeSQL(Exception):
    """The statement is not a bounded, read-only query over allowed tables."""


@dataclass(frozen=True, slots=True)
class CheckedQuery:
    """A statement that has passed every check, and what it touches."""

    sql: str
    tables: frozenset[str]
    row_limit: int


def check(
    sql: str,
    *,
    allowed_tables: set[str] | None = None,
    dialect: str = "sqlite",
    row_limit: int = DEFAULT_ROW_LIMIT,
) -> CheckedQuery:
    """Parse, verify and bound one statement. Raises `UnsafeSQL` with a usable reason.

    `allowed_tables` of `None` means "any table" — correct for a benchmark database that
    contains nothing else, and wrong for a shared one. It is an explicit argument rather
    than a default so that skipping the allowlist is a visible decision at the call site.

    Error messages name the specific problem, because they are fed back to the model as an
    observation: "table `orders` is not available; you may read: customers, deals" gets a
    corrected query, whereas "invalid SQL" gets a retry of the same mistake.
    """
    text = (sql or "").strip().rstrip(";").strip()
    if not text:
        raise UnsafeSQL("empty query")

    try:
        statements = sqlglot.parse(text, dialect=dialect)
    except Exception as exc:  # sqlglot raises several parse error types
        raise UnsafeSQL(f"could not parse as {dialect} SQL: {exc}") from exc

    statements = [s for s in statements if s is not None]
    if not statements:
        raise UnsafeSQL("no statement found")
    if len(statements) > 1:
        # The classic way a read becomes a write. Refused rather than truncated to the
        # first statement, because silently running half of what was asked is worse.
        raise UnsafeSQL(f"{len(statements)} statements found; submit exactly one read-only query")

    statement = statements[0]

    for node in statement.walk():
        if isinstance(node, _FORBIDDEN):
            raise UnsafeSQL(f"{type(node).__name__.upper()} is not permitted; this tool reads only")

    if not isinstance(statement, _READ_ONLY_ROOTS):
        raise UnsafeSQL(
            f"only SELECT queries are permitted, got {type(statement).__name__.upper()}"
        )

    tables = {
        table.name.lower()
        for table in statement.find_all(exp.Table)
        if table.name
        # A CTE is referenced like a table but is defined in the query itself, so it is
        # not an external read and must not be required to be on the allowlist.
        and table.name.lower() not in _cte_names(statement)
    }

    if allowed_tables is not None:
        permitted = {name.lower() for name in allowed_tables}
        unknown = sorted(tables - permitted)
        if unknown:
            raise UnsafeSQL(
                f"table(s) not available: {', '.join(unknown)}. "
                f"You may read: {', '.join(sorted(permitted)) or 'nothing'}"
            )

    bounded = _apply_limit(statement, row_limit)
    return CheckedQuery(
        sql=bounded.sql(dialect=dialect),
        tables=frozenset(tables),
        row_limit=row_limit,
    )


def _cte_names(statement: exp.Expression) -> set[str]:
    return {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE) if cte.alias_or_name}


def _apply_limit(statement: exp.Expression, row_limit: int) -> exp.Expression:
    """Bound the result set, without loosening a stricter limit the author already set.

    An unbounded query against a real table is a memory and cost incident, and the
    observation would be too large to use as evidence anyway. Tightening only: if the
    model asked for 10 rows we do not silently return 1000.
    """
    existing = statement.args.get("limit")
    if existing is not None:
        try:
            asked = int(existing.expression.this)
        except (AttributeError, TypeError, ValueError):
            asked = None
        if asked is not None and asked <= row_limit:
            return statement
    # `limit()` replaces rather than appends, so a looser existing limit is overwritten.
    return statement.limit(row_limit)
