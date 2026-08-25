"""Validation of model-authored SQL.

Security-critical, so the tests are mostly adversarial: every case here is a way a
string-matching check would let a write through, or let a query read a table it should not
see. The point of parsing instead of matching is that these stop being clever.
"""

from __future__ import annotations

import pytest

from cortex.tools.sqlguard import DEFAULT_ROW_LIMIT, UnsafeSQL, check

ALLOWED = {"customers", "deals", "events"}


class TestReadsAreAccepted:
    def test_a_plain_select(self) -> None:
        result = check("SELECT name FROM customers", allowed_tables=ALLOWED)
        assert result.tables == frozenset({"customers"})

    def test_a_join_across_allowed_tables(self) -> None:
        result = check(
            "SELECT c.name, d.amount FROM customers c JOIN deals d ON d.cid = c.id",
            allowed_tables=ALLOWED,
        )
        assert result.tables == frozenset({"customers", "deals"})

    def test_a_cte_is_not_treated_as_an_external_table(self) -> None:
        """A CTE is referenced like a table but defined in the query, so requiring it on
        the allowlist would reject correct SQL."""
        result = check(
            "WITH recent AS (SELECT * FROM deals) SELECT count(*) FROM recent",
            allowed_tables=ALLOWED,
        )
        assert result.tables == frozenset({"deals"})

    def test_a_union_is_a_read(self) -> None:
        result = check(
            "SELECT id FROM customers UNION SELECT id FROM deals", allowed_tables=ALLOWED
        )
        assert result.tables == frozenset({"customers", "deals"})

    def test_a_trailing_semicolon_is_tolerated(self) -> None:
        assert check("SELECT 1 FROM customers;", allowed_tables=ALLOWED).sql


class TestWritesAreRefused:
    @pytest.mark.parametrize(
        "sql",
        [
            "DELETE FROM customers",
            "UPDATE customers SET name = 'x'",
            "DROP TABLE customers",
            "ALTER TABLE customers ADD COLUMN x INT",
            "CREATE TABLE t (a INT)",
            "TRUNCATE TABLE customers",
        ],
    )
    def test_plain_writes(self, sql: str) -> None:
        with pytest.raises(UnsafeSQL):
            check(sql, allowed_tables=ALLOWED)

    def test_an_insert_that_contains_a_select(self) -> None:
        """The case a `startswith("SELECT")` check waves through, and a
        `"SELECT" in sql` check waves through even more enthusiastically."""
        with pytest.raises(UnsafeSQL, match="INSERT"):
            check("INSERT INTO deals SELECT * FROM customers", allowed_tables=ALLOWED)

    def test_a_write_stacked_behind_a_read(self) -> None:
        """How a read becomes a read *and* a write. Refused rather than truncated to the
        first statement: silently running half of what was asked is worse."""
        with pytest.raises(UnsafeSQL, match="statements"):
            check("SELECT 1 FROM customers; DROP TABLE customers", allowed_tables=ALLOWED)

    def test_a_write_hidden_behind_a_comment(self) -> None:
        with pytest.raises(UnsafeSQL, match="DROP"):
            check("/* harmless */ DROP TABLE customers", allowed_tables=ALLOWED)

    def test_a_write_inside_a_cte(self) -> None:
        """Checked across the whole tree, not only at the root, so a mutation nested
        inside a query that *looks* like a SELECT is still caught."""
        with pytest.raises(UnsafeSQL):
            check(
                "WITH x AS (DELETE FROM customers RETURNING id) SELECT * FROM x",
                allowed_tables=ALLOWED,
                dialect="postgres",
            )

    def test_an_unparseable_statement_is_refused_not_guessed(self) -> None:
        with pytest.raises(UnsafeSQL):
            check("SELCT FROM WHERE", allowed_tables=ALLOWED)

    def test_an_empty_query(self) -> None:
        with pytest.raises(UnsafeSQL, match="empty"):
            check("   ", allowed_tables=ALLOWED)


class TestTableAllowlist:
    def test_an_unlisted_table_is_refused(self) -> None:
        with pytest.raises(UnsafeSQL, match="not available"):
            check("SELECT * FROM salaries", allowed_tables=ALLOWED)

    def test_the_message_names_what_may_be_read(self) -> None:
        """The message is fed back to the model as an observation. Naming the available
        tables gets a corrected query; "invalid SQL" gets the same mistake again."""
        with pytest.raises(UnsafeSQL) as caught:
            check("SELECT * FROM salaries", allowed_tables=ALLOWED)
        message = str(caught.value)
        assert "salaries" in message
        assert "customers" in message and "deals" in message

    def test_an_unlisted_table_nested_in_a_subquery(self) -> None:
        """The interesting case: the outer query is entirely legitimate."""
        with pytest.raises(UnsafeSQL, match="salaries"):
            check(
                "SELECT * FROM customers WHERE EXISTS (SELECT 1 FROM salaries)",
                allowed_tables=ALLOWED,
            )

    def test_no_allowlist_permits_any_table(self) -> None:
        """Correct for a benchmark database containing nothing else, and wrong for a
        shared one — which is why it is an explicit argument rather than the default."""
        assert check("SELECT * FROM anything", allowed_tables=None).tables == frozenset(
            {"anything"}
        )


class TestRowLimit:
    def test_an_unbounded_query_is_bounded(self) -> None:
        """An unbounded query against a real table is a cost incident, and the result
        would be too large to use as evidence."""
        result = check("SELECT * FROM customers", allowed_tables=ALLOWED, row_limit=50)
        assert "LIMIT 50" in result.sql.upper()
        assert result.row_limit == 50

    def test_a_stricter_limit_is_preserved(self) -> None:
        """Tightening only. If the model asked for 5 rows, returning 1000 is wrong."""
        result = check("SELECT * FROM customers LIMIT 5", allowed_tables=ALLOWED, row_limit=1000)
        assert "LIMIT 5" in result.sql.upper()

    def test_a_looser_limit_is_tightened(self) -> None:
        result = check(
            "SELECT * FROM customers LIMIT 999999", allowed_tables=ALLOWED, row_limit=100
        )
        assert "LIMIT 100" in result.sql.upper()

    def test_the_default_limit_applies(self) -> None:
        result = check("SELECT * FROM customers", allowed_tables=ALLOWED)
        assert result.row_limit == DEFAULT_ROW_LIMIT
