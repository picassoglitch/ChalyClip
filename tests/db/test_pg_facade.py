"""Unit tests for the Postgres compatibility facade helpers.

These cover the pure translation/parsing logic that lets the existing
aiosqlite-style repos run unchanged on asyncpg. No live database is needed
— asyncpg is imported lazily inside `Database`, so this module loads even
when the driver isn't installed. End-to-end repo tests against real
Postgres land in Phase 4.
"""

from __future__ import annotations

from pathlib import Path

from chalybclip.db.connection import (
    Database,
    _is_postgres_dsn,
    _parse_rowcount,
    _qmark_to_dollar,
    _returns_rows,
)


class TestQmarkToDollar:
    def test_basic_positional(self) -> None:
        assert (
            _qmark_to_dollar("SELECT * FROM t WHERE a = ? AND b = ?")
            == "SELECT * FROM t WHERE a = $1 AND b = $2"
        )

    def test_no_placeholders(self) -> None:
        assert _qmark_to_dollar("SELECT 1") == "SELECT 1"

    def test_question_mark_in_string_literal_is_ignored(self) -> None:
        # The literal '?' must NOT become a parameter; the real param does.
        sql = "INSERT INTO t (msg, k) VALUES ('really?', ?)"
        assert _qmark_to_dollar(sql) == "INSERT INTO t (msg, k) VALUES ('really?', $1)"

    def test_escaped_single_quote_inside_literal(self) -> None:
        sql = "UPDATE t SET note = 'it''s a ? test' WHERE id = ?"
        assert (
            _qmark_to_dollar(sql)
            == "UPDATE t SET note = 'it''s a ? test' WHERE id = $1"
        )

    def test_question_mark_in_quoted_identifier_is_ignored(self) -> None:
        sql = 'SELECT "weird?col" FROM t WHERE id = ?'
        assert _qmark_to_dollar(sql) == 'SELECT "weird?col" FROM t WHERE id = $1'

    def test_numbering_is_sequential_across_many(self) -> None:
        sql = "VALUES (?, ?, ?, ?, ?)"
        assert _qmark_to_dollar(sql) == "VALUES ($1, $2, $3, $4, $5)"


class TestReturnsRows:
    def test_select(self) -> None:
        assert _returns_rows("SELECT * FROM t") is True

    def test_select_leading_whitespace(self) -> None:
        assert _returns_rows("   \n  select 1") is True

    def test_with_cte(self) -> None:
        assert _returns_rows("WITH x AS (SELECT 1) SELECT * FROM x") is True

    def test_values(self) -> None:
        assert _returns_rows("VALUES (1), (2)") is True

    def test_insert_without_returning(self) -> None:
        assert _returns_rows("INSERT INTO t (a) VALUES (?)") is False

    def test_update_without_returning(self) -> None:
        assert _returns_rows("UPDATE t SET a = ? WHERE id = ?") is False

    def test_insert_with_returning(self) -> None:
        assert _returns_rows("INSERT INTO t (a) VALUES (?) RETURNING id") is True

    def test_delete_with_returning(self) -> None:
        assert _returns_rows("DELETE FROM t WHERE id = ? RETURNING id") is True


class TestParseRowcount:
    def test_update(self) -> None:
        assert _parse_rowcount("UPDATE 3") == 3

    def test_delete(self) -> None:
        assert _parse_rowcount("DELETE 2") == 2

    def test_insert_two_numbers(self) -> None:
        # asyncpg INSERT tag is "INSERT <oid> <count>" — count is the tail.
        assert _parse_rowcount("INSERT 0 5") == 5

    def test_select(self) -> None:
        assert _parse_rowcount("SELECT 1") == 1

    def test_empty(self) -> None:
        assert _parse_rowcount("") == 0

    def test_no_trailing_number(self) -> None:
        assert _parse_rowcount("BEGIN") == 0


class TestIsPostgresDsn:
    def test_postgres_scheme(self) -> None:
        assert _is_postgres_dsn("postgres://u@h/db") is True

    def test_postgresql_scheme(self) -> None:
        assert _is_postgres_dsn("postgresql://u@h/db") is True

    def test_path_object(self) -> None:
        assert _is_postgres_dsn(Path("./chalybclip.db")) is False

    def test_plain_file_string(self) -> None:
        assert _is_postgres_dsn("./chalybclip.db") is False


class TestDatabaseBackendSelection:
    def test_sqlite_path_selects_sqlite(self) -> None:
        db = Database(Path("./x.db"))
        assert db.is_postgres is False
        assert db.path == Path("./x.db")

    def test_dsn_string_selects_postgres(self) -> None:
        db = Database("postgresql://u@h/db")
        assert db.is_postgres is True

    def test_plain_string_path_selects_sqlite(self) -> None:
        db = Database("./x.db")
        assert db.is_postgres is False
        assert db.path == Path("./x.db")
