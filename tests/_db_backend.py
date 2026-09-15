"""Shared test-DB backend selection: SQLite (default) or real Postgres.

Set ``CHALYBCLIP_TEST_PG_DSN`` (e.g. ``postgresql://postgres@127.0.0.1:5433/
chalybclip_test``) to run the fixture-backed suites against a real Postgres
instead of a throwaway SQLite file. This exercises the actual production
engine — the asyncpg facade, the ON CONFLICT idioms, the PG baseline — with
no dialect drift.

Isolation model: the Postgres baseline is applied once (idempotent), then
every test starts by TRUNCATE-ing every table (except ``schema_version``,
which would force a baseline re-apply). That composes with the facade's
autocommit-per-statement pool, where a wrap-in-a-transaction-and-rollback
scheme can't. SQLite keeps its file-per-test isolation, unchanged.

The target DB is mutated/truncated continuously — point the DSN only at a
disposable database, never production.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import aiosqlite
import asyncpg

from chalybclip.db import Database, apply_migrations

_PG_DSN = os.environ.get("CHALYBCLIP_TEST_PG_DSN")

if _PG_DSN:
    # In Postgres test mode, point the WHOLE app at the same database — not
    # just the fixtures. Background tasks (clip render, pipeline) resolve
    # their target via settings.db_target() = DATABASE_URL, so without this
    # they'd open the default SQLite file while the fixture data lives in PG.
    # Mirrors production, where DATABASE_URL is set and every path uses it.
    os.environ.setdefault("DATABASE_URL", _PG_DSN)
    from chalybclip.settings import get_settings

    get_settings.cache_clear()

# Backend-agnostic constraint-violation types for tests that assert a DB-level
# integrity error on raw SQL. aiosqlite raises sqlite3.IntegrityError; asyncpg
# raises its IntegrityConstraintViolationError family (FK / RESTRICT / UNIQUE).
INTEGRITY_ERRORS = (
    aiosqlite.IntegrityError,
    asyncpg.exceptions.IntegrityConstraintViolationError,
)


def pg_enabled() -> bool:
    """True when the suites should run against Postgres."""
    return bool(_PG_DSN)


async def _truncate_all(db: Database) -> None:
    """Wipe every public table (keeping the applied schema) for isolation."""
    conn = await db.connect()
    cur = await conn.execute(
        "SELECT tablename FROM pg_tables "
        "WHERE schemaname = 'public' AND tablename <> 'schema_version'"
    )
    names = [row[0] for row in await cur.fetchall()]
    if names:
        quoted = ", ".join(f'"{n}"' for n in names)
        await conn.execute(f"TRUNCATE {quoted} RESTART IDENTITY CASCADE")


async def migrated_database(
    tmp_path: Path, *, name: str = "test.db"
) -> AsyncIterator[Database]:
    """Yield a migrated Database for one test.

    Postgres: shared disposable DB, baseline applied once, tables truncated
    at the start of each test for isolation. SQLite: a fresh file per test
    at ``tmp_path / name`` — callers that also hand the same path to a
    background task (e.g. the clip renderer) pass a matching ``name``.
    """
    if pg_enabled():
        db = Database(_PG_DSN)  # type: ignore[arg-type]
        await apply_migrations(db)  # idempotent — baseline applies once
        await _truncate_all(db)
    else:
        db = Database(tmp_path / name)
        await apply_migrations(db)
    try:
        yield db
    finally:
        await db.close()
