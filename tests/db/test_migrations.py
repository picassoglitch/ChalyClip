"""Tests for the migration runner â€” the lock-down's first line of defense."""

from __future__ import annotations

from pathlib import Path

import pytest

from chalybclip.db import Database, apply_migrations, schema_version
from chalybclip.db.migrations import MigrationError, _discover_migrations

from .._db_backend import pg_enabled

_CURRENT_HEAD = 57  # bumped each time we add a migration (057_platform_cooldowns)

# These assert SQLite implementation details (sqlite_master, PRAGMA). The
# Postgres schema + FK enforcement are covered by test_pg_migration.py.
_sqlite_only = pytest.mark.skipif(
    pg_enabled(), reason="SQLite-specific introspection; PG covered by test_pg_migration.py"
)


async def test_apply_migrations_brings_empty_db_to_current_head(db: Database) -> None:
    v = await apply_migrations(db)
    assert v == _CURRENT_HEAD
    assert await schema_version(db) == _CURRENT_HEAD


async def test_apply_migrations_is_idempotent(db: Database) -> None:
    v1 = await apply_migrations(db)
    v2 = await apply_migrations(db)
    assert v1 == _CURRENT_HEAD
    assert v2 == _CURRENT_HEAD


@_sqlite_only
async def test_schema_creates_all_phase_1_through_3_tables(migrated_db: Database) -> None:
    conn = await migrated_db.connect()
    cur = await conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    names = {row[0] for row in await cur.fetchall()}
    expected = {
        "schema_version",
        "tenants",
        "users",
        "api_tokens",
        "personas",
        "connected_accounts",
        "streams",
        "transcripts",
        "candidates",
        "clips",
        "variants",
        "llm_calls",
        "publish_jobs",
        "events",
        "visual_signals",
        "webhook_subscriptions",  # Phase 2
        "publish_metrics",  # Phase 3 #1
        "webhook_secret_versions",  # Phase 3 #2
        "platform_pacing_rules",  # Growth Engine Phase 1
        "clip_growth_scores",  # Growth Engine Phase 5/6
        "autoprog_locks",  # bulk auto-program idempotency lock
        "platform_cooldowns",  # abuse/rate-limit backoff
    }
    assert expected <= names, f"missing tables: {expected - names}"


@_sqlite_only
async def test_foreign_keys_are_enforced(migrated_db: Database) -> None:
    conn = await migrated_db.connect()
    cur = await conn.execute("PRAGMA foreign_keys")
    assert (await cur.fetchone())[0] == 1


async def test_apply_migrations_refuses_downgrade(db: Database) -> None:
    """If schema_version is newer than any available migration, apply raises."""
    await apply_migrations(db)
    conn = await db.connect()
    await conn.execute(
        "INSERT OR REPLACE INTO schema_version (rowid, version) VALUES (1, 99)"
    )
    await conn.commit()
    with pytest.raises(MigrationError, match="downgrades are not supported"):
        await apply_migrations(db)


def test_discover_migrations_rejects_bad_filenames(tmp_path: Path) -> None:
    (tmp_path / "001_init.sql").write_text("--", encoding="utf-8")
    (tmp_path / "no_number.sql").write_text("--", encoding="utf-8")
    with pytest.raises(MigrationError, match="does not match"):
        _discover_migrations(tmp_path)


def test_discover_migrations_requires_contiguous_versions(tmp_path: Path) -> None:
    """Skipped numbers (1, 3) must be flagged before any apply runs."""
    (tmp_path / "001_init.sql").write_text("--", encoding="utf-8")
    (tmp_path / "003_skip.sql").write_text("--", encoding="utf-8")
    # apply_migrations validates contiguity
    db = Database(tmp_path / "x.db")
    import asyncio

    with pytest.raises(MigrationError, match="contiguous"):
        asyncio.run(apply_migrations(db, migrations_dir=tmp_path))


async def test_apply_migrations_missing_dir_raises(db: Database, tmp_path: Path) -> None:
    with pytest.raises(MigrationError, match="not found"):
        await apply_migrations(db, migrations_dir=tmp_path / "does_not_exist")
