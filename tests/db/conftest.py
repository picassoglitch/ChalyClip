"""Shared fixtures for DB tests.

Each test gets its own SQLite file under `tmp_path` so tests are
independent and can run in parallel later.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest_asyncio

from chalybclip.db import Database

from .._db_backend import migrated_database


@pytest_asyncio.fixture
async def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest_asyncio.fixture
async def db(db_path: Path) -> AsyncIterator[Database]:
    # Intentionally SQLite + un-migrated: the migration-runner tests need a
    # virgin database. The Postgres runner is covered by test_pg_migration.py.
    d = Database(db_path)
    try:
        yield d
    finally:
        await d.close()


@pytest_asyncio.fixture
async def migrated_db(tmp_path: Path) -> AsyncIterator[Database]:
    """A migrated database — Postgres when CHALYBCLIP_TEST_PG_DSN is set, else
    a fresh SQLite file. Repo tests run against the real engine on PG."""
    async for d in migrated_database(tmp_path):
        yield d
