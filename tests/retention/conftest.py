"""Shared fixtures for retention sweeper tests."""

from __future__ import annotations

import datetime as _dt
from collections.abc import AsyncIterator
from pathlib import Path

import pytest_asyncio

from chalybclip.db import Database, apply_migrations


@pytest_asyncio.fixture
async def retention_db(tmp_path: Path) -> AsyncIterator[Database]:
    """Migrated SQLite DB per test."""
    db = Database(tmp_path / "retention.db")
    try:
        await apply_migrations(db)
        yield db
    finally:
        await db.close()


def days_ago_iso(days: int) -> str:
    """ISO 8601 UTC timestamp `days` days in the past — matches the
    `created_at` format every domain table uses."""
    return (_dt.datetime.now(_dt.UTC) - _dt.timedelta(days=days)).isoformat()
