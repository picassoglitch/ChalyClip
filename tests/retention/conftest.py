"""Shared fixtures for retention sweeper tests."""

from __future__ import annotations

import datetime as _dt
from collections.abc import AsyncIterator
from pathlib import Path

import pytest_asyncio

from chalybclip.db import Database

from .._db_backend import migrated_database


@pytest_asyncio.fixture
async def retention_db(tmp_path: Path) -> AsyncIterator[Database]:
    """Migrated DB per test: SQLite, or Postgres when CHALYBCLIP_TEST_PG_DSN
    is set (prod stores created_at as TEXT there too)."""
    async for db in migrated_database(tmp_path, name="retention.db"):
        yield db


def days_ago_iso(days: int) -> str:
    """ISO 8601 UTC timestamp `days` days in the past — matches the
    `created_at` format every domain table uses."""
    return (_dt.datetime.now(_dt.UTC) - _dt.timedelta(days=days)).isoformat()
