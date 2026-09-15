"""Shared fixtures for pipeline-recovery tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest_asyncio

from chalybclip.db import Database, apply_migrations


@pytest_asyncio.fixture
async def recovery_db(tmp_path: Path) -> AsyncIterator[Database]:
    """Migrated SQLite DB per test."""
    db = Database(tmp_path / "recovery.db")
    try:
        await apply_migrations(db)
        yield db
    finally:
        await db.close()
