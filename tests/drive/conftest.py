"""Shared fixtures for Drive watcher tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest_asyncio

from chalybclip.db import Database, apply_migrations


@pytest_asyncio.fixture
async def drive_db(tmp_path: Path) -> AsyncIterator[Database]:
    db = Database(tmp_path / "drive.db")
    try:
        await apply_migrations(db)
        yield db
    finally:
        await db.close()
