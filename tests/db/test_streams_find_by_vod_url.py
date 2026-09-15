"""StreamsRepo.find_by_vod_url — the (tenant, vod_url) dedup lookup."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from chalybclip.db import Database, StreamsRepo, TenantsRepo, apply_migrations
from chalybclip.db.models import StreamRow
from chalybclip.tenancy import bound_tenant


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "s.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


def _row(sid: str, tenant: str, url: str, created_at: str) -> StreamRow:
    return StreamRow(
        id=sid,
        tenant_id=tenant,
        vod_url=url,
        platform="kick",
        duration_s=1.0,
        source_video_path="v.mp4",
        source_audio_path="a.wav",
        created_at=created_at,
    )


@pytest.mark.asyncio
async def test_returns_oldest_match(db: Database) -> None:
    t = await TenantsRepo(db).create(name="A")
    with bound_tenant(t.id):
        await StreamsRepo(db).upsert(_row("str_b", t.id, "url1", "2026-06-16T05:00:00"))
        await StreamsRepo(db).upsert(_row("str_a", t.id, "url1", "2026-06-16T03:00:00"))
        found = await StreamsRepo(db).find_by_vod_url("url1")
    assert found is not None
    assert found.id == "str_a"  # oldest created_at wins → canonical row


@pytest.mark.asyncio
async def test_none_when_absent(db: Database) -> None:
    t = await TenantsRepo(db).create(name="A")
    with bound_tenant(t.id):
        found = await StreamsRepo(db).find_by_vod_url("nope")
    assert found is None


@pytest.mark.asyncio
async def test_tenant_scoped(db: Database) -> None:
    ta = await TenantsRepo(db).create(name="A")
    tb = await TenantsRepo(db).create(name="B")
    with bound_tenant(ta.id):
        await StreamsRepo(db).upsert(_row("str_a", ta.id, "shared", "2026-06-16T01:00:00"))
    with bound_tenant(tb.id):
        found = await StreamsRepo(db).find_by_vod_url("shared")
    assert found is None  # tenant B cannot see tenant A's stream
