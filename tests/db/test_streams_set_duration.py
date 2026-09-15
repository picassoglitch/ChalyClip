"""StreamsRepo.set_duration — backfill the real length onto a row that
was created without one (e.g. a live stream the ChalybOBS webhook inserted
with duration 0)."""

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


def _row(sid: str, tenant: str, duration_s: float) -> StreamRow:
    return StreamRow(
        id=sid,
        tenant_id=tenant,
        vod_url=f"live://chalybobs/{sid}",
        platform="live",
        duration_s=duration_s,
        source_video_path="v.mp4",
        source_audio_path="a.wav",
        created_at="2026-06-16T04:27:24+00:00",
    )


@pytest.mark.asyncio
async def test_backfills_when_row_has_zero(db: Database) -> None:
    t = await TenantsRepo(db).create(name="A")
    with bound_tenant(t.id):
        repo = StreamsRepo(db)
        await repo.upsert(_row("str_live", t.id, 0.0))
        await repo.set_duration("str_live", 3031.9)
        got = await repo.get("str_live")
    assert got is not None
    assert got.duration_s == pytest.approx(3031.9)


@pytest.mark.asyncio
async def test_does_not_clobber_existing_duration(db: Database) -> None:
    t = await TenantsRepo(db).create(name="A")
    with bound_tenant(t.id):
        repo = StreamsRepo(db)
        await repo.upsert(_row("str_done", t.id, 1200.0))  # already known
        await repo.set_duration("str_done", 9999.0)
        got = await repo.get("str_done")
    assert got is not None
    assert got.duration_s == pytest.approx(1200.0)  # untouched


@pytest.mark.asyncio
async def test_noop_when_new_duration_nonpositive(db: Database) -> None:
    t = await TenantsRepo(db).create(name="A")
    with bound_tenant(t.id):
        repo = StreamsRepo(db)
        await repo.upsert(_row("str_z", t.id, 0.0))
        await repo.set_duration("str_z", 0.0)
        got = await repo.get("str_z")
    assert got is not None
    assert got.duration_s == pytest.approx(0.0)  # still 0, no bogus write


@pytest.mark.asyncio
async def test_tenant_scoped(db: Database) -> None:
    ta = await TenantsRepo(db).create(name="A")
    tb = await TenantsRepo(db).create(name="B")
    with bound_tenant(ta.id):
        await StreamsRepo(db).upsert(_row("str_a", ta.id, 0.0))
    with bound_tenant(tb.id):
        # Tenant B cannot backfill tenant A's stream.
        await StreamsRepo(db).set_duration("str_a", 500.0)
    with bound_tenant(ta.id):
        got = await StreamsRepo(db).get("str_a")
    assert got is not None
    assert got.duration_s == pytest.approx(0.0)
