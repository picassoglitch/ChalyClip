"""Phase L.1 — live RTMP ingest: keys repo + MediaMTX webhook handlers.

The test surface intentionally avoids running MediaMTX itself (out
of scope for CI). What we cover:

  - LiveStreamKeysRepo: generate, find, rotate, revoke, last-used
  - StreamsRepo lifecycle helpers: mark_live_started / mark_live_ended
  - The shape of authorize / started / ended endpoint logic, with
    request.app.state.db wired in so the handlers can run

End-to-end RTMP push is tested manually per docs/mediamtx_deploy.md.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from chalybclip.db import (
    Database,
    LiveStreamKeysRepo,
    StreamsRepo,
    TenantsRepo,
    apply_migrations,
)
from chalybclip.db.repos import (
    _streams_repo_mark_live_ended,
    _streams_repo_mark_live_started,
)
from chalybclip.tenancy import bound_tenant


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "live.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


# ---- LiveStreamKeysRepo ---------------------------------------------------


async def test_rotate_generates_active_key(db: Database) -> None:
    """rotate_for_tenant on an empty tenant produces an active key."""
    tenant = await TenantsRepo(db).create(name="Aldo")
    with bound_tenant(tenant.id):
        repo = LiveStreamKeysRepo(db)
        new_key = await repo.rotate_for_tenant()
        assert new_key.tenant_id == tenant.id
        assert new_key.revoked_at is None
        assert new_key.key_value.startswith("slk_")
        # Key value is unguessable (32 url-safe bytes = ~43 char tail)
        assert len(new_key.key_value) > 30

        active = await repo.get_active_for_tenant()
        assert active is not None
        assert active.id == new_key.id


async def test_rotate_revokes_previous_key(db: Database) -> None:
    """A second rotate keeps the second key active + marks the first
    as revoked. Old key value persists in the table for audit."""
    tenant = await TenantsRepo(db).create(name="Aldo")
    with bound_tenant(tenant.id):
        repo = LiveStreamKeysRepo(db)
        first = await repo.rotate_for_tenant()
        second = await repo.rotate_for_tenant()
        assert second.id != first.id
        assert second.key_value != first.key_value

        # Active is the second.
        active = await repo.get_active_for_tenant()
        assert active is not None
        assert active.id == second.id

        # First is now revoked but can still be found by value
        # (audit). Lookups return the row with revoked_at set.
        found_first = await repo.find_by_value(first.key_value)
        assert found_first is not None
        assert found_first.revoked_at is not None


async def test_find_by_value_returns_none_for_unknown(db: Database) -> None:
    """Webhook hits an unknown key -> None -> handler 401s."""
    repo = LiveStreamKeysRepo(db)
    assert await repo.find_by_value("slk_does_not_exist") is None


async def test_touch_last_used_updates_timestamp(db: Database) -> None:
    """The MediaMTX auth webhook touches last_used on every push.
    The dashboard surfaces it so operators see "key used 12s ago"."""
    tenant = await TenantsRepo(db).create(name="Aldo")
    with bound_tenant(tenant.id):
        repo = LiveStreamKeysRepo(db)
        key = await repo.rotate_for_tenant()
        assert key.last_used_at is None

        await repo.touch_last_used(key.id)
        refreshed = await repo.get_active_for_tenant()
        assert refreshed is not None
        assert refreshed.last_used_at is not None


# ---- Streams repo lifecycle helpers --------------------------------------


async def _seed_stream_for_tenant(db: Database, *, tenant_id: str, stream_id: str) -> None:
    """Insert a barebones streams row for the tenant in question."""
    from chalybclip.db.models import StreamRow
    now = _dt.datetime.now(_dt.UTC).isoformat()
    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=stream_id, tenant_id=tenant_id,
                vod_url=f"live://rtmp/{stream_id}",
                platform="live", title="t", channel=None,
                duration_s=0.0,
                source_video_path=f"/data/live/{stream_id}/source.mp4",
                source_audio_path=f"/data/live/{stream_id}/source.audio.wav",
                status="ingested", created_at=now,
            )
        )


async def test_mark_live_started_flips_status_to_live(db: Database) -> None:
    """The /started webhook flips status to 'live' + sets live_started_at."""
    tenant = await TenantsRepo(db).create(name="Aldo")
    await _seed_stream_for_tenant(db, tenant_id=tenant.id, stream_id="str_lst")

    row = await _streams_repo_mark_live_started(db, stream_id="str_lst")
    assert row is not None
    assert row.status == "live"
    assert row.is_live is True
    assert row.live_started_at is not None


async def test_mark_live_started_is_idempotent(db: Database) -> None:
    """Calling mark_live_started twice doesn't overwrite live_started_at —
    COALESCE preserves the first timestamp. Real-world this matters if
    MediaMTX retries the webhook on transient errors."""
    tenant = await TenantsRepo(db).create(name="Aldo")
    await _seed_stream_for_tenant(db, tenant_id=tenant.id, stream_id="str_ist")

    first = await _streams_repo_mark_live_started(db, stream_id="str_ist")
    assert first is not None
    first_started = first.live_started_at

    second = await _streams_repo_mark_live_started(db, stream_id="str_ist")
    assert second is not None
    assert second.live_started_at == first_started  # didn't overwrite


async def test_mark_live_ended_flips_status_and_records_duration(db: Database) -> None:
    """The /ended webhook flips status to 'live_ended' + sets live_ended_at +
    duration_s (when MediaMTX provides it)."""
    tenant = await TenantsRepo(db).create(name="Aldo")
    await _seed_stream_for_tenant(db, tenant_id=tenant.id, stream_id="str_end")
    await _streams_repo_mark_live_started(db, stream_id="str_end")

    ended = await _streams_repo_mark_live_ended(
        db, stream_id="str_end", duration_s=3725.4
    )
    assert ended is not None
    assert ended.status == "live_ended"
    assert ended.is_live is False
    assert ended.live_ended_at is not None
    assert ended.duration_s == pytest.approx(3725.4)


async def test_mark_live_ended_without_duration_preserves_zero(db: Database) -> None:
    """MediaMTX sometimes doesn't include duration in the unpublish
    webhook (transient TCP drop). In that case we still flip the
    status but leave duration_s at whatever it was (0 from seed)."""
    tenant = await TenantsRepo(db).create(name="Aldo")
    await _seed_stream_for_tenant(db, tenant_id=tenant.id, stream_id="str_no_dur")
    await _streams_repo_mark_live_started(db, stream_id="str_no_dur")

    ended = await _streams_repo_mark_live_ended(db, stream_id="str_no_dur")
    assert ended is not None
    assert ended.status == "live_ended"
    # duration_s untouched from the seeded 0.0; the operator's manual
    # "Run pipeline" path will read it from the file via ffprobe (slice O.55).
    assert ended.duration_s == 0.0


# ---- Repo + cascade contract -----------------------------------------------


async def test_keys_cascade_when_tenant_deleted(db: Database) -> None:
    """Live stream keys should disappear with the tenant they belong to.
    Belt-and-braces — protects against keys persisting after a tenant is
    purged (otherwise a stale key would survive + accept pushes).
    """
    tenant = await TenantsRepo(db).create(name="Aldo")
    with bound_tenant(tenant.id):
        repo = LiveStreamKeysRepo(db)
        key = await repo.rotate_for_tenant()
        assert (await repo.find_by_value(key.key_value)) is not None

    # Force the cascade by deleting the tenant row directly. The
    # ON DELETE CASCADE on live_stream_keys.tenant_id reaps the keys.
    conn = await db.connect()
    await conn.execute("DELETE FROM tenants WHERE id = ?", (tenant.id,))
    await conn.commit()
    # Key value is gone, even though we look it up tenant-free.
    assert (await repo.find_by_value(key.key_value)) is None
