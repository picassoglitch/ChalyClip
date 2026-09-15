"""Reprocess-queue cleanup: cancel scheduled posts + reset/re-stamp clips."""

from __future__ import annotations

import pytest

from chalybclip.db import (
    ClipsRepo,
    Database,
    ZernioPublishesRepo,
    apply_migrations,
)
from chalybclip.publish.reprocess import reprocess_scheduled_queue
from chalybclip.settings import Settings
from chalybclip.tenancy import bound_tenant


class _FakeClient:
    """Records the posts we asked Zernio to cancel."""

    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def delete_post(self, post_id: str) -> None:
        self.deleted.append(post_id)


async def _seed(
    db: Database, tid: str, *, clip_id: str, post_id: str,
    clip_path: str = "/clip.mp4", with_stream: bool = True,
) -> None:
    conn = await db.connect()
    await conn.execute("PRAGMA foreign_keys=OFF")
    if with_stream:
        await conn.execute(
            "INSERT INTO tenants (id, name, created_at) VALUES (?, 'T', '2026-01-01')",
            (tid,),
        )
        await conn.execute(
            "INSERT INTO streams (id, tenant_id, vod_url, platform, channel, duration_s, "
            "source_video_path, source_audio_path, status, created_at) "
            "VALUES ('str_1', ?, 'http://x', 'kick', 'aldostream', 100, '/v', '/a', 'done', '2026-01-01')",
            (tid,),
        )
    await conn.execute(
        "INSERT INTO clips (id, stream_id, tenant_id, start_s, end_s, duration_s, "
        "width, height, path, status, created_at) "
        "VALUES (?, 'str_1', ?, 0, 10, 10, 1080, 1920, ?, 'published', '2026-01-01')",
        (clip_id, tid, clip_path),
    )
    await conn.commit()
    with bound_tenant(tid):
        await ZernioPublishesRepo(db).record(
            post_id=post_id, tenant_id=tid, clip_id=clip_id,
            platforms=["kick"], content="old plain caption", status="scheduled",
            scheduled_for="2026-07-01T10:00:00+00:00",
        )


def _real_source(tmp_path, name: str = "clip.mp4") -> str:
    """A clip source file that actually exists on disk — reprocess now skips
    clips whose source retention already reclaimed."""
    p = tmp_path / name
    p.write_bytes(b"fake mp4 bytes")
    return str(p)


@pytest.mark.asyncio
async def test_dry_run_changes_nothing(tmp_path) -> None:
    db = Database(tmp_path / "x.db")
    await apply_migrations(db)
    await _seed(
        db, "ten_a", clip_id="clp_1", post_id="post_1",
        clip_path=_real_source(tmp_path),
    )
    client = _FakeClient()

    summary = await reprocess_scheduled_queue(
        db, "ten_a", client=client, settings=Settings(), dry_run=True,
    )
    assert summary.scheduled_found == 1
    assert summary.clip_ids == ["clp_1"]
    assert summary.canceled == 0
    assert client.deleted == []  # nothing cancelled
    with bound_tenant("ten_a"):
        clip = await ClipsRepo(db).get("clp_1")
    assert clip.status == "published"  # untouched
    await db.close()


@pytest.mark.asyncio
async def test_reprocess_cancels_and_resets(tmp_path) -> None:
    db = Database(tmp_path / "x.db")
    await apply_migrations(db)
    await _seed(
        db, "ten_b", clip_id="clp_1", post_id="post_1",
        clip_path=_real_source(tmp_path),
    )
    client = _FakeClient()

    summary = await reprocess_scheduled_queue(
        db, "ten_b", client=client, settings=Settings(), dry_run=False,
    )
    assert summary.canceled == 1
    assert summary.clips_reset == 1
    assert summary.skipped_missing_source == []
    assert client.deleted == ["post_1"]  # cancelled on Zernio

    pubs = ZernioPublishesRepo(db)
    # Record hard-deleted → clip no longer blocked from re-scheduling.
    assert await pubs.exists_for_clip("ten_b", "clp_1") is False
    with bound_tenant("ten_b"):
        clip = await ClipsRepo(db).get("clp_1")
    assert clip.status == "approved"  # re-eligible
    # Overlay re-stamped with the viral default: captions on + kick marca.
    ov = clip.overlay_config or {}
    assert ov.get("captions", {}).get("enabled") is True
    assert ov.get("banner", {}).get("enabled") is True
    assert ov.get("banner", {}).get("platform") == "kick"
    assert ov.get("banner", {}).get("url") == "aldostream"  # source streamer name
    await db.close()


@pytest.mark.asyncio
async def test_missing_source_clip_left_completely_untouched(tmp_path) -> None:
    """A clip whose SOURCE file retention already reclaimed must be skipped
    BEFORE anything destructive: its scheduled post stays live, its cached
    render stays servable, its status stays 'published' — a reset could never
    re-render it and would strand it with nothing to serve."""
    db = Database(tmp_path / "x.db")
    await apply_migrations(db)
    await _seed(
        db, "ten_d", clip_id="clp_ok", post_id="post_ok",
        clip_path=_real_source(tmp_path),
    )
    await _seed(
        db, "ten_d", clip_id="clp_gone", post_id="post_gone",
        clip_path=str(tmp_path / "reclaimed" / "gone.mp4"),  # never created
        with_stream=False,
    )
    client = _FakeClient()

    summary = await reprocess_scheduled_queue(
        db, "ten_d", client=client, settings=Settings(), dry_run=False,
    )
    assert summary.scheduled_found == 2
    assert summary.skipped_missing_source == ["clp_gone"]
    assert summary.clip_ids == ["clp_ok"]
    assert summary.canceled == 1
    assert summary.clips_reset == 1
    assert client.deleted == ["post_ok"]  # the missing-source post NOT cancelled

    pubs = ZernioPublishesRepo(db)
    # The skipped clip's post record survives (still scheduled)…
    assert await pubs.exists_for_clip("ten_d", "clp_gone") is True
    with bound_tenant("ten_d"):
        gone = await ClipsRepo(db).get("clp_gone")
        ok = await ClipsRepo(db).get("clp_ok")
    # …and its clip state is untouched, while the healthy clip was reset.
    assert gone.status == "published"
    assert ok.status == "approved"
    await db.close()


@pytest.mark.asyncio
async def test_no_scheduled_is_clean_noop(tmp_path) -> None:
    db = Database(tmp_path / "x.db")
    await apply_migrations(db)
    conn = await db.connect()
    await conn.execute("INSERT INTO tenants (id, name, created_at) VALUES ('ten_c','T','2026-01-01')")
    await conn.commit()
    summary = await reprocess_scheduled_queue(
        db, "ten_c", client=_FakeClient(), settings=Settings(), dry_run=False,
    )
    assert summary.scheduled_found == 0
    assert summary.clips_reset == 0
    await db.close()
