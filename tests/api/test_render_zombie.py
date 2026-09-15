"""Render Migration R5 + R6 — pin the recovery contracts.

R5 — the background runner's outer guard catches ANY exception so
     state always exits 'rendering'. The hybrid recorder ALSO
     post-condition-checks that ffmpeg actually wrote the output
     (Railway's silent disk-full failure pattern).

R6 — when state has been 'rendering' for >= 15 min, /render-status
     marks it failed + returns a retry hint, and /download
     auto-recovers by resetting + scheduling a fresh render. Without
     this, a clip whose background task crashed pre-R5 (or got
     killed by Railway / OOM) stays stuck forever with the UI
     polling status indefinitely (the bug the operator hit twice
     today, observable as hundreds of identical /render-status
     200s in the logs).
"""

from __future__ import annotations

import datetime as _dt
import json as _json
from pathlib import Path

import httpx
import pytest

from chalybclip.db import Database
from chalybclip.tenancy import bound_tenant

from .conftest import auth


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


def _ago(minutes: int) -> str:
    """ISO-format timestamp N minutes in the past."""
    return (
        _dt.datetime.now(_dt.UTC)
        - _dt.timedelta(minutes=minutes)
    ).isoformat()


async def _seed_clip_in_state(
    db: Database,
    tenant_id: str,
    *,
    clip_id: str,
    render_state: str,
    render_started_at: str | None,
    tmp_path: Path,
) -> Path:
    stream_id = f"str_{clip_id}"
    clip_dir = tmp_path / "clips" / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    source_path = clip_dir / "clip.mp4"
    source_path.write_bytes(b"fake" * 100)

    with bound_tenant(tenant_id):
        conn = await db.connect()
        await conn.execute(
            "INSERT INTO streams "
            "(id, tenant_id, vod_url, platform, title, channel, "
            "duration_s, source_video_path, source_audio_path, status, "
            "created_at) "
            "VALUES (?, ?, 'https://kick.com/x/v/1', 'kick', NULL, NULL, "
            "30, '/tmp/v.mp4', '/tmp/a.wav', 'ingested', ?) ON CONFLICT DO NOTHING",
            (stream_id, tenant_id, _now()),
        )
        await conn.execute(
            "INSERT INTO clips "
            "(id, stream_id, tenant_id, start_s, end_s, duration_s, "
            "width, height, path, status, created_at, "
            "render_state, render_progress_pct, render_started_at) "
            "VALUES (?, ?, ?, 0, 30, 30, 1080, 1920, ?, 'cut', ?, "
            "?, ?, ?)",
            (
                clip_id, stream_id, tenant_id, str(source_path), _now(),
                render_state, 42, render_started_at,
            ),
        )
        await conn.commit()
    return clip_dir


# ---- R6: /render-status zombie detection ----


async def test_render_status_marks_zombie_failed_after_timeout(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    tmp_path: Path,
) -> None:
    """A render that's been 'rendering' for 20 minutes with no output
    is a zombie. Status endpoint flips it to failed with a retry hint
    so the UI shows a retry button instead of polling forever."""
    tenant_id = tenants["alice"]["id"]
    await _seed_clip_in_state(
        db, tenant_id,
        clip_id="clp_zombie",
        render_state="rendering",
        render_started_at=_ago(20),
        tmp_path=tmp_path,
    )

    r = await client.get(
        "/dashboard/clips/clp_zombie/render-status",
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "failed"
    assert "timed out" in (body.get("error") or "").lower()
    assert body["retry_url"].endswith(
        "render-retry?quality=1080"
    )

    # DB row was flipped to failed (idempotent on subsequent polls).
    with bound_tenant(tenant_id):
        clip = await __import__(
            "chalybclip.db", fromlist=["ClipsRepo"]
        ).ClipsRepo(db).get("clp_zombie")
    assert clip is not None
    assert clip.render_state == "failed"


async def test_render_status_does_not_mark_zombie_within_timeout(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    tmp_path: Path,
) -> None:
    """A render that's only been going for 2 minutes is NOT a zombie.
    Endpoint returns the rendering state unchanged so the UI keeps
    showing the progress bar."""
    tenant_id = tenants["alice"]["id"]
    await _seed_clip_in_state(
        db, tenant_id,
        clip_id="clp_fresh_render",
        render_state="rendering",
        render_started_at=_ago(2),
        tmp_path=tmp_path,
    )

    r = await client.get(
        "/dashboard/clips/clp_fresh_render/render-status",
        headers=auth(tenants["alice"]["token"]),
    )
    body = r.json()
    assert body["state"] == "rendering"
    assert body["progress_pct"] == 42
    assert "retry_url" not in body


async def test_render_status_handles_missing_started_at(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    tmp_path: Path,
) -> None:
    """Legacy clip whose render_state was flipped before T1's
    started_at column was set — don't false-positive as zombie.
    Surface the current state and let the operator click Retry
    if they want."""
    tenant_id = tenants["alice"]["id"]
    await _seed_clip_in_state(
        db, tenant_id,
        clip_id="clp_no_started",
        render_state="rendering",
        render_started_at=None,
        tmp_path=tmp_path,
    )

    r = await client.get(
        "/dashboard/clips/clp_no_started/render-status",
        headers=auth(tenants["alice"]["token"]),
    )
    body = r.json()
    assert body["state"] == "rendering"
    assert "retry_url" not in body


# ---- R6: /download auto-recovery ----


async def test_download_auto_recovers_zombie_render(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operator clicks Download on a clip stuck rendering for 20 min:
    don't return 202+status-poll-forever. Reset to idle, schedule
    a fresh background task, return 202 with the new render's
    status_url. This is the recovery path for the two clips already
    stuck on the operator's Railway deploy."""
    tenant_id = tenants["alice"]["id"]
    await _seed_clip_in_state(
        db, tenant_id,
        clip_id="clp_zombie_dl",
        render_state="rendering",
        render_started_at=_ago(20),
        tmp_path=tmp_path,
    )

    # Block the real renderer; we only want to pin the endpoint's
    # response shape + the state transition.
    scheduled: list[dict] = []
    async def fake_runner(**kwargs: object) -> None:
        scheduled.append(dict(kwargs))
    monkeypatch.setattr(
        "chalybclip.api._clip_render.render_clip_in_background", fake_runner,
    )

    r = await client.get(
        "/dashboard/clips/clp_zombie_dl/download",
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 202
    body = r.json()
    assert body["state"] == "rendering"
    # Fresh schedule fired — the auto-recovery effectively kicked
    # off a new render rather than returning the polled-forever
    # 202 from before.
    assert len(scheduled) == 1
    assert scheduled[0]["clip_id"] == "clp_zombie_dl"


# ---- R9: /render-status race guard ----


async def test_render_status_does_not_unlink_during_active_render(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    tmp_path: Path,
) -> None:
    """The actual root cause that haunted the operator for a third
    session: /render-status is polled ~1/s during a render. When
    ffmpeg is partway through writing, the cache file exists but is
    smaller than the 1 MB servable floor → R3's debris-eviction would
    unlink it. On Linux that orphans the inode (ffmpeg's fd is still
    open) and the encode "succeeds" with rc=0 but the directory entry
    is gone. Operator sees: 'ffmpeg rc=0 but output missing'.

    Pin: while state='rendering', /render-status must NOT touch the
    cache file even if it's currently sub-1MB."""
    tenant_id = tenants["alice"]["id"]
    clip_dir = await _seed_clip_in_state(
        db, tenant_id,
        clip_id="clp_mid_encode",
        render_state="rendering",
        render_started_at=_ago(2),  # within the 15-min zombie window
        tmp_path=tmp_path,
    )
    # Simulate ffmpeg mid-encode: file exists, has the ftyp box
    # ffmpeg wrote on open, but is well under the 1 MB threshold.
    cache_path = clip_dir / "clip_render_1080.mp4"
    cache_path.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 50_000)
    pre_bytes = cache_path.stat().st_size

    # Poll many times — the operator's UI does this every second.
    # Each poll used to unlink the file; now each poll must leave
    # it alone.
    for _ in range(10):
        r = await client.get(
            "/dashboard/clips/clp_mid_encode/render-status",
            headers=auth(tenants["alice"]["token"]),
        )
        assert r.status_code == 200
        body = r.json()
        assert body["state"] == "rendering"

    # The file must STILL be on disk after all those polls. If it
    # vanishes, we've regressed the race.
    assert cache_path.exists(), (
        "/render-status unlinked the cache file while state=rendering "
        "— this races with the in-flight ffmpeg encode and orphans "
        "the inode"
    )
    assert cache_path.stat().st_size == pre_bytes


async def test_render_status_still_evicts_debris_when_not_rendering(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    tmp_path: Path,
) -> None:
    """The R9 guard must NOT break R3's actual purpose: when state is
    'idle' / 'failed' / 'ready' and the file on disk is debris (sub-
    1MB), evict it so the next click triggers a clean render."""
    tenant_id = tenants["alice"]["id"]
    clip_dir = await _seed_clip_in_state(
        db, tenant_id,
        clip_id="clp_stale_debris",
        # state='ready' but cache is debris — the scenario where a
        # pre-R5 render crashed and left a partial file. R3 was
        # designed to clear this.
        render_state="ready",
        render_started_at=_ago(20),
        tmp_path=tmp_path,
    )
    cache_path = clip_dir / "clip_render_1080.mp4"
    cache_path.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 500)
    assert cache_path.exists()

    r = await client.get(
        "/dashboard/clips/clp_stale_debris/render-status",
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 200
    # Debris was correctly evicted (the eviction path still fires when
    # there's no race with a running encode).
    assert not cache_path.exists()


async def test_download_does_not_recover_fresh_render(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A render only 2 minutes in is still legitimately in flight.
    Download click returns 202 status-poll without re-scheduling."""
    tenant_id = tenants["alice"]["id"]
    await _seed_clip_in_state(
        db, tenant_id,
        clip_id="clp_active",
        render_state="rendering",
        render_started_at=_ago(2),
        tmp_path=tmp_path,
    )

    scheduled: list[dict] = []
    async def fake_runner(**kwargs: object) -> None:
        scheduled.append(dict(kwargs))
    monkeypatch.setattr(
        "chalybclip.api._clip_render.render_clip_in_background", fake_runner,
    )

    r = await client.get(
        "/dashboard/clips/clp_active/download",
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 202
    body = r.json()
    assert body["state"] == "rendering"
    # No new schedule — the existing task is presumed alive.
    assert scheduled == []
