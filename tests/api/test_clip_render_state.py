"""Render Migration T1 — background render + state polling endpoints.

Three contracts pinned:
  1. Cache hit on disk → 200 FileResponse direct download (legacy clips
     that were rendered before T1 keep working without a state reset).
  2. Cache miss + state='idle' → flips state to 'rendering' atomically,
     schedules the background runner, returns 202 JSON status with the
     status_url the UI polls.
  3. Cache miss + state='failed' → 409 JSON status with the error
     message + retry_url the UI surfaces as a "Try again" button.

Cookie-login fixture is broken pre-existing — we use bearer-header auth
(same pattern Task 3 + the upload background-runner refactor used).
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import httpx
import pytest

from chalybclip.api import create_app
from chalybclip.db import ClipsRepo, Database
from chalybclip.db.models import ClipRow
from chalybclip.tenancy import bound_tenant

from .conftest import auth


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


async def _seed_clip(
    db: Database,
    tenant_id: str,
    *,
    clip_id: str = "clp_test",
    output_dir: Path,
    render_state: str = "idle",
    render_progress_pct: int = 0,
    render_error: str | None = None,
) -> Path:
    """Insert a clip row + create the source MP4 on disk so the
    download endpoint passes the source-exists check."""
    stream_id = "str_test"
    clip_dir = output_dir / stream_id / "clips" / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / "clip.mp4"
    clip_path.write_bytes(b"\x00fakeclip")

    with bound_tenant(tenant_id):
        conn = await db.connect()
        # Streams row FIRST (clips FK references it).
        await conn.execute(
            "INSERT INTO streams "
            "(id, tenant_id, vod_url, platform, title, channel, "
            "duration_s, source_video_path, source_audio_path, status, "
            "created_at) "
            "VALUES (?, ?, 'https://kick.com/x/v/1', 'kick', NULL, NULL, "
            "30, '/tmp/v.mp4', '/tmp/a.wav', 'ingested', ?) ON CONFLICT DO NOTHING",
            (stream_id, tenant_id, _now()),
        )
        # Then the clip row with the render_* columns the tests assert on.
        await conn.execute(
            "INSERT INTO clips ("
            "id, stream_id, tenant_id, candidate_id, start_s, end_s, "
            "duration_s, width, height, path, status, created_at, "
            "render_state, render_progress_pct, render_error) "
            "VALUES (?, ?, ?, NULL, 0, 30, 30, 1080, 1920, ?, 'cut', ?, "
            "?, ?, ?)",
            (
                clip_id, stream_id, tenant_id, str(clip_path), _now(),
                render_state, render_progress_pct, render_error,
            ),
        )
        await conn.commit()
    return clip_path


async def test_render_status_idle_when_no_cache(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh clip → state=idle, no download URL, progress=0."""
    monkeypatch.setattr(
        "chalybclip.settings.get_settings",
        lambda: type("S", (), {
            "default_output_dir": str(tmp_path),
            "db_path": str(tmp_path / "x.db"),
            "public_url": "",
        })(),
    )
    tenant_id = tenants["alice"]["id"]
    await _seed_clip(db, tenant_id, output_dir=tmp_path)

    r = await client.get(
        "/dashboard/clips/clp_test/render-status",
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "idle"
    assert body["progress_pct"] == 0
    assert body.get("error") is None
    assert "download_url" not in body


async def test_render_status_ready_when_cache_on_disk(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy clip that was already rendered pre-T1 — the cache file
    on disk takes precedence over the DB state column. UI gets a
    download_url and renders the "Download MP4" link."""
    monkeypatch.setattr(
        "chalybclip.settings.get_settings",
        lambda: type("S", (), {
            "default_output_dir": str(tmp_path),
            "db_path": str(tmp_path / "x.db"),
            "public_url": "",
        })(),
    )
    tenant_id = tenants["alice"]["id"]
    clip_path = await _seed_clip(db, tenant_id, output_dir=tmp_path)
    # Pre-create the rendered MP4 in the same dir as the source clip.
    # Render Migration R3 — the cache file must look like a real MP4:
    # ISO BMFF ftyp box at offset 4 + size >= 1 MB. The old test wrote
    # 9 bytes which is correctly debris under the new sanity gate.
    cache = clip_path.parent / "clip_render_1080.mp4"
    cache.write_bytes(
        b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2mp41"
        + b"\x00" * 1_100_000,
    )

    r = await client.get(
        "/dashboard/clips/clp_test/render-status",
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "ready"
    assert body["progress_pct"] == 100
    assert body["download_url"].endswith(
        "/dashboard/clips/clp_test/download?quality=1080"
    )


async def test_download_cache_hit_returns_mp4(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the rendered MP4 is on disk, /download returns the file
    directly. Same behavior whether DB state column says 'idle' or
    'ready' — disk truth wins."""
    monkeypatch.setattr(
        "chalybclip.settings.get_settings",
        lambda: type("S", (), {
            "default_output_dir": str(tmp_path),
            "db_path": str(tmp_path / "x.db"),
            "public_url": "",
        })(),
    )
    tenant_id = tenants["alice"]["id"]
    clip_path = await _seed_clip(db, tenant_id, output_dir=tmp_path)
    rendered = clip_path.parent / "clip_render_1080.mp4"
    # Render Migration R3 — must look like a valid MP4 (ftyp box +
    # >= 1 MB) for the sanity gate to treat it as cache-servable.
    valid_mp4 = (
        b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2mp41"
        + b"\x00" * 1_100_000
    )
    rendered.write_bytes(valid_mp4)

    r = await client.get(
        "/dashboard/clips/clp_test/download",
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
    assert "attachment" in r.headers.get("content-disposition", "")
    # FileResponse streams the file's actual bytes verbatim.
    assert r.content == valid_mp4


async def test_download_idle_state_kicks_off_background_and_returns_202(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator-visible win — clicking Download on a not-yet-
    rendered clip flips state to 'rendering' + returns 202 with the
    status_url IMMEDIATELY instead of blocking on Playwright."""
    monkeypatch.setattr(
        "chalybclip.settings.get_settings",
        lambda: type("S", (), {
            "default_output_dir": str(tmp_path),
            "db_path": str(tmp_path / "x.db"),
            "public_url": "",
        })(),
    )
    tenant_id = tenants["alice"]["id"]
    await _seed_clip(db, tenant_id, output_dir=tmp_path)

    # Block the background runner so the response shape gets pinned
    # without the runner actually firing Playwright.
    scheduled: list[dict] = []

    async def fake_runner(**kwargs: object) -> None:
        scheduled.append(kwargs)

    monkeypatch.setattr(
        "chalybclip.api._clip_render.render_clip_in_background", fake_runner,
    )

    r = await client.get(
        "/dashboard/clips/clp_test/download",
        headers=auth(tenants["alice"]["token"]),
        follow_redirects=False,
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["state"] == "rendering"
    assert body["progress_pct"] == 0
    assert body["status_url"].endswith(
        "/dashboard/clips/clp_test/render-status?quality=1080"
    )

    # DB state column flipped atomically.
    with bound_tenant(tenant_id):
        row = await ClipsRepo(db).get("clp_test")
    assert row is not None
    assert row.render_state == "rendering"
    assert row.render_started_at is not None


async def test_download_failed_state_returns_409_with_retry_url(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed render → 409 + error message + retry endpoint. UI shows
    the error inline + a "Try again" button that POSTs to the retry
    endpoint."""
    monkeypatch.setattr(
        "chalybclip.settings.get_settings",
        lambda: type("S", (), {
            "default_output_dir": str(tmp_path),
            "db_path": str(tmp_path / "x.db"),
            "public_url": "",
        })(),
    )
    tenant_id = tenants["alice"]["id"]
    await _seed_clip(
        db, tenant_id, output_dir=tmp_path,
        render_state="failed",
        render_progress_pct=42,
        render_error="recorder: Chromium OOM",
    )

    r = await client.get(
        "/dashboard/clips/clp_test/download",
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 409
    body = r.json()
    assert body["state"] == "failed"
    assert body["error"] == "recorder: Chromium OOM"
    assert "retry_url" in body
    assert body["retry_url"].endswith(
        "/dashboard/clips/clp_test/render-retry?quality=1080"
    )


async def test_render_retry_resets_state(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /render-retry clears the error + flips state to idle. The
    operator's next Download click kicks off a fresh background task."""
    monkeypatch.setattr(
        "chalybclip.settings.get_settings",
        lambda: type("S", (), {
            "default_output_dir": str(tmp_path),
            "db_path": str(tmp_path / "x.db"),
            "public_url": "",
        })(),
    )
    tenant_id = tenants["alice"]["id"]
    await _seed_clip(
        db, tenant_id, output_dir=tmp_path,
        render_state="failed", render_error="boom",
    )

    r = await client.post(
        "/dashboard/clips/clp_test/render-retry",
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True, "state": "idle"}

    with bound_tenant(tenant_id):
        row = await ClipsRepo(db).get("clp_test")
    assert row is not None
    assert row.render_state == "idle"
    assert row.render_error is None


async def test_download_in_flight_returns_202_without_double_dispatch(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second click on Download while the first render is still
    going does NOT spawn a second background task — it just returns
    the current status. Prevents the thundering-herd render storm."""
    monkeypatch.setattr(
        "chalybclip.settings.get_settings",
        lambda: type("S", (), {
            "default_output_dir": str(tmp_path),
            "db_path": str(tmp_path / "x.db"),
            "public_url": "",
        })(),
    )
    tenant_id = tenants["alice"]["id"]
    await _seed_clip(
        db, tenant_id, output_dir=tmp_path,
        render_state="rendering", render_progress_pct=37,
    )

    scheduled: list[dict] = []

    async def fake_runner(**kwargs: object) -> None:
        scheduled.append(kwargs)

    monkeypatch.setattr(
        "chalybclip.api._clip_render.render_clip_in_background", fake_runner,
    )

    r = await client.get(
        "/dashboard/clips/clp_test/download",
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 202
    body = r.json()
    assert body["state"] == "rendering"
    assert body["progress_pct"] == 37
    # No new background task fired — the existing one keeps going.
    assert scheduled == []
