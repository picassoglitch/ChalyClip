"""Render Migration R1-R3 — pin the new validation gates.

R1 — `validate_rendered_mp4` rejects malformed outputs (no video, no
     audio, truncated, ffprobe error). The background runner uses
     this to gate `mark_render_ready` so a returncode-0-but-corrupt
     ffmpeg output never lands as a "ready" download.

R2 — every failure path in `render_clip_in_background` unlinks the
     output file. Partial bytes never survive across a failed render
     into the next download click.

R3 — the download / status endpoints use `is_servable_cached_mp4`
     before treating a file on disk as a cache hit. A debris file
     (< 1 MB, missing ftyp box, 0 bytes) is unlinked + the
     render-state reset; the request falls through to the
     schedule-new-render path.

Tests deliberately operate on real bytes (smallest possible MP4 made
with ffmpeg testsrc / lavfi) so a future regression that breaks the
magic-byte check on a legitimate ffmpeg output gets caught.
"""

from __future__ import annotations

import datetime as _dt
import json as _json
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest

from chalybclip.api._render_validation import (
    is_servable_cached_mp4,
    validate_rendered_mp4,
)
from chalybclip.db import Database
from chalybclip.tenancy import bound_tenant

from .conftest import auth


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _have_ffprobe() -> bool:
    return shutil.which("ffprobe") is not None


def _make_real_mp4(out: Path, *, duration_s: float = 2.0) -> None:
    """Synthesize a tiny but valid MP4 with both video + audio
    streams. Used by the cache-hit tests so we exercise the real
    success path (not just synthetic bytes)."""
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=duration={duration_s}:size=320x240:rate=15",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_s}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "64k",
            "-t", str(duration_s),
            "-movflags", "+faststart",
            str(out),
        ],
        check=True, capture_output=True, timeout=30,
    )


# =====================================================================
# R3 — `is_servable_cached_mp4` magic-byte + size gate
# =====================================================================


def test_is_servable_rejects_missing_file(tmp_path: Path) -> None:
    """Path that doesn't exist → not servable. Mirrors the "file was
    deleted between cache check and serve" race."""
    assert is_servable_cached_mp4(tmp_path / "nope.mp4") is False


def test_is_servable_rejects_zero_byte_file(tmp_path: Path) -> None:
    """0-byte sentinel left by a crashed ffmpeg → not servable.
    Previously the endpoint's rendered.exists() served this as a
    real download → Windows codec error 0xC00D36C4."""
    p = tmp_path / "empty.mp4"
    p.write_bytes(b"")
    assert is_servable_cached_mp4(p) is False


def test_is_servable_rejects_truncated_file(tmp_path: Path) -> None:
    """A few hundred bytes of a real MP4 → not servable. Below the
    1 MB floor a real 1080p clip would clear."""
    p = tmp_path / "truncated.mp4"
    # Real ftyp header but tiny payload — passes magic check, fails
    # the size floor.
    p.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 500)
    assert is_servable_cached_mp4(p) is False


def test_is_servable_rejects_wrong_magic_bytes(tmp_path: Path) -> None:
    """A large blob without the ISO BMFF ftyp box → not servable.
    Catches the case where the file is large enough to clear the
    size floor but isn't actually an MP4 (e.g. an error-page HTML
    response that got saved to the cache path by mistake)."""
    p = tmp_path / "imposter.mp4"
    p.write_bytes(b"<html>not an mp4</html>" + b"\x00" * 2_000_000)
    assert is_servable_cached_mp4(p) is False


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not on PATH")
def test_is_servable_accepts_real_ffmpeg_mp4(tmp_path: Path) -> None:
    """End-to-end smoke: a real ffmpeg-encoded MP4 passes both the
    size floor and the ftyp magic check. Guards against a future
    regression where we tighten the gate too aggressively."""
    p = tmp_path / "real.mp4"
    # 6 seconds at 15fps + audio clears the 1 MB floor on most
    # libx264 builds. If it doesn't, the encoder we shipped with
    # produces sub-MB outputs which would be unusual and worth
    # surfacing as a test failure rather than masking.
    _make_real_mp4(p, duration_s=6.0)
    assert is_servable_cached_mp4(p) is True


# =====================================================================
# R1 — `validate_rendered_mp4` strict gate (ffprobe-backed)
# =====================================================================


def test_validate_rejects_missing(tmp_path: Path) -> None:
    ok, reason = validate_rendered_mp4(
        tmp_path / "no.mp4", expected_duration_s=10.0,
    )
    assert ok is False
    assert "missing" in (reason or "").lower()


def test_validate_rejects_undersized(tmp_path: Path) -> None:
    p = tmp_path / "tiny.mp4"
    p.write_bytes(b"\x00" * 100)
    ok, reason = validate_rendered_mp4(p, expected_duration_s=10.0)
    assert ok is False
    assert "bytes" in (reason or "").lower()


@pytest.mark.skipif(
    not (_have_ffmpeg() and _have_ffprobe()),
    reason="ffmpeg/ffprobe not on PATH",
)
def test_validate_accepts_real_mp4_with_matching_duration(
    tmp_path: Path,
) -> None:
    """A real ffmpeg-encoded MP4 whose duration matches the request
    → pass. Round-trip with the toolchain so a future regression on
    the ffprobe parse breaks loudly."""
    p = tmp_path / "good.mp4"
    _make_real_mp4(p, duration_s=6.0)
    ok, reason = validate_rendered_mp4(p, expected_duration_s=6.0)
    assert ok is True, f"expected pass, got {reason}"
    assert reason is None


@pytest.mark.skipif(
    not (_have_ffmpeg() and _have_ffprobe()),
    reason="ffmpeg/ffprobe not on PATH",
)
def test_validate_rejects_when_file_much_shorter_than_expected(
    tmp_path: Path,
) -> None:
    """Render claimed 30s but the file is 2s → reject. Catches
    early-EOF / truncated containers that ffmpeg accepted but
    operators would notice as a clipped export."""
    p = tmp_path / "short.mp4"
    _make_real_mp4(p, duration_s=6.0)  # > 1 MB
    ok, reason = validate_rendered_mp4(p, expected_duration_s=30.0)
    # 6s / 30s = 20% — far below the 80% floor.
    assert ok is False
    assert "duration" in (reason or "").lower()


@pytest.mark.skipif(
    not (_have_ffmpeg() and _have_ffprobe()),
    reason="ffmpeg/ffprobe not on PATH",
)
def test_validate_rejects_video_only_no_audio(tmp_path: Path) -> None:
    """A render that lost its audio stream → reject. The recorder
    always muxes source audio; missing audio means the mux failed."""
    p = tmp_path / "no_audio.mp4"
    ffmpeg = shutil.which("ffmpeg")
    subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=duration=6:size=320x240:rate=15",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-an",  # no audio
            "-t", "6",
            "-movflags", "+faststart",
            str(p),
        ],
        check=True, capture_output=True, timeout=30,
    )
    # Synthetic video-only encodes can come in under 1MB; pad so the
    # size floor doesn't reject before ffprobe even runs.
    if p.stat().st_size < 1_000_000:
        with p.open("ab") as f:
            f.write(b"\x00" * (1_000_000 - p.stat().st_size + 1))
    ok, reason = validate_rendered_mp4(p, expected_duration_s=6.0)
    assert ok is False
    assert "audio" in (reason or "").lower()


# =====================================================================
# R3 — download endpoint evicts debris instead of serving it
# =====================================================================


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


async def _seed_clip_with_debris_cache(
    db: Database,
    tenant_id: str,
    *,
    clip_id: str = "clp_debris",
    cache_bytes: bytes = b"",
    render_state: str = "ready",
    tmp_path: Path,
) -> Path:
    """Insert a clip whose on-disk cache file is debris (default
    0-byte). Returns the cache path so the test can poke at it."""
    stream_id = "str_debris"
    clip_dir = tmp_path / "clips" / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    cache_path = clip_dir / "clip_render_1080.mp4"
    cache_path.write_bytes(cache_bytes)
    source_path = clip_dir / "clip.mp4"
    source_path.write_bytes(b"fake-source-bytes" * 1000)

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
            "render_state, render_progress_pct) "
            "VALUES (?, ?, ?, 0, 30, 30, 1080, 1920, ?, 'cut', ?, ?, ?)",
            (
                clip_id, stream_id, tenant_id, str(source_path), _now(),
                render_state, 100 if render_state == "ready" else 0,
            ),
        )
        await conn.commit()

    return cache_path


async def test_download_evicts_zero_byte_cache_and_schedules_fresh(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operator's bug: clicking Download with a 0-byte file at
    clip_render_1080.mp4 used to return a 200 + FileResponse → Windows
    refused to open. Now: endpoint detects debris, unlinks it,
    resets render_state, returns 202 with status_url so the UI
    schedules a fresh render instead."""
    tenant_id = tenants["alice"]["id"]
    cache_path = await _seed_clip_with_debris_cache(
        db, tenant_id, tmp_path=tmp_path,
        cache_bytes=b"",
        render_state="ready",
    )
    assert cache_path.exists() and cache_path.stat().st_size == 0

    # Block the real background renderer — we only want to pin the
    # endpoint response shape, not actually drive Playwright.
    async def fake_runner(**kwargs: object) -> None:
        return None
    monkeypatch.setattr(
        "chalybclip.api._clip_render.render_clip_in_background", fake_runner,
    )

    r = await client.get(
        "/dashboard/clips/clp_debris/download",
        headers=auth(tenants["alice"]["token"]),
    )

    # Cache evicted.
    assert not cache_path.exists()
    # 202 means a fresh render was scheduled (or will be).
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["state"] == "rendering"
    assert body["status_url"].endswith("render-status?quality=1080")


async def test_download_evicts_truncated_cache_and_schedules_fresh(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real-looking but tiny (<1 MB) MP4 → debris. Operator saw
    these as Windows-can't-open downloads."""
    tenant_id = tenants["alice"]["id"]
    cache_path = await _seed_clip_with_debris_cache(
        db, tenant_id, tmp_path=tmp_path,
        # Real ftyp header but 500 bytes — passes magic, fails size.
        cache_bytes=b"\x00\x00\x00\x18ftypisom" + b"\x00" * 500,
        render_state="ready",
        clip_id="clp_truncated",
    )
    assert cache_path.exists() and cache_path.stat().st_size < 1000

    async def fake_runner(**kwargs: object) -> None:
        return None
    monkeypatch.setattr(
        "chalybclip.api._clip_render.render_clip_in_background", fake_runner,
    )

    r = await client.get(
        "/dashboard/clips/clp_truncated/download",
        headers=auth(tenants["alice"]["token"]),
    )
    assert not cache_path.exists()
    assert r.status_code == 202


async def test_download_does_not_evict_while_rendering(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Race guard: if render_state='rendering' and the file currently
    on disk is below the size floor (ffmpeg is mid-write), DO NOT
    unlink — on Linux that yanks the directory entry from under the
    running ffmpeg; on Windows it raises a sharing violation. Just
    return 202 and let the UI keep polling."""
    tenant_id = tenants["alice"]["id"]
    cache_path = await _seed_clip_with_debris_cache(
        db, tenant_id, tmp_path=tmp_path,
        cache_bytes=b"\x00\x00\x00\x18ftypisom" + b"\x00" * 1000,
        render_state="rendering",
        clip_id="clp_mid_write",
    )
    assert cache_path.exists()

    # Belt-and-braces: this branch returns early (state=rendering)
    # without scheduling, so the fake runner is just insurance against
    # an accidental fall-through.
    async def fake_runner(**kwargs: object) -> None:
        return None
    monkeypatch.setattr(
        "chalybclip.api._clip_render.render_clip_in_background", fake_runner,
    )

    r = await client.get(
        "/dashboard/clips/clp_mid_write/download",
        headers=auth(tenants["alice"]["token"]),
    )

    # File must still be on disk — ffmpeg is supposedly writing it.
    assert cache_path.exists()
    assert r.status_code == 202
    assert r.json()["state"] == "rendering"


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not on PATH")
async def test_download_serves_valid_cache_unchanged(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    tmp_path: Path,
) -> None:
    """Sanity: a real, valid MP4 in the cache still gets served
    directly as 200 + video/mp4. The validation gate must not
    accidentally reject legitimate outputs."""
    tenant_id = tenants["alice"]["id"]
    # Seed without writing the cache file ourselves — we want a real
    # ffmpeg-produced MP4.
    cache_path = await _seed_clip_with_debris_cache(
        db, tenant_id, tmp_path=tmp_path,
        cache_bytes=b"",  # placeholder
        render_state="ready",
        clip_id="clp_real",
    )
    cache_path.unlink()  # remove the placeholder
    _make_real_mp4(cache_path, duration_s=6.0)
    assert cache_path.stat().st_size > 1_000_000

    r = await client.get(
        "/dashboard/clips/clp_real/download",
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 200
    assert (r.headers.get("content-type") or "").startswith("video/")
    # Cache file untouched.
    assert cache_path.exists()
