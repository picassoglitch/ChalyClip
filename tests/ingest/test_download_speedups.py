"""Task 2b — yt-dlp HLS speedups: concurrent_fragments + aria2c + format cap.

Tests assert on the `ydl_opts` dict yt-dlp is initialized with. We monkeypatch
`yt_dlp.YoutubeDL` to capture the opts without doing a real network fetch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chalybclip.ingest import service as ingest_service
from chalybclip.settings import Settings, get_settings


class _FakeYDL:
    """Drop-in for yt_dlp.YoutubeDL — records the opts + returns a stub info dict."""

    captured_opts: dict[str, Any] = {}

    def __init__(self, opts: dict[str, Any]):
        _FakeYDL.captured_opts = dict(opts)

    def __enter__(self) -> "_FakeYDL":
        return self

    def __exit__(self, *_a: object) -> None:
        return None

    def extract_info(self, url: str, download: bool = True) -> dict[str, Any]:
        # Create the output file so the surrounding rename + ffprobe paths
        # have something to chew on.
        outtmpl = _FakeYDL.captured_opts.get("outtmpl", "")
        if outtmpl:
            actual = Path(outtmpl.replace("%(ext)s", "mp4"))
            actual.parent.mkdir(parents=True, exist_ok=True)
            actual.write_bytes(b"\x00\x00fakevideo")
        return {
            "duration": 600.0,
            "title": "fake",
            "uploader": "fake",
            "format_id": "1080p",
            "protocol": "m3u8_native",
            "_filename": str(Path(outtmpl.replace("%(ext)s", "mp4")) if outtmpl else ""),
        }


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _patch_settings(
    monkeypatch: pytest.MonkeyPatch,
    **overrides: Any,
) -> None:
    """Force the singleton Settings() to materialize with these overrides."""
    get_settings.cache_clear()
    monkeypatch.setattr(
        ingest_service,
        # Import inside _download_vod is `from chalybclip.settings import get_settings`,
        # so we override the global accessor.
        "_download_vod",  # noqa: B017 — placeholder; real patch below
        ingest_service._download_vod,
    )
    # Patch get_settings directly to return a Settings with our overrides.

    def _get() -> Settings:
        return Settings(**overrides)

    import chalybclip.settings as _settings_mod
    monkeypatch.setattr(_settings_mod, "get_settings", _get)


def test_concurrent_fragments_default_16(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest_service.yt_dlp, "YoutubeDL", _FakeYDL)
    monkeypatch.setattr(ingest_service.yt_dlp.utils, "DownloadError", RuntimeError)

    ingest_service._download_vod(
        vod_url="https://kick.com/x/videos/1",
        target_path=tmp_path / "video.mp4",
    )
    assert _FakeYDL.captured_opts["concurrent_fragment_downloads"] == 16


def test_concurrent_fragments_can_be_overridden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_settings(monkeypatch, ytdlp_concurrent_fragments=4)
    monkeypatch.setattr(ingest_service.yt_dlp, "YoutubeDL", _FakeYDL)
    monkeypatch.setattr(ingest_service.yt_dlp.utils, "DownloadError", RuntimeError)

    ingest_service._download_vod(
        vod_url="https://kick.com/x/videos/1",
        target_path=tmp_path / "video.mp4",
    )
    assert _FakeYDL.captured_opts["concurrent_fragment_downloads"] == 4


def test_format_caps_at_max_source_height(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operator default: never pull >1080 source — clips top out at 1080×1920."""
    monkeypatch.setattr(ingest_service.yt_dlp, "YoutubeDL", _FakeYDL)
    monkeypatch.setattr(ingest_service.yt_dlp.utils, "DownloadError", RuntimeError)

    ingest_service._download_vod(
        vod_url="https://kick.com/x/videos/1",
        target_path=tmp_path / "video.mp4",
    )
    fmt = _FakeYDL.captured_opts["format"]
    # The format string includes "[height<=1080]" in both branches.
    assert "[height<=1080]" in fmt
    assert "bestvideo" in fmt and "bestaudio" in fmt


def test_format_uncapped_when_max_height_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_settings(monkeypatch, max_source_height=0)
    monkeypatch.setattr(ingest_service.yt_dlp, "YoutubeDL", _FakeYDL)
    monkeypatch.setattr(ingest_service.yt_dlp.utils, "DownloadError", RuntimeError)

    ingest_service._download_vod(
        vod_url="https://kick.com/x/videos/1",
        target_path=tmp_path / "video.mp4",
    )
    fmt = _FakeYDL.captured_opts["format"]
    assert "height<=" not in fmt
    assert fmt == "bestvideo*+bestaudio/best"


def test_aria2c_used_when_enabled_and_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_settings(monkeypatch, ytdlp_use_aria2c=True)
    monkeypatch.setattr(ingest_service.yt_dlp, "YoutubeDL", _FakeYDL)
    monkeypatch.setattr(ingest_service.yt_dlp.utils, "DownloadError", RuntimeError)
    monkeypatch.setattr(ingest_service.shutil, "which", lambda b: "/usr/bin/aria2c")

    ingest_service._download_vod(
        vod_url="https://kick.com/x/videos/1",
        target_path=tmp_path / "video.mp4",
    )
    assert _FakeYDL.captured_opts["external_downloader"] == {"default": "aria2c"}
    args = _FakeYDL.captured_opts["external_downloader_args"]["aria2c"]
    assert "-x16" in args
    assert "-s16" in args


def test_aria2c_silently_skipped_when_not_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ytdlp_use_aria2c=True` but binary not installed → fall back to
    native downloader silently (log warning, don't crash the ingest)."""
    _patch_settings(monkeypatch, ytdlp_use_aria2c=True)
    monkeypatch.setattr(ingest_service.yt_dlp, "YoutubeDL", _FakeYDL)
    monkeypatch.setattr(ingest_service.yt_dlp.utils, "DownloadError", RuntimeError)
    monkeypatch.setattr(ingest_service.shutil, "which", lambda b: None)

    ingest_service._download_vod(
        vod_url="https://kick.com/x/videos/1",
        target_path=tmp_path / "video.mp4",
    )
    assert "external_downloader" not in _FakeYDL.captured_opts


def test_aria2c_off_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest_service.yt_dlp, "YoutubeDL", _FakeYDL)
    monkeypatch.setattr(ingest_service.yt_dlp.utils, "DownloadError", RuntimeError)
    monkeypatch.setattr(ingest_service.shutil, "which", lambda b: "/usr/bin/aria2c")

    ingest_service._download_vod(
        vod_url="https://kick.com/x/videos/1",
        target_path=tmp_path / "video.mp4",
    )
    # Default Settings has ytdlp_use_aria2c=False → no external_downloader
    # even when aria2c is present on PATH.
    assert "external_downloader" not in _FakeYDL.captured_opts
