"""Tests for `ingest_vod` with the network/ffmpeg layers stubbed."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from chalybclip.ingest import Stream, ingest_vod
from chalybclip.ingest import service as ingest_service


def _stub_download_writes_file(monkeypatch: pytest.MonkeyPatch, *, info: dict[str, Any]) -> list[Path]:
    """Replace `_download_vod` with a no-network stub that creates the target file."""
    calls: list[Path] = []

    def fake_download(
        *,
        vod_url: str,
        target_path: Path,
        cookies_from_browser: str | None = None,
        cookies_file: str | None = None,
        platform: str = "unknown",
    ) -> dict[str, Any]:
        calls.append(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"\x00\x00fakevideo")
        return info

    monkeypatch.setattr(ingest_service, "_download_vod", fake_download)
    return calls


def _stub_audio_extract(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, Path]]:
    calls: list[tuple[Path, Path]] = []

    def fake_extract(video_path: Path, audio_path: Path) -> None:
        calls.append((video_path, audio_path))
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"RIFFfakewav")

    monkeypatch.setattr(ingest_service, "_extract_audio", fake_extract)
    return calls


def _stub_ffprobe(monkeypatch: pytest.MonkeyPatch, *, duration_s: float) -> None:
    monkeypatch.setattr(ingest_service, "_ffprobe_duration", lambda _p: duration_s)


def test_ingest_creates_layout_and_returns_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = {"duration": 612.5, "title": "test stream", "uploader": "aldovillanueva"}
    download_calls = _stub_download_writes_file(monkeypatch, info=info)
    extract_calls = _stub_audio_extract(monkeypatch)

    stream = asyncio.run(
        ingest_vod(
            tenant_id="default",
            vod_url="https://kick.com/aldovillanueva/videos/abc",
            output_dir=tmp_path,
        )
    )

    assert isinstance(stream, Stream)
    assert stream.id.startswith("str_")
    assert stream.tenant_id == "default"
    assert stream.platform == "kick"
    assert stream.title == "test stream"
    assert stream.channel == "aldovillanueva"
    assert stream.duration_s == pytest.approx(612.5)

    stream_dir = tmp_path / stream.id
    assert (stream_dir / "stream.json").exists()
    assert (stream_dir / "source" / "video.mp4").exists()
    assert (stream_dir / "source" / "audio.wav").exists()

    assert len(download_calls) == 1
    assert download_calls[0].name == "video.mp4"
    assert len(extract_calls) == 1


def test_ingest_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = {"duration": 30.0, "title": "t", "uploader": "u"}
    download_calls = _stub_download_writes_file(monkeypatch, info=info)
    extract_calls = _stub_audio_extract(monkeypatch)

    first = asyncio.run(
        ingest_vod(
            tenant_id="default",
            vod_url="https://kick.com/c/videos/1",
            output_dir=tmp_path,
        )
    )
    second = asyncio.run(
        ingest_vod(
            tenant_id="default",
            vod_url="https://kick.com/c/videos/1",
            output_dir=tmp_path,
            stream_id=first.id,
        )
    )

    assert second == first
    assert len(download_calls) == 1
    assert len(extract_calls) == 1


def test_ingest_resume_redownloads_when_source_reclaimed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every source-reclaim path (delete-on-completion, the pipeline
    runner's failure cleanup, the disk watchdog) leaves stream.json in
    place and relies on 'a re-run re-downloads automatically'. Resuming
    on stream.json alone broke that: the re-run skipped the download and
    died at transcribe on the missing audio."""
    info = {"duration": 30.0, "title": "t", "uploader": "u"}
    download_calls = _stub_download_writes_file(monkeypatch, info=info)
    extract_calls = _stub_audio_extract(monkeypatch)

    first = asyncio.run(
        ingest_vod(
            tenant_id="default",
            vod_url="https://kick.com/c/videos/1",
            output_dir=tmp_path,
        )
    )
    # Reclaim the source the way retention does: files gone, manifest kept.
    source_dir = tmp_path / first.id / "source"
    (source_dir / "video.mp4").unlink()
    (source_dir / "audio.wav").unlink()

    second = asyncio.run(
        ingest_vod(
            tenant_id="default",
            vod_url="https://kick.com/c/videos/1",
            output_dir=tmp_path,
            stream_id=first.id,
        )
    )

    assert second.id == first.id
    assert len(download_calls) == 2  # re-downloaded, not cache-resumed
    assert len(extract_calls) == 2
    assert (source_dir / "video.mp4").exists()
    assert (source_dir / "audio.wav").exists()


def test_ingest_resume_upload_source_reclaimed_raises_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Uploads have no origin to re-fetch — a reclaimed upload source must
    fail with a message that says to upload again, not fall through to
    yt-dlp on the upload:// pseudo-URL."""
    from chalybclip.errors import IngestError

    _stub_download_writes_file(monkeypatch, info={})
    _stub_audio_extract(monkeypatch)

    sid = "str_upload1"
    stream_dir = tmp_path / sid
    (stream_dir / "source").mkdir(parents=True)
    stream = Stream(
        id=sid, tenant_id="default", vod_url="upload://clip.mp4",
        platform="upload", title="clip", channel=None, duration_s=5.0,
        source_video_path=stream_dir / "source" / "video.mp4",
        source_audio_path=stream_dir / "source" / "audio.wav",
    )
    (stream_dir / "stream.json").write_text(
        stream.model_dump_json(indent=2), encoding="utf-8"
    )

    with pytest.raises(IngestError, match="upload the file again"):
        asyncio.run(
            ingest_vod(
                tenant_id="default",
                vod_url="upload://clip.mp4",
                output_dir=tmp_path,
                stream_id=sid,
            )
        )


def test_ingest_force_redownloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = {"duration": 30.0, "title": "t", "uploader": "u"}
    download_calls = _stub_download_writes_file(monkeypatch, info=info)
    extract_calls = _stub_audio_extract(monkeypatch)

    first = asyncio.run(
        ingest_vod(
            tenant_id="default",
            vod_url="https://kick.com/c/videos/1",
            output_dir=tmp_path,
        )
    )
    asyncio.run(
        ingest_vod(
            tenant_id="default",
            vod_url="https://kick.com/c/videos/1",
            output_dir=tmp_path,
            stream_id=first.id,
            force=True,
        )
    )

    assert len(download_calls) == 2
    assert len(extract_calls) == 2


def test_ingest_falls_back_to_ffprobe_for_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = {"duration": 0, "title": "no duration in info", "uploader": "u"}
    _stub_download_writes_file(monkeypatch, info=info)
    _stub_audio_extract(monkeypatch)
    _stub_ffprobe(monkeypatch, duration_s=42.0)

    stream = asyncio.run(
        ingest_vod(
            tenant_id="default",
            vod_url="https://twitch.tv/videos/1",
            output_dir=tmp_path,
        )
    )

    assert stream.duration_s == pytest.approx(42.0)
    assert stream.platform == "twitch"


# ---- Download-completeness guard (truncated recent-upload fetch) ----


def test_assert_download_complete_raises_on_truncation() -> None:
    from chalybclip.errors import IngestError

    # 32-min video, ~8s actually downloaded — the still-processing-upload case.
    with pytest.raises(IngestError, match="truncated"):
        ingest_service._assert_download_complete(
            vod_url="https://youtu.be/x", claimed_s=1924.0, actual_s=8.0
        )


def test_assert_download_complete_allows_near_full_download() -> None:
    # Within 90% — minor container rounding, not a truncation.
    ingest_service._assert_download_complete(
        vod_url="https://youtu.be/x", claimed_s=1924.0, actual_s=1920.0
    )


def test_assert_download_complete_tolerates_small_absolute_shortfall() -> None:
    # Short clip off by a couple seconds: ratio is low but the absolute
    # gap is under tolerance, so it must NOT trip.
    ingest_service._assert_download_complete(
        vod_url="https://youtu.be/x", claimed_s=20.0, actual_s=15.0
    )


def test_assert_download_complete_noop_without_baseline() -> None:
    # No claimed duration, or ffprobe unavailable -> nothing to compare.
    ingest_service._assert_download_complete(
        vod_url="https://youtu.be/x", claimed_s=0.0, actual_s=0.0
    )
    ingest_service._assert_download_complete(
        vod_url="https://youtu.be/x", claimed_s=600.0, actual_s=0.0
    )


def test_ingest_fails_on_truncated_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chalybclip.errors import IngestError

    # Metadata claims 32 min; the file on disk ffprobes to 8s.
    info = {"duration": 1924.0, "title": "fresh upload", "uploader": "u"}
    _stub_download_writes_file(monkeypatch, info=info)
    _stub_audio_extract(monkeypatch)
    _stub_ffprobe(monkeypatch, duration_s=8.0)

    with pytest.raises(IngestError, match="truncated"):
        asyncio.run(
            ingest_vod(
                tenant_id="default",
                vod_url="https://www.youtube.com/watch?v=abc",
                output_dir=tmp_path,
            )
        )
    # No stream.json written -> a re-run re-downloads instead of caching
    # the partial fetch.
    assert not list(tmp_path.glob("*/stream.json"))


def test_ingest_uses_ffprobe_duration_over_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Metadata and the real file are both ~full but differ slightly; the
    # on-disk ffprobe value wins now (authoritative), not the info dict.
    info = {"duration": 600.0, "title": "t", "uploader": "u"}
    _stub_download_writes_file(monkeypatch, info=info)
    _stub_audio_extract(monkeypatch)
    _stub_ffprobe(monkeypatch, duration_s=590.0)

    stream = asyncio.run(
        ingest_vod(
            tenant_id="default",
            vod_url="https://twitch.tv/videos/9",
            output_dir=tmp_path,
        )
    )
    assert stream.duration_s == pytest.approx(590.0)


# ---- Free-disk preflight (disk-exhaustion guard) ----


def test_disk_headroom_raises_when_below_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil as _shutil

    from chalybclip.errors import IngestError
    from chalybclip.settings import get_settings

    monkeypatch.setattr(get_settings(), "min_free_disk_bytes", 3 * 1024**3, raising=False)
    monkeypatch.setattr(
        _shutil, "disk_usage",
        lambda _p: _shutil._ntuple_diskusage(100, 99, 1 * 1024**3),  # 1 GB free
    )
    with pytest.raises(IngestError, match="insufficient disk"):
        ingest_service._assert_disk_headroom(Path("."))


def test_disk_headroom_passes_with_room(monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil as _shutil

    from chalybclip.settings import get_settings

    monkeypatch.setattr(get_settings(), "min_free_disk_bytes", 3 * 1024**3, raising=False)
    monkeypatch.setattr(
        _shutil, "disk_usage",
        lambda _p: _shutil._ntuple_diskusage(100, 50, 50 * 1024**3),  # 50 GB free
    )
    ingest_service._assert_disk_headroom(Path("."))  # no raise


def test_disk_headroom_disabled_when_floor_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil as _shutil

    from chalybclip.settings import get_settings

    monkeypatch.setattr(get_settings(), "min_free_disk_bytes", 0, raising=False)
    # disk_usage must not even be consulted when disabled.
    def _boom(_p: object) -> object:
        raise AssertionError("disk_usage should not be called when floor=0")

    monkeypatch.setattr(_shutil, "disk_usage", _boom)
    ingest_service._assert_disk_headroom(Path("."))  # no raise


# ---- Partial-download cleanup on failure (disk-exhaustion guard) ----


def test_download_failure_sweeps_partial_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed yt-dlp download must not leave partial files behind."""
    import yt_dlp

    from chalybclip.errors import IngestError

    target = tmp_path / "source" / "video.mp4"
    target.parent.mkdir(parents=True)

    class _FakeYDL:
        def __init__(self, *_a: object, **_k: object) -> None:
            pass

        def __enter__(self) -> _FakeYDL:
            return self

        def __exit__(self, *_a: object) -> bool:
            return False

        def extract_info(self, *_a: object, **_k: object) -> dict[str, Any]:
            # Simulate yt-dlp writing partials, then failing mid-fetch.
            (target.parent / "video.mp4.part").write_bytes(b"x" * 1000)
            (target.parent / "video.f137.mp4").write_bytes(b"x" * 1000)
            (target.parent / "video.ytdl").write_bytes(b"x")
            raise yt_dlp.utils.DownloadError("boom: connection reset")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _FakeYDL)

    with pytest.raises(IngestError, match="yt-dlp failed"):
        ingest_service._download_vod(
            vod_url="https://twitch.tv/videos/1",
            target_path=target,
            cookies_from_browser=None,
            cookies_file=None,
            platform="twitch",
        )
    # Every partial under source/ is gone — nothing left to eat the disk.
    leftovers = [p.name for p in target.parent.glob("video*")]
    assert leftovers == [], f"partial files not cleaned: {leftovers}"


def test_stale_partial_resume_sweeps_and_retries_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A killed run leaves `.part`/`.ytdl` resume state whose fragment files
    are gone; yt-dlp's resume then dies with UnavailableVideoError (a SIBLING
    of DownloadError) on a local FileNotFoundError. The download must sweep
    the stale state and retry once from a clean slate instead of failing the
    same way on every re-run.
    """
    import yt_dlp

    target = tmp_path / "source" / "video.mp4"
    target.parent.mkdir(parents=True)
    # Stale state from the killed run.
    (target.parent / "video.mp4.part").write_bytes(b"x" * 1000)
    (target.parent / "video.mp4.ytdl").write_bytes(b"x")

    calls: list[int] = []

    class _FakeYDL:
        def __init__(self, *_a: object, **_k: object) -> None:
            pass

        def __enter__(self) -> _FakeYDL:
            return self

        def __exit__(self, *_a: object) -> bool:
            return False

        def extract_info(self, *_a: object, **_k: object) -> dict[str, Any]:
            calls.append(1)
            if len(calls) == 1:
                raise yt_dlp.utils.UnavailableVideoError(
                    "Unable to download video: [Errno 2] No such file or "
                    "directory: 'video.mp4.part-Frag2086'"
                )
            target.write_bytes(b"video")
            return {}

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _FakeYDL)

    ingest_service._download_vod(
        vod_url="https://twitch.tv/videos/1",
        target_path=target,
        cookies_from_browser=None,
        cookies_file=None,
        platform="twitch",
    )
    assert len(calls) == 2, "expected one clean-slate retry"
    assert target.exists()
    stale = [p.name for p in target.parent.glob("video.mp4.*")]
    assert stale == [], f"stale resume state not swept: {stale}"


def test_unavailable_video_error_is_wrapped_not_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UnavailableVideoError that is NOT stale-resume-shaped must still be
    wrapped as IngestError (with the leftover sweep), not escape raw."""
    import yt_dlp

    from chalybclip.errors import IngestError

    target = tmp_path / "source" / "video.mp4"
    target.parent.mkdir(parents=True)

    class _FakeYDL:
        def __init__(self, *_a: object, **_k: object) -> None:
            pass

        def __enter__(self) -> _FakeYDL:
            return self

        def __exit__(self, *_a: object) -> bool:
            return False

        def extract_info(self, *_a: object, **_k: object) -> dict[str, Any]:
            (target.parent / "video.mp4.part").write_bytes(b"x" * 1000)
            raise yt_dlp.utils.UnavailableVideoError("format unavailable")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _FakeYDL)

    with pytest.raises(IngestError, match="yt-dlp failed"):
        ingest_service._download_vod(
            vod_url="https://twitch.tv/videos/1",
            target_path=target,
            cookies_from_browser=None,
            cookies_file=None,
            platform="twitch",
        )
    leftovers = [p.name for p in target.parent.glob("video*")]
    assert leftovers == [], f"partial files not cleaned: {leftovers}"


# ---- Residential proxy wiring (sustainable bot-gate fix) ----


def _capture_ydl_opts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Patch yt_dlp.YoutubeDL to capture the opts and return a valid download."""
    import yt_dlp

    captured: dict[str, Any] = {}
    target = tmp_path / "source" / "video.mp4"

    class _FakeYDL:
        def __init__(self, opts: dict[str, Any], *_a: object, **_k: object) -> None:
            captured.update(opts)

        def __enter__(self) -> _FakeYDL:
            return self

        def __exit__(self, *_a: object) -> bool:
            return False

        def extract_info(self, *_a: object, **_k: object) -> dict[str, Any]:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"\x00\x00data")
            return {"requested_downloads": [{"filepath": str(target)}]}

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _FakeYDL)
    return captured


def test_download_vod_sets_proxy_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chalybclip.settings import get_settings

    monkeypatch.setattr(
        get_settings(), "ytdlp_proxy", "http://u:p@proxy.host:1234", raising=False
    )
    captured = _capture_ydl_opts(monkeypatch, tmp_path)
    ingest_service._download_vod(
        vod_url="https://twitch.tv/videos/1",
        target_path=tmp_path / "source" / "video.mp4",
        cookies_from_browser=None, cookies_file=None, platform="twitch",
    )
    assert captured.get("proxy") == "http://u:p@proxy.host:1234"


def test_download_vod_no_proxy_when_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chalybclip.settings import get_settings

    monkeypatch.setattr(get_settings(), "ytdlp_proxy", None, raising=False)
    captured = _capture_ydl_opts(monkeypatch, tmp_path)
    ingest_service._download_vod(
        vod_url="https://twitch.tv/videos/1",
        target_path=tmp_path / "source" / "video.mp4",
        cookies_from_browser=None, cookies_file=None, platform="twitch",
    )
    assert "proxy" not in captured


# ---- Free YouTube bot-gate dodge (player_client / PO-token) ----


def test_youtube_extractor_args_player_client() -> None:
    class _S:
        ytdlp_player_client = "tv, web_safari ,mweb"
        ytdlp_po_provider_url = None

    out = ingest_service.youtube_extractor_args(_S())
    assert out == {"player_client": ["tv", "web_safari", "mweb"]}


def test_youtube_extractor_args_po_provider() -> None:
    class _S:
        ytdlp_player_client = None
        ytdlp_po_provider_url = "http://pot:4416"

    out = ingest_service.youtube_extractor_args(_S())
    assert out == {"getpot_bgutil_baseurl": ["http://pot:4416"]}


def test_youtube_extractor_args_none_when_unset() -> None:
    class _S:
        ytdlp_player_client = None
        ytdlp_po_provider_url = ""

    assert ingest_service.youtube_extractor_args(_S()) is None


def test_download_vod_sets_youtube_extractor_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chalybclip.settings import get_settings

    monkeypatch.setattr(
        get_settings(), "ytdlp_player_client", "tv,web_safari", raising=False
    )
    captured = _capture_ydl_opts(monkeypatch, tmp_path)
    ingest_service._download_vod(
        vod_url="https://www.youtube.com/watch?v=abc",
        target_path=tmp_path / "source" / "video.mp4",
        cookies_from_browser=None, cookies_file=None, platform="youtube",
    )
    assert captured.get("extractor_args") == {"youtube": {"player_client": ["tv", "web_safari"]}}
