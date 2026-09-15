"""ingest_uploaded — drop-a-file path that bypasses yt-dlp."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from chalybclip.errors import IngestError
from chalybclip.ingest import Stream, ingest_uploaded


def _stub_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the ffmpeg/ffprobe shell-outs so tests don't need the binaries."""

    def fake_extract_audio(_video: Path, audio: Path) -> None:
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"\x00fakeaudio")

    def fake_ffprobe(_video: Path) -> float:
        return 42.0

    monkeypatch.setattr("chalybclip.ingest.service._extract_audio", fake_extract_audio)
    monkeypatch.setattr("chalybclip.ingest.service._ffprobe_duration", fake_ffprobe)


async def test_ingest_uploaded_writes_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Writes stream.json + source/video.mp4 + source/audio.wav."""
    _stub_ffmpeg(monkeypatch)

    src = tmp_path / "incoming" / "my_recording.mp4"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"\x00fakevideo")

    out = tmp_path / "out"
    stream = await ingest_uploaded(
        tenant_id="default",
        source_path=src,
        output_dir=out,
        title="my_recording.mp4",
    )

    assert isinstance(stream, Stream)
    assert stream.platform == "upload"
    assert stream.duration_s == 42.0
    assert stream.title == "my_recording.mp4"
    assert stream.vod_url == "upload://my_recording.mp4"
    # Video moved into stream_dir/source/video.mp4.
    expected_video = out / stream.id / "source" / "video.mp4"
    assert expected_video.exists()
    assert expected_video.read_bytes() == b"\x00fakevideo"
    # Audio extracted next to it.
    expected_audio = out / stream.id / "source" / "audio.wav"
    assert expected_audio.exists()
    # Manifest persisted.
    assert (out / stream.id / "stream.json").exists()
    # Source tempfile consumed (move semantics).
    assert not src.exists()


async def test_ingest_uploaded_idempotent_on_existing_stream_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running with the same stream_id returns the cached row, no re-ingest."""
    _stub_ffmpeg(monkeypatch)

    src = tmp_path / "v1.mp4"
    src.write_bytes(b"\x00fakevideo")
    out = tmp_path / "out"

    first = await ingest_uploaded(
        tenant_id="default",
        source_path=src,
        output_dir=out,
        stream_id="str_fixed",
    )
    assert first.id == "str_fixed"

    # Second call: a different "upload" file but same stream_id; cache wins
    # and the second source file is left alone (not moved).
    src2 = tmp_path / "v2.mp4"
    src2.write_bytes(b"\x00different")
    second = await ingest_uploaded(
        tenant_id="default",
        source_path=src2,
        output_dir=out,
        stream_id="str_fixed",
    )
    assert second.id == first.id
    assert src2.exists()  # not consumed because cache hit
    # The original on-disk video is untouched.
    video = out / "str_fixed" / "source" / "video.mp4"
    assert video.read_bytes() == b"\x00fakevideo"


async def test_ingest_uploaded_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="upload source not found"):
        await ingest_uploaded(
            tenant_id="default",
            source_path=tmp_path / "does_not_exist.mp4",
            output_dir=tmp_path / "out",
        )


async def test_ingest_uploaded_falls_back_to_copy_on_cross_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When `Path.replace` raises OSError (cross-device on Windows), copy + unlink."""
    _stub_ffmpeg(monkeypatch)

    src = tmp_path / "v.mp4"
    src.write_bytes(b"\x00fakevideo")

    real_replace = Path.replace
    calls = {"replace": 0}

    def flaky_replace(self: Path, target: Path | str) -> Path:
        calls["replace"] += 1
        if calls["replace"] == 1 and Path(self) == src:
            raise OSError("cross-device move")
        return real_replace(self, target)

    with patch.object(Path, "replace", flaky_replace):
        stream = await ingest_uploaded(
            tenant_id="default",
            source_path=src,
            output_dir=tmp_path / "out",
        )

    assert (tmp_path / "out" / stream.id / "source" / "video.mp4").exists()
    # Source still got cleaned up via the copy+unlink fallback.
    assert not src.exists()
