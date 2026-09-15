"""Tests for `load_stream` (Stream rehydration from disk)."""

from __future__ import annotations

from pathlib import Path

import pytest

from chalybclip.errors import IngestError
from chalybclip.ingest import Stream, load_stream


def _seed_stream(tmp_path: Path) -> Stream:
    stream_id = "str_01ABC"
    stream_dir = tmp_path / stream_id
    source_dir = stream_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    video_path = source_dir / "video.mp4"
    audio_path = source_dir / "audio.wav"
    video_path.write_bytes(b"v")
    audio_path.write_bytes(b"a")
    stream = Stream(
        id=stream_id,
        tenant_id="default",
        vod_url="https://kick.com/c/videos/1",
        platform="kick",
        title="t",
        channel="c",
        duration_s=12.0,
        source_video_path=video_path,
        source_audio_path=audio_path,
    )
    (stream_dir / "stream.json").write_text(
        stream.model_dump_json(indent=2), encoding="utf-8"
    )
    return stream


def test_load_stream_round_trips(tmp_path: Path) -> None:
    seeded = _seed_stream(tmp_path)
    loaded = load_stream(tmp_path / seeded.id)
    assert loaded == seeded


def test_load_stream_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="missing stream.json"):
        load_stream(tmp_path / "no_such_stream")
