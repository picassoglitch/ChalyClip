"""`reclaim_stream_source` — delete-on-completion for the raw source VOD.

The pipeline calls this once a stream finishes processing to drop the
heaviest on-disk artifact (downloaded video + extracted audio). These
tests pin the two layouts (conventional `<stream_dir>/source/` and the
live `/data/live/<id>/` layout where the source lives outside the
per-stream dir) and the idempotency guarantee.
"""

from __future__ import annotations

from pathlib import Path

from chalybclip.retention import reclaim_stream_source


def _make_source(stream_dir: Path) -> tuple[Path, Path]:
    src = stream_dir / "source"
    src.mkdir(parents=True, exist_ok=True)
    video = src / "video.mp4"
    audio = src / "audio.wav"
    video.write_bytes(b"\x00" * 4096)
    audio.write_bytes(b"\x00" * 1024)
    return video, audio


def test_reclaims_conventional_source_dir(tmp_path: Path) -> None:
    stream_dir = tmp_path / "str_01"
    video, audio = _make_source(stream_dir)

    freed = reclaim_stream_source(
        stream_dir=stream_dir,
        source_video_path=video,
        source_audio_path=audio,
    )

    assert freed == 4096 + 1024
    assert not (stream_dir / "source").exists()
    assert not video.exists()
    assert not audio.exists()


def test_is_idempotent(tmp_path: Path) -> None:
    stream_dir = tmp_path / "str_02"
    video, audio = _make_source(stream_dir)

    first = reclaim_stream_source(
        stream_dir=stream_dir,
        source_video_path=video,
        source_audio_path=audio,
    )
    second = reclaim_stream_source(
        stream_dir=stream_dir,
        source_video_path=video,
        source_audio_path=audio,
    )

    assert first > 0
    assert second == 0  # nothing left to delete; no error


def test_reclaims_live_layout_outside_stream_dir(tmp_path: Path) -> None:
    """Live recordings live at /data/live/<id>/source.mp4 — OUTSIDE the
    per-stream out dir. The explicit paths must still be unlinked."""
    stream_dir = tmp_path / "out" / "str_03"
    stream_dir.mkdir(parents=True, exist_ok=True)
    live_dir = tmp_path / "live" / "str_03"
    live_dir.mkdir(parents=True, exist_ok=True)
    video = live_dir / "source.mp4"
    audio = live_dir / "source.audio.wav"
    video.write_bytes(b"\x00" * 2048)
    audio.write_bytes(b"\x00" * 256)

    freed = reclaim_stream_source(
        stream_dir=stream_dir,
        source_video_path=video,
        source_audio_path=audio,
    )

    assert freed == 2048 + 256
    assert not video.exists()
    assert not audio.exists()


def test_tolerates_missing_paths(tmp_path: Path) -> None:
    """No source dir, None paths — returns 0, never raises."""
    stream_dir = tmp_path / "str_04"
    stream_dir.mkdir(parents=True, exist_ok=True)

    freed = reclaim_stream_source(
        stream_dir=stream_dir,
        source_video_path=None,
        source_audio_path=None,
    )

    assert freed == 0
