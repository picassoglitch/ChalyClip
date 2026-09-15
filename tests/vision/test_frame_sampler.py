"""Tests for the frame sampler."""

from __future__ import annotations

from pathlib import Path

import pytest

from chalybclip.errors import DetectionError
from chalybclip.vision import sample_frames, save_frames

from ._synth import build_indexed_clip


def test_sample_returns_n_jpeg_blobs(tmp_path: Path) -> None:
    video = tmp_path / "indexed.mp4"
    build_indexed_clip(video, n_seconds=2.0)
    blobs = sample_frames(video, ts=1.0, n=5, spread_s=1.0)
    assert len(blobs) == 5
    # JPEG magic bytes
    for blob in blobs:
        assert blob.startswith(b"\xff\xd8\xff")


def test_sample_n_one_returns_single_frame(tmp_path: Path) -> None:
    video = tmp_path / "indexed.mp4"
    build_indexed_clip(video, n_seconds=2.0)
    blobs = sample_frames(video, ts=1.0, n=1)
    assert len(blobs) == 1


def test_sample_clamps_to_video_bounds(tmp_path: Path) -> None:
    """Asking for frames past the end of the video silently clamps to the
    last frame rather than erroring."""
    video = tmp_path / "indexed.mp4"
    build_indexed_clip(video, n_seconds=1.0)
    blobs = sample_frames(video, ts=10.0, n=3, spread_s=0.5)
    assert len(blobs) == 3


def test_save_frames_writes_to_disk(tmp_path: Path) -> None:
    video = tmp_path / "indexed.mp4"
    build_indexed_clip(video, n_seconds=2.0)
    blobs = sample_frames(video, ts=1.0, n=3, spread_s=0.5)
    paths = save_frames(tmp_path, ts=1.0, frames=blobs)
    assert len(paths) == 3
    for p in paths:
        assert p.exists()
        assert p.suffix == ".jpg"
        assert p.parent.name == "frames"


def test_invalid_n_raises(tmp_path: Path) -> None:
    video = tmp_path / "indexed.mp4"
    build_indexed_clip(video, n_seconds=1.0)
    with pytest.raises(DetectionError, match="n must be"):
        sample_frames(video, ts=0.5, n=0)


def test_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(DetectionError, match="video file missing"):
        sample_frames(tmp_path / "nope.mp4", ts=1.0, n=3)
