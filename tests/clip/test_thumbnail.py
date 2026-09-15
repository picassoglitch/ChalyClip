"""Tests for the auto-thumbnail picker."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from chalybclip.clip.thumbnail import pick_thumbnail, save_thumbnail
from chalybclip.errors import ClipError


def _write_video(path: Path, frames: list[np.ndarray], *, fps: float = 30.0) -> None:
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def _sharp_frame(seed: int = 0) -> np.ndarray:
    """A high-contrast random pattern -> high Laplacian variance."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)


def _blurry_frame() -> np.ndarray:
    """Solid grey -> near-zero Laplacian variance."""
    return np.full((240, 320, 3), 128, dtype=np.uint8)


def _no_faces(*_args: object, **_kwargs: object) -> np.ndarray:
    """Stub for `CascadeClassifier.detectMultiScale` -> no detections."""
    return np.empty((0, 4), dtype=np.int32)


def test_picker_returns_jpeg_bytes(tmp_path: Path) -> None:
    video = tmp_path / "vid.mp4"
    _write_video(video, [_sharp_frame(i) for i in range(60)])
    with patch.object(cv2.CascadeClassifier, "detectMultiScale", _no_faces):
        jpeg, ts, breakdown = pick_thumbnail(video, start_s=0.0, end_s=2.0, sample_n=3)
    assert jpeg.startswith(b"\xff\xd8\xff")
    assert 0.0 <= ts <= 2.0
    assert "score" in breakdown
    assert "sharpness" in breakdown


def test_picker_prefers_face_frames(tmp_path: Path) -> None:
    """Two equally-sharp samples; one has a face. The face frame should win."""
    video = tmp_path / "vid.mp4"
    # 60 frames, all sharp (so the picker decides on face_present, not sharpness).
    frames = [_sharp_frame(i) for i in range(60)]
    _write_video(video, frames)

    face_box = np.array([[10, 10, 60, 60]], dtype=np.int32)
    no_face = np.empty((0, 4), dtype=np.int32)
    # sample_n=3 -> sample times 0.0, 1.0, 2.0. Make the middle one the face.
    side_effect = [no_face, face_box, no_face]

    with patch.object(
        cv2.CascadeClassifier,
        "detectMultiScale",
        side_effect=side_effect,
    ):
        _jpeg, ts, breakdown = pick_thumbnail(
            video, start_s=0.0, end_s=2.0, sample_n=3
        )
    # The face sample (ts=1.0) should win.
    assert ts == pytest.approx(1.0, abs=0.05)
    assert breakdown["has_face"] == 1.0


def test_picker_prefers_sharp_over_blurry(tmp_path: Path) -> None:
    """No face anywhere; sharp frames beat blurry ones on score."""
    video = tmp_path / "vid.mp4"
    # 30 sharp frames, then 30 blurry frames.
    frames = [_sharp_frame(i) for i in range(30)] + [_blurry_frame() for _ in range(30)]
    _write_video(video, frames)
    with patch.object(cv2.CascadeClassifier, "detectMultiScale", _no_faces):
        _jpeg, ts, breakdown = pick_thumbnail(
            video, start_s=0.0, end_s=2.0, sample_n=5
        )
    # Sharp frames are in the first second; the picked timestamp should be
    # in that half. (Exact timestamp depends on which sample was sharpest.)
    assert ts < 1.0
    assert breakdown["sharpness"] > 0.0


def test_save_thumbnail_writes_jpg(tmp_path: Path) -> None:
    clip_dir = tmp_path / "clips" / "clp_X"
    out = save_thumbnail(clip_dir, b"\xff\xd8\xff fake jpeg")
    assert out.exists()
    assert out.suffix == ".jpg"
    assert out.read_bytes() == b"\xff\xd8\xff fake jpeg"


def test_invalid_window_raises(tmp_path: Path) -> None:
    video = tmp_path / "vid.mp4"
    _write_video(video, [_sharp_frame() for _ in range(10)])
    with pytest.raises(ClipError, match="end_s must be"):
        pick_thumbnail(video, start_s=2.0, end_s=1.0)


def test_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(ClipError, match="video file missing"):
        pick_thumbnail(tmp_path / "nope.mp4", start_s=0.0, end_s=1.0)
