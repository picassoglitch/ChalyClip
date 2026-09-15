"""Tests for smart_crop - face-aware 9:16 crop selection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from chalybclip.clip.models import SmartCropBox
from chalybclip.clip.smart_crop import (
    _detect_face_centers,
    compute_smart_crop_box,
    crop_box_to_ffmpeg_filter,
)
from chalybclip.errors import ClipError


def _write_solid_video(path: Path, *, w: int, h: int, fps: float, duration_s: float) -> None:
    n = round(fps * duration_s)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for _ in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_no_face_falls_back_to_centered_crop(tmp_path: Path) -> None:
    """A blank video has no detectable face - crop should be centered."""
    video = tmp_path / "blank.mp4"
    _write_solid_video(video, w=1920, h=1080, fps=30.0, duration_s=2.0)
    box = compute_smart_crop_box(video, start_s=0.0, end_s=2.0)
    expected_w = (1080 * 9) // 16  # 607
    assert box.w == expected_w
    assert box.h == 1080
    # Centered: x_offset = (1920 - 607) / 2 = 656
    assert box.x == (1920 - expected_w) // 2
    assert box.y == 0


def test_portrait_source_no_crop_needed(tmp_path: Path) -> None:
    """A source already at >= 9:16 returns the full frame."""
    video = tmp_path / "portrait.mp4"
    _write_solid_video(video, w=480, h=1080, fps=30.0, duration_s=1.0)
    box = compute_smart_crop_box(video, start_s=0.0, end_s=1.0)
    assert box.x == 0
    assert box.w == 480
    assert box.h == 1080


def test_face_detection_shifts_crop_to_face(tmp_path: Path) -> None:
    """A face on the right side of the frame should pull the crop right."""
    video = tmp_path / "video.mp4"
    _write_solid_video(video, w=1920, h=1080, fps=30.0, duration_s=2.0)

    # Mock the face-center detector to return faces at x=1500 (right side).
    with patch(
        "chalybclip.clip.smart_crop._detect_face_centers",
        return_value=[1500.0, 1500.0, 1500.0],
    ):
        box = compute_smart_crop_box(video, start_s=0.0, end_s=2.0)

    # crop_w = 607; centered on 1500 means x = 1500 - 304 = 1196.
    expected_w = (1080 * 9) // 16
    assert box.w == expected_w
    expected_x = 1500 - expected_w // 2
    # Account for rounding: x should be near 1196.
    assert abs(box.x - expected_x) <= 1


def test_crop_clamps_to_frame_bounds(tmp_path: Path) -> None:
    """A face near the right edge - crop x_offset can't exceed source_w - crop_w."""
    video = tmp_path / "video.mp4"
    _write_solid_video(video, w=1920, h=1080, fps=30.0, duration_s=1.0)

    with patch(
        "chalybclip.clip.smart_crop._detect_face_centers",
        return_value=[1900.0],  # almost at the right edge
    ):
        box = compute_smart_crop_box(video, start_s=0.0, end_s=1.0)

    crop_w = (1080 * 9) // 16
    assert box.x + box.w <= 1920
    assert box.x == 1920 - crop_w


def test_crop_clamps_to_zero_when_face_is_left_edge(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    _write_solid_video(video, w=1920, h=1080, fps=30.0, duration_s=1.0)

    with patch(
        "chalybclip.clip.smart_crop._detect_face_centers",
        return_value=[20.0],
    ):
        box = compute_smart_crop_box(video, start_s=0.0, end_s=1.0)
    assert box.x == 0


def test_invalid_window_raises(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    _write_solid_video(video, w=1920, h=1080, fps=30.0, duration_s=1.0)
    with pytest.raises(ClipError, match="end_s must be"):
        compute_smart_crop_box(video, start_s=2.0, end_s=1.0)


def test_invalid_sample_n_raises(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    _write_solid_video(video, w=1920, h=1080, fps=30.0, duration_s=1.0)
    with pytest.raises(ClipError, match="sample_n"):
        compute_smart_crop_box(video, start_s=0.0, end_s=1.0, sample_n=0)


def test_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(ClipError, match="video file missing"):
        compute_smart_crop_box(tmp_path / "nope.mp4", start_s=0.0, end_s=1.0)


def test_crop_box_to_ffmpeg_filter_string() -> None:
    box = SmartCropBox(x=100, y=0, w=607, h=1080)
    s = crop_box_to_ffmpeg_filter(box, output_w=1080, output_h=1920)
    assert s == "crop=607:1080:100:0,scale=1080:1920"


def test_detect_face_centers_drops_frames_without_faces(tmp_path: Path) -> None:
    """Frames with no detection are skipped, not counted as 'face at center'."""
    video = tmp_path / "video.mp4"
    _write_solid_video(video, w=1920, h=1080, fps=30.0, duration_s=2.0)
    cap = cv2.VideoCapture(str(video))
    try:
        # Three samples: empty, one face at x=1400 w=80 (center=1440), empty.
        no_faces = np.empty((0, 4), dtype=np.int32)
        one_face = np.array([[1400, 500, 80, 80]], dtype=np.int32)
        side_effect = [no_faces, one_face, no_faces]
        with patch.object(
            cv2.CascadeClassifier, "detectMultiScale", side_effect=side_effect
        ):
            centers = _detect_face_centers(
                cap=cap, sample_times=[0.0, 1.0, 2.0], fps=30.0, total_frames=60
            )
    finally:
        cap.release()

    # Only one center, at 1400 + 80/2 = 1440.
    assert len(centers) == 1
    assert centers[0] == pytest.approx(1440.0)


def test_detect_face_centers_picks_largest_when_multiple(tmp_path: Path) -> None:
    """When a frame has multiple detections, the largest by area wins."""
    video = tmp_path / "video.mp4"
    _write_solid_video(video, w=1920, h=1080, fps=30.0, duration_s=1.0)
    cap = cv2.VideoCapture(str(video))
    try:
        # Two faces in one frame: small at x=100 w=40, big at x=1400 w=120.
        # The big one (area=120*120=14400) wins; its center = 1400+60 = 1460.
        faces = np.array([[100, 100, 40, 40], [1400, 100, 120, 120]], dtype=np.int32)
        with patch.object(
            cv2.CascadeClassifier, "detectMultiScale", return_value=faces
        ):
            centers = _detect_face_centers(
                cap=cap, sample_times=[0.0], fps=30.0, total_frames=30
            )
    finally:
        cap.release()
    assert len(centers) == 1
    assert centers[0] == pytest.approx(1460.0)
