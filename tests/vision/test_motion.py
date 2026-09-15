"""Tests for the OpenCV motion-energy detector."""

from __future__ import annotations

from pathlib import Path

import pytest

from chalybclip.errors import DetectionError
from chalybclip.vision import compute_motion_energy

from ._synth import build_motion_clip


def test_motion_higher_in_busy_half_than_quiet_half(tmp_path: Path) -> None:
    video = tmp_path / "motion.mp4"
    build_motion_clip(video, quiet_seconds=2.0, busy_seconds=2.0)
    motions = compute_motion_energy(video, sample_rate_hz=1.0)
    # First MotionFrame is at ts >= 1.0 (we need a previous sample to diff
    # against). Quiet samples should be near zero; busy samples should be
    # well above zero.
    quiet = [m.energy for m in motions if m.ts < 2.0]
    busy = [m.energy for m in motions if m.ts >= 2.0]
    assert quiet, "expected at least one quiet sample"
    assert busy, "expected at least one busy sample"
    assert max(quiet) < 0.05
    assert min(busy) > 0.10


def test_returns_normalized_to_unit_range(tmp_path: Path) -> None:
    video = tmp_path / "motion.mp4"
    build_motion_clip(video, quiet_seconds=1.0, busy_seconds=1.0)
    motions = compute_motion_energy(video)
    for m in motions:
        assert 0.0 <= m.energy <= 1.0


def test_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(DetectionError, match="video file missing"):
        compute_motion_energy(tmp_path / "nope.mp4")
