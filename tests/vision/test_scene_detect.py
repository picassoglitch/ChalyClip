"""Tests for the PySceneDetect wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from chalybclip.errors import DetectionError
from chalybclip.vision import detect_scene_cuts

from ._synth import build_scene_cut_clip, write_video


def test_detects_cut_at_color_change_boundary(tmp_path: Path) -> None:
    """Red for 1 s, then yellow for 1 s — a textbook scene change."""
    video = tmp_path / "scene.mp4"
    build_scene_cut_clip(video, seconds_each=1.0)
    cuts = detect_scene_cuts(video, threshold=27.0)
    # PySceneDetect's first scene starts at 0; the cut is the start of
    # scene #2, which sits near the 1 s boundary. Allow a frame or two
    # of slop because of the encoder's GOP structure.
    assert len(cuts) >= 1
    assert any(0.7 <= c.ts <= 1.3 for c in cuts)


def test_constant_video_yields_no_cuts(tmp_path: Path) -> None:
    """A solid-color video should never trip the detector."""
    video = tmp_path / "solid.mp4"
    from ._synth import solid_frame

    frames = [solid_frame((40, 40, 40)) for _ in range(60)]
    write_video(video, frames)
    cuts = detect_scene_cuts(video, threshold=27.0)
    assert cuts == []


def test_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(DetectionError, match="video file missing"):
        detect_scene_cuts(tmp_path / "nope.mp4")
