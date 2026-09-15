"""Shared frame-pool sampling (Task 1c)."""

from __future__ import annotations

from pathlib import Path

import pytest

from chalybclip.clip.frame_pool import (
    DEFAULT_SAMPLE_N,
    ClipFrameBatch,
    sample_clip_frames,
)


def test_returns_none_on_missing_video(tmp_path: Path) -> None:
    missing = tmp_path / "nope.mp4"
    assert sample_clip_frames(missing, start_s=0.0, end_s=5.0) is None


def test_returns_none_on_invalid_window(tmp_path: Path) -> None:
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"\x00\x00fake")
    # end <= start → None
    assert sample_clip_frames(fake, start_s=5.0, end_s=5.0) is None
    assert sample_clip_frames(fake, start_s=5.0, end_s=4.0) is None


def test_returns_none_on_undecodable_video(tmp_path: Path) -> None:
    """Placeholder bytes look like a file but OpenCV can't decode
    them — sample_clip_frames returns None, callers fall back to
    their legacy per-pass open which raises ClipError and gets caught
    by the _safe_* wrappers. Existing tests rely on this fall-through."""
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"\x00\x00fakevideo")
    assert sample_clip_frames(fake, start_s=0.0, end_s=5.0) is None


def test_default_sample_n_matches_thumbnail_demand() -> None:
    """The shared pool oversamples to satisfy the heaviest consumer
    (pick_thumbnail's default = 10) without making the legacy callers
    aware of it. Asserting the constant in case a refactor accidentally
    drops it below thumbnail's demand."""
    assert DEFAULT_SAMPLE_N >= 10


def test_batch_dataclass_len() -> None:
    """ClipFrameBatch's __len__ returns the actual decoded count so
    consumers can guard on `if not batch.frames_bgr`."""
    batch = ClipFrameBatch(
        frames_bgr=["frame_a", "frame_b"],
        sample_times=[1.0, 2.0],
        source_w=1920,
        source_h=1080,
        fps=30.0,
    )
    assert len(batch) == 2
