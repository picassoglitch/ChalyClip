"""cv2 retry-storm fix — `allow_reopen=False` must skip the per-pass open.

When `sample_clip_frames` fails (resource-starved host, undecodable codec),
the cut pipeline already paid one failed cv2 open. The legacy fallbacks in
each `_safe_*` helper would re-open the SAME source — turning one failed
open into three per candidate and hammering the exhausted resource. With
`allow_reopen=False` the helpers must degrade gracefully WITHOUT touching
cv2 again.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chalybclip.clip import service as clip_service
from chalybclip.clip.models import SmartCropBox


def _boom(*_a: object, **_k: object) -> object:
    raise AssertionError("cv2 re-open must not happen when allow_reopen=False")


def test_smart_crop_no_reopen_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clip_service, "compute_smart_crop_box", _boom)
    out = clip_service._safe_smart_crop(
        video_path=Path("missing.mp4"),
        start_s=0.0, end_s=1.0, batch=None, allow_reopen=False,
    )
    assert out is None


def test_smart_crop_reopens_when_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = SmartCropBox(x=0, y=0, w=10, h=10)
    monkeypatch.setattr(
        clip_service, "compute_smart_crop_box", lambda *_a, **_k: sentinel
    )
    out = clip_service._safe_smart_crop(
        video_path=Path("missing.mp4"),
        start_s=0.0, end_s=1.0, batch=None, allow_reopen=True,
    )
    assert out is sentinel


def test_thumbnail_no_reopen_returns_none_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(clip_service, "pick_thumbnail", _boom)
    out = clip_service._safe_thumbnail(
        video_path=Path("missing.mp4"),
        start_s=0.0, end_s=1.0, clip_dir=tmp_path, batch=None, allow_reopen=False,
    )
    assert out == (None, None)


def test_framing_no_reopen_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    import chalybclip.detect.framing as framing_mod

    monkeypatch.setattr(framing_mod, "analyze_framing", _boom)
    out = clip_service._safe_analyze_framing(
        video_path=Path("missing.mp4"),
        start_s=0.0, end_s=1.0, batch=None, allow_reopen=False,
    )
    assert out is None
