"""Unit tests for slice G.3 framing intelligence.

The full pipeline reads from a real video file — too heavy for unit
tests. Here we exercise the verdict-decision functions directly by
constructing SubjectBox lists that simulate what `_detect_subjects`
would return on different content.
"""

from __future__ import annotations

import pytest

from chalybclip.detect.framing import (
    FramingVerdict,
    SubjectBox,
    _verdict_horizontal,
    _verdict_square,
    _verdict_vertical,
)


def _face_at(cx: float, cy: float = 0.5, w: float = 0.10, h: float = 0.15) -> SubjectBox:
    """Build a SubjectBox centered at (cx, cy)."""
    return SubjectBox(
        x=cx - w / 2,
        y=cy - h / 2,
        w=w,
        h=h,
        score=1.0,
        kind="face",
    )


# ---- horizontal source ----


def test_full_screen_required_when_action_spans_frame() -> None:
    """Two faces, one near left edge and one near right edge → action
    spans the whole frame, no single 9:16 column covers both."""
    subjects = [_face_at(0.08), _face_at(0.92), _face_at(0.10)]
    v = _verdict_horizontal(subjects, sw=1920, sh=1080)
    assert v.recommended_output == "full_screen_horizontal"
    assert v.safe_crop_box is None
    assert v.confidence >= 0.55
    assert "spans" in v.reason.lower() or "edge" in v.reason.lower()


def test_static_crop_when_action_is_centered_and_still() -> None:
    """A single face that stays put across samples → centered static crop."""
    subjects = [_face_at(0.50) for _ in range(5)]
    v = _verdict_horizontal(subjects, sw=1920, sh=1080)
    assert v.recommended_output == "mobile_crop_static"
    assert v.safe_crop_box is not None
    # Crop box should be centered around x=0.5.
    box = v.safe_crop_box
    crop_cx = box.x + box.w / 2
    assert abs(crop_cx - 0.50) < 0.05


def test_tracking_crop_when_subject_moves() -> None:
    """Face center walks 0.30 → 0.50 → 0.70 across the clip → tracking."""
    subjects = [_face_at(0.30), _face_at(0.50), _face_at(0.70)]
    v = _verdict_horizontal(subjects, sw=1920, sh=1080)
    assert v.recommended_output == "mobile_crop_tracking"
    assert v.safe_crop_box is not None
    assert v.confidence >= 0.60


def test_no_subjects_falls_back_to_centered_static_crop() -> None:
    """No face data → conservative default. Confidence is low so the
    operator knows to review."""
    v = _verdict_horizontal([], sw=1920, sh=1080)
    assert v.recommended_output == "mobile_crop_static"
    assert v.confidence <= 0.5
    assert v.safe_crop_box is not None


def test_safe_crop_box_aspect_is_9_16() -> None:
    """The chosen crop column on a 1920x1080 source should be the
    9:16 column width = 1080 * 9 / 16 = 607.5 → ~0.317 of source_w."""
    subjects = [_face_at(0.50) for _ in range(3)]
    v = _verdict_horizontal(subjects, sw=1920, sh=1080)
    assert v.safe_crop_box is not None
    expected_col_w = (1080 * 9 / 16) / 1920
    assert abs(v.safe_crop_box.w - expected_col_w) < 0.01


# ---- vertical source ----


def test_vertical_centered_subject_passes_through() -> None:
    subjects = [_face_at(0.50) for _ in range(3)]
    v = _verdict_vertical(subjects, sw=1080, sh=1920)
    assert v.recommended_output == "already_mobile"
    assert v.confidence >= 0.85


def test_vertical_offcenter_subject_recommends_tracking_recenter() -> None:
    subjects = [_face_at(0.75) for _ in range(3)]
    v = _verdict_vertical(subjects, sw=1080, sh=1920)
    assert v.recommended_output == "mobile_crop_tracking"
    assert "off" in v.reason.lower() or "center" in v.reason.lower()


def test_vertical_no_faces_still_passes_through() -> None:
    v = _verdict_vertical([], sw=1080, sh=1920)
    assert v.recommended_output == "already_mobile"


# ---- square source ----


def test_square_source_recommends_center_crop() -> None:
    v = _verdict_square([], sw=1080, sh=1080)
    assert v.recommended_output == "mobile_crop_static"
    assert v.source_orientation == "square"
    assert v.safe_crop_box is not None


# ---- verdict shape ----


def test_verdict_is_mobile_safe_property() -> None:
    """The convenience flag the renderer reads."""
    static_v = FramingVerdict(
        source_orientation="horizontal",
        recommended_output="mobile_crop_static",
        confidence=0.8,
    )
    assert static_v.is_mobile_safe
    horiz_v = FramingVerdict(
        source_orientation="horizontal",
        recommended_output="full_screen_horizontal",
        confidence=0.8,
    )
    assert not horiz_v.is_mobile_safe
    rejected_v = FramingVerdict(
        source_orientation="horizontal",
        recommended_output="reject_crop",
        confidence=0.8,
    )
    assert not rejected_v.is_mobile_safe


def test_horizontal_action_threshold_boundary() -> None:
    """At exactly the action-spread threshold (0.72) the verdict should
    flip to full_screen. We probe both sides of the boundary."""
    # Just over → full screen
    over = [_face_at(0.07), _face_at(0.80)]
    v_over = _verdict_horizontal(over, sw=1920, sh=1080)
    assert v_over.recommended_output == "full_screen_horizontal"
    # Well under → static or tracking, not full screen
    under = [_face_at(0.45), _face_at(0.55)]
    v_under = _verdict_horizontal(under, sw=1920, sh=1080)
    assert v_under.recommended_output != "full_screen_horizontal"


@pytest.mark.parametrize(
    "sw,sh",
    [(1920, 1080), (3840, 2160), (1280, 720)],
)
def test_horizontal_returns_horizontal_orientation_for_landscape(
    sw: int, sh: int,
) -> None:
    """Sanity — the verdict source_orientation reads back correctly."""
    subjects = [_face_at(0.50)]
    v = _verdict_horizontal(subjects, sw=sw, sh=sh)
    assert v.source_orientation == "horizontal"
