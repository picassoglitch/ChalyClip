"""Framing intelligence — slice G.3.

For every clip, decide HOW it should be exported: full-screen
horizontal, static 9:16 crop, tracking 9:16 crop, already-mobile
recenter, or reject (no safe crop exists). Drives the auto-export
action wiring in slice G.5 so the renderer never produces a
9:16 clip that cuts off the gameplay UI or hides one of two
far-apart speakers.

Pipeline:

  1. Read the source video metadata (orientation, dimensions).
  2. Sample N frames across the clip window.
  3. For each frame, run face detection + a coarse motion-region
     pass.
  4. Aggregate:
       - face center spread (how much does the action move
         across the frame width?)
       - action width fraction (how much of the frame is actually
         carrying meaningful content?)
       - face-area fraction (the closer the subject is, the less
         we need a wide frame)
  5. Pick `recommended_output` from the spec's five options:
       full_screen_horizontal / mobile_crop_static /
       mobile_crop_tracking  / already_mobile / reject_crop
  6. Emit a `safe_crop_box` when a static crop is chosen.

This is heuristic on purpose — the goal is "good enough to drive
the export branch correctly 90% of the time", not vision-LLM-level
perfection. Operators can override per-clip from the dashboard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    # Slice K.3 fix — `chalybclip.clip.models` imports `chalybclip.detect.Candidate`,
    # and `chalybclip.detect.__init__` re-exports this module. Pulling
    # SmartCropBox at module top creates a circular import (clip.models
    # is still loading when framing tries to read it back).
    # Type-check-only import keeps the annotation working without the cycle.
    from chalybclip.clip.frame_pool import ClipFrameBatch
    from chalybclip.clip.models import SmartCropBox

# ---- Type aliases ----

SourceOrientation = Literal["horizontal", "vertical", "square"]
RecommendedOutput = Literal[
    "full_screen_horizontal",
    "mobile_crop_static",
    "mobile_crop_tracking",
    "already_mobile",
    "reject_crop",
]


@dataclass(frozen=True)
class SubjectBox:
    """One detected subject region on a sampled frame.

    Coordinates are FRACTIONAL — multiply by source_w / source_h to
    get pixels. `score` is the detector's confidence (or normalized
    area when no confidence was available)."""

    x: float
    y: float
    w: float
    h: float
    score: float = 1.0
    kind: Literal["face", "motion", "action"] = "face"


@dataclass(frozen=True)
class FramingVerdict:
    """The export-shape decision for one clip.

    Consumed by G.5's auto-export wiring + the dashboard's framing
    label. `safe_crop_box` is fractional like SubjectBox; multiply
    by source_w / source_h to convert to pixels for ffmpeg.
    """

    source_orientation: SourceOrientation
    recommended_output: RecommendedOutput
    confidence: float  # 0.0 - 1.0
    subject_boxes: list[SubjectBox] = field(default_factory=list)
    action_boxes: list[SubjectBox] = field(default_factory=list)
    safe_crop_box: SubjectBox | None = None
    reason: str = ""

    @property
    def is_mobile_safe(self) -> bool:
        """True when the renderer can produce a 9:16 export without
        loss — either the source is already mobile or a safe crop
        was found."""
        return self.recommended_output in (
            "mobile_crop_static",
            "mobile_crop_tracking",
            "already_mobile",
        )


# ---- Tuning constants ----

# A 9:16 column over a horizontal source covers crop_w / source_w of
# the width. Below this ratio of meaningful action AROUND the column,
# we say "full screen required" — the action lives outside the crop.
_ACTION_SPREAD_HORIZONTAL_THRESHOLD = 0.72
"""If detected action spreads across >72% of the source width on a
horizontal video, declare it full-screen-required."""

_TRACKING_VARIANCE_THRESHOLD = 0.18
"""Standard-deviation-of-face-centers (as fraction of source_w) above
which the subject is 'moving' enough to need tracking crop. 0.18 ≈
the subject's face slides across about 18% of the frame on average."""

_MIN_SAFE_CROP_OVERLAP = 0.55
"""When face/action centers are bounded, the resulting 9:16 column
must cover at least 55% of the average action's bbox to be considered
'safe'. Below that → reject_crop."""

_OFF_CENTER_RECENTER_THRESHOLD = 0.12
"""For vertical-source clips, if the average face center deviates more
than 12% from the frame center, recommend a tracking-style recenter."""


# ---- Public entry ----


def analyze_framing(
    video_path: Path,
    *,
    start_s: float,
    end_s: float,
    sample_n: int = 6,
) -> FramingVerdict:
    """Decide how a clip should be exported.

    Defensively guarded — any failure path returns a verdict with
    `recommended_output="mobile_crop_static"` (the conservative
    default) so the pipeline doesn't break on test stub videos /
    unreadable codecs.
    """
    if end_s <= start_s:
        return _fallback_verdict("non-positive clip window")
    if not video_path.exists():
        return _fallback_verdict(f"video missing: {video_path}")

    try:
        import cv2
    except Exception:  # noqa: BLE001
        return _fallback_verdict("opencv not available")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return _fallback_verdict(f"cv2 could not open {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or total <= 0 or sw <= 0 or sh <= 0:
            return _fallback_verdict("unusable video metadata")

        orientation: SourceOrientation
        ratio = sw / max(1, sh)
        if 0.95 <= ratio <= 1.05:
            orientation = "square"
        elif sw > sh:
            orientation = "horizontal"
        else:
            orientation = "vertical"

        sample_times = _sample_times(start_s, end_s, sample_n)
        subjects = _detect_subjects(cap, sample_times, fps, total, sw, sh)

        if orientation == "vertical":
            return _verdict_vertical(subjects, sw=sw, sh=sh)
        if orientation == "square":
            return _verdict_square(subjects, sw=sw, sh=sh)
        return _verdict_horizontal(subjects, sw=sw, sh=sh)
    finally:
        cap.release()


# ---- Internals ----


def _sample_times(start_s: float, end_s: float, sample_n: int) -> list[float]:
    if sample_n <= 1:
        return [(start_s + end_s) / 2.0]
    step = (end_s - start_s) / (sample_n - 1)
    return [start_s + i * step for i in range(sample_n)]


def _detect_subjects(
    cap: object,
    sample_times: list[float],
    fps: float,
    total_frames: int,
    sw: int,
    sh: int,
) -> list[SubjectBox]:
    """For each sample time, return the largest detected face (or an
    empty list when none found). Coordinates are FRACTIONAL relative
    to the source frame."""
    import cv2

    from chalybclip.vision.haar import haar_face_detector

    detector = haar_face_detector()
    if detector is None:
        return []

    boxes: list[SubjectBox] = []
    for ts in sample_times:
        idx = max(0, min(total_frames - 1, round(ts * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # type: ignore[attr-defined]
        ret, frame = cap.read()  # type: ignore[attr-defined]
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            continue
        x, y, w, h = max(faces, key=lambda r: int(r[2]) * int(r[3]))
        boxes.append(SubjectBox(
            x=float(x) / sw,
            y=float(y) / sh,
            w=float(w) / sw,
            h=float(h) / sh,
            score=1.0,
            kind="face",
        ))
    return boxes


def _verdict_horizontal(
    subjects: list[SubjectBox], *, sw: int, sh: int,
) -> FramingVerdict:
    """Choose between full_screen_horizontal / mobile_crop_static /
    mobile_crop_tracking / reject_crop for a horizontal source."""
    # No faces → can't make a confident framing call. Conservative
    # default: mobile_crop_static centered.
    if not subjects:
        return FramingVerdict(
            source_orientation="horizontal",
            recommended_output="mobile_crop_static",
            confidence=0.40,
            subject_boxes=[],
            action_boxes=[],
            safe_crop_box=_centered_safe_crop(),
            reason=(
                "No subjects detected. Defaulting to a centered 9:16 "
                "crop — operator may want to review."
            ),
        )

    # Face-center spread tells us motion across the frame.
    centers_x = [s.x + s.w / 2.0 for s in subjects]
    mean_cx = sum(centers_x) / len(centers_x)
    variance = sum((c - mean_cx) ** 2 for c in centers_x) / len(centers_x)
    stddev = math.sqrt(variance)

    # Action width — how spread out are the subjects horizontally?
    # Take the union bbox left/right edges.
    left_edge = min(s.x for s in subjects)
    right_edge = max(s.x + s.w for s in subjects)
    action_width_frac = right_edge - left_edge

    # 9:16 column width as a fraction of source width.
    target_col_w_frac = (sh * 9.0 / 16.0) / sw if sw > 0 else 1.0

    # Rule A — action spans more than the threshold → full-screen.
    if action_width_frac >= _ACTION_SPREAD_HORIZONTAL_THRESHOLD:
        return FramingVerdict(
            source_orientation="horizontal",
            recommended_output="full_screen_horizontal",
            confidence=min(0.95, 0.55 + (action_width_frac - 0.72) * 1.5),
            subject_boxes=subjects,
            action_boxes=[
                SubjectBox(
                    x=left_edge, y=0.0,
                    w=action_width_frac, h=1.0,
                    score=action_width_frac, kind="action",
                )
            ],
            safe_crop_box=None,
            reason=(
                f"Action spans {action_width_frac:.0%} of the frame width — "
                f"a 9:16 crop would hide subjects on the edges. Export "
                f"full-screen horizontal (or mobile blurred-bg fallback)."
            ),
        )

    # Rule B — face center moves a lot across the clip → tracking crop.
    if stddev > _TRACKING_VARIANCE_THRESHOLD:
        return FramingVerdict(
            source_orientation="horizontal",
            recommended_output="mobile_crop_tracking",
            confidence=min(0.90, 0.60 + (stddev - 0.18) * 2.0),
            subject_boxes=subjects,
            action_boxes=[],
            safe_crop_box=_centered_safe_crop_at(mean_cx, target_col_w_frac),
            reason=(
                f"Subject moves across {stddev:.0%} of the frame width — "
                f"a static crop would lose the subject. Tracking crop "
                f"keyframes follow the face."
            ),
        )

    # Rule C — action contained near mean_cx → static crop centered there.
    crop_box = _centered_safe_crop_at(mean_cx, target_col_w_frac)
    # Sanity: does the crop column actually contain the subjects?
    coverage = _subject_coverage(subjects, crop_box)
    if coverage < _MIN_SAFE_CROP_OVERLAP:
        return FramingVerdict(
            source_orientation="horizontal",
            recommended_output="reject_crop",
            confidence=1.0 - coverage,
            subject_boxes=subjects,
            action_boxes=[],
            safe_crop_box=None,
            reason=(
                f"No safe 9:16 crop — only {coverage:.0%} of subject "
                f"coverage. Operator must export full-screen or trim "
                f"the clip."
            ),
        )
    return FramingVerdict(
        source_orientation="horizontal",
        recommended_output="mobile_crop_static",
        confidence=min(0.95, 0.55 + coverage * 0.4),
        subject_boxes=subjects,
        action_boxes=[],
        safe_crop_box=crop_box,
        reason=(
            f"Action centered around x={mean_cx:.0%} — a static 9:16 "
            f"crop covers {coverage:.0%} of the subject."
        ),
    )


def _verdict_vertical(
    subjects: list[SubjectBox], *, sw: int, sh: int,
) -> FramingVerdict:
    """Already-mobile source. Recommend recenter when the subject
    is off-center; otherwise pass through."""
    if not subjects:
        return FramingVerdict(
            source_orientation="vertical",
            recommended_output="already_mobile",
            confidence=0.70,
            subject_boxes=[],
            action_boxes=[],
            safe_crop_box=None,
            reason="Vertical source — no recentering needed.",
        )
    centers_x = [s.x + s.w / 2.0 for s in subjects]
    mean_cx = sum(centers_x) / len(centers_x)
    off_center = abs(mean_cx - 0.5)
    if off_center > _OFF_CENTER_RECENTER_THRESHOLD:
        return FramingVerdict(
            source_orientation="vertical",
            recommended_output="mobile_crop_tracking",
            confidence=min(0.85, 0.55 + off_center * 2.0),
            subject_boxes=subjects,
            action_boxes=[],
            safe_crop_box=None,
            reason=(
                f"Vertical source but subject is {off_center:.0%} off "
                f"center — recommend tracking-crop recenter."
            ),
        )
    return FramingVerdict(
        source_orientation="vertical",
        recommended_output="already_mobile",
        confidence=0.90,
        subject_boxes=subjects,
        action_boxes=[],
        safe_crop_box=None,
        reason="Vertical source with centered subject — export as-is.",
    )


def _verdict_square(
    subjects: list[SubjectBox], *, sw: int, sh: int,
) -> FramingVerdict:
    """Square 1:1 source — usually crops cleanly to 9:16 by trimming
    the top + bottom. Treat as mobile_crop_static with a centered box."""
    return FramingVerdict(
        source_orientation="square",
        recommended_output="mobile_crop_static",
        confidence=0.80,
        subject_boxes=subjects,
        action_boxes=[],
        safe_crop_box=SubjectBox(x=0.21875, y=0.0, w=0.5625, h=1.0),
        reason="Square source — center-crop to 9:16.",
    )


# ---- Crop helpers ----


def _centered_safe_crop() -> SubjectBox:
    """Centered 9:16 column over the source frame."""
    return SubjectBox(x=(1 - 9 / 16) / 2, y=0.0, w=9 / 16, h=1.0, kind="action")


def _centered_safe_crop_at(center_x: float, col_w_frac: float) -> SubjectBox:
    """9:16 column centered on `center_x`, clamped to the frame."""
    x = max(0.0, min(1.0 - col_w_frac, center_x - col_w_frac / 2.0))
    return SubjectBox(x=x, y=0.0, w=col_w_frac, h=1.0, kind="action")


def _subject_coverage(subjects: list[SubjectBox], crop: SubjectBox) -> float:
    """Mean fraction of each subject bbox area inside the crop column."""
    if not subjects:
        return 0.0
    crop_x0, crop_x1 = crop.x, crop.x + crop.w
    crop_y0, crop_y1 = crop.y, crop.y + crop.h
    totals: list[float] = []
    for s in subjects:
        sx0, sx1 = s.x, s.x + s.w
        sy0, sy1 = s.y, s.y + s.h
        inter_w = max(0.0, min(sx1, crop_x1) - max(sx0, crop_x0))
        inter_h = max(0.0, min(sy1, crop_y1) - max(sy0, crop_y0))
        s_area = max(1e-9, (sx1 - sx0) * (sy1 - sy0))
        totals.append((inter_w * inter_h) / s_area)
    return sum(totals) / len(totals)


def _fallback_verdict(reason: str) -> FramingVerdict:
    """Conservative default when analysis can't run."""
    return FramingVerdict(
        source_orientation="horizontal",
        recommended_output="mobile_crop_static",
        confidence=0.25,
        subject_boxes=[],
        action_boxes=[],
        safe_crop_box=_centered_safe_crop(),
        reason=reason,
    )


def analyze_framing_from_frames(batch: "ClipFrameBatch") -> FramingVerdict:
    """Same contract as `analyze_framing` but consumes a shared
    `ClipFrameBatch` instead of re-opening the source video (Task 1c).

    Same orientation classification + horizontal/vertical/square
    verdict path as the legacy entry — the only difference is the
    frame source. On any failure (no opencv, empty batch, bad
    metadata) returns the same `_fallback_verdict` the legacy path
    would have produced.
    """
    if not batch.frames_bgr:
        return _fallback_verdict("empty frame batch")
    try:
        import cv2
    except Exception:  # noqa: BLE001
        return _fallback_verdict("opencv not available")

    sw, sh = batch.source_w, batch.source_h
    if sw <= 0 or sh <= 0:
        return _fallback_verdict("unusable batch metadata")

    orientation: SourceOrientation
    ratio = sw / max(1, sh)
    if 0.95 <= ratio <= 1.05:
        orientation = "square"
    elif sw > sh:
        orientation = "horizontal"
    else:
        orientation = "vertical"

    from chalybclip.vision.haar import haar_face_detector

    detector = haar_face_detector()
    subjects: list[SubjectBox] = []
    if detector is not None:
        for frame in batch.frames_bgr:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            if len(faces) == 0:
                continue
            x, y, w, h = max(faces, key=lambda r: int(r[2]) * int(r[3]))
            subjects.append(SubjectBox(
                x=float(x) / sw,
                y=float(y) / sh,
                w=float(w) / sw,
                h=float(h) / sh,
                score=1.0,
                kind="face",
            ))

    if orientation == "vertical":
        return _verdict_vertical(subjects, sw=sw, sh=sh)
    if orientation == "square":
        return _verdict_square(subjects, sw=sw, sh=sh)
    return _verdict_horizontal(subjects, sw=sw, sh=sh)


def to_smart_crop_box(box: SubjectBox, *, source_w: int, source_h: int) -> "SmartCropBox":
    """Convert a fractional SubjectBox into the absolute-pixel
    `SmartCropBox` the renderer's `crop=` ffmpeg filter consumes.

    Lazy-imports SmartCropBox so framing.py stays cycle-free with
    `chalybclip.clip.models` (which transitively imports this module
    via `chalybclip.detect.Candidate`).
    """
    from chalybclip.clip.models import SmartCropBox

    return SmartCropBox(
        x=max(0, round(box.x * source_w)),
        y=max(0, round(box.y * source_h)),
        w=max(1, round(box.w * source_w)),
        h=max(1, round(box.h * source_h)),
    )


__all__ = [
    "FramingVerdict",
    "RecommendedOutput",
    "SourceOrientation",
    "SubjectBox",
    "analyze_framing",
    "analyze_framing_from_frames",
    "to_smart_crop_box",
]
