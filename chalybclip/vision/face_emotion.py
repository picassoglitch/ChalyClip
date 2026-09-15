"""Face presence detection via OpenCV's bundled Haar Cascade.

Phase 1 only signals **face-present** (no emotion classification). Haar
gives a bounding box, not landmarks, so the smile-vs-neutral heuristic
that earlier drafts shipped with would need a different model entirely.
We deliberately don't bring one in: Phase 2's multimodal LLM does real
emotion classification (laugh / shock / anger / sad), and a hand-tuned
landmark heuristic for `smile` alone wouldn't earn its calibration cost.

When a face is detected we emit `emotion="neutral"` so the downstream
visual_signals fan-in keeps a stable shape; the strong-emotion edge in
`detect_visual_candidates` simply won't fire on Phase 1 face data, which
is the correct behavior - the LLM upgrade lights it up later.

Sampling: 2 fps by default per PHASE_1.md SS5. The Haar detector loads
once per call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chalybclip.errors import DetectionError

from .models import FaceFrame


def detect_face_emotions(video_path: Path, *, sample_rate_hz: float = 2.0) -> list[FaceFrame]:
    """Return one FaceFrame per sampled frame (default = 2 fps)."""
    import cv2

    if not video_path.exists():
        raise DetectionError(f"video file missing: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise DetectionError(f"cv2 could not open {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            raise DetectionError(f"unknown fps for {video_path}")
        frame_skip = max(1, round(fps / sample_rate_hz))

        from .haar import haar_face_detector

        detector = haar_face_detector()

        frames: list[FaceFrame] = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                ts = frame_idx / fps
                frames.append(_face_frame_from_frame(ts, frame, detector))
            frame_idx += 1
    finally:
        cap.release()

    return frames


def _face_frame_from_frame(ts: float, frame: Any, detector: Any | None) -> FaceFrame:
    """Run Haar on `frame` and return a FaceFrame.

    `detector` may be None when the Haar XML is unavailable - we degrade
    to "no face" rather than raise so downstream still gets per-second rows.
    """
    if detector is None:
        return FaceFrame(ts=ts, has_face=False)
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return FaceFrame(ts=ts, has_face=False)
    return FaceFrame(ts=ts, has_face=True, emotion="neutral", confidence=0.7)
