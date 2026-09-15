"""Shared, failure-tolerant constructor for OpenCV's Haar face detector.

Every face-detection consumer (vision face-presence, detect framing,
clip thumbnails, smart crop) builds the same bundled frontal-face
cascade — and every one of them treats "no detector" as "no faces",
degrading to its non-face fallback. This helper is the ONE place that
constructs it, so a missing/limited cv2 build degrades everywhere
instead of crashing a pipeline step: OpenCV 5 removed the legacy Haar
API entirely (`cv2.CascadeClassifier` raises AttributeError), which an
unpinned prod rebuild shipped and every analyze_video step died on.
"""

from __future__ import annotations

from typing import Any

from chalybclip.logging import get_logger

_log = get_logger("chalybclip.vision.haar")
_warned = False


def haar_face_detector() -> Any | None:
    """The bundled frontal-face CascadeClassifier, or None when the
    running cv2 can't provide one (legacy Haar API removed, bundle XML
    missing/unreadable). Callers treat None as "no faces detected".
    Logs the degradation once per process, not per frame batch."""
    global _warned
    import cv2

    try:
        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
        )
    except AttributeError:
        # OpenCV 5 moved the legacy Haar API out of the main module.
        if not _warned:
            _warned = True
            _log.warning(
                "haar_unavailable",
                cv2_version=getattr(cv2, "__version__", "unknown"),
                reason="cv2 build has no CascadeClassifier — face detection "
                       "degrades to none; pin opencv-python<5",
            )
        return None
    if detector.empty():
        if not _warned:
            _warned = True
            _log.warning(
                "haar_unavailable",
                cv2_version=getattr(cv2, "__version__", "unknown"),
                reason="bundled haarcascade XML missing or unreadable",
            )
        return None
    return detector
