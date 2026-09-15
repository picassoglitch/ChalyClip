"""Tests for the face presence detector.

We can't synthesize a face that Haar Cascade will actually detect, so
the real `detect_face_emotions` function is exercised end-to-end only
in manual smoke tests. These unit tests pin down the FaceFrame shape
returned when the Haar detector finds vs. doesn't find a face.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from chalybclip.vision.face_emotion import _face_frame_from_frame


def _frame() -> np.ndarray:
    """A trivial 64x64 BGR frame."""
    return np.zeros((64, 64, 3), dtype=np.uint8)


def test_face_frame_no_face() -> None:
    """Detector returns zero detections -> has_face is False, no emotion."""
    detector = MagicMock()
    detector.detectMultiScale.return_value = np.empty((0, 4), dtype=np.int32)
    f = _face_frame_from_frame(ts=2.0, frame=_frame(), detector=detector)
    assert f.has_face is False
    assert f.emotion is None
    assert f.ts == 2.0


def test_face_frame_with_face_returns_neutral() -> None:
    """A detection -> has_face=True, emotion='neutral' (Phase 1 floor).

    Phase 1 doesn't do emotion classification - the multimodal LLM in
    Phase 2 does. So every detected face is reported as 'neutral' and
    the strong-emotion edge in the visual fan-in stays quiet until then.
    """
    detector = MagicMock()
    detector.detectMultiScale.return_value = np.array(
        [[10, 10, 40, 40]], dtype=np.int32
    )
    f = _face_frame_from_frame(ts=5.0, frame=_frame(), detector=detector)
    assert f.has_face is True
    assert f.emotion == "neutral"
    assert 0.0 < f.confidence <= 1.0


def test_face_frame_with_no_detector_degrades_to_no_face() -> None:
    """Missing Haar XML -> caller passes detector=None -> no face."""
    f = _face_frame_from_frame(ts=0.0, frame=_frame(), detector=None)
    assert f.has_face is False
    assert f.emotion is None
