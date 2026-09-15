"""haar_face_detector — the ONE constructor for the Haar face cascade.

Pins the degradation contract: a cv2 build without the legacy Haar API
(OpenCV 5 removed cv2.CascadeClassifier — the unpinned prod rebuild that
crashed every analyze_video step) yields None, and every consumer treats
None as "no faces" instead of raising.
"""

from __future__ import annotations

import pytest

from chalybclip.vision.haar import haar_face_detector


def test_returns_working_detector_on_healthy_cv2() -> None:
    detector = haar_face_detector()
    assert detector is not None
    assert not detector.empty()


def test_returns_none_when_cascade_api_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cv2

    monkeypatch.delattr(cv2, "CascadeClassifier")
    assert haar_face_detector() is None


def test_framing_degrades_without_cascade_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The analyze_video path must not die on a Haar-less cv2 — subject
    detection returns empty and framing falls back to its no-face verdict."""
    import cv2

    from chalybclip.detect.framing import _detect_subjects

    monkeypatch.delattr(cv2, "CascadeClassifier")
    assert _detect_subjects(
        cap=object(), sample_times=[0.0, 1.0], fps=30.0,
        total_frames=60, sw=1920, sh=1080,
    ) == []
