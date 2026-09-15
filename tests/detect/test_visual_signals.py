"""Tests for the visual-signals detector (Task 6 fusion)."""

from __future__ import annotations

import pytest

from chalybclip.config import VisualConfig
from chalybclip.detect import detect_visual_candidates
from chalybclip.errors import DetectionError
from chalybclip.vision import VisualSignal, VisualSignalTrack

from ._fixtures import make_stream


def _track(*signals: VisualSignal) -> VisualSignalTrack:
    return VisualSignalTrack(
        stream_id="str_01TEST", tenant_id="default", signals=list(signals)
    )


def _config(
    *,
    enabled: bool = True,
    weight: float = 0.6,
    cut_weight: float = 1.0,
    emotion_weight: float = 0.7,
    motion_weight: float = 0.5,
    motion_baseline_window_s: float = 30.0,
    motion_spike_ratio: float = 2.0,
    emotion_labels: list[str] | None = None,
) -> VisualConfig:
    return VisualConfig(
        enabled=enabled,
        weight=weight,
        cut_weight=cut_weight,
        emotion_weight=emotion_weight,
        motion_weight=motion_weight,
        motion_baseline_window_s=motion_baseline_window_s,
        motion_spike_ratio=motion_spike_ratio,
        emotion_labels=emotion_labels or ["smile", "laugh", "shock"],
    )


def test_disabled_detector_returns_empty() -> None:
    track = _track(VisualSignal(ts_offset_s=0.0, scene_cut=True))
    assert detect_visual_candidates(
        "default", make_stream(), track, _config(enabled=False)
    ) == []


def test_empty_track_returns_empty() -> None:
    assert detect_visual_candidates(
        "default", make_stream(), _track(), _config()
    ) == []


def test_scene_cut_emits_one_candidate() -> None:
    """A scene cut at t=5 produces one visual candidate."""
    track = _track(
        VisualSignal(ts_offset_s=0.0),
        VisualSignal(ts_offset_s=1.0),
        VisualSignal(ts_offset_s=5.0, scene_cut=True),
        VisualSignal(ts_offset_s=10.0),
    )
    cands = detect_visual_candidates("default", make_stream(), track, _config())
    assert len(cands) == 1
    c = cands[0]
    assert c.timestamp == 5.0
    assert c.reason == "visual"
    assert c.evidence["scene_cut"] is True
    assert "scene_cut" in c.evidence["sub_scores"]


def test_emotion_transition_fires_only_at_edge() -> None:
    """A smile starting at t=2 and continuing through t=5 fires once,
    at t=2. The middle of the smile (t=3, 4, 5) doesn't re-fire."""
    track = _track(
        VisualSignal(ts_offset_s=0.0, face_emotion="neutral"),
        VisualSignal(ts_offset_s=1.0, face_emotion="neutral"),
        VisualSignal(ts_offset_s=2.0, face_emotion="smile"),
        VisualSignal(ts_offset_s=3.0, face_emotion="smile"),
        VisualSignal(ts_offset_s=4.0, face_emotion="smile"),
        VisualSignal(ts_offset_s=5.0, face_emotion="smile"),
    )
    cands = detect_visual_candidates("default", make_stream(), track, _config())
    assert len(cands) == 1
    assert cands[0].timestamp == 2.0
    assert cands[0].evidence["face_emotion"] == "smile"


def test_emotion_re_transition_fires_again() -> None:
    """neutral → smile → neutral → smile gives two visual candidates."""
    track = _track(
        VisualSignal(ts_offset_s=0.0, face_emotion="neutral"),
        VisualSignal(ts_offset_s=1.0, face_emotion="smile"),
        VisualSignal(ts_offset_s=2.0, face_emotion="neutral"),
        VisualSignal(ts_offset_s=3.0, face_emotion="smile"),
    )
    cands = detect_visual_candidates("default", make_stream(), track, _config())
    assert [c.timestamp for c in cands] == [1.0, 3.0]


def test_neutral_emotion_does_not_fire() -> None:
    track = _track(
        VisualSignal(ts_offset_s=0.0, face_emotion="neutral"),
        VisualSignal(ts_offset_s=1.0, face_emotion="neutral"),
    )
    assert detect_visual_candidates(
        "default", make_stream(), track, _config()
    ) == []


def test_motion_spike_against_rolling_baseline_fires() -> None:
    """5 seconds of low motion (0.05) followed by a 0.3 spike at t=5.
    Baseline is 0.05 → ratio 6 → well over spike_ratio=2.0."""
    track = _track(
        *[VisualSignal(ts_offset_s=float(s), motion_energy=0.05) for s in range(5)],
        VisualSignal(ts_offset_s=5.0, motion_energy=0.3),
    )
    cfg = _config(motion_baseline_window_s=10.0, motion_spike_ratio=2.0)
    cands = detect_visual_candidates("default", make_stream(), track, cfg)
    assert len(cands) == 1
    assert cands[0].timestamp == 5.0
    assert "motion" in cands[0].evidence["sub_scores"]


def test_motion_below_baseline_ratio_does_not_fire() -> None:
    """Modest 1.3x bump shouldn't trip spike_ratio=2.0."""
    track = _track(
        *[VisualSignal(ts_offset_s=float(s), motion_energy=0.05) for s in range(5)],
        VisualSignal(ts_offset_s=5.0, motion_energy=0.065),
    )
    cfg = _config(motion_baseline_window_s=10.0, motion_spike_ratio=2.0)
    assert detect_visual_candidates(
        "default", make_stream(), track, cfg
    ) == []


def test_motion_first_second_no_baseline() -> None:
    """The very first sample has no history; can't fire."""
    track = _track(VisualSignal(ts_offset_s=0.0, motion_energy=0.9))
    assert detect_visual_candidates(
        "default", make_stream(), track, _config()
    ) == []


def test_all_three_signals_at_same_second_compose() -> None:
    """Scene cut + emotion transition + motion spike at the same second
    add up (capped at config.weight)."""
    quiet = [VisualSignal(ts_offset_s=float(s), motion_energy=0.05) for s in range(5)]
    spike = VisualSignal(
        ts_offset_s=5.0,
        scene_cut=True,
        face_emotion="laugh",
        motion_energy=0.4,
    )
    track = _track(*quiet, spike)
    cfg = _config(weight=1.0, motion_baseline_window_s=10.0)
    cands = detect_visual_candidates("default", make_stream(), track, cfg)
    assert len(cands) == 1
    c = cands[0]
    sub = c.evidence["sub_scores"]
    assert "scene_cut" in sub and "emotion" in sub and "motion" in sub
    # All three contribute; outer weight saturates the ratio at 1.0.
    assert c.score == pytest.approx(1.0)


def test_score_saturates_at_outer_weight() -> None:
    """Outer weight=0.6 caps any composite at 0.6 even when all three
    signals fire."""
    quiet = [VisualSignal(ts_offset_s=float(s), motion_energy=0.05) for s in range(5)]
    spike = VisualSignal(
        ts_offset_s=5.0,
        scene_cut=True,
        face_emotion="smile",
        motion_energy=0.5,
    )
    track = _track(*quiet, spike)
    cfg = _config(weight=0.6, motion_baseline_window_s=10.0)
    cands = detect_visual_candidates("default", make_stream(), track, cfg)
    assert cands[0].score == pytest.approx(0.6)


def test_custom_emotion_labels_filter_what_fires() -> None:
    """Restricting emotion_labels to ['shock'] suppresses smile/laugh."""
    track = _track(
        VisualSignal(ts_offset_s=0.0, face_emotion="neutral"),
        VisualSignal(ts_offset_s=1.0, face_emotion="smile"),
        VisualSignal(ts_offset_s=2.0, face_emotion="shock"),
    )
    cfg = _config(emotion_labels=["shock"])
    cands = detect_visual_candidates("default", make_stream(), track, cfg)
    assert [c.timestamp for c in cands] == [2.0]


def test_tenant_mismatch_raises() -> None:
    track = VisualSignalTrack(
        stream_id="str_01TEST",
        tenant_id="other",
        signals=[VisualSignal(ts_offset_s=0.0, scene_cut=True)],
    )
    with pytest.raises(DetectionError, match="tenant mismatch"):
        detect_visual_candidates(
            "default", make_stream(tenant_id="default"), track, _config()
        )


def test_stream_mismatch_raises() -> None:
    track = VisualSignalTrack(
        stream_id="str_OTHER",
        tenant_id="default",
        signals=[VisualSignal(ts_offset_s=0.0, scene_cut=True)],
    )
    with pytest.raises(DetectionError, match="stream/track mismatch"):
        detect_visual_candidates("default", make_stream(), track, _config())
