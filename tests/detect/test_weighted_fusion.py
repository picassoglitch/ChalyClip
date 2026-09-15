"""Unit tests for slice G.1 weighted multi-signal fusion."""

from __future__ import annotations

import pytest

from chalybclip.detect.fusion import (
    FusionBonuses,
    FusionConfig,
    FusionWeights,
    fuse_candidates,
)
from chalybclip.detect.models import Candidate


def _c(
    ts: float,
    score: float,
    reason: str,
    **evidence: object,
) -> Candidate:
    return Candidate(
        timestamp=ts,
        score=score,
        reason=reason,  # type: ignore[arg-type]
        evidence=dict(evidence),
    )


# ---- weighted-sum formula ----


def test_single_voice_candidate_uses_voice_weight() -> None:
    """A lone voice candidate at score 1.0 should fuse to 0.35
    (the voice weight). No bonuses fire."""
    fused = fuse_candidates([_c(ts=10.0, score=1.0, reason="voice")])
    assert len(fused) == 1
    assert fused[0].score == pytest.approx(0.35)


def test_five_detectors_at_max_score_clamp_at_one() -> None:
    """All 5 detectors at 1.0: weighted = 0.95 + bonuses → clamp to 1.0."""
    fused = fuse_candidates([
        _c(ts=10.0, score=1.0, reason="voice"),
        _c(ts=10.5, score=1.0, reason="visual"),
        _c(ts=11.0, score=1.0, reason="audio"),
        _c(ts=11.5, score=1.0, reason="chat"),
        _c(ts=12.0, score=1.0, reason="viral"),
    ])
    assert len(fused) == 1
    assert fused[0].score == pytest.approx(1.0)


def test_max_score_no_longer_wins_alone() -> None:
    """One voice at 0.9 vs three detectors at 0.5 each within 10s:
    Three-detector consensus should win — this is the WHOLE POINT
    of slice G.1."""
    voice_only = fuse_candidates([_c(ts=100.0, score=0.9, reason="voice")])
    consensus = fuse_candidates([
        _c(ts=200.0, score=0.5, reason="visual"),
        _c(ts=201.0, score=0.5, reason="chat"),
        _c(ts=202.0, score=0.5, reason="viral"),
    ])
    assert voice_only[0].score < consensus[0].score


def test_repeated_same_detector_does_not_double_count() -> None:
    """Two audio candidates in the same cluster use the HIGHER score —
    a chatty detector firing in burst is one signal, not two."""
    fused = fuse_candidates([
        _c(ts=10.0, score=0.5, reason="audio"),
        _c(ts=11.0, score=0.9, reason="audio"),
    ])
    # weighted = 0.15 × 0.9 = 0.135. Same detector → no overlap bonus.
    assert fused[0].score == pytest.approx(0.135)


# ---- overlap bonus ----


def test_two_detector_overlap_bonus_fires_at_2_distinct() -> None:
    fused = fuse_candidates([
        _c(ts=10.0, score=0.5, reason="voice"),
        _c(ts=15.0, score=0.5, reason="visual"),
    ])
    # weighted = 0.35×0.5 + 0.20×0.5 = 0.275, + 0.05 bonus = 0.325
    assert fused[0].score == pytest.approx(0.325)


def test_three_detector_overlap_bonus_jumps_to_010() -> None:
    fused = fuse_candidates([
        _c(ts=10.0, score=0.4, reason="voice"),
        _c(ts=12.0, score=0.4, reason="audio"),
        _c(ts=14.0, score=0.4, reason="viral"),
    ])
    # weighted = 0.35*0.4 + 0.15*0.4 + 0.10*0.4 = 0.240
    # bonus = 0.10 (3 distinct detectors within 10s of anchor)
    assert fused[0].score == pytest.approx(0.340)


def test_no_overlap_bonus_when_detectors_outside_window() -> None:
    """Same cluster (≤30s) but outside the 10s overlap window → no bonus."""
    cfg = FusionConfig(cluster_window_s=30.0, overlap_window_s=10.0)
    fused = fuse_candidates(
        [
            _c(ts=0.0, score=0.5, reason="voice"),
            _c(ts=25.0, score=0.5, reason="visual"),
        ],
        config=cfg,
    )
    # weighted = 0.35*0.5 + 0.20*0.5 = 0.275. No bonus.
    # Note: the anchor pick may put the anchor next to ONE of them,
    # so the OTHER might be outside its 10s window. Either way no
    # bonus fires because only 1 distinct detector is local.
    assert fused[0].score == pytest.approx(0.275)


# ---- face / strong-signal bonuses ----


def test_face_present_adds_005_bonus() -> None:
    fused = fuse_candidates([
        _c(ts=10.0, score=0.4, reason="visual", face_emotion="smile"),
    ])
    # weighted = 0.20 * 0.4 = 0.08, +0.05 face, +0.05 strong (smile) = 0.18
    assert fused[0].score == pytest.approx(0.18)


def test_motion_spike_adds_strong_signal_bonus() -> None:
    fused = fuse_candidates([
        _c(ts=10.0, score=0.5, reason="visual", kind="motion_spike"),
    ])
    # weighted = 0.20 * 0.5 = 0.10, +0.05 strong = 0.15
    assert fused[0].score == pytest.approx(0.15)


# ---- anchor selection ----


def test_anchor_picks_timestamp_with_most_corroborating_detectors() -> None:
    """Cluster spanning 25s:
      - t=0: voice (no corroboration nearby)
      - t=15..18: visual+audio+viral all firing within 5s of each other
    New behavior: anchor at t=15ish where 3 detectors agree, not t=0."""
    candidates = [
        _c(ts=0.0, score=0.9, reason="voice"),
        _c(ts=15.0, score=0.4, reason="visual"),
        _c(ts=17.0, score=0.4, reason="audio"),
        _c(ts=18.0, score=0.4, reason="viral"),
    ]
    fused = fuse_candidates(candidates)
    assert len(fused) == 1
    assert fused[0].timestamp >= 15.0


def test_single_candidate_cluster_keeps_its_timestamp() -> None:
    fused = fuse_candidates([_c(ts=42.0, score=0.7, reason="audio")])
    assert fused[0].timestamp == 42.0


# ---- evidence preservation ----


def test_evidence_carries_fusion_breakdown_and_matches() -> None:
    """Output must preserve all input evidence under `matches` AND
    record the fusion math for explainability."""
    fused = fuse_candidates([
        _c(ts=10.0, score=0.6, reason="voice", phrase="clipea esto"),
        _c(ts=12.0, score=0.7, reason="audio", rms_ratio=3.2),
    ])
    ev = fused[0].evidence
    assert "matches" in ev
    assert len(ev["matches"]) == 2
    assert "fusion" in ev
    fusion = ev["fusion"]
    assert "weighted_sum" in fusion
    assert "overlap_bonus" in fusion
    assert "per_detector_best" in fusion
    assert fusion["cluster_size"] == 2


# ---- clustering behavior ----


def test_distant_candidates_stay_separate() -> None:
    """Candidates >30s apart should NOT cluster — each fuses alone."""
    fused = fuse_candidates([
        _c(ts=10.0, score=0.5, reason="voice"),
        _c(ts=100.0, score=0.5, reason="voice"),
    ])
    assert len(fused) == 2
    assert fused[0].score == fused[1].score
    assert fused[0].score == pytest.approx(0.35 * 0.5)


def test_empty_input_returns_empty() -> None:
    assert fuse_candidates([]) == []


def test_zero_cluster_window_passes_through_sorted() -> None:
    cfg = FusionConfig(cluster_window_s=0.0)
    fused = fuse_candidates(
        [
            _c(ts=10.0, score=0.5, reason="voice"),
            _c(ts=5.0, score=0.5, reason="audio"),
        ],
        config=cfg,
    )
    assert [c.timestamp for c in fused] == [5.0, 10.0]


# ---- config tunability ----


def test_custom_weights_change_outcome() -> None:
    cfg = FusionConfig(
        weights=FusionWeights(
            voice=0.0,
            visual=1.0,
            audio=0.0,
            chat=0.0,
            viral=0.0,
            transcript_hook=0.0,
        ),
        bonuses=FusionBonuses(
            two_detectors=0.0,
            three_plus_detectors=0.0,
            face_visible=0.0,
            strong_signal=0.0,
        ),
    )
    fused = fuse_candidates(
        [_c(ts=10.0, score=0.7, reason="visual")],
        config=cfg,
    )
    # weighted = 1.0 * 0.7 = 0.7. No bonuses configured.
    assert fused[0].score == pytest.approx(0.7)
