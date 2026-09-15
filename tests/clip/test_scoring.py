"""AI score-card derivation from `ClipBreakdown`."""

from __future__ import annotations

from chalybclip.clip import ClipBreakdown, compute_ai_scores


def _bd(
    *,
    heuristic_score: float = 0.5,
    heuristic_reason: str = "voice",
    motion_score: float | None = None,
    face_presence: float | None = None,
    speaking_intensity: float | None = None,
    reaction_confidence: float | None = None,
    rescore_delta: float | None = None,
    rescore_reason: str | None = None,
    duration_s: float = 30.0,
) -> ClipBreakdown:
    return ClipBreakdown(
        clip_id="clp_x",
        duration_s=duration_s,
        motion_score=motion_score,
        face_presence=face_presence,
        speaking_intensity=speaking_intensity,
        reaction_confidence=reaction_confidence,
        rescore_delta=rescore_delta,
        rescore_reason=rescore_reason,
        heuristic_reason=heuristic_reason,
        heuristic_score=heuristic_score,
    )


# ---- viral_score ----


def test_viral_score_uses_rescore_when_present() -> None:
    """A high vision rescore dominates a low heuristic — that's the
    explicit weighting (rescore = 70%, heuristic = 30%)."""
    s = compute_ai_scores(_bd(heuristic_score=0.2, reaction_confidence=0.9))
    # 0.9*70 + 0.2*30 = 69
    assert s.viral_score == 69
    assert "vision-LLM rescore" in s.viral_why


def test_viral_score_falls_back_to_heuristic_when_no_rescore() -> None:
    s = compute_ai_scores(_bd(heuristic_score=0.7))
    # 0.7 * 90 = 63
    assert s.viral_score == 63
    assert "not yet vision-rescored" in s.viral_why


def test_viral_score_face_presence_boosts_up_to_10() -> None:
    """Face in frame >40% of clip seconds adds up to +10."""
    no_face = compute_ai_scores(_bd(heuristic_score=0.5))
    with_face = compute_ai_scores(_bd(heuristic_score=0.5, face_presence=1.0))
    # 0.5*90 = 45; with face 1.0: +min(10, (1.0-0.4)*20) = +10 → 55
    assert no_face.viral_score == 45
    assert with_face.viral_score == 55
    assert "face-presence boost" in with_face.viral_why


def test_viral_score_face_presence_below_threshold_no_boost() -> None:
    s = compute_ai_scores(_bd(heuristic_score=0.5, face_presence=0.3))
    assert s.viral_score == 45  # below 0.4 → no boost


def test_viral_score_clamped_to_0_100() -> None:
    """Even a perfect rescore + max face boost stays ≤ 100."""
    s = compute_ai_scores(_bd(
        heuristic_score=1.0,
        reaction_confidence=1.0,
        face_presence=1.0,
    ))
    assert s.viral_score == 100


# ---- hook_strength ----


def test_hook_strength_high_at_75_plus() -> None:
    s = compute_ai_scores(_bd(heuristic_score=1.0, reaction_confidence=1.0))
    assert s.hook_strength == "HIGH"
    assert "push this clip first" in s.hook_why


def test_hook_strength_medium_in_band() -> None:
    s = compute_ai_scores(_bd(heuristic_score=0.7))  # → 63
    assert s.hook_strength == "MEDIUM"


def test_hook_strength_developing_below_55() -> None:
    s = compute_ai_scores(_bd(heuristic_score=0.3))  # → 27
    assert s.hook_strength == "DEVELOPING"
    assert "needs a strong title" in s.hook_why


# ---- caption_readability ----


def test_caption_readability_good_in_sweet_spot() -> None:
    for wps in (0.8, 1.5, 2.2):
        s = compute_ai_scores(_bd(speaking_intensity=wps))
        assert s.caption_readability == "GOOD", f"wps={wps}"


def test_caption_readability_ok_at_edges() -> None:
    for wps in (0.6, 2.5):
        s = compute_ai_scores(_bd(speaking_intensity=wps))
        assert s.caption_readability == "OK", f"wps={wps}"


def test_caption_readability_check_outside_band() -> None:
    for wps in (0.2, 4.0):
        s = compute_ai_scores(_bd(speaking_intensity=wps))
        assert s.caption_readability == "CHECK", f"wps={wps}"


def test_caption_readability_check_when_no_transcript() -> None:
    s = compute_ai_scores(_bd(speaking_intensity=None))
    assert s.caption_readability == "CHECK"
    assert "no transcript" in s.readability_why


# ---- dead_air_risk ----


def test_dead_air_low_when_talking_through() -> None:
    s = compute_ai_scores(_bd(speaking_intensity=1.5))
    assert s.dead_air_risk == "LOW"


def test_dead_air_med_when_quiet_stretches() -> None:
    s = compute_ai_scores(_bd(speaking_intensity=0.5))
    assert s.dead_air_risk == "MED"


def test_dead_air_high_when_long_quiet() -> None:
    s = compute_ai_scores(_bd(speaking_intensity=0.2))
    assert s.dead_air_risk == "HIGH"
    assert "trim or skip" in s.dead_air_why


def test_dead_air_med_when_no_transcript() -> None:
    s = compute_ai_scores(_bd(speaking_intensity=None))
    assert s.dead_air_risk == "MED"


# ---- end-to-end smoke ----


def test_compute_ai_scores_returns_all_fields_with_explanations() -> None:
    """Every score has a paired *_why string — the UI surfaces these
    on hover so the operator understands why the score is what it is."""
    s = compute_ai_scores(_bd(
        heuristic_score=0.6,
        speaking_intensity=1.2,
        face_presence=0.8,
        reaction_confidence=0.75,
    ))
    assert isinstance(s.viral_score, int)
    assert s.viral_why
    assert s.hook_why
    assert s.readability_why
    assert s.dead_air_why
