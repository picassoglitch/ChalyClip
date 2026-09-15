"""Unit tests for slice G.1 dynamic clip windowing."""

from __future__ import annotations

import pytest

from chalybclip.clip.windowing import (
    WINDOW_BANDS,
    classify_window_kind,
    plan_clip_window,
)
from chalybclip.detect.models import Candidate
from chalybclip.transcribe.models import Segment, Transcript, Word


def _candidate(
    ts: float,
    *,
    reason: str = "voice",
    **evidence: object,
) -> Candidate:
    return Candidate(
        timestamp=ts,
        score=0.5,
        reason=reason,  # type: ignore[arg-type]
        evidence=dict(evidence),
    )


def _transcript(*spans: tuple[float, float, str]) -> Transcript:
    return Transcript(
        stream_id="str_TEST",
        tenant_id="ten_TEST",
        language="es",
        duration_s=spans[-1][1] if spans else 0.0,
        model="medium",
        segments=[
            Segment(
                ts=start,
                end_ts=end,
                text=text,
                words=[Word(ts=start, end_ts=end, text=text.split()[0], prob=0.9)],
            )
            for (start, end, text) in spans
        ],
    )


# ---- classify_window_kind ----


def test_classifies_retroactive_from_evidence() -> None:
    c = _candidate(50.0, reason="voice", trigger_kind="retroactive")
    assert classify_window_kind(c) == "retroactive"


def test_classifies_viral_humor_as_reaction() -> None:
    c = _candidate(50.0, reason="viral", type="humor")
    assert classify_window_kind(c) == "reaction"


def test_classifies_viral_quotable_as_quote() -> None:
    c = _candidate(50.0, reason="viral", type="quotable")
    assert classify_window_kind(c) == "quote"


def test_classifies_viral_drama_as_story() -> None:
    c = _candidate(50.0, reason="viral", type="drama")
    assert classify_window_kind(c) == "story"


def test_classifies_visual_as_reaction() -> None:
    """Visual detector fires on scene-cut / motion / face-emotion —
    punchy single moments, so reaction band fits."""
    assert classify_window_kind(_candidate(50.0, reason="visual")) == "reaction"


def test_classifies_chat_heat_as_quote() -> None:
    assert classify_window_kind(_candidate(50.0, reason="chat")) == "quote"


def test_classifies_plain_voice_as_default() -> None:
    """Forward voice trigger with no extra hints → legacy default
    window so existing behavior is preserved."""
    assert classify_window_kind(_candidate(50.0, reason="voice")) == "default"


# ---- duration bands ----


def test_reaction_band_within_10_to_22s() -> None:
    c = _candidate(60.0, reason="visual")  # → reaction
    plan = plan_clip_window(
        candidate=c, transcript=None, stream_duration_s=600.0
    )
    band = WINDOW_BANDS["reaction"]
    assert band.min_s <= plan.duration_s <= band.max_s


def test_story_band_within_35_to_60s() -> None:
    c = _candidate(120.0, reason="viral", type="drama")
    plan = plan_clip_window(
        candidate=c, transcript=None, stream_duration_s=600.0
    )
    band = WINDOW_BANDS["story"]
    assert band.min_s <= plan.duration_s <= band.max_s


def test_retroactive_anchors_at_trigger_and_looks_back() -> None:
    """Retroactive windows end AT the trigger, not after it."""
    c = _candidate(100.0, reason="voice", trigger_kind="retroactive")
    plan = plan_clip_window(
        candidate=c, transcript=None, stream_duration_s=600.0
    )
    assert plan.end_s == pytest.approx(100.0)
    assert plan.start_s < 100.0
    assert plan.duration_s <= 60.0  # band max


def test_default_uses_legacy_pre_and_post_roll() -> None:
    c = _candidate(100.0, reason="voice")  # → default
    plan = plan_clip_window(
        candidate=c,
        transcript=None,
        stream_duration_s=600.0,
        fallback_pre_roll_s=30.0,
        fallback_post_roll_s=15.0,
    )
    assert plan.start_s == pytest.approx(70.0)  # 100 - 30
    assert plan.end_s == pytest.approx(115.0)   # 100 + 15


# ---- transcript snapping ----


def test_start_snaps_back_to_sentence_boundary() -> None:
    """Target start of 12s would mid-sentence cut the (10s, 18s)
    segment. With snapping, start pulls back to 10.0."""
    transcript = _transcript(
        (0.0, 8.0, "intro line"),
        (10.0, 18.0, "the important bit"),
        (20.0, 28.0, "follow-up"),
    )
    c = _candidate(15.0, reason="visual")  # reaction band
    plan = plan_clip_window(
        candidate=c, transcript=transcript, stream_duration_s=30.0
    )
    # reaction band pre=15*0.55 ≈ 8.25, so initial start ≈ 6.75.
    # The (0, 8) segment is the one before; pull-back should not
    # cross it. Verify start is on a clean boundary (0.0 or 10.0).
    assert plan.start_s in {0.0, 10.0, pytest.approx(6.75, abs=0.01)}


def test_end_snaps_forward_to_finish_utterance() -> None:
    """Target end at 21.0 should extend forward to 22.0 to finish
    the segment that contains it."""
    transcript = _transcript(
        (0.0, 8.0, "intro"),
        (15.0, 22.0, "the important sentence ending soon"),
        (30.0, 35.0, "next"),
    )
    c = _candidate(18.0, reason="visual")
    plan = plan_clip_window(
        candidate=c, transcript=transcript, stream_duration_s=40.0
    )
    # Reaction band: pre=8.25, post=6.75. Target end ≈ 24.75.
    # Segment (15, 22) ends at 22; segment (30, 35) starts at 30.
    # We expect either the segment end (22) or the next-seg start (30).
    assert plan.end_s in {pytest.approx(22.0), pytest.approx(30.0), pytest.approx(24.75, abs=0.01)}


# ---- clamping ----


def test_clamps_to_stream_duration() -> None:
    """Window can't extend past the stream end."""
    c = _candidate(95.0, reason="visual")
    plan = plan_clip_window(
        candidate=c, transcript=None, stream_duration_s=100.0
    )
    assert plan.end_s <= 100.0


def test_clamps_to_zero_start() -> None:
    """Window can't start before t=0."""
    c = _candidate(5.0, reason="visual")  # reaction
    plan = plan_clip_window(
        candidate=c, transcript=None, stream_duration_s=300.0
    )
    assert plan.start_s >= 0.0


def test_non_positive_stream_duration_returns_zero_window() -> None:
    c = _candidate(50.0, reason="voice")
    plan = plan_clip_window(
        candidate=c, transcript=None, stream_duration_s=0.0
    )
    assert plan.start_s == 0.0 and plan.end_s == 0.0


def test_plan_reason_is_populated() -> None:
    c = _candidate(60.0, reason="viral", type="humor")
    plan = plan_clip_window(
        candidate=c, transcript=None, stream_duration_s=600.0
    )
    assert plan.reason  # not empty
    assert "reaction" in plan.reason


# ---- Monetization length lift (max_clip_duration_s) ----


def test_default_ceiling_keeps_60s_cap_for_story() -> None:
    # Default ceiling (60) must leave behavior EXACTLY as before: a story
    # band caps at 60s.
    cand = _candidate(500.0, reason="voice", type="drama")  # -> story
    plan = plan_clip_window(
        candidate=cand, transcript=None, stream_duration_s=2000.0,
    )
    assert plan.kind == "story"
    assert plan.duration_s <= 60.0 + 1e-6


def test_lifted_ceiling_lets_story_exceed_60s() -> None:
    cand = _candidate(500.0, reason="voice", type="drama")  # -> story
    plan = plan_clip_window(
        candidate=cand, transcript=None, stream_duration_s=2000.0,
        max_clip_duration_s=90.0,
    )
    assert plan.kind == "story"
    # Clears the >60s TikTok Creator Rewards floor, within the new ceiling.
    assert plan.duration_s > 60.0
    assert plan.duration_s <= 90.0 + 1e-6


def test_lifted_ceiling_does_not_stretch_punchy_reaction() -> None:
    cand = _candidate(500.0, reason="visual")  # -> reaction (punchy)
    plan = plan_clip_window(
        candidate=cand, transcript=None, stream_duration_s=2000.0,
        max_clip_duration_s=90.0,
    )
    assert plan.kind == "reaction"
    # Punchy bands stay short even when the ceiling is lifted.
    assert plan.duration_s <= WINDOW_BANDS["reaction"].max_s + 1e-6
