"""Viral picks carry exact clip edges and stay separate clips."""

from __future__ import annotations

from unittest.mock import AsyncMock

from chalybclip.config import ViralConfig
from chalybclip.detect import ViralMoment, ViralMomentList, detect_viral_moments
from chalybclip.detect.fusion import FusionConfig, fuse_candidates
from chalybclip.detect.models import Candidate
from chalybclip.detect.viral import _dedupe_near_identical, _format_for_llm

from ._fixtures import make_stream, make_transcript


def _moment(ts: float, dur: float, score: float = 0.8) -> ViralMoment:
    return ViralMoment(
        timestamp_s=ts,
        duration_s=dur,
        score=score,
        reason="strong opinion, quotable",
        type="hot_take",
        transcript_snippet="nadie te dice esto",
    )


def test_format_for_llm_lists_sentences_with_word_times() -> None:
    t = make_transcript(
        words=[
            (0.5, 0.9, "Hola", 0.9),
            (1.0, 1.4, "chicas.", 0.9),
            (1.6, 2.0, "Nadie", 0.9),
            (2.1, 2.6, "dice", 0.9),
            (4.0, 4.5, "esto", 0.9),  # 1.4s pause → new line
        ]
    )
    out = _format_for_llm(t, duration_s=10.0).splitlines()
    assert out == [
        "[0.50-1.40] Hola chicas.",
        "[1.60-2.60] Nadie dice",
        "[4.00-4.50] esto",
    ]


def test_format_for_llm_uses_windows_for_long_vods() -> None:
    t = make_transcript(words=[(0.0, 1.0, "uno", 0.9)])
    assert _format_for_llm(t, duration_s=3 * 3600).startswith("[00:00:00]")


def test_dedupe_keeps_hook_and_full_cut_drops_near_identical() -> None:
    full = _moment(10.0, 40.0, score=0.9)
    hook = _moment(12.0, 12.0, score=0.85)  # inside full, IoU 0.3 → kept
    twin = _moment(11.0, 39.0, score=0.7)  # ~same window as full → dropped
    kept = _dedupe_near_identical([full, hook, twin])
    assert kept == [full, hook]


async def test_candidates_carry_exact_edges() -> None:
    router = AsyncMock()
    router.complete = AsyncMock(
        return_value=ViralMomentList(moments=[_moment(20.0, 15.0), _moment(3.0, 9.0)])
    )
    cands = await detect_viral_moments(
        tenant_id="default",
        stream=make_stream(),
        transcript=make_transcript(words=[(0.0, 1.0, "hola", 0.9)]),
        router=router,
        config=ViralConfig(enabled=True, min_score=0.0),
    )
    assert [(c.evidence["start_s"], c.evidence["end_s"]) for c in cands] == [
        (3.0, 12.0),
        (20.0, 35.0),
    ]


def test_fusion_never_merges_two_viral_picks() -> None:
    picks = [
        Candidate(timestamp=t, score=0.8, reason="viral", evidence={"viral_type": "hot_take"})
        for t in (5.0, 25.0, 50.0)
    ]
    audio = Candidate(timestamp=6.0, score=0.5, reason="audio", evidence={})
    fused = fuse_candidates([*picks, audio], config=FusionConfig(cluster_window_s=30.0))
    assert len(fused) == 3
    assert fused[0].evidence["merged_count"] == 2  # viral + nearby audio
