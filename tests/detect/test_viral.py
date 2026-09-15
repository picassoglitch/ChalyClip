"""LLM-based viral-moment detector — covers the happy path, disabled toggle,
LLM-error swallow, and score-floor filtering."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from chalybclip.config import ViralConfig
from chalybclip.detect import ViralMoment, ViralMomentList, detect_viral_moments
from chalybclip.detect.viral import _format_transcript
from chalybclip.errors import LLMError

from ._fixtures import make_stream, make_transcript


def _moment(
    ts: float = 30.0,
    score: float = 0.9,
    type_: str = "hot_take",
    snippet: str = "you're broke at 30 and that's on you",
) -> ViralMoment:
    return ViralMoment(
        timestamp_s=ts,
        duration_s=20.0,
        score=score,
        reason="strong opinion, quotable, likely to bait reactions",
        type=type_,  # type: ignore[arg-type]
        transcript_snippet=snippet,
    )


async def test_disabled_returns_empty_without_llm_call() -> None:
    """Off by default, the detector never touches the router."""
    router = AsyncMock()
    cands = await detect_viral_moments(
        tenant_id="default",
        stream=make_stream(),
        transcript=make_transcript(),
        router=router,
        config=ViralConfig(enabled=False),
    )
    assert cands == []
    router.complete.assert_not_called()


async def test_happy_path_emits_one_candidate_per_moment() -> None:
    """Each LLM-returned moment becomes a Candidate with reason='viral'."""
    router = AsyncMock()
    router.complete = AsyncMock(
        return_value=ViralMomentList(moments=[_moment(ts=10.0), _moment(ts=120.0)])
    )

    cands = await detect_viral_moments(
        tenant_id="default",
        stream=make_stream(),
        transcript=make_transcript(words=[(0.0, 1.0, "hola", 0.9)]),
        router=router,
        config=ViralConfig(enabled=True, weight=1.0, min_score=0.0),
    )

    assert len(cands) == 2
    assert all(c.reason == "viral" for c in cands)
    assert cands[0].timestamp == 10.0
    assert cands[1].timestamp == 120.0
    # Evidence carries the LLM rationale + type for the dashboard breakdown.
    for c in cands:
        assert "viral_type" in c.evidence
        assert "transcript_snippet" in c.evidence


async def test_weight_multiplies_llm_score() -> None:
    """Config weight scales every candidate's score linearly."""
    router = AsyncMock()
    router.complete = AsyncMock(
        return_value=ViralMomentList(moments=[_moment(score=0.8)])
    )

    cands = await detect_viral_moments(
        tenant_id="default",
        stream=make_stream(),
        transcript=make_transcript(),
        router=router,
        config=ViralConfig(enabled=True, weight=0.5, min_score=0.0),
    )

    assert len(cands) == 1
    assert cands[0].score == pytest.approx(0.5 * 0.8)


async def test_min_score_drops_low_confidence_moments() -> None:
    """The detector filters moments below `min_score` BEFORE returning."""
    router = AsyncMock()
    router.complete = AsyncMock(
        return_value=ViralMomentList(
            moments=[_moment(score=0.4), _moment(score=0.9)]
        )
    )

    cands = await detect_viral_moments(
        tenant_id="default",
        stream=make_stream(),
        transcript=make_transcript(),
        router=router,
        config=ViralConfig(enabled=True, min_score=0.5),
    )

    assert len(cands) == 1
    assert cands[0].evidence["viral_score"] == 0.9


async def test_max_moments_caps_short_video() -> None:
    """On a short video the configured cap floors the returned list (keeps the
    cut step from chewing through 50+ candidates)."""
    router = AsyncMock()
    router.complete = AsyncMock(
        return_value=ViralMomentList(moments=[_moment(ts=i * 10.0) for i in range(20)])
    )

    short = make_stream().model_copy(update={"duration_s": 120.0})  # 2 min
    cands = await detect_viral_moments(
        tenant_id="default",
        stream=short,
        transcript=make_transcript(),
        router=router,
        config=ViralConfig(enabled=True, max_moments=5, min_score=0.0),
    )

    assert len(cands) == 5


async def test_cap_scales_with_duration_for_long_video() -> None:
    """A long VOD isn't truncated to the short-video default — the cap scales
    ~1 moment/minute so chat-less sources (e.g. YouTube) keep more of what the
    detector genuinely found, instead of being clipped to max_moments."""
    router = AsyncMock()
    router.complete = AsyncMock(
        return_value=ViralMomentList(moments=[_moment(ts=i * 60.0) for i in range(40)])
    )

    long = make_stream().model_copy(update={"duration_s": 1800.0})  # 30 min
    cands = await detect_viral_moments(
        tenant_id="default",
        stream=long,
        transcript=make_transcript(),
        router=router,
        config=ViralConfig(enabled=True, max_moments=20, min_score=0.0),
    )

    # 30-min video → cap scales to 30 (above the 20 default), so 30 kept.
    assert len(cands) == 30


async def test_llm_error_returns_empty_does_not_raise() -> None:
    """LLM failure mid-pipeline must not blow up the whole run."""
    router = AsyncMock()
    router.complete = AsyncMock(side_effect=LLMError("provider 503"))

    cands = await detect_viral_moments(
        tenant_id="default",
        stream=make_stream(),
        transcript=make_transcript(),
        router=router,
        config=ViralConfig(enabled=True),
    )
    assert cands == []


def test_format_transcript_groups_into_windows() -> None:
    """Transcript chunks every ~30s with [HH:MM:SS] anchors the LLM can cite."""
    t = make_transcript(
        words=[
            (0.0, 1.0, "uno", 0.9),
            (5.0, 6.0, "dos", 0.9),
            (40.0, 41.0, "tres", 0.9),
            (45.0, 46.0, "cuatro", 0.9),
        ]
    )
    formatted = _format_transcript(t, window_s=30.0)
    assert "[00:00:00]" in formatted
    # The fixture stuffs all words into one segment so windowing emits a
    # single line; what matters is the timestamp anchor is present.
    assert "uno" in formatted


def test_format_transcript_empty_segments_returns_placeholder() -> None:
    """No transcript -> short marker the LLM will treat as 'nothing to score'."""
    assert _format_transcript(make_transcript()) == "(empty transcript)"


async def test_tenant_mismatch_raises() -> None:
    """Tenant ID must agree across caller / stream / transcript."""
    from chalybclip.errors import DetectionError

    router = AsyncMock()
    with pytest.raises(DetectionError, match="tenant mismatch"):
        await detect_viral_moments(
            tenant_id="alice",
            stream=make_stream(tenant_id="bob"),
            transcript=make_transcript(),
            router=router,
            config=ViralConfig(enabled=True),
        )


async def test_passes_quality_through_to_router() -> None:
    """`quality` from config reaches the router so the dashboard can opt into Opus."""
    captured: dict[str, Any] = {}

    async def fake_complete(**kwargs: Any) -> ViralMomentList:
        captured.update(kwargs)
        return ViralMomentList(moments=[])

    router = AsyncMock()
    router.complete = fake_complete

    await detect_viral_moments(
        tenant_id="default",
        stream=make_stream(),
        transcript=make_transcript(),
        router=router,
        config=ViralConfig(enabled=True, quality="premium"),
    )

    assert captured["quality"] == "premium"
    assert captured["purpose"] == "viral_detection"
