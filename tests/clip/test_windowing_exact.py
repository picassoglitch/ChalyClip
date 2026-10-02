"""Viral picks are cut on their exact edges, snapped to word boundaries."""

from __future__ import annotations

from chalybclip.clip.windowing import plan_clip_window
from chalybclip.detect.models import Candidate
from chalybclip.transcribe.models import Segment, Transcript, Word


def _transcript(words: list[tuple[float, float, str]]) -> Transcript:
    ws = [Word(ts=a, end_ts=b, text=t, prob=0.9) for a, b, t in words]
    return Transcript(
        stream_id="str_TEST",
        tenant_id="ten_TEST",
        language="es",
        duration_s=ws[-1].end_ts,
        model="assemblyai",
        segments=[Segment(ts=ws[0].ts, end_ts=ws[-1].end_ts, text="x", words=ws)],
    )


# 0-2s warm-up, hook at 3.0, payoff ends 10.8, then a pause and the phone tap.
WORDS = [
    (0.2, 0.6, "okay"),
    (0.8, 1.2, "espera."),
    (3.0, 3.4, "Nadie"),
    (3.5, 3.9, "te"),
    (4.0, 4.6, "dice"),
    (5.0, 6.0, "esto"),
    (6.2, 8.0, "pero"),
    (8.2, 10.8, "funciona."),
    (14.0, 14.4, "listo"),
]


def _viral(start: float, end: float, **extra: object) -> Candidate:
    return Candidate(
        timestamp=start,
        score=0.8,
        reason="viral",
        evidence={"viral_type": "hot_take", "start_s": start, "end_s": end, **extra},
    )


def test_cuts_on_hook_and_payoff_words() -> None:
    plan = plan_clip_window(
        candidate=_viral(2.9, 11.0),
        transcript=_transcript(WORDS),
        stream_duration_s=15.0,
    )
    assert plan.start_s == 3.0 - 0.12
    assert plan.end_s == 10.8 + 0.35
    assert "exact edges" in plan.reason


def test_padding_never_reaches_neighbouring_words() -> None:
    words = [(0.0, 2.95, "antes"), *WORDS[2:8], (10.9, 11.5, "después")]
    plan = plan_clip_window(
        candidate=_viral(3.0, 10.8),
        transcript=_transcript(words),
        stream_duration_s=15.0,
    )
    assert plan.start_s == 2.95
    assert plan.end_s == 10.9


def test_too_short_pick_runs_to_sentence_end() -> None:
    plan = plan_clip_window(
        candidate=_viral(3.0, 4.6),
        transcript=_transcript(WORDS),
        stream_duration_s=15.0,
    )
    assert plan.end_s == 10.8 + 0.35  # "funciona." closes the sentence


def test_edges_found_in_fusion_matches() -> None:
    audio_anchor = Candidate(
        timestamp=4.0,
        score=0.9,
        reason="audio",
        evidence={
            "matches": [
                {"viral_type": "hot_take", "start_s": 3.0, "end_s": 10.8},
                {},
            ]
        },
    )
    plan = plan_clip_window(
        candidate=audio_anchor, transcript=_transcript(WORDS), stream_duration_s=15.0
    )
    assert plan.start_s == 3.0 - 0.12


def test_no_edges_keeps_band_logic() -> None:
    plain = Candidate(timestamp=5.0, score=0.8, reason="viral", evidence={"viral_type": "humor"})
    plan = plan_clip_window(
        candidate=plain, transcript=_transcript(WORDS), stream_duration_s=15.0
    )
    assert "exact edges" not in plan.reason
