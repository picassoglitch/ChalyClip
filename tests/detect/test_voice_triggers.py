"""Tests for `detect_voice_triggers`."""

from __future__ import annotations

import pytest

from chalybclip.config import DetectionConfig, VoiceDetectorConfig
from chalybclip.detect import detect_voice_triggers
from chalybclip.errors import DetectionError

from ._fixtures import es_only_config, make_stream, make_transcript


def test_detects_exact_phrase_match() -> None:
    transcript = make_transcript(
        words=[
            (0.0, 0.5, " hola", 0.95),
            (0.6, 1.0, " mundo", 0.92),
            (10.0, 11.0, " clipéalo", 0.90),
            (20.0, 20.5, " gracias", 0.88),
        ]
    )
    candidates = detect_voice_triggers(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=es_only_config(["clipéalo"]),
    )
    assert len(candidates) == 1
    c = candidates[0]
    assert c.timestamp == pytest.approx(10.0)
    assert c.reason == "voice"
    assert c.evidence["phrase"] == "clipéalo"
    assert c.evidence["language"] == "es"
    assert c.evidence["distance"] == 0
    assert c.score == pytest.approx(0.90)
    assert "clipéalo" in c.evidence["transcript_snippet"]


def test_detects_multi_word_phrase() -> None:
    transcript = make_transcript(
        words=[
            (5.0, 5.4, " saca", 0.9),
            (5.5, 5.8, " un", 0.85),
            (5.9, 6.4, " clip", 0.95),
        ]
    )
    candidates = detect_voice_triggers(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=es_only_config(["saca un clip"]),
    )
    assert len(candidates) == 1
    assert candidates[0].timestamp == pytest.approx(5.0)
    assert candidates[0].evidence["distance"] == 0


def test_fuzzy_match_within_distance() -> None:
    """`clipealo` (no accent) is distance 1 from `clipéalo` — should match at fd=2."""
    transcript = make_transcript(
        words=[(8.0, 9.0, " clipealo", 0.7)],
    )
    candidates = detect_voice_triggers(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=es_only_config(["clipéalo"], fuzzy_distance=2),
    )
    assert len(candidates) == 1
    assert candidates[0].evidence["distance"] >= 1


def test_fuzzy_match_beyond_distance_rejected() -> None:
    transcript = make_transcript(
        words=[(8.0, 9.0, " completamente_diferente", 0.7)],
    )
    candidates = detect_voice_triggers(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=es_only_config(["clipéalo"], fuzzy_distance=2),
    )
    assert candidates == []


def test_score_is_weight_times_confidence() -> None:
    """score = phrase_weight * mean(word probabilities)"""
    transcript = make_transcript(
        words=[
            (0.0, 0.4, " saca", 0.8),
            (0.5, 0.8, " un", 0.6),
            (0.9, 1.4, " clip", 1.0),
        ]
    )
    candidates = detect_voice_triggers(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=es_only_config(["saca un clip"], weight=0.5),
    )
    assert len(candidates) == 1
    expected = 0.5 * (0.8 + 0.6 + 1.0) / 3
    assert candidates[0].score == pytest.approx(expected)


def test_merges_close_candidates_with_evidence_union() -> None:
    """Two phrase hits within `merge_window_s` collapse to one cluster.

    The per-speaker cooldown (added in slice B.3) would otherwise drop the
    second hit before merging — explicitly disable cooldown here so the
    merge code path gets exercised. The cooldown's own behavior is
    covered by the slice-B.3 tests below.
    """
    transcript = make_transcript(
        words=[
            (10.0, 11.0, " clipéalo", 0.95),
            (15.0, 16.0, " clipéalo", 0.88),
            (60.0, 61.0, " clipéalo", 0.99),
        ]
    )
    cfg = es_only_config(["clipéalo"], merge_window_s=30.0)
    cfg.voice.cooldown_s = 0.0  # disable cooldown for this test
    candidates = detect_voice_triggers(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=cfg,
    )
    # First two merge (both within 30s), third is its own cluster
    assert len(candidates) == 2
    merged = candidates[0]
    assert merged.evidence["merged_count"] == 2
    assert len(merged.evidence["matches"]) == 2
    assert merged.score == pytest.approx(0.95)
    assert candidates[1].timestamp == pytest.approx(60.0)


def test_disabled_voice_detector_returns_empty() -> None:
    transcript = make_transcript(words=[(0.0, 1.0, " clipéalo", 1.0)])
    config = DetectionConfig(
        voice=VoiceDetectorConfig(enabled=False, phrases={"es": ["clipéalo"]}),
    )
    assert detect_voice_triggers(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=config,
    ) == []


def test_empty_transcript_returns_empty() -> None:
    candidates = detect_voice_triggers(
        tenant_id="default",
        stream=make_stream(),
        transcript=make_transcript(words=[]),
        config=es_only_config(),
    )
    assert candidates == []


def test_tenant_mismatch_raises() -> None:
    with pytest.raises(DetectionError, match="tenant mismatch"):
        detect_voice_triggers(
            tenant_id="bob",
            stream=make_stream(tenant_id="alice"),
            transcript=make_transcript(tenant_id="alice"),
            config=es_only_config(),
        )


def test_stream_transcript_mismatch_raises() -> None:
    with pytest.raises(DetectionError, match="stream/transcript mismatch"):
        detect_voice_triggers(
            tenant_id="default",
            stream=make_stream(stream_id="str_A"),
            transcript=make_transcript(stream_id="str_B"),
            config=es_only_config(),
        )


def test_punctuation_does_not_break_match() -> None:
    transcript = make_transcript(
        words=[(2.0, 3.0, " ¡clipéalo!", 0.9)],
    )
    candidates = detect_voice_triggers(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=es_only_config(["clipéalo"], fuzzy_distance=0),
    )
    assert len(candidates) == 1


# ---- retroactive trigger family ('clipeaste eso' → backward clip) ----


def test_retroactive_phrase_tags_evidence() -> None:
    """A retroactive phrase emits a candidate with trigger_kind=retroactive
    and the lookback duration in evidence, so the cut step knows to
    extend backward rather than around the timestamp."""
    transcript = make_transcript(
        words=[
            (10.0, 10.3, " no", 0.9),
            (10.4, 10.8, " manches", 0.9),
            (10.9, 11.3, " clipeaste", 0.92),
            (11.4, 11.7, " eso", 0.91),
        ]
    )
    candidates = detect_voice_triggers(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=es_only_config(
            [],  # no forward phrases
            retroactive_phrases=["clipeaste eso"],
            retroactive_lookback_s=60.0,
        ),
    )
    assert len(candidates) == 1
    c = candidates[0]
    assert c.evidence["trigger_kind"] == "retroactive"
    assert c.evidence["retroactive_lookback_s"] == 60.0
    assert c.evidence["phrase"] == "clipeaste eso"


def test_forward_and_retroactive_coexist() -> None:
    """Same VOD with both phrase families fires distinct candidates."""
    transcript = make_transcript(
        words=[
            (5.0, 5.5, " clipéalo", 0.9),   # forward
            (60.0, 60.4, " clipeaste", 0.9),  # retroactive
            (60.5, 60.8, " eso", 0.9),
        ]
    )
    candidates = detect_voice_triggers(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=es_only_config(
            ["clipéalo"],
            retroactive_phrases=["clipeaste eso"],
        ),
    )
    kinds = sorted(c.evidence["trigger_kind"] for c in candidates)
    assert kinds == ["forward", "retroactive"]


def test_forward_phrase_carries_forward_kind() -> None:
    """Backward-compatible: existing forward triggers still emit
    trigger_kind=forward (defaults exercised, no retroactive list)."""
    transcript = make_transcript(
        words=[
            (10.0, 11.0, " clipéalo", 0.92),
        ]
    )
    candidates = detect_voice_triggers(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=es_only_config(["clipéalo"]),
    )
    assert candidates[0].evidence["trigger_kind"] == "forward"


# ---- per-speaker trigger attribution + cooldown (slice B.3) ----


from chalybclip.diarize.models import (  # noqa: E402
    Diarization,
    DiarizationSegment,
)


def _diar_with_two_speakers() -> Diarization:
    """Speaker A holds the floor 0-30s; Speaker B holds 30-60s."""
    return Diarization(
        stream_id="str_01TEST",
        tenant_id="default",
        segments=[
            DiarizationSegment(ts=0.0, end_ts=30.0, speaker_label="SPEAKER_00"),
            DiarizationSegment(ts=30.0, end_ts=60.0, speaker_label="SPEAKER_01"),
        ],
    )


def test_candidate_carries_speaker_label_from_diarization() -> None:
    """A trigger inside SPEAKER_00's turn gets the SPEAKER_00 label."""
    transcript = make_transcript(
        words=[
            (10.0, 10.4, " no", 0.9),
            (10.5, 11.0, " manches", 0.9),
            (11.1, 12.0, " clipéalo", 0.93),
        ]
    )
    candidates = detect_voice_triggers(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=es_only_config(["clipéalo"]),
        diarization=_diar_with_two_speakers(),
    )
    assert len(candidates) == 1
    assert candidates[0].evidence["speaker_label"] == "SPEAKER_00"


def test_two_speakers_firing_same_phrase_both_emit_in_cooldown_window() -> None:
    """When two different speakers say the same trigger within the global
    cooldown window, both candidates survive — cooldown is per-speaker."""
    transcript = make_transcript(
        words=[
            # Speaker A @ 15s
            (15.0, 16.0, " clipéalo", 0.92),
            # Speaker B @ 35s — within 30s cooldown but a different speaker
            (35.0, 36.0, " clipéalo", 0.90),
        ]
    )
    cfg = es_only_config(["clipéalo"], merge_window_s=0.0)
    cfg.voice.cooldown_s = 30.0
    candidates = detect_voice_triggers(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=cfg,
        diarization=_diar_with_two_speakers(),
    )
    assert len(candidates) == 2
    labels = sorted(c.evidence["speaker_label"] for c in candidates)
    assert labels == ["SPEAKER_00", "SPEAKER_01"]


def test_same_speaker_repeated_phrase_within_cooldown_drops_second() -> None:
    """One speaker firing the same trigger twice within the cooldown only
    emits the first hit."""
    transcript = make_transcript(
        words=[
            (5.0, 6.0, " clipéalo", 0.92),
            (10.0, 11.0, " clipéalo", 0.92),  # 5s later, within 30s cooldown
        ]
    )
    cfg = es_only_config(["clipéalo"], merge_window_s=0.0)
    cfg.voice.cooldown_s = 30.0
    candidates = detect_voice_triggers(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=cfg,
        diarization=_diar_with_two_speakers(),
    )
    assert len(candidates) == 1
    assert candidates[0].timestamp == pytest.approx(5.0)


def test_no_diarization_applies_global_cooldown() -> None:
    """Without diarization (None or skipped=True), all candidates share
    one cooldown bucket — equivalent to the spec's pre-diarization
    behavior."""
    transcript = make_transcript(
        words=[
            (5.0, 6.0, " clipéalo", 0.92),
            (10.0, 11.0, " clipéalo", 0.92),
        ]
    )
    cfg = es_only_config(["clipéalo"], merge_window_s=0.0)
    cfg.voice.cooldown_s = 30.0
    # Skipped diarization: same effect as None.
    skipped = Diarization(stream_id="str_01TEST", tenant_id="default", skipped=True)
    candidates = detect_voice_triggers(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=cfg,
        diarization=skipped,
    )
    assert len(candidates) == 1


# ---- per-kit custom_trigger_phrases (slice C.2) ----


from chalybclip.db.models import CustomTriggerPhrases  # noqa: E402


def test_kit_phrase_fires_only_for_assigned_speaker() -> None:
    """A kit-added phrase for SPEAKER_00 must NOT fire on SPEAKER_01's
    words, even when both speakers say the exact phrase."""
    transcript = make_transcript(
        words=[
            # Speaker A @ 15s says 'monchi eso' (Aldo's kit phrase).
            (15.0, 15.4, " monchi", 0.9),
            (15.5, 16.0, " eso", 0.9),
            # Speaker B @ 35s says 'monchi eso' too — should NOT fire,
            # because B's kit doesn't include this phrase.
            (35.0, 35.4, " monchi", 0.9),
            (35.5, 36.0, " eso", 0.9),
        ]
    )
    candidates = detect_voice_triggers(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=es_only_config([]),  # no base forward phrases
        diarization=_diar_with_two_speakers(),
        extra_phrases_by_speaker={
            "SPEAKER_00": CustomTriggerPhrases(forward=["monchi eso"]),
        },
    )
    assert len(candidates) == 1
    assert candidates[0].evidence["speaker_label"] == "SPEAKER_00"
    assert candidates[0].evidence["phrase"] == "monchi eso"


def test_kit_phrase_layers_on_top_of_base_phrases() -> None:
    """Base phrases still fire for everyone; kit phrases ADDITIVELY extend
    that for the assigned speaker. merge_window_s=0 here so the
    same-speaker merge doesn't collapse the two distinct candidates."""
    transcript = make_transcript(
        words=[
            (5.0, 5.5, " clipéalo", 0.92),       # base phrase, Speaker A
            (35.0, 35.4, " monchi", 0.9),         # kit phrase, Speaker B
            (35.5, 36.0, " eso", 0.9),
        ]
    )
    candidates = detect_voice_triggers(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=es_only_config(["clipéalo"], merge_window_s=0.0),
        diarization=_diar_with_two_speakers(),
        extra_phrases_by_speaker={
            "SPEAKER_01": CustomTriggerPhrases(forward=["monchi eso"]),
        },
    )
    assert len(candidates) == 2
    phrases = sorted(c.evidence["phrase"] for c in candidates)
    assert phrases == ["clipéalo", "monchi eso"]
    labels = sorted(c.evidence["speaker_label"] for c in candidates)
    assert labels == ["SPEAKER_00", "SPEAKER_01"]


def test_kit_retroactive_phrase_carries_retroactive_kind() -> None:
    """Per-kit retroactive phrases get the retroactive trigger_kind so
    the cut step uses the backward window."""
    transcript = make_transcript(
        words=[
            (15.0, 15.4, " córtalo", 0.9),
        ]
    )
    candidates = detect_voice_triggers(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=es_only_config([], retroactive_lookback_s=45.0),
        diarization=_diar_with_two_speakers(),
        extra_phrases_by_speaker={
            "SPEAKER_00": CustomTriggerPhrases(retroactive=["córtalo"]),
        },
    )
    assert len(candidates) == 1
    assert candidates[0].evidence["trigger_kind"] == "retroactive"
    assert candidates[0].evidence["retroactive_lookback_s"] == 45.0
