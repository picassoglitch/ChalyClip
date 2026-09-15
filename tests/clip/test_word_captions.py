"""Word-level caption builder (slice F.7-F — creator-weapon upgrade)."""

from __future__ import annotations

import json

from chalybclip.clip.word_captions import (
    LINE_BREAK_PAUSE_S,
    MAX_LINE_DURATION_S,
    MAX_WORDS_PER_LINE,
    captions_for_clip,
    chunk_words_to_lines,
    classify_emphasis,
    lines_to_json,
)

# ---- classify_emphasis ----


def test_classify_normal_word() -> None:
    text, emp = classify_emphasis("hello")
    assert text == "hello"
    assert emp == "normal"


def test_classify_all_caps_is_shout() -> None:
    text, emp = classify_emphasis("EPIC")
    assert text == "EPIC"
    assert emp == "shout"


def test_classify_terminal_bang_is_emphasis() -> None:
    text, emp = classify_emphasis("now!")
    assert text == "now!"
    assert emp == "emphasis"


def test_classify_question_terminal_is_emphasis() -> None:
    _, emp = classify_emphasis("what?")
    assert emp == "emphasis"


def test_classify_spanish_inverted_punctuation() -> None:
    _, emp = classify_emphasis("¡bueno")
    assert emp == "emphasis"
    _, emp2 = classify_emphasis("¿qué")
    assert emp2 == "emphasis"


def test_classify_laughter_token_is_reaction() -> None:
    _, emp = classify_emphasis("lmao")
    assert emp == "reaction"
    _, emp2 = classify_emphasis("JAJAJA")
    assert emp2 == "reaction"  # case-folded


def test_classify_two_letter_caps_is_not_shout() -> None:
    """Two-letter ALL-CAPS like 'OK' / 'AI' isn't shouting — too short."""
    _, emp = classify_emphasis("AI")
    assert emp == "normal"


def test_classify_empty_string_returns_normal() -> None:
    text, emp = classify_emphasis("")
    assert text == ""
    assert emp == "normal"


def test_classify_strips_punctuation_for_lookup_but_keeps_terminal_for_emphasis() -> None:
    """Word `lmao!` should classify as reaction (the laughter token
    wins over the terminal-bang emphasis tag)."""
    _, emp = classify_emphasis("lmao!")
    assert emp == "reaction"


# ---- chunk_words_to_lines ----


def _w(ts: float, end_ts: float, text: str) -> dict[str, object]:
    return {"ts": ts, "end_ts": end_ts, "text": text}


def test_chunk_filters_to_clip_window_and_shifts_timing() -> None:
    """Words before clip_start or after clip_end are dropped; in-window
    words get shifted to clip-relative timing (0 at clip_start)."""
    words = [
        _w(100.0, 100.3, "before"),  # outside (clip starts at 105)
        _w(106.0, 106.3, "inside"),  # inside → clip-relative 1.0
        _w(200.0, 200.3, "after"),   # outside
    ]
    lines = chunk_words_to_lines(
        words, clip_start_s=105.0, clip_end_s=115.0
    )
    assert len(lines) == 1
    assert lines[0].text == "inside"
    assert lines[0].ts == 1.0  # 106 - 105
    assert lines[0].end_ts > 1.0


def test_chunk_max_words_per_line_caps_chunk_size() -> None:
    """Three normal words in tight succession all land on ONE line,
    then the fourth starts a fresh line."""
    words = [
        _w(0.0, 0.3, "uno"),
        _w(0.3, 0.6, "dos"),
        _w(0.6, 0.9, "tres"),
        _w(0.9, 1.2, "cuatro"),
    ]
    lines = chunk_words_to_lines(words, clip_start_s=0, clip_end_s=10)
    assert len(lines) == 2
    assert len(lines[0].words) == MAX_WORDS_PER_LINE
    assert lines[1].text == "cuatro"


def test_chunk_pause_breaks_line() -> None:
    """A pause >= LINE_BREAK_PAUSE_S between two words starts a new line."""
    words = [
        _w(0.0, 0.3, "uno"),
        _w(0.3 + LINE_BREAK_PAUSE_S + 0.1, 1.0, "dos"),  # long pause
    ]
    lines = chunk_words_to_lines(words, clip_start_s=0, clip_end_s=10)
    assert len(lines) == 2


def test_chunk_shout_word_gets_its_own_line() -> None:
    """ALL-CAPS forces a flush — it stands alone for the visual punch."""
    words = [
        _w(0.0, 0.3, "the"),
        _w(0.3, 0.6, "INSANE"),  # shout
        _w(0.6, 0.9, "thing"),
    ]
    lines = chunk_words_to_lines(words, clip_start_s=0, clip_end_s=10)
    assert len(lines) == 3
    assert lines[1].words[0].text == "INSANE"
    assert lines[1].words[0].emphasis == "shout"
    assert lines[1].emphasis == "shout"


def test_chunk_reaction_word_gets_its_own_line() -> None:
    words = [
        _w(0.0, 0.3, "no"),
        _w(0.3, 0.6, "haha"),
        _w(0.6, 0.9, "way"),
    ]
    lines = chunk_words_to_lines(words, clip_start_s=0, clip_end_s=10)
    kinds = [ln.emphasis for ln in lines]
    assert "reaction" in kinds


def test_chunk_clamps_words_at_window_edges() -> None:
    """A word straddling clip_end_s gets clipped to end at the window."""
    words = [_w(2.0, 6.0, "long")]  # clip ends at 4
    lines = chunk_words_to_lines(words, clip_start_s=0, clip_end_s=4)
    assert lines[0].words[0].end_ts <= 4.0


def test_chunk_line_span_caps_at_max_duration() -> None:
    """A single line never spans more than MAX_LINE_DURATION_S."""
    words = [
        _w(0.0, 0.4, "one"),
        _w(0.5, 0.9, "two"),
        _w(1.0, MAX_LINE_DURATION_S + 0.2, "three"),
    ]
    lines = chunk_words_to_lines(words, clip_start_s=0, clip_end_s=10)
    for ln in lines:
        assert (ln.end_ts - ln.ts) <= MAX_LINE_DURATION_S + 0.5


def test_chunk_empty_input_returns_empty_list() -> None:
    assert chunk_words_to_lines([], clip_start_s=0, clip_end_s=10) == []


# ---- captions_for_clip (end-to-end with the DB shape) ----


def test_captions_for_clip_reads_real_db_shape_ts_end_ts() -> None:
    """The DB shape is `{ts, end_ts}` (Pydantic field names from
    transcribe.models.Segment). build_srt and build_ass MUST read
    these, not the legacy `{start, end}` keys. This is the bug that
    silently produced empty captions on production data."""
    segments = [
        {
            "ts": 100.0,
            "end_ts": 101.5,
            "text": "hello world test",
            "words": [
                {"ts": 100.0, "end_ts": 100.3, "text": "hello", "prob": 0.9},
                {"ts": 100.3, "end_ts": 100.7, "text": "world", "prob": 0.9},
                {"ts": 100.7, "end_ts": 101.5, "text": "test!", "prob": 0.9},
            ],
        },
    ]
    lines = captions_for_clip(
        json.dumps(segments), clip_start_s=99.0, clip_end_s=105.0
    )
    assert lines
    # First word should be at clip-relative 1.0 (100 - 99).
    assert lines[0].words[0].ts == 1.0
    # The bang-terminated word fires emphasis.
    flat = [w for ln in lines for w in ln.words]
    assert any(w.emphasis == "emphasis" for w in flat)


def test_captions_for_clip_falls_back_to_segment_text_without_words() -> None:
    """Legacy segments without word-level data still produce lines —
    the segment text is split evenly across the segment's duration."""
    segments = [
        {"ts": 0.0, "end_ts": 3.0, "text": "the quick brown fox jumps"},
    ]
    lines = captions_for_clip(
        json.dumps(segments), clip_start_s=0, clip_end_s=10
    )
    assert lines
    # 5 words distributed across 3s → words available.
    flat = [w for ln in lines for w in ln.words]
    assert len(flat) == 5


def test_captions_for_clip_tolerates_legacy_start_end_keys() -> None:
    """The `start`/`end` shape (older fixtures, or migration cruft)
    is also accepted — fallback path inside _read_word_dicts."""
    segments = [
        {"start": 0.0, "end": 1.0, "text": "fallback test"},
    ]
    lines = captions_for_clip(
        json.dumps(segments), clip_start_s=0, clip_end_s=10
    )
    assert lines


def test_captions_for_clip_returns_empty_on_invalid_json() -> None:
    assert captions_for_clip("not json", clip_start_s=0, clip_end_s=10) == []


def test_captions_for_clip_returns_empty_on_non_list_payload() -> None:
    assert captions_for_clip("{}", clip_start_s=0, clip_end_s=10) == []


def test_captions_for_clip_returns_empty_on_empty_string() -> None:
    assert captions_for_clip("", clip_start_s=0, clip_end_s=10) == []


# ---- lines_to_json ----


def test_lines_to_json_serializes_word_level_data() -> None:
    """The dashboard endpoint serves whatever lines_to_json returns.
    The shape must match what the editor JS consumes:
    {ts, end_ts, text, emphasis, words: [{ts, end_ts, text, emphasis}]}"""
    words = [_w(0.0, 0.5, "BIG"), _w(0.5, 1.0, "moment")]
    lines = chunk_words_to_lines(words, clip_start_s=0, clip_end_s=10)
    out = lines_to_json(lines)
    assert isinstance(out, list)
    first = out[0]
    assert "ts" in first and "end_ts" in first
    assert "text" in first
    assert "emphasis" in first
    assert isinstance(first["words"], list)
    assert first["words"][0]["emphasis"] == "shout"  # BIG is ALL-CAPS


def test_lines_to_json_rounds_timestamps_to_millis() -> None:
    """3 decimal places — sub-millisecond precision is noise."""
    words = [_w(0.123456789, 0.5, "x")]
    lines = chunk_words_to_lines(words, clip_start_s=0, clip_end_s=10)
    out = lines_to_json(lines)
    assert out[0]["ts"] == 0.123  # rounded
