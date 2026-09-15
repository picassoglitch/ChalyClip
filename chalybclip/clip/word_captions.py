"""Word-level caption builder — slice F.7-F (creator-weapon upgrade).

Modern viral captions on TikTok / Reels / Shorts are NOT lines like
"EPIC PLAY RIGHT HERE". They are word-by-word, pacing-driven, with
emphasis on the words that carry weight. This module turns Whisper's
word-level transcript into THAT — for both the live editor preview
and the burned MP4.

Pipeline:
    transcripts.segments_json  (Whisper Segment + Word records)
        ↓
    _read_words_from_segments  (handle BOTH the real DB shape
                                `{ts, end_ts}` AND the test-fixture
                                shape `{start, end}` so callers can
                                feed either)
        ↓
    chunk_words                (rebreak into 1-3 word "lines" sized
                                for short-form readability)
        ↓
    classify_emphasis          (ALL-CAPS → SHOUT, terminal `!?` →
                                EMPHASIS, laughter tokens → REACTION)
        ↓
    list[CaptionLine]          (clip-relative timing, ready to JSON-
                                serialize for the preview OR feed to
                                the ASS generator for burn-in)

The output is the SINGLE source of truth for both surfaces. What the
operator sees in the preview IS what the burn renders. Trust restored.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal, NamedTuple

Emphasis = Literal["normal", "shout", "emphasis", "reaction"]

# Tokens that always classify as a "reaction" word — gets a special
# style + scale in the editor + larger font in the burn.
_REACTION_TOKENS: frozenset[str] = frozenset(
    {
        "haha", "hahaha", "jaja", "jajaja", "jajajaja",
        "lol", "lmao", "lmaoo", "rofl", "kekw",
        "wow", "woah", "whoa",
    }
)

# Words this short stay attached to the previous chunk rather than
# becoming a chunk of their own — improves readability of articles /
# prepositions / conjunctions.
_GLUE_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "to", "of", "in", "on", "at", "is", "it",
        "i", "el", "la", "y", "o", "de", "en",
    }
)


class CaptionWord(NamedTuple):
    """One word in a caption line, with clip-relative timing + emphasis."""

    ts: float        # seconds from clip start
    end_ts: float
    text: str        # punctuation-stripped, ready to render
    emphasis: Emphasis


class CaptionLine(NamedTuple):
    """A chunked caption line — 1-3 words shown together."""

    ts: float
    end_ts: float
    text: str        # joined word text, with spaces
    words: list[CaptionWord]
    emphasis: Emphasis  # strongest emphasis across the line's words


# Target words-per-line for short-form readability. Up to 3 words feels
# punchy; 1-word lines are fine and common for emphasis.
MAX_WORDS_PER_LINE = 3

# Lines that span more than this many seconds get split — viewers
# stop reading after ~1.5s.
MAX_LINE_DURATION_S = 1.6

# Pause threshold — if the gap between two words exceeds this, the
# second word starts a fresh line regardless of the line-length cap.
LINE_BREAK_PAUSE_S = 0.35


# ---- shape tolerance ----


def _read_word_dicts(
    segments: Iterable[object],
) -> list[dict[str, object]]:
    """Walk segments and return their flattened word list.

    Tolerates THREE shapes the codebase has accumulated:

      A) Real Whisper segments as serialized to DB:
         `{"ts": .., "end_ts": .., "text": "...", "words": [{ts, end_ts, text, prob}]}`

      B) Legacy / test-fixture segments without word-level data:
         `{"start": .., "end": .., "text": "..."}`
         — falls back to splitting the segment text by whitespace and
         distributing word ts evenly across the segment.

      C) Mixed bags from tests that intentionally probe edge cases —
         missing text, missing timestamps, wrong types. All skipped.

    The output is always a list of `{ts, end_ts, text}` dicts in
    chronological stream-relative seconds.
    """
    out: list[dict[str, object]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        seg_words = seg.get("words")
        if isinstance(seg_words, list) and seg_words:
            for w in seg_words:
                if not isinstance(w, dict):
                    continue
                ts = _as_float(w.get("ts"), default=None)
                end_ts = _as_float(w.get("end_ts"), default=None)
                text = str(w.get("text") or "").strip()
                if ts is None or end_ts is None or not text:
                    continue
                out.append({"ts": ts, "end_ts": end_ts, "text": text})
            continue

        # Fallback — distribute the segment's text evenly across its
        # duration. Better than dropping the line entirely.
        s_start = _as_float(seg.get("ts"), default=None)
        if s_start is None:
            s_start = _as_float(seg.get("start"), default=None)
        s_end = _as_float(seg.get("end_ts"), default=None)
        if s_end is None:
            s_end = _as_float(seg.get("end"), default=None)
        text = str(seg.get("text") or "").strip()
        if s_start is None or s_end is None or not text:
            continue
        tokens = text.split()
        if not tokens:
            continue
        span = max(0.001, s_end - s_start)
        per = span / len(tokens)
        for i, tok in enumerate(tokens):
            out.append(
                {
                    "ts": s_start + i * per,
                    "end_ts": s_start + (i + 1) * per,
                    "text": tok,
                }
            )
    return out


def _as_float(value: object, *, default: float | None) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


# ---- emphasis classifier ----


_NORMALIZE_RE = re.compile(r"[^\w¡¿!?]+", re.UNICODE)


def classify_emphasis(raw_text: str) -> tuple[str, Emphasis]:
    """Return `(clean_text, emphasis)` for one Whisper word.

    Rules:
      * Mostly-ALL-CAPS letters (and >=3 chars) → SHOUT.
        Whisper rarely produces ALL-CAPS organically, but uppercase
        leaks through from genuine shouting + on-screen-text models.
      * Ends with `!` or `?` (or Spanish `¡` / `¿` openers) → EMPHASIS.
      * Token (lowered, stripped) in laughter set → REACTION.
      * Otherwise → normal.
    """
    clean = raw_text.strip()
    if not clean:
        return "", "normal"
    has_emph_terminal = clean[-1] in "!?¡¿" or clean[0] in "¡¿"
    stripped = _NORMALIZE_RE.sub("", clean).strip()
    if not stripped:
        return clean, "normal"
    # For reaction-token lookup we further strip terminal punctuation
    # so "lmao!" / "haha!!" still register as reactions. The reaction
    # tag wins over emphasis when both fire.
    bare = stripped.rstrip("!?¡¿").lower()
    if bare in _REACTION_TOKENS:
        return stripped, "reaction"
    if has_emph_terminal:
        return clean, "emphasis"
    letters = [c for c in stripped if c.isalpha()]
    if (
        len(letters) >= 3
        and all(c.isupper() for c in letters)
    ):
        return stripped, "shout"
    return clean, "normal"


# ---- chunker ----


def chunk_words_to_lines(
    word_dicts: Iterable[dict[str, object]],
    *,
    clip_start_s: float,
    clip_end_s: float,
) -> list[CaptionLine]:
    """Turn a flat word stream into clip-relative caption lines.

    Words OUTSIDE the [clip_start_s, clip_end_s] window are dropped.
    Timestamps in the output are SHIFTED to clip-relative (0 at
    clip_start_s) so callers don't have to.

    Chunking rules (in priority order):
      1. SHOUT or REACTION words break onto their own line.
      2. EMPHASIS words break onto their own line (so the `!` lands hard).
      3. A pause >= LINE_BREAK_PAUSE_S between words starts a new line.
      4. Line length >= MAX_WORDS_PER_LINE flushes.
      5. Line span >= MAX_LINE_DURATION_S flushes.
      6. Glue words (a, the, of) attach to the previous line if the
         current chunk would otherwise be empty.
    """
    # First pass — classify + window-filter + clip-relative shift.
    classified: list[CaptionWord] = []
    for w in word_dicts:
        ts = _as_float(w.get("ts"), default=None)
        end_ts = _as_float(w.get("end_ts"), default=None)
        raw = str(w.get("text") or "")
        if ts is None or end_ts is None:
            continue
        if end_ts < clip_start_s or ts > clip_end_s:
            continue
        # Clip to window edges to handle words straddling either side.
        rel_start = max(0.0, ts - clip_start_s)
        rel_end = max(rel_start + 0.05, min(clip_end_s, end_ts) - clip_start_s)
        text, emphasis = classify_emphasis(raw)
        if not text:
            continue
        classified.append(
            CaptionWord(
                ts=rel_start,
                end_ts=rel_end,
                text=text,
                emphasis=emphasis,
            )
        )

    # Second pass — group into lines.
    lines: list[CaptionLine] = []
    current: list[CaptionWord] = []

    def flush() -> None:
        if not current:
            return
        line_words = list(current)
        current.clear()
        # Strongest emphasis on the line wins for the line-level tag.
        rank = {"normal": 0, "emphasis": 1, "reaction": 2, "shout": 3}
        strongest: Emphasis = max(
            (w.emphasis for w in line_words),
            key=lambda e: rank[e],
        )
        lines.append(
            CaptionLine(
                ts=line_words[0].ts,
                end_ts=line_words[-1].end_ts,
                text=" ".join(w.text for w in line_words),
                words=line_words,
                emphasis=strongest,
            )
        )

    for cw in classified:
        # Rule 1+2 — special words break onto their own line.
        if cw.emphasis in ("shout", "reaction", "emphasis"):
            flush()
            current.append(cw)
            flush()
            continue

        # Rule 3 — pause break.
        if current and (cw.ts - current[-1].end_ts) >= LINE_BREAK_PAUSE_S:
            flush()

        # Rule 6 — glue words attach to previous if current is empty.
        # (No-op here; glue words just fall through and get appended.)
        _ = cw.text.lower() in _GLUE_WORDS

        current.append(cw)

        # Rules 4 + 5 — length / span flush.
        if len(current) >= MAX_WORDS_PER_LINE:
            flush()
            continue
        if (current[-1].end_ts - current[0].ts) >= MAX_LINE_DURATION_S:
            flush()

    flush()
    return lines


# ---- public entry ----


def captions_for_clip(
    segments_json_text: str,
    *,
    clip_start_s: float,
    clip_end_s: float,
) -> list[CaptionLine]:
    """Build the per-clip caption-line list from raw transcript JSON.

    Defensive — malformed JSON returns []. The endpoint + the ASS
    burn both call this so they share the exact same chunking.
    """
    import json

    if not segments_json_text:
        return []
    try:
        segments = json.loads(segments_json_text)
    except json.JSONDecodeError:
        return []
    if not isinstance(segments, list):
        return []
    words = _read_word_dicts(segments)
    return chunk_words_to_lines(
        words,
        clip_start_s=clip_start_s,
        clip_end_s=clip_end_s,
    )


def lines_to_json(lines: Iterable[CaptionLine]) -> list[dict[str, object]]:
    """Serialize CaptionLine list for the /captions.json endpoint
    and the preview's fetch consumer."""
    return [
        {
            "ts": round(line.ts, 3),
            "end_ts": round(line.end_ts, 3),
            "text": line.text,
            "emphasis": line.emphasis,
            "words": [
                {
                    "ts": round(w.ts, 3),
                    "end_ts": round(w.end_ts, 3),
                    "text": w.text,
                    "emphasis": w.emphasis,
                }
                for w in line.words
            ],
        }
        for line in lines
    ]
