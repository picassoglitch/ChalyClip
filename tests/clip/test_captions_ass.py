"""R16 — ASS subtitle generator for ffmpeg libass burn-in.

The hybrid recorder's screenshot pipeline could not capture dynamic
caption DOM on headless Chromium reliably (documented in the R11-R15
chain). We switched to burning captions via ffmpeg's `ass=` filter.
These tests pin the file format and the per-word karaoke timing so
a refactor doesn't ship a malformed ASS to libass.
"""
from __future__ import annotations

from pathlib import Path

from chalybclip.clip.captions_ass import generate_ass


def _lines(words: list[tuple[str, float, float]], *, end_pad: float = 0.0):
    """Build a single caption-line dict from (text, ts, end_ts) tuples."""
    if not words:
        return [{"ts": 0.0, "end_ts": 0.0, "text": "", "words": []}]
    return [
        {
            "ts": words[0][1],
            "end_ts": words[-1][2] + end_pad,
            "text": " ".join(w[0] for w in words),
            "words": [
                {"ts": ts, "end_ts": ets, "text": text, "emphasis": None}
                for text, ts, ets in words
            ],
        }
    ]


def test_writes_valid_ass_header(tmp_path: Path) -> None:
    out = tmp_path / "captions.ass"
    generate_ass(
        lines=_lines([("hello", 0.5, 1.0), ("world", 1.0, 1.5)]),
        width=1080,
        height=1920,
        output_path=out,
    )
    text = out.read_text(encoding="utf-8")
    assert text.startswith("[Script Info]")
    assert "ScriptType: v4.00+" in text
    assert "PlayResX: 1080" in text
    assert "PlayResY: 1920" in text
    assert "[V4+ Styles]" in text
    assert "[Events]" in text
    assert "Style: Default,Inter," in text


def test_emits_one_dialogue_per_word(tmp_path: Path) -> None:
    """For karaoke-style highlighting, every word gets its own dialogue
    line spanning [word.ts, next_word.ts). Three words → three
    Dialogue lines, each with the active word color-shifted."""
    out = tmp_path / "captions.ass"
    generate_ass(
        lines=_lines([
            ("hello", 0.5, 0.9),
            ("clipped", 0.9, 1.3),
            ("world", 1.3, 1.7),
        ]),
        width=1080,
        height=1920,
        output_path=out,
    )
    text = out.read_text(encoding="utf-8")
    dialogues = [ln for ln in text.splitlines() if ln.startswith("Dialogue:")]
    assert len(dialogues) == 3, dialogues
    # Each dialogue must include all three words (the line text is
    # constant; only the highlight position moves between dialogues).
    for d in dialogues:
        assert "hello" in d
        assert "clipped" in d
        assert "world" in d
    # And each highlights exactly one word — easiest pin: each
    # dialogue carries one `{\1c&H...&}` open + one close.
    for d in dialogues:
        assert d.count("\\1c&H") == 2, (
            "expected one highlight open + one close per dialogue, "
            f"got: {d}"
        )


def test_highlight_color_is_bgr_not_rgb(tmp_path: Path) -> None:
    """ASS color tags use BGR hex. `#ffd84a` (RGB) → `4AD8FF` (BGR)."""
    out = tmp_path / "captions.ass"
    generate_ass(
        lines=_lines([("word", 0.0, 0.5)]),
        width=1080, height=1920, output_path=out,
        highlight_color_hex="#ffd84a",
    )
    text = out.read_text(encoding="utf-8")
    assert "&H4AD8FF&" in text, "expected BGR `4AD8FF` for RGB `#ffd84a`"
    assert "&H00FFD84A&" not in text, "ASS doesn't take RGB"


def test_position_lower_third_pushes_margin_v_to_36_pct(tmp_path: Path) -> None:
    """`lower_third` baseline sits ~64% from top of a 1920-tall canvas
    → MarginV (distance from bottom for alignment 2) ≈ 36% of height.
    Bumped from 32% to 36% after the first R16 render put the caption
    text uncomfortably close to the Kick banner's top edge."""
    out = tmp_path / "captions.ass"
    generate_ass(
        lines=_lines([("w", 0.0, 0.5)]),
        width=1080, height=1920, output_path=out,
        position="lower_third",
    )
    text = out.read_text(encoding="utf-8")
    # Default style line ends with the MarginV value before encoding.
    style_line = next(ln for ln in text.splitlines() if ln.startswith("Style: Default,"))
    # Fields are comma-separated; MarginV is the 22nd "data" field.
    fields = style_line.split(",")
    margin_v = int(fields[-2])
    assert 680 <= margin_v <= 700, margin_v


def test_position_centered_uses_half_height(tmp_path: Path) -> None:
    out = tmp_path / "captions.ass"
    generate_ass(
        lines=_lines([("w", 0.0, 0.5)]),
        width=1080, height=1920, output_path=out,
        position="centered",
    )
    text = out.read_text(encoding="utf-8")
    style_line = next(ln for ln in text.splitlines() if ln.startswith("Style: Default,"))
    margin_v = int(style_line.split(",")[-2])
    assert margin_v == 960


def test_empty_lines_produces_header_only(tmp_path: Path) -> None:
    """No captions → still write the [Script Info]/[Styles]/[Events]
    skeleton so ffmpeg's libass doesn't reject the file. Zero
    Dialogue lines is fine."""
    out = tmp_path / "captions.ass"
    generate_ass(
        lines=[],
        width=1080, height=1920, output_path=out,
    )
    text = out.read_text(encoding="utf-8")
    assert "[Events]" in text
    assert "Dialogue:" not in text


def test_curly_braces_in_word_text_are_stripped(tmp_path: Path) -> None:
    """ASS uses `{...}` for override tags. A `{` in user-typed text
    would break the parser; the generator strips them defensively."""
    out = tmp_path / "captions.ass"
    generate_ass(
        lines=_lines([("hel{lo}", 0.0, 0.5)]),
        width=1080, height=1920, output_path=out,
    )
    text = out.read_text(encoding="utf-8")
    dialogue = next(ln for ln in text.splitlines() if ln.startswith("Dialogue:"))
    # Inside the highlight override the only braces should be the
    # `\1c&H...&` opens + closes — not the user's `{lo}`.
    assert "{lo}" not in dialogue
    assert "hello" in dialogue


def test_word_dialogue_covers_to_next_word_start(tmp_path: Path) -> None:
    """Per-word dialogue ends at the NEXT word's start, not the
    current word's end_ts. Keeps the caption continuously visible
    across word boundaries — only the highlight position moves."""
    out = tmp_path / "captions.ass"
    generate_ass(
        lines=_lines([
            ("a", 0.0, 0.2),
            ("b", 0.5, 0.7),  # gap 0.3-0.5 between a.end and b.start
        ]),
        width=1080, height=1920, output_path=out,
    )
    text = out.read_text(encoding="utf-8")
    dialogues = [ln for ln in text.splitlines() if ln.startswith("Dialogue:")]
    # First word's dialogue should run until 0.5 (when b starts), not
    # 0.2 (when a ended). Pin via the End time field.
    first = dialogues[0]
    # Dialogue: 0,0:00:00.00,0:00:00.50,Default,...
    parts = first.split(",")
    end_time = parts[2]
    assert end_time == "0:00:00.50", end_time


def test_malformed_hex_color_falls_back_to_white(tmp_path: Path) -> None:
    """A broken highlight_color shouldn't crash the burn. The generator
    falls back to white so ffmpeg still produces a valid file."""
    out = tmp_path / "captions.ass"
    generate_ass(
        lines=_lines([("word", 0.0, 0.5)]),
        width=1080, height=1920, output_path=out,
        highlight_color_hex="not-a-color",
    )
    text = out.read_text(encoding="utf-8")
    # SecondaryColour in the style line should be white BGR.
    style_line = next(ln for ln in text.splitlines() if ln.startswith("Style: Default,"))
    assert "&H00FFFFFF&" in style_line, style_line
