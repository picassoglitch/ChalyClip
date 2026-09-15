"""Generate ASS subtitle files from word-level caption data.

R16 — switching from in-page DOM caption rendering to ffmpeg's
`ass=` filter. The hybrid recorder's screenshot pipeline could
not reliably rasterize dynamic innerHTML on headless Chromium
(documented in the R11-R15 chain across this codebase). ASS via
libass sidesteps the browser entirely: captions get baked into
the video's pixels by ffmpeg, identical across platforms, no
DOM/composite/screenshot timing to chase. This is how Opus Clip,
CapCut, and every serious short-form pipeline ships captions.

The generator emits ONE dialogue line per word with the full line
as text and the active word switched to the highlight color via
an inline `\1c` override. Each word's dialogue covers `[word.ts,
next_word.ts)` so the caption stays visible across the whole line
with the highlighted "leading edge" hopping word-to-word at the
transcription's word boundaries. Matches the operator-visible
TikTok-style karaoke effect without paying for separate base +
highlight layers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


# ASS uses BGR hex, not RGB. The CSS `--pv-highlight` default is
# `#ffd84a` → BGR = 4AD8FF.
_WHITE_BGR = "FFFFFF"
_BLACK_BGR = "000000"


def _hex_to_bgr(rgb_hex: str | None) -> str:
    """Convert `#RRGGBB` → `BBGGRR` for ASS color tags.

    Falls back to white on a malformed input so the burn step
    can never crash on a hex-parse error.
    """
    if not rgb_hex:
        return _WHITE_BGR
    h = rgb_hex.lstrip("#").strip()
    if len(h) != 6:
        return _WHITE_BGR
    try:
        # Validate it's actually hex
        int(h, 16)
    except ValueError:
        return _WHITE_BGR
    return (h[4:6] + h[2:4] + h[0:2]).upper()


def _format_time(seconds: float) -> str:
    """`12.345` → `0:00:12.34` (ASS `H:MM:SS.cc`).

    libass reads at centisecond precision; rounding here matches.
    """
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _escape_ass_text(text: str) -> str:
    r"""Strip ASS-significant characters from user-typed words so a
    rogue `{` or `}` in a transcript doesn't break the override
    parser. Backslashes are also stripped to defang `\h` / `\n`
    that could change layout."""
    return (
        (text or "")
        .replace("\\", "")
        .replace("{", "")
        .replace("}", "")
    )


def generate_ass(
    *,
    lines: Sequence[dict[str, Any]],
    width: int,
    height: int,
    output_path: Path,
    font_family: str = "Inter",
    font_size: int = 64,
    highlight_color_hex: str = "#ffd84a",
    position: str = "lower_third",
) -> Path:
    """Write a valid ASS subtitle file at `output_path`.

    `lines` must match the shape `clip_captions_for_clip` produces:

        [
          {
            "ts": float,        # clip-relative line start
            "end_ts": float,    # clip-relative line end
            "text": str,        # joined line text
            "words": [
              {
                "ts": float,
                "end_ts": float,
                "text": str,
                "emphasis": str | None,
              },
              ...
            ],
          },
          ...
        ]

    The output is UTF-8 (libass requires it) and uses bottom-center
    alignment with MarginV tuned to the requested vertical position.
    """
    highlight_bgr = _hex_to_bgr(highlight_color_hex)

    # MarginV is the distance from the screen edge (bottom for
    # alignment 2). lower_third nudged up from 0.32 to 0.36 after the
    # first operator-visible R16 render put the karaoke text a hair
    # too close to the top edge of the Kick banner (~50px gap). 0.36
    # gives ~115px between caption baseline and banner top, matching
    # the editor preview's visual breathing room.
    position_map = {
        "upper_third": int(height * 0.76),   # baseline ~24% from top
        "centered":    int(height * 0.50),
        "lower_third": int(height * 0.36),   # baseline ~64% from top
        "bottom":      int(height * 0.22),   # baseline ~78% from top
    }
    margin_v = position_map.get(position, position_map["lower_third"])

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "ScaledBorderAndShadow: yes\n"
        "WrapStyle: 2\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, "
        "SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, "
        "MarginV, Encoding\n"
        # Default style: bold white text, black 4px outline, drop
        # shadow, bottom-center anchored, MarginV pushes it up to
        # the requested vertical position.
        f"Style: Default,{font_family},{font_size},"
        f"&H00{_WHITE_BGR}&,&H00{highlight_bgr}&,"
        f"&H00{_BLACK_BGR}&,&H80000000,"
        f"1,0,0,0,100,100,0,0,1,4,2,"
        f"2,80,80,{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\n"
    )

    events: list[str] = []
    for line in lines:
        words = line.get("words") or []
        line_end = float(line.get("end_ts") or 0.0)
        if not words:
            text = _escape_ass_text(line.get("text") or "")
            if not text:
                continue
            events.append(
                f"Dialogue: 0,"
                f"{_format_time(float(line.get('ts') or 0.0))},"
                f"{_format_time(line_end)},Default,,0,0,0,,{text}"
            )
            continue

        for i, w in enumerate(words):
            start = float(w.get("ts") or 0.0)
            # The dialogue line stays visible until the next word's
            # start (or the line's end for the last word) so the
            # caption never blinks out between word boundaries —
            # only the highlighted word "advances" along the line.
            if i + 1 < len(words):
                end = float(words[i + 1].get("ts") or start + 0.1)
            else:
                end = line_end if line_end > start else start + 0.1

            parts: list[str] = []
            for j, ww in enumerate(words):
                text_w = _escape_ass_text(ww.get("text") or "")
                if not text_w:
                    continue
                if j == i:
                    # Active word: highlight color, switch back after.
                    parts.append(
                        f"{{\\1c&H{highlight_bgr}&}}"
                        f"{text_w}"
                        f"{{\\1c&H{_WHITE_BGR}&}}"
                    )
                else:
                    parts.append(text_w)
            line_text = " ".join(parts)
            if not line_text:
                continue

            events.append(
                f"Dialogue: 0,{_format_time(start)},"
                f"{_format_time(end)},Default,,0,0,0,,{line_text}"
            )

    body = header + "\n".join(events) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(body, encoding="utf-8")
    return output_path


def escape_ass_path_for_filter(ass_file_path: Path) -> str:
    """Escape a filesystem path for ffmpeg's `ass='<path>'` filter arg.

    ffmpeg filter syntax requires single quotes around the filename AND
    escaping of `:` and `\\` inside it. On POSIX the cache path is
    `/data/out/.../*.ass` — no problematic chars. The escape handles
    Windows / containerized paths defensively. Shared by BOTH recorders
    (hybrid composite + legacy fallback encode) so the escaping rules
    can't drift between them.
    """
    return (
        str(ass_file_path)
        .replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
    )


__all__ = ["escape_ass_path_for_filter", "generate_ass"]
