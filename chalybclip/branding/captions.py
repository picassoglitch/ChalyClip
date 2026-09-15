"""Caption preset library — voice-markers spec §3.4.

The actual ASS-subtitle generation + ffmpeg burn-in happens in the
renderer (slice E task — re-transcribes at word level for tight sync).
This module is the SOURCE OF TRUTH for what a preset *is*:

  * `karaoke_pop`   — word-by-word highlight, scale bounce (default)
  * `typewriter`    — char-by-char reveal
  * `bold_block`    — 2-3 word chunks, hard cut
  * `subtle`        — single line bottom, no animation (serious content)

The schema covers everything the renderer needs to lay out captions:
font, sizing, fills, stroke, position, animation, shadow. Each preset
also carries an `id` (the storage key) and `display_name` (UI label).

Stored on `brand_kits.caption_style_json` as the full CaptionStyle JSON
so brand kits can override individual fields without re-doing the
preset selector. The UI flow: pick a preset → schema is copied in →
user tweaks the few fields they care about → save.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CaptionPresetId = Literal["karaoke_pop", "typewriter", "bold_block", "subtle"]
CaptionPosition = Literal["upper_third", "centered", "lower_third", "bottom"]
# Slice F.7-H added "pop" / "slide" as editor-surface animation choices;
# kept the legacy "pop_in" / "none" to round-trip older saved kits.
CaptionAnimation = Literal["pop", "slide", "fade", "typewriter", "pop_in", "none"]
CaptionFontSize = Literal["small", "medium", "large", "xl"]
HighlightMode = Literal["active_word", "active_chunk", "none"]


class CaptionShadow(BaseModel):
    """Drop-shadow params. `enabled=False` → renderer skips the shadow pass."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    offset_y: int = 3
    blur: int = 6
    color: str = "rgba(0,0,0,0.6)"


class CaptionStyle(BaseModel):
    """Complete caption render contract. One per brand kit.

    Fields map 1:1 to ASS-subtitle / drawtext properties the renderer
    needs. Adding a new field here is a schema-compatible change as
    long as it has a default — old kits will round-trip cleanly.
    """

    model_config = ConfigDict(extra="forbid")

    preset: CaptionPresetId = "karaoke_pop"
    font_family: str = "Inter"
    font_weight: int = Field(default=900, ge=100, le=900)
    font_size_px: int = Field(default=72, gt=0)
    fill_color: str = "#FFFFFF"
    stroke_color: str = "#000000"
    stroke_width_px: int = Field(default=5, ge=0)
    highlight_color: str = "#FFD700"
    highlight_mode: HighlightMode = "active_word"
    position: CaptionPosition = "upper_third"
    max_words_per_line: int = Field(default=4, ge=1, le=20)
    animation: CaptionAnimation = "pop_in"
    shadow: CaptionShadow = Field(default_factory=CaptionShadow)
    censor_swears: bool = True

    # Slice H.1 — editor-controlled knobs the operator picks from the
    # right panel. Persisted to brand_kits.caption_style_json so they
    # survive across clips. font_size is a coarse band (the renderer
    # multiplies font_size_px by the band's scale factor — small=0.78,
    # medium=1.0, large=1.24, xl=1.5). lead_ms shifts caption timing
    # FORWARD in the preview AND the ASS burn so words can read-ahead.
    font_size: CaptionFontSize = "medium"
    lead_ms: int = Field(default=0, ge=0, le=500)


def builtin_presets() -> dict[CaptionPresetId, CaptionStyle]:
    """The four shipped presets. UI populates the preset dropdown from this.

    Each preset is a fully-realized CaptionStyle so the user can apply
    one and tweak individual fields without losing the rest of the
    coherent design.
    """
    return {
        "karaoke_pop": CaptionStyle(
            preset="karaoke_pop",
            font_family="Inter",
            font_weight=900,
            font_size_px=72,
            fill_color="#FFFFFF",
            stroke_color="#000000",
            stroke_width_px=5,
            highlight_color="#FFD700",
            highlight_mode="active_word",
            position="upper_third",
            max_words_per_line=4,
            animation="pop_in",
            shadow=CaptionShadow(enabled=True, offset_y=3, blur=6),
            censor_swears=True,
        ),
        "typewriter": CaptionStyle(
            preset="typewriter",
            font_family="JetBrains Mono",
            font_weight=700,
            font_size_px=56,
            fill_color="#FFFFFF",
            stroke_color="#000000",
            stroke_width_px=3,
            highlight_mode="none",
            position="centered",
            max_words_per_line=8,
            animation="typewriter",
            shadow=CaptionShadow(enabled=True, offset_y=2, blur=4),
            censor_swears=False,
        ),
        "bold_block": CaptionStyle(
            preset="bold_block",
            font_family="Inter",
            font_weight=900,
            font_size_px=88,
            fill_color="#FFFFFF",
            stroke_color="#000000",
            stroke_width_px=7,
            highlight_color="#FFD700",
            highlight_mode="active_chunk",
            position="upper_third",
            max_words_per_line=3,
            animation="none",
            shadow=CaptionShadow(enabled=False),
            censor_swears=True,
        ),
        "subtle": CaptionStyle(
            preset="subtle",
            font_family="Inter",
            font_weight=500,
            font_size_px=48,
            fill_color="#FFFFFF",
            stroke_color="#000000",
            stroke_width_px=2,
            highlight_mode="none",
            position="lower_third",
            max_words_per_line=8,
            animation="fade",
            shadow=CaptionShadow(enabled=True, offset_y=1, blur=3),
            censor_swears=False,
        ),
    }


def preset_choices() -> list[tuple[CaptionPresetId, str]]:
    """(id, ui-label) list for the brand-kit form's preset dropdown."""
    return [
        ("karaoke_pop", "Karaoke pop — word-by-word highlight, scale bounce"),
        ("typewriter", "Typewriter — char-by-char reveal"),
        ("bold_block", "Bold block — 2-3 word chunks, hard cut"),
        ("subtle", "Subtle — single line bottom, no animation"),
    ]


def caption_style_or_default(raw: dict[str, object] | None) -> CaptionStyle:
    """Coerce a stored caption_style dict (or None) into a CaptionStyle.

    Used by the brand-kit form + the renderer. Tolerates partial data —
    missing fields fall back to the karaoke_pop defaults rather than
    raising on malformed/old kits.
    """
    if not raw:
        return builtin_presets()["karaoke_pop"]
    try:
        return CaptionStyle.model_validate(raw)
    except Exception:
        # Stored dict has unknown fields or is malformed. Try to peel
        # the preset id and fall back to that preset's defaults.
        preset_id = raw.get("preset")
        if isinstance(preset_id, str):
            return _preset_by_id(preset_id)
        return builtin_presets()["karaoke_pop"]


def _preset_by_id(preset_id: str) -> CaptionStyle:
    """Lookup a preset by string id with karaoke_pop fallback.

    Wraps the dict access so callers can pass arbitrary strings (form
    input) without tripping mypy's Literal-key strictness on the
    `builtin_presets()` dict.
    """
    presets = builtin_presets()
    for known_id, style in presets.items():
        if known_id == preset_id:
            return style
    return presets["karaoke_pop"]
