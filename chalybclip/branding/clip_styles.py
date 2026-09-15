"""Clip Style presets — slice I.1.

A "Clip Style" is the keystone editor abstraction: a single named preset
the operator picks (e.g. "Repost Page Viral") that bundles every visual
decision (banner variant, top hook box, caption preset, animation
intensity, safezone behavior) into one click. Without this layer the
editor was 12+ unrelated dropdowns that the user had to coherently set.

Stored on `brand_kits.clip_style` so it survives across clips. The
existing per-clip overrides (`clips.overlay_config_json`) still win on
each individual field — the preset is the *default* the per-clip fields
are layered on top of.

Each `ClipStyle` instance is the source of truth for what the renderer
+ preview produce. The preview reads `data-clip-style="repost_page_viral"`
to drive CSS; the burn reads the same name to pick the banner filter
chain. Single mapping, both sides.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---- Type aliases ----

ClipStyleId = Literal[
    "repost_page_viral",
    "clean_creator",
    "gaming_chaos",
    "documentary",
    "minimal_native",
]
"""The 5 shipped presets. Adding a 6th means: add a Literal value, add a
preset to BUILTIN_STYLES, add a CSS hook in clip_detail.html, and (if
the burn differs from existing variants) add a renderer dispatch case
in overlay_burn.py."""


BannerVariantId = Literal[
    "kick_black_bar_classic",
    "kick_green_block",
    "kick_minimal_url",
    "kick_repost_page",
    "platform_band",
]
"""Source-credit banner styles. `kick_repost_page` is the spec's
recommended default for KICK clips — short black bottom bar, huge KICK
wordmark on the left, blocky URL center, optional LIVE NOW badge right.
`platform_band` is the sibling for the OTHER stream platforms (YouTube /
Twitch / X): a full-width bar in the platform's brand color with a white
monochrome glyph + the source URL. Matches the viral repost-page look
operators actually see in their feed."""


AnimationIntensity = Literal["low", "medium", "high", "max"]
"""How aggressive caption + element entrance animations are. Maps to
per-word scale bounce magnitude + emphasis modifier in the renderer."""


TopHookStyleId = Literal["white_rounded", "black_solid", "subtle"]
"""Top-of-frame headline box style. `white_rounded` is the viral
repost-page default (white bg, rounded corners, black bold text)."""


# ---- Pydantic models ----


class TopHookConfig(BaseModel):
    """Top hook title box — the white rounded headline above the
    subject's face. Optional; off by default in minimal styles."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    style: TopHookStyleId = "white_rounded"
    text: str = ""
    max_lines: int = Field(default=2, ge=1, le=3)


class ClipStyle(BaseModel):
    """One named preset bundling every visual decision."""

    model_config = ConfigDict(extra="forbid")

    id: ClipStyleId
    name: str
    description: str

    # Bottom banner ----------------------------------------------------
    banner_variant: BannerVariantId
    """Which Kick banner family to render."""
    banner_height_frac: float = Field(default=0.07, ge=0.0, le=0.25)
    """Banner height as a fraction of output_h. 0.0 = no banner (minimal_native).
    Repost-page chunky = 0.07; gaming-block tall = 0.08."""

    # Top hook box -----------------------------------------------------
    top_hook_default: TopHookConfig = Field(default_factory=TopHookConfig)

    # Captions ---------------------------------------------------------
    caption_preset_default: str = "karaoke_pop"
    caption_position_default: str = "upper_third"
    animation_intensity: AnimationIntensity = "medium"

    # Safe-zone behavior ----------------------------------------------
    safe_zone_platform_default: str = "reels_tiktok"


# ---- Built-in presets ----


BUILTIN_STYLES: dict[ClipStyleId, ClipStyle] = {
    "repost_page_viral": ClipStyle(
        id="repost_page_viral",
        name="Repost Page Viral",
        description=(
            "The default. Kick-style black bottom banner with huge logo + "
            "URL, white rounded headline up top, karaoke word-by-word "
            "captions, high animation. Looks like the viral Reels reposts "
            "operators actually see in their feed."
        ),
        banner_variant="kick_repost_page",
        banner_height_frac=0.07,
        top_hook_default=TopHookConfig(
            enabled=True,
            style="white_rounded",
            text="",
            max_lines=2,
        ),
        caption_preset_default="karaoke_pop",
        caption_position_default="upper_third",
        animation_intensity="high",
        safe_zone_platform_default="reels_tiktok",
    ),
    "clean_creator": ClipStyle(
        id="clean_creator",
        name="Clean Creator",
        description=(
            "Minimal URL only at the bottom, subtle captions, no top hook. "
            "For creators who want their face to be the moment."
        ),
        banner_variant="kick_minimal_url",
        banner_height_frac=0.04,
        top_hook_default=TopHookConfig(enabled=False),
        caption_preset_default="subtle",
        caption_position_default="lower_third",
        animation_intensity="low",
        safe_zone_platform_default="reels_tiktok",
    ),
    "gaming_chaos": ClipStyle(
        id="gaming_chaos",
        name="Gaming Chaos",
        description=(
            "Heavy emphasis, max animation, larger captions, green Kick "
            "block on the side. For raid moments, jumpscares, big plays."
        ),
        banner_variant="kick_green_block",
        banner_height_frac=0.08,
        top_hook_default=TopHookConfig(
            enabled=True, style="white_rounded", text=""
        ),
        caption_preset_default="bold_block",
        caption_position_default="centered",
        animation_intensity="max",
        safe_zone_platform_default="reels_tiktok",
    ),
    "documentary": ClipStyle(
        id="documentary",
        name="Documentary",
        description=(
            "Long-form story style. Smaller bottom banner, no top hook by "
            "default, typewriter captions, low animation."
        ),
        banner_variant="kick_black_bar_classic",
        banner_height_frac=0.05,
        top_hook_default=TopHookConfig(enabled=False),
        caption_preset_default="typewriter",
        caption_position_default="bottom",
        animation_intensity="low",
        safe_zone_platform_default="reels_tiktok",
    ),
    "minimal_native": ClipStyle(
        id="minimal_native",
        name="Minimal Native",
        description=(
            "No banner. No hook. Captions only. For native-feel posts that "
            "should look like phone recordings."
        ),
        banner_variant="kick_minimal_url",
        banner_height_frac=0.0,
        top_hook_default=TopHookConfig(enabled=False),
        caption_preset_default="subtle",
        caption_position_default="lower_third",
        animation_intensity="low",
        safe_zone_platform_default="reels_tiktok",
    ),
}


def get_clip_style(style_id: str | None) -> ClipStyle:
    """Lookup with a safe default — operators on a fresh install or a
    clip with `clip_style` unset see the recommended `repost_page_viral`."""
    if isinstance(style_id, str) and style_id in BUILTIN_STYLES:
        return BUILTIN_STYLES[style_id]  # type: ignore[literal-required,index,unused-ignore]
    return BUILTIN_STYLES["repost_page_viral"]


def style_choices() -> list[tuple[ClipStyleId, str, str]]:
    """For UI dropdown / preset cards: `(id, name, description)`."""
    return [(s.id, s.name, s.description) for s in BUILTIN_STYLES.values()]


__all__ = [
    "BUILTIN_STYLES",
    "AnimationIntensity",
    "BannerVariantId",
    "ClipStyle",
    "ClipStyleId",
    "TopHookConfig",
    "TopHookStyleId",
    "get_clip_style",
    "style_choices",
]
