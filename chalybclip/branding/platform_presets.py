"""Target-platform auto-config bundles — slice K.5.

The operator picks WHERE they're posting (TikTok / Reels / Shorts /
Kick / All); the editor writes every downstream setting to match.

Each preset bundles:
  - safe_zone_platform      (which UI band catalog to test against)
  - fit_mode                (smart_crop / blurred_bg / letterbox / dynamic_zoom)
  - captions_position       (upper_third / centered / lower_third / bottom)
  - captions_font_size      (small / medium / large / xl)
  - captions_animation      (pop / slide / typewriter / fade)
  - captions_lead_ms        (0-500)
  - banner_variant          (kick_repost_page / kick_black_bar_classic /
                             kick_green_block / kick_minimal_url)
  - banner_live_badge       (bool)
  - top_hook_enabled        (bool)
  - top_hook_style          (white_rounded / black_solid / subtle)
  - clip_style              (the existing Clip Style preset id —
                             Repost Page Viral / Clean Creator / etc)

Sourced from publicly-observable creator-tool conventions on each
platform — not from a research database. Treat as defaults the
operator can override; tune by experiment.

Cross-platform "all" picks the safest intersection so the same MP4
posts cleanly on every short-form feed at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TargetPlatformId = Literal["tiktok", "reels", "shorts", "kick", "all"]


@dataclass(frozen=True)
class PlatformPreset:
    """Every editor setting picked for one target platform."""

    id: TargetPlatformId
    name: str
    emoji: str
    blurb: str
    # Safety + framing
    safe_zone_platform: str
    fit_mode: str
    # Captions
    captions_position: str
    captions_font_size: str
    captions_animation: str
    captions_lead_ms: int
    # Banner / hook
    banner_variant: str
    banner_enabled: bool
    banner_live_badge: bool
    top_hook_enabled: bool
    top_hook_style: str
    # Clip Style preset (cascades down to whatever it owns)
    clip_style: str


# ---- the shipped presets ----

PLATFORM_PRESETS: dict[TargetPlatformId, PlatformPreset] = {
    "tiktok": PlatformPreset(
        id="tiktok",
        name="TikTok",
        emoji="🎵",
        blurb="Full subject · blurred backdrop · captions lower-middle",
        safe_zone_platform="tiktok",
        # Slice K.7 — flipped from smart_crop → blurred_bg. The naive
        # 16:9 → 9:16 cover-crop was producing the half-face super-
        # zoom operator complained about. blurred_bg keeps the full
        # horizontal frame visible (subject centered, no eye-level
        # crop) and fills the surrounding 9:16 area with a blurred
        # backdrop, matching what top creators (e.g. Clavicular on
        # @reelonkick) actually ship.
        fit_mode="blurred_bg",
        # Slice L.4 — captions moved from upper_third → lower_third.
        # The spec calls for the lower-middle (~65-72%) sweet spot:
        # below the face / reaction, above the description + dock.
        # upper_third was competing with the main viral hook at 18%.
        captions_position="lower_third",
        captions_font_size="large",
        captions_animation="pop",
        captions_lead_ms=120,
        # Slice K.7 — bumped from kick_minimal_url to kick_repost_page
        # so the operator gets the big black-bar + green KICK + URL
        # combo by default. Minimal_url was too understated.
        banner_variant="kick_repost_page",
        banner_enabled=True,
        banner_live_badge=False,
        top_hook_enabled=True,
        top_hook_style="white_rounded",
        clip_style="repost_page_viral",
    ),
    "reels": PlatformPreset(
        id="reels",
        name="Reels",
        emoji="📸",
        blurb="Full subject · starfield backdrop · captions lower-middle",
        safe_zone_platform="reels",
        # Slice K.7 — same reasoning as tiktok. Reels in particular
        # frequently ships with a "starfield" backdrop on portrait
        # video; blurred_bg is the closest CSS approximation.
        fit_mode="blurred_bg",
        # Slice L.4 — captions to lower-middle per the spec.
        captions_position="lower_third",
        captions_font_size="large",
        captions_animation="pop",
        captions_lead_ms=100,
        banner_variant="kick_repost_page",
        banner_enabled=True,
        banner_live_badge=False,
        top_hook_enabled=True,
        top_hook_style="white_rounded",
        clip_style="repost_page_viral",
    ),
    "shorts": PlatformPreset(
        id="shorts",
        name="Shorts",
        emoji="▶",
        blurb="Full subject · lower-middle captions · clean KICK banner",
        safe_zone_platform="shorts",
        # Slice K.7 — same fit-mode flip as TikTok/Reels.
        fit_mode="blurred_bg",
        # Slice L.4 — was "centered" (50%); moved to lower_third
        # (68%) per the spec. Centered overlapped the face on most
        # Kick streamer clips.
        captions_position="lower_third",
        captions_font_size="medium",
        captions_animation="slide",
        captions_lead_ms=80,
        # Slice K.7 — repost page on Shorts too. The viewer brain has
        # learned what a Kick repost looks like; minimal_url asks
        # them to do the visual work themselves.
        banner_variant="kick_repost_page",
        banner_enabled=True,
        banner_live_badge=False,
        top_hook_enabled=True,
        top_hook_style="white_rounded",
        clip_style="repost_page_viral",
    ),
    "kick": PlatformPreset(
        id="kick",
        name="Kick",
        emoji="💚",
        blurb="Native streamer banner · LIVE pill · letterbox-safe",
        safe_zone_platform="kick",
        fit_mode="letterbox",
        captions_position="bottom",
        captions_font_size="medium",
        captions_animation="pop",
        captions_lead_ms=120,
        banner_variant="kick_repost_page",
        banner_enabled=True,
        banner_live_badge=True,
        top_hook_enabled=False,
        top_hook_style="white_rounded",
        clip_style="repost_page_viral",
    ),
    "all": PlatformPreset(
        id="all",
        name="All platforms",
        emoji="🌐",
        blurb="Safest cross-platform export — full subject on every feed",
        # Strictest safe zone = TikTok (covers Reels + Shorts implicitly)
        safe_zone_platform="tiktok",
        # Slice K.7 — blurred_bg as the cross-platform default. The
        # full-subject framing reads correctly everywhere; smart_crop
        # routinely chopped faces in half on operator-reported clips.
        fit_mode="blurred_bg",
        # Slice L.4 — lower-middle captions match the spec across
        # every short-form feed.
        captions_position="lower_third",
        captions_font_size="medium",
        captions_animation="pop",
        captions_lead_ms=120,
        # Slice K.7 — repost-page banner = the prominent KICK + URL
        # bar that real @reelonkick clips ship with; far more visible
        # than the minimal URL pill we used to default to.
        banner_variant="kick_repost_page",
        banner_enabled=True,
        banner_live_badge=False,
        top_hook_enabled=True,
        top_hook_style="white_rounded",
        clip_style="repost_page_viral",
    ),
}


def get_platform_preset(target_id: str | None) -> PlatformPreset:
    """Lookup with `all` as the safe default — operators on a fresh
    install or a clip with `target_platform` unset see the cross-
    platform preset, which works everywhere."""
    if isinstance(target_id, str) and target_id in PLATFORM_PRESETS:
        return PLATFORM_PRESETS[target_id]  # type: ignore[literal-required,index,unused-ignore]
    return PLATFORM_PRESETS["all"]


def platform_target_choices() -> list[tuple[TargetPlatformId, str, str, str]]:
    """For UI chip row: `(id, emoji, name, blurb)`."""
    return [
        (p.id, p.emoji, p.name, p.blurb)
        for p in PLATFORM_PRESETS.values()
    ]


__all__ = [
    "PLATFORM_PRESETS",
    "PlatformPreset",
    "TargetPlatformId",
    "get_platform_preset",
    "platform_target_choices",
]
