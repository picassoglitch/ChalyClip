"""Brand kit resolution + asset management (voice-markers spec slice C).

The resolver picks the right brand kit per candidate / clip at render
time. Priority order:

    1. speaker.preferred_brand_kit_id (if the candidate's speaker is
       resolved AND that speaker has an explicit preferred kit)
    2. tenant's default brand kit (`brand_kits.is_default = True`)
    3. None — renderer falls back to system defaults (Pico styles,
       no logo, no custom captions)

A future Storage abstraction (docs/production_deploy.md §3) will swap
asset path resolution from local-filesystem to S3/R2 without touching
this module.
"""

from __future__ import annotations

from .captions import (
    CaptionPresetId,
    CaptionShadow,
    CaptionStyle,
    builtin_presets,
    caption_style_or_default,
    preset_choices,
)
from .clip_styles import (
    BUILTIN_STYLES,
    BannerVariantId,
    ClipStyle,
    ClipStyleId,
    TopHookConfig,
    get_clip_style,
    style_choices,
)
from .platform_presets import (
    PLATFORM_PRESETS,
    PlatformPreset,
    TargetPlatformId,
    get_platform_preset,
    platform_target_choices,
)
from .platform_zones import (
    CollisionWarning,
    OverlayRect,
    PlatformId,
    SafeZone,
    detect_collisions,
    platform_choices,
    zones_for_platform,
)
from .logo import (
    LogoSVG,
    generate_logo,
    is_rasterization_available,
    rasterize_svg_to_png,
    sanitize_svg,
)
from .service import (
    merged_trigger_phrases_for_speaker,
    outro_enabled_for_clip,
    resolve_brand_kit_for_candidate,
    resolve_brand_kit_for_speaker,
)

__all__ = [
    "BUILTIN_STYLES",
    "BannerVariantId",
    "CaptionPresetId",
    "CaptionShadow",
    "CaptionStyle",
    "ClipStyle",
    "ClipStyleId",
    "CollisionWarning",
    "LogoSVG",
    "OverlayRect",
    "PLATFORM_PRESETS",
    "PlatformId",
    "PlatformPreset",
    "SafeZone",
    "TargetPlatformId",
    "TopHookConfig",
    "builtin_presets",
    "caption_style_or_default",
    "detect_collisions",
    "generate_logo",
    "get_clip_style",
    "get_platform_preset",
    "is_rasterization_available",
    "merged_trigger_phrases_for_speaker",
    "outro_enabled_for_clip",
    "platform_choices",
    "platform_target_choices",
    "preset_choices",
    "rasterize_svg_to_png",
    "resolve_brand_kit_for_candidate",
    "resolve_brand_kit_for_speaker",
    "sanitize_svg",
    "style_choices",
    "zones_for_platform",
]
