"""Default burn-in overlay config for auto-generated clips.

A clip that's never opened in the editor used to ship with `overlay_config =
NULL`, so auto-published clips came out plain — no hook card, no source-credit
"marca" banner (captions defaulted on, but that was the only thing). This builds
a sensible viral default so auto-publish / auto-program clips look like the
manually-edited ones: captions on, the generated hook as the top headline, and
a source-credit banner carrying the original streamer's name, styled for the
platform the VOD came from.

Pure + deterministic so the pipeline can stamp it and tests can pin it. The
operator can still override any of it in the editor (which overwrites the
whole `overlay_config`).
"""
from __future__ import annotations

from typing import Any

# Source platform → how the source-credit handle reads. Kick gets the full
# repost-page banner (wordmark + KICK.COM/handle); the other stream platforms
# get their own branded band (`platform_band` — brand color + a white
# monochrome glyph + the platform-correct domain) so we never stamp a Kick
# logo on a YouTube/Twitch/X clip. Uploads / unknown sources fall to the
# logo-free `kick_minimal_url` text pill.
_SOURCE_DOMAINS = {
    "youtube": "youtube.com/@{h}",
    "twitch": "twitch.tv/{h}",
    "twitter": "x.com/{h}",
    "x": "x.com/{h}",
}


def source_banner(platform: str | None, channel: str | None) -> dict[str, Any] | None:
    """Banner config crediting the SOURCE streamer, or None when there's no
    channel to credit (e.g. a raw upload with no known creator)."""
    handle = (channel or "").strip().lstrip("@").strip("/").strip()
    if not handle:
        return None
    p = (platform or "").strip().lower()
    if p == "kick":
        # Full Kick repost-page chrome (wordmark + KICK.COM/handle).
        return {
            "enabled": True, "platform": "kick",
            "variant": "kick_repost_page", "url": handle,
        }
    domain = _SOURCE_DOMAINS.get(p)
    if domain:
        # Branded band — platform brand color + white glyph + domain URL.
        return {
            "enabled": True, "platform": p,
            "variant": "platform_band", "url": domain.format(h=handle),
        }
    # Unknown source (upload / unknown) → minimal text, just the handle.
    return {
        "enabled": True, "platform": p or "kick",
        "variant": "kick_minimal_url", "url": handle,
    }


def default_overlay_config(
    *,
    platform: str | None = None,
    channel: str | None = None,
    hook: str = "",
) -> dict[str, Any]:
    """The viral default for a freshly-generated clip: captions on, the hook as
    the top headline box, and a source-credit banner when we know the channel."""
    cfg: dict[str, Any] = {
        "clip_style": "repost_page_viral",
        "captions": {"enabled": True, "position": "lower_third", "font_size": "medium"},
    }
    hook = (hook or "").strip()
    if hook:
        cfg["top_hook"] = {"enabled": True, "text": hook, "style": "white_rounded"}
    banner = source_banner(platform, channel)
    if banner:
        cfg["banner"] = banner
    return cfg
