"""apply_ai_fixes Fix 4 — banner variant recommendation is platform-aware.

Kick-branded variants (repost_page / green_block / black_bar_classic) only
get recommended for a KICK source; other stream platforms map the banner-ful
styles to `platform_band`. The logo-free minimal variant is platform-agnostic.
"""

from __future__ import annotations

from chalybclip.clip.ai_fixes import apply_ai_fixes


def _variant_for(*, style: str, platform: str) -> str | None:
    """Run the engine on a fresh (variant-unset) banner and return the
    variant it recommended, or None if it left the banner variant unset."""
    res = apply_ai_fixes(
        overlay_config={"clip_style": style, "banner": {"url": "aldo"}},
        source_platform=platform,
    )
    banner = res.new_overlay_config.get("banner")
    assert isinstance(banner, dict)
    return banner.get("variant")


def test_kick_keeps_kick_branded_variants() -> None:
    assert _variant_for(style="repost_page_viral", platform="kick") == "kick_repost_page"
    assert _variant_for(style="gaming_chaos", platform="kick") == "kick_green_block"
    assert _variant_for(style="documentary", platform="kick") == "kick_black_bar_classic"


def test_non_kick_stream_maps_bannerful_styles_to_platform_band() -> None:
    for platform in ("youtube", "twitch"):
        assert _variant_for(style="repost_page_viral", platform=platform) == "platform_band"
        assert _variant_for(style="gaming_chaos", platform=platform) == "platform_band"
        assert _variant_for(style="documentary", platform=platform) == "platform_band"


def test_minimal_styles_stay_platform_agnostic() -> None:
    # kick_minimal_url is logo-free — the burn's _banner_minimal_url already
    # gates the kick.com rewrite on platform, so it's safe everywhere.
    for platform in ("kick", "youtube", "twitch"):
        assert _variant_for(style="clean_creator", platform=platform) == "kick_minimal_url"
        assert _variant_for(style="minimal_native", platform=platform) == "kick_minimal_url"


def test_non_stream_source_gets_no_variant_recommendation() -> None:
    # Uploads / unknown → banner fixes are skipped entirely.
    assert _variant_for(style="repost_page_viral", platform="upload") is None
    assert _variant_for(style="repost_page_viral", platform="") is None


def test_deliberate_operator_variant_is_not_overridden() -> None:
    # A variant already set (operator pick) is left alone even if it
    # doesn't match the style/platform.
    res = apply_ai_fixes(
        overlay_config={
            "clip_style": "repost_page_viral",
            "banner": {"url": "aldo", "variant": "kick_minimal_url"},
        },
        source_platform="youtube",
    )
    banner = res.new_overlay_config.get("banner")
    assert isinstance(banner, dict)
    assert banner.get("variant") == "kick_minimal_url"
