"""Default viral overlay config for auto-generated clips."""

from __future__ import annotations

from chalybclip.clip.overlay_defaults import default_overlay_config, source_banner


def test_captions_and_hook_on_by_default() -> None:
    cfg = default_overlay_config(platform="kick", channel="aldo", hook="INSANE clutch")
    assert cfg["captions"]["enabled"] is True
    assert cfg["top_hook"]["enabled"] is True
    assert cfg["top_hook"]["text"] == "INSANE clutch"
    assert cfg["clip_style"] == "repost_page_viral"


def test_no_hook_when_empty() -> None:
    cfg = default_overlay_config(platform="kick", channel="aldo", hook="")
    assert "top_hook" not in cfg  # no empty hook box
    assert cfg["captions"]["enabled"] is True


def test_kick_source_gets_repost_page_banner() -> None:
    b = source_banner("kick", "AldoStream")
    assert b == {
        "enabled": True, "platform": "kick",
        "variant": "kick_repost_page", "url": "AldoStream",
    }


def test_youtube_source_credits_youtube_not_kick() -> None:
    b = source_banner("youtube", "@SomeStreamer")
    assert b["platform"] == "youtube"
    # Branded band (platform color + white glyph), never Kick chrome.
    assert b["variant"] == "platform_band"
    assert b["url"] == "youtube.com/@SomeStreamer"


def test_twitch_and_x_domains() -> None:
    tw = source_banner("twitch", "ninja")
    assert tw["variant"] == "platform_band"
    assert tw["url"] == "twitch.tv/ninja"
    x = source_banner("x", "elonmusk")
    assert x["variant"] == "platform_band"
    assert x["url"] == "x.com/elonmusk"


def test_no_channel_no_banner() -> None:
    assert source_banner("kick", "") is None
    assert source_banner("youtube", None) is None
    cfg = default_overlay_config(platform="upload", channel=None, hook="hi")
    assert "banner" not in cfg


def test_unknown_platform_uses_minimal_handle() -> None:
    # Upload / unknown has no domain + no glyph → logo-free text pill.
    b = source_banner("upload", "creator")
    assert b["variant"] == "kick_minimal_url"
    assert b["url"] == "creator"
