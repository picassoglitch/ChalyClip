"""Platform glyph rasterizer — the white monochrome marks the branded
`platform_band` source banner composites (YouTube / Twitch / X)."""

from __future__ import annotations

from chalybclip.clip.platform_glyph_png import platform_glyph_png_path


def test_supported_platforms_render_a_png() -> None:
    from PIL import Image

    for platform in ("youtube", "twitch", "x", "twitter"):
        path = platform_glyph_png_path(platform)
        assert path is not None, platform
        assert path.exists()
        assert path.suffix == ".png"
        img = Image.open(path)
        # Transparent RGBA canvas with actual opaque (white) pixels drawn.
        assert img.mode == "RGBA"
        alpha_extrema = img.getchannel("A").getextrema()
        assert alpha_extrema[1] > 0, f"{platform} glyph is fully transparent"


def test_case_and_whitespace_insensitive() -> None:
    assert platform_glyph_png_path("  YouTube ") == platform_glyph_png_path("youtube")


def test_twitter_aliases_x() -> None:
    # Same mark for both handles on the X/Twitter brand.
    assert platform_glyph_png_path("twitter") == platform_glyph_png_path("x")


def test_platforms_without_a_glyph_return_none() -> None:
    # Kick uses its full wordmark (kick_logo_png), not a glyph; uploads /
    # unknown sources get no logo overlay at all.
    assert platform_glyph_png_path("kick") is None
    assert platform_glyph_png_path("upload") is None
    assert platform_glyph_png_path("") is None
    assert platform_glyph_png_path("instagram") is None


def test_idempotent_same_path() -> None:
    a = platform_glyph_png_path("twitch")
    b = platform_glyph_png_path("twitch")
    assert a == b
