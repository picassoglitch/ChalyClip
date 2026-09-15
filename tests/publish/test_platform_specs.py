"""Per-platform publish spec helpers — caption fitting + duration gating."""

from __future__ import annotations

from chalybclip.publish.platform_specs import (
    DEFAULT_SPEC,
    PLATFORM_SPECS,
    fit_caption,
    fits_duration,
    per_platform_caption_overrides,
    spec_for,
)


def test_spec_for_known_and_unknown() -> None:
    assert spec_for("TikTok").caption_limit == 2200
    assert spec_for("x_unknown") is DEFAULT_SPEC
    assert spec_for("") is DEFAULT_SPEC


def test_fits_duration_per_platform() -> None:
    # Bluesky caps at 60s, IG/FB at 90s, YT Shorts at 180s.
    assert fits_duration("bluesky", 45.0) is True
    assert fits_duration("bluesky", 75.0) is False
    assert fits_duration("instagram", 75.0) is True
    assert fits_duration("instagram", 95.0) is False
    assert fits_duration("youtube", 120.0) is True
    assert fits_duration("tiktok", 120.0) is True


def test_fits_duration_allows_unknown_length() -> None:
    # No length to judge -> never block.
    assert fits_duration("bluesky", None) is True
    assert fits_duration("bluesky", 0.0) is True


def test_fit_caption_truncates_to_x_limit_at_word_boundary() -> None:
    long = "word " * 100  # 500 chars
    out = fit_caption(long, "twitter")  # 280-char cap
    assert len(out) <= 280
    assert not out.endswith("wor")  # didn't cut mid-word
    assert out  # non-empty


def test_fit_caption_noop_when_within_limit() -> None:
    short = "just a short caption #clip"
    assert fit_caption(short, "tiktok") == short
    assert fit_caption(short, "twitter") == short


def test_fit_caption_caps_hashtags_for_instagram() -> None:
    body = "great moment"
    tags = " ".join(f"#tag{i}" for i in range(40))
    out = fit_caption(f"{body} {tags}", "instagram")  # cap 30
    assert out.count("#") == 30
    assert out.startswith(body)


def test_fit_caption_empty_safe() -> None:
    assert fit_caption("", "twitter") == ""
    assert fit_caption(None, "twitter") == ""


def test_per_platform_overrides_only_includes_changed() -> None:
    caption = "x" * 400  # over X's 280, under TikTok's 2200
    overrides = per_platform_caption_overrides(caption, ["tiktok", "twitter"])
    # TikTok takes it as-is (not in overrides); X is truncated (included).
    assert "tiktok" not in overrides
    assert "twitter" in overrides
    assert len(overrides["twitter"]) <= 280


def test_per_platform_overrides_empty_when_all_fit() -> None:
    caption = "short enough for everyone"
    assert per_platform_caption_overrides(caption, ["tiktok", "twitter", "bluesky"]) == {}


def test_every_supported_platform_has_a_spec() -> None:
    # The Zernio-supported clip targets must each resolve to a real spec
    # (not the fallback) so nothing silently uses default limits.
    for p in ("tiktok", "youtube", "instagram", "facebook", "twitter",
              "linkedin", "threads", "pinterest", "bluesky"):
        assert p in PLATFORM_SPECS, p
