"""Per-platform publish specs — the single source of truth for what each
target will actually accept, applied right before we hand a post to the
vendor (Zernio).

Today the publish path sends the same caption + clip to every platform.
That silently breaks on the platforms with a SHORT caption cap (X is 280
chars, Threads 500, Bluesky ~300) — a 2,200-char TikTok-style caption is
rejected or ugly-truncated by the vendor — and would send an over-long clip
to a surface that won't take it (Bluesky maxes at 60s, IG/FB Reels at 90s).

`fit_caption` truncates per platform; `fits_duration` gates a target by the
clip length. Both are pure functions so the call sites (manual publish +
hands-free auto-publish) stay thin.

Numbers are the STABLE specs (durations, aspect, caption/hashtag limits) —
verified mid-2026. The volatile part (monetization eligibility thresholds)
lives in product docs, NOT here: this file only encodes what makes a post
land. Conservative where a platform is fuzzy; widen a cap only with a source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HASHTAG_RE = re.compile(r"#\w+")


@dataclass(frozen=True)
class PlatformSpec:
    """What one platform's short-form surface accepts.

    max_duration_s: hard ceiling for the monetizable short surface (over it
        the post is rejected or demoted to a non-short — we skip the target).
    caption_limit: max caption/description characters the platform stores.
    hashtag_limit: max hashtags that count (over it some platforms drop ALL).
    """

    max_duration_s: float
    caption_limit: int
    hashtag_limit: int


# Keyed by the platform id Zernio uses (note: `twitter`, not `x`). Values are
# deliberately conservative — the goal is "the post always lands", not
# "squeeze the absolute max the platform allows".
PLATFORM_SPECS: dict[str, PlatformSpec] = {
    # TikTok: up to 10 min; caption expanded to 2,200.
    "tiktok": PlatformSpec(max_duration_s=600.0, caption_limit=2200, hashtag_limit=8),
    # YouTube Shorts: <=3 min counts as a Short; description 5,000; >15
    # hashtags makes YouTube ignore ALL of them.
    "youtube": PlatformSpec(max_duration_s=180.0, caption_limit=5000, hashtag_limit=15),
    # Instagram Reels: up to 90s; caption 2,200; hard cap 30 hashtags.
    "instagram": PlatformSpec(max_duration_s=90.0, caption_limit=2200, hashtag_limit=30),
    # Facebook Reels: up to 90s; generous caption.
    "facebook": PlatformSpec(max_duration_s=90.0, caption_limit=2200, hashtag_limit=30),
    # X / Twitter: video ~2:20; caption 280 (non-Premium) — the binding cap.
    "twitter": PlatformSpec(max_duration_s=140.0, caption_limit=280, hashtag_limit=5),
    # LinkedIn: video up to 15 min; ~3,000 char post.
    "linkedin": PlatformSpec(max_duration_s=600.0, caption_limit=3000, hashtag_limit=10),
    # Threads: video up to 5 min; 500-char post.
    "threads": PlatformSpec(max_duration_s=300.0, caption_limit=500, hashtag_limit=10),
    # Pinterest: video pins; ~500-char description.
    "pinterest": PlatformSpec(max_duration_s=900.0, caption_limit=500, hashtag_limit=20),
    # Bluesky: short video, ~300-char post — the tightest caption + duration.
    "bluesky": PlatformSpec(max_duration_s=60.0, caption_limit=300, hashtag_limit=8),
}

# Fallback for an unknown platform id — permissive enough not to mangle a
# caption, tight enough to stay sane (matches TikTok-ish limits).
DEFAULT_SPEC = PlatformSpec(max_duration_s=600.0, caption_limit=2200, hashtag_limit=30)


def spec_for(platform: str) -> PlatformSpec:
    """The spec for a platform id (case-insensitive), or DEFAULT_SPEC."""
    return PLATFORM_SPECS.get((platform or "").strip().lower(), DEFAULT_SPEC)


def fits_duration(platform: str, duration_s: float | None) -> bool:
    """True when a clip of `duration_s` is within the platform's ceiling.

    Unknown duration (None / <=0) is allowed through — we don't have a
    length to judge, and blocking on a missing value would drop good posts.
    """
    if duration_s is None or duration_s <= 0:
        return True
    return duration_s <= spec_for(platform).max_duration_s


def _cap_hashtags(text: str, limit: int) -> str:
    """Drop hashtags past `limit`. Cuts the string at the start of the first
    surplus hashtag — hashtags are conventionally a trailing block, so this
    keeps the body and the kept tags intact without reflowing whitespace."""
    matches = list(_HASHTAG_RE.finditer(text))
    if len(matches) <= limit:
        return text
    return text[: matches[limit].start()].rstrip()


def _truncate(text: str, limit: int) -> str:
    """Hard-cap `text` to `limit` chars, backing off to a word/line boundary
    so we never cut mid-word or mid-hashtag (unless the first token alone
    already exceeds the limit)."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = max(cut.rfind(" "), cut.rfind("\n"))
    # Only honor the boundary if it doesn't throw away most of the budget.
    if boundary >= int(limit * 0.6):
        cut = cut[:boundary]
    return cut.rstrip()


def fit_caption(caption: str | None, platform: str) -> str:
    """Make `caption` compliant for `platform`: cap hashtag count, then
    truncate to the caption-length limit at a word boundary. Idempotent and
    safe on empty input."""
    text = caption or ""
    if not text:
        return text
    spec = spec_for(platform)
    text = _cap_hashtags(text, spec.hashtag_limit)
    return _truncate(text, spec.caption_limit)


def per_platform_caption_overrides(
    caption: str | None, platforms: list[str]
) -> dict[str, str]:
    """Build the `custom_content` map (platform -> fitted caption) for the
    vendor's createPost. Only includes a platform when its fitted caption
    DIFFERS from the base `caption` — platforms that take the caption as-is
    are left out so they inherit the root `content` unchanged.
    """
    base = caption or ""
    out: dict[str, str] = {}
    for p in platforms:
        key = (p or "").strip().lower()
        fitted = fit_caption(base, key)
        if fitted != base:
            out[key] = fitted
    return out
