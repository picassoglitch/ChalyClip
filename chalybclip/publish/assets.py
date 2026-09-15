"""Per-platform assets — every platform gets a different asset, not one clip.

The spec's Phase 3: instead of shipping the same `clip.mp4` everywhere, each
platform should receive its own filename, caption, hashtags, title, and first
frame so the platforms don't see identical uploads (which they down-rank as
cross-posted spam).

What this module owns is the *metadata* asset matrix — the part that's pure and
testable: per-platform caption (length-conformed + style-aware), hashtag set
(count-conformed), SEO title, first-comment, and a distinct filename. The video
re-encode / first-frame / thumbnail variants are produced by the render
pipeline; `PlatformAsset.render_preset` carries the platform key it should key
off (the branding `platform_presets` already vary crop/captions per platform).
"""
from __future__ import annotations

from dataclasses import dataclass

from chalybclip.publish.hub import FIRST_COMMENT_PLATFORMS
from chalybclip.publish.pacing import PlatformRule, canonical_platform, default_rule_for


@dataclass(frozen=True)
class PlatformAsset:
    platform: str
    caption: str
    hashtags: list[str]
    title: str | None
    first_comment: str | None
    filename: str
    render_preset: str
    caption_style: str

    def caption_with_tags(self) -> str:
        """Caption with hashtags appended (the form most platforms post)."""
        if not self.hashtags:
            return self.caption
        tag_line = " ".join(f"#{h.lstrip('#')}" for h in self.hashtags)
        return f"{self.caption}\n\n{tag_line}".strip()


def _platform_filename(clip_id: str, platform: str) -> str:
    """A distinct, stable filename per (clip, platform) so no two platforms
    upload a byte-identical file name."""
    safe = clip_id.replace("/", "_")
    return f"{safe}_{platform}.mp4"


def _platform_title(
    base_title: str | None, hook: str, caption: str, rule: PlatformRule
) -> str | None:
    """The title field, where the platform uses one (YouTube). Prefer an
    explicit title, then the hook, then the caption's first line — trimmed to
    the platform's caption band so an SEO title isn't a wall of text."""
    if rule.platform != "youtube":
        # Only YouTube consumes a separate title in our payload today; others
        # carry everything in the caption.
        return base_title
    # NB: the ternary binds the whole `or` chain, so without parentheses an
    # empty caption discarded an existing base_title/hook.
    raw = (base_title or hook or (caption.splitlines()[0] if caption else "")) or ""
    raw = raw.strip()
    if not raw:
        return base_title
    return rule.clamp_caption(raw)


def build_platform_assets(
    *,
    clip_id: str,
    base_caption: str,
    base_hashtags: list[str],
    base_title: str | None,
    hook: str,
    platforms: list[str],
    rules: dict[str, PlatformRule] | None = None,
    first_comment: str | None = None,
) -> dict[str, PlatformAsset]:
    """Build the per-platform asset matrix from one clip's base content.

    Each platform's caption is clamped to its length band and its hashtags to
    its count band (Phase 1 rules), the title is platform-appropriate, the
    first-comment lands only where the platform supports it, and the filename
    is distinct. Returns `{platform: PlatformAsset}` keyed by canonical id.
    """
    rules = rules or {}
    out: dict[str, PlatformAsset] = {}
    for p in platforms:
        key = canonical_platform(p)
        if key in out:
            continue
        rule = rules.get(key) or default_rule_for(key)
        hashtags = rule.clamp_hashtags(base_hashtags)
        # The caption band applies to the POSTED content — which is
        # `caption_with_tags()`, caption + "\n\n" + tag line. Reserve the
        # tag line's length before clamping the prose, otherwise a
        # 280-clamped caption plus appended tags overflows X's hard cap
        # and the vendor rejects the post. If the tags alone would eat
        # the whole band, drop trailing tags until prose fits.
        tag_line = " ".join(f"#{h.lstrip('#')}" for h in hashtags)
        while hashtags and len(tag_line) + 2 >= rule.caption_max_chars:
            hashtags = hashtags[:-1]
            tag_line = " ".join(f"#{h.lstrip('#')}" for h in hashtags)
        reserve = (len(tag_line) + 2) if tag_line else 0
        caption = rule.clamp_caption(base_caption, max_chars=rule.caption_max_chars - reserve)
        title = _platform_title(base_title, hook, base_caption, rule)
        fc = first_comment if key in FIRST_COMMENT_PLATFORMS else None
        out[key] = PlatformAsset(
            platform=key,
            caption=caption,
            hashtags=hashtags,
            title=title,
            first_comment=fc,
            filename=_platform_filename(clip_id, key),
            render_preset=key,
            caption_style=rule.caption_style,
        )
    return out
