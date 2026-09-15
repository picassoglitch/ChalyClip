"""Per-platform publishing rulebook — optimize BEFORE publishing.

The growth thesis: account health is the top priority, above impressions,
watch time, and engagement. You protect it by never out-pacing what each
platform's algorithm reads as natural posting behaviour. So every platform
gets a hard rulebook — a daily ceiling, a minimum gap between posts, a
caption-length band, a hashtag count band, and a randomized delay so the
cadence never looks botted.

These defaults are intentionally conservative (the spec's lower-to-mid end of
each range). A tenant can override any of them per platform from the Publish
Center; `PlatformPacingRulesRepo` persists those overrides and falls back to
`DEFAULT_PLATFORM_RULES` for anything unset. Everything here is pure and
deterministic (the jitter takes an injected RNG) so the scheduler is testable.
"""
from __future__ import annotations

import datetime as _dt
import random as _random
from collections.abc import Iterable

from pydantic import BaseModel, Field

# Canonical platform ids match the lowercased Zernio `account.platform`
# values used everywhere else (see hub.FIRST_COMMENT_PLATFORMS). The spec
# writes "X"; Zernio (and our payload builder) calls it "twitter", so we
# alias it. Aliases normalize caller input before a rule lookup.
PLATFORM_ALIASES = {
    "x": "twitter",
    "twitter/x": "twitter",
    "reels": "instagram",
    "instagram_reels": "instagram",
    "shorts": "youtube",
    "youtube_shorts": "youtube",
    "fb": "facebook",
    "facebook_reels": "facebook",
    "ig": "instagram",
    "yt": "youtube",
    "li": "linkedin",
}


def canonical_platform(platform: str) -> str:
    """Normalize a platform id (alias, casing, whitespace) to its canonical key."""
    key = (platform or "").strip().lower()
    return PLATFORM_ALIASES.get(key, key)


class PlatformRule(BaseModel):
    """The hard rulebook for one platform. All knobs are per-platform-per-day
    where a day means a UTC calendar day (matching the existing cap accounting
    in `HubPublishJobsRepo.count_for_day`)."""

    platform: str
    max_per_day: int = Field(ge=0, description="Hard ceiling of posts per UTC day.")
    min_gap_minutes: int = Field(
        ge=0, description="Minimum spacing between two posts on this platform."
    )
    caption_min_chars: int = Field(ge=0, default=0)
    caption_max_chars: int = Field(gt=0, default=2200)
    hashtag_min: int = Field(ge=0, default=0)
    hashtag_max: int = Field(ge=0, default=30)
    jitter_minutes: int = Field(
        ge=0, default=0, description="Scheduled times are nudged ± this many minutes."
    )
    # The caption *style* the per-platform asset generator should aim for. Not
    # enforced as a number — it steers Phase-3 caption rewriting.
    caption_style: str = Field(default="default")
    enabled: bool = True

    def clamp_hashtags(self, hashtags: Iterable[str]) -> list[str]:
        """Trim a hashtag list to this platform's max (keep the first N — they
        are assumed already ranked by relevance). Min is advisory: we can't
        invent tags, so we never pad, only cap."""
        tags = [h for h in hashtags if h and h.strip()]
        if self.hashtag_max <= 0:
            return []
        return tags[: self.hashtag_max]

    def clamp_caption(self, caption: str, *, max_chars: int | None = None) -> str:
        """Trim a caption to this platform's max length, on a word boundary
        when one is close to the cut so we don't slice a word in half.
        `max_chars` overrides the platform band for callers that must
        reserve room inside it (the asset matrix reserves the appended
        hashtag line's length)."""
        limit = self.caption_max_chars if max_chars is None else max(0, max_chars)
        text = (caption or "").strip()
        if len(text) <= limit:
            return text
        cut = text[:limit].rstrip()
        # Prefer the last whitespace if it isn't far back (avoid a tiny tail).
        sp = cut.rfind(" ")
        if sp >= limit - 24:
            cut = cut[:sp].rstrip()
        return cut


# The shipped rulebook. Values are the conservative end of the spec's ranges:
# fewer posts, wider gaps — health first. Caption bands use the spec's numbers
# where given (TikTok 80-150) and platform-sane defaults elsewhere.
DEFAULT_PLATFORM_RULES: dict[str, PlatformRule] = {
    "tiktok": PlatformRule(
        platform="tiktok", max_per_day=5, min_gap_minutes=180,
        caption_min_chars=80, caption_max_chars=150,
        hashtag_min=3, hashtag_max=5, jitter_minutes=20, caption_style="punchy",
    ),
    "instagram": PlatformRule(
        platform="instagram", max_per_day=4, min_gap_minutes=240,
        caption_min_chars=60, caption_max_chars=300,
        hashtag_min=3, hashtag_max=5, jitter_minutes=25, caption_style="question_cta",
    ),
    "youtube": PlatformRule(
        platform="youtube", max_per_day=3, min_gap_minutes=300,
        caption_min_chars=40, caption_max_chars=100,
        hashtag_min=2, hashtag_max=3, jitter_minutes=30, caption_style="seo_title",
    ),
    "facebook": PlatformRule(
        platform="facebook", max_per_day=6, min_gap_minutes=150,
        caption_min_chars=60, caption_max_chars=400,
        hashtag_min=2, hashtag_max=4, jitter_minutes=20, caption_style="story",
    ),
    "twitter": PlatformRule(
        platform="twitter", max_per_day=12, min_gap_minutes=40,
        caption_min_chars=0, caption_max_chars=280,
        hashtag_min=0, hashtag_max=2, jitter_minutes=10, caption_style="thread",
    ),
    "linkedin": PlatformRule(
        platform="linkedin", max_per_day=1, min_gap_minutes=720,
        caption_min_chars=80, caption_max_chars=1200,
        hashtag_min=0, hashtag_max=0, jitter_minutes=0, caption_style="story",
    ),
    "pinterest": PlatformRule(
        platform="pinterest", max_per_day=7, min_gap_minutes=60,
        caption_min_chars=40, caption_max_chars=300,
        hashtag_min=5, hashtag_max=5, jitter_minutes=15, caption_style="seo",
    ),
}

# Fallback for any platform without an explicit rule (a new Zernio target we
# haven't tuned). Deliberately cautious.
_GENERIC_RULE = PlatformRule(
    platform="_generic", max_per_day=3, min_gap_minutes=180,
    caption_max_chars=600, hashtag_max=5, jitter_minutes=20,
)


def default_rule_for(platform: str) -> PlatformRule:
    """The shipped default rule for a platform (generic fallback if untuned)."""
    key = canonical_platform(platform)
    base = DEFAULT_PLATFORM_RULES.get(key)
    if base is None:
        return _GENERIC_RULE.model_copy(update={"platform": key or "_generic"})
    return base


def jitter_delta(
    rule: PlatformRule, *, rng: _random.Random | None = None
) -> _dt.timedelta:
    """A random ± offset within the platform's jitter window. Pass a seeded
    `rng` for deterministic output (tests, reproducible schedules)."""
    if rule.jitter_minutes <= 0:
        return _dt.timedelta()
    r = rng or _random
    minutes = r.uniform(-rule.jitter_minutes, rule.jitter_minutes)
    return _dt.timedelta(minutes=minutes)


def plan_platform_times(
    count: int,
    *,
    rule: PlatformRule,
    now: _dt.datetime,
    existing_today: int = 0,
    existing_by_day: dict[str, int] | None = None,
    rng: _random.Random | None = None,
) -> list[_dt.datetime]:
    """Schedule `count` posts on ONE platform under its hard rulebook.

    Two rules bound the timeline, in this order of authority:

      * `max_per_day` — at most this many posts land on any UTC day. Day 0
        already carries `existing_today` posts, so its remaining capacity is
        `max_per_day - existing_today`. When a day fills, the next post rolls
        to the start of the following day's posting window.
      * `min_gap_minutes` — consecutive posts are at least this far apart. The
        first post fires ~`now` (clamped just past it) when the platform has no
        posts yet today, since the gap spaces consecutive posts, not the first;
        if `existing_today > 0` the first stays a gap out to be safe.

    `existing_by_day` maps 'YYYY-MM-DD' → posts a PREVIOUS planning run already
    placed on that day. Without it, a second sweep the same day (a new VOD)
    rolled its own `max_per_day` worth of overflow onto days that were already
    full — stacking twice the cap on every future day. Today's count still comes
    from `existing_today` (the authoritative today number the callers already
    compute); `existing_by_day` seeds every OTHER day the timeline touches.

    Each slot is then nudged by `jitter_delta` so the cadence reads as human.
    Jitter never reorders posts and never pulls one before `now`. Returns the
    sorted list; `[]` when `count <= 0` or the platform's daily cap is 0.

    Deterministic given a seeded `rng`, so the schedule is testable.
    """
    if count <= 0 or rule.max_per_day <= 0:
        return []
    r = rng or _random
    by_day = existing_by_day or {}

    def _seed(d: _dt.date) -> int:
        # `now`'s own day defers to existing_today (never double-count a
        # caller that included today in both); other days pull their prior
        # sweeps' load from the by-day map.
        if d == now.date():
            return existing_today
        return max(0, int(by_day.get(d.isoformat(), 0)))

    def _jit() -> _dt.timedelta:
        # Non-negative jitter (0..jitter_minutes) ADDED to each gap, so the gap
        # only ever grows — `min_gap` stays a true floor between consecutive
        # posts, never shrunk (independent ± jitter could push two posts closer
        # than min_gap, violating the safeguard).
        return _dt.timedelta(minutes=r.uniform(0, max(0, rule.jitter_minutes)))

    gap = _dt.timedelta(minutes=max(1, rule.min_gap_minutes))
    out: list[_dt.datetime] = []
    # First post fires ~now (+small jitter) when nothing has gone out today —
    # min_gap spaces CONSECUTIVE posts, it shouldn't delay the first. When the
    # platform already has posts today, stay a full gap out to be safe (we only
    # know the count of prior posts, not their times).
    cursor = now + _jit() if existing_today <= 0 else now + gap + _jit()
    day = cursor.date()
    placed_on_day = _seed(day)

    while len(out) < count:
        if cursor.date() != day:
            day = cursor.date()
            placed_on_day = _seed(day)
        if placed_on_day >= rule.max_per_day:
            # Day is full: roll to early next day (then the gap cadence spreads
            # posts through it). The cap is enforced on this clean timeline.
            # The rolled-into day starts at ITS seed, so a day a prior sweep
            # already filled rolls again — the loop walks forward until a day
            # with spare capacity.
            next_day = _dt.datetime.combine(
                day + _dt.timedelta(days=1),
                _dt.time(hour=0, minute=0),
                tzinfo=cursor.tzinfo,
            )
            cursor = next_day + _jit()
            day = cursor.date()
            placed_on_day = _seed(day)
            continue
        out.append(cursor)
        placed_on_day += 1
        cursor = cursor + gap + _jit()

    return out
