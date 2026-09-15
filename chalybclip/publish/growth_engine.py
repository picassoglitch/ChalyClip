"""Growth Engine planner — turn scored clips into a publish schedule.

This is the deterministic core that ties Phases 1-6 together. Given clips that
already carry a Growth Score card + their composed content, plus the tenant's
rulebook and today's posting counts, it decides:

    1. fatigue spacing   — hold near-duplicate themes (Phase 4),
    2. publish/skip       — drop platform fits below the floor (Phase 5),
    3. allocation         — fill each platform up to its daily cap, best-fit
                            first, within the day's clip budget (Phase 2/6),
    4. scheduling         — place each post under the platform's min-gap +
                            jitter rulebook (Phase 1),
    5. assets             — conform caption/hashtags/title per platform (Phase 3).

Pure: scores + rules + counts in, a `GrowthPlan` out. The executor (the
hands-free sweep) renders + posts + records; none of that lives here, so the
whole decision is unit-testable without a DB, an LLM, or Zernio.
"""
from __future__ import annotations

import datetime as _dt
import math
import random as _random
from dataclasses import dataclass, field

from chalybclip.llm import GrowthScoreCard
from chalybclip.publish.allocation import ClipPlatformFit, allocate
from chalybclip.publish.assets import PlatformAsset, build_platform_assets
from chalybclip.publish.fatigue import assess_batch_fatigue
from chalybclip.publish.pacing import (
    PlatformRule,
    canonical_platform,
    default_rule_for,
    plan_platform_times,
)


@dataclass(frozen=True)
class ClipContent:
    """A scored, composed clip ready for the planner."""

    clip_id: str
    caption: str
    card: GrowthScoreCard
    hashtags: list[str] = field(default_factory=list)
    hook: str = ""
    title: str | None = None
    first_comment: str | None = None


@dataclass(frozen=True)
class ScheduledPost:
    clip_id: str
    platform: str
    when: _dt.datetime  # always scheduled — the engine never bursts (health first)
    asset: PlatformAsset
    score: int


@dataclass(frozen=True)
class GrowthPlan:
    posts: list[ScheduledPost] = field(default_factory=list)
    held_fatigue: list[str] = field(default_factory=list)
    held_allocation: list[str] = field(default_factory=list)
    queued: list[str] = field(default_factory=list)

    @property
    def per_platform_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.posts:
            out[p.platform] = out.get(p.platform, 0) + 1
        return out


def plan_growth_publish(
    clips: list[ClipContent],
    *,
    connected: list[str],
    rules: dict[str, PlatformRule],
    now: _dt.datetime,
    budget: int | None = None,
    min_score: int = 40,
    existing_today: dict[str, int] | None = None,
    existing_by_day: dict[str, dict[str, int]] | None = None,
    recent_tags: list[list[str]] | None = None,
    platform_weights: dict[str, float] | None = None,
    fatigue_threshold: float = 0.6,
    fatigue_window: int = 2,
    rng: _random.Random | None = None,
) -> GrowthPlan:
    """Plan today's publishing for a batch of scored clips. See module docstring
    for the pipeline. `existing_today` is per-platform posts already placed
    today (subtracted from each platform's cap); `existing_by_day` is the
    per-platform {YYYY-MM-DD: count} of posts prior sweeps already scheduled
    on FUTURE days, so overflow rolled forward never stacks past a day's cap
    across runs. `platform_weights` are the continuous-learning multipliers
    that shift volume toward platforms that actually earn views."""
    existing_today = dict(existing_today or {})
    existing_by_day = dict(existing_by_day or {})
    target = [canonical_platform(p) for p in connected]
    by_id = {c.clip_id: c for c in clips}

    # --- Phase 4: fatigue spacing. Order by overall score so the strongest
    # clip of a theme claims the slot and weaker near-duplicates are held. ---
    ordered = sorted(clips, key=lambda c: -c.card.overall_score)
    verdicts = assess_batch_fatigue(
        [(c.clip_id, c.card.content_tags) for c in ordered],
        recent_tags=recent_tags,
        threshold=fatigue_threshold,
        max_similar_per_window=fatigue_window,
    )
    held_fatigue = [v.clip_id for v in verdicts if v.hold]
    fresh_ids = {v.clip_id for v in verdicts if not v.hold}

    # --- Phase 5/6: build (clip, platform) fits from the cards, keeping only
    # 'publish' verdicts at/above the floor on connected platforms. ---
    fits: list[ClipPlatformFit] = []
    for c in ordered:
        if c.clip_id not in fresh_ids:
            continue
        for ps in c.card.platforms:
            p = canonical_platform(ps.platform)
            if p not in target or ps.verdict != "publish":
                continue
            fits.append(
                ClipPlatformFit(
                    clip_id=c.clip_id, platform=p,
                    score=float(ps.score), overall=float(c.card.overall_score),
                )
            )

    # --- Phase 2: allocate within the day's budget + per-platform caps. ---
    plan = allocate(
        fits, platforms=target, rules=rules, budget=budget,
        existing_today=existing_today, min_score=float(min_score),
        platform_weights=platform_weights,
    )

    # --- Phase 1 + 3: schedule each platform's clips under its rulebook and
    # build the per-platform asset for each post. ---
    score_by_clip_platform = {
        (f.clip_id, f.platform): int(f.score) for f in fits
    }
    posts: list[ScheduledPost] = []
    for platform, clip_ids in plan.by_platform.items():
        rule = rules.get(platform) or default_rule_for(platform)
        times = plan_platform_times(
            len(clip_ids), rule=rule, now=now,
            existing_today=int(existing_today.get(platform, 0)),
            existing_by_day=existing_by_day.get(platform), rng=rng,
        )
        for clip_id, when in zip(clip_ids, times, strict=False):
            content = by_id[clip_id]
            assets = build_platform_assets(
                clip_id=clip_id, base_caption=content.caption,
                base_hashtags=content.hashtags, base_title=content.title,
                hook=content.hook, platforms=[platform], rules=rules,
                first_comment=content.first_comment,
            )
            posts.append(
                ScheduledPost(
                    clip_id=clip_id, platform=platform, when=when,
                    asset=assets[platform],
                    score=score_by_clip_platform.get((clip_id, platform), 0),
                )
            )

    return GrowthPlan(
        posts=sorted(posts, key=lambda p: p.when),
        held_fatigue=held_fatigue,
        held_allocation=plan.held,
        queued=plan.queued,
    )


def plan_backlog_schedule(
    clips: list[ClipContent],
    *,
    connected: list[str],
    rules: dict[str, PlatformRule],
    now: _dt.datetime,
    min_score: int = 0,
    existing_today: dict[str, int] | None = None,
    existing_by_day: dict[str, dict[str, int]] | None = None,
    recent_tags: list[list[str]] | None = None,
    platform_weights: dict[str, float] | None = None,
    fatigue_threshold: float = 0.6,
    fatigue_window: int = 2,
    rng: _random.Random | None = None,
) -> GrowthPlan:
    """Schedule a whole approved BACKLOG across days under the rulebook.

    Unlike `plan_growth_publish` (which allocates a single day's pool and
    queues the rest — right for a per-VOD sweep), this spreads the ENTIRE
    eligible backlog: each connected platform gets its publishable clips paced
    by its own `max_per_day` + `min_gap`, rolling overflow to following days
    (via `plan_platform_times`). Best clips by overall score claim the earliest
    slots. Every eligible clip cross-posts to every allowed platform on which
    its verdict is 'publish' and its fit score clears `min_score`.

    This is what the dashboard's "Auto-program everything" button uses, so a
    39-clip YouTube backlog becomes ~3/day, 5h apart, over ~2 weeks — instead
    of 39 posts in one day every 30 minutes. Pure + deterministic given `rng`.

    `existing_by_day` ({platform: {YYYY-MM-DD: count}}) carries what PRIOR
    runs already scheduled on future days, so re-running the auto-program
    never stacks a second batch past `max_per_day` on any day.
    """
    existing_today = dict(existing_today or {})
    existing_by_day = dict(existing_by_day or {})
    target = [canonical_platform(p) for p in connected]
    by_id = {c.clip_id: c for c in clips}

    # Best overall score first → earliest slots on every platform.
    ordered = sorted(clips, key=lambda c: -c.card.overall_score)

    # Fatigue spacing: hold near-duplicate themes (no-op when content_tags are
    # empty, e.g. the non-LLM publishability path).
    verdicts = assess_batch_fatigue(
        [(c.clip_id, c.card.content_tags) for c in ordered],
        recent_tags=recent_tags,
        threshold=fatigue_threshold,
        max_similar_per_window=fatigue_window,
    )
    held_fatigue = [v.clip_id for v in verdicts if v.hold]
    held = set(held_fatigue)
    fresh = [c for c in ordered if c.clip_id not in held]

    weights = platform_weights or {}
    posts: list[ScheduledPost] = []
    scheduled_ids: set[str] = set()
    for platform in target:
        rule = rules.get(platform) or default_rule_for(platform)
        if not rule.enabled:
            continue
        eligible: list[tuple[str, int]] = []
        for c in fresh:
            for ps in c.card.platforms:
                if (
                    canonical_platform(ps.platform) == platform
                    and ps.verdict == "publish"
                    and ps.score >= min_score
                ):
                    eligible.append((c.clip_id, int(ps.score)))
                    break
        # Continuous learning: a low-performing platform receives only its
        # top-weight fraction of the backlog (best clips first — `fresh` is
        # score-ordered), shifting volume to platforms that earn views. A
        # positive list never drops below 1 (keep re-testing).
        w = weights.get(platform, 1.0)
        if w < 1.0 and eligible:
            keep = max(1, math.ceil(len(eligible) * w))
            eligible = eligible[:keep]
        times = plan_platform_times(
            len(eligible), rule=rule, now=now,
            existing_today=int(existing_today.get(platform, 0)),
            existing_by_day=existing_by_day.get(platform), rng=rng,
        )
        for (clip_id, score), when in zip(eligible, times, strict=False):
            content = by_id[clip_id]
            asset = build_platform_assets(
                clip_id=clip_id, base_caption=content.caption,
                base_hashtags=content.hashtags, base_title=content.title,
                hook=content.hook, platforms=[platform], rules=rules,
                first_comment=content.first_comment,
            )[platform]
            posts.append(
                ScheduledPost(
                    clip_id=clip_id, platform=platform, when=when,
                    asset=asset, score=score,
                )
            )
            scheduled_ids.add(clip_id)

    # Clips that never reached any platform (all 'skip' / below floor).
    held_allocation = [
        c.clip_id for c in fresh if c.clip_id not in scheduled_ids
    ]
    return GrowthPlan(
        posts=sorted(posts, key=lambda p: p.when),
        held_fatigue=held_fatigue,
        held_allocation=held_allocation,
        queued=[],
    )
