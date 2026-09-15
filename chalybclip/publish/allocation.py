"""Growth Publish allocation — decide WHICH clips go to WHICH platforms.

The shift the spec asks for: stop "Publish All" (blast every clip to every
platform) and start "Growth Publish" — the operator says *how many clips
today*, and the engine distributes them so each platform receives only the
clips that fit it, never more than its daily rulebook allows. The rest wait
in the queue.

Two ranked inputs drive it:

  * an overall growth score per clip → picks the day's *pool* (the best N),
  * a per-(clip, platform) fit score → picks, within each platform's daily
    capacity, the clips most likely to perform THERE (Phase 6 priority).

This module is pure: it takes scores + rules + today's counts and returns a
plan. Computing the scores (LLM) and executing the plan (Zernio) live
elsewhere, so the allocation logic is unit-testable in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from chalybclip.publish.pacing import PlatformRule, canonical_platform, default_rule_for


@dataclass(frozen=True)
class ClipPlatformFit:
    """How well one clip fits one platform. `score` is 0-100 (a Growth Score
    per Phase 6); `overall` is the clip's platform-independent growth score,
    used to rank the day's pool."""

    clip_id: str
    platform: str
    score: float
    overall: float


@dataclass(frozen=True)
class AllocationPlan:
    # platform -> clip_ids chosen for today, best-fit first.
    by_platform: dict[str, list[str]] = field(default_factory=dict)
    # Clips that fell outside today's pool (the "rest wait in queue").
    queued: list[str] = field(default_factory=list)
    # Clips inside the pool that no platform accepted (score floor / no cap).
    held: list[str] = field(default_factory=list)

    @property
    def total_posts(self) -> int:
        return sum(len(v) for v in self.by_platform.values())

    @property
    def per_platform_counts(self) -> dict[str, int]:
        return {p: len(v) for p, v in self.by_platform.items()}


def allocate(
    fits: list[ClipPlatformFit],
    *,
    platforms: list[str],
    rules: dict[str, PlatformRule] | None = None,
    budget: int | None = None,
    existing_today: dict[str, int] | None = None,
    min_score: float = 0.0,
    platform_weights: dict[str, float] | None = None,
) -> AllocationPlan:
    """Distribute clips across platforms under the rulebook.

    Args:
        fits: every candidate (clip, platform) pair with its fit + overall
            scores. A clip may appear once per platform (cross-posting).
        platforms: the connected/target platforms to allocate onto. Fits for
            platforms outside this list are ignored.
        rules: per-platform rulebook (defaults filled in for any missing).
        budget: "how many clips today" — the pool is the top-`budget` clips by
            overall score. `None` means no pool limit (only the per-platform
            caps bound the day).
        existing_today: posts already placed today per platform; subtracted
            from each platform's `max_per_day` capacity.
        min_score: Phase-5 publish floor — a (clip, platform) fit below this is
            never scheduled (the clip is held off that platform).
        platform_weights: continuous-learning multipliers (0-1) per platform;
            a platform's remaining daily capacity is scaled by its weight so a
            consistently low-performing platform gets fewer clips (never zero —
            it keeps a trickle to keep re-testing). Absent → full capacity.

    Returns an `AllocationPlan`: chosen clips per platform, plus the clips that
    queued (outside the pool) or were held (in the pool, no platform took them).
    """
    rules = rules or {}
    existing_today = existing_today or {}
    weights = platform_weights or {}
    target = {canonical_platform(p) for p in platforms}

    # Rank clips by their overall growth score to form today's pool.
    overall_by_clip: dict[str, float] = {}
    for f in fits:
        overall_by_clip[f.clip_id] = max(overall_by_clip.get(f.clip_id, 0.0), f.overall)
    ranked_clips = [
        c for c, _ in sorted(overall_by_clip.items(), key=lambda kv: -kv[1])
    ]
    pool = set(ranked_clips if budget is None else ranked_clips[: max(0, budget)])
    queued = [c for c in ranked_clips if c not in pool]

    # Remaining capacity per platform after today's existing posts.
    capacity: dict[str, int] = {}
    for p in target:
        rule = rules.get(p) or default_rule_for(p)
        if not rule.enabled:
            capacity[p] = 0
            continue
        cap = max(0, rule.max_per_day - int(existing_today.get(p, 0)))
        w = weights.get(p, 1.0)
        # Scale by the learned performance weight; a positive capacity never
        # drops below 1 (keep a trickle so a recovering platform re-samples).
        capacity[p] = max(1, round(cap * w)) if (cap > 0 and w > 0) else round(cap * w)

    # Greedily fill each platform with its best-fitting in-pool clips. We walk
    # fits globally by descending fit score so the strongest matches claim
    # capacity first across all platforms.
    by_platform: dict[str, list[str]] = {p: [] for p in target}
    accepted: set[str] = set()
    eligible = sorted(
        (
            f
            for f in fits
            if canonical_platform(f.platform) in target
            and f.clip_id in pool
            and f.score >= min_score
        ),
        key=lambda f: -f.score,
    )
    for f in eligible:
        p = canonical_platform(f.platform)
        if capacity.get(p, 0) <= 0:
            continue
        if f.clip_id in by_platform[p]:
            continue
        by_platform[p].append(f.clip_id)
        capacity[p] -= 1
        accepted.add(f.clip_id)

    by_platform = {p: v for p, v in by_platform.items() if v}
    held = [c for c in pool if c not in accepted]
    return AllocationPlan(by_platform=by_platform, queued=queued, held=held)
