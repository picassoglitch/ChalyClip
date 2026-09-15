"""Growth Engine planner integration test (Phases 1-6 wired together)."""

from __future__ import annotations

import datetime as _dt
import random

from chalybclip.llm import GrowthScoreCard, PlatformGrowthScore
from chalybclip.publish.growth_engine import ClipContent, plan_growth_publish
from chalybclip.publish.pacing import DEFAULT_PLATFORM_RULES

_NOW = _dt.datetime(2026, 6, 10, 8, 0, tzinfo=_dt.UTC)


def _card(overall: int, platforms: dict[str, tuple[int, str]], tags: list[str]) -> GrowthScoreCard:
    return GrowthScoreCard(
        overall_score=overall,
        decision="publish_select",
        platforms=[
            PlatformGrowthScore(platform=p, score=s, verdict=v, label="x", reason="r")
            for p, (s, v) in platforms.items()
        ],
        content_tags=tags,
    )


def _clip(cid: str, overall: int, platforms: dict, tags: list[str]) -> ClipContent:
    return ClipContent(
        clip_id=cid, caption=f"caption for {cid}", hook="hook",
        hashtags=["gaming", "viral", "clip"], card=_card(overall, platforms, tags),
    )


def test_skip_verdict_and_floor_drop_weak_platforms() -> None:
    clips = [
        _clip("c1", 92, {"tiktok": (95, "publish"), "linkedin": (12, "skip")}, ["a"]),
    ]
    plan = plan_growth_publish(
        clips, connected=["tiktok", "linkedin"], rules=DEFAULT_PLATFORM_RULES,
        now=_NOW, min_score=40, rng=random.Random(1),
    )
    platforms = {p.platform for p in plan.posts}
    assert platforms == {"tiktok"}  # linkedin skipped (verdict + floor)


def test_per_platform_cap_holds_the_rest() -> None:
    # 8 clips all great on tiktok (cap 5) → exactly 5 scheduled today, the rest
    # held for the next sweep (Phase 2: "the rest wait in the queue").
    clips = [
        _clip(f"c{i}", 90 - i, {"tiktok": (90 - i, "publish")}, [f"theme{i}"])
        for i in range(8)
    ]
    plan = plan_growth_publish(
        clips, connected=["tiktok"], rules=DEFAULT_PLATFORM_RULES, now=_NOW,
        rng=random.Random(2),
    )
    tiktok_posts = [p for p in plan.posts if p.platform == "tiktok"]
    cap = DEFAULT_PLATFORM_RULES["tiktok"].max_per_day
    assert len(tiktok_posts) == cap  # 5 best-fit clips today
    assert {p.clip_id for p in tiktok_posts} == {f"c{i}" for i in range(cap)}
    assert set(plan.held_allocation) == {f"c{i}" for i in range(cap, 8)}  # 3 held


def test_budget_limits_pool() -> None:
    clips = [_clip(f"c{i}", 90 - i * 5, {"tiktok": (90 - i * 5, "publish")}, [f"t{i}"])
             for i in range(5)]
    plan = plan_growth_publish(
        clips, connected=["tiktok"], rules=DEFAULT_PLATFORM_RULES, now=_NOW,
        budget=2, rng=random.Random(3),
    )
    scheduled_clips = {p.clip_id for p in plan.posts}
    assert scheduled_clips == {"c0", "c1"}
    assert set(plan.queued) == {"c2", "c3", "c4"}


def test_fatigue_holds_similar_clips() -> None:
    clips = [
        _clip("c1", 95, {"tiktok": (95, "publish")}, ["warzone", "verdansk", "ak47"]),
        _clip("c2", 90, {"tiktok": (90, "publish")}, ["warzone", "verdansk", "ak47"]),
        _clip("c3", 85, {"tiktok": (85, "publish")}, ["warzone", "verdansk", "ak47"]),
    ]
    plan = plan_growth_publish(
        clips, connected=["tiktok"], rules=DEFAULT_PLATFORM_RULES, now=_NOW,
        fatigue_window=2, fatigue_threshold=0.5, rng=random.Random(4),
    )
    assert "c3" in plan.held_fatigue
    assert {p.clip_id for p in plan.posts} == {"c1", "c2"}


def test_two_consecutive_sweeps_respect_future_day_caps() -> None:
    # Two hands-free sweeps the same day (two VODs), starting in the evening
    # so min_gap pushes part of each allocation past midnight: sweep 1 spills
    # 2 posts onto tomorrow, and sweep 2 — seeded with sweep 1's per-day
    # counts — must top tomorrow up to the cap at most, never past it.
    from chalybclip.publish.pacing import PlatformRule

    rules = {"tiktok": PlatformRule(
        platform="tiktok", max_per_day=3, min_gap_minutes=240, jitter_minutes=0,
    )}
    evening = _NOW.replace(hour=20)  # 20:00 → gap-spaced posts cross midnight
    run1 = plan_growth_publish(
        [_clip(f"a{i}", 90, {"tiktok": (90, "publish")}, [f"ta{i}"]) for i in range(3)],
        connected=["tiktok"], rules=rules, now=evening, rng=random.Random(6),
    )
    by_day: dict[str, int] = {}
    for p in run1.posts:
        d = p.when.date().isoformat()
        by_day[d] = by_day.get(d, 0) + 1
    tomorrow = (evening.date() + _dt.timedelta(days=1)).isoformat()
    assert by_day.get(tomorrow, 0) >= 1  # premise: run 1 spilled forward
    run2 = plan_growth_publish(
        [_clip(f"b{i}", 90, {"tiktok": (90, "publish")}, [f"tb{i}"]) for i in range(3)],
        connected=["tiktok"], rules=rules,
        now=evening + _dt.timedelta(minutes=30),
        existing_today={"tiktok": by_day.get(evening.date().isoformat(), 0)},
        existing_by_day={"tiktok": by_day},
        rng=random.Random(7),
    )
    combined = dict(by_day)
    for p in run2.posts:
        d = p.when.date().isoformat()
        combined[d] = combined.get(d, 0) + 1
    assert run2.posts  # premise: sweep 2 still had capacity to place
    assert all(n <= 3 for n in combined.values())


def test_schedule_honours_min_gap_and_assets_conformed() -> None:
    clips = [
        _clip("c1", 95, {"tiktok": (95, "publish")}, ["a"]),
        _clip("c2", 90, {"tiktok": (90, "publish")}, ["b"]),
    ]
    plan = plan_growth_publish(
        clips, connected=["tiktok"], rules=DEFAULT_PLATFORM_RULES, now=_NOW,
        rng=random.Random(5),
    )
    posts = sorted(plan.posts, key=lambda p: p.when)
    gap = posts[1].when - posts[0].when
    # tiktok min_gap 180 minus jitter window (±20*2) → still clearly spaced.
    assert gap >= _dt.timedelta(minutes=180 - 40)
    # Per-platform asset: caption clamped to tiktok 150, hashtags to 5.
    for p in posts:
        assert len(p.asset.caption) <= 150
        assert len(p.asset.hashtags) <= 5
        assert p.asset.filename.endswith("_tiktok.mp4")
