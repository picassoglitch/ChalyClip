"""Backlog scheduler — the auto-program safeguard fix (per-platform caps/gaps
across days, instead of a flat 30-min drip to one platform)."""

from __future__ import annotations

import datetime as _dt
import random
from itertools import pairwise

from chalybclip.publish.growth_engine import ClipContent, plan_backlog_schedule
from chalybclip.publish.pacing import DEFAULT_PLATFORM_RULES
from chalybclip.score.growth import GrowthInput, fallback_card

_NOW = _dt.datetime(2026, 6, 10, 8, 0, tzinfo=_dt.UTC)


def _synthetic_clip(cid: str, publishability: int, platforms: list[str]) -> ClipContent:
    """A clip with a fallback (publishability-derived) card — the non-LLM path."""
    card = fallback_card(GrowthInput(
        clip_id=cid, duration_s=12, caption=f"caption {cid}",
        platforms=platforms, publishability_score=publishability,
    ))
    return ClipContent(clip_id=cid, caption=f"caption {cid}", card=card,
                       hashtags=["gaming", "clip"], hook="hook")


def test_youtube_backlog_spreads_across_days_not_one() -> None:
    # 39 clips, YouTube only — the exact production bug. Must cap at YouTube's
    # max_per_day with its min_gap, rolling across days.
    clips = [_synthetic_clip(f"c{i}", 80, ["youtube"]) for i in range(39)]
    plan = plan_backlog_schedule(
        clips, connected=["youtube"], rules=DEFAULT_PLATFORM_RULES, now=_NOW,
        rng=random.Random(1),
    )
    assert len(plan.posts) == 39  # all scheduled (across days), none dropped
    cap = DEFAULT_PLATFORM_RULES["youtube"].max_per_day  # 3
    gap = DEFAULT_PLATFORM_RULES["youtube"].min_gap_minutes  # 300
    by_day: dict[str, int] = {}
    for p in plan.posts:
        by_day[p.when.date().isoformat()] = by_day.get(p.when.date().isoformat(), 0) + 1
    assert all(n <= cap for n in by_day.values())  # never more than 3/day
    assert len(by_day) >= 13  # 39 / 3 → spread over ~13 days
    # Consecutive YouTube posts respect the min gap (minus jitter slack).
    yt = sorted(p.when for p in plan.posts)
    for a, b in pairwise(yt):
        if a.date() == b.date():
            assert (b - a) >= _dt.timedelta(minutes=gap - 60)


def test_spreads_across_connected_platforms() -> None:
    clips = [_synthetic_clip(f"c{i}", 85, ["tiktok", "youtube"]) for i in range(6)]
    plan = plan_backlog_schedule(
        clips, connected=["tiktok", "youtube"], rules=DEFAULT_PLATFORM_RULES,
        now=_NOW, rng=random.Random(2),
    )
    platforms = {p.platform for p in plan.posts}
    assert platforms == {"tiktok", "youtube"}  # cross-posted to both


def test_weak_clips_held_off_skip_band() -> None:
    # Publishability < 40 → fallback card marks verdict 'skip' → never scheduled.
    clips = [
        _synthetic_clip("good", 80, ["youtube"]),
        _synthetic_clip("weak", 20, ["youtube"]),
    ]
    plan = plan_backlog_schedule(
        clips, connected=["youtube"], rules=DEFAULT_PLATFORM_RULES, now=_NOW,
        rng=random.Random(3),
    )
    assert {p.clip_id for p in plan.posts} == {"good"}
    assert "weak" in plan.held_allocation


def test_best_clips_get_earliest_slots() -> None:
    clips = [
        _synthetic_clip("low", 50, ["youtube"]),
        _synthetic_clip("high", 95, ["youtube"]),
        _synthetic_clip("mid", 70, ["youtube"]),
    ]
    plan = plan_backlog_schedule(
        clips, connected=["youtube"], rules=DEFAULT_PLATFORM_RULES, now=_NOW,
        rng=random.Random(4),
    )
    order = [p.clip_id for p in sorted(plan.posts, key=lambda p: p.when)]
    assert order[0] == "high"  # best clip first


def test_two_consecutive_runs_respect_future_day_caps() -> None:
    # Re-running auto-program (run 1 already rolled overflow onto future days)
    # must NOT stack a second max_per_day batch onto those days. Run 2 is
    # seeded with run 1's per-day counts (what the callers now fetch via
    # ZernioPublishesRepo.count_by_platform_by_day).
    cap = DEFAULT_PLATFORM_RULES["youtube"].max_per_day  # 3
    run1 = plan_backlog_schedule(
        [_synthetic_clip(f"a{i}", 80, ["youtube"]) for i in range(7)],
        connected=["youtube"], rules=DEFAULT_PLATFORM_RULES, now=_NOW,
        rng=random.Random(6),
    )
    by_day: dict[str, int] = {}
    for p in run1.posts:
        d = p.when.date().isoformat()
        by_day[d] = by_day.get(d, 0) + 1
    run2 = plan_backlog_schedule(
        [_synthetic_clip(f"b{i}", 80, ["youtube"]) for i in range(7)],
        connected=["youtube"], rules=DEFAULT_PLATFORM_RULES,
        now=_NOW + _dt.timedelta(hours=1),
        existing_today={"youtube": by_day.get(_NOW.date().isoformat(), 0)},
        existing_by_day={"youtube": by_day},
        rng=random.Random(7),
    )
    combined = dict(by_day)
    for p in run2.posts:
        d = p.when.date().isoformat()
        combined[d] = combined.get(d, 0) + 1
    assert len(run1.posts) == 7
    assert len(run2.posts) == 7
    assert all(n <= cap for n in combined.values())


def test_assets_conformed_per_platform() -> None:
    clips = [_synthetic_clip("c1", 80, ["youtube"])]
    plan = plan_backlog_schedule(
        clips, connected=["youtube"], rules=DEFAULT_PLATFORM_RULES, now=_NOW,
        rng=random.Random(5),
    )
    post = plan.posts[0]
    assert post.asset.filename.endswith("_youtube.mp4")
    assert len(post.asset.caption) <= DEFAULT_PLATFORM_RULES["youtube"].caption_max_chars
