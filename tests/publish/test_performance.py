"""Continuous-learning performance weights + their effect on allocation."""

from __future__ import annotations

import datetime as _dt

from chalybclip.publish.allocation import ClipPlatformFit, allocate
from chalybclip.publish.growth_engine import ClipContent, plan_backlog_schedule
from chalybclip.publish.pacing import DEFAULT_PLATFORM_RULES
from chalybclip.score.growth import GrowthInput, fallback_card
from chalybclip.score.performance import compute_platform_performance, platform_weights

_NOW = _dt.datetime(2026, 6, 30, 12, 0, tzinfo=_dt.UTC)


def _post(platform_views: dict[str, float], *, age_h: float = 48) -> dict:
    published = (_NOW - _dt.timedelta(hours=age_h)).isoformat()
    return {
        "published_at": published,
        "per_platform": [
            {"platform": p, "metrics": {"views": v}} for p, v in platform_views.items()
        ],
    }


def test_mature_zero_view_platform_is_penalized() -> None:
    # YouTube matured at ~0 views; TikTok at ~50. Enough samples each.
    posts = [_post({"youtube": 0, "tiktok": 50}) for _ in range(5)]
    perf = compute_platform_performance(posts, min_mature_posts=4, now=_NOW)
    assert perf["tiktok"].weight == 1.0
    assert perf["youtube"].weight == 0.2  # floored, not zero
    assert perf["youtube"].learning is False


def test_cold_start_not_penalized() -> None:
    # Only 2 samples → not enough to judge → full weight, marked learning.
    posts = [_post({"youtube": 0}) for _ in range(2)]
    perf = compute_platform_performance(posts, min_mature_posts=4, now=_NOW)
    assert perf["youtube"].learning is True
    assert perf["youtube"].weight == 1.0


def test_fresh_posts_excluded_by_maturity() -> None:
    # 5 YouTube posts but all 1h old → not mature → not judged.
    posts = [_post({"youtube": 0}, age_h=1) for _ in range(5)]
    perf = compute_platform_performance(posts, maturity_hours=24, now=_NOW)
    assert perf == {}  # nothing mature enough to sample


def test_unmeasured_views_dont_count() -> None:
    # views=None (Zernio hasn't measured) must not be read as a real 0.
    posts = [{"published_at": (_NOW - _dt.timedelta(hours=48)).isoformat(),
              "per_platform": [{"platform": "youtube", "metrics": {"views": None}}]}
             for _ in range(5)]
    perf = compute_platform_performance(posts, now=_NOW)
    assert "youtube" not in perf


def test_weights_reduce_allocation_capacity() -> None:
    # TikTok cap 5; weight 0.2 → ~1 clip allocated (trickle), not 5.
    fits = [ClipPlatformFit(f"c{i}", "tiktok", 90 - i, 90 - i) for i in range(5)]
    plan = allocate(fits, platforms=["tiktok"], rules=DEFAULT_PLATFORM_RULES,
                    platform_weights={"tiktok": 0.2})
    assert len(plan.by_platform.get("tiktok", [])) == 1


def test_backlog_weight_shifts_volume() -> None:
    # 10 clips publishable on both; youtube weight 0.2 → ~2 clips, tiktok all 10.
    clips = []
    for i in range(10):
        card = fallback_card(GrowthInput(
            clip_id=f"c{i}", duration_s=12, caption="x",
            platforms=["tiktok", "youtube"], publishability_score=80,
        ))
        clips.append(ClipContent(clip_id=f"c{i}", caption="x", card=card))
    plan = plan_backlog_schedule(
        clips, connected=["tiktok", "youtube"], rules=DEFAULT_PLATFORM_RULES,
        now=_NOW, platform_weights={"youtube": 0.2, "tiktok": 1.0},
    )
    counts = plan.per_platform_counts
    assert counts.get("tiktok") == 10
    assert counts.get("youtube") == 2  # ceil(10 * 0.2)


def test_platform_weights_helper() -> None:
    posts = [_post({"youtube": 0, "tiktok": 100}) for _ in range(5)]
    w = platform_weights(compute_platform_performance(posts, now=_NOW))
    assert w["tiktok"] == 1.0 and w["youtube"] == 0.2
