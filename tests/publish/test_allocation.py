"""Growth Publish allocation unit tests (Phase 2 + 6).

Pins the distribution logic: today's pool from the budget, per-platform fill
under daily caps, the publish-floor (Phase 5), and the queued/held buckets.
"""

from __future__ import annotations

from chalybclip.publish.allocation import ClipPlatformFit, allocate
from chalybclip.publish.pacing import PlatformRule


def _fit(clip: str, platform: str, score: float, overall: float | None = None) -> ClipPlatformFit:
    return ClipPlatformFit(clip, platform, score, overall if overall is not None else score)


def test_budget_limits_todays_pool() -> None:
    # 4 clips, budget 2 → only the top-2 by overall score are eligible today.
    fits = [
        _fit("c1", "tiktok", 90),
        _fit("c2", "tiktok", 80),
        _fit("c3", "tiktok", 70),
        _fit("c4", "tiktok", 60),
    ]
    plan = allocate(fits, platforms=["tiktok"], budget=2)
    assert set(plan.by_platform["tiktok"]) == {"c1", "c2"}
    assert plan.queued == ["c3", "c4"]


def test_per_platform_cap_enforced() -> None:
    rules = {"tiktok": PlatformRule(platform="tiktok", max_per_day=2, min_gap_minutes=60)}
    fits = [_fit(f"c{i}", "tiktok", 100 - i) for i in range(5)]
    plan = allocate(fits, platforms=["tiktok"], rules=rules)
    assert len(plan.by_platform["tiktok"]) == 2  # capped
    assert plan.by_platform["tiktok"] == ["c0", "c1"]  # best fit first


def test_existing_today_subtracts_from_capacity() -> None:
    rules = {"tiktok": PlatformRule(platform="tiktok", max_per_day=3, min_gap_minutes=60)}
    fits = [_fit(f"c{i}", "tiktok", 100 - i) for i in range(5)]
    plan = allocate(fits, platforms=["tiktok"], rules=rules, existing_today={"tiktok": 2})
    assert len(plan.by_platform["tiktok"]) == 1


def test_min_score_floor_holds_weak_clips() -> None:
    # Phase 5: a clip scoring below the floor on a platform is not scheduled.
    fits = [_fit("good", "tiktok", 95), _fit("weak", "tiktok", 12)]
    plan = allocate(fits, platforms=["tiktok"], min_score=50)
    assert plan.by_platform["tiktok"] == ["good"]
    assert "weak" in plan.held


def test_cross_post_one_clip_many_platforms() -> None:
    fits = [
        _fit("c1", "tiktok", 96),
        _fit("c1", "instagram", 94),
        _fit("c1", "linkedin", 12),
    ]
    plan = allocate(fits, platforms=["tiktok", "instagram", "linkedin"], min_score=50)
    assert plan.by_platform["tiktok"] == ["c1"]
    assert plan.by_platform["instagram"] == ["c1"]
    assert "linkedin" not in plan.by_platform  # below floor → skipped


def test_x_alias_resolves_in_allocation() -> None:
    fits = [_fit("c1", "x", 90)]
    plan = allocate(fits, platforms=["twitter"])
    assert plan.by_platform["twitter"] == ["c1"]


def test_disabled_platform_gets_nothing() -> None:
    rules = {"tiktok": PlatformRule(platform="tiktok", max_per_day=5, min_gap_minutes=60, enabled=False)}
    fits = [_fit("c1", "tiktok", 90)]
    plan = allocate(fits, platforms=["tiktok"], rules=rules)
    assert "tiktok" not in plan.by_platform
    assert "c1" in plan.held
