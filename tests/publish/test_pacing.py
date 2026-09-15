"""Per-platform rulebook + scheduler unit tests (Phase 1).

Pins the pure logic: caption/hashtag clamping, alias normalization, and the
per-platform scheduler honouring max_per_day, min_gap, and jitter.
"""

from __future__ import annotations

import datetime as _dt
import random
from itertools import pairwise

from chalybclip.publish.pacing import (
    DEFAULT_PLATFORM_RULES,
    PlatformRule,
    canonical_platform,
    default_rule_for,
    jitter_delta,
    plan_platform_times,
)

_NOW = _dt.datetime(2026, 6, 10, 8, 0, tzinfo=_dt.UTC)  # Wednesday 08:00 UTC


# ---- aliases + defaults ----


def test_x_aliases_to_twitter() -> None:
    assert canonical_platform("X") == "twitter"
    assert canonical_platform("  Reels ") == "instagram"
    assert canonical_platform("shorts") == "youtube"
    assert default_rule_for("x").platform == "twitter"


def test_unknown_platform_gets_cautious_generic() -> None:
    rule = default_rule_for("mastodon")
    assert rule.platform == "mastodon"
    assert rule.max_per_day == 3  # the generic ceiling


def test_every_shipped_rule_is_internally_consistent() -> None:
    for key, rule in DEFAULT_PLATFORM_RULES.items():
        assert rule.platform == key
        assert rule.caption_min_chars <= rule.caption_max_chars
        assert rule.hashtag_min <= rule.hashtag_max


# ---- caption / hashtag clamping ----


def test_clamp_caption_trims_on_word_boundary() -> None:
    rule = PlatformRule(platform="t", max_per_day=1, min_gap_minutes=1, caption_max_chars=20)
    out = rule.clamp_caption("the quick brown fox jumps over")
    assert len(out) <= 20
    assert not out.endswith(" ")
    # Did not slice mid-word near the boundary.
    assert out in {"the quick brown fox", "the quick brown"}


def test_clamp_hashtags_caps_to_max_keeping_order() -> None:
    rule = DEFAULT_PLATFORM_RULES["tiktok"]  # hashtag_max=5
    out = rule.clamp_hashtags(["a", "b", "c", "d", "e", "f", "g"])
    assert out == ["a", "b", "c", "d", "e"]


def test_clamp_hashtags_zero_max_returns_empty() -> None:
    rule = DEFAULT_PLATFORM_RULES["linkedin"]  # hashtag_max=0
    assert rule.clamp_hashtags(["a", "b"]) == []


# ---- scheduler: caps + gaps ----


def test_respects_min_gap() -> None:
    rule = PlatformRule(platform="t", max_per_day=99, min_gap_minutes=60, jitter_minutes=0)
    times = plan_platform_times(4, rule=rule, now=_NOW)
    assert len(times) == 4
    for a, b in pairwise(times):
        assert (b - a) >= _dt.timedelta(minutes=60)


def test_respects_max_per_day_rolls_forward() -> None:
    rule = PlatformRule(platform="t", max_per_day=3, min_gap_minutes=60, jitter_minutes=0)
    times = plan_platform_times(7, rule=rule, now=_NOW)
    assert len(times) == 7
    by_day: dict[str, int] = {}
    for t in times:
        by_day[t.date().isoformat()] = by_day.get(t.date().isoformat(), 0) + 1
    assert all(n <= 3 for n in by_day.values())
    assert sum(by_day.values()) == 7


def test_existing_today_reduces_day0_capacity() -> None:
    rule = PlatformRule(platform="t", max_per_day=3, min_gap_minutes=30, jitter_minutes=0)
    times = plan_platform_times(3, rule=rule, now=_NOW, existing_today=2)
    # Only 1 slot fits today (3 cap - 2 existing); the rest roll to tomorrow.
    day0 = [t for t in times if t.date() == _NOW.date()]
    assert len(day0) == 1


def test_zero_cap_or_count_yields_nothing() -> None:
    rule = PlatformRule(platform="t", max_per_day=0, min_gap_minutes=30)
    assert plan_platform_times(5, rule=rule, now=_NOW) == []
    rule2 = DEFAULT_PLATFORM_RULES["tiktok"]
    assert plan_platform_times(0, rule=rule2, now=_NOW) == []


def test_first_slot_is_immediate_rest_future() -> None:
    # The first post fires ~now (min_gap spaces consecutive posts, not the
    # first); the rest are strictly later and chronological.
    rule = DEFAULT_PLATFORM_RULES["tiktok"]
    times = plan_platform_times(5, rule=rule, now=_NOW, rng=random.Random(7))
    assert all(t >= _NOW for t in times)
    assert times[0] < _NOW + _dt.timedelta(minutes=rule.min_gap_minutes)
    assert times == sorted(times)


def test_existing_posts_today_delays_first_slot() -> None:
    # When the platform already posted today, the first new post stays a full
    # gap out (we only know the count of prior posts, not their times).
    rule = DEFAULT_PLATFORM_RULES["tiktok"]
    times = plan_platform_times(1, rule=rule, now=_NOW, existing_today=1, rng=random.Random(7))
    assert times[0] >= _NOW + _dt.timedelta(minutes=rule.min_gap_minutes - rule.jitter_minutes)


def test_existing_by_day_rolls_past_already_full_future_days() -> None:
    # Tomorrow already carries a full day from a PREVIOUS sweep — this run's
    # overflow must roll past it, not stack a second batch on top.
    rule = PlatformRule(platform="t", max_per_day=3, min_gap_minutes=60, jitter_minutes=0)
    tomorrow = (_NOW.date() + _dt.timedelta(days=1)).isoformat()
    times = plan_platform_times(
        4, rule=rule, now=_NOW, existing_today=3, existing_by_day={tomorrow: 3},
    )
    assert len(times) == 4
    assert all(t.date() >= _NOW.date() + _dt.timedelta(days=2) for t in times)


def test_existing_by_day_partial_day_gets_only_the_remainder() -> None:
    rule = PlatformRule(platform="t", max_per_day=3, min_gap_minutes=60, jitter_minutes=0)
    tomorrow = (_NOW.date() + _dt.timedelta(days=1)).isoformat()
    times = plan_platform_times(
        5, rule=rule, now=_NOW, existing_today=3, existing_by_day={tomorrow: 2},
    )
    day1 = [t for t in times if t.date().isoformat() == tomorrow]
    assert len(day1) == 1  # 3 cap - 2 already placed
    assert all(n <= 3 for n in _count_by_day(times).values())


def test_today_defers_to_existing_today_never_double_counted() -> None:
    # Callers pass the repo's from_day=today map, which INCLUDES today's
    # count — today must come from existing_today alone, not both.
    rule = PlatformRule(platform="t", max_per_day=3, min_gap_minutes=30, jitter_minutes=0)
    today = _NOW.date().isoformat()
    times = plan_platform_times(
        3, rule=rule, now=_NOW, existing_today=2, existing_by_day={today: 2},
    )
    day0 = [t for t in times if t.date() == _NOW.date()]
    assert len(day0) == 1  # cap 3 - existing 2, not 3 - (2+2)


def _count_by_day(times: list[_dt.datetime]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in times:
        out[t.date().isoformat()] = out.get(t.date().isoformat(), 0) + 1
    return out


def test_two_consecutive_runs_never_exceed_daily_cap() -> None:
    # The production bug: run 1 (VOD A) rolls 3 posts to tomorrow, run 2
    # (VOD B, same day) rolled 3 MORE onto tomorrow — 6 vs max_per_day 3.
    # With the second run seeded from the first's per-day counts, the
    # combined schedule never exceeds the cap on ANY day.
    rule = PlatformRule(platform="t", max_per_day=3, min_gap_minutes=60, jitter_minutes=0)
    run1 = plan_platform_times(7, rule=rule, now=_NOW)
    by_day = _count_by_day(run1)
    run2 = plan_platform_times(
        5, rule=rule, now=_NOW + _dt.timedelta(minutes=30),
        existing_today=by_day.get(_NOW.date().isoformat(), 0),
        existing_by_day=by_day,
    )
    combined = _count_by_day(run1 + run2)
    assert len(run1) + len(run2) == 12
    assert all(n <= 3 for n in combined.values())


# ---- jitter ----


def test_jitter_is_bounded_and_deterministic() -> None:
    rule = DEFAULT_PLATFORM_RULES["tiktok"]  # jitter 20 min
    a = jitter_delta(rule, rng=random.Random(1))
    b = jitter_delta(rule, rng=random.Random(1))
    assert a == b  # same seed → same nudge
    assert abs(a.total_seconds()) <= 20 * 60


def test_no_jitter_when_zero() -> None:
    rule = DEFAULT_PLATFORM_RULES["linkedin"]  # jitter 0 (scheduled, not random)
    assert jitter_delta(rule, rng=random.Random(1)) == _dt.timedelta()
