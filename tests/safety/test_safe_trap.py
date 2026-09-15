"""Publishing safe trap — anti-shadowban window scoring + slot finding."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from chalybclip.safety import (
    PlatformSafetyRule,
    SafetyPolicy,
    evaluate_post_window,
    next_safe_slot,
    policy_for_kit,
)


class _FakeTimes:
    """Stand-in RecentPostTimesRepo — returns a fixed list of ISO times."""

    def __init__(self, times: list[str]) -> None:
        self._times = times

    async def recent_post_times(
        self, *, platform: str, since: str, exclude_job_id: str | None = None
    ) -> list[str]:
        return list(self._times)


def _policy(**rule: object) -> SafetyPolicy:
    return SafetyPolicy(timezone="UTC", rules={"tiktok": PlatformSafetyRule(**rule)})


# ---- policy_for_kit ----


def test_policy_for_kit_overlays_overrides_field_by_field() -> None:
    kit = SimpleNamespace(
        content_timezone="America/Mexico_City",
        safety_policy={"tiktok": {"daily_cap": 2}},  # only override the cap
    )
    pol = policy_for_kit(kit)
    assert pol.timezone == "America/Mexico_City"
    rule = pol.rules["tiktok"]
    assert rule.daily_cap == 2          # overridden
    assert rule.min_spacing_min == 150  # kept from the built-in default


def test_policy_for_kit_unknown_platform_falls_back() -> None:
    pol = policy_for_kit(SimpleNamespace(content_timezone="UTC", safety_policy=None))
    # merged() returns the built-in default for a platform with no override.
    assert pol.merged("instagram").daily_cap == 3


# ---- evaluate_post_window ----


@pytest.mark.asyncio
async def test_safe_window_when_clear() -> None:
    pol = _policy(min_spacing_min=120, daily_cap=4, quiet_hours_start=1, quiet_hours_end=7)
    v = await evaluate_post_window(
        _FakeTimes([]), platform="tiktok", policy=pol,
        at="2026-06-10T15:00:00+00:00",  # 15:00, no prior posts
    )
    assert v.risk == "safe"
    assert v.allowed is True
    assert v.recommended_at == v.requested_at


@pytest.mark.asyncio
async def test_blocked_inside_quiet_hours_recommends_later() -> None:
    pol = _policy(min_spacing_min=0, daily_cap=0, quiet_hours_start=1, quiet_hours_end=7)
    v = await evaluate_post_window(
        _FakeTimes([]), platform="tiktok", policy=pol,
        at="2026-06-10T03:00:00+00:00",  # inside 01:00-07:00 quiet
    )
    assert v.risk == "blocked"
    assert v.allowed is False
    assert "quiet hours" in v.reason
    # Recommended slot is pushed out of the quiet window.
    assert v.recommended_at > v.requested_at


@pytest.mark.asyncio
async def test_blocked_when_too_close_to_another_post() -> None:
    pol = _policy(min_spacing_min=120, daily_cap=0)
    v = await evaluate_post_window(
        _FakeTimes(["2026-06-10T14:30:00+00:00"]), platform="tiktok", policy=pol,
        at="2026-06-10T15:00:00+00:00",  # only 30min after the last post
    )
    assert v.risk == "blocked"
    assert "spacing" in v.reason


@pytest.mark.asyncio
async def test_daily_cap_blocks() -> None:
    pol = _policy(min_spacing_min=0, daily_cap=2)
    times = ["2026-06-10T08:00:00+00:00", "2026-06-10T12:00:00+00:00"]  # 2 already today
    v = await evaluate_post_window(
        _FakeTimes(times), platform="tiktok", policy=pol,
        at="2026-06-10T16:00:00+00:00",
    )
    assert v.risk == "blocked"
    assert "daily cap" in v.reason


# ---- next_safe_slot ----


@pytest.mark.asyncio
async def test_next_safe_slot_clears_spacing() -> None:
    pol = _policy(min_spacing_min=120, daily_cap=0, jitter_min=0)
    slot = await next_safe_slot(
        _FakeTimes(["2026-06-10T15:00:00+00:00"]), platform="tiktok", policy=pol,
        earliest="2026-06-10T15:30:00+00:00",
    )
    # Must land at least 120min after the 15:00 post → >= 17:00.
    import datetime as _dt
    assert _dt.datetime.fromisoformat(slot) >= _dt.datetime.fromisoformat(
        "2026-06-10T17:00:00+00:00"
    )


@pytest.mark.asyncio
async def test_next_safe_slot_is_deterministic() -> None:
    pol = _policy(min_spacing_min=90, daily_cap=4, quiet_hours_start=1,
                  quiet_hours_end=7, jitter_min=15)
    kw = dict(platform="tiktok", policy=pol, earliest="2026-06-10T15:00:00+00:00")
    a = await next_safe_slot(_FakeTimes([]), **kw)
    b = await next_safe_slot(_FakeTimes([]), **kw)
    assert a == b  # no now()/RNG — same inputs, same slot (jitter is hashed)


@pytest.mark.asyncio
async def test_next_safe_slot_skips_quiet_hours() -> None:
    import datetime as _dt
    pol = _policy(min_spacing_min=0, daily_cap=0, quiet_hours_start=1,
                  quiet_hours_end=7, jitter_min=0)
    slot = await next_safe_slot(
        _FakeTimes([]), platform="tiktok", policy=pol,
        earliest="2026-06-10T03:00:00+00:00",  # inside quiet
    )
    # Pushed to the end of the quiet window (07:00).
    assert _dt.datetime.fromisoformat(slot).hour >= 7
