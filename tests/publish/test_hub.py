"""Hub publish service unit tests (phase 3) — the deterministic parts.

The API surface is covered end-to-end in tests/api/test_internal_publish.py;
these pin the pure logic: batch time planning (anti-spam cap), best-time
resolution, media-URL validation, and service-token parsing.
"""

from __future__ import annotations

import datetime as _dt

import httpx
import pytest
import respx

from chalybclip.publish.hub import (
    HubPublishError,
    next_best_time,
    plan_batch_times,
    validate_media_url,
)
from chalybclip.settings import Settings

_NOW = _dt.datetime(2026, 6, 10, 8, 0, tzinfo=_dt.UTC)  # Wednesday 08:00 UTC


# ---- plan_batch_times ----


def test_batch_spreads_within_daily_cap() -> None:
    times = plan_batch_times(6, now=_NOW, cap_per_day=4, existing_today=0)
    assert len(times) == 6
    by_day: dict[str, int] = {}
    for t in times:
        by_day[t.date().isoformat()] = by_day.get(t.date().isoformat(), 0) + 1
    # Never more than the cap on any single day; overflow rolls forward.
    assert all(n <= 4 for n in by_day.values())
    assert by_day["2026-06-10"] == 4
    assert by_day["2026-06-11"] == 2
    # Strictly future (> now + 30 min) and chronological per day.
    assert all(t > _NOW + _dt.timedelta(minutes=30) for t in times)
    assert times == sorted(times)


def test_batch_counts_existing_posts_against_today() -> None:
    times = plan_batch_times(3, now=_NOW, cap_per_day=4, existing_today=3)
    # Only one slot left today; the rest roll to tomorrow.
    assert sum(1 for t in times if t.date() == _NOW.date()) == 1
    assert sum(1 for t in times if t.date() == _NOW.date() + _dt.timedelta(days=1)) == 2


def test_batch_full_day_rolls_everything_forward() -> None:
    times = plan_batch_times(2, now=_NOW, cap_per_day=4, existing_today=4)
    assert all(t.date() > _NOW.date() for t in times)


def test_batch_late_day_skips_past_hours() -> None:
    late = _NOW.replace(hour=23)
    times = plan_batch_times(3, now=late, cap_per_day=4, existing_today=0)
    assert len(times) == 3
    assert all(t > late + _dt.timedelta(minutes=30) for t in times)


def test_batch_prefers_best_time_hours_for_matching_weekday() -> None:
    # Wednesday = weekday 2; high-engagement slot at 18:00 UTC.
    slots = [
        {"day_of_week": 2, "hour": 18, "avg_engagement": 500.0, "post_count": 9},
        {"day_of_week": 0, "hour": 9, "avg_engagement": 300.0, "post_count": 5},
    ]
    times = plan_batch_times(
        1, now=_NOW, cap_per_day=1, existing_today=0, best_slots=slots,
    )
    assert times[0].hour == 18
    assert times[0].date() == _NOW.date()


def test_engagement_path_does_not_pad_with_fallback_hours() -> None:
    # One engagement slot, on Wednesdays only, at a NON-fallback hour.
    from chalybclip.publish.hub import _FALLBACK_HOURS
    slots = [{"day_of_week": 2, "hour": 14, "avg_engagement": 500.0, "post_count": 9}]
    times = plan_batch_times(
        3, now=_NOW, cap_per_day=4, existing_today=0, best_slots=slots,
    )
    assert len(times) == 3
    # Pure engagement path: every post is at the one proven hour (14:00),
    # never a generic fallback hour, and a day with no slot spills forward —
    # so the three posts land on three successive Wednesdays.
    assert all(t.hour == 14 for t in times)
    assert 14 not in _FALLBACK_HOURS  # guards the test's premise
    assert all(t.weekday() == 2 for t in times)
    assert len({t.date() for t in times}) == 3


def test_fallback_path_uses_fallback_hours_when_no_slots() -> None:
    from chalybclip.publish.hub import _FALLBACK_HOURS
    times = plan_batch_times(3, now=_NOW, cap_per_day=4, existing_today=0)
    assert len(times) == 3
    assert all(t.hour in _FALLBACK_HOURS for t in times)


def test_engagement_short_falls_back_as_last_resort() -> None:
    # One proven slot (Wed 14:00) but a batch too big for the Wednesdays in
    # the horizon to absorb → leftovers must land on fallback hours, NOT be
    # dropped (which would make the caller post them immediately).
    from chalybclip.publish.hub import _FALLBACK_HOURS
    slots = [{"day_of_week": 2, "hour": 14, "avg_engagement": 500.0, "post_count": 9}]
    times = plan_batch_times(
        8, now=_NOW, cap_per_day=4, existing_today=0, best_slots=slots,
    )
    # Every clip gets a real slot — none fall through to "immediate".
    assert len(times) == 8
    assert times == sorted(times)
    eng = [t for t in times if t.hour == 14]
    fallback = [t for t in times if t.hour in _FALLBACK_HOURS]
    # Engagement is exhausted first (one 14:00 per Wednesday in the horizon),
    # then the fixed hours top up the rest.
    assert all(t.weekday() == 2 for t in eng)
    assert len(eng) >= 1
    assert len(fallback) >= 1
    assert len(eng) + len(fallback) == 8


# ---- next_best_time ----


def test_next_best_time_picks_top_slot_next_occurrence() -> None:
    slots = [
        {"day_of_week": 2, "hour": 18, "avg_engagement": 510.3, "post_count": 15},
        {"day_of_week": 0, "hour": 9, "avg_engagement": 342.5, "post_count": 12},
    ]
    best = next_best_time(slots, now=_NOW)
    assert best is not None
    # Wednesday 18:00 is later today — the top slot wins.
    assert (best.weekday(), best.hour) == (2, 18)
    assert best.date() == _NOW.date()


def test_next_best_time_rolls_a_week_when_slot_already_passed() -> None:
    slots = [{"day_of_week": 2, "hour": 6, "avg_engagement": 100.0}]
    best = next_best_time(slots, now=_NOW)  # 08:00 Wed > 06:00 Wed
    assert best is not None
    assert best - _NOW == _dt.timedelta(days=7) - _dt.timedelta(hours=2)


def test_next_best_time_none_without_usable_slots() -> None:
    assert next_best_time([], now=_NOW) is None
    assert next_best_time([{"day_of_week": "x", "hour": "y"}], now=_NOW) is None


def test_next_best_time_skips_slot_violating_min_gap() -> None:
    # Top slot Wed 18:00, but a post is already scheduled 17:30 — closer than
    # the platform's 240-min gap. Fall through to the NEXT-ranked slot instead
    # of clustering (the rulebook's whole point).
    slots = [
        {"day_of_week": 2, "hour": 18, "avg_engagement": 510.3, "post_count": 15},
        {"day_of_week": 4, "hour": 9, "avg_engagement": 342.5, "post_count": 12},
    ]
    taken = _NOW.replace(hour=17, minute=30)  # Wed 17:30, already scheduled
    best = next_best_time(
        slots, now=_NOW, min_gap_minutes=240, recent_times=[taken],
    )
    assert best is not None
    assert (best.weekday(), best.hour) == (4, 9)  # second slot won


def test_next_best_time_gap_none_when_all_slots_violate() -> None:
    # Every ranked slot violates the gap → None, so the caller's fallback
    # spread takes over rather than forcing a clustered slot.
    slots = [{"day_of_week": 2, "hour": 18, "avg_engagement": 510.3}]
    taken = _NOW.replace(hour=18)
    assert next_best_time(
        slots, now=_NOW, min_gap_minutes=240, recent_times=[taken],
    ) is None


def test_next_best_time_gap_checks_future_scheduled_posts_too() -> None:
    # The clash can sit AHEAD of the candidate (a post scheduled for 19:00
    # blocks an 18:00 pick just as much as one at 17:30 does).
    slots = [{"day_of_week": 2, "hour": 18, "avg_engagement": 510.3}]
    future_taken = _NOW.replace(hour=19)
    assert next_best_time(
        slots, now=_NOW, min_gap_minutes=120, recent_times=[future_taken],
    ) is None


def test_next_best_time_ignores_gap_when_no_recent_times() -> None:
    slots = [{"day_of_week": 2, "hour": 18, "avg_engagement": 510.3}]
    best = next_best_time(slots, now=_NOW, min_gap_minutes=240, recent_times=[])
    assert best is not None and (best.weekday(), best.hour) == (2, 18)


# ---- validate_media_url ----


@pytest.mark.asyncio
async def test_media_url_ok_video() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.head("https://cdn.test/clip.mp4").mock(
                return_value=httpx.Response(
                    200, headers={"content-type": "video/mp4"}
                )
            )
            await validate_media_url("https://cdn.test/clip.mp4", http=http)


@pytest.mark.asyncio
async def test_media_url_head_405_falls_back_to_ranged_get() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            mock.head("https://cdn.test/clip.mp4").mock(
                return_value=httpx.Response(405)
            )
            get = mock.get("https://cdn.test/clip.mp4").mock(
                return_value=httpx.Response(
                    206, headers={"content-type": "video/mp4"}
                )
            )
            await validate_media_url("https://cdn.test/clip.mp4", http=http)
    assert get.calls.last.request.headers["Range"] == "bytes=0-0"


@pytest.mark.asyncio
async def test_media_url_404_is_unreachable() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.head("https://cdn.test/gone.mp4").mock(
                return_value=httpx.Response(404)
            )
            with pytest.raises(HubPublishError) as ei:
                await validate_media_url("https://cdn.test/gone.mp4", http=http)
    assert ei.value.code == "media_url_unreachable"


@pytest.mark.asyncio
async def test_media_url_html_is_not_video() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.head("https://cdn.test/page").mock(
                return_value=httpx.Response(
                    200, headers={"content-type": "text/html; charset=utf-8"}
                )
            )
            with pytest.raises(HubPublishError) as ei:
                await validate_media_url("https://cdn.test/page", http=http)
    assert ei.value.code == "media_url_not_video"


@pytest.mark.asyncio
async def test_media_url_non_http_rejected() -> None:
    with pytest.raises(HubPublishError) as ei:
        await validate_media_url("file:///etc/passwd")
    assert ei.value.code == "media_url_invalid"


# ---- service-token parsing ----


def test_hub_service_token_map_parses_pairs() -> None:
    s = Settings(hub_service_tokens="chalybobs:tok_a, chalyb:tok_b,broken,:empty")
    assert s.hub_service_token_map() == {"tok_a": "chalybobs", "tok_b": "chalyb"}


def test_hub_service_token_map_empty_when_unset() -> None:
    assert Settings(hub_service_tokens=None).hub_service_token_map() == {}
