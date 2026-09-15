"""Safe-trap scoring + slot finding.

Two entry points:

  * `evaluate_post_window` — advisory. Score a proposed post against the
    policy *right now* and return a `SafetyVerdict` (risk + reason +
    recommended time). The dashboard surfaces this; the drain stamps it.

  * `next_safe_slot` — the auto-schedule primitive. Walk forward from an
    `earliest` timestamp to the first instant that satisfies spacing,
    daily cap, and quiet hours, then add deterministic jitter. Pure and
    idempotent: it never reads the wall clock or RNG, so re-running with
    the same inputs yields the same slot.

Both read recent post times through a tiny `RecentPostTimesRepo` protocol
so tests can inject a fake without a database.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from typing import Protocol

from .models import PlatformSafetyRule, Risk, SafetyPolicy, SafetyVerdict

# How far back to look when measuring spacing / counting the local day.
# Comfortably covers the largest spacing window + a full day either side.
_LOOKBACK_HOURS = 48

# Hard cap on the slot search so a pathological policy can't spin forever.
_MAX_ITERS = 1000


class RecentPostTimesRepo(Protocol):
    """The slice of `PublishJobsRepo` the safe trap needs."""

    async def recent_post_times(
        self, *, platform: str, since: str, exclude_job_id: str | None = None
    ) -> list[str]: ...


# ---- public entry ----


async def evaluate_post_window(
    jobs_repo: RecentPostTimesRepo,
    *,
    platform: str,
    policy: SafetyPolicy,
    at: str | None = None,
    exclude_job_id: str | None = None,
) -> SafetyVerdict:
    """Score posting `platform` at `at` (default: now) against `policy`."""
    rule = policy.merged(platform)
    tz = _zone(policy.timezone)
    now = _parse(at) if at is not None else _dt.datetime.now(_dt.UTC)

    times = await _load_times(
        jobs_repo, platform=platform, around=now, exclude_job_id=exclude_job_id
    )

    reasons: list[str] = []
    blocked = False
    caution = False

    if _in_quiet_hours(now, rule, tz):
        blocked = True
        reasons.append(
            f"within quiet hours "
            f"({rule.quiet_hours_start:02d}:00-{rule.quiet_hours_end:02d}:00 "
            f"{policy.timezone})"
        )

    if rule.daily_cap:
        local_day = now.astimezone(tz).date()
        day_count = sum(1 for t in times if t.astimezone(tz).date() == local_day)
        if day_count >= rule.daily_cap:
            blocked = True
            reasons.append(
                f"daily cap reached ({day_count}/{rule.daily_cap} on {platform})"
            )

    spacing_s = rule.min_spacing_min * 60
    nearest = min(
        (abs((now - t).total_seconds()) for t in times), default=None
    )
    if nearest is not None and spacing_s:
        if nearest < spacing_s:
            blocked = True
            reasons.append(
                f"too close to another post "
                f"({int(nearest // 60)}min < {rule.min_spacing_min}min spacing)"
            )
        elif nearest < spacing_s * 1.5:
            caution = True
            reasons.append("posting cadence is tight")

    risk: Risk = "blocked" if blocked else ("caution" if caution else "safe")
    if blocked:
        recommended = await next_safe_slot(
            jobs_repo,
            platform=platform,
            policy=policy,
            earliest=now.isoformat(),
            exclude_job_id=exclude_job_id,
        )
    else:
        recommended = now.isoformat()

    return SafetyVerdict(
        platform=platform,
        allowed=not blocked,
        risk=risk,
        reason="; ".join(reasons) or "within safe posting window",
        requested_at=now.isoformat(),
        recommended_at=recommended,
    )


async def next_safe_slot(
    jobs_repo: RecentPostTimesRepo,
    *,
    platform: str,
    policy: SafetyPolicy,
    earliest: str,
    exclude_job_id: str | None = None,
) -> str:
    """First instant at/after `earliest` that satisfies every rule.

    Deterministic given the same inputs — no `now()`, no RNG (jitter is a
    stable hash of platform + slot). Safe to call from idempotent enqueue.
    """
    rule = policy.merged(platform)
    tz = _zone(policy.timezone)
    spacing = _dt.timedelta(minutes=rule.min_spacing_min)
    start = _parse(earliest)

    times = sorted(
        await _load_times(
            jobs_repo, platform=platform, around=start, exclude_job_id=exclude_job_id
        )
    )

    cand = start
    for _ in range(_MAX_ITERS):
        # 1. Quiet hours — jump to the end of the quiet window.
        if _in_quiet_hours(cand, rule, tz):
            cand = _next_quiet_end(cand, rule, tz)
            continue
        # 2. Daily cap — jump to the next local midnight.
        if rule.daily_cap:
            local_day = cand.astimezone(tz).date()
            day_count = sum(1 for t in times if t.astimezone(tz).date() == local_day)
            if day_count >= rule.daily_cap:
                cand = _next_local_midnight(cand, tz)
                continue
        # 3. Spacing — push past the nearest conflicting post.
        if spacing.total_seconds() > 0:
            conflict = next(
                (
                    t
                    for t in times
                    if abs((t - cand).total_seconds()) < spacing.total_seconds()
                ),
                None,
            )
            if conflict is not None:
                cand = conflict + spacing
                continue
        break

    # Deterministic forward jitter, dropped if it would re-enter quiet hours.
    if rule.jitter_min > 0:
        jittered = cand + _dt.timedelta(
            minutes=_jitter_minutes(platform, cand, rule.jitter_min)
        )
        if not _in_quiet_hours(jittered, rule, tz):
            cand = jittered

    return cand.isoformat()


# ---- helpers ----


async def _load_times(
    jobs_repo: RecentPostTimesRepo,
    *,
    platform: str,
    around: _dt.datetime,
    exclude_job_id: str | None,
) -> list[_dt.datetime]:
    since = (around - _dt.timedelta(hours=_LOOKBACK_HOURS)).isoformat()
    raw = await jobs_repo.recent_post_times(
        platform=platform, since=since, exclude_job_id=exclude_job_id
    )
    return [_parse(t) for t in raw]


def _parse(ts: str) -> _dt.datetime:
    d = _dt.datetime.fromisoformat(ts)
    if d.tzinfo is None:
        d = d.replace(tzinfo=_dt.UTC)
    return d


def _zone(name: str) -> _dt.tzinfo:
    # Imported lazily so a missing tzdata install degrades to UTC instead
    # of crashing the publish drain.
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:
        return _dt.UTC


def _in_quiet_hours(
    at: _dt.datetime, rule: PlatformSafetyRule, tz: _dt.tzinfo
) -> bool:
    s, e = rule.quiet_hours_start, rule.quiet_hours_end
    if s is None or e is None or s == e:
        return False
    hour = at.astimezone(tz).hour
    if s < e:
        return s <= hour < e
    # Window wraps past midnight (e.g. 23 -> 6).
    return hour >= s or hour < e


def _next_quiet_end(
    cand: _dt.datetime, rule: PlatformSafetyRule, tz: _dt.tzinfo
) -> _dt.datetime:
    assert rule.quiet_hours_end is not None
    local = cand.astimezone(tz)
    target = local.replace(
        hour=rule.quiet_hours_end, minute=0, second=0, microsecond=0
    )
    if target <= local:
        target = target + _dt.timedelta(days=1)
    return target.astimezone(_dt.UTC)


def _next_local_midnight(cand: _dt.datetime, tz: _dt.tzinfo) -> _dt.datetime:
    local = cand.astimezone(tz)
    nxt = (local + _dt.timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return nxt.astimezone(_dt.UTC)


def _jitter_minutes(platform: str, cand: _dt.datetime, jitter_min: int) -> int:
    """Stable 0..jitter_min offset derived from platform + slot.

    Deterministic on purpose: the same (platform, slot) always yields the
    same nudge, so recomputing a scheduled job doesn't move it."""
    digest = hashlib.sha256(f"{platform}|{cand.isoformat()}".encode()).hexdigest()
    return int(digest, 16) % (jitter_min + 1)
