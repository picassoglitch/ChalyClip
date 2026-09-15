"""Publish & Engagement Hub — internal service-API publish orchestration.

The business logic behind /api/internal/v1/* (routers/internal_publish.py
is a thin wrapper). ChalyOBS Auto-Clip Mode and Chalyb engines publish
through here with a service token; they never talk to Zernio directly.

Every function takes `tenant_id` explicitly (the caller addresses
tenants by id — there's no session) and every query filters on it.
Zernio errors never leak raw: 402 becomes the structured `plan_limit`
error code, everything else `zernio_error` with the message attached.
"""
from __future__ import annotations

import datetime as _dt
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from chalybclip.db import Database, HubPublishJobsRepo, TenantsRepo
from chalybclip.db.models import HubPublishJobRow
from chalybclip.integrations.zernio.client import ZernioClient, ZernioError

_log = logging.getLogger("chalybclip.publish.hub")

VALID_MODES = ("now", "queue", "schedule", "draft")
VALID_SOURCES = ("chalybobs", "chalybclip", "chalyb")

# Platforms whose platformSpecificData supports firstComment per the
# OpenAPI spec (Facebook/Instagram/LinkedIn/YouTube schemas document
# it; TikTok/Twitter/etc. don't — we silently skip those).
FIRST_COMMENT_PLATFORMS = frozenset({"facebook", "instagram", "linkedin", "youtube"})

# Fallback posting hours (UTC) when an account has no best-time data:
# spread through the engagement day, cap-many per day. Used by the
# internal service-API batch (`hub_batch`) and the manual "best hour"
# suggestion. The dashboard publishing paths now schedule through the
# per-platform rulebook (chalybclip.publish.pacing), not a flat interval drip.
_FALLBACK_HOURS = (10, 13, 16, 19, 22, 7)


class HubPublishError(Exception):
    """Structured publish failure. `code` is machine-readable (the
    internal API returns it verbatim); `http_status` maps it onto the
    response. Never a raw Zernio status."""

    def __init__(self, code: str, message: str, *, http_status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


@dataclass(frozen=True)
class HubPublishResult:
    job: HubPublishJobRow
    platforms: list[dict[str, str]]
    duplicate: bool = False


@dataclass(frozen=True)
class PublishOptions:
    per_platform_captions: dict[str, Any] = field(default_factory=dict)
    first_comment: str | None = None
    use_best_time: bool = False


def _tiktok_defaults() -> dict[str, Any]:
    """Mandatory TikTok consents + sane public defaults (same contract
    the dashboard publish path uses)."""
    return {
        "privacy_level": "PUBLIC_TO_EVERYONE",
        "allow_comment": True,
        "content_preview_confirmed": True,
        "express_consent_given": True,
    }


async def validate_media_url(
    url: str, *, http: httpx.AsyncClient | None = None
) -> None:
    """Raise HubPublishError unless `url` looks publicly fetchable video.

    HEAD first (cheap); servers that reject HEAD (405/501) get a 1-byte
    ranged GET. Content-type must be video/* or octet-stream when the
    server reports one — Zernio downloads this URL itself, so catching
    a broken link here saves a guaranteed post.failed later.
    """
    if not url.startswith(("https://", "http://")):
        raise HubPublishError(
            "media_url_invalid", f"video_url must be http(s), got: {url[:50]}",
        )
    own = http is None
    client = http or httpx.AsyncClient(timeout=8.0, follow_redirects=True)
    try:
        try:
            resp = await client.head(url)
            if resp.status_code in (405, 501):
                resp = await client.get(url, headers={"Range": "bytes=0-0"})
        except httpx.HTTPError as e:
            raise HubPublishError(
                "media_url_unreachable",
                f"video_url did not respond: {e}",
                http_status=422,
            ) from e
        if resp.status_code >= 400:
            raise HubPublishError(
                "media_url_unreachable",
                f"video_url returned HTTP {resp.status_code}",
                http_status=422,
            )
        ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        if ctype and not (
            ctype.startswith("video/")
            or ctype in ("application/octet-stream", "binary/octet-stream")
        ):
            raise HubPublishError(
                "media_url_not_video",
                f"video_url content-type is {ctype!r}, expected video/*",
                http_status=422,
            )
    finally:
        if own:
            await client.aclose()


async def _resolve_accounts(
    client: ZernioClient,
    *,
    profile_id: str,
    targets: list[str] | str,
) -> list[tuple[str, str]]:
    """Resolve requested targets against the tenant's connected
    accounts → [(platform, account_id)]. `targets="all_connected"`
    fans out to everything connected."""
    try:
        accounts = await client.list_accounts(profile_id=profile_id)
    except ZernioError as e:
        raise _map_zernio_error(e, during="list_accounts") from e
    by_platform = {a.platform.lower(): a.account_id for a in accounts}
    if not by_platform:
        raise HubPublishError(
            "no_connected_accounts",
            "This tenant has no connected social accounts on Zernio.",
            http_status=409,
        )
    if targets == "all_connected":
        return sorted(by_platform.items())
    assert isinstance(targets, list)
    resolved: list[tuple[str, str]] = []
    missing: list[str] = []
    for t in targets:
        key = t.strip().lower()
        if key in by_platform:
            resolved.append((key, by_platform[key]))
        else:
            missing.append(key)
    if missing:
        raise HubPublishError(
            "target_not_connected",
            f"Not connected for this tenant: {', '.join(sorted(missing))}. "
            f"Connected: {', '.join(sorted(by_platform))}.",
            http_status=409,
        )
    if not resolved:
        raise HubPublishError("invalid_targets", "targets resolved to nothing")
    return resolved


def _map_zernio_error(e: ZernioError, *, during: str) -> HubPublishError:
    """Zernio failure → structured hub error. 402 is the plan-limit
    contract: internal consumers get {error: "plan_limit"}, never a
    raw upstream 402 body."""
    if e.status_code == 402:
        return HubPublishError(
            "plan_limit",
            "Zernio plan limit reached — the connected-account or plan "
            "quota is full. Free Zernio plans allow 2 connected accounts.",
            http_status=402,
        )
    if e.status_code == 409:
        return HubPublishError(
            "duplicate_content",
            f"Zernio rejected duplicate content (24h window): {e}",
            http_status=409,
        )
    return HubPublishError(
        "zernio_error", f"Zernio {during} failed: {e}", http_status=502,
    )


def _parse_iso_times(raw: list[str]) -> list[_dt.datetime]:
    """ISO strings → tz-aware datetimes for `next_best_time`'s gap check.
    Naive timestamps are pinned to UTC (everything we persist is UTC ISO);
    unparseable ones are dropped rather than failing the publish."""
    out: list[_dt.datetime] = []
    for s in raw:
        try:
            t = _dt.datetime.fromisoformat(str(s))
        except ValueError:
            continue
        out.append(t if t.tzinfo is not None else t.replace(tzinfo=_dt.UTC))
    return out


def next_best_time(
    slots: list[dict[str, Any]],
    *,
    now: _dt.datetime,
    not_before: _dt.datetime | None = None,
    min_gap_minutes: int = 0,
    recent_times: list[_dt.datetime] | None = None,
) -> _dt.datetime | None:
    """Earliest future datetime matching the highest-engagement slot.

    Slots are {day_of_week: 0=Mon..6=Sun, hour: UTC}. Picks the best-
    engagement slot's next occurrence after max(now, not_before);
    returns None when there's no usable slot data (caller falls back).

    `min_gap_minutes` + `recent_times` make the pick rulebook-aware: a
    candidate closer than the platform's min gap to ANY known scheduled/
    published time (past or future — scheduled posts sit ahead of now) is
    skipped, falling through to the next-ranked slot. Otherwise repeated
    best-time publishes all landed on the same top slot, clustering tighter
    than the tenant's rulebook allows.
    """
    floor = max(now, not_before) if not_before else now
    gap = _dt.timedelta(minutes=max(0, min_gap_minutes))
    taken = [t for t in (recent_times or []) if t.tzinfo is not None]
    best: _dt.datetime | None = None
    ranked = sorted(
        (s for s in slots if isinstance(s.get("hour"), int)),
        key=lambda s: -float(s.get("avg_engagement") or 0),
    )
    for slot in ranked[:3]:  # top slots only — the rest is noise
        dow = slot.get("day_of_week")
        hour = int(slot["hour"])
        if not isinstance(dow, int) or not (0 <= dow <= 6 and 0 <= hour <= 23):
            continue
        candidate = floor.replace(hour=hour, minute=0, second=0, microsecond=0)
        days_ahead = (dow - candidate.weekday()) % 7
        candidate = candidate + _dt.timedelta(days=days_ahead)
        if candidate <= floor:
            candidate += _dt.timedelta(days=7)
        if gap and any(abs(candidate - t) < gap for t in taken):
            continue  # violates min_gap against an existing post → next slot
        if best is None or candidate < best:
            best = candidate
    return best


async def hub_publish(
    db: Database,
    tenant_id: str,
    *,
    video_url: str,
    title: str | None,
    caption: str | None,
    targets: list[str] | str,
    mode: str,
    scheduled_for: str | None,
    options: PublishOptions,
    source: str,
    idempotency_key: str | None,
    client: ZernioClient,
    http: httpx.AsyncClient | None = None,
    now: _dt.datetime | None = None,
) -> HubPublishResult:
    """Publish ONE clip through the hub. The /internal/v1/publish core.

    Raises HubPublishError with a structured code on every failure
    path. Idempotent on (tenant_id, idempotency_key): a replayed key
    returns the ORIGINAL job with duplicate=True and no Zernio call.
    """
    jobs = HubPublishJobsRepo(db)

    if idempotency_key:
        existing = await jobs.find_by_idempotency(tenant_id, idempotency_key)
        if existing is not None:
            return HubPublishResult(
                job=existing,
                platforms=_pending_platforms(existing.targets),
                duplicate=True,
            )

    if mode not in VALID_MODES:
        raise HubPublishError(
            "invalid_mode", f"mode must be one of {VALID_MODES}, got {mode!r}",
        )
    tenant = await TenantsRepo(db).get(tenant_id)
    if tenant is None:
        raise HubPublishError(
            "unknown_tenant", f"no tenant {tenant_id!r}", http_status=404,
        )
    profile_id = tenant.zernio_profile_id
    if not profile_id:
        raise HubPublishError(
            "no_zernio_profile",
            "Tenant has no Zernio profile yet — create one from the "
            "ChalyClip Publish Center first.",
            http_status=409,
        )

    resolved = await _resolve_accounts(
        client, profile_id=profile_id, targets=targets,
    )
    platform_keys = [p for p, _ in resolved]

    await validate_media_url(video_url, http=http)

    # --- resolve the effective schedule ---
    now_dt = now or _dt.datetime.now(_dt.UTC)
    effective_scheduled: str | None = None
    if mode == "schedule":
        if options.use_best_time:
            try:
                slots = await client.best_time_slots(
                    profile_id=profile_id,
                    platform=platform_keys[0] if len(platform_keys) == 1 else None,
                )
            except ZernioError:
                slots = []  # analytics add-on missing / no data → fallback
            # Single-platform best-time picks respect that platform's
            # min_gap against the tenant's recent/scheduled hub posts —
            # otherwise every use_best_time publish landed on the same top
            # slot, clustering tighter than the rulebook allows. Multi-
            # platform posts have no single gap to enforce (mirrors the
            # platform=None analytics call above), so they stay as-is.
            gap_minutes = 0
            recent: list[_dt.datetime] = []
            if len(platform_keys) == 1:
                from chalybclip.db import PlatformPacingRulesRepo
                from chalybclip.publish.pacing import (
                    canonical_platform,
                    default_rule_for,
                )

                key = canonical_platform(platform_keys[0])
                rules = await PlatformPacingRulesRepo(db).effective_rules(
                    tenant_id
                )
                rule = rules.get(key) or default_rule_for(key)
                gap_minutes = rule.min_gap_minutes
                recent = _parse_iso_times(
                    await jobs.recent_times_for_platform(
                        tenant_id, platform=platform_keys[0],
                    )
                )
            best = next_best_time(
                slots, now=now_dt,
                min_gap_minutes=gap_minutes, recent_times=recent,
            )
            effective_scheduled = (
                best or _next_fallback_hour(now_dt)
            ).isoformat()
        elif scheduled_for:
            effective_scheduled = scheduled_for
        else:
            raise HubPublishError(
                "missing_scheduled_for",
                "mode=schedule needs scheduled_for or options.use_best_time",
            )

    job = await jobs.create(
        tenant_id,
        source=source,
        mode=mode,
        targets=platform_keys,
        video_url=video_url,
        title=title,
        caption=caption,
        scheduled_for=effective_scheduled,
        idempotency_key=idempotency_key,
    )

    custom_content, platform_data = build_per_platform_payload(
        options, platform_keys, title=title,
    )
    try:
        result = await client.create_post(
            profile_id=profile_id,
            content=caption or "",
            media_url=video_url,
            platforms=resolved,
            title=title,
            is_draft=(mode == "draft"),
            queued_from_profile=profile_id if mode == "queue" else None,
            scheduled_for=effective_scheduled,
            timezone="UTC" if effective_scheduled else None,
            tiktok_settings=_tiktok_defaults() if "tiktok" in platform_keys else None,
            custom_content=custom_content,
            platform_specific_data=platform_data,
        )
    except ZernioError as e:
        mapped = _map_zernio_error(e, during="create_post")
        await jobs.set_error(tenant_id, job.job_id, error=mapped.code)
        raise mapped from e

    status = {
        "draft": "draft",
        "queue": "queued",
        "schedule": "scheduled",
        "now": "publishing",
    }[mode]
    await jobs.set_zernio_post(
        tenant_id, job.job_id, post_id=result.post_id, status=status,
    )
    final = await jobs.get(tenant_id, job.job_id)
    assert final is not None
    _log.info(
        "hub.publish source=%s tenant=%s job=%s post=%s mode=%s targets=%s",
        source, tenant_id, job.job_id, result.post_id, mode,
        ",".join(platform_keys),
    )
    return HubPublishResult(
        job=final, platforms=_pending_platforms(final.targets),
    )


async def hub_batch(
    db: Database,
    tenant_id: str,
    *,
    clips: list[dict[str, Any]],
    targets: list[str] | str,
    source: str,
    options: PublishOptions,
    idempotency_key: str | None,
    cap_per_day: int,
    client: ZernioClient,
    http: httpx.AsyncClient | None = None,
    now: _dt.datetime | None = None,
) -> list[dict[str, Any]]:
    """Batch-publish one stream session's clips, spread across days.

    Anti-spam policy: at most `cap_per_day` hub posts per platform per
    UTC day (counting what's already planned today), scheduled at the
    tenant's best-time slots when requested, else the fallback spread —
    never all-at-once. Per-clip failures don't abort the batch; each
    entry reports ok/error individually.
    """
    if not clips:
        raise HubPublishError("empty_batch", "clips must not be empty")
    now_dt = now or _dt.datetime.now(_dt.UTC)
    today = now_dt.date().isoformat()

    tenant = await TenantsRepo(db).get(tenant_id)
    if tenant is None:
        raise HubPublishError(
            "unknown_tenant", f"no tenant {tenant_id!r}", http_status=404,
        )
    profile_id = tenant.zernio_profile_id
    if not profile_id:
        raise HubPublishError(
            "no_zernio_profile",
            "Tenant has no Zernio profile yet.",
            http_status=409,
        )

    resolved = await _resolve_accounts(
        client, profile_id=profile_id, targets=targets,
    )
    platform_keys = [p for p, _ in resolved]

    jobs = HubPublishJobsRepo(db)
    existing_today = 0
    for p in platform_keys:
        existing_today = max(
            existing_today,
            await jobs.count_for_day(tenant_id, platform=p, day=today),
        )

    best_slots: list[dict[str, Any]] | None = None
    if options.use_best_time:
        try:
            best_slots = await client.best_time_slots(profile_id=profile_id)
        except ZernioError:
            best_slots = None  # no analytics → fallback spread

    # Bound the caller's cap by the per-platform rulebook so the internal API
    # can't out-pace platform norms either: the tightest involved platform's
    # max_per_day is a hard ceiling (account health is the top priority).
    from chalybclip.db import PlatformPacingRulesRepo
    from chalybclip.publish.pacing import canonical_platform, default_rule_for

    rulebook = await PlatformPacingRulesRepo(db).effective_rules(tenant_id)
    rulebook_cap = min(
        (
            (rulebook.get(canonical_platform(p)) or default_rule_for(p)).max_per_day
            for p in platform_keys
        ),
        default=cap_per_day,
    )
    effective_cap = max(1, min(cap_per_day, rulebook_cap))

    times = plan_batch_times(
        len(clips),
        now=now_dt,
        cap_per_day=effective_cap,
        existing_today=existing_today,
        best_slots=best_slots,
    )
    if len(times) < len(clips):  # defensive — the planner is bounded
        raise HubPublishError(
            "scheduling_failed",
            f"could only plan {len(times)} of {len(clips)} slots",
            http_status=500,
        )

    per_clip_options = PublishOptions(
        per_platform_captions=options.per_platform_captions,
        first_comment=options.first_comment,
        use_best_time=False,  # the batch planner already picked the times
    )
    results: list[dict[str, Any]] = []
    for i, clip in enumerate(clips):
        key = f"{idempotency_key}:{i}" if idempotency_key else None
        try:
            r = await hub_publish(
                db,
                tenant_id,
                video_url=str(clip.get("video_url") or ""),
                title=clip.get("title"),
                caption=clip.get("caption_default"),
                targets=list(platform_keys),
                mode="schedule",
                scheduled_for=times[i].isoformat(),
                options=per_clip_options,
                source=source,
                idempotency_key=key,
                client=client,
                http=http,
                now=now_dt,
            )
            results.append(
                {
                    "ok": True,
                    "job_id": r.job.job_id,
                    "zernio_post_id": r.job.zernio_post_id,
                    "scheduled_for": r.job.scheduled_for,
                    "status": r.job.status,
                    "duplicate": r.duplicate,
                }
            )
        except HubPublishError as e:
            results.append({"ok": False, "error": e.code, "message": e.message})
    return results


def _pending_platforms(targets_csv: str) -> list[dict[str, str]]:
    return [
        {"platform": p, "status": "pending"}
        for p in targets_csv.split(",")
        if p
    ]


def build_per_platform_payload(
    options: PublishOptions,
    platforms: list[str],
    *,
    title: str | None,
    tiktok_privacy: str | None = None,
) -> tuple[dict[str, str] | None, dict[str, dict[str, Any]] | None]:
    """The ONE builder for per-target createPost extras — shared by the
    internal API and the dashboard publish path so the payload matrix
    has a single source of truth. Returns (customContent overrides,
    platformSpecificData per target).

    per_platform_captions values: a string is the caption override; a
    dict uses its "caption" key, and its "title" key rides via
    platformSpecificData (YouTube). firstComment lands ONLY on
    platforms whose spec supports it (silently skipped elsewhere).
    tiktok_privacy must be a value from TikTok's creator-info options.
    """
    content: dict[str, str] = {}
    data: dict[str, dict[str, Any]] = {}
    for p, v in options.per_platform_captions.items():
        key = p.strip().lower()
        if key not in platforms:
            continue
        if isinstance(v, str) and v:
            content[key] = v
        elif isinstance(v, dict):
            if isinstance(v.get("caption"), str) and v["caption"]:
                content[key] = v["caption"]
            if isinstance(v.get("title"), str) and v["title"]:
                data.setdefault(key, {})["title"] = v["title"]
    if "youtube" in platforms and title and "title" not in data.get("youtube", {}):
        data.setdefault("youtube", {})["title"] = title
    if options.first_comment:
        for p in platforms:
            if p in FIRST_COMMENT_PLATFORMS:
                data.setdefault(p, {})["firstComment"] = options.first_comment
    if tiktok_privacy and "tiktok" in platforms:
        data.setdefault("tiktok", {})["privacyLevel"] = tiktok_privacy
    return content or None, data or None


def _next_fallback_hour(now: _dt.datetime) -> _dt.datetime:
    """Sane best-time fallback: the next _FALLBACK_HOURS slot at least
    30 minutes out."""
    floor = now + _dt.timedelta(minutes=30)
    for day in range(0, 2):
        base = (now + _dt.timedelta(days=day)).replace(
            minute=0, second=0, microsecond=0,
        )
        for hour in sorted(_FALLBACK_HOURS):
            candidate = base.replace(hour=hour)
            if candidate >= floor:
                return candidate
    return floor.replace(minute=0, second=0, microsecond=0) + _dt.timedelta(hours=1)


def plan_batch_times(
    count: int,
    *,
    now: _dt.datetime,
    cap_per_day: int,
    existing_today: int,
    best_slots: list[dict[str, Any]] | None = None,
) -> list[_dt.datetime]:
    """Distribute `count` posts across days, at most `cap_per_day` per
    UTC day (counting `existing_today` already-planned posts on day 0).

    Used by the internal service-API batch (`hub_batch`), with its cap bounded
    by the per-platform rulebook; the dashboard publishing paths schedule
    through the rulebook directly (`chalybclip.publish.pacing`).

    Two distinct hour sources, never blended into each other day-by-day:

      - `best_slots` given → ENGAGEMENT pass: only each weekday's
        analytics-ranked hours. A weekday with no proven slot places
        nothing and spills forward — we post at known-good times, not
        generic ones.
      - FALLBACK pass: the fixed `_FALLBACK_HOURS` spread. This is the
        sole pass when there are no slots, AND a LAST-RESORT top-up when
        the engagement pass couldn't place every clip within the horizon
        (e.g. a tenant with one good slot on one weekday and a big batch).
        Without it those leftover clips would fall through to posting
        immediately — a generic-but-decent hour beats that.

    Padding *every* engagement day with generic hours is what muddied the
    best-time signal, so that never happens; fallback only fills the
    remainder after engagement has taken everything it can. Deterministic
    either way, so the batch behavior is testable.
    """
    out: list[_dt.datetime] = []
    taken: set[_dt.datetime] = set()
    floor = now + _dt.timedelta(minutes=30)  # too soon = that's mode=now

    def _fill(hours_for_day: Callable[[_dt.datetime], list[int]]) -> None:
        """Place posts at each day's hours until `out` reaches `count` or
        the horizon is exhausted. Respects `cap_per_day` across BOTH passes
        (existing_today on day 0 + whatever an earlier pass already placed)
        and never reuses a slot already chosen."""
        # Safety bound: even at 1 post/day the batch fits in count+31 days.
        for day_offset in range(count + 31):
            if len(out) >= count:
                return
            day = (now + _dt.timedelta(days=day_offset)).replace(
                minute=0, second=0, microsecond=0,
            )
            already = sum(1 for t in out if t.date() == day.date())
            base_used = existing_today if day_offset == 0 else 0
            capacity = cap_per_day - base_used - already
            if capacity <= 0:
                continue
            picked = 0
            for hour in dict.fromkeys(hours_for_day(day)):
                if picked >= capacity or len(out) >= count:
                    break
                candidate = day.replace(hour=hour)
                if candidate <= floor or candidate in taken:
                    continue  # past, or already taken by an earlier pass
                taken.add(candidate)
                out.append(candidate)
                picked += 1

    if best_slots:
        def _engagement_hours(day: _dt.datetime) -> list[int]:
            return [
                int(s["hour"])
                for s in sorted(
                    best_slots, key=lambda s: -float(s.get("avg_engagement") or 0)
                )
                if isinstance(s.get("hour"), int)
                and s.get("day_of_week") == day.weekday()
            ]

        _fill(_engagement_hours)

    if len(out) < count:
        _fill(lambda _day: sorted(_FALLBACK_HOURS))

    return sorted(out)[:count]
