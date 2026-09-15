"""Channel-poll loop — auto-ingest new VODs from creator channels.

The auto-ingest counterpart to `chalybclip.drive.service`. Per-watch:

  1. List the channel's recent uploads via yt-dlp (flat — metadata only,
     no media download).
  2. For each video whose id is NOT in `seen_video_ids`, call
     `ingest_callback(tenant_id, vod_url, video_id, persona_id, language)`
     — which ingests the VOD and kicks off the pipeline.
  3. Append ingested ids to `seen_video_ids`; advance `last_polled_at`
     only on a clean pass (mirrors the drive watcher's failure handling).

First-poll guard: a brand-new watch only ingests the newest
`max_per_poll` videos and marks the rest of the listed window as seen, so
connecting a channel with a deep back-catalog doesn't flood the pipeline.
Later polls ingest up to `max_per_poll` fresh uploads per pass.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import re
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

import structlog

from chalybclip.db import ChannelWatchesRepo, Database, TenantsRepo
from chalybclip.db.models import ChannelWatchRow
from chalybclip.tenancy import bound_tenant

from .models import ChannelPollReport, ChannelVOD

_log = structlog.get_logger(__name__)

# How many entries to pull off the channel each poll. We list a window a
# bit wider than the per-poll ingest cap so a burst of uploads between
# polls is still caught (deduped against the seen-set).
_DEFAULT_LIST_WINDOW = 10

# Slack on the due-check so the loop's tick granularity doesn't push each
# day's polls progressively later (a 1/day watch becomes due a touch
# before the full 24h rather than always just after).
_DUE_SLACK = _dt.timedelta(minutes=5)

# How many polls a failing video is retried before it's parked. A transient
# error (YouTube bot-check, a flaky network, a not-yet-published premiere)
# usually clears within a poll or two; a genuinely-unavailable video (private
# / deleted / region-locked — yt-dlp "This video is not available") never
# will. After this many failures we add it to the seen-set and stop retrying
# it, so it doesn't fail on every single poll forever.
_MAX_INGEST_ATTEMPTS = 3


ChannelIngestCallback = Callable[[str, str, str, str, str | None], Awaitable[None]]
"""(tenant_id, vod_url, video_id, persona_id, language) -> None.

The poller hands you a freshly-detected VOD; you ingest it and kick off
the pipeline. In production this is `make_channel_ingest_callback`."""


class ChannelVODLister(Protocol):
    """Lists a channel's recent videos. Injected so tests can fake it."""

    async def __call__(
        self, platform: str, channel_url: str, *, limit: int
    ) -> list[ChannelVOD]: ...


# ---- VOD listing (yt-dlp, flat) ----


def _list_channel_vods_sync(channel_url: str, limit: int) -> list[dict[str, object]]:
    # Imported here so yt-dlp stays out of module-load time.
    import yt_dlp

    from chalybclip.settings import get_settings

    s = get_settings()
    opts: dict[str, object] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        # Flat: list playlist/channel entries without resolving each video.
        "extract_flat": "in_playlist",
        "playlistend": limit,
    }
    # Same egress auth as the VOD download path — a channel listing hits the
    # same bot-gate / Cloudflare, so route it through the residential proxy
    # and any configured cookies. Without this, auto-polling silently 403s.
    proxy = (getattr(s, "ytdlp_proxy", None) or "").strip()
    if proxy:
        opts["proxy"] = proxy
    if s.cookies_file:
        opts["cookiefile"] = str(s.cookies_file)
    elif s.cookies_from_browser:
        opts["cookiesfrombrowser"] = (s.cookies_from_browser.strip().lower(),)
    # Same free YouTube bot-gate dodge as the download path (player_client
    # / PO-token) — a channel listing hits the same gate.
    from chalybclip.ingest.service import youtube_extractor_args
    yt_args = youtube_extractor_args(s)
    if yt_args:
        opts["extractor_args"] = {"youtube": yt_args}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
    entries = (info or {}).get("entries") or []
    return [e for e in entries if isinstance(e, dict)]


def _canonical_url(platform: str, entry: dict[str, object]) -> str | None:
    """Resolve a flat entry to a single-video URL ingest_vod can download."""
    url = entry.get("url")
    if isinstance(url, str) and url.startswith("http"):
        return url
    vid = entry.get("id")
    if not isinstance(vid, str) or not vid:
        return url if isinstance(url, str) and url else None
    if platform == "youtube":
        return f"https://www.youtube.com/watch?v={vid}"
    if platform == "twitch":
        return f"https://www.twitch.tv/videos/{vid}"
    # Kick / unknown — fall back to whatever flat gave us.
    return url if isinstance(url, str) and url else None


# ---- Kick VOD listing (Kick API via browser impersonation) ----
#
# yt-dlp has NO Kick channel-VOD-list extractor (only kick:live/vod/clips),
# so kick.com/<slug>/videos falls through to kick:live and 403s behind
# Cloudflare. We hit Kick's own API instead, with curl_cffi impersonating a
# browser so Cloudflare lets us through. The VOD download (ingest_vod) gets
# the same impersonation via yt-dlp's `impersonate` target.

_KICK_SLUG_RE = re.compile(r"kick\.com/(?P<slug>[\w-]+)", re.IGNORECASE)
# Path segments that are NOT a channel slug (kick.com/video/... etc.).
_KICK_RESERVED = frozenset({"video", "videos", "api", "search", "categories"})


def _kick_channel_slug(channel_url: str) -> str | None:
    """Extract the channel slug from a Kick channel URL (with or without
    a trailing `/videos`). Returns None when the URL isn't a channel."""
    m = _KICK_SLUG_RE.search(channel_url)
    if not m:
        return None
    slug = m.group("slug")
    return None if slug.lower() in _KICK_RESERVED else slug


def _fetch_kick_vods_sync(slug: str) -> list[dict[str, Any]]:
    # curl_cffi imported here so it stays out of module-load time and a
    # missing wheel only bites the Kick path, not the whole service.
    from curl_cffi import requests as _creq

    from chalybclip.settings import get_settings

    url = f"https://kick.com/api/v2/channels/{slug}/videos"
    # Route through the same residential proxy as the yt-dlp paths so Kick's
    # Cloudflare sees a home IP, not the datacenter. curl_cffi takes the
    # standard requests `proxies` mapping.
    proxy = (getattr(get_settings(), "ytdlp_proxy", None) or "").strip()
    proxies = {"http": proxy, "https": proxy} if proxy else None
    resp = _creq.get(url, impersonate="chrome", timeout=30, proxies=proxies)
    resp.raise_for_status()
    data = resp.json()
    return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else []


async def list_kick_channel_vods(
    channel_url: str, *, limit: int
) -> list[ChannelVOD]:
    """List a Kick channel's newest `limit` VODs via Kick's API.

    Each entry's `video.uuid` is the stable id; the ingestable URL is
    `kick.com/<slug>/videos/<uuid>` (matches yt-dlp's kick:vod extractor).
    """
    slug = _kick_channel_slug(channel_url)
    if not slug:
        raise ValueError(f"not a Kick channel URL: {channel_url!r}")
    entries = await asyncio.to_thread(_fetch_kick_vods_sync, slug)
    # Newest-first regardless of API order (created_at is ISO-ish, sorts
    # lexically). Defensive: don't rely on Kick's response ordering.
    entries.sort(key=lambda e: str(e.get("created_at") or ""), reverse=True)
    vods: list[ChannelVOD] = []
    for e in entries[:limit]:
        video = e.get("video")
        uuid = video.get("uuid") if isinstance(video, dict) else None
        if not isinstance(uuid, str) or not uuid:
            continue
        title = e.get("session_title")
        vods.append(
            ChannelVOD(
                video_id=uuid,
                url=f"https://kick.com/{slug}/videos/{uuid}",
                title=title if isinstance(title, str) and title else None,
            )
        )
    return vods


async def list_channel_vods(
    platform: str, channel_url: str, *, limit: int
) -> list[ChannelVOD]:
    """List a channel's newest `limit` videos (newest first), no download."""
    if platform == "kick":
        return await list_kick_channel_vods(channel_url, limit=limit)
    entries = await asyncio.to_thread(_list_channel_vods_sync, channel_url, limit)
    vods: list[ChannelVOD] = []
    for e in entries[:limit]:
        vid = e.get("id")
        url = _canonical_url(platform, e)
        if not isinstance(vid, str) or not vid or not url:
            continue
        title = e.get("title")
        vods.append(
            ChannelVOD(
                video_id=vid,
                url=url,
                title=title if isinstance(title, str) else None,
            )
        )
    return vods


# ---- public entry ----


def _is_due(watch: ChannelWatchRow, now: _dt.datetime) -> bool:
    """Has this watch's scheduled cadence elapsed since its last poll?

    `polls_per_day` derives the interval (24h / N); a small slack absorbs
    the loop's tick granularity so the cadence doesn't drift later each
    day. A never-polled watch is always due."""
    if watch.last_polled_at is None:
        return True
    ppd = max(watch.polls_per_day, 1)
    interval = _dt.timedelta(hours=24 / ppd)
    try:
        last = _dt.datetime.fromisoformat(watch.last_polled_at)
    except ValueError:
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=_dt.UTC)
    return (now - last) >= (interval - _DUE_SLACK)


async def poll_channel_watches(
    db: Database,
    *,
    ingest_callback: ChannelIngestCallback,
    list_vods: ChannelVODLister = list_channel_vods,
    tenant_id: str | None = None,
    watch_id: str | None = None,
    respect_schedule: bool = False,
    now: _dt.datetime | None = None,
) -> list[ChannelPollReport]:
    """Poll every enabled channel watch for new VODs.

    Args:
        db: Database handle. Migrations must already be applied.
        ingest_callback: Called per newly-detected VOD with
            `(tenant_id, vod_url, video_id, persona_id, language)`.
        list_vods: Channel lister (defaults to the yt-dlp one). Injected
            in tests.
        tenant_id: When set, restricts to that tenant; otherwise polls
            every tenant.
        watch_id: When set, restricts to that single watch (within the
            tenant). Used by the dashboard's on-demand "scan now" button so
            the operator can pull a channel immediately instead of waiting
            for its cadence. Pair with `respect_schedule=False`.
        respect_schedule: When True (the background loop), skip watches
            whose `polls_per_day` cadence hasn't elapsed — no yt-dlp call.
            When False (manual `channel poll` / tests), poll regardless.
        now: Clock override for the due-check (tests).

    Returns:
        One report per watch row scanned (including schedule-skips).
    """
    now_dt = now or _dt.datetime.now(_dt.UTC)
    tenants_repo = TenantsRepo(db)
    if tenant_id is not None:
        t = await tenants_repo.get(tenant_id)
        tenants = [t] if t is not None else []
    else:
        tenants = await tenants_repo.list_all()

    reports: list[ChannelPollReport] = []
    for t in tenants:
        with bound_tenant(t.id):
            watches = await ChannelWatchesRepo(db).list_for_tenant()
            if watch_id is not None:
                watches = [w for w in watches if w.id == watch_id]
            for watch in watches:
                if (
                    respect_schedule
                    and watch.enabled
                    and not _is_due(watch, now_dt)
                ):
                    reports.append(
                        ChannelPollReport(
                            watch_id=watch.id,
                            tenant_id=watch.tenant_id,
                            channel_url=watch.channel_url,
                            videos_seen=0,
                            videos_ingested=0,
                            videos_failed=0,
                            skipped_disabled=False,
                            skipped_not_due=True,
                        )
                    )
                    continue
                report = await _poll_one_watch(
                    db=db,
                    watch=watch,
                    ingest_callback=ingest_callback,
                    list_vods=list_vods,
                )
                reports.append(report)
                _log.info(
                    "channel.poll_done",
                    watch_id=watch.id,
                    tenant_id=watch.tenant_id,
                    platform=watch.platform,
                    videos_seen=report.videos_seen,
                    videos_ingested=report.videos_ingested,
                    videos_failed=report.videos_failed,
                    skipped_disabled=report.skipped_disabled,
                )
    return reports


async def _poll_one_watch(
    *,
    db: Database,
    watch: ChannelWatchRow,
    ingest_callback: ChannelIngestCallback,
    list_vods: ChannelVODLister,
) -> ChannelPollReport:
    """Per-watch body. Tenant is already bound."""
    if not watch.enabled:
        return ChannelPollReport(
            watch_id=watch.id,
            tenant_id=watch.tenant_id,
            channel_url=watch.channel_url,
            videos_seen=0,
            videos_ingested=0,
            videos_failed=0,
            skipped_disabled=True,
        )

    window = max(watch.max_per_poll, _DEFAULT_LIST_WINDOW)
    try:
        vods = await list_vods(watch.platform, watch.channel_url, limit=window)
    except Exception as e:
        _log.warning(
            "channel.list_failed",
            watch_id=watch.id,
            platform=watch.platform,
            channel_url=watch.channel_url,
            error=str(e),
        )
        return ChannelPollReport(
            watch_id=watch.id,
            tenant_id=watch.tenant_id,
            channel_url=watch.channel_url,
            videos_seen=0,
            videos_ingested=0,
            videos_failed=1,
            skipped_disabled=False,
        )

    seen = set(watch.seen_video_ids)
    fail_counts = dict(watch.failed_video_attempts)
    new_vods = [v for v in vods if v.video_id not in seen]

    if watch.last_polled_at is None:
        # First sight of this channel: ingest only the newest few; treat the
        # rest of the listed window as back-catalog and mark it seen so it
        # never reprocesses.
        to_ingest = new_vods[: watch.max_per_poll]
        for v in new_vods[watch.max_per_poll :]:
            seen.add(v.video_id)
    else:
        # Steady state: ingest fresh uploads, capped per pass as a flood
        # guard. Any overflow stays unseen and lands on the next poll.
        to_ingest = new_vods[: watch.max_per_poll]

    ingested = 0
    failed = 0
    for v in to_ingest:
        try:
            await ingest_callback(
                watch.tenant_id, v.url, v.video_id, watch.persona_id, watch.language
            )
            seen.add(v.video_id)
            fail_counts.pop(v.video_id, None)  # cleared on success
            ingested += 1
        except Exception as e:
            failed += 1
            attempts = fail_counts.get(v.video_id, 0) + 1
            if attempts >= _MAX_INGEST_ATTEMPTS:
                # Retried to the ceiling and still failing — treat as
                # permanently unavailable. Park it in the seen-set so it
                # stops being retried (and reported as a failure) every poll.
                seen.add(v.video_id)
                fail_counts.pop(v.video_id, None)
                _log.warning(
                    "channel.video_parked",
                    watch_id=watch.id,
                    video_id=v.video_id,
                    vod_url=v.url,
                    attempts=attempts,
                    error=str(e),
                )
            else:
                # Stays UNSEEN so it retries next poll, with the bumped count.
                fail_counts[v.video_id] = attempts
                _log.warning(
                    "channel.ingest_failed",
                    watch_id=watch.id,
                    video_id=v.video_id,
                    vod_url=v.url,
                    attempts=attempts,
                    error=str(e),
                )

    # The listing succeeded (a listing failure returns early above), so the
    # poll itself ran — advance last_polled_at unconditionally. Per-video
    # ingest failures must NOT freeze it: a transiently-failed video stays
    # UNSEEN (only successes are added to `seen` above) so it retries on the
    # next scheduled poll, with its bumped failure count persisted. After
    # `_MAX_INGEST_ATTEMPTS` it's parked into `seen` instead, so a genuinely
    # dead video stops failing every poll. Freezing on any failure was the bug
    # behind "última revisión: nunca" — one perpetually-failing video (e.g. a
    # YouTube bot-check) kept the watch permanently "due", which both hid the
    # real last-checked time AND re-listed the channel every loop tick.
    new_polled_at = _now_iso()
    await ChannelWatchesRepo(db).mark_polled(
        watch.id,
        seen_video_ids=sorted(seen),
        last_polled_at=new_polled_at,
        failed_video_attempts=fail_counts,
    )

    return ChannelPollReport(
        watch_id=watch.id,
        tenant_id=watch.tenant_id,
        channel_url=watch.channel_url,
        videos_seen=len(vods),
        videos_ingested=ingested,
        videos_failed=failed,
        skipped_disabled=False,
    )


# ---- helpers ----


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()
