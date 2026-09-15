"""Per-platform engagement-metric fetchers.

Each fetcher is a coroutine that takes a `(publish_job, account, http)`
triple and returns a `NormalizedMetric`. Stats absent from a platform's
response stay `None` rather than zero - the calibration loop skips
NULL columns rather than treating them as zero engagement.

Phase 3 ships:
  * YouTube Data API v3      `videos.list?part=statistics`
  * TikTok analytics         /v2/research/video/query/  (research API
                             tier; for sandbox accounts we simply
                             record `views=None` which the calibration
                             skips - no fabricated zeros).
  * Buffer                   no analytics endpoint we use; returns a
                             trivial NormalizedMetric so the worker
                             still records the fetch attempt for audit.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from chalybclip.db.models import ConnectedAccount, PublishJob

_log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class NormalizedMetric:
    """Normalized stats payload that the service writes via the repo."""

    views: int | None = None
    likes: int | None = None
    comments: int | None = None
    shares: int | None = None
    retention_pct: float | None = None
    ctr: float | None = None
    raw_metadata: dict[str, Any] | None = None


# A fetcher takes the publish_job + account + an httpx client + the OAuth
# token (already refreshed by the caller). Returns the normalized metric or
# raises on non-recoverable errors.
FetcherProtocol = Callable[
    [PublishJob, ConnectedAccount, httpx.AsyncClient, str],
    Awaitable[NormalizedMetric],
]


# ---------------------------------------------------------------------------
# YouTube
# ---------------------------------------------------------------------------


_YT_VIDEOS_LIST_URL = "https://www.googleapis.com/youtube/v3/videos"


async def fetch_youtube_metric(
    job: PublishJob,
    account: ConnectedAccount,
    http: httpx.AsyncClient,
    access_token: str,
) -> NormalizedMetric:
    """Pull statistics for one video from the YouTube Data API.

    `external_id` on the publish_job is the YouTube videoId recorded when
    the post was published.
    """
    if not job.external_id:
        return NormalizedMetric(raw_metadata={"skipped": "no external_id"})

    resp = await http.get(
        _YT_VIDEOS_LIST_URL,
        params={"part": "statistics,contentDetails", "id": job.external_id},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    if resp.status_code >= 400:
        # Non-fatal for the worker - we still record the fetch attempt as a
        # NULL-stats row so the audit trail shows we tried.
        _log.warning(
            "youtube_stats_http_error",
            video_id=job.external_id,
            status=resp.status_code,
            body=resp.text[:200],
        )
        return NormalizedMetric(
            raw_metadata={
                "error": f"youtube {resp.status_code}",
                "body": resp.text[:500],
            }
        )

    body = resp.json()
    items = body.get("items") if isinstance(body, dict) else None
    if not items or not isinstance(items, list):
        return NormalizedMetric(raw_metadata={"empty_response": True})
    item = items[0]
    stats = item.get("statistics") or {}
    return NormalizedMetric(
        views=_to_int(stats.get("viewCount")),
        likes=_to_int(stats.get("likeCount")),
        comments=_to_int(stats.get("commentCount")),
        shares=None,  # YT Data API doesn't expose shares to most scopes
        retention_pct=None,  # requires the YT Analytics API (separate scope)
        ctr=None,
        raw_metadata=item,
    )


# ---------------------------------------------------------------------------
# TikTok
# ---------------------------------------------------------------------------


_TT_QUERY_URL = "https://open.tiktokapis.com/v2/research/video/query/"


async def fetch_tiktok_metric(
    job: PublishJob,
    account: ConnectedAccount,
    http: httpx.AsyncClient,
    access_token: str,
) -> NormalizedMetric:
    """Pull stats for one TikTok video.

    The Research API tier exposes view/like/comment/share counts; sandbox
    accounts return 403 and we record a NULL-stats row. Either way the
    worker stays alive.
    """
    if not job.external_id:
        return NormalizedMetric(raw_metadata={"skipped": "no external_id"})

    resp = await http.post(
        _TT_QUERY_URL,
        params={"fields": "id,view_count,like_count,comment_count,share_count"},
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json={
            "query": {"and": [{"operation": "EQ", "field_name": "id", "field_values": [job.external_id]}]},
            "max_count": 1,
        },
    )
    if resp.status_code >= 400:
        _log.warning(
            "tiktok_stats_http_error",
            video_id=job.external_id,
            status=resp.status_code,
            body=resp.text[:200],
        )
        return NormalizedMetric(
            raw_metadata={
                "error": f"tiktok {resp.status_code}",
                "body": resp.text[:500],
            }
        )

    body = resp.json()
    data = body.get("data") if isinstance(body, dict) else None
    videos = (data or {}).get("videos") or []
    if not videos:
        return NormalizedMetric(raw_metadata={"empty_response": True})
    item = videos[0]
    return NormalizedMetric(
        views=_to_int(item.get("view_count")),
        likes=_to_int(item.get("like_count")),
        comments=_to_int(item.get("comment_count")),
        shares=_to_int(item.get("share_count")),
        retention_pct=None,
        ctr=None,
        raw_metadata=item,
    )


# ---------------------------------------------------------------------------
# Instagram
# ---------------------------------------------------------------------------


_IG_GRAPH_BASE = "https://graph.facebook.com/v22.0"
_IG_INSIGHTS_METRICS = "plays,likes,comments,shares,reach"


async def fetch_instagram_metric(
    job: PublishJob,
    account: ConnectedAccount,
    http: httpx.AsyncClient,
    access_token: str,
) -> NormalizedMetric:
    """Pull stats for one IG Reels post via the Graph Insights API.

    `external_id` on the publish_job is the IG media id recorded when the
    post was published.
    Sandbox apps return 200 with an empty `data` array; we record the
    audit row regardless.
    """
    if not job.external_id:
        return NormalizedMetric(raw_metadata={"skipped": "no external_id"})

    resp = await http.get(
        f"{_IG_GRAPH_BASE}/{job.external_id}/insights",
        params={
            "metric": _IG_INSIGHTS_METRICS,
            "access_token": access_token,
        },
    )
    if resp.status_code >= 400:
        _log.warning(
            "instagram_stats_http_error",
            media_id=job.external_id,
            status=resp.status_code,
            body=resp.text[:200],
        )
        return NormalizedMetric(
            raw_metadata={
                "error": f"instagram {resp.status_code}",
                "body": resp.text[:500],
            }
        )

    body = resp.json()
    items = body.get("data") if isinstance(body, dict) else None
    if not items or not isinstance(items, list):
        return NormalizedMetric(raw_metadata={"empty_response": True})

    # Insights API returns one row per metric: each row has `name` + `values`
    # (a list of {value, end_time}). We take the latest value per metric.
    pulled: dict[str, int | None] = {}
    for row in items:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        values = row.get("values")
        if not isinstance(name, str) or not isinstance(values, list) or not values:
            continue
        latest = values[-1]
        if isinstance(latest, dict):
            pulled[name] = _to_int(latest.get("value"))

    return NormalizedMetric(
        views=pulled.get("plays"),
        likes=pulled.get("likes"),
        comments=pulled.get("comments"),
        shares=pulled.get("shares"),
        retention_pct=None,  # IG doesn't expose retention on this scope
        ctr=None,
        raw_metadata={"insights": items},
    )


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------


async def fetch_buffer_metric(
    job: PublishJob,
    account: ConnectedAccount,
    http: httpx.AsyncClient,
    access_token: str,
) -> NormalizedMetric:
    """Buffer doesn't expose analytics on the tier we use.

    We still record an empty fetch so the audit trail can show we tried.
    Phase 3+ swap-in: pull from the user's downstream platform via the
    Buffer-recorded external_id when that becomes available.
    """
    return NormalizedMetric(raw_metadata={"skipped": "buffer_no_analytics"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
