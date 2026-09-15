"""Analytics normalization — turn Zernio's raw metric payloads into
the shape the Rendimiento tab and the internal analytics API render.

The hard rule (plan phase 7): NO fake zeros. A metric Zernio omits is
unavailable on that platform → it stays `None` and the UI renders "—".
A metric Zernio sends — even 0 — is a real value and shows the number.
So normalization NEVER defaults a missing key to 0.
"""
from __future__ import annotations

from typing import Any, Final

# The metric keys Zernio's `analytics` object uses (per the OpenAPI
# AnalyticsSinglePostResponse / list schemas). engagementRate is a
# derived float; the rest are counts.
METRIC_KEYS: Final[tuple[str, ...]] = (
    "impressions",
    "reach",
    "likes",
    "comments",
    "shares",
    "saves",
    "clicks",
    "views",
    "engagementRate",
)

# The subset the recent-clips table shows by default (the streamer-
# facing four + views). Order matters for the table columns.
HEADLINE_METRICS: Final[tuple[str, ...]] = (
    "views",
    "likes",
    "comments",
    "shares",
)


def _num(value: Any) -> float | int | None:
    """A metric value, or None when absent/non-numeric. A present 0 is
    kept (real zero); only missing/garbage becomes None."""
    if isinstance(value, bool):  # guard: bool is an int subclass
        return None
    if isinstance(value, int | float):
        return value
    return None


def normalize_metrics(analytics: Any) -> dict[str, float | int | None]:
    """One metric dict → {key: value|None} over METRIC_KEYS.

    A key Zernio didn't send stays None (renders "—"); a key it sent
    (incl. 0) keeps its value. Never invents a 0."""
    out: dict[str, float | int | None] = {k: None for k in METRIC_KEYS}
    if isinstance(analytics, dict):
        for k in METRIC_KEYS:
            if k in analytics:
                out[k] = _num(analytics[k])
    return out


def normalize_post(post: dict[str, Any]) -> dict[str, Any]:
    """One analytics post → a flat, render-ready row:
    {post_id, content, published_at, platform_post_url, metrics,
     per_platform:[{platform, username, url, metrics}]}.
    Metrics follow the no-fake-zeros rule throughout."""
    platforms_raw = post.get("platforms")
    if not isinstance(platforms_raw, list):
        platforms_raw = post.get("platformAnalytics")
    per_platform: list[dict[str, Any]] = []
    if isinstance(platforms_raw, list):
        for p in platforms_raw:
            if not isinstance(p, dict):
                continue
            per_platform.append(
                {
                    "platform": p.get("platform"),
                    "username": p.get("accountUsername"),
                    "url": p.get("platformPostUrl"),
                    "metrics": normalize_metrics(p.get("analytics")),
                }
            )
    return {
        "post_id": post.get("_id") or post.get("id") or post.get("postId"),
        "content": (post.get("content") or "")[:120],
        "published_at": post.get("publishedAt") or post.get("scheduledFor"),
        "platform_post_url": post.get("platformPostUrl"),
        "metrics": normalize_metrics(post.get("analytics")),
        "per_platform": per_platform,
    }


def normalize_list(body: dict[str, Any]) -> list[dict[str, Any]]:
    """The /analytics list response → render-ready post rows."""
    posts = body.get("posts")
    if not isinstance(posts, list):
        return []
    return [normalize_post(p) for p in posts if isinstance(p, dict)]


def sum_headline(rows: list[dict[str, Any]]) -> dict[str, int | None]:
    """Totals header over post rows for the headline metrics.

    A total is None (renders "—") only when EVERY row lacks that metric
    — i.e. no platform reported it at all. As soon as one row has a
    real value, the total is the sum of the available ones."""
    totals: dict[str, int | None] = {}
    for metric in HEADLINE_METRICS:
        present = [
            r["metrics"][metric]
            for r in rows
            if isinstance(r.get("metrics"), dict)
            and r["metrics"].get(metric) is not None
        ]
        totals[metric] = int(sum(present)) if present else None
    return totals


__all__ = [
    "HEADLINE_METRICS",
    "METRIC_KEYS",
    "normalize_list",
    "normalize_metrics",
    "normalize_post",
    "sum_headline",
]
