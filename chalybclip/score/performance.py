"""Continuous learning — turn real per-platform performance into allocation
weights so the engine sends clips where they actually win views.

The spec's last principle: "every post updates the account's profile so the
next publishing decisions become smarter." This reads the per-post, per-platform
analytics (views) we already collect and produces a 0-1 weight per platform.
Allocation scales each platform's daily capacity by its weight, so a platform
that keeps earning ~0 views (e.g. YouTube on a channel it doesn't fit) gets
fewer clips and the winners get more.

Cold-start safety is the whole game: a brand-new channel's 0 views are "not
sampled yet", NOT "bad fit". So a platform only earns a reduced weight once it
has `min_mature_posts` posts with a REAL measured views number (Zernio's
no-fake-zeros rule leaves unmeasured views as None — those don't count). Until
then it stays at weight 1.0 (full allocation) so we keep gathering data. And a
penalized platform never drops below `_FLOOR` — it keeps a trickle so it can
recover if it starts performing.

Pure + deterministic: analytics in, weights out. Tested without a network.
"""
from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from chalybclip.publish.pacing import canonical_platform

# A penalized platform keeps at least this share of its capacity — enough to
# re-test it, never a hard zero (which would stop us ever re-learning).
_FLOOR = 0.2


@dataclass(frozen=True)
class PlatformPerf:
    platform: str
    mature_posts: int  # posts with a real measured views number, old enough
    avg_views: float
    learning: bool  # too little data yet → don't penalize
    weight: float  # 0-1 allocation multiplier


def _views(metrics: Any) -> float | None:
    if not isinstance(metrics, dict):
        return None
    v = metrics.get("views")
    if isinstance(v, bool) or not isinstance(v, int | float):
        return None
    return float(v)


def _mature(published_at: Any, now: _dt.datetime, maturity_hours: float) -> bool:
    """Old enough to trust its view count. Unknown timestamp → treat as mature
    (the snapshot fallback carries a coarse `day`, not a full timestamp; a
    measured views number on it is already a day-old reading)."""
    if not isinstance(published_at, str) or not published_at:
        return True
    try:
        ts = _dt.datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=_dt.UTC)
    return (now - ts) >= _dt.timedelta(hours=maturity_hours)


def compute_platform_performance(
    posts: Sequence[dict[str, Any]],
    *,
    min_mature_posts: int = 4,
    maturity_hours: float = 24.0,
    now: _dt.datetime | None = None,
) -> dict[str, PlatformPerf]:
    """Per-platform performance from the analytics `posts` list (the shape
    `internal_analytics` returns: each post has `per_platform:[{platform,
    metrics:{views}}]` and optionally `published_at`)."""
    now = now or _dt.datetime.now(_dt.UTC)
    samples: dict[str, list[float]] = {}
    for post in posts:
        published_at = post.get("published_at") or post.get("publishedAt")
        per_platform = post.get("per_platform")
        if not isinstance(per_platform, list):
            continue
        if not _mature(published_at, now, maturity_hours):
            continue
        for entry in per_platform:
            if not isinstance(entry, dict):
                continue
            views = _views(entry.get("metrics"))
            if views is None:
                continue  # unmeasured → not a real sample
            key = canonical_platform(str(entry.get("platform") or ""))
            if key:
                samples.setdefault(key, []).append(views)

    # Rank by average views among platforms that have enough data; learning
    # platforms (too few samples) stay at full weight.
    avgs: dict[str, float] = {
        p: (sum(v) / len(v)) for p, v in samples.items()
        if len(v) >= min_mature_posts
    }
    best = max(avgs.values(), default=0.0)

    out: dict[str, PlatformPerf] = {}
    for p, vlist in samples.items():
        n = len(vlist)
        avg = sum(vlist) / n if n else 0.0
        learning = p not in avgs
        if learning:
            weight = 1.0
        elif best <= 0:
            weight = 1.0  # nobody's getting views yet → don't tilt
        else:
            weight = max(_FLOOR, min(1.0, avg / best))
        out[p] = PlatformPerf(
            platform=p, mature_posts=n, avg_views=round(avg, 1),
            learning=learning, weight=round(weight, 3),
        )
    return out


def platform_weights(perf: dict[str, PlatformPerf]) -> dict[str, float]:
    """Just the {platform: weight} map allocation consumes."""
    return {p: s.weight for p, s in perf.items()}
