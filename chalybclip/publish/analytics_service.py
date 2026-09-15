"""Analytics service — the Rendimiento tab + daily snapshot job + the
internal analytics read for Chalyb engines (Hub phase 7).

Business logic behind the analytics routes; the routes stay thin. The
no-fake-zeros rule lives in integrations.zernio.analytics and flows
through everything here unchanged.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import dataclass
from typing import Any

from chalybclip.db import Database, TenantsRepo, ZernioPublishSnapshotsRepo
from chalybclip.integrations.zernio.analytics import normalize_list, sum_headline
from chalybclip.integrations.zernio.client import ZernioClient, ZernioError

_log = logging.getLogger("chalybclip.publish.analytics")


@dataclass(frozen=True)
class PerformanceView:
    """What the Rendimiento tab renders: post rows + totals header."""

    rows: list[dict[str, Any]]
    totals: dict[str, int | None]
    days: int


async def performance_for_tenant(
    db: Database,
    tenant_id: str,
    *,
    days: int = 30,
    client: ZernioClient | None = None,
    now: _dt.datetime | None = None,
) -> PerformanceView:
    """Recent-clips performance for the tenant over the last `days`.

    Reads LIVE from Zernio's analytics (the tab wants current numbers),
    normalized to the no-fake-zeros shape. Empty view (not an error)
    when there's no profile / no key / Zernio is unreachable, so the
    tab degrades to a clean "sin datos" state."""
    tenant = await TenantsRepo(db).get(tenant_id)
    profile_id = tenant.zernio_profile_id if tenant else None
    if profile_id is None or client is None:
        return PerformanceView(rows=[], totals=sum_headline([]), days=days)
    now_dt = now or _dt.datetime.now(_dt.UTC)
    from_date = (now_dt - _dt.timedelta(days=days)).date().isoformat()
    to_date = now_dt.date().isoformat()
    try:
        body = await client.post_analytics_list(
            profile_id=profile_id, from_date=from_date, to_date=to_date,
        )
    except ZernioError as e:
        _log.warning("analytics.performance_failed tenant=%s err=%s", tenant_id, e)
        return PerformanceView(rows=[], totals=sum_headline([]), days=days)
    rows = normalize_list(body)
    return PerformanceView(rows=rows, totals=sum_headline(rows), days=days)


async def snapshot_tenant(
    db: Database,
    tenant_id: str,
    *,
    client: ZernioClient,
    now: _dt.datetime | None = None,
) -> int:
    """Persist today's per-post metric snapshot for one tenant.

    Idempotent per (post, UTC day) — a same-day re-run refreshes rows
    instead of duplicating. Returns the number of posts snapshotted.
    Persist-only: NO scoring, NO ML — just history for the future
    clip-selection feedback loop."""
    tenant = await TenantsRepo(db).get(tenant_id)
    profile_id = tenant.zernio_profile_id if tenant else None
    if profile_id is None:
        return 0
    now_dt = now or _dt.datetime.now(_dt.UTC)
    day = now_dt.date().isoformat()
    try:
        body = await client.post_analytics_list(
            profile_id=profile_id,
            from_date=(now_dt - _dt.timedelta(days=30)).date().isoformat(),
            to_date=day,
        )
    except ZernioError as e:
        _log.warning("analytics.snapshot_failed tenant=%s err=%s", tenant_id, e)
        return 0
    rows = normalize_list(body)
    repo = ZernioPublishSnapshotsRepo(db)
    n = 0
    for row in rows:
        post_id = row.get("post_id")
        if not isinstance(post_id, str) or not post_id:
            continue
        await repo.upsert(
            tenant_id,
            post_id=post_id,
            day=day,
            metrics_json=json.dumps(row["metrics"]),
            platforms_json=json.dumps(row["per_platform"]),
        )
        n += 1
    _log.info("analytics.snapshot tenant=%s posts=%d day=%s", tenant_id, n, day)
    return n


async def internal_analytics(
    db: Database,
    tenant_id: str,
    *,
    days: int = 30,
    client: ZernioClient | None = None,
    now: _dt.datetime | None = None,
) -> dict[str, Any]:
    """Performance read for Chalyb engines (GET /internal/v1/analytics).

    Prefers LIVE Zernio analytics; falls back to the latest stored
    snapshots when Zernio is unreachable so an engine still gets data.
    Returns {tenant_id, days, source, totals, posts:[...]}."""
    view = await performance_for_tenant(
        db, tenant_id, days=days, client=client, now=now,
    )
    if view.rows:
        return {
            "tenant_id": tenant_id,
            "days": days,
            "source": "live",
            "totals": view.totals,
            "posts": view.rows,
        }
    # Live empty/failed → serve the stored snapshots.
    snaps = await ZernioPublishSnapshotsRepo(db).latest_for_tenant(tenant_id)
    posts: list[dict[str, Any]] = []
    for s in snaps:
        try:
            metrics = json.loads(s["metrics_json"])
            per_platform = json.loads(s["platforms_json"] or "[]")
        except (ValueError, TypeError):
            continue
        posts.append(
            {
                "post_id": s["post_id"],
                "day": s["day"],
                "metrics": metrics,
                "per_platform": per_platform,
            }
        )
    return {
        "tenant_id": tenant_id,
        "days": days,
        "source": "snapshot",
        "totals": sum_headline(posts),
        "posts": posts,
    }


async def send_weekly_digest(
    db: Database,
    tenant_id: str,
    *,
    client: ZernioClient,
    now: _dt.datetime | None = None,
) -> bool:
    """Post a weekly metrics digest to the tenant's community channel,
    if the digest toggle is on. Reuses the phase-7 headline totals.
    Returns True when a digest was sent. Best-effort."""
    from chalybclip.db import TenantsRepo, ZernioCommunityRepo
    from chalybclip.integrations.zernio.client import ZernioError
    from chalybclip.integrations.zernio.community import build_weekly_digest_text

    community = ZernioCommunityRepo(db)
    settings_row = await community.get_settings(tenant_id)
    if not settings_row or not settings_row.get("weekly_digest"):
        return False
    discord_id = settings_row.get("discord_account_id")
    telegram_id = settings_row.get("telegram_account_id")
    if not (discord_id or telegram_id):
        return False
    tenant = await TenantsRepo(db).get(tenant_id)
    profile_id = tenant.zernio_profile_id if tenant else None
    if not profile_id:
        return False
    view = await performance_for_tenant(
        db, tenant_id, days=7, client=client, now=now,
    )
    text = build_weekly_digest_text(view.totals, days=7)
    platforms: list[tuple[str, str]] = []
    if discord_id:
        platforms.append(("discord", discord_id))
    if telegram_id:
        platforms.append(("telegram", telegram_id))
    try:
        await client.create_community_post(
            profile_id=profile_id, content=text, platforms=platforms,
        )
    except ZernioError as e:
        _log.warning("analytics.digest_failed tenant=%s err=%s", tenant_id, e)
        return False
    _log.info("analytics.digest_sent tenant=%s", tenant_id)
    return True


__all__ = [
    "PerformanceView",
    "internal_analytics",
    "performance_for_tenant",
    "send_weekly_digest",
    "snapshot_tenant",
]
