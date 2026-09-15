"""Metrics-ingest worker.

`run_metrics_ingest(tenant_id, db, ...)` walks every `sent` publish_job
for the tenant whose latest `publish_metrics` reading is older than
`refetch_after_s` (default: 1 hour). For each job it:

  1. Pulls the matching ConnectedAccount + access token (refreshing
     when near expiry, same path as the publisher).
  2. Picks the platform's fetcher.
  3. Calls the fetcher with a shared httpx client.
  4. Writes one publish_metrics row regardless of whether stats came
     back populated - even an empty fetch creates an audit trail.

Failure modes are local: one platform 5xx skips that job and logs
`metrics.fetch_failed`; the rest of the drain continues.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from chalybclip.db import (
    ConnectedAccountsRepo,
    Database,
    PublishJobsRepo,
    PublishMetricsRepo,
)
from chalybclip.db.models import ConnectedAccount, PublishJob
from chalybclip.tenancy import bound_tenant

from .fetchers import (
    FetcherProtocol,
    fetch_buffer_metric,
    fetch_instagram_metric,
    fetch_tiktok_metric,
    fetch_youtube_metric,
)

_log = structlog.get_logger(__name__)

_DEFAULT_FETCHERS: dict[str, FetcherProtocol] = {
    "youtube": fetch_youtube_metric,
    "tiktok": fetch_tiktok_metric,
    "instagram": fetch_instagram_metric,
    "buffer": fetch_buffer_metric,
}

# Skip jobs whose latest reading is younger than this window.
_DEFAULT_REFETCH_AFTER_S = 3600.0


@dataclass(frozen=True)
class IngestOutcome:
    """Roll-up of one drain pass."""

    fetched: int
    skipped_recent: int
    skipped_no_account: int
    failed: int


def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC)


def _is_stale(latest_fetched_at: str | None, *, refetch_after_s: float, now: _dt.datetime) -> bool:
    if latest_fetched_at is None:
        return True
    try:
        last = _dt.datetime.fromisoformat(latest_fetched_at)
    except ValueError:
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=_dt.UTC)
    return (now - last).total_seconds() >= refetch_after_s


async def run_metrics_ingest(
    tenant_id: str,
    db: Database,
    *,
    http: httpx.AsyncClient | None = None,
    fetchers: dict[str, FetcherProtocol] | None = None,
    refetch_after_s: float = _DEFAULT_REFETCH_AFTER_S,
    clock: Callable[[], _dt.datetime] | None = None,
    max_jobs: int = 200,
) -> IngestOutcome:
    """Drain stats for every eligible `sent` publish_job under `tenant_id`.

    `fetchers` overrides the default platform map (tests pass stubs).
    `refetch_after_s` controls how stale a row has to be before we re-fetch.
    """
    fetcher_map = fetchers if fetchers is not None else _DEFAULT_FETCHERS
    own_http = http is None
    http_client = http or httpx.AsyncClient(timeout=30.0)
    now = (clock or _utc_now)()

    fetched = 0
    skipped_recent = 0
    skipped_no_account = 0
    failed = 0

    try:
        with bound_tenant(tenant_id):
            jobs_repo = PublishJobsRepo(db)
            metrics_repo = PublishMetricsRepo(db)
            accounts_repo = ConnectedAccountsRepo(db)

            sent_jobs = await _list_sent_jobs(db, tenant_id, limit=max_jobs)
            if not sent_jobs:
                return IngestOutcome(0, 0, 0, 0)

            accounts_index = {
                a.id: a for a in await accounts_repo.list_for_tenant()
            }

            for job in sent_jobs:
                latest = await metrics_repo.latest_for_job(job.id)
                if not _is_stale(
                    latest.fetched_at if latest else None,
                    refetch_after_s=refetch_after_s,
                    now=now,
                ):
                    skipped_recent += 1
                    continue

                account = accounts_index.get(job.account_id)
                if account is None:
                    skipped_no_account += 1
                    continue
                token = _access_token_for(account)
                if not token:
                    skipped_no_account += 1
                    continue

                fetcher = fetcher_map.get(job.platform)
                if fetcher is None:
                    _log.info(
                        "metrics_no_fetcher_for_platform",
                        platform=job.platform,
                        job_id=job.id,
                    )
                    failed += 1
                    continue

                try:
                    metric = await fetcher(job, account, http_client, token)
                except Exception as e:
                    _log.warning(
                        "metrics_fetch_failed",
                        job_id=job.id,
                        platform=job.platform,
                        error=str(e),
                    )
                    failed += 1
                    continue

                await metrics_repo.record(
                    publish_job_id=job.id,
                    platform=job.platform,
                    fetched_at=now.isoformat(),
                    views=metric.views,
                    likes=metric.likes,
                    comments=metric.comments,
                    shares=metric.shares,
                    retention_pct=metric.retention_pct,
                    ctr=metric.ctr,
                    raw_metadata=metric.raw_metadata,
                )
                fetched += 1

            _ = jobs_repo  # used implicitly via _list_sent_jobs
    finally:
        if own_http:
            await http_client.aclose()

    if fetched or failed or skipped_recent or skipped_no_account:
        _log.info(
            "metrics_drain",
            tenant_id=tenant_id,
            fetched=fetched,
            skipped_recent=skipped_recent,
            skipped_no_account=skipped_no_account,
            failed=failed,
        )
    return IngestOutcome(
        fetched=fetched,
        skipped_recent=skipped_recent,
        skipped_no_account=skipped_no_account,
        failed=failed,
    )


async def _list_sent_jobs(
    db: Database, tenant_id: str, *, limit: int
) -> list[PublishJob]:
    """All `sent` publish_jobs for this tenant, newest first."""
    conn = await db.connect()
    cur = await conn.execute(
        "SELECT * FROM publish_jobs WHERE tenant_id = ? AND status = 'sent' "
        "ORDER BY created_at DESC LIMIT ?",
        (tenant_id, limit),
    )
    rows = await cur.fetchall()
    out: list[PublishJob] = []
    import json

    for r in rows:
        d = dict(r)
        meta = d.pop("platform_metadata_json", None)
        d["platform_metadata"] = json.loads(meta) if meta else None
        out.append(PublishJob.model_validate(d))
    return out


def _access_token_for(account: ConnectedAccount) -> str | None:
    blob: dict[str, Any] = account.oauth_blob or {}
    raw = blob.get("access_token")
    return raw if isinstance(raw, str) and raw else None
