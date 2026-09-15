"""Vision-rescore calibration loop.

Pairs each `sent` publish_job with:
    * the source candidate's `rescore_score` (the LLM's pre-publish prediction)
    * the latest `publish_metrics` reading (the actual outcome - views)

Computes Pearson r per platform. The dashboard's calibration page renders
the scatter as a table; future Phase 3 work uses this same shape to gate
auto-publish thresholds (the user's "earned automation" rule from
PHASE_3.md).

This is deliberately *not* a generic stats library - it's the smallest
piece of math the calibration loop needs:
    Pearson r over (rescore_score, views) for one platform.

When a job lacks either a rescore (not all clips get rescored) or a metric
reading, we drop it from the correlation but keep it in the per-clip
table so reviewers see "we don't have a verdict yet" cells.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass

from chalybclip.db import (
    CandidatesRepo,
    ClipsRepo,
    Database,
    PublishMetricsRepo,
)
from chalybclip.tenancy import current_tenant_id


@dataclass(frozen=True)
class CalibrationRow:
    """One (clip, rescore, outcome) tuple for the dashboard table."""

    publish_job_id: str
    clip_id: str
    rescore_score: float | None
    views: int | None
    likes: int | None
    fetched_at: str | None


@dataclass(frozen=True)
class CalibrationReport:
    """Result of one platform's calibration scan."""

    platform: str
    rows: list[CalibrationRow]
    pearson_r: float | None  # None when fewer than 3 paired samples
    n_paired: int  # rows with BOTH a rescore_score AND views


def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC)


async def compute_calibration(
    db: Database,
    *,
    platform: str,
    since_days: int = 30,
    limit: int = 500,
) -> CalibrationReport:
    """Build the calibration report for one platform.

    Walks every metric reading for the bound tenant in the last `since_days`,
    collapses to the latest reading per publish_job, joins back to the clip
    + source candidate, and computes Pearson r over the paired
    (rescore_score, views) values.
    """
    tenant_id = current_tenant_id()  # noqa: F841 - asserted via repo calls
    since = (_utc_now() - _dt.timedelta(days=since_days)).isoformat()

    metrics = await PublishMetricsRepo(db).latest_per_job_since(
        platform=platform, since=since, limit=limit
    )
    if not metrics:
        return CalibrationReport(platform=platform, rows=[], pearson_r=None, n_paired=0)

    # Look up each metric's publish_job + clip + source candidate.
    rows: list[CalibrationRow] = []
    paired: list[tuple[float, float]] = []
    for metric in metrics:
        job = await _get_publish_job(db, metric.publish_job_id)
        if job is None:
            continue
        clip = await ClipsRepo(db).get(job.clip_id)
        if clip is None or clip.candidate_id is None:
            rows.append(
                CalibrationRow(
                    publish_job_id=job.id,
                    clip_id=job.clip_id,
                    rescore_score=None,
                    views=metric.views,
                    likes=metric.likes,
                    fetched_at=metric.fetched_at,
                )
            )
            continue
        candidates = await CandidatesRepo(db).list_for_stream(clip.stream_id)
        cand = next((c for c in candidates if c.id == clip.candidate_id), None)
        rescore = cand.rescore_score if cand else None
        rows.append(
            CalibrationRow(
                publish_job_id=job.id,
                clip_id=clip.id,
                rescore_score=rescore,
                views=metric.views,
                likes=metric.likes,
                fetched_at=metric.fetched_at,
            )
        )
        if rescore is not None and metric.views is not None:
            paired.append((float(rescore), float(metric.views)))

    pearson = _pearson(paired) if len(paired) >= 3 else None
    return CalibrationReport(
        platform=platform,
        rows=rows,
        pearson_r=pearson,
        n_paired=len(paired),
    )


async def _get_publish_job(db: Database, job_id: str):  # type: ignore[no-untyped-def]
    """Load one publish_job for the bound tenant.

    `PublishJobsRepo` doesn't expose a `get(id)` (intentionally - reads go
    through `list_for_clip` / `list_pending`); we hit the table directly
    here since we already have the id from the metric row.
    """
    conn = await db.connect()
    cur = await conn.execute(
        "SELECT * FROM publish_jobs WHERE id = ? AND tenant_id = ?",
        (job_id, current_tenant_id()),
    )
    row = await cur.fetchone()
    if row is None:
        return None
    import json

    from chalybclip.db.models import PublishJob

    d = dict(row)
    meta = d.pop("platform_metadata_json", None)
    d["platform_metadata"] = json.loads(meta) if meta else None
    return PublishJob.model_validate(d)


def _pearson(pairs: list[tuple[float, float]]) -> float | None:
    """Pearson correlation coefficient. Returns None when undefined."""
    n = len(pairs)
    if n < 2:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = sum(xs) / n
    my = sum(ys) / n
    sx2 = sum((x - mx) ** 2 for x in xs)
    sy2 = sum((y - my) ** 2 for y in ys)
    if sx2 == 0 or sy2 == 0:
        return None  # zero variance on either side -> undefined
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    denom = math.sqrt(sx2 * sy2)
    if denom == 0:
        return None
    return sxy / denom
