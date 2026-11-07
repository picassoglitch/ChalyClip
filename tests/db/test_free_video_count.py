"""count_videos_since — what counts as Free's one video of the month."""

from __future__ import annotations

import datetime as dt

from chalybclip.db import Database, StreamsRepo, TenantsRepo
from chalybclip.db.models import StreamRow
from chalybclip.db.usage_repos import UsageJobsRepo, count_videos_since
from chalybclip.tenancy import bound_tenant

_NOW = dt.datetime.now(dt.UTC)
_START = dt.datetime(_NOW.year, _NOW.month, 1, tzinfo=dt.UTC)
MONTH = _START.isoformat()
LAST_MONTH = (_START - dt.timedelta(seconds=1)).isoformat()


async def _stream(db: Database, tenant_id: str, stream_id: str) -> None:
    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=stream_id, tenant_id=tenant_id, vod_url="upload://v.mp4",
                platform="upload", title="v", channel=None, duration_s=0.0,
                source_video_path="/tmp/v.mp4", source_audio_path="/tmp/v.wav",
                status="pending", created_at=MONTH,
            )
        )


async def _job(db: Database, job_id: str, stream_id: str, status: str, *,
               tenant_id: str = "ten_a", operation: str = "clips.pipeline",
               created_at: str | None = None) -> None:
    await UsageJobsRepo(db).upsert(
        job_id=job_id, tenant_id=tenant_id, stream_id=stream_id, operation=operation,
        job_class="job", reservation_id=None, lane="standard", boost=None,
        upload_mb=0, source_minutes=1.0, status=status,
    )
    if created_at:
        conn = await db.connect()
        await conn.execute("UPDATE usage_jobs SET created_at = ? WHERE id = ?",
                           (created_at, job_id))
        await conn.commit()


async def test_only_started_runs_this_month_count(migrated_db: Database) -> None:
    db = migrated_db
    for t in ("ten_a", "ten_b"):
        await TenantsRepo(db).create(tenant_id=t, name=t)
    await _stream(db, "ten_a", "str_live")
    await _stream(db, "ten_b", "str_b")

    async def count(exclude: str | None = None) -> int:
        return await count_videos_since(db, tenant_id="ten_a", operation="clips.pipeline",
                                        since_iso=MONTH, exclude_job_id=exclude)

    assert await count() == 0
    # An upload admitted but never completed: no stream row → not a video.
    await _job(db, "j_abandoned", "str_never", "admitted")
    await _job(db, "j_cancel", "str_live", "cancelled")
    await _job(db, "j_refused", "str_live", "refused")
    await _job(db, "j_old", "str_live", "succeeded", created_at=LAST_MONTH)
    await _job(db, "j_other_tenant", "str_b", "succeeded", tenant_id="ten_b")
    await _job(db, "j_store", "str_live", "succeeded", operation="transcribe")
    assert await count() == 0

    await _job(db, "j_running", "str_live", "admitted")
    assert await count() == 1
    assert await count(exclude="j_running") == 0  # a re-admit doesn't count itself
    await _job(db, "j_failed", "str_x", "failed")
    await _job(db, "j_done", "str_y", "succeeded")
    assert await count() == 3
