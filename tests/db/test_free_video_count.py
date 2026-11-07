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
STALE_BEFORE = (_NOW - dt.timedelta(hours=3)).isoformat()


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
               created_at: str | None = None, updated_at: str | None = None) -> None:
    await UsageJobsRepo(db).upsert(
        job_id=job_id, tenant_id=tenant_id, stream_id=stream_id, operation=operation,
        job_class="job", reservation_id=None, lane="standard", boost=None,
        upload_mb=0, source_minutes=1.0, status=status,
    )
    conn = await db.connect()
    for col, v in (("created_at", created_at), ("updated_at", updated_at)):
        if v:
            await conn.execute(f"UPDATE usage_jobs SET {col} = ? WHERE id = ?", (v, job_id))
    await conn.commit()


async def test_claimed_runs_this_month_count(migrated_db: Database) -> None:
    db = migrated_db
    for t in ("ten_a", "ten_b"):
        await TenantsRepo(db).create(tenant_id=t, name=t)
    await _stream(db, "ten_a", "str_live")
    await _stream(db, "ten_b", "str_b")

    async def count(exclude: str | None = None) -> int:
        return await count_videos_since(db, tenant_id="ten_a", operation="clips.pipeline",
                                        since_iso=MONTH, stale_before_iso=STALE_BEFORE,
                                        exclude_job_id=exclude)

    assert await count() == 0
    # Released: failed/cancelled/refused, last month, another tenant or operation,
    # and an upload abandoned before its stream existed, past the hold's TTL.
    await _job(db, "j_failed", "str_x", "failed")
    await _job(db, "j_cancel", "str_live", "cancelled")
    await _job(db, "j_refused", "str_live", "refused")
    await _job(db, "j_old", "str_live", "succeeded", created_at=LAST_MONTH)
    await _job(db, "j_other_tenant", "str_b", "succeeded", tenant_id="ten_b")
    await _job(db, "j_other_op", "str_live", "succeeded", operation="transcribe")
    stale = (_NOW - dt.timedelta(hours=4)).isoformat()
    await _job(db, "j_abandoned", "str_never", "admitted", updated_at=stale)
    assert await count() == 0

    # Claimed: an upload in flight (no stream row yet) counts at once.
    await _job(db, "j_uploading", "str_pending", "admitted")
    assert await count() == 1
    assert await count(exclude="j_uploading") == 0  # a re-admit doesn't count itself
    # Running and succeeded count; so does any other in-flight status.
    await _job(db, "j_running", "str_live", "running")
    await _job(db, "j_done", "str_y", "succeeded")
    assert await count() == 3
