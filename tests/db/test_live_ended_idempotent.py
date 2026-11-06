"""A repeat live 'ended' (relay retry) must not reset a claimed stream.

Runs on Postgres when CHALYBCLIP_TEST_PG_DSN is set (CI), else SQLite.
"""

from __future__ import annotations

import datetime as _dt

from chalybclip.db import Database, StreamsRepo, TenantsRepo
from chalybclip.db.models import StreamRow
from chalybclip.db.repos import (
    _streams_repo_mark_live_ended,
    _streams_repo_mark_live_started,
    _streams_repo_set_status,
    _streams_repo_try_claim_for_processing,
)
from chalybclip.tenancy import bound_tenant


async def test_repeat_ended_keeps_claim_and_first_end(migrated_db: Database) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="Aldo")
    with bound_tenant(tenant.id):
        await StreamsRepo(migrated_db).upsert(
            StreamRow(
                id="str_pg_r",
                tenant_id=tenant.id,
                vod_url="live://rtmp/str_pg_r",
                platform="live",
                title="En vivo",
                channel=None,
                duration_s=0.0,
                source_video_path="/data/live/str_pg_r/source.mp4",
                source_audio_path="/data/live/str_pg_r/source.audio.wav",
                status="pending",
                created_at=_dt.datetime.now(_dt.UTC).isoformat(),
            )
        )
    await _streams_repo_mark_live_started(migrated_db, stream_id="str_pg_r")

    first = await _streams_repo_mark_live_ended(migrated_db, stream_id="str_pg_r", duration_s=60.0)
    assert first is not None and first.status == "live_ended"
    assert await _streams_repo_try_claim_for_processing(migrated_db, stream_id="str_pg_r")

    again = await _streams_repo_mark_live_ended(migrated_db, stream_id="str_pg_r", duration_s=99.0)
    assert again is not None
    assert again.status == "processing"
    assert again.live_ended_at == first.live_ended_at
    assert again.duration_s == 60.0
    assert not await _streams_repo_try_claim_for_processing(migrated_db, stream_id="str_pg_r")

    await _streams_repo_set_status(migrated_db, stream_id="str_pg_r", status="done")
    after_done = await _streams_repo_mark_live_ended(migrated_db, stream_id="str_pg_r")
    assert after_done is not None and after_done.status == "done"
    assert not await _streams_repo_try_claim_for_processing(migrated_db, stream_id="str_pg_r")
