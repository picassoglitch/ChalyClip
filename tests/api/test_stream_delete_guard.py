"""POST /dashboard/streams/{id}/delete — pending-publish guard.

Regression for the "viral @doomscroll" failure: a stream was deleted from
the dashboard while a Zernio post scheduled two days earlier still pointed
at one of its clips. The FK cascade removed the clip row, and when the
post's slot arrived Zernio fetched our signed media URL and got a 404 —
surfacing to the user as a "post failed to publish" email with no visible
cause. Deletion must refuse while any pre-flight publish references the
stream's clips.
"""

from __future__ import annotations

import datetime as _dt

import httpx

from chalybclip.db import (
    ClipsRepo,
    Database,
    StreamsRepo,
    ZernioPublishesRepo,
)
from chalybclip.db.models import ClipRow, StreamRow
from chalybclip.tenancy import bound_tenant

from .conftest import auth


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


async def _seed_stream_with_clip(
    db: Database, *, tenant_id: str, stream_id: str, clip_id: str
) -> None:
    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=stream_id, tenant_id=tenant_id,
                vod_url="https://kick.com/x/videos/1", platform="kick",
                title="t", channel="c", duration_s=300.0,
                source_video_path=f"/tmp/{stream_id}.mp4",
                source_audio_path=f"/tmp/{stream_id}.wav",
                status="done", created_at=_now(),
            )
        )
        await ClipsRepo(db).upsert_many(
            [
                ClipRow(
                    id=clip_id, stream_id=stream_id, tenant_id=tenant_id,
                    candidate_id=None, start_s=10.0, end_s=55.0,
                    duration_s=45.0, width=1080, height=1920,
                    path=f"/tmp/{clip_id}.mp4", status="approved",
                    created_at=_now(),
                )
            ]
        )


async def test_delete_refused_while_post_scheduled(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    tid = tenants["alice"]["id"]
    await _seed_stream_with_clip(
        db, tenant_id=tid, stream_id="str_guard", clip_id="clp_guard"
    )
    await ZernioPublishesRepo(db).record(
        post_id="p_sched", tenant_id=tid, clip_id="clp_guard",
        platforms=["youtube"], content="viral @doomscroll",
        status="scheduled", scheduled_for=_now(),
    )

    r = await client.post(
        "/dashboard/streams/str_guard/delete",
        headers=auth(tenants["alice"]["token"]),
        follow_redirects=False,
    )

    assert r.status_code == 409
    assert "p_sched" in r.text
    with bound_tenant(tid):
        assert await StreamsRepo(db).get("str_guard") is not None


async def test_delete_proceeds_once_post_is_terminal(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    """published / failed / cancelled posts don't block — only pre-flight
    states hold the stream hostage."""
    tid = tenants["alice"]["id"]
    await _seed_stream_with_clip(
        db, tenant_id=tid, stream_id="str_free", clip_id="clp_free"
    )
    await ZernioPublishesRepo(db).record(
        post_id="p_done", tenant_id=tid, clip_id="clp_free",
        platforms=["youtube"], content="x", status="scheduled",
    )
    await ZernioPublishesRepo(db).set_status("p_done", status="published")

    r = await client.post(
        "/dashboard/streams/str_free/delete",
        headers=auth(tenants["alice"]["token"]),
        follow_redirects=False,
    )

    assert r.status_code == 303
    with bound_tenant(tid):
        assert await StreamsRepo(db).get("str_free") is None
