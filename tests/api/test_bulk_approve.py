"""Bulk 'Aprobar todos' — flip a stream's pending clips to approved in one
POST so the operator doesn't click through them individually."""

from __future__ import annotations

import datetime as _dt

import httpx
import pytest

from chalybclip.db import ClipsRepo, Database, StreamsRepo
from chalybclip.db.models import ClipRow, StreamRow
from chalybclip.tenancy import bound_tenant

from .conftest import auth


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


async def _seed(
    db: Database, tid: str, *, stream_id: str, clips: dict[str, str]
) -> None:
    now = _now()
    with bound_tenant(tid):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=stream_id, tenant_id=tid, vod_url="x", platform="kick",
                title=None, channel=None, duration_s=60.0,
                source_video_path="/v", source_audio_path="/a",
                status="ingested", created_at=now,
            )
        )
        await ClipsRepo(db).upsert_many([
            ClipRow(
                id=cid, stream_id=stream_id, tenant_id=tid, candidate_id=None,
                start_s=0.0, end_s=10.0, duration_s=10.0, width=1080,
                height=1920, path="/c.mp4", status=status, created_at=now,
            )
            for cid, status in clips.items()
        ])


@pytest.mark.asyncio
async def test_approve_all_flips_pending_clips_only(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    tid = tenants["alice"]["id"]
    await _seed(db, tid, stream_id="str_b", clips={
        "clp_cut": "cut",
        "clp_rev": "ready_for_review",
        "clp_appr": "approved",   # already approved — untouched
        "clp_rej": "rejected",    # rejected — must NOT be approved
    })
    r = await client.post(
        "/dashboard/streams/str_b/clips/approve-all",
        headers=auth(tenants["alice"]["token"]), follow_redirects=False,
    )
    assert r.status_code == 303
    assert "approved=2" in r.headers["location"]  # cut + ready_for_review

    with bound_tenant(tid):
        status = {c.id: c.status for c in await ClipsRepo(db).list_for_stream("str_b")}
    assert status["clp_cut"] == "approved"
    assert status["clp_rev"] == "approved"
    assert status["clp_appr"] == "approved"
    assert status["clp_rej"] == "rejected"  # untouched


@pytest.mark.asyncio
async def test_approve_all_cross_tenant_is_isolated(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    """Bob can't approve Alice's clips."""
    await _seed(db, tenants["alice"]["id"], stream_id="str_a", clips={"clp_a": "cut"})
    await client.post(
        "/dashboard/streams/str_a/clips/approve-all",
        headers=auth(tenants["bob"]["token"]), follow_redirects=False,
    )
    with bound_tenant(tenants["alice"]["id"]):
        status = {c.id: c.status for c in await ClipsRepo(db).list_for_stream("str_a")}
    assert status["clp_a"] == "cut"  # untouched by Bob
