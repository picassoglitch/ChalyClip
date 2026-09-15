"""Dashboard surfaces the confidence-breakdown panel on /clips/{id}."""

from __future__ import annotations

import datetime as _dt

import httpx

from chalybclip.db import (
    CandidatesRepo,
    ClipsRepo,
    Database,
    StreamsRepo,
)
from chalybclip.db.models import (
    CandidateRow,
    ClipRow,
    StreamRow,
)
from chalybclip.tenancy import bound_tenant


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


async def test_dashboard_clip_page_renders_breakdown_panel(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    tenant_id = tenants["alice"]["id"]
    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_dh",
                tenant_id=tenant_id,
                vod_url="x",
                platform="kick",
                title="t",
                channel="c",
                duration_s=60.0,
                source_video_path="/tmp/x.mp4",
                source_audio_path="/tmp/x.wav",
                status="ingested",
                created_at=_now(),
            )
        )
        await CandidatesRepo(db).upsert_many(
            [
                CandidateRow(
                    id="cnd_dh",
                    stream_id="str_dh",
                    tenant_id=tenant_id,
                    ts=10.0,
                    score=0.5,
                    reason="voice",
                    evidence={},
                    created_at=_now(),
                )
            ]
        )
        await ClipsRepo(db).upsert_many(
            [
                ClipRow(
                    id="clp_dh",
                    stream_id="str_dh",
                    tenant_id=tenant_id,
                    candidate_id="cnd_dh",
                    start_s=10.0,
                    end_s=20.0,
                    duration_s=10.0,
                    width=1080,
                    height=1920,
                    path="/tmp/c.mp4",
                    status="cut",
                    created_at=_now(),
                )
            ]
        )
        await CandidatesRepo(db).update_rescore(
            "cnd_dh",
            rescore_score=0.95,
            rescore_reason="big shock face onset",
            rescore_model="claude-opus-4-7",
        )

    await client.post("/dashboard/login", data={"token": tenants["alice"]["token"]})
    r = await client.get("/dashboard/clips/clp_dh")
    assert r.status_code == 200
    assert "Why this clip?" in r.text
    assert "voice" in r.text
    assert "0.950" in r.text
    assert "+0.450" in r.text  # rescore_delta = 0.95 - 0.50
    assert "big shock face onset" in r.text


async def test_dashboard_clip_page_renders_panel_without_rescore(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """No rescore yet -> panel still renders, with 'not rescored' notice."""
    tenant_id = tenants["alice"]["id"]
    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_n",
                tenant_id=tenant_id,
                vod_url="x",
                platform="kick",
                title=None,
                channel=None,
                duration_s=60.0,
                source_video_path="/tmp/x.mp4",
                source_audio_path="/tmp/x.wav",
                status="ingested",
                created_at=_now(),
            )
        )
        await CandidatesRepo(db).upsert_many(
            [
                CandidateRow(
                    id="cnd_n",
                    stream_id="str_n",
                    tenant_id=tenant_id,
                    ts=10.0,
                    score=0.5,
                    reason="audio",
                    evidence={},
                    created_at=_now(),
                )
            ]
        )
        await ClipsRepo(db).upsert_many(
            [
                ClipRow(
                    id="clp_n",
                    stream_id="str_n",
                    tenant_id=tenant_id,
                    candidate_id="cnd_n",
                    start_s=10.0,
                    end_s=20.0,
                    duration_s=10.0,
                    width=1080,
                    height=1920,
                    path="/tmp/c.mp4",
                    status="cut",
                    created_at=_now(),
                )
            ]
        )

    await client.post("/dashboard/login", data={"token": tenants["alice"]["token"]})
    r = await client.get("/dashboard/clips/clp_n")
    assert r.status_code == 200
    assert "Why this clip?" in r.text
    assert "not rescored" in r.text
