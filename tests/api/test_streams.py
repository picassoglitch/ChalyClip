"""Stream + candidate + clip listing - tenant isolation is the load-bearing assertion."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import httpx
import pytest

from chalybclip.api import create_app
from chalybclip.db import (
    CandidatesRepo,
    ClipsRepo,
    Database,
    StreamsRepo,
)
from chalybclip.db.models import CandidateRow, ClipRow, StreamRow
from chalybclip.tenancy import bound_tenant

from .conftest import auth


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


async def _seed_stream(
    db: Database, *, tenant_id: str, stream_id: str, vod_url: str = "https://example.com/x"
) -> StreamRow:
    with bound_tenant(tenant_id):
        return await StreamsRepo(db).upsert(
            StreamRow(
                id=stream_id,
                tenant_id=tenant_id,
                vod_url=vod_url,
                platform="kick",
                title="Test stream",
                channel="aldo",
                duration_s=300.0,
                source_video_path=f"/tmp/{stream_id}.mp4",
                source_audio_path=f"/tmp/{stream_id}.wav",
                status="ingested",
                created_at=_now(),
            )
        )


async def _seed_candidate(
    db: Database, *, tenant_id: str, stream_id: str, candidate_id: str, ts: float
) -> None:
    with bound_tenant(tenant_id):
        await CandidatesRepo(db).upsert_many(
            [
                CandidateRow(
                    id=candidate_id,
                    stream_id=stream_id,
                    tenant_id=tenant_id,
                    ts=ts,
                    score=0.9,
                    reason="voice",
                    evidence={"phrase": "clipéalo"},
                    created_at=_now(),
                )
            ]
        )


async def _seed_clip(
    db: Database, *, tenant_id: str, stream_id: str, clip_id: str, candidate_id: str
) -> ClipRow:
    with bound_tenant(tenant_id):
        await ClipsRepo(db).upsert_many(
            [
                ClipRow(
                    id=clip_id,
                    stream_id=stream_id,
                    tenant_id=tenant_id,
                    candidate_id=candidate_id,
                    start_s=10.0,
                    end_s=55.0,
                    duration_s=45.0,
                    width=1080,
                    height=1920,
                    path=f"/tmp/{clip_id}.mp4",
                    smart_crop_box=None,
                    thumbnail_frame_path=None,
                    status="cut",
                    created_at=_now(),
                )
            ]
        )
        clip = await ClipsRepo(db).get(clip_id)
    assert clip is not None
    return clip


async def test_list_streams_returns_only_my_tenant(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    await _seed_stream(db, tenant_id=tenants["alice"]["id"], stream_id="str_a1")
    await _seed_stream(db, tenant_id=tenants["alice"]["id"], stream_id="str_a2")
    await _seed_stream(db, tenant_id=tenants["bob"]["id"], stream_id="str_b1")

    r = await client.get("/streams", headers=auth(tenants["alice"]["token"]))
    assert r.status_code == 200
    ids = {s["id"] for s in r.json()}
    assert ids == {"str_a1", "str_a2"}

    r = await client.get("/streams", headers=auth(tenants["bob"]["token"]))
    ids = {s["id"] for s in r.json()}
    assert ids == {"str_b1"}


async def test_get_other_tenants_stream_returns_404(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """Bob cannot read Alice's stream even though the id exists."""
    await _seed_stream(db, tenant_id=tenants["alice"]["id"], stream_id="str_alice")
    r = await client.get("/streams/str_alice", headers=auth(tenants["bob"]["token"]))
    assert r.status_code == 404


async def test_candidates_endpoint_filters_by_tenant(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    await _seed_stream(db, tenant_id=tenants["alice"]["id"], stream_id="str_alice")
    await _seed_candidate(
        db,
        tenant_id=tenants["alice"]["id"],
        stream_id="str_alice",
        candidate_id="cnd_a1",
        ts=12.0,
    )
    # Same stream id from Bob's perspective is invisible.
    r = await client.get(
        "/streams/str_alice/candidates", headers=auth(tenants["bob"]["token"])
    )
    assert r.status_code == 404
    # Alice sees her one candidate.
    r = await client.get(
        "/streams/str_alice/candidates", headers=auth(tenants["alice"]["token"])
    )
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["id"] == "cnd_a1"


async def test_clips_endpoint_lists_per_stream(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    await _seed_stream(db, tenant_id=tenants["alice"]["id"], stream_id="str_alice")
    await _seed_candidate(
        db,
        tenant_id=tenants["alice"]["id"],
        stream_id="str_alice",
        candidate_id="cnd_a1",
        ts=12.0,
    )
    await _seed_clip(
        db,
        tenant_id=tenants["alice"]["id"],
        stream_id="str_alice",
        clip_id="clp_a1",
        candidate_id="cnd_a1",
    )
    r = await client.get(
        "/streams/str_alice/clips", headers=auth(tenants["alice"]["token"])
    )
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["id"] == "clp_a1"
    assert rows[0]["status"] == "cut"


async def test_post_streams_kicks_off_pipeline(
    db: Database,
    tenants: dict[str, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`POST /streams` calls ingest, persists, schedules pipeline, returns 202."""
    from chalybclip.api import PipelineKickoff
    from chalybclip.ingest.models import Stream

    fake_video = tmp_path / "vid.mp4"
    fake_video.write_bytes(b"\x00fakevideo")
    fake_audio = tmp_path / "aud.wav"
    fake_audio.write_bytes(b"\x00fakeaudio")

    captured: list[PipelineKickoff] = []

    async def fake_ingest(*, vod_url: str, tenant_id: str, output_dir: Path) -> Stream:
        return Stream(
            id="str_fresh",
            tenant_id=tenant_id,
            vod_url=vod_url,
            platform="kick",
            title="Fresh",
            channel="x",
            duration_s=42.0,
            source_video_path=fake_video,
            source_audio_path=fake_audio,
        )

    async def fake_runner(kickoff: PipelineKickoff) -> None:
        captured.append(kickoff)

    monkeypatch.setattr("chalybclip.ingest.ingest_vod", fake_ingest)

    app = create_app(db=db, pipeline_runner=fake_runner)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as c:
        r = await c.post(
            "/streams",
            json={"vod_url": "https://example.com/x", "persona_id": "aldo"},
            headers=auth(tenants["alice"]["token"]),
        )
    assert r.status_code == 202
    body = r.json()
    assert body["id"] == "str_fresh"
    assert body["duration_s"] == 42.0
    # BackgroundTask runs after the response sends.
    assert len(captured) == 1
    assert captured[0].tenant_id == tenants["alice"]["id"]
    assert captured[0].stream.id == "str_fresh"
    assert captured[0].persona_id == "aldo"

    # Stream row landed in the DB under Alice's tenant.
    with bound_tenant(tenants["alice"]["id"]):
        row = await StreamsRepo(db).get("str_fresh")
    assert row is not None
    assert row.tenant_id == tenants["alice"]["id"]
