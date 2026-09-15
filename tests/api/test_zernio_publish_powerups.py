"""Publish power-ups tests (Hub phase 4) — dashboard publish path.

Pins the form→createPost payload matrix (per-platform captions, first
comment gating, YouTube title, TikTok privacy), the YouTube-without-
title validation, and the full draft lifecycle (guardar → listar →
publicar ahora / programar → eliminar), including tenant isolation.

Zernio is respx-mocked; clip rendering is stubbed (the publish path
renders the edited MP4 before minting the signed URL — that's covered
by the render tests, not here).
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
import respx

import chalybclip.api._clip_render as _clip_render
from chalybclip.db import (
    CandidatesRepo,
    ClipsRepo,
    Database,
    StreamsRepo,
    TenantsRepo,
    ZernioPublishesRepo,
)
from chalybclip.db.models import CandidateRow, ClipRow, StreamRow
from chalybclip.integrations.chalyb.service import sync_tenant_tier
from chalybclip.settings import get_settings
from chalybclip.tenancy import bound_tenant

from .conftest import auth

_ZBASE = "https://zernio.com/api/v1"


@pytest.fixture
def publish_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("CHALYBCLIP_ZERNIO_API_KEY", "sk_test_p4")
    monkeypatch.setenv("CHALYBCLIP_INTERNAL_SIGNING_SECRET", "sign_me_p4")
    get_settings.cache_clear()

    async def _rendered(**_kw: Any) -> None:
        return None

    monkeypatch.setattr(_clip_render, "ensure_clip_rendered", _rendered)
    try:
        yield
    finally:
        get_settings.cache_clear()


@pytest_asyncio.fixture
async def alice(
    db: Database, tenants: dict[str, dict[str, str]]
) -> dict[str, str]:
    """Alice: paid tier, Zernio profile, one approved clip (clp_p4)."""
    tid = tenants["alice"]["id"]
    await sync_tenant_tier(db, tenant_id=tid, tier="all_access")
    await TenantsRepo(db).set_zernio_profile(
        tid, profile_id="prof_alice", profile_name="Alice",
    )
    now = _dt.datetime.now(_dt.UTC).isoformat()
    with bound_tenant(tid):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_p4", tenant_id=tid, vod_url="x", platform="kick",
                title=None, channel=None, duration_s=60.0,
                source_video_path="/tmp/v", source_audio_path="/tmp/a",
                status="ingested", created_at=now,
            )
        )
        await CandidatesRepo(db).upsert_many(
            [
                CandidateRow(
                    id="cnd_p4", stream_id="str_p4", tenant_id=tid,
                    ts=10.0, score=0.5, reason="voice", evidence={},
                    created_at=now,
                )
            ]
        )
        await ClipsRepo(db).upsert_many(
            [
                ClipRow(
                    id="clp_p4", stream_id="str_p4", tenant_id=tid,
                    candidate_id="cnd_p4", start_s=0.0, end_s=10.0,
                    duration_s=10.0, width=1080, height=1920,
                    path="/tmp/c.mp4", status="approved", created_at=now,
                )
            ]
        )
    return tenants["alice"]


def _mock_accounts(mock: respx.Router, *platforms: str) -> None:
    mock.get(f"{_ZBASE}/accounts").mock(
        return_value=httpx.Response(
            200,
            json={
                "accounts": [
                    {"platform": p, "_id": f"acct_{p}", "profileId": "prof_alice"}
                    for p in platforms
                ]
            },
        )
    )


def _mock_create_post(mock: respx.Router, post_id: str) -> respx.Route:
    return mock.post(f"{_ZBASE}/posts").mock(
        return_value=httpx.Response(
            201, json={"success": True, "post": {"_id": post_id}}
        )
    )


async def _save_draft(
    client: httpx.AsyncClient, alice: dict[str, str], *, post_id: str
) -> None:
    """Save clp_p4 as a draft with the full set of power-up extras."""
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok", "youtube")
        _mock_create_post(mock, post_id)
        resp = await client.post(
            "/dashboard/publish/zernio/post/clp_p4",
            data={
                "platforms": "tiktok,youtube",
                "title": "Mi título",
                "description": "Caption general",
                "mode": "draft",
                "first_comment": "Link en mi canal 👉",
                "caption_tiktok": "Caption TikTok",
                "yt_title": "Título YT",
                "tiktok_privacy": "SELF_ONLY",
            },
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 303
    assert resp.headers["location"].endswith("?draft=saved")


# ---- payload matrix ----


@pytest.mark.asyncio
async def test_publish_now_payload_includes_powerups(
    publish_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok", "youtube")
        post_route = _mock_create_post(mock, "post_p4_now")
        resp = await client.post(
            "/dashboard/publish/zernio/post/clp_p4",
            data={
                "platforms": "tiktok,youtube",
                "title": "Mi título",
                "description": "Caption general",
                "first_comment": "Sígueme 👉",
                "caption_tiktok": "Caption TikTok",
                "yt_title": "Título YT",
                "tiktok_privacy": "MUTUAL_FOLLOW_FRIENDS",
            },
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 303, resp.text
    assert "queued=post_p4_now" in resp.headers["location"]

    payload = json.loads(post_route.calls.last.request.content.decode())
    assert payload["publishNow"] is True
    assert payload["title"] == "Mi título"
    by_platform = {p["platform"]: p for p in payload["platforms"]}
    # customContent override only where given.
    assert by_platform["tiktok"]["customContent"] == "Caption TikTok"
    assert "customContent" not in by_platform["youtube"]
    # YouTube: title + firstComment via platformSpecificData.
    yt = by_platform["youtube"]["platformSpecificData"]
    assert yt["title"] == "Título YT"
    assert yt["firstComment"] == "Sígueme 👉"
    # TikTok: privacy override, NO firstComment (unsupported there).
    tt = by_platform["tiktok"]["platformSpecificData"]
    assert tt["privacyLevel"] == "MUTUAL_FOLLOW_FRIENDS"
    assert "firstComment" not in tt
    # Root TikTok consents still ride along.
    assert payload["tiktokSettings"]["express_consent_given"] is True


@pytest.mark.asyncio
async def test_youtube_without_title_is_400_before_zernio(
    publish_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    # No respx mock at all: the validation must reject BEFORE any
    # Zernio call (an unmocked request would error loudly here).
    resp = await client.post(
        "/dashboard/publish/zernio/post/clp_p4",
        data={"platforms": "youtube", "description": "solo caption"},
        headers=auth(alice["token"]),
    )
    assert resp.status_code == 400
    assert "título" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_schedule_mode_sets_scheduledfor(
    publish_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok")
        post_route = _mock_create_post(mock, "post_p4_sched")
        resp = await client.post(
            "/dashboard/publish/zernio/post/clp_p4",
            data={
                "platforms": "tiktok",
                "description": "programada",
                "mode": "schedule",
                "scheduled_for": "2026-12-31T23:00:00.000Z",
            },
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 303
    payload = json.loads(post_route.calls.last.request.content.decode())
    assert payload["scheduledFor"] == "2026-12-31T23:00:00.000Z"
    assert payload["timezone"] == "UTC"
    assert "publishNow" not in payload


@pytest.mark.asyncio
async def test_schedule_without_datetime_is_400(
    publish_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    resp = await client.post(
        "/dashboard/publish/zernio/post/clp_p4",
        data={"platforms": "tiktok", "mode": "schedule"},
        headers=auth(alice["token"]),
    )
    assert resp.status_code == 400


# ---- draft lifecycle ----


@pytest.mark.asyncio
async def test_draft_saves_locally_with_snapshot_and_keeps_clip_approved(
    publish_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    await _save_draft(client, alice, post_id="post_p4_draft")
    row = await ZernioPublishesRepo(db).get_by_post_id("post_p4_draft")
    assert row is not None
    assert row.status == "draft"
    snapshot = json.loads(row.options_json or "{}")
    assert snapshot["title"] == "Mi título"
    assert snapshot["per_platform_captions"]["tiktok"] == "Caption TikTok"
    assert snapshot["per_platform_captions"]["youtube"]["title"] == "Título YT"
    assert snapshot["first_comment"] == "Link en mi canal 👉"
    assert snapshot["tiktok_privacy"] == "SELF_ONLY"
    # The clip is NOT out the door — it stays approved, not published.
    with bound_tenant(alice["id"]):
        clip = await ClipsRepo(db).get("clp_p4")
    assert clip is not None
    assert clip.status == "approved"


@pytest.mark.asyncio
async def test_draft_publish_now_recreates_with_snapshot_and_deletes_draft(
    publish_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    await _save_draft(client, alice, post_id="post_p4_d1")
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok", "youtube")
        post_route = _mock_create_post(mock, "post_p4_live")
        delete_route = mock.delete(f"{_ZBASE}/posts/post_p4_d1").mock(
            return_value=httpx.Response(200, json={"message": "deleted"})
        )
        resp = await client.post(
            "/dashboard/publish/zernio/draft/post_p4_d1/publish",
            data={},
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 303
    assert "queued=post_p4_live" in resp.headers["location"]
    assert delete_route.called

    # Re-created payload carries the snapshot extras.
    payload = json.loads(post_route.calls.last.request.content.decode())
    assert payload["publishNow"] is True
    by_platform = {p["platform"]: p for p in payload["platforms"]}
    assert by_platform["tiktok"]["customContent"] == "Caption TikTok"
    assert by_platform["youtube"]["platformSpecificData"]["title"] == "Título YT"
    assert by_platform["tiktok"]["platformSpecificData"]["privacyLevel"] == "SELF_ONLY"

    old = await ZernioPublishesRepo(db).get_by_post_id("post_p4_d1")
    assert old is not None and old.status == "deleted"
    new = await ZernioPublishesRepo(db).get_by_post_id("post_p4_live")
    assert new is not None and new.status is None  # live → webhook-fed


@pytest.mark.asyncio
async def test_draft_schedule_sets_scheduledfor(
    publish_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    await _save_draft(client, alice, post_id="post_p4_d2")
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok", "youtube")
        post_route = _mock_create_post(mock, "post_p4_sched2")
        mock.delete(f"{_ZBASE}/posts/post_p4_d2").mock(
            return_value=httpx.Response(200, json={"message": "ok"})
        )
        resp = await client.post(
            "/dashboard/publish/zernio/draft/post_p4_d2/publish",
            data={"scheduled_for": "2027-01-15T10:00:00.000Z"},
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 303
    payload = json.loads(post_route.calls.last.request.content.decode())
    assert payload["scheduledFor"] == "2027-01-15T10:00:00.000Z"
    new = await ZernioPublishesRepo(db).get_by_post_id("post_p4_sched2")
    assert new is not None and new.status == "scheduled"


@pytest.mark.asyncio
async def test_draft_delete_removes_remote_and_tombstones_local(
    publish_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    await _save_draft(client, alice, post_id="post_p4_d3")
    with respx.mock(assert_all_called=True) as mock:
        delete_route = mock.delete(f"{_ZBASE}/posts/post_p4_d3").mock(
            return_value=httpx.Response(200, json={"message": "deleted"})
        )
        resp = await client.post(
            "/dashboard/publish/zernio/draft/post_p4_d3/delete",
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 303
    assert delete_route.called
    row = await ZernioPublishesRepo(db).get_by_post_id("post_p4_d3")
    assert row is not None and row.status == "deleted"


@pytest.mark.asyncio
async def test_draft_actions_enforce_tenant_ownership(
    publish_env: None,
    client: httpx.AsyncClient,
    alice: dict[str, str],
    tenants: dict[str, dict[str, str]],
) -> None:
    await _save_draft(client, alice, post_id="post_p4_d4")
    resp = await client.post(
        "/dashboard/publish/zernio/draft/post_p4_d4/delete",
        headers=auth(tenants["bob"]["token"]),
    )
    # bob hits the paywall (free tier, 402) or the ownership check
    # (404) — either way, never a cross-tenant delete.
    assert resp.status_code in (402, 404)


@pytest.mark.asyncio
async def test_non_draft_post_rejects_draft_actions(
    publish_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    await ZernioPublishesRepo(db).record(
        post_id="post_p4_live2",
        tenant_id=alice["id"],
        clip_id="clp_p4",
        platforms=["tiktok"],
        content="ya publicado",
    )
    resp = await client.post(
        "/dashboard/publish/zernio/draft/post_p4_live2/delete",
        headers=auth(alice["token"]),
    )
    assert resp.status_code == 409
