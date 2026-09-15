"""Scheduling tests (Hub phase 5) — Programación routes.

Pins the recurring-slots CRUD, the upcoming (scheduled) list, best-time
panel data, and cancel — including the slot validation that turns a bad
weekday/time into a clean 400 before any Zernio call, and the
dayOfWeek convention (queue slots 0=Sunday, best-time 0=Monday).
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest
import pytest_asyncio
import respx

from chalybclip.db import Database, TenantsRepo, ZernioPublishesRepo
from chalybclip.integrations.chalyb.service import sync_tenant_tier
from chalybclip.settings import get_settings

from .conftest import auth

_ZBASE = "https://zernio.com/api/v1"


@pytest.fixture
def zernio_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("CHALYBCLIP_ZERNIO_API_KEY", "sk_test_sched")
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


@pytest_asyncio.fixture
async def alice(
    db: Database, tenants: dict[str, dict[str, str]]
) -> dict[str, str]:
    tid = tenants["alice"]["id"]
    await sync_tenant_tier(db, tenant_id=tid, tier="all_access")
    await TenantsRepo(db).set_zernio_profile(
        tid, profile_id="prof_alice", profile_name="Alice",
    )
    return tenants["alice"]


# ---- schedule.json (queues + upcoming) ----


@pytest.mark.asyncio
async def test_schedule_json_returns_queues_and_upcoming(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        q_route = mock.get(f"{_ZBASE}/queue/slots").mock(
            return_value=httpx.Response(
                200,
                json={
                    "queues": [
                        {"_id": "q1", "name": "Default", "isDefault": True,
                         "timezone": "UTC",
                         "slots": [{"dayOfWeek": 1, "time": "09:00"}],
                         "active": True}
                    ],
                    "count": 1,
                },
            )
        )
        p_route = mock.get(f"{_ZBASE}/posts").mock(
            return_value=httpx.Response(
                200,
                json={
                    "posts": [
                        {"_id": "post_s1", "content": "Programada",
                         "status": "scheduled",
                         "scheduledFor": "2026-12-31T10:00:00Z",
                         "platforms": [{"platform": "tiktok"}]}
                    ]
                },
            )
        )
        resp = await client.get(
            "/dashboard/publish/zernio/schedule.json",
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["queues"][0]["_id"] == "q1"
    assert body["upcoming"][0]["post_id"] == "post_s1"
    assert body["upcoming"][0]["platforms"] == ["tiktok"]
    # Upcoming asks for scheduled, soonest-first.
    assert p_route.calls.last.request.url.params.get("status") == "scheduled"
    assert p_route.calls.last.request.url.params.get("sortBy") == "scheduled-asc"
    assert q_route.calls.last.request.url.params.get("all") == "true"


@pytest.mark.asyncio
async def test_schedule_json_empty_without_profile(
    zernio_env: None, client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    # bob has no Zernio profile → clean empty state, no Zernio calls.
    # schedule.json is read-only (no paid-tier gate), so a profile-less
    # tenant still gets the clean empty payload.
    resp = await client.get(
        "/dashboard/publish/zernio/schedule.json",
        headers=auth(tenants["bob"]["token"]),
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "queues": [], "upcoming": []}


# ---- save slots (validation + upsert) ----


@pytest.mark.asyncio
async def test_save_slots_upserts_default_queue(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock(assert_all_called=True) as mock:
        route = mock.put(f"{_ZBASE}/queue/slots").mock(
            return_value=httpx.Response(
                200, json={"success": True, "schedule": {"_id": "q9"}}
            )
        )
        resp = await client.post(
            "/dashboard/publish/zernio/schedule/slots",
            json={
                "timezone": "America/Mexico_City",
                "slots": [
                    {"dayOfWeek": 1, "time": "09:00"},
                    {"dayOfWeek": 5, "time": "18:30"},
                ],
            },
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "queue_id": "q9"}
    payload = json.loads(route.calls.last.request.content.decode())
    assert payload["timezone"] == "America/Mexico_City"
    assert payload["slots"] == [
        {"dayOfWeek": 1, "time": "09:00"},
        {"dayOfWeek": 5, "time": "18:30"},
    ]
    assert "queueId" not in payload  # default queue


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_slot",
    [
        {"dayOfWeek": 7, "time": "09:00"},   # weekday out of range
        {"dayOfWeek": -1, "time": "09:00"},
        {"dayOfWeek": 1, "time": "25:00"},   # bad hour
        {"dayOfWeek": 1, "time": "9:00"},    # not HH:mm
        {"dayOfWeek": 1},                    # missing time
        {"time": "09:00"},                   # missing weekday
    ],
)
async def test_save_slots_rejects_bad_slot_before_zernio(
    zernio_env: None,
    client: httpx.AsyncClient,
    alice: dict[str, str],
    bad_slot: dict,
) -> None:
    # No PUT mocked — a bad slot must 400 before any Zernio call.
    resp = await client.post(
        "/dashboard/publish/zernio/schedule/slots",
        json={"slots": [bad_slot]},
        headers=auth(alice["token"]),
    )
    assert resp.status_code == 400
    assert resp.json()["ok"] is False


@pytest.mark.asyncio
async def test_save_slots_empty_list_is_400(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    resp = await client.post(
        "/dashboard/publish/zernio/schedule/slots",
        json={"slots": []},
        headers=auth(alice["token"]),
    )
    assert resp.status_code == 400


# ---- delete queue ----


@pytest.mark.asyncio
async def test_delete_queue_calls_zernio(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock(assert_all_called=True) as mock:
        route = mock.delete(f"{_ZBASE}/queue/slots").mock(
            return_value=httpx.Response(200, json={"success": True})
        )
        resp = await client.post(
            "/dashboard/publish/zernio/schedule/queue/q1/delete",
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    sent = route.calls.last.request
    assert sent.url.params.get("queueId") == "q1"
    assert sent.url.params.get("profileId") == "prof_alice"


# ---- best-time ----


@pytest.mark.asyncio
async def test_best_time_returns_slots(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        mock.get(f"{_ZBASE}/analytics/best-time").mock(
            return_value=httpx.Response(
                200,
                json={"slots": [{"day_of_week": 2, "hour": 18, "avg_engagement": 5}]},
            )
        )
        resp = await client.get(
            "/dashboard/publish/zernio/best-time.json",
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    assert resp.json()["slots"][0]["hour"] == 18


@pytest.mark.asyncio
async def test_best_time_403_addon_returns_empty_not_error(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    """No Analytics add-on / no history → empty list, never a 5xx —
    the panel shows a "sin datos" state."""
    with respx.mock() as mock:
        mock.get(f"{_ZBASE}/analytics/best-time").mock(
            return_value=httpx.Response(403, json={"error": "Analytics add-on"})
        )
        resp = await client.get(
            "/dashboard/publish/zernio/best-time.json",
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "slots": []}


# ---- cancel ----


@pytest.mark.asyncio
async def test_cancel_scheduled_deletes_and_tombstones(
    zernio_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    await ZernioPublishesRepo(db).record(
        post_id="post_cancel_1",
        tenant_id=alice["id"],
        clip_id="clp_x",
        platforms=["tiktok"],
        content="programada",
        status="scheduled",
    )
    with respx.mock(assert_all_called=True) as mock:
        mock.delete(f"{_ZBASE}/posts/post_cancel_1").mock(
            return_value=httpx.Response(200, json={"message": "deleted"})
        )
        resp = await client.post(
            "/dashboard/publish/zernio/schedule/cancel/post_cancel_1",
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    row = await ZernioPublishesRepo(db).get_by_post_id("post_cancel_1")
    assert row is not None and row.status == "cancelled"


@pytest.mark.asyncio
async def test_cancel_published_post_is_409(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        mock.delete(f"{_ZBASE}/posts/post_pub").mock(
            return_value=httpx.Response(400, json={"error": "cannot delete published"})
        )
        resp = await client.post(
            "/dashboard/publish/zernio/schedule/cancel/post_pub",
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 409
    assert resp.json()["ok"] is False


@pytest.mark.asyncio
async def test_cancel_other_tenants_post_is_404_and_never_reaches_zernio(
    zernio_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
    tenants: dict[str, dict[str, str]],
) -> None:
    """Tenant isolation: the Zernio API key is company-wide, so the
    ownership gate must fire BEFORE the vendor delete — a 404 with zero
    outbound calls (no DELETE route is mocked; a call would error)."""
    await ZernioPublishesRepo(db).record(
        post_id="post_of_bob",
        tenant_id=tenants["bob"]["id"],
        clip_id="clp_bob",
        platforms=["tiktok"],
        content="de bob",
        status="scheduled",
    )
    with respx.mock():
        resp = await client.post(
            "/dashboard/publish/zernio/schedule/cancel/post_of_bob",
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 404
    row = await ZernioPublishesRepo(db).get_by_post_id("post_of_bob")
    assert row is not None and row.status == "scheduled"  # untouched


@pytest.mark.asyncio
async def test_cancel_returns_clip_to_approved_pool(
    zernio_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    """Cancelling a clip's last live post puts the clip back in play:
    status back to 'approved' and exists_for_clip no longer blocks —
    previously the clip stayed 'published' behind a cancelled tombstone
    with no surface able to ever schedule it again."""
    import datetime as _dt

    from chalybclip.db import ClipsRepo, StreamsRepo
    from chalybclip.db.models import ClipRow, StreamRow
    from chalybclip.tenancy import bound_tenant

    tid = alice["id"]
    now = _dt.datetime.now(_dt.UTC).isoformat()
    with bound_tenant(tid):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_cx", tenant_id=tid, vod_url="https://kick.com/x",
                platform="kick", title="t", channel="c", duration_s=60.0,
                source_video_path="/tmp/v", source_audio_path="/tmp/a",
                status="ingested", created_at=now,
            )
        )
        await ClipsRepo(db).upsert_many(
            [
                ClipRow(
                    id="clp_cx", stream_id="str_cx", tenant_id=tid,
                    start_s=0.0, end_s=10.0, duration_s=10.0,
                    width=1080, height=1920, path="/tmp/c.mp4",
                    status="published", created_at=now,
                )
            ]
        )
    await ZernioPublishesRepo(db).record(
        post_id="post_cx", tenant_id=tid, clip_id="clp_cx",
        platforms=["tiktok"], content="programada", status="scheduled",
        scheduled_for=now,
    )

    with respx.mock(assert_all_called=True) as mock:
        mock.delete(f"{_ZBASE}/posts/post_cx").mock(
            return_value=httpx.Response(200, json={"message": "deleted"})
        )
        resp = await client.post(
            "/dashboard/publish/zernio/schedule/cancel/post_cx",
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200

    pubs = ZernioPublishesRepo(db)
    assert await pubs.exists_for_clip(tid, "clp_cx") is False  # re-schedulable
    with bound_tenant(tid):
        clip = await ClipsRepo(db).get("clp_cx")
    assert clip is not None and clip.status == "approved"
