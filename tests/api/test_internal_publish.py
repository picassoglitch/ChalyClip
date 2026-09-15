"""Internal Publish API tests (Hub phase 3) — the ChalybOBS contract.

Covers service-token auth, idempotency replay, the mode→payload matrix
(now/queue/schedule/draft, best-time, per-platform captions, first
comment gating), media-URL validation failures, the structured
plan_limit error, batch distribution, and the status endpoint fed by
the phase-2 webhook processor.

Zernio + the media CDN are respx-mocked; the app runs over ASGI.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
import respx

from chalybclip.db import Database, TenantsRepo, ZernioEventsRepo
from chalybclip.integrations.zernio.events import process_zernio_event
from chalybclip.settings import get_settings

_ZBASE = "https://zernio.com/api/v1"
_TOKEN_OBS = "tok_hub_chalybobs_1"
_MEDIA = "https://cdn.test/clip.mp4"


@pytest.fixture
def hub_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("CHALYBCLIP_ZERNIO_API_KEY", "sk_test_hub")
    monkeypatch.setenv(
        "CHALYBCLIP_HUB_SERVICE_TOKENS",
        f"chalybobs:{_TOKEN_OBS},chalyb:tok_hub_chalyb_1",
    )
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


def _auth(token: str = _TOKEN_OBS) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def alice(
    db: Database, tenants: dict[str, dict[str, str]]
) -> dict[str, str]:
    await TenantsRepo(db).set_zernio_profile(
        tenants["alice"]["id"], profile_id="prof_alice", profile_name="Alice",
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


def _mock_media(mock: respx.Router, url: str = _MEDIA) -> None:
    mock.head(url).mock(
        return_value=httpx.Response(200, headers={"content-type": "video/mp4"})
    )


def _mock_create_post(mock: respx.Router, post_id: str = "post_hub_1") -> respx.Route:
    return mock.post(f"{_ZBASE}/posts").mock(
        return_value=httpx.Response(
            201, json={"success": True, "post": {"_id": post_id}}
        )
    )


def _publish_body(tenant_id: str, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "tenant_id": tenant_id,
        "clip": {
            "video_url": _MEDIA,
            "title": "Mi clip",
            "caption_default": "Caption por defecto",
            "duration_s": 45,
        },
        "targets": ["tiktok", "youtube"],
        "mode": "now",
        "source": "chalybobs",
        "idempotency_key": "11111111-1111-1111-1111-111111111111",
    }
    body.update(overrides)
    return body


# ---- auth ----


@pytest.mark.asyncio
async def test_no_tokens_configured_is_503(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CHALYBCLIP_HUB_SERVICE_TOKENS", raising=False)
    get_settings.cache_clear()
    try:
        resp = await client.post(
            "/api/internal/v1/publish", json={}, headers=_auth(),
        )
    finally:
        get_settings.cache_clear()
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_missing_token_is_401(
    hub_env: None, client: httpx.AsyncClient
) -> None:
    resp = await client.post("/api/internal/v1/publish", json={})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_unknown_token_is_401(
    hub_env: None, client: httpx.AsyncClient
) -> None:
    resp = await client.post(
        "/api/internal/v1/publish", json={}, headers=_auth("tok_wrong"),
    )
    assert resp.status_code == 401


# ---- publish: mode → payload matrix ----


@pytest.mark.asyncio
async def test_publish_now_builds_full_payload(
    hub_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok", "youtube", "instagram")
        _mock_media(mock)
        post_route = _mock_create_post(mock)
        resp = await client.post(
            "/api/internal/v1/publish",
            json=_publish_body(
                alice["id"],
                options={
                    "per_platform_captions": {
                        "tiktok": "Caption TikTok",
                        "youtube": {"title": "Título YT", "caption": "Caption YT"},
                    },
                    "first_comment": "Sígueme en Kick 👉",
                },
            ),
            headers=_auth(),
        )
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["ok"] is True
    assert out["job_id"].startswith("hpj_")
    assert out["zernio_post_id"] == "post_hub_1"
    assert out["status"] == "publishing"
    assert out["platforms"] == [
        {"platform": "tiktok", "status": "pending"},
        {"platform": "youtube", "status": "pending"},
    ]

    payload = json.loads(post_route.calls.last.request.content.decode())
    assert payload["publishNow"] is True
    assert "scheduledFor" not in payload
    assert payload["title"] == "Mi clip"
    assert payload["content"] == "Caption por defecto"
    assert payload["mediaItems"] == [{"type": "video", "url": _MEDIA}]
    by_platform = {p["platform"]: p for p in payload["platforms"]}
    assert set(by_platform) == {"tiktok", "youtube"}
    assert by_platform["tiktok"]["accountId"] == "acct_tiktok"
    # Per-platform caption overrides.
    assert by_platform["tiktok"]["customContent"] == "Caption TikTok"
    assert by_platform["youtube"]["customContent"] == "Caption YT"
    # YouTube gets its title via platformSpecificData; firstComment only
    # lands on platforms that support it (youtube sí, tiktok no).
    yt_data = by_platform["youtube"]["platformSpecificData"]
    assert yt_data["title"] == "Título YT"
    assert yt_data["firstComment"] == "Sígueme en Kick 👉"
    assert "platformSpecificData" not in by_platform["tiktok"] or (
        "firstComment" not in by_platform["tiktok"].get("platformSpecificData", {})
    )
    # TikTok legal consents ride at root level.
    assert payload["tiktokSettings"]["content_preview_confirmed"] is True


@pytest.mark.asyncio
async def test_publish_draft_sets_isdraft(
    hub_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok")
        _mock_media(mock)
        post_route = _mock_create_post(mock)
        resp = await client.post(
            "/api/internal/v1/publish",
            json=_publish_body(
                alice["id"], mode="draft", targets=["tiktok"],
                idempotency_key=None,
            ),
            headers=_auth(),
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "draft"
    payload = json.loads(post_route.calls.last.request.content.decode())
    assert payload["isDraft"] is True
    assert "publishNow" not in payload
    assert "scheduledFor" not in payload


@pytest.mark.asyncio
async def test_publish_queue_uses_queued_from_profile(
    hub_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok")
        _mock_media(mock)
        post_route = _mock_create_post(mock)
        resp = await client.post(
            "/api/internal/v1/publish",
            json=_publish_body(
                alice["id"], mode="queue", targets=["tiktok"],
                idempotency_key=None,
            ),
            headers=_auth(),
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"
    payload = json.loads(post_route.calls.last.request.content.decode())
    # Zernio assigns the slot itself — we never precompute next-slot.
    assert payload["queuedFromProfile"] == "prof_alice"
    assert "publishNow" not in payload
    assert "scheduledFor" not in payload


@pytest.mark.asyncio
async def test_publish_schedule_with_explicit_time(
    hub_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok")
        _mock_media(mock)
        post_route = _mock_create_post(mock)
        resp = await client.post(
            "/api/internal/v1/publish",
            json=_publish_body(
                alice["id"], mode="schedule", targets=["tiktok"],
                scheduled_for="2026-12-31T23:00:00+00:00",
                idempotency_key=None,
            ),
            headers=_auth(),
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "scheduled"
    payload = json.loads(post_route.calls.last.request.content.decode())
    assert payload["scheduledFor"] == "2026-12-31T23:00:00+00:00"
    assert payload["timezone"] == "UTC"


@pytest.mark.asyncio
async def test_publish_schedule_best_time_resolves_from_slots(
    hub_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok")
        _mock_media(mock)
        post_route = _mock_create_post(mock)
        mock.get(f"{_ZBASE}/analytics/best-time").mock(
            return_value=httpx.Response(
                200,
                json={
                    "slots": [
                        {"day_of_week": 2, "hour": 18,
                         "avg_engagement": 510.3, "post_count": 15},
                    ]
                },
            )
        )
        resp = await client.post(
            "/api/internal/v1/publish",
            json=_publish_body(
                alice["id"], mode="schedule", targets=["tiktok"],
                options={"use_best_time": True}, idempotency_key=None,
            ),
            headers=_auth(),
        )
    assert resp.status_code == 200
    payload = json.loads(post_route.calls.last.request.content.decode())
    # The resolved time matches the best slot: a Wednesday at 18:00 UTC.
    import datetime as _dt

    resolved = _dt.datetime.fromisoformat(payload["scheduledFor"])
    assert (resolved.weekday(), resolved.hour) == (2, 18)


@pytest.mark.asyncio
async def test_publish_schedule_best_time_falls_back_without_data(
    hub_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok")
        _mock_media(mock)
        post_route = _mock_create_post(mock)
        mock.get(f"{_ZBASE}/analytics/best-time").mock(
            return_value=httpx.Response(403, json={"error": "Analytics add-on"})
        )
        resp = await client.post(
            "/api/internal/v1/publish",
            json=_publish_body(
                alice["id"], mode="schedule", targets=["tiktok"],
                options={"use_best_time": True}, idempotency_key=None,
            ),
            headers=_auth(),
        )
    assert resp.status_code == 200  # sane fallback, not an error
    payload = json.loads(post_route.calls.last.request.content.decode())
    assert payload["scheduledFor"]  # a concrete future time was chosen


@pytest.mark.asyncio
async def test_publish_schedule_without_time_or_best_time_is_400(
    hub_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok")
        _mock_media(mock)
        resp = await client.post(
            "/api/internal/v1/publish",
            json=_publish_body(
                alice["id"], mode="schedule", targets=["tiktok"],
                idempotency_key=None,
            ),
            headers=_auth(),
        )
    assert resp.status_code == 400
    assert resp.json()["error"] == "missing_scheduled_for"


@pytest.mark.asyncio
async def test_publish_all_connected_fans_out(
    hub_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok", "instagram")
        _mock_media(mock)
        post_route = _mock_create_post(mock)
        resp = await client.post(
            "/api/internal/v1/publish",
            json=_publish_body(
                alice["id"], targets="all_connected", idempotency_key=None,
            ),
            headers=_auth(),
        )
    assert resp.status_code == 200
    payload = json.loads(post_route.calls.last.request.content.decode())
    assert {p["platform"] for p in payload["platforms"]} == {"tiktok", "instagram"}


# ---- idempotency ----


@pytest.mark.asyncio
async def test_idempotency_replay_returns_original_without_second_post(
    hub_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok", "youtube")
        _mock_media(mock)
        post_route = _mock_create_post(mock)
        first = await client.post(
            "/api/internal/v1/publish",
            json=_publish_body(alice["id"]),
            headers=_auth(),
        )
        second = await client.post(
            "/api/internal/v1/publish",
            json=_publish_body(alice["id"]),
            headers=_auth(),
        )
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert second.json()["job_id"] == first.json()["job_id"]
    assert len(post_route.calls) == 1  # Zernio was called exactly once


# ---- failure paths ----


@pytest.mark.asyncio
async def test_unknown_tenant_is_404(
    hub_env: None, client: httpx.AsyncClient
) -> None:
    resp = await client.post(
        "/api/internal/v1/publish",
        json=_publish_body("ten_nope", idempotency_key=None),
        headers=_auth(),
    )
    assert resp.status_code == 404
    assert resp.json()["error"] == "unknown_tenant"


@pytest.mark.asyncio
async def test_target_not_connected_is_409(
    hub_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok")  # youtube NOT connected
        resp = await client.post(
            "/api/internal/v1/publish",
            json=_publish_body(alice["id"], idempotency_key=None),
            headers=_auth(),
        )
    assert resp.status_code == 409
    body = resp.json()
    assert body["error"] == "target_not_connected"
    assert "youtube" in body["message"]


@pytest.mark.asyncio
async def test_media_url_unreachable_is_422(
    hub_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok", "youtube")
        mock.head(_MEDIA).mock(return_value=httpx.Response(404))
        resp = await client.post(
            "/api/internal/v1/publish",
            json=_publish_body(alice["id"], idempotency_key=None),
            headers=_auth(),
        )
    assert resp.status_code == 422
    assert resp.json()["error"] == "media_url_unreachable"


@pytest.mark.asyncio
async def test_zernio_402_maps_to_structured_plan_limit(
    hub_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok", "youtube")
        _mock_media(mock)
        mock.post(f"{_ZBASE}/posts").mock(
            return_value=httpx.Response(402, json={"error": "upgrade required"})
        )
        resp = await client.post(
            "/api/internal/v1/publish",
            json=_publish_body(alice["id"], idempotency_key=None),
            headers=_auth(),
        )
    assert resp.status_code == 402
    body = resp.json()
    # The structured contract: never Zernio's raw 402 body.
    assert body["ok"] is False
    assert body["error"] == "plan_limit"
    assert "upgrade required" not in json.dumps(body)


# ---- status + accounts ----


@pytest.mark.asyncio
async def test_status_endpoint_reflects_webhook_updates(
    hub_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok")
        _mock_media(mock)
        _mock_create_post(mock, post_id="post_wh_1")
        created = await client.post(
            "/api/internal/v1/publish",
            json=_publish_body(alice["id"], targets=["tiktok"], idempotency_key=None),
            headers=_auth(),
        )
    job_id = created.json()["job_id"]

    # Simulate the phase-2 webhook landing for that Zernio post.
    await ZernioEventsRepo(db).insert_dedup(
        event_id="evt_hub_pub",
        type="post.published",
        payload=json.dumps(
            {
                "id": "evt_hub_pub",
                "event": "post.published",
                "post": {
                    "id": "post_wh_1",
                    "status": "published",
                    "platforms": [{"platform": "tiktok", "status": "published"}],
                },
            }
        ),
    )
    await process_zernio_event(db, "evt_hub_pub")

    resp = await client.get(
        f"/api/internal/v1/publish/{job_id}",
        params={"tenant_id": alice["id"]},
        headers=_auth(),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "published"
    assert "tiktok" in (body["platforms_json"] or "")

    # Tenant isolation: another tenant id can't read the job.
    other = await client.get(
        f"/api/internal/v1/publish/{job_id}",
        params={"tenant_id": "ten_other"},
        headers=_auth(),
    )
    assert other.status_code == 404


@pytest.mark.asyncio
async def test_accounts_endpoint_lists_connected(
    hub_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok", "youtube")
        resp = await client.get(
            "/api/internal/v1/accounts",
            params={"tenant_id": alice["id"]},
            headers=_auth(),
        )
    assert resp.status_code == 200
    assert resp.json()["connected"] == ["tiktok", "youtube"]


# ---- batch ----


@pytest.mark.asyncio
async def test_batch_distributes_under_daily_cap(
    hub_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    clips = [
        {"video_url": f"https://cdn.test/clip{i}.mp4", "title": f"Clip {i}"}
        for i in range(6)
    ]
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok")
        post_route = _mock_create_post(mock)
        for i in range(6):
            _mock_media(mock, f"https://cdn.test/clip{i}.mp4")
        resp = await client.post(
            "/api/internal/v1/batch",
            json={
                "tenant_id": alice["id"],
                "clips": clips,
                "targets": ["tiktok"],
                "source": "chalybobs",
                "idempotency_key": "batch-1",
            },
            headers=_auth(),
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["scheduled"] == 6
    assert body["failed"] == 0
    # Anti-spam: never more than 4 (default cap) on one UTC date, and
    # never fired all at once — every clip got a concrete future slot.
    dates: dict[str, int] = {}
    for r in body["results"]:
        assert r["ok"] is True
        assert r["status"] == "scheduled"
        day = r["scheduled_for"][:10]
        dates[day] = dates.get(day, 0) + 1
    assert all(n <= 4 for n in dates.values())
    assert len(dates) >= 2
    assert len(post_route.calls) == 6


@pytest.mark.asyncio
async def test_batch_replay_is_idempotent_per_clip(
    hub_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    clips = [{"video_url": _MEDIA, "title": "Solo"}]
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok")
        _mock_media(mock)
        post_route = _mock_create_post(mock)
        body = {
            "tenant_id": alice["id"],
            "clips": clips,
            "targets": ["tiktok"],
            "source": "chalybobs",
            "idempotency_key": "batch-replay",
        }
        first = await client.post(
            "/api/internal/v1/batch", json=body, headers=_auth(),
        )
        second = await client.post(
            "/api/internal/v1/batch", json=body, headers=_auth(),
        )
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["results"][0]["duplicate"] is True
    assert len(post_route.calls) == 1


# ---- internal analytics ----


@pytest.mark.asyncio
async def test_internal_analytics_requires_service_token(
    hub_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    resp = await client.get(
        "/api/internal/v1/analytics", params={"tenant_id": alice["id"]},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_internal_analytics_returns_normalized_performance(
    hub_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        mock.get(f"{_ZBASE}/analytics").mock(
            return_value=httpx.Response(
                200,
                json={
                    "posts": [
                        {"_id": "p1", "content": "Clip A",
                         "analytics": {"likes": 10, "views": 100},
                         "platforms": []},
                    ]
                },
            )
        )
        resp = await client.get(
            "/api/internal/v1/analytics",
            params={"tenant_id": alice["id"], "days": 7},
            headers=_auth(),
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["source"] == "live"
    assert body["totals"]["likes"] == 10
    # no fake zeros: nobody reported comments → null
    assert body["totals"]["comments"] is None
    assert body["posts"][0]["post_id"] == "p1"


@pytest.mark.asyncio
async def test_internal_analytics_unknown_tenant_404(
    hub_env: None, client: httpx.AsyncClient
) -> None:
    resp = await client.get(
        "/api/internal/v1/analytics",
        params={"tenant_id": "ten_nope"},
        headers=_auth(),
    )
    assert resp.status_code == 404
