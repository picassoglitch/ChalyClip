"""/api/hub/v1/publish — the hub's Publish Center API (Zernio).

Pins:
  * only the hub's Bearer launch token reaches these routes (no session
    token / cookie), and the tenant comes from the token;
  * reused dashboard handlers answer in the hub's error shape: engine copy
    for 4xx, generic copy for 5xx (never a raw Zernio body);
  * form routes' 303 redirects come back as JSON results;
  * the OAuth popup returns to the HUB (CHALYB_BASE_URL origin only);
  * ids from the URL are ownership-gated BEFORE the company-wide Zernio key
    is used (posts, automations, sequences).

Zernio is respx-mocked; the app runs over ASGI.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Iterator
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import pytest_asyncio
import respx

from chalybclip.db import (
    CandidatesRepo,
    ClipsRepo,
    Database,
    StreamsRepo,
    TenantsRepo,
    ZernioPublishesRepo,
)
from chalybclip.db.models import CandidateRow, ClipRow, StreamRow
from chalybclip.integrations.chalyb.sso import sign_sso_token
from chalybclip.settings import get_settings
from chalybclip.tenancy import bound_tenant

from .conftest import auth

_SECRET = "hub_publish_test_secret"
_ZBASE = "https://zernio.com/api/v1"
_HUB = "https://www.chalyb.com"


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("CHALYB_SSO_SECRET", _SECRET)
    monkeypatch.setenv("CHALYBCLIP_ZERNIO_API_KEY", "sk_test_hub_publish")
    monkeypatch.setenv("CHALYB_BASE_URL", _HUB)
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


def _hub(tenant_id: str, tier: str = "all_access") -> dict[str, str]:
    token = sign_sso_token(
        user_id=f"user_{tenant_id}",
        email=f"{tenant_id}@example.com",
        tenant_id=tenant_id,
        secret=_SECRET,
        tier=tier,
    )
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def alice(db: Database, tenants: dict[str, dict[str, str]]) -> str:
    tid = tenants["alice"]["id"]
    await TenantsRepo(db).set_zernio_profile(tid, profile_id="prof_alice", profile_name="Alice")
    return tid


@pytest_asyncio.fixture
async def bob(db: Database, tenants: dict[str, dict[str, str]]) -> str:
    tid = tenants["bob"]["id"]
    await TenantsRepo(db).set_zernio_profile(tid, profile_id="prof_bob", profile_name="Bob")
    return tid


def _accounts(*rows: tuple[str, str, str]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "accounts": [
                {"_id": aid, "platform": plat, "profileId": pid} for aid, plat, pid in rows
            ]
        },
    )


async def _approved_clip(db: Database, tenant_id: str, title: str = "Gran jugada") -> str:
    clip_id = f"clp_{tenant_id[-6:]}"
    now = _dt.datetime.now(_dt.UTC).isoformat()
    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=f"str_{tenant_id[-6:]}",
                tenant_id=tenant_id,
                vod_url="x",
                platform="kick",
                title=None,
                channel=None,
                duration_s=60.0,
                source_video_path="/tmp/v",
                source_audio_path="/tmp/a",
                status="ingested",
                created_at=now,
            )
        )
        await CandidatesRepo(db).upsert_many(
            [
                CandidateRow(
                    id=f"cnd_{clip_id}",
                    stream_id=f"str_{tenant_id[-6:]}",
                    tenant_id=tenant_id,
                    ts=10.0,
                    score=0.7,
                    reason="voice",
                    evidence={},
                    created_at=now,
                )
            ]
        )
        await ClipsRepo(db).upsert_many(
            [
                ClipRow(
                    id=clip_id,
                    stream_id=f"str_{tenant_id[-6:]}",
                    tenant_id=tenant_id,
                    candidate_id=f"cnd_{clip_id}",
                    start_s=0.0,
                    end_s=20.0,
                    duration_s=20.0,
                    width=1080,
                    height=1920,
                    path="/tmp/c.mp4",
                    status="approved",
                    created_at=now,
                )
            ]
        )
        await ClipsRepo(db).set_overlay_config(clip_id, overlay_config={"title_text": title})
    return clip_id


def _err(r: httpx.Response, status: int, code: str) -> dict:
    assert r.status_code == status, r.text
    body = r.json()
    assert body["ok"] is False
    assert body["error"]["code"] == code
    return body["error"]


# ---- auth ----


async def test_requires_hub_token(
    client: httpx.AsyncClient, env: None, alice: str, tenants: dict[str, dict[str, str]]
) -> None:
    r = await client.get("/api/hub/v1/publish/accounts.json")
    _err(r, 401, "unauthorized")
    # A dashboard api_token is NOT a hub credential.
    r = await client.get(
        "/api/hub/v1/publish/accounts.json",
        headers=auth(tenants["alice"]["token"]),
    )
    _err(r, 401, "unauthorized")
    # Forged signature.
    r = await client.get(
        "/api/hub/v1/publish/accounts.json",
        headers={"Authorization": "Bearer " + "x" * 40 + "." + "y" * 43},
    )
    _err(r, 401, "unauthorized")


async def test_tenant_comes_from_token(
    client: httpx.AsyncClient, env: None, alice: str, bob: str
) -> None:
    with respx.mock(assert_all_called=True) as mock:
        route = mock.get(f"{_ZBASE}/accounts").mock(
            return_value=_accounts(("acc_b", "tiktok", "prof_bob")),
        )
        r = await client.get("/api/hub/v1/publish/accounts.json", headers=_hub(bob))
    assert r.status_code == 200, r.text
    assert r.json()["connected"] == ["tiktok"]
    assert route.calls.last.request.url.params["profileId"] == "prof_bob"


# ---- overview ----


async def test_overview_shapes_dashboard_context(
    client: httpx.AsyncClient,
    env: None,
    db: Database,
    alice: str,
) -> None:
    clip_id = await _approved_clip(db, alice)
    await ZernioPublishesRepo(db).record(
        post_id="p_draft",
        tenant_id=alice,
        clip_id=clip_id,
        platforms=["tiktok"],
        content="borrador",
        status="draft",
    )
    with respx.mock() as mock:
        mock.get(f"{_ZBASE}/accounts").mock(
            return_value=_accounts(("acc_tt", "tiktok", "prof_alice")),
        )
        mock.get(f"{_ZBASE}/posts").mock(return_value=httpx.Response(200, json={"posts": []}))
        r = await client.get("/api/hub/v1/publish/overview", headers=_hub(alice))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True and body["configured"] is True
    assert body["profile"] == {"id": "prof_alice", "name": "Alice"}
    assert body["accounts"] == [{"platform": "tiktok", "account_id": "acc_tt"}]
    assert body["account_limit"] is None  # all_access
    assert [c["id"] for c in body["publishable"]] == [clip_id]
    assert body["publishable"][0]["title"] == "Gran jugada"
    assert body["publishable_total"] == 1
    assert [d["post_id"] for d in body["drafts"]] == ["p_draft"]
    assert body["drafts"][0]["platforms"] == ["tiktok"]
    assert body["is_pro"] is True
    assert {p["id"] for p in body["platforms"]} >= {"tiktok", "youtube", "twitter"}
    assert set(body["stats"]) == {"published", "failed", "scheduled", "total"}
    # Never the template's own engine URLs.
    assert "/dashboard/" not in r.text


async def test_overview_free_tier_is_not_pro(
    client: httpx.AsyncClient,
    env: None,
    alice: str,
) -> None:
    with respx.mock() as mock:
        mock.get(f"{_ZBASE}/accounts").mock(return_value=_accounts())
        r = await client.get("/api/hub/v1/publish/overview", headers=_hub(alice, tier="free"))
    assert r.status_code == 200, r.text
    assert r.json()["is_pro"] is False


# ---- connect: the popup comes back to the hub ----


async def test_connect_redirects_to_hub_landing(
    client: httpx.AsyncClient,
    env: None,
    alice: str,
) -> None:
    with respx.mock(assert_all_called=True) as mock:
        route = mock.get(f"{_ZBASE}/connect/facebook").mock(
            return_value=httpx.Response(200, json={"authUrl": "https://zernio.com/oauth/x"}),
        )
        r = await client.post(
            "/api/hub/v1/publish/connect",
            headers=_hub(alice),
            json={
                "platform": "facebook",
                "return_url": f"{_HUB}/app/clips/publicar/conectado?platform=evil",
            },
        )
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "auth_url": "https://zernio.com/oauth/x"}
    params = route.calls.last.request.url.params
    assert params["profileId"] == "prof_alice"
    assert params["headless"] == "true"
    back = urlsplit(params["redirect_url"])
    assert f"{back.scheme}://{back.netloc}" == _HUB
    assert back.path == "/app/clips/publicar/conectado"
    assert parse_qs(back.query) == {"platform": ["facebook"]}


@pytest.mark.parametrize(
    "return_url",
    [
        "https://evil.example/app/clips/publicar/conectado",
        "https://www.chalyb.com.evil.example/x",
        "https://user@www.chalyb.com/x",
        "http://www.chalyb.com/x",  # plain http only for localhost
        "javascript:alert(1)",
        "",
    ],
)
async def test_connect_refuses_foreign_return_url(
    client: httpx.AsyncClient,
    env: None,
    alice: str,
    return_url: str,
) -> None:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(f"{_ZBASE}/connect/tiktok").mock(
            return_value=httpx.Response(200, json={"authUrl": "https://zernio.com/x"}),
        )
        r = await client.post(
            "/api/hub/v1/publish/connect",
            headers=_hub(alice),
            json={"platform": "tiktok", "return_url": return_url},
        )
    err = _err(r, 400, "bad_request")
    assert err["reason"] == "bad_return_url"
    assert not route.called


async def test_connect_bare_domain_matches_www(
    client: httpx.AsyncClient,
    env: None,
    alice: str,
) -> None:
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{_ZBASE}/connect/tiktok").mock(
            return_value=httpx.Response(200, json={"authUrl": "https://zernio.com/x"}),
        )
        r = await client.post(
            "/api/hub/v1/publish/connect",
            headers=_hub(alice),
            json={
                "platform": "tiktok",
                "return_url": "https://chalyb.com/app/clips/publicar/conectado",
            },
        )
    assert r.status_code == 200, r.text


async def test_connect_free_tier_is_needs_plan(
    client: httpx.AsyncClient,
    env: None,
    alice: str,
) -> None:
    r = await client.post(
        "/api/hub/v1/publish/connect",
        headers=_hub(alice, tier="free"),
        json={"platform": "tiktok", "return_url": f"{_HUB}/app/clips/publicar/conectado"},
    )
    assert r.status_code == 402, r.text
    assert r.json()["error"]["code"] == "needs_plan"


async def test_connect_pro_account_cap_is_needs_plan(
    client: httpx.AsyncClient,
    env: None,
    alice: str,
) -> None:
    with respx.mock() as mock:
        mock.get(f"{_ZBASE}/accounts").mock(
            return_value=_accounts(("acc_tt", "tiktok", "prof_alice")),
        )
        r = await client.post(
            "/api/hub/v1/publish/connect",
            headers=_hub(alice, tier="pro"),
            json={"platform": "youtube", "return_url": f"{_HUB}/app/clips/publicar/conectado"},
        )
    err = _err(r, 402, "needs_plan")
    assert err["reason"] == "account_limit"
    assert "1" in err["message"]


# ---- error shaping + redirect → JSON ----


async def test_engine_4xx_copy_is_kept(client: httpx.AsyncClient, env: None, alice: str) -> None:
    r = await client.post(
        "/api/hub/v1/publish/schedule/slots",
        headers=_hub(alice),
        json={"slots": [{"dayOfWeek": 9, "time": "25:00"}]},
    )
    err = _err(r, 400, "bad_request")
    assert "dayOfWeek" in err["message"]


async def test_engine_5xx_never_leaks_vendor_text(
    client: httpx.AsyncClient,
    env: None,
    alice: str,
) -> None:
    with respx.mock(assert_all_called=True) as mock:
        mock.put(f"{_ZBASE}/queue/slots").mock(
            return_value=httpx.Response(500, json={"error": "SECRET internal vendor trace"}),
        )
        r = await client.post(
            "/api/hub/v1/publish/schedule/slots",
            headers=_hub(alice),
            json={"slots": [{"dayOfWeek": 1, "time": "09:00"}], "timezone": "UTC"},
        )
    assert r.status_code == 502, r.text
    body = r.json()
    assert body["ok"] is False and body["error"]["code"] == "internal"
    assert "SECRET" not in r.text


async def test_mark_published_redirect_becomes_json(
    client: httpx.AsyncClient,
    env: None,
    db: Database,
    alice: str,
) -> None:
    clip_id = await _approved_clip(db, alice)
    r = await client.post(
        f"/api/hub/v1/publish/clip/{clip_id}/mark-published",
        headers=_hub(alice),
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True}
    with bound_tenant(alice):
        assert (await ClipsRepo(db).get(clip_id)).status == "published"  # type: ignore[union-attr]
    # Second time it's a state conflict, in the hub shape.
    r = await client.post(
        f"/api/hub/v1/publish/clip/{clip_id}/mark-published",
        headers=_hub(alice),
    )
    _err(r, 409, "conflict")
    # Reopen comes back with its flash count.
    r = await client.post("/api/hub/v1/publish/reopen-all", headers=_hub(alice))
    assert r.json() == {"ok": True, "reopened": "1"}


async def test_other_tenants_clip_is_not_found(
    client: httpx.AsyncClient,
    env: None,
    db: Database,
    alice: str,
    bob: str,
) -> None:
    clip_id = await _approved_clip(db, alice)
    r = await client.post(
        f"/api/hub/v1/publish/clip/{clip_id}/mark-published",
        headers=_hub(bob),
    )
    _err(r, 404, "not_found")
    with bound_tenant(alice):
        assert (await ClipsRepo(db).get(clip_id)).status == "approved"  # type: ignore[union-attr]


async def test_post_validation_copy(
    client: httpx.AsyncClient, env: None, db: Database, alice: str
) -> None:
    clip_id = await _approved_clip(db, alice)
    r = await client.post(
        f"/api/hub/v1/publish/post/{clip_id}",
        headers=_hub(alice),
        json={"platforms": ["youtube"], "mode": "now"},
    )
    err = _err(r, 400, "bad_request")
    assert "YouTube necesita un título" in err["message"]


@pytest.fixture
def render_spy(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Stub the renderer; record what auth it was handed."""
    import chalybclip.api._clip_render as _clip_render

    monkeypatch.setenv("CHALYBCLIP_INTERNAL_SIGNING_SECRET", "sign_me_hub")
    get_settings.cache_clear()
    calls: list[dict[str, Any]] = []

    async def _rendered(**kw: Any) -> None:
        calls.append(kw)
        return None

    monkeypatch.setattr(_clip_render, "ensure_clip_rendered", _rendered)
    return calls


def _mock_create_post(mock: respx.Router, post_id: str) -> respx.Route:
    return mock.post(f"{_ZBASE}/posts").mock(
        return_value=httpx.Response(201, json={"success": True, "post": {"_id": post_id}})
    )


async def test_post_json_publishes_with_powerups(
    client: httpx.AsyncClient,
    env: None,
    render_spy: list[dict[str, Any]],
    db: Database,
    alice: str,
) -> None:
    clip_id = await _approved_clip(db, alice)
    with respx.mock() as mock:
        mock.get(f"{_ZBASE}/accounts").mock(
            return_value=_accounts(
                ("acc_tt", "tiktok", "prof_alice"), ("acc_yt", "youtube", "prof_alice")
            ),
        )
        post_route = _mock_create_post(mock, "post_hub_now")
        r = await client.post(
            f"/api/hub/v1/publish/post/{clip_id}",
            headers=_hub(alice),
            json={
                "platforms": ["tiktok", "youtube"],
                "title": "Mi título",
                "description": "Caption general",
                "mode": "now",
                "first_comment": "Sígueme 👉",
                "caption_tiktok": "Caption TikTok",
                "yt_title": "Título YT",
                "tiktok_privacy": "MUTUAL_FOLLOW_FRIENDS",
                "scheduled_for": None,
            },
        )
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "queued": "post_hub_now"}
    payload = json.loads(post_route.calls.last.request.content.decode())
    by_platform = {p["platform"]: p for p in payload["platforms"]}
    assert by_platform["tiktok"]["customContent"] == "Caption TikTok"
    assert by_platform["youtube"]["platformSpecificData"]["title"] == "Título YT"
    assert by_platform["tiktok"]["platformSpecificData"]["privacyLevel"] == "MUTUAL_FOLLOW_FRIENDS"
    # The hub has no session cookie: the recorder gets a signed render query.
    assert render_spy and render_spy[-1]["auth_cookie_value"] is None
    assert render_spy[-1]["auth_query"]
    row = await ZernioPublishesRepo(db).get_by_post_id("post_hub_now")
    assert row is not None and row.tenant_id == alice


async def test_draft_then_schedule_from_hub(
    client: httpx.AsyncClient,
    env: None,
    render_spy: list[dict[str, Any]],
    db: Database,
    alice: str,
) -> None:
    clip_id = await _approved_clip(db, alice)
    with respx.mock() as mock:
        mock.get(f"{_ZBASE}/accounts").mock(
            return_value=_accounts(("acc_tt", "tiktok", "prof_alice")),
        )
        _mock_create_post(mock, "post_draft")
        r = await client.post(
            f"/api/hub/v1/publish/post/{clip_id}",
            headers=_hub(alice),
            json={"platforms": "tiktok", "description": "hola", "mode": "draft"},
        )
    assert r.json() == {"ok": True, "draft": "saved"}
    with respx.mock() as mock:
        mock.get(f"{_ZBASE}/accounts").mock(
            return_value=_accounts(("acc_tt", "tiktok", "prof_alice")),
        )
        post_route = _mock_create_post(mock, "post_sched")
        mock.delete(f"{_ZBASE}/posts/post_draft").mock(return_value=httpx.Response(204))
        r = await client.post(
            "/api/hub/v1/publish/draft/post_draft/publish",
            headers=_hub(alice),
            json={"scheduled_for": "2031-01-02T15:00:00.000Z"},
        )
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "queued": "post_sched"}
    payload = json.loads(post_route.calls.last.request.content.decode())
    assert payload["scheduledFor"].startswith("2031-01-02T15:00")


async def test_json_body_must_be_object_for_form_routes(
    client: httpx.AsyncClient, env: None, db: Database, alice: str
) -> None:
    clip_id = await _approved_clip(db, alice)
    r = await client.post(
        f"/api/hub/v1/publish/post/{clip_id}",
        headers={**_hub(alice), "content-type": "application/json"},
        content=b"[1,2]",
    )
    _err(r, 400, "bad_request")


async def test_claim_json(client: httpx.AsyncClient, env: None, db: Database, bob: str) -> None:
    # Alice's profile is bound to Alice → Bob gets the not-found copy.
    await TenantsRepo(db).set_zernio_profile(
        (await TenantsRepo(db).create(name="Carol")).id, profile_id="prof_carol"
    )
    r = await client.post(
        "/api/hub/v1/publish/accounts/claim", headers=_hub(bob), json={"profile_id": "prof_carol"}
    )
    _err(r, 404, "not_found")
    with respx.mock() as mock:
        mock.get(f"{_ZBASE}/accounts").mock(
            return_value=_accounts(("acc_x", "tiktok", "prof_free")),
        )
        r = await client.post(
            "/api/hub/v1/publish/accounts/claim",
            headers=_hub(bob),
            json={"profile_id": "prof_free"},
        )
    assert r.json() == {"ok": True, "claimed": "prof_free"}


# ---- ownership gates ----


async def test_status_of_other_tenants_post_is_404_without_vendor_call(
    client: httpx.AsyncClient,
    env: None,
    db: Database,
    alice: str,
    bob: str,
) -> None:
    await ZernioPublishesRepo(db).record(
        post_id="p_bob",
        tenant_id=bob,
        clip_id="clp_b",
        platforms=["tiktok"],
        content="x",
        status="published",
    )
    with respx.mock(assert_all_called=False) as mock:
        live = mock.get(f"{_ZBASE}/posts/p_bob").mock(
            return_value=httpx.Response(
                200, json={"post": {"_id": "p_bob", "status": "published"}}
            ),
        )
        r = await client.get("/api/hub/v1/publish/status/p_bob.json", headers=_hub(alice))
        _err(r, 404, "not_found")
        # Unknown post (no local record) is not looked up either.
        r = await client.get("/api/hub/v1/publish/status/p_zzz.json", headers=_hub(alice))
        _err(r, 404, "not_found")
    assert not live.called


async def test_status_of_own_post(
    client: httpx.AsyncClient, env: None, db: Database, alice: str
) -> None:
    await ZernioPublishesRepo(db).record(
        post_id="p_a",
        tenant_id=alice,
        clip_id="clp_a",
        platforms=["tiktok"],
        content="x",
    )
    with respx.mock() as mock:
        mock.get(f"{_ZBASE}/posts/p_a").mock(
            return_value=httpx.Response(
                200,
                json={
                    "post": {
                        "_id": "p_a",
                        "status": "published",
                        "platforms": [{"platform": "tiktok", "status": "published"}],
                    }
                },
            ),
        )
        r = await client.get("/api/hub/v1/publish/status/p_a.json", headers=_hub(alice))
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "PUBLISHED"


async def test_retry_gate(
    client: httpx.AsyncClient, env: None, db: Database, alice: str, bob: str
) -> None:
    await ZernioPublishesRepo(db).record(
        post_id="p_bob",
        tenant_id=bob,
        clip_id="clp_b",
        platforms=["tiktok"],
        content="x",
        status="failed",
    )
    with respx.mock(assert_all_called=False) as mock:
        retry_bob = mock.post(f"{_ZBASE}/posts/p_bob/retry").mock(
            return_value=httpx.Response(200, json={"post": {"_id": "p_bob"}}),
        )
        mock.get(f"{_ZBASE}/posts").mock(
            return_value=httpx.Response(200, json={"posts": [{"_id": "p_mine_remote"}]}),
        )
        retry_mine = mock.post(f"{_ZBASE}/posts/p_mine_remote/retry").mock(
            return_value=httpx.Response(200, json={"post": {"_id": "p_mine_remote"}}),
        )
        retry_other = mock.post(f"{_ZBASE}/posts/p_unlisted/retry").mock(
            return_value=httpx.Response(200, json={"post": {"_id": "p_unlisted"}}),
        )
        r = await client.post("/api/hub/v1/publish/retry/p_bob", headers=_hub(alice))
        _err(r, 404, "not_found")
        r = await client.post("/api/hub/v1/publish/retry/p_unlisted", headers=_hub(alice))
        _err(r, 404, "not_found")
        r = await client.post("/api/hub/v1/publish/retry/p_mine_remote", headers=_hub(alice))
        assert r.status_code == 200, r.text
    assert not retry_bob.called and not retry_other.called and retry_mine.called


async def test_cancel_gate(
    client: httpx.AsyncClient, env: None, db: Database, alice: str, bob: str
) -> None:
    await ZernioPublishesRepo(db).record(
        post_id="p_bob",
        tenant_id=bob,
        clip_id="clp_b",
        platforms=["tiktok"],
        content="x",
        status="scheduled",
    )
    with respx.mock(assert_all_called=False) as mock:
        delete = mock.delete(f"{_ZBASE}/posts/p_bob").mock(return_value=httpx.Response(204))
        r = await client.post("/api/hub/v1/publish/schedule/cancel/p_bob", headers=_hub(alice))
    _err(r, 404, "not_found")
    assert not delete.called


async def test_sequence_and_automation_gates(
    client: httpx.AsyncClient,
    env: None,
    alice: str,
) -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"{_ZBASE}/sequences").mock(
            return_value=httpx.Response(200, json={"sequences": [{"_id": "seq_mine"}]}),
        )
        mock.get(f"{_ZBASE}/comment-automations").mock(
            return_value=httpx.Response(200, json={"automations": [{"_id": "auto_mine"}]}),
        )
        foreign_seq = mock.post(url__regex=r".*/sequences/seq_other/.*").mock(
            return_value=httpx.Response(200, json={}),
        )
        foreign_auto = mock.patch(url__regex=r".*/comment-automations/auto_other.*").mock(
            return_value=httpx.Response(200, json={}),
        )
        for path in (
            "/growth/sequences/seq_other/toggle",
            "/growth/sequences/seq_other/enroll",
            "/growth/automations/auto_other/toggle",
        ):
            r = await client.post(
                f"/api/hub/v1/publish{path}",
                headers=_hub(alice),
                json={"active": True},
            )
            _err(r, 404, "not_found")
    assert not foreign_seq.called and not foreign_auto.called


async def test_every_route_is_hub_only(
    client: httpx.AsyncClient, env: None, tenants: dict[str, dict[str, str]]
) -> None:
    """No route under /api/hub/v1/publish answers a dashboard credential."""
    from chalybclip.api.routers.hub_publish import router

    paths = {
        (next(iter(r.methods)), r.path)  # type: ignore[attr-defined]
        for r in router.routes
    }
    assert len(paths) >= 60
    for method, path in sorted(paths):
        url = path.replace("{", "").replace("}", "")
        r = await client.request(method, url, headers=auth(tenants["alice"]["token"]))
        assert r.status_code == 401, (method, path, r.status_code)
        assert json.loads(r.text)["error"]["code"] == "unauthorized"
