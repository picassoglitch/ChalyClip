"""Zernio connect-flow router tests (Publish Hub phase 1).

Pins the white-label connect contract:
  - connect URLs carry redirect_url back to OUR /connected page with
    the platform on the query string, and headless=true for platforms
    that need a post-OAuth selection (Facebook page)
  - /connected postMessages `zernio:connected` + platform and closes;
    on a headless Facebook redirect (tempToken, no accountId) it
    renders OUR page picker instead
  - /fb-pages and /fb-page/select proxy Zernio's select-page endpoints
    with the temp token in the BODY (never the URL) and the profileId
    resolved server-side from the tenant

Zernio is respx-mocked; the app is exercised over ASGI (respx leaves
ASGITransport alone, so only the router's outbound calls are faked).
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest
import pytest_asyncio
import respx

from chalybclip.db import Database, TenantsRepo
from chalybclip.integrations.chalyb.service import sync_tenant_tier
from chalybclip.settings import get_settings

from .conftest import auth

_ZBASE = "https://zernio.com/api/v1"


@pytest.fixture
def zernio_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Configure the company-wide Zernio key + bust the settings cache."""
    monkeypatch.setenv("CHALYBCLIP_ZERNIO_API_KEY", "sk_test_router")
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


@pytest_asyncio.fixture
async def alice(
    db: Database, tenants: dict[str, dict[str, str]]
) -> dict[str, str]:
    """Alice with a paid tier (no account cap → no list_accounts call on
    connect) and a bound Zernio profile."""
    await sync_tenant_tier(db, tenant_id=tenants["alice"]["id"], tier="all_access")
    await TenantsRepo(db).set_zernio_profile(
        tenants["alice"]["id"], profile_id="prof_alice", profile_name="Alice",
    )
    return tenants["alice"]


# ---- POST /connect — redirect_url + headless ----


@pytest.mark.asyncio
async def test_dashboard_autoprovisions_profile_so_tabs_appear(
    zernio_env: None, client: httpx.AsyncClient, db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """A tenant with NO Zernio profile opens the Publish Center → the
    profile is auto-provisioned and the tabs render (no manual step)."""
    tid = tenants["alice"]["id"]  # no set_zernio_profile → profile_id is None
    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"{_ZBASE}/profiles").mock(
            return_value=httpx.Response(200, json={"profiles": []})
        )
        mock.post(f"{_ZBASE}/profiles").mock(
            return_value=httpx.Response(
                201, json={"profile": {"_id": "prof_auto", "name": f"ChalyClip {tid}"}}
            )
        )
        mock.get(f"{_ZBASE}/accounts").mock(
            return_value=httpx.Response(200, json={"accounts": []})
        )
        mock.get(f"{_ZBASE}/posts").mock(
            return_value=httpx.Response(200, json={"posts": []})
        )
        resp = await client.get(
            "/dashboard/publish/zernio", headers=auth(tenants["alice"]["token"]),
        )
    assert resp.status_code == 200
    # Tabs render now (gated on profile_id, which auto-provision filled).
    assert 'data-tab="single"' in resp.text
    # And it persisted.
    tenant = await TenantsRepo(db).get(tid)
    assert tenant is not None and tenant.zernio_profile_id == "prof_auto"


@pytest.mark.asyncio
async def test_connect_facebook_is_headless_with_platform_in_redirect(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock(assert_all_called=True) as mock:
        route = mock.get(f"{_ZBASE}/connect/facebook").mock(
            return_value=httpx.Response(200, json={"authUrl": "https://fb/oauth"})
        )
        resp = await client.post(
            "/dashboard/publish/zernio/connect",
            json={"platform": "facebook"},
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "auth_url": "https://fb/oauth"}
    sent = route.calls.last.request
    assert sent.url.params.get("profileId") == "prof_alice"
    # Facebook needs a page selection → headless mode, we render the picker.
    assert sent.url.params.get("headless") == "true"
    # The platform rides on the redirect so /connected can tell the opener.
    assert sent.url.params.get("redirect_url") == (
        "http://api.test/dashboard/publish/zernio/connected?platform=facebook"
    )


@pytest.mark.asyncio
async def test_connect_tiktok_standard_mode_no_headless(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock(assert_all_called=True) as mock:
        route = mock.get(f"{_ZBASE}/connect/tiktok").mock(
            return_value=httpx.Response(200, json={"authUrl": "https://tt/oauth"})
        )
        resp = await client.post(
            "/dashboard/publish/zernio/connect",
            json={"platform": "tiktok"},
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    sent = route.calls.last.request
    assert "headless" not in sent.url.params
    assert sent.url.params.get("redirect_url", "").endswith(
        "/dashboard/publish/zernio/connected?platform=tiktok"
    )


# ---- GET /connected — close page vs headless picker ----


@pytest.mark.asyncio
async def test_connected_page_posts_message_with_platform_and_closes(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    resp = await client.get(
        "/dashboard/publish/zernio/connected?platform=tiktok",
        headers=auth(alice["token"]),
    )
    assert resp.status_code == 200
    html = resp.text
    assert "zernio:connected" in html
    assert "postMessage" in html
    assert "window.close()" in html
    # Platform is read client-side from the query string (works for our
    # ?platform=X and for Zernio's standard ?connected=X append).
    assert 'get("platform")' in html
    assert 'get("connected")' in html


@pytest.mark.asyncio
async def test_connected_headless_facebook_renders_page_picker(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    resp = await client.get(
        "/dashboard/publish/zernio/connected?platform=facebook&tempToken=EAAtmp",
        headers=auth(alice["token"]),
    )
    assert resp.status_code == 200
    html = resp.text
    # The picker, not the close page.
    assert "fb-pages" in html
    assert "fb-page/select" in html
    assert "window.close()" not in html.split("fb-form")[0]  # no eager close
    # The temp token must NOT be interpolated server-side (JS reads
    # location.search itself).
    assert "EAAtmp" not in html


@pytest.mark.asyncio
async def test_connected_with_accountid_closes_even_with_temptoken(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    """accountId on the redirect = Zernio already created the account
    (no selection needed) → plain close page."""
    resp = await client.get(
        "/dashboard/publish/zernio/connected"
        "?platform=facebook&tempToken=EAAtmp&accountId=acct_1",
        headers=auth(alice["token"]),
    )
    assert resp.status_code == 200
    assert "zernio:connected" in resp.text
    assert "fb-pages" not in resp.text


# ---- POST /fb-pages ----


@pytest.mark.asyncio
async def test_fb_pages_lists_pages_with_server_side_profile(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    body = {"pages": [{"id": "123", "name": "Mi Página", "category": "Brand"}]}
    with respx.mock(assert_all_called=True) as mock:
        route = mock.get(f"{_ZBASE}/connect/facebook/select-page").mock(
            return_value=httpx.Response(200, json=body)
        )
        resp = await client.post(
            "/dashboard/publish/zernio/fb-pages",
            json={"temp_token": "EAAtmp"},
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    assert resp.json() == {
        "ok": True,
        "pages": [
            {"page_id": "123", "name": "Mi Página", "username": None, "category": "Brand"}
        ],
    }
    sent = route.calls.last.request
    # profileId comes from the tenant row, never from the popup.
    assert sent.url.params.get("profileId") == "prof_alice"
    assert sent.url.params.get("tempToken") == "EAAtmp"


@pytest.mark.asyncio
async def test_fb_pages_missing_token_is_400(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    resp = await client.post(
        "/dashboard/publish/zernio/fb-pages",
        json={},
        headers=auth(alice["token"]),
    )
    assert resp.status_code == 400
    assert resp.json()["ok"] is False


# ---- POST /fb-page/select ----


@pytest.mark.asyncio
async def test_fb_select_posts_selection_and_decodes_user_profile(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    zbody = {"account": {"accountId": "acct_fb_9", "platform": "facebook"}}
    user_profile = {"id": "987", "name": "Alice", "profilePicture": "https://p/x.jpg"}
    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(f"{_ZBASE}/connect/facebook/select-page").mock(
            return_value=httpx.Response(200, json=zbody)
        )
        resp = await client.post(
            "/dashboard/publish/zernio/fb-page/select",
            json={
                "page_id": "123",
                "temp_token": "EAAtmp",
                "user_profile_raw": json.dumps(user_profile),
            },
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "account_id": "acct_fb_9"}
    payload = json.loads(route.calls.last.request.content.decode())
    assert payload == {
        "profileId": "prof_alice",
        "pageId": "123",
        "tempToken": "EAAtmp",
        "userProfile": user_profile,
    }


@pytest.mark.asyncio
async def test_fb_select_zernio_402_maps_to_friendly_error(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        mock.post(f"{_ZBASE}/connect/facebook/select-page").mock(
            return_value=httpx.Response(402, json={"error": "limit"})
        )
        resp = await client.post(
            "/dashboard/publish/zernio/fb-page/select",
            json={"page_id": "1", "temp_token": "t", "user_profile_raw": "{}"},
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 502
    body = resp.json()
    assert body["ok"] is False
    assert "plan limit" in body["error"].lower()
    assert body["status"] == 402


@pytest.mark.asyncio
async def test_fb_select_missing_fields_is_400(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    resp = await client.post(
        "/dashboard/publish/zernio/fb-page/select",
        json={"page_id": "123"},
        headers=auth(alice["token"]),
    )
    assert resp.status_code == 400


# ---- GET /accounts-panel — reload-free chip refresh ----


@pytest.mark.asyncio
async def test_accounts_panel_renders_connected_chip_and_connect_buttons(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    zbody = {"accounts": [{"platform": "tiktok", "_id": "acct_tt_1"}]}
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{_ZBASE}/accounts").mock(
            return_value=httpx.Response(200, json=zbody)
        )
        resp = await client.get(
            "/dashboard/publish/zernio/accounts-panel",
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    html = resp.text
    # TikTok shows as a connected chip with its disconnect button…
    assert "js-zn-disconnect" in html
    assert "acct_tt_1" in html
    # …while not-yet-connected platforms keep their connect buttons.
    assert "js-zn-connect" in html
    assert 'data-platform="instagram"' in html


@pytest.mark.asyncio
async def test_accounts_panel_502_when_zernio_unreachable(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    """A failed Zernio fetch must NOT render an empty (all-disconnected)
    panel — 502 makes the popup JS fall back to a full reload."""
    with respx.mock() as mock:
        mock.get(f"{_ZBASE}/accounts").mock(
            side_effect=httpx.ReadTimeout("slow")
        )
        resp = await client.get(
            "/dashboard/publish/zernio/accounts-panel",
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 502
