"""Growth layer routes (Hub phase 10): automations, sequences,
broadcasts — platform enforcement, step validation, daily cap, Pro
gating."""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest
import pytest_asyncio
import respx

from chalybclip.db import Database, TenantsRepo, ZernioInboxRepo
from chalybclip.integrations.chalyb.service import sync_tenant_tier
from chalybclip.settings import get_settings

from .conftest import auth

_ZBASE = "https://zernio.com/api/v1"
_IG = "acct_ig_1"
_TT = "acct_tt_1"


@pytest.fixture
def zernio_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("CHALYBCLIP_ZERNIO_API_KEY", "sk_test_growth")
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
    await sync_tenant_tier(db, tenant_id=tid, tier="all_access")  # Pro+
    await TenantsRepo(db).set_zernio_profile(
        tid, profile_id="prof_alice", profile_name="Alice",
    )
    return tenants["alice"]


def _mock_accounts(mock: respx.Router) -> None:
    mock.get(f"{_ZBASE}/accounts").mock(
        return_value=httpx.Response(
            200,
            json={"accounts": [
                {"platform": "instagram", "_id": _IG, "profileId": "prof_alice"},
                {"platform": "tiktok", "_id": _TT, "profileId": "prof_alice"},
            ]},
        )
    )


# ---- Pro gating ----


@pytest.mark.asyncio
async def test_growth_blocked_for_free_tier(
    zernio_env: None,
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    # bob is free → 402 (the JS shows upsell), never a 500.
    resp = await client.get(
        "/dashboard/publish/zernio/growth/contacts.json",
        headers=auth(tenants["bob"]["token"]),
    )
    assert resp.status_code == 402


# ---- automations: IG/FB enforcement ----


@pytest.mark.asyncio
async def test_create_automation_on_instagram_ok(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock)
        route = mock.post(f"{_ZBASE}/comment-automations").mock(
            return_value=httpx.Response(200, json={"id": "auto1"})
        )
        resp = await client.post(
            "/dashboard/publish/zernio/growth/automations",
            json={"account_id": _IG, "name": "Clip link",
                  "dm_message": "Aquí 👉 {url}", "keywords": ["CLIP"],
                  "platform_post_id": "pp1"},
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    payload = json.loads(route.calls.last.request.content.decode())
    assert payload["accountId"] == _IG
    assert payload["keywords"] == ["CLIP"]


@pytest.mark.asyncio
async def test_create_automation_on_tiktok_rejected(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock)  # no automations route → must not be called
        resp = await client.post(
            "/dashboard/publish/zernio/growth/automations",
            json={"account_id": _TT, "name": "x", "dm_message": "y"},
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 409
    assert "instagram" in resp.json()["error"].lower()


@pytest.mark.asyncio
async def test_create_automation_unowned_account_403(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock)
        resp = await client.post(
            "/dashboard/publish/zernio/growth/automations",
            json={"account_id": "acct_not_mine", "name": "x", "dm_message": "y"},
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 403


# ---- sequences: step validation + template ----


@pytest.mark.asyncio
async def test_create_sequence_bienvenida_template(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock)
        route = mock.post(f"{_ZBASE}/sequences").mock(
            return_value=httpx.Response(200, json={"id": "seq1"})
        )
        resp = await client.post(
            "/dashboard/publish/zernio/growth/sequences",
            json={"account_id": _IG, "name": "Bienvenida Chalyb",
                  "template": "bienvenida"},
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    payload = json.loads(route.calls.last.request.content.decode())
    assert len(payload["steps"]) == 3
    assert payload["steps"][0]["delayMinutes"] == 0
    assert payload["platform"] == "instagram"


@pytest.mark.asyncio
async def test_create_sequence_bad_steps_400(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock)  # no sequences route → not called
        resp = await client.post(
            "/dashboard/publish/zernio/growth/sequences",
            json={"account_id": _IG, "name": "Mala",
                  "steps": [{"delayMinutes": -5, "message": {"text": ""}}]},
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 400


# ---- broadcasts: daily cap guardrail ----


@pytest.mark.asyncio
async def test_broadcast_requires_confirm(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    # The confirm check fires before any account resolution.
    resp = await client.post(
        "/dashboard/publish/zernio/growth/broadcasts/send",
        json={"account_id": _IG, "name": "Promo", "message": "hola"},
        headers=auth(alice["token"]),
    )
    assert resp.status_code == 400
    assert "confirma" in resp.json()["error"].lower()


@pytest.mark.asyncio
async def test_broadcast_send_then_daily_cap_blocks_second(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock)
        mock.post(f"{_ZBASE}/broadcasts").mock(
            return_value=httpx.Response(200, json={"id": "b1"})
        )
        mock.post(f"{_ZBASE}/broadcasts/b1/recipients").mock(
            return_value=httpx.Response(200, json={"added": 5})
        )
        mock.post(f"{_ZBASE}/broadcasts/b1/send").mock(
            return_value=httpx.Response(200, json={"status": "sending", "sent": 5})
        )
        body = {"account_id": _IG, "name": "Promo", "message": "¡Nuevo clip!",
                "confirm": True}
        first = await client.post(
            "/dashboard/publish/zernio/growth/broadcasts/send",
            json=body, headers=auth(alice["token"]),
        )
        # Second send same UTC day → blocked by the cap (default 1).
        second = await client.post(
            "/dashboard/publish/zernio/growth/broadcasts/send",
            json=body, headers=auth(alice["token"]),
        )
    assert first.status_code == 200
    assert first.json()["broadcast_id"] == "b1"
    assert second.status_code == 429
    assert second.json()["reason"] == "daily_cap"


@pytest.mark.asyncio
async def test_broadcast_unowned_account_403(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock)
        resp = await client.post(
            "/dashboard/publish/zernio/growth/broadcasts/send",
            json={"account_id": "acct_nope", "name": "x", "message": "y",
                  "confirm": True},
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 403


# ---- contacts feed (seeded in phase 9) ----


@pytest.mark.asyncio
async def test_contacts_feed_with_tag_filter(
    zernio_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    inbox = ZernioInboxRepo(db)
    await inbox.upsert_contact(
        account_id=_IG, contact_key="u1", platform="instagram",
        name="Fan", username="fan1", tag="comment-lead",
    )
    await inbox.upsert_contact(
        account_id=_IG, contact_key="u2", platform="instagram",
        name="Cliente", username="cli", tag="dm-lead",
    )
    with respx.mock() as mock:
        _mock_accounts(mock)
        resp = await client.get(
            "/dashboard/publish/zernio/growth/contacts.json?tag=dm-lead",
            headers=auth(alice["token"]),
        )
    rows = resp.json()["contacts"]
    assert {r["contact_key"] for r in rows} == {"u2"}
