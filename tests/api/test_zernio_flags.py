"""Feature-flagged extras (Hub phase 12): flags off → 404; on → happy
paths for ads boost / campaigns + whatsapp number status."""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest
import pytest_asyncio
import respx

from chalybclip.db import (
    Database,
    TenantsRepo,
    ZernioEventsRepo,
    ZernioWhatsappNumbersRepo,
)
from chalybclip.integrations.chalyb.service import sync_tenant_tier
from chalybclip.integrations.zernio.events import process_zernio_event
from chalybclip.settings import get_settings

from .conftest import auth

_ZBASE = "https://zernio.com/api/v1"
_ACC = "acct_ig_1"


def _env(monkeypatch: pytest.MonkeyPatch, *, ads: bool, whatsapp: bool) -> None:
    monkeypatch.setenv("CHALYBCLIP_ZERNIO_API_KEY", "sk_test_flags")
    monkeypatch.setenv("CHALYBCLIP_FEATURE_ADS", "1" if ads else "0")
    monkeypatch.setenv("CHALYBCLIP_FEATURE_WHATSAPP", "1" if whatsapp else "0")
    get_settings.cache_clear()


@pytest.fixture
def flags_off(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    _env(monkeypatch, ads=False, whatsapp=False)
    try:
        yield
    finally:
        get_settings.cache_clear()


@pytest.fixture
def flags_on(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    _env(monkeypatch, ads=True, whatsapp=True)
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


def _mock_accounts(mock: respx.Router) -> None:
    mock.get(f"{_ZBASE}/accounts").mock(
        return_value=httpx.Response(
            200,
            json={"accounts": [
                {"platform": "instagram", "_id": _ACC, "profileId": "prof_alice"}
            ]},
        )
    )


# ---- flags off → 404 (invisible surface) ----


@pytest.mark.asyncio
async def test_ads_campaigns_404_when_flag_off(
    flags_off: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    resp = await client.get(
        "/dashboard/publish/zernio/ads/campaigns.json",
        headers=auth(alice["token"]),
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_ads_boost_404_when_flag_off(
    flags_off: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    resp = await client.post(
        "/dashboard/publish/zernio/ads/boost",
        json={"confirm": True, "account_id": _ACC},
        headers=auth(alice["token"]),
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_whatsapp_status_404_when_flag_off(
    flags_off: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    resp = await client.get(
        "/dashboard/publish/zernio/whatsapp/status.json",
        headers=auth(alice["token"]),
    )
    assert resp.status_code == 404


# ---- flags on → happy paths ----


@pytest.mark.asyncio
async def test_ads_campaigns_listed_when_flag_on(
    flags_on: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        mock.get(f"{_ZBASE}/ads/campaigns").mock(
            return_value=httpx.Response(
                200, json={"campaigns": [{"id": "c1", "name": "Promo", "status": "active"}]}
            )
        )
        resp = await client.get(
            "/dashboard/publish/zernio/ads/campaigns.json",
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    assert resp.json()["campaigns"][0]["id"] == "c1"


@pytest.mark.asyncio
async def test_ads_boost_requires_confirm(
    flags_on: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    resp = await client.post(
        "/dashboard/publish/zernio/ads/boost",
        json={"account_id": _ACC, "ad_account_id": "act_1", "name": "x",
              "goal": "engagement", "budget_amount": 10, "budget_type": "daily"},
        headers=auth(alice["token"]),
    )
    assert resp.status_code == 400  # confirm missing


@pytest.mark.asyncio
async def test_ads_boost_happy_path(
    flags_on: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock)
        route = mock.post(f"{_ZBASE}/ads/boost").mock(
            return_value=httpx.Response(200, json={"campaignId": "camp1"})
        )
        resp = await client.post(
            "/dashboard/publish/zernio/ads/boost",
            json={"account_id": _ACC, "ad_account_id": "act_1", "name": "Boost clip",
                  "goal": "engagement", "budget_amount": 15, "budget_type": "daily",
                  "platform_post_id": "pp1", "confirm": True},
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    payload = json.loads(route.calls.last.request.content.decode())
    assert payload["adAccountId"] == "act_1"
    assert payload["goal"] == "engagement"
    assert payload["budget"] == {"amount": 15.0, "type": "daily"}
    assert payload["platformPostId"] == "pp1"


@pytest.mark.asyncio
async def test_ads_boost_unowned_account_403(
    flags_on: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock)
        resp = await client.post(
            "/dashboard/publish/zernio/ads/boost",
            json={"account_id": "acct_nope", "ad_account_id": "act_1", "name": "x",
                  "goal": "engagement", "budget_amount": 10, "budget_type": "daily",
                  "confirm": True},
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_whatsapp_status_from_webhook_when_flag_on(
    flags_on: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    # A whatsapp.number.activated webhook records the status…
    await ZernioEventsRepo(db).insert_dedup(
        event_id="ev_wa", type="whatsapp.number.activated",
        payload=json.dumps({
            "id": "ev_wa", "event": "whatsapp.number.activated",
            "account": {"id": _ACC, "platform": "whatsapp"},
        }),
    )
    await process_zernio_event(db, "ev_wa")
    with respx.mock() as mock:
        _mock_accounts(mock)
        resp = await client.get(
            "/dashboard/publish/zernio/whatsapp/status.json",
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    nums = resp.json()["numbers"]
    assert nums[0]["account_id"] == _ACC
    assert nums[0]["status"] == "activated"


@pytest.mark.asyncio
async def test_whatsapp_number_webhook_records_status(
    flags_on: None, db: Database
) -> None:
    await ZernioEventsRepo(db).insert_dedup(
        event_id="ev1", type="whatsapp.number.suspended",
        payload=json.dumps({
            "id": "ev1", "event": "whatsapp.number.suspended",
            "account": {"id": "acct_x"}, "reason": "policy",
        }),
    )
    await process_zernio_event(db, "ev1")
    rows = await ZernioWhatsappNumbersRepo(db).list_for_accounts(["acct_x"])
    assert rows[0]["status"] == "suspended"
    assert rows[0]["detail"] == "policy"
