"""Community settings routes (phase 11)."""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
import pytest_asyncio
import respx

from chalybclip.db import Database, TenantsRepo, ZernioCommunityRepo
from chalybclip.integrations.chalyb.service import sync_tenant_tier
from chalybclip.settings import get_settings

from .conftest import auth

_ZBASE = "https://zernio.com/api/v1"


@pytest.fixture
def zernio_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("CHALYBCLIP_ZERNIO_API_KEY", "sk_test_comm")
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


def _mock_channels(mock: respx.Router) -> None:
    mock.get(f"{_ZBASE}/accounts").mock(
        return_value=httpx.Response(
            200,
            json={"accounts": [
                {"platform": "discord", "_id": "acct_discord", "profileId": "prof_alice"},
                {"platform": "telegram", "_id": "acct_tg", "profileId": "prof_alice"},
                {"platform": "tiktok", "_id": "acct_tt", "profileId": "prof_alice"},
            ]},
        )
    )


@pytest.mark.asyncio
async def test_settings_json_offers_only_community_channels(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_channels(mock)
        resp = await client.get(
            "/dashboard/publish/zernio/community/settings.json",
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    channels = resp.json()["channels"]
    platforms = {c["platform"] for c in channels}
    # Only discord/telegram are offered as community channels (not tiktok).
    assert platforms == {"discord", "telegram"}


@pytest.mark.asyncio
async def test_save_settings_persists(
    zernio_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_channels(mock)
        resp = await client.post(
            "/dashboard/publish/zernio/community/settings",
            json={"enabled": True, "discord_account_id": "acct_discord",
                  "brand_name": "Mi Canal", "weekly_digest": True},
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    s = await ZernioCommunityRepo(db).get_settings(alice["id"])
    assert s is not None
    assert s["enabled"] is True
    assert s["discord_account_id"] == "acct_discord"
    assert s["brand_name"] == "Mi Canal"
    assert s["weekly_digest"] is True


@pytest.mark.asyncio
async def test_save_settings_rejects_unowned_channel(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_channels(mock)
        resp = await client.post(
            "/dashboard/publish/zernio/community/settings",
            json={"enabled": True, "discord_account_id": "acct_not_mine"},
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_connect_allows_discord(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    """Discord/Telegram are connectable (community channels) even though
    they're not publish targets."""
    with respx.mock() as mock:
        mock.get(f"{_ZBASE}/connect/discord").mock(
            return_value=httpx.Response(200, json={"authUrl": "https://discord/oauth"})
        )
        resp = await client.post(
            "/dashboard/publish/zernio/connect",
            json={"platform": "discord"},
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    assert resp.json()["auth_url"] == "https://discord/oauth"
