"""Regression: a VIP tenant gets the FULL hub — not the free/upsell
view. Chalyb renamed its top tier ALL_ACCESS → VIP; `vip` must
resolve to all_access everywhere a gate reads the tier.

Pins the end-to-end path: SSO sync normalizes vip → all_access, the
dashboard renders the Crecimiento panels (not the upsell card), and the
Pro-gated growth routes accept the VIP (no 402).
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
import pytest_asyncio
import respx

from chalybclip.db import Database, TenantsRepo
from chalybclip.integrations.chalyb.service import sync_tenant_tier
from chalybclip.settings import get_settings
from chalybclip.tiers import normalize_tier, zernio_account_limit

from .conftest import auth

_ZBASE = "https://zernio.com/api/v1"


@pytest.fixture
def zernio_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("CHALYBCLIP_ZERNIO_API_KEY", "sk_test_vip")
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


@pytest_asyncio.fixture
async def vip(
    db: Database, tenants: dict[str, dict[str, str]]
) -> dict[str, str]:
    """Alice synced as a VIP (the label Chalyb sends) + a Zernio profile."""
    tid = tenants["alice"]["id"]
    await sync_tenant_tier(db, tenant_id=tid, tier="vip")
    await TenantsRepo(db).set_zernio_profile(
        tid, profile_id="prof_alice", profile_name="Alice",
    )
    return tenants["alice"]


# ---- pure: vip resolves to the top tier everywhere ----


def test_vip_normalizes_to_all_access() -> None:
    assert normalize_tier("vip") == "all_access"
    assert normalize_tier("VIP") == "all_access"
    # VIP is unlimited connected accounts (the top-tier cap).
    assert zernio_account_limit("vip") is None


@pytest.mark.asyncio
async def test_sync_writes_canonical_for_vip(
    db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    tid = tenants["alice"]["id"]
    await sync_tenant_tier(db, tenant_id=tid, tier="vip")
    tenant = await TenantsRepo(db).get(tid)
    assert tenant is not None
    assert tenant.tier == "all_access"  # stored canonical, not "vip"


# ---- end-to-end: VIP sees the full dashboard, not the upsell ----


@pytest.mark.asyncio
async def test_vip_dashboard_renders_growth_panels_not_upsell(
    zernio_env: None, client: httpx.AsyncClient, vip: dict[str, str]
) -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"{_ZBASE}/accounts").mock(
            return_value=httpx.Response(200, json={"accounts": []})
        )
        mock.get(f"{_ZBASE}/posts").mock(
            return_value=httpx.Response(200, json={"posts": []})
        )
        resp = await client.get(
            "/dashboard/publish/zernio", headers=auth(vip["token"]),
        )
    assert resp.status_code == 200
    html = resp.text
    # The Crecimiento tab is present…
    assert 'data-tab="crecimiento"' in html
    # …and shows the Pro panels (automations/broadcast), NOT the upsell.
    assert "Crecimiento es una función Pro" not in html
    assert 'id="zn-auto-create"' in html  # automation panel rendered


@pytest.mark.asyncio
async def test_vip_passes_growth_route(
    zernio_env: None, client: httpx.AsyncClient, vip: dict[str, str]
) -> None:
    with respx.mock() as mock:
        mock.get(f"{_ZBASE}/accounts").mock(
            return_value=httpx.Response(200, json={"accounts": []})
        )
        resp = await client.get(
            "/dashboard/publish/zernio/growth/contacts.json",
            headers=auth(vip["token"]),
        )
    # Pro gate accepts VIP → 200, not 402.
    assert resp.status_code == 200
