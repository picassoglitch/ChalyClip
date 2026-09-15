"""Growth Engine API: settings knobs, the per-platform rulebook, growth card."""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
import pytest_asyncio

from chalybclip.db import AutopublishSettingsRepo, Database, PlatformPacingRulesRepo, TenantsRepo
from chalybclip.integrations.chalyb.service import sync_tenant_tier
from chalybclip.settings import get_settings

from .conftest import auth


@pytest.fixture
def zernio_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("CHALYBCLIP_ZERNIO_API_KEY", "sk_test_ge")
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


@pytest_asyncio.fixture
async def alice(db: Database, tenants: dict[str, dict[str, str]]) -> dict[str, str]:
    tid = tenants["alice"]["id"]
    await sync_tenant_tier(db, tenant_id=tid, tier="all_access")
    await TenantsRepo(db).set_zernio_profile(tid, profile_id="prof_alice", profile_name="Alice")
    return tenants["alice"]


_BASE = "/dashboard/publish/zernio"


@pytest.mark.asyncio
async def test_save_persists_growth_knobs(
    zernio_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    resp = await client.post(
        f"{_BASE}/autopublish/save",
        json={
            "enabled": True, "mode": "hands_free", "targets": ["tiktok"],
            "post_mode": "queue", "daily_cap": 5,
            "growth_min_score": 60, "daily_clip_budget": 15,
        },
        headers=auth(alice["token"]),
    )
    assert resp.status_code == 200
    s = await AutopublishSettingsRepo(db).get(alice["id"])
    assert s["growth_min_score"] == 60
    assert s["daily_clip_budget"] == 15


@pytest.mark.asyncio
async def test_growth_knobs_clamped(
    zernio_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    resp = await client.post(
        f"{_BASE}/autopublish/save",
        json={"enabled": True, "growth_min_score": 999, "daily_clip_budget": "bad"},
        headers=auth(alice["token"]),
    )
    assert resp.status_code == 200
    s = await AutopublishSettingsRepo(db).get(alice["id"])
    assert s["growth_min_score"] == 100  # clamped
    assert s["daily_clip_budget"] is None  # invalid → null


@pytest.mark.asyncio
async def test_rules_json_returns_defaults(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    resp = await client.get(f"{_BASE}/autopublish/rules.json", headers=auth(alice["token"]))
    assert resp.status_code == 200
    rules = {r["platform"]: r for r in resp.json()["rules"]}
    assert rules["tiktok"]["max_per_day"] == 5
    assert rules["tiktok"]["overridden"] is False
    assert rules["linkedin"]["hashtag_max"] == 0


@pytest.mark.asyncio
async def test_rules_save_override_and_reset(
    zernio_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    # Override tiktok cap.
    resp = await client.post(
        f"{_BASE}/autopublish/rules/save",
        json={"platform": "tiktok", "max_per_day": 2, "min_gap_minutes": 300},
        headers=auth(alice["token"]),
    )
    assert resp.status_code == 200
    eff = await PlatformPacingRulesRepo(db).effective_rules(alice["id"])
    assert eff["tiktok"].max_per_day == 2

    # It shows as overridden.
    rules = {r["platform"]: r for r in (
        await client.get(f"{_BASE}/autopublish/rules.json", headers=auth(alice["token"]))
    ).json()["rules"]}
    assert rules["tiktok"]["overridden"] is True

    # Reset → back to default.
    resp = await client.post(
        f"{_BASE}/autopublish/rules/save",
        json={"platform": "tiktok", "reset": True},
        headers=auth(alice["token"]),
    )
    assert resp.status_code == 200
    eff2 = await PlatformPacingRulesRepo(db).effective_rules(alice["id"])
    assert eff2["tiktok"].max_per_day == 5  # default restored


@pytest.mark.asyncio
async def test_rules_save_x_alias(
    zernio_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    resp = await client.post(
        f"{_BASE}/autopublish/rules/save",
        json={"platform": "x", "max_per_day": 9},
        headers=auth(alice["token"]),
    )
    assert resp.status_code == 200
    eff = await PlatformPacingRulesRepo(db).effective_rules(alice["id"])
    assert eff["twitter"].max_per_day == 9  # x normalized to twitter


@pytest.mark.asyncio
async def test_clip_growth_json_404_when_unscored(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    resp = await client.get(
        f"{_BASE}/clips/clp_missing/growth.json", headers=auth(alice["token"])
    )
    assert resp.status_code == 404
    assert resp.json()["error"] == "not_scored"
