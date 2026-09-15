"""Auto-publish ("Piloto automático") settings routes + engine guard."""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
import pytest_asyncio
import respx

from chalybclip.db import AutopublishSettingsRepo, Database, TenantsRepo
from chalybclip.integrations.chalyb.service import sync_tenant_tier
from chalybclip.settings import get_settings

from .conftest import auth

_ZBASE = "https://zernio.com/api/v1"


@pytest.fixture
def zernio_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("CHALYBCLIP_ZERNIO_API_KEY", "sk_test_ap")
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


def _mock_accounts(mock: respx.Router) -> None:
    mock.get(f"{_ZBASE}/accounts").mock(
        return_value=httpx.Response(
            200,
            json={"accounts": [
                {"platform": "tiktok", "_id": "acct_tt", "profileId": "prof_alice"},
                {"platform": "instagram", "_id": "acct_ig", "profileId": "prof_alice"},
            ]},
        )
    )


@pytest.mark.asyncio
async def test_autopublish_json_defaults_and_platforms(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    """No row yet → safe defaults (off, on_approve); platforms come from
    the tenant's connected Zernio accounts."""
    with respx.mock() as mock:
        _mock_accounts(mock)
        resp = await client.get(
            "/dashboard/publish/zernio/autopublish.json",
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["settings"]["enabled"] is False
    assert body["settings"]["mode"] == "on_approve"
    assert set(body["platforms"]) == {"tiktok", "instagram"}
    assert body["used_today"] == 0


@pytest.mark.asyncio
async def test_autopublish_save_persists(
    zernio_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    resp = await client.post(
        "/dashboard/publish/zernio/autopublish/save",
        json={
            "enabled": True, "mode": "hands_free",
            "targets": ["tiktok", "instagram"], "post_mode": "queue",
            "daily_cap": 5, "score_threshold": 0.75,
        },
        headers=auth(alice["token"]),
    )
    assert resp.status_code == 200
    s = await AutopublishSettingsRepo(db).get(alice["id"])
    assert s is not None
    assert s["enabled"] is True
    assert s["mode"] == "hands_free"
    assert s["targets"] == "tiktok,instagram"
    assert s["post_mode"] == "queue"
    assert s["daily_cap"] == 5
    assert s["score_threshold"] == 0.75


@pytest.mark.asyncio
async def test_autopublish_save_clamps_invalid_mode(
    zernio_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    """An unknown mode/post_mode falls back to the safe default."""
    resp = await client.post(
        "/dashboard/publish/zernio/autopublish/save",
        json={"enabled": True, "mode": "bogus", "post_mode": "bogus", "targets": []},
        headers=auth(alice["token"]),
    )
    assert resp.status_code == 200
    s = await AutopublishSettingsRepo(db).get(alice["id"])
    assert s is not None
    assert s["mode"] == "on_approve"
    assert s["post_mode"] == "queue"


@pytest.mark.asyncio
async def test_engine_no_op_when_disabled(
    db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    """maybe_autopublish_on_approve returns None (and never raises) when
    auto-publish is off — the common case must not touch Zernio."""
    from chalybclip.api.routers.zernio import maybe_autopublish_on_approve

    tid = tenants["alice"]["id"]
    # No settings row at all → disabled by default.
    out = await maybe_autopublish_on_approve(
        request=None,  # unused on the disabled early-return path
        db=db, tenant_id=tid, clip_id="clp_x",
    )
    assert out is None


@pytest.mark.asyncio
async def test_handsfree_sweep_no_op_when_disabled(
    db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    """The hands-free sweep returns 0 and never raises when auto-publish is
    off — the pipeline calls it unconditionally and must not be affected."""
    from chalybclip.api.routers.zernio import autopublish_hands_free_sweep

    n = await autopublish_hands_free_sweep(
        db=db, tenant_id=tenants["alice"]["id"], base_url="https://x.test",
        clip_scores=[("clp_a", 0.9)],
    )
    assert n == 0


@pytest.mark.asyncio
async def test_handsfree_sweep_no_op_in_on_approve_mode(
    db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    """Enabled but mode=on_approve → the hands-free sweep stays out of it."""
    from chalybclip.api.routers.zernio import autopublish_hands_free_sweep

    tid = tenants["alice"]["id"]
    await AutopublishSettingsRepo(db).upsert(
        tid, enabled=True, mode="on_approve", targets="tiktok",
        post_mode="queue", daily_cap=10, score_threshold=0.5,
    )
    n = await autopublish_hands_free_sweep(
        db=db, tenant_id=tid, base_url="https://x.test",
        clip_scores=[("clp_a", 0.9)],
    )
    assert n == 0
