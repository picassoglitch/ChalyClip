"""Failed list + retry routes (Hub phase 6)."""

from __future__ import annotations

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
    monkeypatch.setenv("CHALYBCLIP_ZERNIO_API_KEY", "sk_test_retry")
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


def _failed_post(post_id: str, category: str = "platform_error") -> dict:
    return {
        "_id": post_id,
        "content": "Falló esto",
        "status": "failed",
        "platforms": [
            {"platform": "tiktok", "status": "failed",
             "errorCategory": category, "errorMessage": "boom"},
        ],
    }


# ---- failed.json ----


@pytest.mark.asyncio
async def test_failed_json_lists_with_hints(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock(assert_all_called=True) as mock:
        route = mock.get(f"{_ZBASE}/posts").mock(
            return_value=httpx.Response(
                200, json={"posts": [_failed_post("pf1", "auth_expired")]}
            )
        )
        resp = await client.get(
            "/dashboard/publish/zernio/failed.json", headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["failed"][0]["post_id"] == "pf1"
    plat = body["failed"][0]["platforms"][0]
    assert plat["platform"] == "tiktok"
    assert plat["category"] == "auth_expired"
    assert "reconé" in plat["hint"].lower()
    assert plat["transient"] is False
    # Uses GET /posts?status=failed (no /posts/failed endpoint).
    assert route.calls.last.request.url.params.get("status") == "failed"


# ---- retry one ----


@pytest.mark.asyncio
async def test_retry_one_calls_zernio_and_flips_local_status(
    zernio_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    await ZernioPublishesRepo(db).record(
        post_id="pf2", tenant_id=alice["id"], clip_id="clp_x",
        platforms=["tiktok"], content="x", status="failed",
    )
    with respx.mock(assert_all_called=True) as mock:
        mock.post(f"{_ZBASE}/posts/pf2/retry").mock(
            return_value=httpx.Response(200, json={"post": {"_id": "pf2"}})
        )
        resp = await client.post(
            "/dashboard/publish/zernio/retry/pf2", headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    row = await ZernioPublishesRepo(db).get_by_post_id("pf2")
    assert row is not None and row.status == "publishing"


@pytest.mark.asyncio
async def test_retry_one_429_maps_to_rate_limit_message(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        mock.post(f"{_ZBASE}/posts/pf3/retry").mock(
            return_value=httpx.Response(429, json={"error": "rate limited"})
        )
        resp = await client.post(
            "/dashboard/publish/zernio/retry/pf3", headers=auth(alice["token"]),
        )
    assert resp.status_code == 429
    assert "frecuencia" in resp.json()["error"].lower()


# ---- retry all ----


@pytest.mark.asyncio
async def test_retry_all_retries_each_failed_post(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        mock.get(f"{_ZBASE}/posts").mock(
            return_value=httpx.Response(
                200,
                json={"posts": [_failed_post("pa1"), _failed_post("pa2")]},
            )
        )
        r1 = mock.post(f"{_ZBASE}/posts/pa1/retry").mock(
            return_value=httpx.Response(200, json={"post": {"_id": "pa1"}})
        )
        r2 = mock.post(f"{_ZBASE}/posts/pa2/retry").mock(
            return_value=httpx.Response(200, json={"post": {"_id": "pa2"}})
        )
        resp = await client.post(
            "/dashboard/publish/zernio/retry-all", headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["retried"] == 2
    assert r1.called and r2.called


@pytest.mark.asyncio
async def test_retry_all_isolates_per_post_failures(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        mock.get(f"{_ZBASE}/posts").mock(
            return_value=httpx.Response(
                200,
                json={"posts": [_failed_post("pb1"), _failed_post("pb2")]},
            )
        )
        mock.post(f"{_ZBASE}/posts/pb1/retry").mock(
            return_value=httpx.Response(200, json={"post": {"_id": "pb1"}})
        )
        mock.post(f"{_ZBASE}/posts/pb2/retry").mock(
            return_value=httpx.Response(500, text="nope")
        )
        resp = await client.post(
            "/dashboard/publish/zernio/retry-all", headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    body = resp.json()
    # One ok, one failed — the batch didn't abort.
    assert body["retried"] == 1
    outcomes = {r["post_id"]: r["ok"] for r in body["results"]}
    assert outcomes == {"pb1": True, "pb2": False}
