"""Job detail page + single-post status poll: graceful degradation when
Zernio's GET /posts/{id} can't serve the post.

Regression for the "Couldn't load status from Zernio: ... returned HTTP
404 → page spins on Cargando… forever" bug. A 404 must either fall back
to our webhook-fed local record (zernio_publishes) or settle into a
terminal UNAVAILABLE state so the poller stops.
"""

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
    monkeypatch.setenv("CHALYBCLIP_ZERNIO_API_KEY", "sk_test_job")
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


# ---- status.json poll ----


@pytest.mark.asyncio
async def test_status_live_ok_returns_zernio_payload(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{_ZBASE}/posts/p_live").mock(
            return_value=httpx.Response(
                200,
                json={
                    "post": {
                        "_id": "p_live",
                        "status": "published",
                        "platforms": [{"platform": "tiktok", "status": "published"}],
                    }
                },
            )
        )
        resp = await client.get(
            "/dashboard/publish/zernio/status/p_live.json",
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "PUBLISHED"
    assert body["platforms"][0]["platform"] == "tiktok"


@pytest.mark.asyncio
async def test_status_404_falls_back_to_local_record(
    zernio_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    # A webhook already wrote the real result locally; Zernio now 404s.
    await ZernioPublishesRepo(db).record(
        post_id="p_gone", tenant_id=alice["id"], clip_id="clp_x",
        platforms=["tiktok"], content="x", status="failed",
    )
    await ZernioPublishesRepo(db).set_status(
        "p_gone", status="failed",
        platforms_json='[{"platform": "tiktok", "status": "failed"}]',
    )
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{_ZBASE}/posts/p_gone").mock(
            return_value=httpx.Response(404, json={"error": "not found"})
        )
        resp = await client.get(
            "/dashboard/publish/zernio/status/p_gone.json",
            headers=auth(alice["token"]),
        )
    # 200 (not 502) so the poller settles; status comes from our record.
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "FAILED"
    assert body["platforms"][0]["platform"] == "tiktok"


@pytest.mark.asyncio
async def test_status_404_no_local_record_settles_unavailable(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{_ZBASE}/posts/p_unknown").mock(
            return_value=httpx.Response(404, json={"error": "not found"})
        )
        resp = await client.get(
            "/dashboard/publish/zernio/status/p_unknown.json",
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200  # terminal status, not a 502 retry-loop
    assert resp.json()["status"] == "UNAVAILABLE"


@pytest.mark.asyncio
async def test_status_transient_error_stays_502_for_retry(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{_ZBASE}/posts/p_5xx").mock(
            return_value=httpx.Response(500, text="boom")
        )
        resp = await client.get(
            "/dashboard/publish/zernio/status/p_5xx.json",
            headers=auth(alice["token"]),
        )
    # No local row + non-404 → keep 502 so the poller retries silently.
    assert resp.status_code == 502
    assert resp.json()["status"] == "ERROR"


@pytest.mark.asyncio
async def test_status_404_local_row_without_status_settles_unavailable(
    zernio_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    # A "publish now" row seeds status=None (no post.* webhook yet). When
    # Zernio also 404s the post, the old code returned "UNKNOWN" — NOT
    # terminal — so the page polled /status every 3s forever. It must
    # settle on terminal UNAVAILABLE instead.
    await ZernioPublishesRepo(db).record(
        post_id="p_nostatus", tenant_id=alice["id"], clip_id="clp_n",
        platforms=["tiktok"], content="x",  # status defaults to None
    )
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{_ZBASE}/posts/p_nostatus").mock(
            return_value=httpx.Response(404, json={"error": "not found"})
        )
        resp = await client.get(
            "/dashboard/publish/zernio/status/p_nostatus.json",
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "UNAVAILABLE"  # terminal — poller stops


@pytest.mark.asyncio
async def test_status_404_ignores_other_tenants_record(
    zernio_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
    tenants: dict[str, dict[str, str]],
) -> None:
    # Bob owns the local row; Alice must NOT see it on her 404 fallback.
    await ZernioPublishesRepo(db).record(
        post_id="p_bob", tenant_id=tenants["bob"]["id"], clip_id="clp_b",
        platforms=["tiktok"], content="x", status="published",
    )
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{_ZBASE}/posts/p_bob").mock(
            return_value=httpx.Response(404, json={"error": "not found"})
        )
        resp = await client.get(
            "/dashboard/publish/zernio/status/p_bob.json",
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    # Falls through to UNAVAILABLE — Bob's published result is not leaked.
    assert resp.json()["status"] == "UNAVAILABLE"


# ---- job detail page ----


@pytest.mark.asyncio
async def test_job_page_404_renders_unavailable_not_raw_error(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{_ZBASE}/posts/p_x").mock(
            return_value=httpx.Response(404, json={"error": "not found"})
        )
        resp = await client.get(
            "/dashboard/publish/zernio/job/p_x", headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    html = resp.text
    assert 'data-s="UNAVAILABLE"' in html
    assert "no longer available" in html
    # The raw transport error must not leak onto the page.
    assert "returned HTTP 404" not in html


@pytest.mark.asyncio
async def test_job_page_404_uses_local_record_when_present(
    zernio_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    await ZernioPublishesRepo(db).record(
        post_id="p_local", tenant_id=alice["id"], clip_id="clp_x",
        platforms=["tiktok"], content="x", status="published",
    )
    await ZernioPublishesRepo(db).set_status(
        "p_local", status="published",
        platforms_json='[{"platform": "tiktok", "status": "published"}]',
    )
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{_ZBASE}/posts/p_local").mock(
            return_value=httpx.Response(404, json={"error": "not found"})
        )
        resp = await client.get(
            "/dashboard/publish/zernio/job/p_local", headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    assert 'data-s="PUBLISHED"' in resp.text
