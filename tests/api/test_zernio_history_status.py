"""Publish Center history: a post's status must survive Zernio's feed
missing it.

Regression for the "stuck in queued forever" bug. The dashboard derived
each history row's status purely from Zernio's company-key GET /posts
feed and fell back to "queued" when the feed missed the post — even
though the post.* webhooks had already written the real outcome into
zernio_publishes.status. So a published/failed post Zernio no longer
serves (deleted → GET /posts/{id} 404, beyond page 1, or a different
workspace) read "queued" indefinitely.

Status precedence is now: live feed → local webhook status → "queued".
"""

from __future__ import annotations

import re
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
    monkeypatch.setenv("CHALYBCLIP_ZERNIO_API_KEY", "sk_test_hist")
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


@pytest.mark.asyncio
async def test_history_uses_local_webhook_status_when_feed_misses_post(
    zernio_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    # A real publish whose post.published webhook already landed locally…
    await ZernioPublishesRepo(db).record(
        post_id="p_done", tenant_id=alice["id"], clip_id="clp_done",
        platforms=["tiktok"], content="x",
    )
    await ZernioPublishesRepo(db).set_status("p_done", status="published")

    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"{_ZBASE}/accounts").mock(
            return_value=httpx.Response(200, json={"accounts": []})
        )
        # …but Zernio's company-key feed no longer returns the post.
        mock.get(f"{_ZBASE}/posts").mock(
            return_value=httpx.Response(200, json={"posts": []})
        )
        resp = await client.get(
            "/dashboard/publish/zernio", headers=auth(alice["token"]),
        )

    assert resp.status_code == 200
    html = resp.text
    # The row is present and reads "published" (success chip) — NOT the
    # "queued" fallback (which would render the neutral nc-chip--code).
    assert "/dashboard/publish/zernio/job/p_done" in html
    assert re.search(r'nc-chip--success">\s*published', html)
