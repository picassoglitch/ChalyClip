"""calendar.json route — merges hub/scheduled/external (phase 8)."""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
import pytest_asyncio
import respx

from chalybclip.db import (
    Database,
    TenantsRepo,
    ZernioCalendarRepo,
    ZernioPublishesRepo,
)
from chalybclip.integrations.chalyb.service import sync_tenant_tier
from chalybclip.settings import get_settings

from .conftest import auth

_ZBASE = "https://zernio.com/api/v1"


@pytest.fixture
def zernio_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("CHALYBCLIP_ZERNIO_API_KEY", "sk_test_cal")
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
async def test_calendar_merges_three_sources(
    zernio_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    # Source 1+2: a hub publish + a scheduled one.
    await ZernioPublishesRepo(db).record(
        post_id="post_hub", tenant_id=alice["id"], clip_id="clp_1",
        platforms=["tiktok"], content="Publicado por el hub", status=None,
    )
    await ZernioPublishesRepo(db).record(
        post_id="post_sched", tenant_id=alice["id"], clip_id="clp_2",
        platforms=["youtube"], content="Programado", status="scheduled",
    )
    # Source 3: an external post on Alice's connected account.
    await ZernioCalendarRepo(db).upsert(
        account_id="acct_tiktok", post_id="ext1", platform="tiktok",
        content="Posteado nativo", url="https://tiktok.com/ext1",
        thumbnail_url=None, media_type="video", published_at="2026-06-05T10:00:00Z",
    )

    with respx.mock() as mock:
        # The route resolves external posts via the tenant's accounts.
        mock.get(f"{_ZBASE}/accounts").mock(
            return_value=httpx.Response(
                200,
                json={"accounts": [
                    {"platform": "tiktok", "_id": "acct_tiktok",
                     "profileId": "prof_alice"}
                ]},
            )
        )
        resp = await client.get(
            "/dashboard/publish/zernio/calendar.json", headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    entries = resp.json()["entries"]
    by_source: dict[str, list] = {}
    for e in entries:
        by_source.setdefault(e["source"], []).append(e)
    assert {"hub", "scheduled", "external"} <= set(by_source)
    assert by_source["external"][0]["post_id"] == "ext1"
    assert by_source["external"][0]["url"] == "https://tiktok.com/ext1"
    assert by_source["scheduled"][0]["post_id"] == "post_sched"


@pytest.mark.asyncio
async def test_calendar_excludes_drafts(
    zernio_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    await ZernioPublishesRepo(db).record(
        post_id="post_draft", tenant_id=alice["id"], clip_id="clp_d",
        platforms=["tiktok"], content="Borrador", status="draft",
    )
    with respx.mock() as mock:
        mock.get(f"{_ZBASE}/accounts").mock(
            return_value=httpx.Response(200, json={"accounts": []})
        )
        resp = await client.get(
            "/dashboard/publish/zernio/calendar.json", headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    assert all(e["post_id"] != "post_draft" for e in resp.json()["entries"])


@pytest.mark.asyncio
async def test_calendar_external_isolation_by_account(
    zernio_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    # An external post on an account Alice does NOT own must not show.
    await ZernioCalendarRepo(db).upsert(
        account_id="acct_someone_else", post_id="ext_x", platform="tiktok",
        content="No es de Alice", url=None, thumbnail_url=None,
        media_type=None, published_at="2026-06-05T10:00:00Z",
    )
    with respx.mock() as mock:
        mock.get(f"{_ZBASE}/accounts").mock(
            return_value=httpx.Response(
                200,
                json={"accounts": [
                    {"platform": "tiktok", "_id": "acct_alice_only",
                     "profileId": "prof_alice"}
                ]},
            )
        )
        resp = await client.get(
            "/dashboard/publish/zernio/calendar.json", headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    assert all(e.get("post_id") != "ext_x" for e in resp.json()["entries"])


@pytest.mark.asyncio
async def test_calendar_date_range_filter(
    zernio_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    await ZernioCalendarRepo(db).upsert(
        account_id="acct_tiktok", post_id="june", platform="tiktok",
        content="junio", url=None, thumbnail_url=None, media_type=None,
        published_at="2026-06-15T10:00:00Z",
    )
    await ZernioCalendarRepo(db).upsert(
        account_id="acct_tiktok", post_id="july", platform="tiktok",
        content="julio", url=None, thumbnail_url=None, media_type=None,
        published_at="2026-07-15T10:00:00Z",
    )
    with respx.mock() as mock:
        mock.get(f"{_ZBASE}/accounts").mock(
            return_value=httpx.Response(
                200,
                json={"accounts": [
                    {"platform": "tiktok", "_id": "acct_tiktok",
                     "profileId": "prof_alice"}
                ]},
            )
        )
        resp = await client.get(
            "/dashboard/publish/zernio/calendar.json"
            "?date_from=2026-06-01&date_to=2026-06-30",
            headers=auth(alice["token"]),
        )
    posts = {e["post_id"] for e in resp.json()["entries"]}
    assert "june" in posts
    assert "july" not in posts


@pytest.mark.asyncio
async def test_calendar_window_filters_hub_rows_in_sql(
    zernio_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    # Hub rows are windowed by their PUBLISH day in SQL — the old
    # newest-200 slice + in-Python filter dropped older-created rows,
    # so weeks of scheduled posts vanished from the month view.
    await ZernioPublishesRepo(db).record(
        post_id="post_june", tenant_id=alice["id"], clip_id="clp_j1",
        platforms=["tiktok"], content="junio", status="scheduled",
        scheduled_for="2026-06-20T18:00:00Z",
    )
    await ZernioPublishesRepo(db).record(
        post_id="post_july", tenant_id=alice["id"], clip_id="clp_j2",
        platforms=["tiktok"], content="julio", status="scheduled",
        scheduled_for="2026-07-20T18:00:00Z",
    )
    with respx.mock() as mock:
        mock.get(f"{_ZBASE}/accounts").mock(
            return_value=httpx.Response(200, json={"accounts": []})
        )
        resp = await client.get(
            "/dashboard/publish/zernio/calendar.json"
            "?date_from=2026-06-01&date_to=2026-06-30",
            headers=auth(alice["token"]),
        )
    posts = {e["post_id"] for e in resp.json()["entries"]}
    assert "post_june" in posts
    assert "post_july" not in posts


@pytest.mark.asyncio
async def test_calendar_keeps_published_post_on_its_scheduled_day(
    zernio_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    # Once a scheduled post publishes (status flips via webhook), the day
    # it went out doesn't change. Regression: bucketing keyed on
    # status == "scheduled" snapped published posts back onto the
    # batch-creation day, piling a run's worth of dots on the day the
    # operator clicked Auto-programar and leaving the real days empty.
    repo = ZernioPublishesRepo(db)
    await repo.record(
        post_id="post_done", tenant_id=alice["id"], clip_id="clp_done",
        platforms=["tiktok"], content="Ya salió", status="scheduled",
        scheduled_for="2026-06-10T20:00:00Z",
    )
    await repo.set_status("post_done", status="published")
    with respx.mock() as mock:
        mock.get(f"{_ZBASE}/accounts").mock(
            return_value=httpx.Response(200, json={"accounts": []})
        )
        resp = await client.get(
            "/dashboard/publish/zernio/calendar.json", headers=auth(alice["token"]),
        )
    entry = next(
        e for e in resp.json()["entries"] if e["post_id"] == "post_done"
    )
    assert entry["date"][:10] == "2026-06-10"
    assert entry["source"] == "hub"  # published → no longer "scheduled"


@pytest.mark.asyncio
async def test_calendar_buckets_scheduled_by_publish_date(
    zernio_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    # A scheduled post must land on the day it PUBLISHES (scheduled_for),
    # not the day it was queued (created_at). Regression: the calendar used
    # to bucket every scheduled clip onto the batch-creation day.
    await ZernioPublishesRepo(db).record(
        post_id="post_future", tenant_id=alice["id"], clip_id="clp_f",
        platforms=["youtube"], content="Sale el 23", status="scheduled",
        scheduled_for="2026-06-23T22:00:00Z",
    )
    with respx.mock() as mock:
        mock.get(f"{_ZBASE}/accounts").mock(
            return_value=httpx.Response(200, json={"accounts": []})
        )
        resp = await client.get(
            "/dashboard/publish/zernio/calendar.json", headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    entry = next(
        e for e in resp.json()["entries"] if e["post_id"] == "post_future"
    )
    assert entry["date"][:10] == "2026-06-23"
