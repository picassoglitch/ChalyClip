"""Analytics service: snapshot job + internal read fallback (phase 7)."""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
import respx

from chalybclip.db import (
    Database,
    TenantsRepo,
    ZernioPublishSnapshotsRepo,
    apply_migrations,
)
from chalybclip.integrations.zernio.client import ZernioClient
from chalybclip.publish.analytics_service import (
    internal_analytics,
    performance_for_tenant,
    snapshot_tenant,
)

_ZBASE = "https://zernio.com/api/v1"
_NOW = _dt.datetime(2026, 6, 10, 12, 0, tzinfo=_dt.UTC)


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "an.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


@pytest_asyncio.fixture
async def alice(db: Database) -> str:
    t = await TenantsRepo(db).create(name="Alice")
    await TenantsRepo(db).set_zernio_profile(
        t.id, profile_id="prof_alice", profile_name="Alice",
    )
    return t.id


def _client(http: httpx.AsyncClient) -> ZernioClient:
    return ZernioClient(api_key="sk_test", http=http)


def _analytics_body() -> dict:
    return {
        "overview": {"totalPosts": 2},
        "posts": [
            {"_id": "p1", "content": "Clip A",
             "analytics": {"likes": 10, "views": 100},  # no comments/shares
             "platforms": [{"platform": "tiktok",
                            "analytics": {"likes": 10, "views": 100}}]},
            {"_id": "p2", "content": "Clip B",
             "analytics": {"likes": 5, "comments": 2},
             "platforms": []},
        ],
    }


# ---- performance_for_tenant ----


@pytest.mark.asyncio
async def test_performance_normalizes_and_totals(
    db: Database, alice: str
) -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.get(f"{_ZBASE}/analytics").mock(
                return_value=httpx.Response(200, json=_analytics_body())
            )
            view = await performance_for_tenant(
                db, alice, days=7, client=_client(http), now=_NOW,
            )
    assert len(view.rows) == 2
    assert view.totals["likes"] == 15
    assert view.totals["views"] == 100   # only p1 had it → sum of available
    assert view.totals["shares"] is None  # nobody had it → "—"
    # 7-day window passed through.
    assert route.calls.last.request.url.params.get("fromDate") == "2026-06-03"


@pytest.mark.asyncio
async def test_performance_empty_without_profile(db: Database) -> None:
    t = await TenantsRepo(db).create(name="NoProfile")
    async with httpx.AsyncClient() as http:
        view = await performance_for_tenant(db, t.id, client=_client(http))
    assert view.rows == []
    assert all(v is None for v in view.totals.values())


# ---- snapshot job ----


@pytest.mark.asyncio
async def test_snapshot_persists_per_post_and_is_idempotent(
    db: Database, alice: str
) -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.get(f"{_ZBASE}/analytics").mock(
                return_value=httpx.Response(200, json=_analytics_body())
            )
            n1 = await snapshot_tenant(db, alice, client=_client(http), now=_NOW)
            # Same UTC day again → refresh, not duplicate.
            n2 = await snapshot_tenant(db, alice, client=_client(http), now=_NOW)
    assert n1 == 2
    assert n2 == 2
    assert await ZernioPublishSnapshotsRepo(db).count_for_tenant(alice) == 2
    snaps = await ZernioPublishSnapshotsRepo(db).latest_for_tenant(alice)
    by_post = {s["post_id"]: json.loads(s["metrics_json"]) for s in snaps}
    assert by_post["p1"]["views"] == 100
    assert by_post["p1"]["shares"] is None  # no fake zero in the snapshot


# ---- internal_analytics: live + snapshot fallback ----


@pytest.mark.asyncio
async def test_internal_analytics_prefers_live(db: Database, alice: str) -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.get(f"{_ZBASE}/analytics").mock(
                return_value=httpx.Response(200, json=_analytics_body())
            )
            out = await internal_analytics(
                db, alice, days=30, client=_client(http), now=_NOW,
            )
    assert out["source"] == "live"
    assert len(out["posts"]) == 2
    assert out["totals"]["likes"] == 15


@pytest.mark.asyncio
async def test_internal_analytics_falls_back_to_snapshots(
    db: Database, alice: str
) -> None:
    # Seed a snapshot, then make live return empty → fallback serves it.
    await ZernioPublishSnapshotsRepo(db).upsert(
        alice, post_id="p9", day="2026-06-09",
        metrics_json=json.dumps({"likes": 3, "views": None}),
        platforms_json=json.dumps([]),
    )
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.get(f"{_ZBASE}/analytics").mock(
                return_value=httpx.Response(200, json={"posts": []})
            )
            out = await internal_analytics(
                db, alice, days=30, client=_client(http), now=_NOW,
            )
    assert out["source"] == "snapshot"
    assert out["posts"][0]["post_id"] == "p9"
    assert out["totals"]["likes"] == 3
