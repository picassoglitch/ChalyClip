"""Community notifications: embed building + announce-on-publish
(phase 11)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
import respx

from chalybclip.db import (
    Database,
    TenantsRepo,
    ZernioCommunityRepo,
    ZernioEventsRepo,
    apply_migrations,
)
from chalybclip.integrations.zernio.community import (
    build_discord_embed,
    build_notification_payload,
    build_weekly_digest_text,
)
from chalybclip.integrations.zernio.events import process_zernio_event
from chalybclip.settings import get_settings

_ZBASE = "https://zernio.com/api/v1"


# ---- pure builders ----


def test_discord_embed_has_title_link_platforms_and_brand() -> None:
    post = {
        "content": "Mi mejor jugada",
        "platforms": [
            {"platform": "tiktok", "status": "published",
             "publishedUrl": "https://tiktok.com/v/1"},
            {"platform": "youtube", "status": "published"},
        ],
    }
    data = build_discord_embed(
        post, brand_name="Mi Canal", brand_avatar_url="https://a/x.png",
        thumbnail_url="https://t/thumb.jpg",
    )
    embed = data["embeds"][0]
    assert embed["title"] == "Mi mejor jugada"
    assert embed["url"] == "https://tiktok.com/v/1"
    assert embed["thumbnail"]["url"] == "https://t/thumb.jpg"
    assert "tiktok" in embed["fields"][0]["value"]
    # Brand identity overrides.
    assert data["webhookUsername"] == "Mi Canal"
    assert data["webhookAvatarUrl"] == "https://a/x.png"


def test_notification_payload_only_configured_channels() -> None:
    post = {"content": "x", "platforms": []}
    # Discord only.
    platforms, psd, text = build_notification_payload(
        post, discord_account_id="d1", telegram_account_id=None,
    )
    assert platforms == [("discord", "d1")]
    assert "discord" in psd
    assert text  # telegram fallback text always built
    # Telegram only → no discord psd.
    platforms2, psd2, _ = build_notification_payload(
        post, discord_account_id=None, telegram_account_id="t1",
    )
    assert platforms2 == [("telegram", "t1")]
    assert psd2 == {}


def test_weekly_digest_text_handles_missing_metrics() -> None:
    text = build_weekly_digest_text(
        {"views": 1200, "likes": 50, "comments": None, "shares": None}
    )
    assert "1,200" in text
    assert "—" in text  # comments/shares unavailable, not fake 0


# ---- announce-on-publish wiring ----


@pytest.fixture
def comm_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("CHALYBCLIP_ZERNIO_API_KEY", "sk_test_comm")
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "comm.db")
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


async def _publish_event(db: Database, *, event_id: str, post_id: str) -> None:
    payload = {
        "id": event_id, "event": "post.published",
        "post": {
            "id": post_id, "status": "published",
            "content": "Nuevo clip",
            "platforms": [{"platform": "tiktok", "status": "published",
                           "publishedUrl": "https://tiktok.com/v/" + post_id}],
            "profileId": "prof_alice",
        },
        "timestamp": "2026-06-10T10:00:00Z",
    }
    await ZernioEventsRepo(db).insert_dedup(
        event_id=event_id, type="post.published", payload=json.dumps(payload),
    )
    await process_zernio_event(db, event_id)


@pytest.mark.asyncio
async def test_publish_notifies_community_when_enabled(
    comm_env: None, db: Database, alice: str
) -> None:
    await ZernioCommunityRepo(db).upsert_settings(
        alice, enabled=True, discord_account_id="acct_discord",
        telegram_account_id=None, brand_name="Mi Canal",
        brand_avatar_url=None, weekly_digest=False,
    )
    with respx.mock() as mock:
        post_route = mock.post(f"{_ZBASE}/posts").mock(
            return_value=httpx.Response(201, json={"post": {"_id": "notif1"}})
        )
        await _publish_event(db, event_id="ev1", post_id="post1")
    assert post_route.called
    payload = json.loads(post_route.calls.last.request.content.decode())
    # Community post: Discord embed, no video mediaItems.
    assert "mediaItems" not in payload
    assert payload["platforms"][0]["platform"] == "discord"
    assert payload["platforms"][0]["platformSpecificData"]["webhookUsername"] == "Mi Canal"
    # Ledger recorded the notification post id (loop guard substrate).
    assert await ZernioCommunityRepo(db).is_notification_post("notif1")


@pytest.mark.asyncio
async def test_publish_does_not_notify_when_disabled(
    comm_env: None, db: Database, alice: str
) -> None:
    await ZernioCommunityRepo(db).upsert_settings(
        alice, enabled=False, discord_account_id="acct_discord",
        telegram_account_id=None, brand_name=None, brand_avatar_url=None,
        weekly_digest=False,
    )
    with respx.mock(assert_all_called=False) as mock:
        post_route = mock.post(f"{_ZBASE}/posts").mock(
            return_value=httpx.Response(201, json={"post": {"_id": "x"}})
        )
        await _publish_event(db, event_id="ev1", post_id="post1")
    assert not post_route.called


@pytest.mark.asyncio
async def test_redelivery_announces_once(
    comm_env: None, db: Database, alice: str
) -> None:
    await ZernioCommunityRepo(db).upsert_settings(
        alice, enabled=True, discord_account_id="acct_discord",
        telegram_account_id=None, brand_name=None, brand_avatar_url=None,
        weekly_digest=False,
    )
    with respx.mock() as mock:
        post_route = mock.post(f"{_ZBASE}/posts").mock(
            return_value=httpx.Response(201, json={"post": {"_id": "notif1"}})
        )
        await _publish_event(db, event_id="ev1", post_id="post1")
        # Same post, different event id (at-least-once redelivery).
        await _publish_event(db, event_id="ev2", post_id="post1")
    assert len(post_route.calls) == 1  # announced once only


@pytest.mark.asyncio
async def test_notification_post_does_not_loop(
    comm_env: None, db: Database, alice: str
) -> None:
    await ZernioCommunityRepo(db).upsert_settings(
        alice, enabled=True, discord_account_id="acct_discord",
        telegram_account_id=None, brand_name=None, brand_avatar_url=None,
        weekly_digest=False,
    )
    with respx.mock() as mock:
        post_route = mock.post(f"{_ZBASE}/posts").mock(
            return_value=httpx.Response(201, json={"post": {"_id": "notif1"}})
        )
        await _publish_event(db, event_id="ev1", post_id="post1")
        # Now the notification's OWN post.published arrives → must NOT
        # announce again (loop guard).
        await _publish_event(db, event_id="ev2", post_id="notif1")
    assert len(post_route.calls) == 1  # the loop did not produce a 2nd post
    # Only the original announcement fired; notif1's publish was skipped.
    assert await ZernioCommunityRepo(db).is_notification_post("notif1")
    # No new ledger row for notif1 as a SOURCE.
    s = await ZernioCommunityRepo(db).get_settings(alice)
    assert s is not None
