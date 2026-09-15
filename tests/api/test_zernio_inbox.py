"""Inbox routes: comments + DMs feed/reply/like/hide/archive (phase 9)."""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest
import pytest_asyncio
import respx

from chalybclip.db import Database, TenantsRepo, ZernioInboxRepo
from chalybclip.integrations.chalyb.service import sync_tenant_tier
from chalybclip.settings import get_settings

from .conftest import auth

_ZBASE = "https://zernio.com/api/v1"
_ACC = "acct_ig_1"


@pytest.fixture
def zernio_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("CHALYBCLIP_ZERNIO_API_KEY", "sk_test_inbox")
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


def _mock_accounts(mock: respx.Router, *ids: str) -> None:
    mock.get(f"{_ZBASE}/accounts").mock(
        return_value=httpx.Response(
            200,
            json={"accounts": [
                {"platform": "instagram", "_id": i, "profileId": "prof_alice"}
                for i in ids
            ]},
        )
    )


async def _seed_comment(db: Database, *, platform: str = "instagram") -> None:
    await ZernioInboxRepo(db).upsert_comment(
        account_id=_ACC, comment_id="c1", post_id=None, platform_post_id="pp1",
        platform=platform, text="Buen clip", author_id="u1",
        author_name="Fan", author_username="fan1", is_reply=False,
        parent_id=None, created_at="2026-06-05T10:00:00Z",
    )


# ---- comments feed + capability flag ----


@pytest.mark.asyncio
async def test_comments_json_resolves_by_account_and_flags_hide(
    zernio_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    await _seed_comment(db, platform="instagram")  # hide-capable
    with respx.mock() as mock:
        _mock_accounts(mock, _ACC)
        resp = await client.get(
            "/dashboard/publish/zernio/inbox/comments.json",
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    rows = resp.json()["comments"]
    assert rows[0]["comment_id"] == "c1"
    assert rows[0]["can_hide"] is True  # instagram supports hide


@pytest.mark.asyncio
async def test_comments_isolation_account_not_owned(
    zernio_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    await _seed_comment(db)
    with respx.mock() as mock:
        # Alice owns a DIFFERENT account → the seeded comment is hidden.
        _mock_accounts(mock, "acct_other")
        resp = await client.get(
            "/dashboard/publish/zernio/inbox/comments.json",
            headers=auth(alice["token"]),
        )
    assert resp.json()["comments"] == []


# ---- reply / like / hide ----


@pytest.mark.asyncio
async def test_comment_reply_posts_to_zernio(
    zernio_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock, _ACC)
        route = mock.post(f"{_ZBASE}/inbox/comments/pp1").mock(
            return_value=httpx.Response(200, json={"success": True})
        )
        resp = await client.post(
            "/dashboard/publish/zernio/inbox/comments/reply",
            json={"account_id": _ACC, "post_id": "pp1", "comment_id": "c1",
                  "message": "¡Gracias!"},
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    payload = json.loads(route.calls.last.request.content.decode())
    assert payload == {"accountId": _ACC, "message": "¡Gracias!", "commentId": "c1"}


@pytest.mark.asyncio
async def test_comment_reply_rejects_unowned_account(
    zernio_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock, "acct_alice_real")
        resp = await client.post(
            "/dashboard/publish/zernio/inbox/comments/reply",
            json={"account_id": "acct_someone_else", "post_id": "pp1",
                  "message": "hola"},
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_hide_gated_on_unsupported_platform(
    zernio_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock, _ACC)
        resp = await client.post(
            "/dashboard/publish/zernio/inbox/comments/hide",
            json={"account_id": _ACC, "post_id": "pp1", "comment_id": "c1",
                  "platform": "bluesky"},  # bluesky can't hide
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 409
    assert "soportado" in resp.json()["error"].lower()


@pytest.mark.asyncio
async def test_hide_supported_marks_local_hidden(
    zernio_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    await _seed_comment(db, platform="instagram")
    with respx.mock() as mock:
        _mock_accounts(mock, _ACC)
        mock.post(f"{_ZBASE}/inbox/comments/pp1/c1/hide").mock(
            return_value=httpx.Response(200, json={"success": True})
        )
        resp = await client.post(
            "/dashboard/publish/zernio/inbox/comments/hide",
            json={"account_id": _ACC, "post_id": "pp1", "comment_id": "c1",
                  "platform": "instagram"},
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    rows = await ZernioInboxRepo(db).list_comments([_ACC])
    assert rows[0]["status"] == "hidden"


# ---- DMs ----


@pytest.mark.asyncio
async def test_conversations_and_messages_feed(
    zernio_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    inbox = ZernioInboxRepo(db)
    await inbox.upsert_conversation(
        account_id=_ACC, conversation_id="conv1", platform="instagram",
        participant_id="p1", participant_name="Cliente",
        participant_username="cli", status="active",
        last_message_at="2026-06-05T10:00:00Z",
    )
    await inbox.upsert_message(
        account_id=_ACC, message_id="m1", conversation_id="conv1",
        platform="instagram", direction="incoming", text="hola",
        sent_at="2026-06-05T10:00:00Z", is_read=False,
    )
    with respx.mock() as mock:
        _mock_accounts(mock, _ACC)
        convs = await client.get(
            "/dashboard/publish/zernio/inbox/conversations.json",
            headers=auth(alice["token"]),
        )
        msgs = await client.get(
            "/dashboard/publish/zernio/inbox/messages.json"
            "?conversation_id=conv1",
            headers=auth(alice["token"]),
        )
    assert convs.json()["conversations"][0]["conversation_id"] == "conv1"
    assert msgs.json()["messages"][0]["text"] == "hola"


@pytest.mark.asyncio
async def test_dm_reply_text_sends(
    zernio_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock, _ACC)
        route = mock.post(f"{_ZBASE}/inbox/conversations/conv1/messages").mock(
            return_value=httpx.Response(200, json={"success": True})
        )
        resp = await client.post(
            "/dashboard/publish/zernio/inbox/conversations/reply",
            json={"account_id": _ACC, "conversation_id": "conv1",
                  "message": "¡Hola!", "platform": "instagram"},
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    payload = json.loads(route.calls.last.request.content.decode())
    assert payload == {"accountId": _ACC, "message": "¡Hola!"}


@pytest.mark.asyncio
async def test_dm_attachment_gated_on_bluesky(
    zernio_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    with respx.mock() as mock:
        _mock_accounts(mock, _ACC)
        resp = await client.post(
            "/dashboard/publish/zernio/inbox/conversations/reply",
            json={"account_id": _ACC, "conversation_id": "conv1",
                  "message": "mira esto", "attachment_url": "https://x/y.jpg",
                  "platform": "bluesky"},  # text-only DMs
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 409
    assert "texto" in resp.json()["error"].lower()


@pytest.mark.asyncio
async def test_dm_archive_updates_local_and_remote(
    zernio_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    inbox = ZernioInboxRepo(db)
    await inbox.upsert_conversation(
        account_id=_ACC, conversation_id="conv1", platform="instagram",
        participant_id="p1", participant_name="C", participant_username="c",
        status="active", last_message_at="2026-06-05T10:00:00Z",
    )
    with respx.mock() as mock:
        _mock_accounts(mock, _ACC)
        mock.put(f"{_ZBASE}/inbox/conversations/conv1").mock(
            return_value=httpx.Response(200, json={"success": True})
        )
        resp = await client.post(
            "/dashboard/publish/zernio/inbox/conversations/archive",
            json={"account_id": _ACC, "conversation_id": "conv1"},
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    convs = await inbox.list_conversations([_ACC])
    assert convs[0]["status"] == "archived"
