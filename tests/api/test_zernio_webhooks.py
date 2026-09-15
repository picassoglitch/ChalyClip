"""Zernio webhook receiver tests (Publish Hub phase 2).

Pins the event backbone end-to-end over ASGI: HMAC verification,
at-least-once dedup on payload.id, fast-ACK + background processing
(publish-status updates, tenant resolution, fan-out to subscriber
webhooks via the existing HMAC dispatcher).

Background tasks run inside the ASGI response cycle, so effects are
visible right after the POST returns.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
import respx

from chalybclip.db import (
    Database,
    EventsRepo,
    TenantsRepo,
    WebhookSubscriptionsRepo,
    ZernioEventsRepo,
    ZernioPublishesRepo,
)
from chalybclip.settings import get_settings
from chalybclip.tenancy import bound_tenant

_SECRET = "zernio_webhook_secret_x"
_PATH = "/api/webhooks/zernio"


@pytest.fixture
def webhook_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("CHALYBCLIP_ZERNIO_WEBHOOK_SECRET", _SECRET)
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


def _sign(body: bytes, secret: str = _SECRET) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


async def _post_event(
    client: httpx.AsyncClient, payload: dict[str, Any], *, secret: str = _SECRET
) -> httpx.Response:
    body = json.dumps(payload).encode()
    return await client.post(
        _PATH,
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Zernio-Signature": _sign(body, secret),
        },
    )


@pytest_asyncio.fixture
async def alice(
    db: Database, tenants: dict[str, dict[str, str]]
) -> dict[str, str]:
    """Alice bound to Zernio profile prof_alice."""
    await TenantsRepo(db).set_zernio_profile(
        tenants["alice"]["id"], profile_id="prof_alice", profile_name="Alice",
    )
    return tenants["alice"]


# ---- transport auth ----


@pytest.mark.asyncio
async def test_missing_secret_config_is_503(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CHALYBCLIP_ZERNIO_WEBHOOK_SECRET", raising=False)
    get_settings.cache_clear()
    try:
        resp = await client.post(_PATH, content=b"{}")
    finally:
        get_settings.cache_clear()
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_bad_signature_is_403(
    webhook_env: None, client: httpx.AsyncClient
) -> None:
    body = b'{"id": "evt_x", "event": "post.published"}'
    resp = await client.post(
        _PATH, content=body, headers={"X-Zernio-Signature": "deadbeef"}
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_missing_signature_is_403(
    webhook_env: None, client: httpx.AsyncClient
) -> None:
    resp = await client.post(_PATH, content=b"{}")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_valid_signature_with_sha256_prefix_accepted(
    webhook_env: None, client: httpx.AsyncClient, db: Database
) -> None:
    body = json.dumps({"id": "evt_pfx", "event": "webhook.test"}).encode()
    resp = await client.post(
        _PATH,
        content=body,
        headers={"X-Zernio-Signature": "sha256=" + _sign(body)},
    )
    assert resp.status_code == 200
    assert (await ZernioEventsRepo(db).get("evt_pfx")) is not None


# ---- body validation ----


@pytest.mark.asyncio
async def test_non_json_body_is_400(
    webhook_env: None, client: httpx.AsyncClient
) -> None:
    body = b"not json at all"
    resp = await client.post(
        _PATH, content=body, headers={"X-Zernio-Signature": _sign(body)}
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_json_non_object_body_is_400(
    webhook_env: None, client: httpx.AsyncClient
) -> None:
    body = b'["an", "array"]'
    resp = await client.post(
        _PATH, content=body, headers={"X-Zernio-Signature": _sign(body)}
    )
    assert resp.status_code == 400


# ---- dedup ----


@pytest.mark.asyncio
async def test_duplicate_event_id_acks_200_and_processes_once(
    webhook_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    payload = {
        "id": "evt_dup",
        "event": "account.connected",
        "account": {
            "accountId": "acct_1",
            "profileId": "prof_alice",
            "platform": "tiktok",
            "username": "alice_tt",
        },
        "timestamp": "2026-06-10T12:00:00Z",
    }
    first = await _post_event(client, payload)
    second = await _post_event(client, payload)
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json().get("duplicate") is True
    # Exactly ONE fan-out event row despite two deliveries.
    with bound_tenant(alice["id"]):
        rows = await EventsRepo(db).list_for_tenant(limit=50)
    zernio_rows = [r for r in rows if r.type == "zernio.account.connected"]
    assert len(zernio_rows) == 1


# ---- processing: post.* feeds zernio_publishes ----


@pytest.mark.asyncio
async def test_post_published_updates_publish_status_and_resolves_tenant(
    webhook_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    await ZernioPublishesRepo(db).record(
        post_id="post_1",
        tenant_id=alice["id"],
        clip_id="clp_1",
        platforms=["tiktok"],
        content="hola",
    )
    payload = {
        "id": "evt_pub",
        "event": "post.published",
        "post": {
            "id": "post_1",
            "content": "hola",
            "status": "published",
            "scheduledFor": "2026-06-10T12:00:00Z",
            "platforms": [
                {
                    "platform": "tiktok",
                    "status": "published",
                    "publishedUrl": "https://tiktok.com/v/1",
                }
            ],
        },
        "timestamp": "2026-06-10T12:00:01Z",
    }
    resp = await _post_event(client, payload)
    assert resp.status_code == 200

    pub = await ZernioPublishesRepo(db).get_by_post_id("post_1")
    assert pub is not None
    assert pub.status == "published"
    assert pub.platforms_json is not None
    assert "tiktok.com/v/1" in pub.platforms_json

    ev = await ZernioEventsRepo(db).get("evt_pub")
    assert ev is not None
    assert ev.processed is True
    assert ev.tenant_id == alice["id"]  # resolved via the post id

    with bound_tenant(alice["id"]):
        rows = await EventsRepo(db).list_for_tenant(limit=10)
    assert any(r.type == "zernio.post.published" for r in rows)


@pytest.mark.asyncio
async def test_post_failed_event_marks_publish_failed(
    webhook_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    await ZernioPublishesRepo(db).record(
        post_id="post_2",
        tenant_id=alice["id"],
        clip_id="clp_2",
        platforms=["youtube"],
        content=None,
    )
    # No post.status in the payload → status falls back to the event name.
    payload = {"id": "evt_fail", "event": "post.failed", "post": {"id": "post_2"}}
    resp = await _post_event(client, payload)
    assert resp.status_code == 200
    pub = await ZernioPublishesRepo(db).get_by_post_id("post_2")
    assert pub is not None
    assert pub.status == "failed"


# ---- processing: every subscribed type ACKs and stores ----


@pytest.mark.parametrize(
    "event_type",
    [
        "post.scheduled",
        "post.published",
        "post.failed",
        "post.partial",
        "post.cancelled",
        "post.external.created",
        "post.external.updated",
        "post.external.deleted",
        "account.connected",
        "account.disconnected",
        "comment.received",
        "message.received",
        "message.sent",
        "conversation.started",  # not subscribable, still handled if sent
    ],
)
@pytest.mark.asyncio
async def test_every_event_type_acks_and_stores(
    webhook_env: None,
    client: httpx.AsyncClient,
    db: Database,
    event_type: str,
) -> None:
    """Minimal (even underspecified) payloads must never 500 — they
    ACK, store and mark processed (unresolved tenants stay NULL)."""
    payload = {"id": f"evt_{event_type}", "event": event_type}
    resp = await _post_event(client, payload)
    assert resp.status_code == 200
    row = await ZernioEventsRepo(db).get(f"evt_{event_type}")
    assert row is not None
    assert row.type == event_type
    assert row.processed is True


@pytest.mark.asyncio
async def test_event_without_id_still_stored_under_generated_id(
    webhook_env: None, client: httpx.AsyncClient, db: Database
) -> None:
    resp = await _post_event(client, {"event": "webhook.test"})
    assert resp.status_code == 200
    pending = await ZernioEventsRepo(db).list_unprocessed(limit=10)
    # Processed in the background — nothing left pending, and the row
    # exists under a generated id (can't dedup without Zernio's id).
    assert pending == []


# ---- fan-out to subscriber webhooks ----


@pytest.mark.asyncio
async def test_fanout_delivers_to_tenant_subscriber_with_hmac(
    webhook_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    with bound_tenant(alice["id"]):
        await WebhookSubscriptionsRepo(db).create(
            url="https://chalybobs.test/hooks/chalybclip",
            types=["zernio.*"],
            secret="sub_secret_1",
        )
    payload = {
        "id": "evt_fan",
        "event": "account.connected",
        "account": {
            "accountId": "acct_9",
            "profileId": "prof_alice",
            "platform": "instagram",
            "username": "alice_ig",
        },
    }
    with respx.mock() as mock:
        route = mock.post("https://chalybobs.test/hooks/chalybclip").mock(
            return_value=httpx.Response(200)
        )
        resp = await _post_event(client, payload)
    assert resp.status_code == 200
    assert route.called
    delivered = route.calls.last.request
    # Our own HMAC, verifiable by the subscriber.
    sig = delivered.headers["X-Chalybclip-Signature"]
    expected = hmac.new(
        b"sub_secret_1", delivered.content, hashlib.sha256
    ).hexdigest()
    assert hmac.compare_digest(sig, expected)
    body = json.loads(delivered.content)
    assert body["type"] == "zernio.account.connected"
    assert body["tenant_id"] == alice["id"]
    # Relay carries ids/status only — never message/comment text.
    assert body["payload"].get("account_id") == "acct_9"
