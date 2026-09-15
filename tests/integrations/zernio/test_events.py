"""Zernio event processor + webhook registration tests (phase 2).

The receiver path is covered end-to-end in tests/api/test_zernio_webhooks.py;
these pin the processor's edge cases (unresolved tenant, malformed
payload, idempotent re-runs, the pending sweep) and the idempotent
create-or-update registration against Zernio's /webhooks/settings.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
import respx

from chalybclip.db import (
    Database,
    EventsRepo,
    TenantsRepo,
    ZernioEventsRepo,
    apply_migrations,
)
from chalybclip.integrations.zernio.client import ZernioClient
from chalybclip.integrations.zernio.events import (
    SUBSCRIBED_EVENTS,
    extract_profile_id,
    process_pending,
    process_zernio_event,
)
from chalybclip.integrations.zernio.webhooks import register_zernio_webhook
from chalybclip.tenancy import bound_tenant

_ZBASE = "https://zernio.com/api/v1"


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "events.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


async def _store(
    db: Database,
    *,
    event_id: str,
    type_: str,
    payload: dict[str, Any] | str,
) -> None:
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    await ZernioEventsRepo(db).insert_dedup(
        event_id=event_id, type=type_, payload=raw,
    )


# ---- extract_profile_id ----


def test_extract_profile_id_handles_all_shapes() -> None:
    assert extract_profile_id({"profileId": "p1"}) == "p1"
    assert extract_profile_id({"account": {"profileId": "p2"}}) == "p2"
    assert extract_profile_id({"post": {"profileId": "p3"}}) == "p3"
    # The nested-object regression shape.
    assert extract_profile_id({"account": {"profileId": {"_id": "p4"}}}) == "p4"
    assert extract_profile_id({}) is None


# ---- processor edge cases ----


@pytest.mark.asyncio
async def test_unresolved_profile_marks_processed_without_event(
    db: Database,
) -> None:
    await _store(
        db,
        event_id="evt_orphan",
        type_="account.connected",
        payload={
            "id": "evt_orphan",
            "event": "account.connected",
            "account": {"accountId": "a", "profileId": "prof_nobody"},
        },
    )
    await process_zernio_event(db, "evt_orphan")
    row = await ZernioEventsRepo(db).get("evt_orphan")
    assert row is not None
    assert row.processed is True
    assert row.tenant_id is None  # kept, not dropped — just unresolved


@pytest.mark.asyncio
async def test_malformed_payload_row_does_not_crash(db: Database) -> None:
    await _store(
        db, event_id="evt_bad", type_="post.published", payload="{not json",
    )
    await process_zernio_event(db, "evt_bad")  # must not raise
    row = await ZernioEventsRepo(db).get("evt_bad")
    assert row is not None
    assert row.processed is True


@pytest.mark.asyncio
async def test_reprocessing_is_a_noop(db: Database) -> None:
    tenant = await TenantsRepo(db).create(name="Rerun Co")
    await TenantsRepo(db).set_zernio_profile(
        tenant.id, profile_id="prof_rerun", profile_name="R",
    )
    await _store(
        db,
        event_id="evt_rerun",
        type_="account.connected",
        payload={
            "id": "evt_rerun",
            "event": "account.connected",
            "account": {"accountId": "a", "profileId": "prof_rerun"},
        },
    )
    await process_zernio_event(db, "evt_rerun")
    await process_zernio_event(db, "evt_rerun")  # already processed
    with bound_tenant(tenant.id):
        rows = await EventsRepo(db).list_for_tenant(limit=20)
    assert len([r for r in rows if r.type == "zernio.account.connected"]) == 1


@pytest.mark.asyncio
async def test_process_pending_sweeps_backlog(db: Database) -> None:
    for i in range(3):
        await _store(
            db,
            event_id=f"evt_sweep_{i}",
            type_="webhook.test",
            payload={"id": f"evt_sweep_{i}", "event": "webhook.test"},
        )
    n = await process_pending(db)
    assert n == 3
    assert await ZernioEventsRepo(db).list_unprocessed() == []
    assert await process_pending(db) == 0  # idempotent


# ---- register_zernio_webhook (idempotent create-or-update) ----


def _client(http: httpx.AsyncClient) -> ZernioClient:
    return ZernioClient(api_key="sk_test_abc", http=http)


@pytest.mark.asyncio
async def test_register_creates_when_absent() -> None:
    created = {"webhook": {"_id": "wh_1", "name": "ChalyClip Hub"}}
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            mock.get(f"{_ZBASE}/webhooks/settings").mock(
                return_value=httpx.Response(200, json={"webhooks": []})
            )
            post = mock.post(f"{_ZBASE}/webhooks/settings").mock(
                return_value=httpx.Response(200, json=created)
            )
            result = await register_zernio_webhook(
                _client(http),
                url="https://hub.test/api/webhooks/zernio",
                secret="whsec_1",
            )
    assert result["action"] == "created"
    assert result["webhook_id"] == "wh_1"
    payload = json.loads(post.calls.last.request.content.decode())
    assert payload["name"] == "ChalyClip Hub"
    assert payload["url"] == "https://hub.test/api/webhooks/zernio"
    assert payload["secret"] == "whsec_1"
    assert payload["isActive"] is True
    assert payload["events"] == list(SUBSCRIBED_EVENTS)


@pytest.mark.asyncio
async def test_register_updates_in_place_when_present() -> None:
    existing = {
        "webhooks": [
            {"_id": "wh_9", "name": "ChalyClip Hub",
             "url": "https://old.test/hook", "isActive": False},
        ]
    }
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            mock.get(f"{_ZBASE}/webhooks/settings").mock(
                return_value=httpx.Response(200, json=existing)
            )
            put = mock.put(f"{_ZBASE}/webhooks/settings").mock(
                return_value=httpx.Response(
                    200, json={"webhook": {"_id": "wh_9"}}
                )
            )
            result = await register_zernio_webhook(
                _client(http),
                url="https://hub.test/api/webhooks/zernio",
                secret="whsec_2",
            )
    assert result["action"] == "updated"
    assert result["webhook_id"] == "wh_9"
    payload = json.loads(put.calls.last.request.content.decode())
    assert payload["_id"] == "wh_9"
    assert payload["url"] == "https://hub.test/api/webhooks/zernio"
    # Re-activates a webhook Zernio auto-disabled after failures.
    assert payload["isActive"] is True
