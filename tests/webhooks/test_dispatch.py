"""Webhook dispatcher — HMAC signature, retry/disable, type filtering."""

from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest_asyncio
import respx

from chalybclip.db import (
    Database,
    EventsRepo,
    TenantsRepo,
    WebhookSubscriptionsRepo,
    apply_migrations,
)
from chalybclip.tenancy import bound_tenant
from chalybclip.webhooks import (
    HMAC_HEADER,
    SIGNED_TS_HEADER,
    run_webhook_dispatch,
    sign_payload,
)
from chalybclip.webhooks.service import MAX_FAILURES_BEFORE_DISABLE


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "wh.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


# ---- HMAC helper ----


def test_sign_payload_matches_hmac_sha256() -> None:
    body = b'{"hello":"world"}'
    secret = "topsecret"
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert sign_payload(secret=secret, body=body) == expected


# ---- Dispatcher integration ----


@respx.mock
async def test_drain_delivers_signed_post_for_each_pending_event(
    db: Database,
) -> None:
    tenant = await TenantsRepo(db).create(name="A")
    with bound_tenant(tenant.id):
        sub = await WebhookSubscriptionsRepo(db).create(
            url="https://example.com/x",
            types=[],
            secret="sekret",
        )
        await EventsRepo(db).emit(type="clip.published", payload={"clip_id": "c1"})
        await EventsRepo(db).emit(type="clip.approved", payload={"clip_id": "c2"})

    route = respx.post("https://example.com/x").mock(
        return_value=httpx.Response(200, text="ok")
    )
    outcome = await run_webhook_dispatch(tenant.id, db)
    assert outcome.delivered == 2
    assert outcome.failed == 0
    assert route.call_count == 2

    # Each request carried the HMAC + timestamp headers and was POSTed JSON.
    req = route.calls[0].request
    assert HMAC_HEADER in req.headers
    assert SIGNED_TS_HEADER in req.headers
    body = req.content
    assert json.loads(body)["type"] in {"clip.published", "clip.approved"}
    expected_sig = sign_payload(secret="sekret", body=body)
    assert req.headers[HMAC_HEADER] == expected_sig

    # `last_dispatch_ts` advanced.
    with bound_tenant(tenant.id):
        refreshed = await WebhookSubscriptionsRepo(db).get(sub.id)
    assert refreshed is not None
    assert refreshed.last_dispatch_ts is not None
    assert refreshed.failure_count == 0


@respx.mock
async def test_drain_filters_by_subscribed_types(db: Database) -> None:
    """Only events matching `types` get delivered; prefix wildcards work."""
    tenant = await TenantsRepo(db).create(name="A")
    with bound_tenant(tenant.id):
        await WebhookSubscriptionsRepo(db).create(
            url="https://example.com/c",
            types=["clip.*"],
            secret="s",
        )
        await EventsRepo(db).emit(type="clip.published", payload={})
        await EventsRepo(db).emit(type="publish_job.failed", payload={})
        await EventsRepo(db).emit(type="clip.approved", payload={})

    route = respx.post("https://example.com/c").mock(
        return_value=httpx.Response(200)
    )
    outcome = await run_webhook_dispatch(tenant.id, db)
    # publish_job.failed must NOT match.
    assert outcome.delivered == 2
    assert route.call_count == 2


@respx.mock
async def test_drain_5xx_increments_failure_count(db: Database) -> None:
    tenant = await TenantsRepo(db).create(name="A")
    with bound_tenant(tenant.id):
        sub = await WebhookSubscriptionsRepo(db).create(
            url="https://example.com/broken/x",
            types=[],
            secret="s",
        )
        await EventsRepo(db).emit(type="clip.published", payload={})

    respx.post("https://example.com/broken/x").mock(
        return_value=httpx.Response(503)
    )
    outcome = await run_webhook_dispatch(tenant.id, db)
    assert outcome.delivered == 0
    assert outcome.failed == 1

    with bound_tenant(tenant.id):
        refreshed = await WebhookSubscriptionsRepo(db).get(sub.id)
    assert refreshed is not None
    assert refreshed.failure_count == 1
    assert refreshed.status == "active"  # not disabled yet


@respx.mock
async def test_drain_disables_after_max_failures(db: Database) -> None:
    """After MAX_FAILURES_BEFORE_DISABLE consecutive failures, status flips."""
    tenant = await TenantsRepo(db).create(name="A")
    with bound_tenant(tenant.id):
        sub = await WebhookSubscriptionsRepo(db).create(
            url="https://example.com/broken/x",
            types=[],
            secret="s",
        )
        # Pre-set the failure count to one short of the cap.
        for _ in range(MAX_FAILURES_BEFORE_DISABLE - 1):
            await WebhookSubscriptionsRepo(db).record_failure(sub.id)
        await EventsRepo(db).emit(type="clip.published", payload={})

    respx.post("https://example.com/broken/x").mock(return_value=httpx.Response(500))
    outcome = await run_webhook_dispatch(tenant.id, db)
    assert outcome.disabled == 1
    with bound_tenant(tenant.id):
        flipped = await WebhookSubscriptionsRepo(db).get(sub.id)
    assert flipped is not None
    assert flipped.status == "disabled"


@respx.mock
async def test_drain_skips_disabled_subscriptions(db: Database) -> None:
    """A subscription with status='disabled' is invisible to the drain."""
    tenant = await TenantsRepo(db).create(name="A")
    with bound_tenant(tenant.id):
        sub = await WebhookSubscriptionsRepo(db).create(
            url="https://example.com/x",
            types=[],
            secret="s",
        )
        # Disable directly (mimicking a previous over-failure flip).
        conn = await db.connect()
        await conn.execute(
            "UPDATE webhook_subscriptions SET status = 'disabled' WHERE id = ?",
            (sub.id,),
        )
        await conn.commit()
        await EventsRepo(db).emit(type="clip.published", payload={})

    route = respx.post("https://example.com/x").mock(
        return_value=httpx.Response(200)
    )
    outcome = await run_webhook_dispatch(tenant.id, db)
    assert outcome.delivered == 0
    assert route.call_count == 0


@respx.mock
async def test_drain_only_sends_events_after_last_dispatch_ts(
    db: Database,
) -> None:
    """Re-running a drain delivers only NEW events."""
    tenant = await TenantsRepo(db).create(name="A")
    with bound_tenant(tenant.id):
        sub = await WebhookSubscriptionsRepo(db).create(
            url="https://example.com/x",
            types=[],
            secret="s",
        )
        await EventsRepo(db).emit(type="clip.published", payload={"n": 1})

    respx.post("https://example.com/x").mock(return_value=httpx.Response(200))
    out1 = await run_webhook_dispatch(tenant.id, db)
    assert out1.delivered == 1

    # New event -> only one more delivered next pass.
    with bound_tenant(tenant.id):
        await EventsRepo(db).emit(type="clip.published", payload={"n": 2})

    out2 = await run_webhook_dispatch(tenant.id, db)
    assert out2.delivered == 1

    # Third drain with no new events -> nothing delivered.
    out3 = await run_webhook_dispatch(tenant.id, db)
    assert out3.delivered == 0
    _ = sub
