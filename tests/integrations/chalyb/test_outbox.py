"""Durable usage outbox — batching (≤100/request), reservation_id per
event, backoff on transient failures, dead-lettering of permanent 4xx
(isolated from the rest of their batch), and settle delivery."""

from __future__ import annotations

import datetime as _dt
import json as _json

import httpx
import pytest
import respx

from chalybclip.db import Database, TenantsRepo, apply_migrations
from chalybclip.db.usage_repos import UsageOutboxRepo
from chalybclip.integrations.chalyb.outbox import (
    bind_reservation,
    drain_outbox,
    enqueue_settle,
    enqueue_usage,
    reset_reservation,
)
from chalybclip.settings import get_settings

_USAGE = "https://chalyb.test/api/engines/chalybclip/usage"
_SETTLE = "https://chalyb.test/api/engines/chalybclip/usage/settle"


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


def _later(hours: int = 2) -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC) + _dt.timedelta(hours=hours)


@pytest.fixture
async def db(tmp_path) -> Database:
    d = Database(tmp_path / "t.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


@pytest.fixture
def chalyb_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CHALYB_BASE_URL", "https://chalyb.test")
    monkeypatch.setenv("CHALYB_ADMIN_TOKEN", "admintok")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def _tenant(db: Database, ext: str | None = "user-uuid") -> str:
    t = await TenantsRepo(db).create(name="T")
    if ext:
        await TenantsRepo(db).set_external_user_id(t.id, ext)
    return t.id


async def _enqueue(db: Database, tid: str, n: int, *, prefix: str = "ev") -> None:
    for i in range(n):
        await enqueue_usage(
            db, tenant_id=tid, kind="llm.tokens", amount=10 + i,
            cost_usd_micros=40 + i, source_id=f"{prefix}{i}", occurred_at_iso=_now(),
            provider="anthropic", operation="hook_generation", kick=False,
        )


@respx.mock
async def test_drain_batches_at_most_100_events_per_request(
    db: Database, chalyb_env
) -> None:
    tid = await _tenant(db)
    await _enqueue(db, tid, 150)
    route = respx.post(_USAGE).mock(return_value=httpx.Response(
        200, json={"balance": {"remaining": 7, "unlimited": False, "monthlyUsed": 1}}
    ))
    report = await drain_outbox(db)
    assert report.sent == 150
    sizes = [len(_json.loads(c.request.content)["events"]) for c in route.calls]
    assert sizes == [100, 50]
    body = _json.loads(route.calls[0].request.content)
    assert body["external_user_id"] == "user-uuid"
    ev = body["events"][0]
    assert {"kind", "amount", "cost_usd_micros", "source_id", "occurred_at"} <= set(ev)
    assert (await UsageOutboxRepo(db).count_by_status()) == {"sent": 150}
    # A second pass has nothing left to send.
    assert (await drain_outbox(db)).sent == 0
    assert route.call_count == 2


@respx.mock
async def test_events_carry_the_bound_reservation(db: Database, chalyb_env) -> None:
    tid = await _tenant(db)
    token = bind_reservation("res-42")
    try:
        await _enqueue(db, tid, 1)
    finally:
        reset_reservation(token)
    await _enqueue(db, tid, 1, prefix="free")
    route = respx.post(_USAGE).mock(return_value=httpx.Response(200, json={}))
    await drain_outbox(db)
    events = _json.loads(route.calls.last.request.content)["events"]
    by_id = {e["source_id"]: e for e in events}
    assert by_id["ev0"]["reservation_id"] == "res-42"
    assert "reservation_id" not in by_id["free0"]


@respx.mock
async def test_transient_failure_backs_off_then_delivers(db: Database, chalyb_env) -> None:
    tid = await _tenant(db)
    await _enqueue(db, tid, 3)
    route = respx.post(_USAGE).mock(side_effect=[
        httpx.Response(429, text="slow down"),
        httpx.Response(200, json={}),
    ])
    first = await drain_outbox(db)
    assert first.retried == 3 and first.sent == 0
    row = await UsageOutboxRepo(db).get("ev0")
    assert row.status == "pending" and row.attempts == 1
    assert row.next_attempt_at > _now()  # backed off into the future
    # Not due yet → no request.
    assert (await drain_outbox(db)).sent == 0
    assert route.call_count == 1
    assert (await drain_outbox(db, now=_later())).sent == 3
    assert route.call_count == 2


@respx.mock
async def test_permanent_4xx_dead_letters_only_the_offender(
    db: Database, chalyb_env
) -> None:
    """A 422 on a batch is split per event; the good events land, the bad
    one is marked dead (kept, not dropped)."""
    tid = await _tenant(db)
    await _enqueue(db, tid, 3)

    def handler(request: httpx.Request) -> httpx.Response:
        events = _json.loads(request.content)["events"]
        if any(e["source_id"] == "ev1" for e in events):
            return httpx.Response(422, text="occurred_at too old")
        return httpx.Response(200, json={})

    route = respx.post(_USAGE).mock(side_effect=handler)
    report = await drain_outbox(db)
    assert report.sent == 2 and report.dead == 1
    # 1 batch attempt + 3 single-event resends.
    assert route.call_count == 4
    dead = await UsageOutboxRepo(db).get("ev1")
    assert dead.status == "dead" and "422" in (dead.last_error or "")
    assert (await UsageOutboxRepo(db).get("ev0")).status == "sent"
    # Dead rows are never resent.
    await drain_outbox(db, now=_later())
    assert route.call_count == 4


@respx.mock
async def test_unlinked_tenant_events_are_dead_not_dropped(
    db: Database, chalyb_env
) -> None:
    tid = await _tenant(db, ext=None)
    await _enqueue(db, tid, 2)
    route = respx.post(_USAGE).mock(return_value=httpx.Response(200, json={}))
    report = await drain_outbox(db)
    assert report.dead == 2 and not route.called
    assert (await UsageOutboxRepo(db).count_by_status()) == {"dead": 2}


@respx.mock
async def test_settle_is_queued_once_and_delivered(db: Database, chalyb_env) -> None:
    tid = await _tenant(db)
    assert await enqueue_settle(
        db, tenant_id=tid, reservation_id="res-1", outcome="succeeded", kick=False
    )
    # First terminal outcome wins — settling twice is a no-op.
    assert not await enqueue_settle(
        db, tenant_id=tid, reservation_id="res-1", outcome="failed", kick=False
    )
    route = respx.post(_SETTLE).mock(return_value=httpx.Response(200, json={"ok": True}))
    await drain_outbox(db)
    assert _json.loads(route.calls.last.request.content) == {
        "reservation_id": "res-1", "outcome": "succeeded",
    }
    assert (await UsageOutboxRepo(db).get("settle_res-1")).status == "sent"


async def test_enqueue_is_idempotent_on_source_id(db: Database, chalyb_env) -> None:
    tid = await _tenant(db)
    assert await enqueue_usage(
        db, tenant_id=tid, kind="compute.seconds", amount=5, cost_usd_micros=440,
        source_id="dup", occurred_at_iso=_now(), kick=False,
    )
    assert not await enqueue_usage(
        db, tenant_id=tid, kind="compute.seconds", amount=5, cost_usd_micros=440,
        source_id="dup", occurred_at_iso=_now(), kick=False,
    )


async def test_cost_is_clamped_to_contract_bounds(db: Database, chalyb_env) -> None:
    tid = await _tenant(db)
    await enqueue_usage(
        db, tenant_id=tid, kind="llm.tokens", amount=10**15,
        cost_usd_micros=10**12, source_id="huge", occurred_at_iso=_now(), kick=False,
    )
    row = await UsageOutboxRepo(db).get("huge")
    assert row.payload["amount"] == 10**12
    assert row.payload["cost_usd_micros"] == 10**9
