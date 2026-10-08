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


# ---- config / auth errors are retried, not dead-lettered -------------------


@pytest.mark.parametrize(
    "resp",
    [
        httpx.Response(401, json={"error": "missing bearer token"}),
        httpx.Response(403, json={"error": "invalid bearer token"}),
        httpx.Response(404, json={"error": "unknown engine: chalybclip"}),
        httpx.Response(404, json={"error": "unknown user_id"}),
        httpx.Response(404, text="<html>This page could not be found</html>"),
    ],
    ids=["401", "403", "404-engine", "404-user", "404-wrong-base-url"],
)
@respx.mock
async def test_auth_or_config_rejection_keeps_events_pending(
    db: Database, chalyb_env, resp: httpx.Response
) -> None:
    """A rotated/mismatched CHALYB_ADMIN_TOKEN (or any other config 4xx)
    says nothing about the events: they stay pending with backoff, the
    tenant's status shows the error, and they land once it's fixed."""
    tid = await _tenant(db)
    await _enqueue(db, tid, 3)
    route = respx.post(_USAGE).mock(side_effect=[resp, httpx.Response(200, json={})])
    report = await drain_outbox(db)
    assert report.dead == 0 and report.retried == 3
    assert route.call_count == 1  # not split per event: it isn't about the events
    repo = UsageOutboxRepo(db)
    assert (await repo.count_by_status()) == {"pending": 3}
    row = await repo.get("ev0")
    assert row.attempts == 1 and row.next_attempt_at > _now()
    assert str(resp.status_code) in (row.last_error or "")
    tenant = await TenantsRepo(db).get(tid)
    assert tenant.last_usage_report_ok in (0, False)
    if resp.status_code in (401, 403):
        assert "CHALYB_ADMIN_TOKEN" in (tenant.last_usage_report_error or "")
    # Config fixed → next pass after the backoff delivers everything.
    assert (await drain_outbox(db, now=_later())).sent == 3
    assert (await repo.count_by_status()) == {"sent": 3}


@respx.mock
async def test_400_event_rejection_is_still_dead(db: Database, chalyb_env) -> None:
    tid = await _tenant(db)
    await _enqueue(db, tid, 1)
    respx.post(_USAGE).mock(return_value=httpx.Response(
        400, json={"error": "invalid kind", "index": 0}
    ))
    report = await drain_outbox(db)
    assert report.dead == 1
    assert (await UsageOutboxRepo(db).get("ev0")).status == "dead"


@pytest.mark.parametrize(
    "resp",
    [
        httpx.Response(401, json={"error": "missing bearer token"}),
        httpx.Response(403, json={"error": "invalid bearer token"}),
        httpx.Response(400, json={"ok": False, "error": "engine not registered: chalybclip"}),
        httpx.Response(404, text="not found"),
    ],
    ids=["401", "403", "400-engine", "404-page"],
)
@respx.mock
async def test_settle_config_rejection_is_retried(
    db: Database, chalyb_env, resp: httpx.Response
) -> None:
    tid = await _tenant(db)
    await enqueue_settle(db, tenant_id=tid, reservation_id="res-1", outcome="succeeded", kick=False)
    respx.post(_SETTLE).mock(side_effect=[resp, httpx.Response(200, json={"ok": True})])
    report = await drain_outbox(db)
    assert report.dead == 0 and report.retried == 1
    assert (await UsageOutboxRepo(db).get("settle_res-1")).status == "pending"
    await drain_outbox(db, now=_later())
    assert (await UsageOutboxRepo(db).get("settle_res-1")).status == "sent"


@respx.mock
async def test_settle_for_unknown_reservation_is_dead(db: Database, chalyb_env) -> None:
    tid = await _tenant(db)
    await enqueue_settle(db, tenant_id=tid, reservation_id="res-x", outcome="failed", kick=False)
    respx.post(_SETTLE).mock(return_value=httpx.Response(
        404, json={"error": "unknown reservation"}
    ))
    report = await drain_outbox(db)
    assert report.dead == 1
    assert (await UsageOutboxRepo(db).get("settle_res-x")).status == "dead"


# ---- settle waits for its job's usage ---------------------------------------


@respx.mock
async def test_settle_waits_until_the_jobs_usage_has_landed(db: Database, chalyb_env) -> None:
    """If the run's usage is backing off, the settle must not close the
    hold first (that would free tokens the job already spent)."""
    tid = await _tenant(db)
    token = bind_reservation("res-7")
    try:
        await _enqueue(db, tid, 2)
    finally:
        reset_reservation(token)
    await enqueue_settle(db, tenant_id=tid, reservation_id="res-7", outcome="succeeded", kick=False)
    usage = respx.post(_USAGE).mock(side_effect=[
        httpx.Response(503, text="down"), httpx.Response(200, json={}),
    ])
    settle = respx.post(_SETTLE).mock(return_value=httpx.Response(200, json={"ok": True}))

    await drain_outbox(db)
    assert usage.call_count == 1 and not settle.called
    row = await UsageOutboxRepo(db).get("settle_res-7")
    assert row.status == "pending" and row.attempts == 0
    assert "waiting" in (row.last_error or "")

    # Later: usage lands first in the pass, then the settle goes out.
    await drain_outbox(db, now=_later())
    assert usage.call_count == 2 and settle.call_count == 1
    assert (await UsageOutboxRepo(db).count_by_status()) == {"sent": 3}


@respx.mock
async def test_settle_is_not_held_by_other_jobs_or_dead_usage(db: Database, chalyb_env) -> None:
    tid = await _tenant(db)
    token = bind_reservation("res-other")
    try:
        await _enqueue(db, tid, 1, prefix="other")
    finally:
        reset_reservation(token)
    await enqueue_settle(db, tenant_id=tid, reservation_id="res-8", outcome="failed", kick=False)
    respx.post(_USAGE).mock(return_value=httpx.Response(503))
    settle = respx.post(_SETTLE).mock(return_value=httpx.Response(200, json={"ok": True}))
    await drain_outbox(db)
    assert settle.call_count == 1


# ---- lease is atomic --------------------------------------------------------


async def test_claim_gives_each_row_to_one_drain(db: Database, chalyb_env) -> None:
    tid = await _tenant(db)
    await _enqueue(db, tid, 3)
    repo = UsageOutboxRepo(db)
    now = _now()
    until = _later().isoformat()
    ids = [r.id for r in await repo.due(now_iso=now, limit=10)]
    # Two drains read the same due rows; only the first claim wins them.
    first = await repo.claim(ids, now_iso=now, until_iso=until)
    second = await repo.claim(ids, now_iso=now, until_iso=until)
    assert sorted(first) == ["ev0", "ev1", "ev2"] and second == []


@respx.mock
async def test_concurrent_drains_send_each_event_once(db: Database, chalyb_env) -> None:
    import asyncio

    tid = await _tenant(db)
    await _enqueue(db, tid, 5)
    sent: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.01)
        sent.extend(e["source_id"] for e in _json.loads(request.content)["events"])
        return httpx.Response(200, json={})

    respx.post(_USAGE).mock(side_effect=handler)
    a, b = await asyncio.gather(drain_outbox(db), drain_outbox(db))
    assert a.sent + b.sent == 5
    assert sorted(sent) == [f"ev{i}" for i in range(5)]
