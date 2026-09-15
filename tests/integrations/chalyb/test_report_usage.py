"""Token T4 — the unified, provider-agnostic usage reporter.

Every event now carries BOTH the native amount+kind AND the real
cost_usd_micros, so Chalyb can price off true spend across ALL
providers (Claude tokens, AssemblyAI seconds, …) — not just Claude.

Pins the POST body shape for: a generic report_usage call, the LLM
wrapper (kind=llm.tokens + cost), and a transcription event
(kind=transcription.seconds + cost). The captured request body is the
contract Chalyb's /usage endpoint consumes.
"""

from __future__ import annotations

import datetime as _dt
import json as _json

import httpx
import pytest
import respx

from chalybclip.db import Database, TenantsRepo, apply_migrations
from chalybclip.integrations.chalyb import reporter as reporter_mod
from chalybclip.integrations.chalyb.reporter import (
    report_llm_usage,
    report_usage,
    schedule_usage,
)
from chalybclip.settings import get_settings

_URL = "https://chalyb.test/api/engines/chalybclip/usage"


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


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


async def _tenant(db: Database) -> str:
    t = await TenantsRepo(db).create(name="T")
    await TenantsRepo(db).set_external_user_id(t.id, "auth0|u")
    return t.id


def _ok_response() -> httpx.Response:
    return httpx.Response(
        200, json={"balance": {"remaining": 10, "unlimited": False, "monthlyUsed": 1}},
    )


@respx.mock
async def test_generic_event_carries_kind_amount_and_cost(
    db: Database, chalyb_env
) -> None:
    tid = await _tenant(db)
    route = respx.post(_URL).mock(return_value=_ok_response())
    await report_usage(
        db, tenant_id=tid, kind="transcription.seconds", amount=125,
        cost_usd_micros=43_000, source_id="llmc_x", occurred_at_iso=_now(),
        provider="assemblyai", operation="transcribe",
    )
    assert route.called
    body = _json.loads(route.calls.last.request.content)
    ev = body["events"][0]
    assert ev["kind"] == "transcription.seconds"
    assert ev["amount"] == 125
    assert ev["cost_usd_micros"] == 43_000
    assert ev["source_id"] == "llmc_x"
    assert ev["provider"] == "assemblyai"
    assert ev["operation"] == "transcribe"
    assert body["external_user_id"] == "auth0|u"


@respx.mock
async def test_llm_wrapper_sends_llm_tokens_kind_plus_cost(
    db: Database, chalyb_env
) -> None:
    """report_llm_usage delegates to the generic with kind=llm.tokens,
    amount = input+output, and the router's real cost."""
    tid = await _tenant(db)
    route = respx.post(_URL).mock(return_value=_ok_response())
    await report_llm_usage(
        db, tenant_id=tid, llm_call_id="llm_a",
        input_tokens=20_000, output_tokens=17_000,
        cost_usd_micros=111_000, occurred_at_iso=_now(), operation="variants",
    )
    ev = _json.loads(route.calls.last.request.content)["events"][0]
    assert ev["kind"] == "llm.tokens"
    assert ev["amount"] == 37_000
    assert ev["cost_usd_micros"] == 111_000
    assert ev["source_id"] == "llm_a"
    # Provider defaults to "anthropic" for the LLM wrapper.
    assert ev["provider"] == "anthropic"


@respx.mock
async def test_cost_only_event_still_sends(db: Database, chalyb_env) -> None:
    """A provider with zero 'amount' but real cost still reports — the
    skip gate is amount<=0 AND cost<=0, not amount alone."""
    tid = await _tenant(db)
    route = respx.post(_URL).mock(return_value=_ok_response())
    await report_usage(
        db, tenant_id=tid, kind="vision.frames", amount=0,
        cost_usd_micros=5_000, source_id="v1", occurred_at_iso=_now(),
    )
    assert route.called
    ev = _json.loads(route.calls.last.request.content)["events"][0]
    assert ev["cost_usd_micros"] == 5_000


@respx.mock
async def test_zero_amount_zero_cost_is_skipped(db: Database, chalyb_env) -> None:
    tid = await _tenant(db)
    route = respx.post(_URL).mock(return_value=_ok_response())
    await report_usage(
        db, tenant_id=tid, kind="llm.tokens", amount=0,
        cost_usd_micros=0, source_id="z", occurred_at_iso=_now(),
    )
    assert not route.called


@respx.mock
async def test_db_close_drains_scheduled_report(db: Database, chalyb_env) -> None:
    """Shutdown-ordering regression (Modal worker, 2026-07-15): the run's
    teardown closed the DB while a scheduled usage report was still in its
    HTTP leg, so the report's follow-up writes (balance cache, report
    status) died with "pool is closed" and were silently lost. close() must
    drain the scheduled report first."""
    tid = await _tenant(db)
    route = respx.post(_URL).mock(return_value=_ok_response())

    before = set(reporter_mod._BACKGROUND_TASKS)
    schedule_usage(
        db, tenant_id=tid, kind="llm.tokens", amount=37,
        cost_usd_micros=1_000, source_id="lc_drain", occurred_at_iso=_now(),
        provider="openllm", operation="variants",
    )
    spawned = set(reporter_mod._BACKGROUND_TASKS) - before
    assert len(spawned) == 1
    task = spawned.pop()

    # Teardown immediately — pre-fix this closed the backend under the
    # in-flight report task instead of waiting for it.
    await db.close()
    assert task.done()
    assert route.called

    check = Database(db.target)
    try:
        t = await TenantsRepo(check).get(tid)
        assert t is not None
        assert t.last_usage_report_ok == 1
        assert t.cached_balance_remaining == 10
    finally:
        await check.close()
