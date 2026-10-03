"""Hub admission client — /usage/admit request shape, refusal → HTTP
mapping, and the fail-closed policy (hub down ⇒ refuse, unless no hub is
configured at all)."""

from __future__ import annotations

import json as _json

import httpx
import pytest
import respx

from chalybclip.db import Database, TenantsRepo, apply_migrations
from chalybclip.integrations.chalyb import admission as admission_mod
from chalybclip.integrations.chalyb.admission import (
    REFUSAL_MESSAGES,
    AdmissionRefused,
    AdmitRequest,
    admit,
    heartbeat,
)
from chalybclip.settings import get_settings

_ADMIT = "https://chalyb.test/api/engines/chalybclip/usage/admit"
_SETTLE = "https://chalyb.test/api/engines/chalybclip/usage/settle"


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
    monkeypatch.setattr(admission_mod, "_ADMIT_RETRY_DELAY_S", 0.0)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def _linked_tenant(db: Database) -> str:
    t = await TenantsRepo(db).create(name="T")
    await TenantsRepo(db).set_external_user_id(t.id, "user-uuid")
    return t.id


@pytest.mark.parametrize(
    ("reason", "status"),
    [
        ("upload_too_large", 413),
        ("video_too_long", 413),
        ("storage_full", 413),
        ("no_tokens", 402),
        ("boost_unavailable", 402),
        ("concurrency", 429),
        ("jobs_cap", 429),
        ("minutes_cap", 429),
        ("streams_cap", 429),
        ("hub_unavailable", 503),
        ("some_future_reason", 429),
    ],
)
def test_refusal_reason_maps_to_status_and_spanish_message(reason: str, status: int) -> None:
    e = AdmissionRefused(reason)
    assert e.status_code == status
    assert e.reason == reason
    # Spanish, user-facing — never the raw reason code.
    assert reason not in e.user_message
    assert any(w in e.user_message for w in ("tu", "Tu", "No ", "Ya ", "El ", "Este"))


def test_every_contract_reason_has_a_message() -> None:
    contract = {
        "upload_too_large", "video_too_long", "storage_full", "minutes_cap",
        "jobs_cap", "concurrency", "streams_cap", "no_tokens", "boost_unavailable",
    }
    assert contract <= set(REFUSAL_MESSAGES)


@respx.mock
async def test_admitted_response_is_parsed_and_request_shape_matches_contract(
    db: Database, chalyb_env
) -> None:
    tid = await _linked_tenant(db)
    route = respx.post(_ADMIT).mock(return_value=httpx.Response(200, json={
        "ok": True, "allowed": True, "reservation_id": "res-1", "lane": "boost",
        "boost_fee_tokens": 0,
        "limits": {"max_upload_mb": 20480, "max_source_minutes": 480},
        "balance": {"remaining": 4_812_345, "reserved": 40_000},
    }))
    result = await admit(db, tenant_id=tid, request=AdmitRequest(
        external_job_id="ujob_1", est_tokens=40_000, upload_mb=812.4,
        source_minutes=95.5, storage_mb_after=3120, boost=None, ttl_seconds=10800,
    ))
    assert result.allowed and result.reservation_id == "res-1"
    assert result.lane == "boost"
    assert result.max_upload_bytes == 20480 * 1024 * 1024
    req = route.calls.last.request
    assert req.headers["authorization"] == "Bearer admintok"
    body = _json.loads(req.content)
    assert body == {
        "external_user_id": "user-uuid",
        "external_job_id": "ujob_1",
        "class": "job",
        "operation": "clips.pipeline",
        "est_tokens": 40_000,
        "upload_mb": 812.4,
        "source_minutes": 95.5,
        "storage_mb_after": 3120,
        "boost": None,
        "ttl_seconds": 10800,
    }


@respx.mock
async def test_refused_response_raises_with_reason(db: Database, chalyb_env) -> None:
    tid = await _linked_tenant(db)
    respx.post(_ADMIT).mock(return_value=httpx.Response(200, json={
        "ok": True, "allowed": False, "reason": "no_tokens", "remaining": 3, "needed": 9,
    }))
    with pytest.raises(AdmissionRefused) as ei:
        await admit(db, tenant_id=tid, request=AdmitRequest(external_job_id="j"))
    assert ei.value.reason == "no_tokens"
    assert ei.value.status_code == 402
    assert ei.value.detail["needed"] == 9


@respx.mock
async def test_hub_down_fails_closed(db: Database, chalyb_env) -> None:
    """5xx twice (one retry) → refuse; never run unmetered."""
    tid = await _linked_tenant(db)
    route = respx.post(_ADMIT).mock(return_value=httpx.Response(503))
    with pytest.raises(AdmissionRefused) as ei:
        await admit(db, tenant_id=tid, request=AdmitRequest(external_job_id="j"))
    assert ei.value.reason == "hub_unavailable"
    assert ei.value.status_code == 503
    assert route.call_count == 2


@respx.mock
async def test_network_error_fails_closed(db: Database, chalyb_env) -> None:
    tid = await _linked_tenant(db)
    respx.post(_ADMIT).mock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(AdmissionRefused) as ei:
        await admit(db, tenant_id=tid, request=AdmitRequest(external_job_id="j"))
    assert ei.value.reason == "hub_unavailable"


@respx.mock
async def test_auth_rejection_fails_closed(db: Database, chalyb_env) -> None:
    tid = await _linked_tenant(db)
    route = respx.post(_ADMIT).mock(return_value=httpx.Response(403, text="nope"))
    with pytest.raises(AdmissionRefused) as ei:
        await admit(db, tenant_id=tid, request=AdmitRequest(external_job_id="j"))
    assert ei.value.reason == "hub_unavailable"
    assert route.call_count == 1  # a 4xx isn't retried


async def test_no_hub_configured_admits_locally(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local dev: CHALYB_BASE_URL unset → allowed (warning), no reservation."""
    monkeypatch.setenv("CHALYB_BASE_URL", "")
    get_settings.cache_clear()
    try:
        tid = await _linked_tenant(db)
        result = await admit(db, tenant_id=tid, request=AdmitRequest(external_job_id="j"))
    finally:
        get_settings.cache_clear()
    assert result.allowed and result.local and result.reservation_id is None


@respx.mock
async def test_missing_admin_token_fails_closed(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHALYB_BASE_URL", "https://chalyb.test")
    monkeypatch.setenv("CHALYB_ADMIN_TOKEN", "")
    get_settings.cache_clear()
    try:
        tid = await _linked_tenant(db)
        with pytest.raises(AdmissionRefused) as ei:
            await admit(db, tenant_id=tid, request=AdmitRequest(external_job_id="j"))
    finally:
        get_settings.cache_clear()
    assert ei.value.reason == "hub_unavailable"


@respx.mock
async def test_heartbeat_posts_settle_heartbeat(chalyb_env) -> None:
    route = respx.post(_SETTLE).mock(return_value=httpx.Response(200, json={"ok": True}))
    assert await heartbeat("res-9") is True
    assert _json.loads(route.calls.last.request.content) == {
        "reservation_id": "res-9", "outcome": "heartbeat",
    }
