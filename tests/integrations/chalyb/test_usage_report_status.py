"""Token T1 — the usage reporter records its outcome so a failing
balance-sync stops being invisible.

Pins: success clears the status, an HTTP-rejected report records a
failure with a token hint on 401/403, a missing admin token records a
config failure, and a network error records a failure — all WITHOUT
the reporter ever raising (it's a fire-and-forget background path).
"""

from __future__ import annotations

import datetime as _dt

import httpx
import pytest
import respx

from chalybclip.db import Database, TenantsRepo, apply_migrations
from chalybclip.integrations.chalyb import reporter as reporter_mod
from chalybclip.integrations.chalyb.reporter import report_llm_usage
from chalybclip.settings import get_settings


@pytest.fixture(autouse=True)
def _zero_retry_delays(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the retry schedule (attempt count) but drop the sleeps so
    transient-failure tests don't wait out real backoff."""
    monkeypatch.setattr(reporter_mod, "_RETRY_DELAYS_S", (0.0, 0.0))


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


async def _tenant_linked(db: Database) -> str:
    """A tenant linked to Chalyb (external_user_id set) so the reporter
    doesn't early-skip."""
    t = await TenantsRepo(db).create(name="T")
    await TenantsRepo(db).set_external_user_id(t.id, "auth0|user")
    return t.id


_URL = "https://chalyb.test/api/engines/chalybclip/usage"


@respx.mock
async def test_success_records_ok_and_clears_error(
    db: Database, chalyb_env
) -> None:
    tid = await _tenant_linked(db)
    respx.post(_URL).mock(return_value=httpx.Response(
        200, json={"balance": {"remaining": 1_900_000, "unlimited": False, "monthlyUsed": 100_000}},
    ))
    await report_llm_usage(
        db, tenant_id=tid, llm_call_id="lc1",
        input_tokens=20_000, output_tokens=17_000, occurred_at_iso=_now(),
    )
    t = await TenantsRepo(db).get(tid)
    assert t.last_usage_report_ok == 1
    assert t.last_usage_report_error is None
    # Balance cache also updated on success.
    assert t.cached_balance_remaining == 1_900_000


@respx.mock
async def test_http_403_records_failure_with_token_hint(
    db: Database, chalyb_env
) -> None:
    """The exact prod scenario: a misaligned admin token → Chalyb 403s
    the report → recorded as a failure that names the token."""
    tid = await _tenant_linked(db)
    respx.post(_URL).mock(return_value=httpx.Response(403, text="forbidden"))
    await report_llm_usage(
        db, tenant_id=tid, llm_call_id="lc2",
        input_tokens=10_000, output_tokens=0, occurred_at_iso=_now(),
    )
    t = await TenantsRepo(db).get(tid)
    assert t.last_usage_report_ok == 0
    assert "403" in (t.last_usage_report_error or "")
    assert "CHALYB_ADMIN_TOKEN" in (t.last_usage_report_error or "")


async def test_missing_admin_token_records_config_failure(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    tid = await _tenant_linked(db)
    monkeypatch.setenv("CHALYB_BASE_URL", "https://chalyb.test")
    # Empty-string override (a .env file may supply a value, which delenv
    # wouldn't clear); "" is falsy so the reporter hits the token-unset
    # early return.
    monkeypatch.setenv("CHALYB_ADMIN_TOKEN", "")
    get_settings.cache_clear()
    try:
        await report_llm_usage(
            db, tenant_id=tid, llm_call_id="lc3",
            input_tokens=5_000, output_tokens=0, occurred_at_iso=_now(),
        )
    finally:
        get_settings.cache_clear()
    t = await TenantsRepo(db).get(tid)
    assert t.last_usage_report_ok == 0
    assert "CHALYB_ADMIN_TOKEN" in (t.last_usage_report_error or "")


@respx.mock
async def test_network_error_records_failure_without_raising(
    db: Database, chalyb_env
) -> None:
    tid = await _tenant_linked(db)
    respx.post(_URL).mock(side_effect=httpx.ConnectError("boom"))
    # Must NOT raise — fire-and-forget path.
    await report_llm_usage(
        db, tenant_id=tid, llm_call_id="lc4",
        input_tokens=5_000, output_tokens=0, occurred_at_iso=_now(),
    )
    t = await TenantsRepo(db).get(tid)
    assert t.last_usage_report_ok == 0
    assert (t.last_usage_report_error or "") != ""


@respx.mock
async def test_timeout_retries_then_succeeds(db: Database, chalyb_env) -> None:
    """The prod incident: a transient timeout while the host is saturated.
    The report must retry instead of dropping the event (Chalyb would
    never deduct it) and leave the chip green after the retry lands."""
    tid = await _tenant_linked(db)
    route = respx.post(_URL).mock(side_effect=[
        httpx.ReadTimeout("slow"),
        httpx.Response(200, json={"balance": {
            "remaining": 1_949_000, "unlimited": False, "monthlyUsed": 3_050_000,
        }}),
    ])
    await report_llm_usage(
        db, tenant_id=tid, llm_call_id="lc6",
        input_tokens=1_000, output_tokens=0, occurred_at_iso=_now(),
    )
    assert route.call_count == 2
    t = await TenantsRepo(db).get(tid)
    assert t.last_usage_report_ok == 1
    assert t.last_usage_report_error is None
    assert t.cached_balance_remaining == 1_949_000


@respx.mock
async def test_all_attempts_time_out_records_failure(
    db: Database, chalyb_env
) -> None:
    tid = await _tenant_linked(db)
    route = respx.post(_URL).mock(side_effect=httpx.ReadTimeout("slow"))
    await report_llm_usage(
        db, tenant_id=tid, llm_call_id="lc7",
        input_tokens=1_000, output_tokens=0, occurred_at_iso=_now(),
    )
    assert route.call_count == 3  # 1 + the 2 zeroed retry delays
    t = await TenantsRepo(db).get(tid)
    assert t.last_usage_report_ok == 0
    assert "timeout" in (t.last_usage_report_error or "")


@respx.mock
async def test_5xx_is_retried(db: Database, chalyb_env) -> None:
    """Server errors are transient → retried until the schedule runs out."""
    tid = await _tenant_linked(db)
    route = respx.post(_URL).mock(return_value=httpx.Response(503, text="down"))
    await report_llm_usage(
        db, tenant_id=tid, llm_call_id="lc8",
        input_tokens=1_000, output_tokens=0, occurred_at_iso=_now(),
    )
    assert route.call_count == 3
    assert "503" in ((await TenantsRepo(db).get(tid)).last_usage_report_error or "")


@respx.mock
async def test_4xx_rejection_is_not_retried(db: Database, chalyb_env) -> None:
    """Payload/config rejections are permanent → one attempt only."""
    tid = await _tenant_linked(db)
    route = respx.post(_URL).mock(return_value=httpx.Response(422, text="bad kind"))
    await report_llm_usage(
        db, tenant_id=tid, llm_call_id="lc9",
        input_tokens=1_000, output_tokens=0, occurred_at_iso=_now(),
    )
    assert route.call_count == 1
    assert "422" in ((await TenantsRepo(db).get(tid)).last_usage_report_error or "")


@respx.mock
async def test_failure_then_success_clears_the_warning(
    db: Database, chalyb_env
) -> None:
    """A recovered sync flips the status back to ok so the chip stops
    warning."""
    tid = await _tenant_linked(db)
    respx.post(_URL).mock(return_value=httpx.Response(403, text="no"))
    await report_llm_usage(
        db, tenant_id=tid, llm_call_id="lc5a",
        input_tokens=1_000, output_tokens=0, occurred_at_iso=_now(),
    )
    assert (await TenantsRepo(db).get(tid)).last_usage_report_ok == 0

    respx.post(_URL).mock(return_value=httpx.Response(
        200, json={"balance": {"remaining": 5, "unlimited": False, "monthlyUsed": 1}},
    ))
    await report_llm_usage(
        db, tenant_id=tid, llm_call_id="lc5b",
        input_tokens=1_000, output_tokens=0, occurred_at_iso=_now(),
    )
    t = await TenantsRepo(db).get(tid)
    assert t.last_usage_report_ok == 1
    assert t.last_usage_report_error is None
