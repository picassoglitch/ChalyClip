"""Dashboard surfaces the cost projection cards on /dashboard/llm-calls.

This HTML view shows raw USD economics, so it's ADMIN-ONLY (creators see
tokens, never cost). Each test flags alice as admin via
CHALYBCLIP_ADMIN_TENANT_IDS so the page renders.
"""

from __future__ import annotations

import datetime as _dt

import httpx
import pytest

from chalybclip.db import Database, LLMCallsRepo
from chalybclip.db.models import LLMCallRow
from chalybclip.tenancy import bound_tenant

from .conftest import auth


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


def _make_admin(monkeypatch: pytest.MonkeyPatch, tenant_id: str) -> None:
    """Flag a tenant as admin so the USD cost pages render for it."""
    from chalybclip.settings import get_settings

    monkeypatch.setenv("CHALYBCLIP_ADMIN_TENANT_IDS", tenant_id)
    get_settings.cache_clear()


async def test_dashboard_llm_calls_renders_projection_cards(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant_id = tenants["alice"]["id"]
    _make_admin(monkeypatch, tenant_id)
    with bound_tenant(tenant_id):
        await LLMCallsRepo(db).record(
            LLMCallRow(
                id="llm_one",
                tenant_id=tenant_id,
                purpose="variant_generation",
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                quality="standard",
                input_tokens=100,
                output_tokens=50,
                cost_usd_micros=2_500_000,
                status="ok",
                error=None,
                attempts=1,
                ts=_now(),
            )
        )

    r = await client.get(
        "/dashboard/llm-calls", headers=auth(tenants["alice"]["token"])
    )
    assert r.status_code == 200
    assert "Today" in r.text
    assert "Month-to-date" in r.text
    assert "Projected EOM" in r.text
    assert "By purpose (MTD)" in r.text
    assert "By model (MTD)" in r.text


async def test_dashboard_llm_calls_renders_budget_bar_when_set(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the tenant has a budget cap, the headroom bar shows."""
    from chalybclip.db import TenantsRepo

    tenant_id = tenants["alice"]["id"]
    _make_admin(monkeypatch, tenant_id)
    await TenantsRepo(db).set_budget(
        tenant_id, daily_llm_budget_usd_micros=10_000_000
    )
    with bound_tenant(tenant_id):
        await LLMCallsRepo(db).record(
            LLMCallRow(
                id="llm_b",
                tenant_id=tenant_id,
                purpose="variant_generation",
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                quality="standard",
                input_tokens=100,
                output_tokens=50,
                cost_usd_micros=2_500_000,
                status="ok",
                error=None,
                attempts=1,
                ts=_now(),
            )
        )
    r = await client.get(
        "/dashboard/llm-calls", headers=auth(tenants["alice"]["token"])
    )
    assert r.status_code == 200
    assert "daily ceiling" in r.text
    assert "consumed" in r.text
