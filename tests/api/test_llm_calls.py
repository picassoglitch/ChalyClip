"""`GET /llm-calls` returns the tenant's LLM-call audit log."""

from __future__ import annotations

import datetime as _dt

import httpx

from chalybclip.db import Database, LLMCallsRepo
from chalybclip.db.models import LLMCallRow
from chalybclip.tenancy import bound_tenant

from .conftest import auth


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


async def _seed_call(db: Database, tenant_id: str, *, call_id: str) -> None:
    with bound_tenant(tenant_id):
        await LLMCallsRepo(db).record(
            LLMCallRow(
                id=call_id,
                tenant_id=tenant_id,
                purpose="variant_generation",
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                quality="standard",
                input_tokens=200,
                output_tokens=80,
                cost_usd_micros=600,
                status="ok",
                error=None,
                attempts=1,
                ts=_now(),
            )
        )


async def test_lists_only_my_tenant_llm_calls(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    await _seed_call(db, tenants["alice"]["id"], call_id="llm_a1")
    await _seed_call(db, tenants["alice"]["id"], call_id="llm_a2")
    await _seed_call(db, tenants["bob"]["id"], call_id="llm_b1")

    r = await client.get("/llm-calls", headers=auth(tenants["alice"]["token"]))
    assert r.status_code == 200
    ids = {row["id"] for row in r.json()}
    assert ids == {"llm_a1", "llm_a2"}

    r = await client.get("/llm-calls", headers=auth(tenants["bob"]["token"]))
    ids = {row["id"] for row in r.json()}
    assert ids == {"llm_b1"}


async def test_limit_param_caps_result_size(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    for i in range(5):
        await _seed_call(db, tenants["alice"]["id"], call_id=f"llm_a{i}")
    r = await client.get(
        "/llm-calls?limit=3", headers=auth(tenants["alice"]["token"])
    )
    assert r.status_code == 200
    assert len(r.json()) == 3
