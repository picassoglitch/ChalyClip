"""LLMRouter dual-writes: cost-tracking row in the DB plus the JSONL breadcrumb."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import BaseModel

from chalybclip.db import Database, LLMCallsRepo, TenantsRepo, apply_migrations
from chalybclip.llm import LLMRouter
from chalybclip.tenancy import bound_tenant

from ._fakes import FakeProvider
from ._fixtures import make_llm_config


class TinySchema(BaseModel):
    answer: str


def _factory(provider: FakeProvider):
    def _build(name, _config, _api_key):
        return provider if name == "anthropic" else None

    return _build


async def _seed_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "test.db")
    await apply_migrations(db)
    await TenantsRepo(db).create(tenant_id="ten_a", name="A")
    return db


def test_router_writes_llm_calls_row_in_addition_to_jsonl(tmp_path: Path) -> None:
    db = asyncio.run(_seed_db(tmp_path))
    log_path = tmp_path / "llm_calls.jsonl"

    fake = FakeProvider("anthropic")
    fake.queue_success({"answer": "x"}, input_tokens=1000, output_tokens=500)
    config = make_llm_config(retry_attempts=1, pricing_input=0.80, pricing_output=4.00)
    router = LLMRouter(
        config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory(fake),
        call_log_path=log_path,
        db=db,
    )

    async def go() -> None:
        with bound_tenant("ten_a"):
            await router.complete(
                tenant_id="ten_a",
                purpose="variant_generation",
                system="s",
                user="u",
                schema=TinySchema,
            )

    asyncio.run(go())

    # JSONL has the row.
    lines = [
        json.loads(line) for line in log_path.read_text("utf-8").splitlines() if line
    ]
    assert len(lines) == 1
    assert lines[0]["cost_usd_micros"] == 2800

    # DB has the row.
    async def fetch() -> list:
        with bound_tenant("ten_a"):
            return await LLMCallsRepo(db).list_for_tenant()

    rows = asyncio.run(fetch())
    asyncio.run(db.close())
    assert len(rows) == 1
    assert rows[0].cost_usd_micros == 2800
    assert rows[0].purpose == "variant_generation"


def test_router_db_failure_does_not_break_call(tmp_path: Path) -> None:
    """Even if the DB write fails (e.g. tenant missing), the LLM call still
    succeeds — JSONL captures the row for later reconciliation."""
    db = Database(tmp_path / "test.db")
    asyncio.run(apply_migrations(db))
    # Note: we deliberately do NOT seed a tenant, so the FK insert will fail.
    log_path = tmp_path / "llm_calls.jsonl"

    fake = FakeProvider("anthropic")
    fake.queue_success({"answer": "x"})
    config = make_llm_config(retry_attempts=1)
    router = LLMRouter(
        config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory(fake),
        call_log_path=log_path,
        db=db,
    )

    async def go() -> TinySchema:
        return await router.complete(
            tenant_id="ten_ghost",
            purpose="variant_generation",
            system="s",
            user="u",
            schema=TinySchema,
        )

    result = asyncio.run(go())
    asyncio.run(db.close())
    assert result.answer == "x"
    # JSONL still has the row even though the DB write was swallowed.
    assert log_path.exists() and log_path.read_text("utf-8").strip()


def test_router_without_db_still_writes_jsonl(tmp_path: Path) -> None:
    """Phase 0 backward-compat: no db -> JSONL only, like before."""
    log_path = tmp_path / "llm_calls.jsonl"
    fake = FakeProvider("anthropic")
    fake.queue_success({"answer": "x"})
    router = LLMRouter(
        make_llm_config(retry_attempts=1),
        api_keys={"anthropic": "k"},
        provider_factory=_factory(fake),
        call_log_path=log_path,
        db=None,
    )

    async def go() -> None:
        await router.complete(
            tenant_id="t",
            purpose="variant_generation",
            system="s",
            user="u",
            schema=TinySchema,
        )

    asyncio.run(go())
    assert log_path.exists() and log_path.read_text("utf-8").strip()
