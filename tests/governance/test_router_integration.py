"""LLMRouter consults BudgetGovernor before each call when one is wired in."""

from __future__ import annotations

import asyncio
import datetime as _dt
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from pydantic import BaseModel

from chalybclip.db import (
    Database,
    EventsRepo,
    LLMCallsRepo,
    TenantsRepo,
    apply_migrations,
)
from chalybclip.db.models import LLMCallRow
from chalybclip.errors import BudgetExceeded
from chalybclip.governance import BudgetGovernor
from chalybclip.llm import LLMRouter
from chalybclip.llm.config import ProviderConfig
from chalybclip.tenancy import bound_tenant
from tests.llm._fakes import FakeProvider  # type: ignore[import]
from tests.llm._fixtures import make_llm_config  # type: ignore[import]


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "router_gov.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


def _factory(providers: dict[str, FakeProvider]):
    def _build(name: str, _config: ProviderConfig, _api_key: str) -> FakeProvider | None:
        return providers.get(name)

    return _build


class TinySchema(BaseModel):
    answer: str


async def _seed_tenant(db: Database, *, daily_llm_budget_usd_micros: int | None) -> str:
    t = await TenantsRepo(db).create(name="Aldo")
    if daily_llm_budget_usd_micros is not None:
        await TenantsRepo(db).set_budget(
            t.id, daily_llm_budget_usd_micros=daily_llm_budget_usd_micros
        )
    return t.id


async def test_router_calls_provider_when_under_budget(db: Database) -> None:
    tenant_id = await _seed_tenant(db, daily_llm_budget_usd_micros=10_000_000)
    fake = FakeProvider("anthropic")
    fake.queue_success({"answer": "ok"})
    gov = BudgetGovernor(db)
    router = LLMRouter(
        config=make_llm_config(),
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
        db=db,
        governor=gov,
    )
    res = await router.complete(
        tenant_id=tenant_id,
        purpose="variant_generation",
        system="s",
        user="u",
        schema=TinySchema,
    )
    assert res.answer == "ok"
    assert len(fake.calls) == 1


async def test_router_refuses_when_budget_exhausted(db: Database) -> None:
    """Today's spend already at cap -> next complete() raises BudgetExceeded."""
    tenant_id = await _seed_tenant(db, daily_llm_budget_usd_micros=1_000)
    # Seed today's spend at the cap.
    with bound_tenant(tenant_id):
        await LLMCallsRepo(db).record(
            LLMCallRow(
                id="llm_seed",
                tenant_id=tenant_id,
                purpose="variant_generation",
                provider="anthropic",
                model="m",
                quality="standard",
                input_tokens=1,
                output_tokens=1,
                cost_usd_micros=1_000,
                status="ok",
                error=None,
                attempts=1,
                ts=_now(),
            )
        )
    fake = FakeProvider("anthropic")
    fake.queue_success({"answer": "would_run"})  # should never be popped
    gov = BudgetGovernor(db)
    router = LLMRouter(
        config=make_llm_config(),
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
        db=db,
        governor=gov,
    )
    with pytest.raises(BudgetExceeded):
        await router.complete(
            tenant_id=tenant_id,
            purpose="variant_generation",
            system="s",
            user="u",
            schema=TinySchema,
        )
    # Provider was never asked.
    assert fake.calls == []


async def test_budget_exhaustion_emits_event(db: Database) -> None:
    """`llm.budget_exhausted` event lands in the events table on refusal."""
    tenant_id = await _seed_tenant(db, daily_llm_budget_usd_micros=1_000)
    with bound_tenant(tenant_id):
        await LLMCallsRepo(db).record(
            LLMCallRow(
                id="llm_seed",
                tenant_id=tenant_id,
                purpose="variant_generation",
                provider="anthropic",
                model="m",
                quality="standard",
                input_tokens=1,
                output_tokens=1,
                cost_usd_micros=1_500,
                status="ok",
                error=None,
                attempts=1,
                ts=_now(),
            )
        )
    fake = FakeProvider("anthropic")
    gov = BudgetGovernor(db)
    router = LLMRouter(
        config=make_llm_config(),
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
        db=db,
        governor=gov,
    )
    with pytest.raises(BudgetExceeded):
        await router.complete(
            tenant_id=tenant_id,
            purpose="variant_generation",
            system="s",
            user="u",
            schema=TinySchema,
        )
    # An event row was emitted on the way out.
    with bound_tenant(tenant_id):
        events = await EventsRepo(db).list_for_tenant(type="llm.budget_exhausted")
    assert len(events) == 1
    assert events[0].payload.get("purpose") == "variant_generation"


async def test_router_without_governor_is_unrestricted(db: Database) -> None:
    """Backward-compat: routers without a governor never gate calls."""
    tenant_id = await _seed_tenant(db, daily_llm_budget_usd_micros=1_000)
    with bound_tenant(tenant_id):
        await LLMCallsRepo(db).record(
            LLMCallRow(
                id="llm_seed",
                tenant_id=tenant_id,
                purpose="variant_generation",
                provider="anthropic",
                model="m",
                quality="standard",
                input_tokens=1,
                output_tokens=1,
                cost_usd_micros=99_999,
                status="ok",
                error=None,
                attempts=1,
                ts=_now(),
            )
        )
    fake = FakeProvider("anthropic")
    fake.queue_success({"answer": "ok"})
    router = LLMRouter(
        config=make_llm_config(),
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
        # no governor -> no enforcement
    )
    res = await router.complete(
        tenant_id=tenant_id,
        purpose="variant_generation",
        system="s",
        user="u",
        schema=TinySchema,
    )
    assert res.answer == "ok"


# Silence the unused-import warning that comes from the asyncio import in
# this module while the tests above don't currently spawn extra tasks.
_ = asyncio
