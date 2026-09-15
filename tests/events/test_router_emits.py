"""LLMRouter emits llm.fallback (per chain step) and llm.exhausted (final)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel

from chalybclip.db import Database, EventsRepo, TenantsRepo, apply_migrations
from chalybclip.errors import LLMError
from chalybclip.events import LLM_EXHAUSTED, LLM_FALLBACK
from chalybclip.llm import LLMRouter
from chalybclip.tenancy import bound_tenant

from tests.llm._fakes import FakeProvider  # type: ignore[import]
from tests.llm._fixtures import make_llm_config  # type: ignore[import]


class TinySchema(BaseModel):
    answer: str


def _factory(providers: dict[str, FakeProvider]):
    def _build(name, _config, _api_key):
        return providers.get(name)

    return _build


async def _seed_tenant(tmp_path: Path) -> Database:
    db = Database(tmp_path / "test.db")
    await apply_migrations(db)
    await TenantsRepo(db).create(tenant_id="ten_a", name="A")
    return db


def _events(db: Database, tenant_id: str) -> list:
    async def go() -> list:
        with bound_tenant(tenant_id):
            return await EventsRepo(db).list_for_tenant()

    return asyncio.run(go())


def test_router_emits_llm_fallback_when_primary_exhausts(tmp_path: Path) -> None:
    db = asyncio.run(_seed_tenant(tmp_path))
    primary = FakeProvider("anthropic")
    for _ in range(3):
        primary.queue_retryable("503")
    fallback = FakeProvider("openai")
    fallback.queue_success({"answer": "ok"})

    config = make_llm_config(
        primary="anthropic",
        fallbacks=["openai"],
        retry_attempts=3,
        initial_backoff_s=0.0,
    )
    router = LLMRouter(
        config,
        api_keys={"anthropic": "k", "openai": "k"},
        provider_factory=_factory({"anthropic": primary, "openai": fallback}),
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
    events = _events(db, "ten_a")
    asyncio.run(db.close())

    # One fallback row from the primary failure; no exhausted row because
    # the secondary succeeded.
    fb = [e for e in events if e.type == LLM_FALLBACK]
    assert len(fb) == 1
    assert fb[0].payload["provider"] == "anthropic"
    assert fb[0].payload["purpose"] == "variant_generation"
    assert "503" in fb[0].payload["error"]

    assert [e for e in events if e.type == LLM_EXHAUSTED] == []


def test_router_emits_llm_exhausted_when_all_providers_fail(tmp_path: Path) -> None:
    db = asyncio.run(_seed_tenant(tmp_path))
    primary = FakeProvider("anthropic")
    for _ in range(3):
        primary.queue_retryable("503")
    fallback = FakeProvider("openai")
    for _ in range(3):
        fallback.queue_retryable("503")

    config = make_llm_config(
        primary="anthropic",
        fallbacks=["openai"],
        retry_attempts=3,
        initial_backoff_s=0.0,
    )
    router = LLMRouter(
        config,
        api_keys={"anthropic": "k", "openai": "k"},
        provider_factory=_factory({"anthropic": primary, "openai": fallback}),
        db=db,
    )

    async def go() -> None:
        with bound_tenant("ten_a"):
            with pytest.raises(LLMError, match="all providers failed"):
                await router.complete(
                    tenant_id="ten_a",
                    purpose="variant_generation",
                    system="s",
                    user="u",
                    schema=TinySchema,
                )

    asyncio.run(go())
    events = _events(db, "ten_a")
    asyncio.run(db.close())

    # One fallback (primary -> secondary), one exhausted (secondary failed too).
    assert [e.type for e in events if e.type == LLM_FALLBACK] == [LLM_FALLBACK]
    exhausted = [e for e in events if e.type == LLM_EXHAUSTED]
    assert len(exhausted) == 1
    assert exhausted[0].payload["providers_tried"] == ["anthropic", "openai"]
    assert exhausted[0].payload["purpose"] == "variant_generation"


def test_router_no_fallback_event_when_only_one_provider(tmp_path: Path) -> None:
    """If there's no provider after the failing one, no llm.fallback row;
    only llm.exhausted at the end."""
    db = asyncio.run(_seed_tenant(tmp_path))
    primary = FakeProvider("anthropic")
    for _ in range(3):
        primary.queue_retryable("503")

    config = make_llm_config(
        primary="anthropic",
        fallbacks=[],  # no fallbacks
        retry_attempts=3,
        initial_backoff_s=0.0,
    )
    router = LLMRouter(
        config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": primary}),
        db=db,
    )

    async def go() -> None:
        with bound_tenant("ten_a"):
            with pytest.raises(LLMError):
                await router.complete(
                    tenant_id="ten_a",
                    purpose="variant_generation",
                    system="s",
                    user="u",
                    schema=TinySchema,
                )

    asyncio.run(go())
    events = _events(db, "ten_a")
    asyncio.run(db.close())

    assert [e for e in events if e.type == LLM_FALLBACK] == []
    assert [e.type for e in events if e.type == LLM_EXHAUSTED] == [LLM_EXHAUSTED]


def test_router_without_db_emits_no_events(tmp_path: Path) -> None:
    """Phase 0 backward-compat: db=None => no event side effects."""
    primary = FakeProvider("anthropic")
    for _ in range(3):
        primary.queue_retryable("503")
    config = make_llm_config(retry_attempts=3, initial_backoff_s=0.0)
    router = LLMRouter(
        config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": primary}),
        db=None,
    )

    async def go() -> None:
        with pytest.raises(LLMError):
            await router.complete(
                tenant_id="ten_a",
                purpose="variant_generation",
                system="s",
                user="u",
                schema=TinySchema,
            )

    asyncio.run(go())  # no DB, no failure
