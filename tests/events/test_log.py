"""Tests for the `emit()` event-log helper."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from chalybclip.db import Database, EventsRepo, TenantsRepo, apply_migrations
from chalybclip.errors import TenancyError
from chalybclip.events import (
    CLIP_READY_FOR_REVIEW,
    LLM_EXHAUSTED,
    LLM_FALLBACK,
    STREAM_CREATED,
    STREAM_PROCESSED,
    emit,
)
from chalybclip.tenancy import bound_tenant


def test_canonical_event_types_have_stable_strings() -> None:
    # Schema-version-equivalent contract: these strings persist in the DB
    # forever once emitted, so renaming them silently is forbidden.
    assert STREAM_CREATED == "stream.created"
    assert STREAM_PROCESSED == "stream.processed"
    assert CLIP_READY_FOR_REVIEW == "clip.ready_for_review"
    assert LLM_FALLBACK == "llm.fallback"
    assert LLM_EXHAUSTED == "llm.exhausted"


def test_emit_with_db_writes_row(tmp_path: Path) -> None:
    async def go() -> None:
        db = Database(tmp_path / "test.db")
        await apply_migrations(db)
        await TenantsRepo(db).create(tenant_id="ten_a", name="A")
        with bound_tenant("ten_a"):
            await emit(db, STREAM_CREATED, {"stream_id": "str_1"})
            rows = await EventsRepo(db).list_for_tenant()
        assert len(rows) == 1
        assert rows[0].type == STREAM_CREATED
        assert rows[0].payload == {"stream_id": "str_1"}
        await db.close()

    asyncio.run(go())


def test_emit_with_db_none_is_noop(tmp_path: Path) -> None:
    """Phase 0-style invocations (no DB) silently skip — no error."""

    async def go() -> None:
        await emit(None, STREAM_CREATED, {"x": 1})

    asyncio.run(go())  # no exception


def test_emit_swallows_failures(tmp_path: Path) -> None:
    """A bad payload (or unbound tenant) doesn't propagate up to the pipeline."""

    async def go() -> None:
        db = Database(tmp_path / "test.db")
        await apply_migrations(db)
        # No tenant bound -> EventsRepo.emit will raise TenancyError, but
        # `emit()` swallows it.
        await emit(db, STREAM_CREATED, {"x": 1})
        await db.close()

    asyncio.run(go())  # must not raise


def test_emit_uses_current_tenant_not_caller_arg(tmp_path: Path) -> None:
    """Tenant comes from contextvars, not a kwarg — defense in depth."""

    async def go() -> tuple[list, list]:
        db = Database(tmp_path / "test.db")
        await apply_migrations(db)
        await TenantsRepo(db).create(tenant_id="ten_a", name="A")
        await TenantsRepo(db).create(tenant_id="ten_b", name="B")
        with bound_tenant("ten_a"):
            await emit(db, STREAM_CREATED, {"x": 1})
        with bound_tenant("ten_b"):
            await emit(db, STREAM_CREATED, {"x": 2})
            b = await EventsRepo(db).list_for_tenant()
        with bound_tenant("ten_a"):
            a = await EventsRepo(db).list_for_tenant()
        await db.close()
        return a, b

    a_events, b_events = asyncio.run(go())
    assert len(a_events) == 1 and a_events[0].payload == {"x": 1}
    assert len(b_events) == 1 and b_events[0].payload == {"x": 2}
