"""Shutdown-ordering regression — close() vs fire-and-forget writers.

Prod failure (Modal worker, 2026-07-15, str_01KXKGMM1WEKZB8CF4STGKC01C):
right after "pipeline.done", a background write (a queued step event / a
usage report's follow-up status write) died with asyncpg
InterfaceError("pool is closed") — the run's `db_session` teardown closed
the pool while those tasks were still pending, silently dropping the run's
last DB writes.

Two guarantees pinned here:

  1. `Database.close()` drains tasks registered via
     `track_background_task()` BEFORE tearing down the backend, so a late
     write lands instead of dying (bounded by `_CLOSE_DRAIN_TIMEOUT_S`).
  2. `close()` detaches the pool/connection handle BEFORE awaiting its
     close — asyncpg's Pool.close() yields while waiting for acquired
     connections, and a straggler calling connect() in that window must
     never be handed the closing pool.

Plus the spawner side: the pipeline's Postgres-path step-event emit
registers its task with the owning Database.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import structlog

from chalybclip.db import connection as connection_mod
from chalybclip.db.connection import Database


async def test_close_waits_for_tracked_background_write(tmp_path: Path) -> None:
    """A registered in-flight write completes before close() returns, and
    its row is durable in the database file."""
    path = tmp_path / "t.db"
    db = Database(path)
    conn = await db.connect()
    await conn.execute("CREATE TABLE notes (body TEXT)")
    await conn.commit()

    async def late_write() -> None:
        # Still sleeping when close() begins — the drain must wait it out.
        await asyncio.sleep(0.05)
        c = await db.connect()
        await c.execute("INSERT INTO notes (body) VALUES (?)", ("late",))
        await c.commit()

    task = asyncio.create_task(late_write())
    db.track_background_task(task)
    await db.close()

    assert task.done()
    assert task.exception() is None
    assert db._bg_tasks == set()  # done-callback hygiene

    check = Database(path)
    try:
        c = await check.connect()
        cur = await c.execute("SELECT COUNT(*) FROM notes")
        row = await cur.fetchone()
        assert row is not None and row[0] == 1
    finally:
        await check.close()


class _StubPool:
    """Stands in for an asyncpg pool whose close() blocks mid-await, the
    window in which the prod race fired."""

    def __init__(self) -> None:
        self.close_started = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

    async def close(self) -> None:
        self.close_started.set()
        await self.release.wait()
        self.closed = True


async def test_pg_pool_detached_before_close_await() -> None:
    """While Pool.close() is still awaiting, the handle must already be
    detached — so a concurrent connect() can never receive the closing pool
    (the InterfaceError("pool is closed") race)."""
    db = Database("postgresql://u:p@db.invalid:5432/x")
    stub = _StubPool()
    db._pool = stub  # type: ignore[assignment]

    close_task = asyncio.create_task(db.close())
    await asyncio.wait_for(stub.close_started.wait(), timeout=2.0)
    assert db._pool is None
    stub.release.set()
    await asyncio.wait_for(close_task, timeout=2.0)
    assert stub.closed


async def test_close_drain_timeout_leaves_straggler_running(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A writer that outlives the drain budget (e.g. a usage report riding
    its retry backoff) must not wedge close() — it's left running, not
    cancelled."""
    monkeypatch.setattr(connection_mod, "_CLOSE_DRAIN_TIMEOUT_S", 0.05)
    db = Database(tmp_path / "t.db")
    await db.connect()

    async def straggler() -> None:
        await asyncio.sleep(30)

    task = asyncio.create_task(straggler())
    db.track_background_task(task)
    await asyncio.wait_for(db.close(), timeout=2.0)

    assert not task.done()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


class _StubPgDb:
    """Database-shaped stub: Postgres-flagged, records registrations, and
    accepts EventsRepo's INSERT so the emit task can run to completion."""

    is_postgres = True

    def __init__(self) -> None:
        self.tracked: list[asyncio.Task[Any]] = []
        self.executed: list[str] = []

    def track_background_task(self, task: asyncio.Task[Any]) -> None:
        self.tracked.append(task)

    async def connect(self) -> _StubPgDb:
        return self

    async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        self.executed.append(sql)

    async def commit(self) -> None:
        return None


async def test_step_event_pg_task_registered_with_db() -> None:
    """The Postgres-path step-event emit registers its fire-and-forget task
    with the owning Database, so db.close() drains it."""
    from chalybclip.pipeline import _record_step_event
    from chalybclip.tenancy import bound_tenant

    db = _StubPgDb()
    structlog.contextvars.bind_contextvars(tenant_id="ten_x")
    try:
        with bound_tenant("ten_x"):
            _record_step_event(
                db, "pipeline.step.done", "cut", "str_x", {"duration_s": 1.0}
            )
    finally:
        structlog.contextvars.clear_contextvars()

    assert len(db.tracked) == 1
    await asyncio.gather(*db.tracked)
    assert any("INSERT INTO events" in sql for sql in db.executed)
