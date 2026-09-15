"""Integration test for the Postgres backend (migration runner + facade).

Skipped unless ``CHALYBCLIP_TEST_PG_DSN`` points at a disposable Postgres
database (e.g. ``postgresql://postgres@127.0.0.1:5433/chalybclip_test``). The
target database is mutated, so never point this at production data.

This is the seed of the Phase 4 "tests on real Postgres" work; Phase 4
generalizes it to per-test isolation via testcontainers / a CI service.
"""

from __future__ import annotations

import os

import pytest

from chalybclip.db import Database, apply_migrations
from chalybclip.db.migrations import _pg_baseline_version

_DSN = os.environ.get("CHALYBCLIP_TEST_PG_DSN")
pytestmark = pytest.mark.skipif(
    not _DSN, reason="set CHALYBCLIP_TEST_PG_DSN to run Postgres integration tests"
)


@pytest.mark.asyncio
async def test_apply_baseline_and_roundtrip() -> None:
    db = Database(_DSN)  # type: ignore[arg-type]
    assert db.is_postgres is True
    try:
        version = await apply_migrations(db)
        assert version == _pg_baseline_version()

        # Idempotent: a second run is a no-op at the same version.
        assert await apply_migrations(db) == version

        conn = await db.connect()
        # The full object set is present.
        cur = await conn.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        )
        assert (await cur.fetchone())[0] >= 42

        # A `?`-parameterized write/read roundtrips through the facade, and
        # the row is dict-accessible the way the repos' _model() expects.
        await conn.execute(
            "INSERT INTO tenants (id, name, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT (id) DO NOTHING",
            ("ten_pgtest", "PG Test", "2026-06-16T00:00:00Z"),
        )
        cur = await conn.execute(
            "SELECT id, name FROM tenants WHERE id = ?", ("ten_pgtest",)
        )
        row = await cur.fetchone()
        assert dict(row) == {"id": "ten_pgtest", "name": "PG Test"}

        # rowcount comes from the asyncpg command tag.
        cur = await conn.execute(
            "DELETE FROM tenants WHERE id = ?", ("ten_pgtest",)
        )
        assert cur.rowcount == 1
    finally:
        await db.close()
