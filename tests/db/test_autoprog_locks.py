"""AutoprogLocksRepo — the cross-worker bulk auto-program lock (migration 055)."""

from __future__ import annotations

import pytest

from chalybclip.db import AutoprogLocksRepo, Database, TenantsRepo


@pytest.mark.asyncio
async def test_second_acquire_blocked_until_release(migrated_db: Database) -> None:
    await TenantsRepo(migrated_db).create(tenant_id="ten_lk", name="LK")
    repo = AutoprogLocksRepo(migrated_db)

    t1 = await repo.acquire("ten_lk")
    assert t1 is not None  # first run wins

    t2 = await repo.acquire("ten_lk")
    assert t2 is None  # second concurrent run is blocked

    await repo.release("ten_lk", t1)
    t3 = await repo.acquire("ten_lk")
    assert t3 is not None  # released → next run can take it


@pytest.mark.asyncio
async def test_stale_lock_is_reclaimed(migrated_db: Database) -> None:
    await TenantsRepo(migrated_db).create(tenant_id="ten_st", name="ST")
    repo = AutoprogLocksRepo(migrated_db)

    t1 = await repo.acquire("ten_st")
    assert t1 is not None
    # A run that "crashed" leaves a lock behind; age it past the staleness
    # window so the next acquire treats it as dead and reclaims it.
    conn = await migrated_db.connect()
    await conn.execute(
        "UPDATE autoprog_locks SET claimed_at = ? WHERE tenant_id = ?",
        ("2020-01-01T00:00:00+00:00", "ten_st"),
    )
    await conn.commit()
    t2 = await repo.acquire("ten_st")
    assert t2 is not None
    assert t2 != t1


@pytest.mark.asyncio
async def test_release_with_wrong_token_is_noop(migrated_db: Database) -> None:
    await TenantsRepo(migrated_db).create(tenant_id="ten_wr", name="WR")
    repo = AutoprogLocksRepo(migrated_db)
    t1 = await repo.acquire("ten_wr")
    await repo.release("ten_wr", "not-the-token")
    # Lock still held → a new acquire is blocked.
    assert await repo.acquire("ten_wr") is None
    await repo.release("ten_wr", t1)
