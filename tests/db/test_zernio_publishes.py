"""ZernioPublishesRepo — local, tenant-scoped publish history.

The whole reason this table exists: Zernio's GET /posts is scoped to
the company API key, so per-tenant history MUST come from local rows.
These tests pin the isolation (tenant A never sees tenant B's posts)
and the idempotent record() the duplicate-resolved publish path needs.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from chalybclip.db import Database, TenantsRepo, ZernioPublishesRepo
from chalybclip.db.migrations import apply_migrations
from chalybclip.tenancy import bound_tenant


@pytest.fixture
async def db(tmp_path: Path):
    db = Database(tmp_path / "zp.db")
    await db.connect()
    await apply_migrations(db)
    for tid in ("ten_A", "ten_B"):
        with bound_tenant(tid):
            await TenantsRepo(db).create(tenant_id=tid, name=tid)
    yield db
    await db.close()


@pytest.mark.asyncio
async def test_record_and_list_roundtrip(db: Database) -> None:
    repo = ZernioPublishesRepo(db)
    await repo.record(
        post_id="post_1",
        tenant_id="ten_A",
        clip_id="clp_1",
        platforms=["instagram", "tiktok"],
        content="Hola mundo",
    )
    with bound_tenant("ten_A"):
        rows = await repo.list_for_tenant()
    assert len(rows) == 1
    row = rows[0]
    assert row.post_id == "post_1"
    assert row.clip_id == "clp_1"
    assert row.platforms == "instagram,tiktok"
    assert row.content == "Hola mundo"
    assert row.created_at  # stamped


@pytest.mark.asyncio
async def test_history_is_tenant_isolated(db: Database) -> None:
    """Tenant A must NEVER see tenant B's publishes — the bug this
    table fixes (the raw Zernio feed was company-wide)."""
    repo = ZernioPublishesRepo(db)
    await repo.record(
        post_id="post_a", tenant_id="ten_A", clip_id="clp_a",
        platforms=["instagram"], content=None,
    )
    await repo.record(
        post_id="post_b", tenant_id="ten_B", clip_id="clp_b",
        platforms=["tiktok"], content=None,
    )
    with bound_tenant("ten_A"):
        rows_a = await repo.list_for_tenant()
    with bound_tenant("ten_B"):
        rows_b = await repo.list_for_tenant()
    assert [r.post_id for r in rows_a] == ["post_a"]
    assert [r.post_id for r in rows_b] == ["post_b"]


@pytest.mark.asyncio
async def test_record_is_idempotent_on_post_id(db: Database) -> None:
    """The duplicate-resolved publish path (Zernio 409 → existing post)
    records the same post id again — must not error or duplicate."""
    repo = ZernioPublishesRepo(db)
    for _ in range(3):
        await repo.record(
            post_id="post_dup", tenant_id="ten_A", clip_id="clp_1",
            platforms=["instagram"], content="x",
        )
    with bound_tenant("ten_A"):
        rows = await repo.list_for_tenant()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_exists_for_clip_ignores_cancelled_rows(db: Database) -> None:
    """A cancelled post must NOT block the clip from being re-scheduled —
    the tombstone used to black-hole the clip permanently (every
    scheduling surface skipped it, with no UI path back)."""
    repo = ZernioPublishesRepo(db)
    await repo.record(
        post_id="post_c1", tenant_id="ten_A", clip_id="clp_9",
        platforms=["tiktok"], content=None, status="scheduled",
    )
    assert await repo.exists_for_clip("ten_A", "clp_9") is True

    await repo.set_status("post_c1", status="cancelled")
    assert await repo.exists_for_clip("ten_A", "clp_9") is False


@pytest.mark.asyncio
async def test_count_by_platform_today_skips_cancelled_and_future(
    db: Database,
) -> None:
    """The daily-cap accounting: a cancelled post frees its slot (that's
    the point of cancelling to make room), and a post scheduled for a
    FUTURE day doesn't consume today's cap."""
    import datetime as _dt

    repo = ZernioPublishesRepo(db)
    now = _dt.datetime.now(_dt.UTC)
    today = now.isoformat()
    tomorrow = (now + _dt.timedelta(days=1)).isoformat()

    await repo.record(
        post_id="p_live", tenant_id="ten_A", clip_id="clp_a",
        platforms=["tiktok"], content=None, status="scheduled",
        scheduled_for=today,
    )
    await repo.record(
        post_id="p_tomorrow", tenant_id="ten_A", clip_id="clp_b",
        platforms=["tiktok"], content=None, status="scheduled",
        scheduled_for=tomorrow,
    )
    await repo.record(
        post_id="p_cancel", tenant_id="ten_A", clip_id="clp_c",
        platforms=["tiktok"], content=None, status="scheduled",
        scheduled_for=today,
    )
    await repo.set_status("p_cancel", status="cancelled")

    counts = await repo.count_by_platform_today("ten_A")
    assert counts.get("tiktok", 0) == 1  # only today's live post


@pytest.mark.asyncio
async def test_count_by_platform_by_day_buckets_future_days(
    db: Database,
) -> None:
    """The cross-sweep cap accounting: FUTURE days a previous planning run
    already rolled overflow onto must be visible to the next run, per
    platform per day, with cancelled rows freeing their slot and platform
    ids canonicalized — else two sweeps stack 2x max_per_day on a day."""
    import datetime as _dt

    repo = ZernioPublishesRepo(db)
    now = _dt.datetime.now(_dt.UTC)
    today = now.strftime("%Y-%m-%d")
    d1 = (now + _dt.timedelta(days=1)).strftime("%Y-%m-%d")
    d2 = (now + _dt.timedelta(days=2)).strftime("%Y-%m-%d")

    await repo.record(
        post_id="q_today", tenant_id="ten_A", clip_id="c1",
        platforms=["youtube"], content=None, status="scheduled",
        scheduled_for=now.isoformat(),
    )
    for i in range(2):
        await repo.record(
            post_id=f"q_d1_{i}", tenant_id="ten_A", clip_id=f"c{2 + i}",
            platforms=["shorts"],  # alias → canonical "youtube"
            content=None, status="scheduled",
            scheduled_for=(now + _dt.timedelta(days=1)).isoformat(),
        )
    await repo.record(
        post_id="q_d2", tenant_id="ten_A", clip_id="c4",
        platforms=["youtube"], content=None, status="scheduled",
        scheduled_for=(now + _dt.timedelta(days=2)).isoformat(),
    )
    await repo.set_status("q_d2", status="cancelled")  # frees its slot
    # Another tenant's rows never leak into this tenant's day counts.
    await repo.record(
        post_id="q_other", tenant_id="ten_B", clip_id="cb",
        platforms=["youtube"], content=None, status="scheduled",
        scheduled_for=(now + _dt.timedelta(days=1)).isoformat(),
    )

    by_day = await repo.count_by_platform_by_day("ten_A", from_day=today)
    assert by_day.get("youtube", {}).get(today) == 1
    assert by_day.get("youtube", {}).get(d1) == 2
    assert d2 not in by_day.get("youtube", {})  # cancelled → not counted


@pytest.mark.asyncio
async def test_list_for_calendar_windows_by_publish_day(db: Database) -> None:
    """The calendar window filters on the day a post goes out
    (scheduled_for, else created_at) IN SQL — no newest-N slice that
    could drop older rows — and never returns drafts or tombstones."""
    repo = ZernioPublishesRepo(db)
    await repo.record(
        post_id="w_in", tenant_id="ten_A", clip_id="c_in",
        platforms=["tiktok"], content=None, status="scheduled",
        scheduled_for="2026-06-15T10:00:00+00:00",
    )
    await repo.record(
        post_id="w_out", tenant_id="ten_A", clip_id="c_out",
        platforms=["tiktok"], content=None, status="scheduled",
        scheduled_for="2026-07-15T10:00:00+00:00",
    )
    await repo.record(
        post_id="w_draft", tenant_id="ten_A", clip_id="c_d",
        platforms=["tiktok"], content=None, status="draft",
        scheduled_for="2026-06-16T10:00:00+00:00",
    )
    await repo.record(  # no scheduled_for → windows on created_at (now)
        post_id="w_now", tenant_id="ten_A", clip_id="c_n",
        platforms=["tiktok"], content=None,
    )
    with bound_tenant("ten_A"):
        june = await repo.list_for_calendar(
            date_from="2026-06-01", date_to="2026-06-30",
        )
    assert [r.post_id for r in june] == ["w_in"]
    with bound_tenant("ten_A"):
        unbounded = await repo.list_for_calendar()
    ids = {r.post_id for r in unbounded}
    assert {"w_in", "w_out", "w_now"} <= ids
    assert "w_draft" not in ids


@pytest.mark.asyncio
async def test_count_status_groups_exact_and_manual_as_published(
    db: Database,
) -> None:
    """The stats strip counts REAL totals per status. Rows with no status
    land under ''; synthetic manual_<clip_id> marks count as 'published'
    even though they're stored statusless (no webhook ever updates them)."""
    repo = ZernioPublishesRepo(db)
    for i in range(3):
        await repo.record(
            post_id=f"s_sched_{i}", tenant_id="ten_A", clip_id=f"cs{i}",
            platforms=["tiktok"], content=None, status="scheduled",
        )
    await repo.record(
        post_id="s_pub", tenant_id="ten_A", clip_id="cp",
        platforms=["tiktok"], content=None, status="published",
    )
    await repo.record(  # statusless immediate publish → '' bucket
        post_id="s_none", tenant_id="ten_A", clip_id="cn",
        platforms=["tiktok"], content=None,
    )
    await repo.record(  # manual mark: statusless but counts as published
        post_id="manual_cm", tenant_id="ten_A", clip_id="cm",
        platforms=[], content=None,
    )
    await repo.record(  # other tenants never leak into the counts
        post_id="s_other", tenant_id="ten_B", clip_id="cb",
        platforms=["tiktok"], content=None, status="scheduled",
    )
    with bound_tenant("ten_A"):
        groups = await repo.count_status_groups()
    assert groups.get("scheduled") == 3
    assert groups.get("published") == 2  # explicit row + manual mark
    assert groups.get("") == 1


@pytest.mark.asyncio
async def test_recorded_clip_ids_is_uncapped_membership(db: Database) -> None:
    repo = ZernioPublishesRepo(db)
    await repo.record(
        post_id="m_1", tenant_id="ten_A", clip_id="clp_x",
        platforms=["tiktok"], content=None,
    )
    await repo.record(
        post_id="m_2", tenant_id="ten_B", clip_id="clp_y",
        platforms=["tiktok"], content=None,
    )
    with bound_tenant("ten_A"):
        ids = await repo.recorded_clip_ids()
    assert ids == {"clp_x"}
