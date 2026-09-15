"""PublishJobsRepo.list_recent_for_tenant — used by `chalybclip queue list`."""

from __future__ import annotations

import datetime as _dt

import pytest

from chalybclip.db import (
    CandidatesRepo,
    ClipsRepo,
    ConnectedAccountsRepo,
    Database,
    PersonasRepo,
    PublishJobsRepo,
    StreamsRepo,
    TenantsRepo,
    VariantsRepo,
)
from chalybclip.db.models import (
    CandidateRow,
    ClipRow,
    StreamRow,
    VariantRow,
)
from chalybclip.errors import TenancyError
from chalybclip.tenancy import bound_tenant


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


async def _seed_account(db: Database, tenant_id: str) -> str:
    """Minimum scaffolding so a publish_job FK chain holds."""
    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_q",
                tenant_id=tenant_id,
                vod_url="x",
                platform="kick",
                title=None,
                channel=None,
                duration_s=60.0,
                source_video_path="/tmp/v",
                source_audio_path="/tmp/a",
                status="ingested",
                created_at=_now(),
            )
        )
        await CandidatesRepo(db).upsert_many(
            [
                CandidateRow(
                    id="cnd_q",
                    stream_id="str_q",
                    tenant_id=tenant_id,
                    ts=10.0,
                    score=0.5,
                    reason="voice",
                    evidence={},
                    created_at=_now(),
                )
            ]
        )
        await ClipsRepo(db).upsert_many(
            [
                ClipRow(
                    id="clp_q",
                    stream_id="str_q",
                    tenant_id=tenant_id,
                    candidate_id="cnd_q",
                    start_s=0.0,
                    end_s=10.0,
                    duration_s=10.0,
                    width=1080,
                    height=1920,
                    path="/tmp/c.mp4",
                    status="cut",
                    created_at=_now(),
                )
            ]
        )
        await PersonasRepo(db).create(
            persona_id="p1",
            name="P",
            primary_language="es",
            target_languages=["es"],
            voice_prompt="v",
        )
        await VariantsRepo(db).replace_for_clip_persona(
            "clp_q",
            "p1",
            [
                VariantRow(
                    id="var_q",
                    clip_id="clp_q",
                    tenant_id=tenant_id,
                    persona_id="p1",
                    language="es",
                    caption="c",
                    title_card_text="",
                    hashtags=[],
                    model=None,
                    created_at=_now(),
                )
            ],
        )
        acct = await ConnectedAccountsRepo(db).create(
            platform="tiktok", external_id="u"
        )
    return acct.id


async def test_list_recent_no_status_filter_returns_all(migrated_db: Database) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="A")
    account_id = await _seed_account(migrated_db, tenant.id)
    with bound_tenant(tenant.id):
        repo = PublishJobsRepo(migrated_db)
        for _ in range(3):
            await repo.enqueue(
                clip_id="clp_q",
                variant_id="var_q",
                account_id=account_id,
                platform="tiktok",
            )
        rows = await repo.list_recent_for_tenant(limit=10)
    assert len(rows) == 3


async def test_list_recent_filters_by_status(migrated_db: Database) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="A")
    account_id = await _seed_account(migrated_db, tenant.id)
    with bound_tenant(tenant.id):
        repo = PublishJobsRepo(migrated_db)
        a = await repo.enqueue(
            clip_id="clp_q", variant_id="var_q", account_id=account_id, platform="tiktok"
        )
        b = await repo.enqueue(
            clip_id="clp_q", variant_id="var_q", account_id=account_id, platform="tiktok"
        )
        # Flip a -> sent, b -> failed.
        conn = await migrated_db.connect()
        await conn.execute("UPDATE publish_jobs SET status='sent' WHERE id=?", (a.id,))
        await conn.execute(
            "UPDATE publish_jobs SET status='failed', last_error='timeout' WHERE id=?",
            (b.id,),
        )
        await conn.commit()

        sent = await repo.list_recent_for_tenant(status="sent", limit=10)
        failed = await repo.list_recent_for_tenant(status="failed", limit=10)
        pending = await repo.list_recent_for_tenant(status="pending", limit=10)
    assert {r.id for r in sent} == {a.id}
    assert {r.id for r in failed} == {b.id}
    assert pending == []


async def test_list_recent_returns_newest_first(migrated_db: Database) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="A")
    account_id = await _seed_account(migrated_db, tenant.id)
    with bound_tenant(tenant.id):
        repo = PublishJobsRepo(migrated_db)
        # Enqueue three with deliberately spaced created_at via direct SQL
        # (the repo's _now() is sub-millisecond on a fast machine and the
        # test's intent is just "ordering by created_at desc works").
        ids = []
        for i in range(3):
            j = await repo.enqueue(
                clip_id="clp_q",
                variant_id="var_q",
                account_id=account_id,
                platform="tiktok",
            )
            ids.append(j.id)
            conn = await migrated_db.connect()
            await conn.execute(
                "UPDATE publish_jobs SET created_at = ? WHERE id = ?",
                (f"2026-05-08T0{i}:00:00+00:00", j.id),
            )
            await conn.commit()
        rows = await repo.list_recent_for_tenant(limit=10)
    # Newest (i=2) -> first; oldest (i=0) -> last.
    assert [r.id for r in rows] == [ids[2], ids[1], ids[0]]


async def test_list_recent_isolated_per_tenant(migrated_db: Database) -> None:
    alice = await TenantsRepo(migrated_db).create(name="Alice")
    bob = await TenantsRepo(migrated_db).create(name="Bob")
    aid = await _seed_account(migrated_db, alice.id)
    with bound_tenant(alice.id):
        await PublishJobsRepo(migrated_db).enqueue(
            clip_id="clp_q", variant_id="var_q", account_id=aid, platform="tiktok"
        )
    with bound_tenant(bob.id):
        rows = await PublishJobsRepo(migrated_db).list_recent_for_tenant()
    assert rows == []


async def test_list_recent_requires_bound_tenant(migrated_db: Database) -> None:
    with pytest.raises(TenancyError, match="no tenant bound"):
        await PublishJobsRepo(migrated_db).list_recent_for_tenant()
