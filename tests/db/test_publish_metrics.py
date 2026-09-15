"""PublishMetricsRepo — record / latest_for_job / list_for_job / calibration query."""

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
    PublishMetricsRepo,
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


async def _seed_publish_job(
    db: Database, tenant_id: str, *, job_suffix: str = "a", platform: str = "youtube"
) -> str:
    """Minimal scaffolding so a publish_metrics row can FK in."""
    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=f"str_m_{job_suffix}",
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
                    id=f"cnd_m_{job_suffix}",
                    stream_id=f"str_m_{job_suffix}",
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
                    id=f"clp_m_{job_suffix}",
                    stream_id=f"str_m_{job_suffix}",
                    tenant_id=tenant_id,
                    candidate_id=f"cnd_m_{job_suffix}",
                    start_s=0.0,
                    end_s=10.0,
                    duration_s=10.0,
                    width=1080,
                    height=1920,
                    path="/tmp/c.mp4",
                    status="approved",
                    created_at=_now(),
                )
            ]
        )
        # Reuse a single persona+account across calls within the same tenant.
        if await PersonasRepo(db).get("p1") is None:
            await PersonasRepo(db).create(
                persona_id="p1",
                name="P",
                primary_language="es",
                target_languages=["es"],
                voice_prompt="v",
            )
            acct = await ConnectedAccountsRepo(db).create(
                platform=platform, external_id="u"
            )
            account_id = acct.id
        else:
            accts = await ConnectedAccountsRepo(db).list_for_tenant()
            account_id = accts[0].id
        await VariantsRepo(db).replace_for_clip_persona(
            f"clp_m_{job_suffix}",
            "p1",
            [
                VariantRow(
                    id=f"var_m_{job_suffix}",
                    clip_id=f"clp_m_{job_suffix}",
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
        job = await PublishJobsRepo(db).enqueue(
            clip_id=f"clp_m_{job_suffix}",
            variant_id=f"var_m_{job_suffix}",
            account_id=account_id,
            platform=platform,
        )
    return job.id


# ---- record + read-back ----


async def test_record_round_trips_all_fields(migrated_db: Database) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="A")
    job_id = await _seed_publish_job(migrated_db, tenant.id)
    with bound_tenant(tenant.id):
        repo = PublishMetricsRepo(migrated_db)
        metric = await repo.record(
            publish_job_id=job_id,
            platform="youtube",
            fetched_at="2026-05-07T12:00:00+00:00",
            views=1500,
            likes=42,
            comments=7,
            shares=3,
            retention_pct=0.65,
            ctr=0.04,
            raw_metadata={"snippet": "...", "stats": {"plays": 1500}},
        )
    assert metric.id.startswith("met_")
    assert metric.views == 1500
    assert metric.retention_pct == 0.65
    assert metric.raw_metadata is not None
    assert metric.raw_metadata.get("snippet") == "..."


async def test_record_supports_partial_metrics(migrated_db: Database) -> None:
    """Some platforms only expose a subset of stats - NULLs are valid."""
    tenant = await TenantsRepo(migrated_db).create(name="A")
    job_id = await _seed_publish_job(migrated_db, tenant.id, platform="tiktok")
    with bound_tenant(tenant.id):
        m = await PublishMetricsRepo(migrated_db).record(
            publish_job_id=job_id,
            platform="tiktok",
            fetched_at="2026-05-07T12:00:00+00:00",
            views=200,
        )
    assert m.views == 200
    assert m.likes is None
    assert m.retention_pct is None
    assert m.ctr is None


# ---- latest_for_job ----


async def test_latest_for_job_returns_most_recent_row(migrated_db: Database) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="A")
    job_id = await _seed_publish_job(migrated_db, tenant.id)
    with bound_tenant(tenant.id):
        repo = PublishMetricsRepo(migrated_db)
        await repo.record(
            publish_job_id=job_id,
            platform="youtube",
            fetched_at="2026-05-07T08:00:00+00:00",
            views=100,
        )
        await repo.record(
            publish_job_id=job_id,
            platform="youtube",
            fetched_at="2026-05-07T14:00:00+00:00",
            views=300,
        )
        await repo.record(
            publish_job_id=job_id,
            platform="youtube",
            fetched_at="2026-05-07T11:00:00+00:00",
            views=200,
        )
        latest = await repo.latest_for_job(job_id)
    assert latest is not None
    assert latest.views == 300


async def test_latest_for_job_none_when_no_metrics(migrated_db: Database) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="A")
    job_id = await _seed_publish_job(migrated_db, tenant.id)
    with bound_tenant(tenant.id):
        latest = await PublishMetricsRepo(migrated_db).latest_for_job(job_id)
    assert latest is None


# ---- list_for_job (time series) ----


async def test_list_for_job_returns_chronological_series(migrated_db: Database) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="A")
    job_id = await _seed_publish_job(migrated_db, tenant.id)
    timestamps = [
        "2026-05-07T08:00:00+00:00",
        "2026-05-07T14:00:00+00:00",
        "2026-05-07T11:00:00+00:00",
    ]
    with bound_tenant(tenant.id):
        repo = PublishMetricsRepo(migrated_db)
        for ts in timestamps:
            await repo.record(
                publish_job_id=job_id,
                platform="youtube",
                fetched_at=ts,
                views=100,
            )
        series = await repo.list_for_job(job_id)
    assert [m.fetched_at for m in series] == sorted(timestamps)


# ---- latest_per_job_since (calibration query) ----


async def test_latest_per_job_since_collapses_to_one_per_job(
    migrated_db: Database,
) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="A")
    job_a = await _seed_publish_job(migrated_db, tenant.id, job_suffix="a")
    job_b = await _seed_publish_job(migrated_db, tenant.id, job_suffix="b")
    with bound_tenant(tenant.id):
        repo = PublishMetricsRepo(migrated_db)
        # Two readings on job_a, one on job_b.
        await repo.record(
            publish_job_id=job_a,
            platform="youtube",
            fetched_at="2026-05-07T08:00:00+00:00",
            views=100,
        )
        await repo.record(
            publish_job_id=job_a,
            platform="youtube",
            fetched_at="2026-05-07T14:00:00+00:00",
            views=300,
        )
        await repo.record(
            publish_job_id=job_b,
            platform="youtube",
            fetched_at="2026-05-07T13:00:00+00:00",
            views=500,
        )
        results = await repo.latest_per_job_since(
            platform="youtube", since="2026-05-07T00:00:00+00:00"
        )
    by_job = {m.publish_job_id: m for m in results}
    assert len(by_job) == 2
    assert by_job[job_a].views == 300  # latest reading wins
    assert by_job[job_b].views == 500


async def test_latest_per_job_since_filters_by_platform(migrated_db: Database) -> None:
    """A YT scan must not pull TikTok readings."""
    tenant = await TenantsRepo(migrated_db).create(name="A")
    yt_job = await _seed_publish_job(
        migrated_db, tenant.id, job_suffix="yt", platform="youtube"
    )
    with bound_tenant(tenant.id):
        repo = PublishMetricsRepo(migrated_db)
        await repo.record(
            publish_job_id=yt_job,
            platform="youtube",
            fetched_at="2026-05-07T12:00:00+00:00",
            views=100,
        )
        await repo.record(
            publish_job_id=yt_job,  # same job_id but different platform string
            platform="tiktok",
            fetched_at="2026-05-07T12:00:00+00:00",
            views=200,
        )
        yt_only = await repo.latest_per_job_since(
            platform="youtube", since="2026-05-07T00:00:00+00:00"
        )
    assert all(m.platform == "youtube" for m in yt_only)
    assert len(yt_only) == 1


# ---- tenancy ----


async def test_publish_metrics_requires_bound_tenant(migrated_db: Database) -> None:
    with pytest.raises(TenancyError, match="no tenant bound"):
        await PublishMetricsRepo(migrated_db).record(
            publish_job_id="x",
            platform="youtube",
            fetched_at=_now(),
            views=1,
        )


async def test_publish_metrics_isolated_per_tenant(migrated_db: Database) -> None:
    """Bob can't see Alice's metrics even if he guesses the publish_job_id."""
    alice = await TenantsRepo(migrated_db).create(name="Alice")
    bob = await TenantsRepo(migrated_db).create(name="Bob")
    job_id = await _seed_publish_job(migrated_db, alice.id, job_suffix="alice")
    with bound_tenant(alice.id):
        await PublishMetricsRepo(migrated_db).record(
            publish_job_id=job_id,
            platform="youtube",
            fetched_at=_now(),
            views=100,
        )
    with bound_tenant(bob.id):
        latest = await PublishMetricsRepo(migrated_db).latest_for_job(job_id)
        series = await PublishMetricsRepo(migrated_db).list_for_job(job_id)
    assert latest is None
    assert series == []
