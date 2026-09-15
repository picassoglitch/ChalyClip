"""Calibration loop — Pearson r over (rescore_score, views) per platform."""

from __future__ import annotations

import datetime as _dt
from collections.abc import AsyncIterator
from pathlib import Path

import pytest_asyncio

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
    apply_migrations,
)
from chalybclip.db.models import (
    CandidateRow,
    ClipRow,
    StreamRow,
    VariantRow,
)
from chalybclip.metrics import compute_calibration
from chalybclip.tenancy import bound_tenant


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "calib.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


async def _seed_published(
    db: Database,
    tenant_id: str,
    *,
    suffix: str,
    rescore_score: float | None,
    views: int | None,
    platform: str = "youtube",
) -> str:
    """Seed one stream/candidate/clip/variant/publish_job + one metric row."""
    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=f"str_{suffix}",
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
                    id=f"cnd_{suffix}",
                    stream_id=f"str_{suffix}",
                    tenant_id=tenant_id,
                    ts=10.0,
                    score=0.5,
                    reason="voice",
                    evidence={},
                    created_at=_now(),
                )
            ]
        )
        if rescore_score is not None:
            await CandidatesRepo(db).update_rescore(
                f"cnd_{suffix}",
                rescore_score=rescore_score,
                rescore_reason="r",
                rescore_model="m",
            )
        await ClipsRepo(db).upsert_many(
            [
                ClipRow(
                    id=f"clp_{suffix}",
                    stream_id=f"str_{suffix}",
                    tenant_id=tenant_id,
                    candidate_id=f"cnd_{suffix}",
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
        if await PersonasRepo(db).get("p1") is None:
            await PersonasRepo(db).create(
                persona_id="p1",
                name="P",
                primary_language="es",
                target_languages=["es"],
                voice_prompt="v",
            )
        accts = await ConnectedAccountsRepo(db).list_for_tenant()
        existing = next((a for a in accts if a.platform == platform), None)
        if existing is None:
            account = await ConnectedAccountsRepo(db).create(
                platform=platform, external_id="u"
            )
        else:
            account = existing
        await VariantsRepo(db).replace_for_clip_persona(
            f"clp_{suffix}",
            "p1",
            [
                VariantRow(
                    id=f"var_{suffix}",
                    clip_id=f"clp_{suffix}",
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
            clip_id=f"clp_{suffix}",
            variant_id=f"var_{suffix}",
            account_id=account.id,
            platform=platform,
        )
        # Always record a metric row — views may legitimately be NULL when
        # the platform's analytics API returned 200 but empty stats. That's
        # an audit row, not the absence of one (the service-layer fetcher
        # records empty rows on 4xx for the same reason).
        await PublishMetricsRepo(db).record(
            publish_job_id=job.id,
            platform=platform,
            fetched_at=_now(),
            views=views,
        )
    return job.id


async def test_calibration_no_data_returns_empty_report(db: Database) -> None:
    tenant = await TenantsRepo(db).create(name="A")
    with bound_tenant(tenant.id):
        report = await compute_calibration(db, platform="youtube")
    assert report.rows == []
    assert report.pearson_r is None
    assert report.n_paired == 0


async def test_calibration_perfect_correlation_yields_r_one(db: Database) -> None:
    """Linear (rescore, views) -> Pearson r == 1.0."""
    tenant = await TenantsRepo(db).create(name="A")
    # rescore 0.2 -> 200 views; 0.5 -> 500; 0.8 -> 800; 0.9 -> 900.
    samples = [(0.2, 200), (0.5, 500), (0.8, 800), (0.9, 900)]
    for i, (r, v) in enumerate(samples):
        await _seed_published(db, tenant.id, suffix=f"a{i}", rescore_score=r, views=v)

    with bound_tenant(tenant.id):
        report = await compute_calibration(db, platform="youtube")
    assert report.n_paired == 4
    assert report.pearson_r is not None
    assert abs(report.pearson_r - 1.0) < 1e-9


async def test_calibration_weak_correlation_returns_low_r(db: Database) -> None:
    """Anti-correlated (rescore, views) -> r near -1."""
    tenant = await TenantsRepo(db).create(name="A")
    samples = [(0.9, 100), (0.7, 300), (0.5, 500), (0.2, 800)]
    for i, (r, v) in enumerate(samples):
        await _seed_published(db, tenant.id, suffix=f"a{i}", rescore_score=r, views=v)

    with bound_tenant(tenant.id):
        report = await compute_calibration(db, platform="youtube")
    assert report.pearson_r is not None
    assert report.pearson_r < -0.9


async def test_calibration_drops_unrescored_jobs_from_pearson(db: Database) -> None:
    """Rows without rescore stay in `rows` but don't count toward pearson."""
    tenant = await TenantsRepo(db).create(name="A")
    # 3 paired + 1 unrescored (still appears in rows).
    await _seed_published(db, tenant.id, suffix="a", rescore_score=0.2, views=200)
    await _seed_published(db, tenant.id, suffix="b", rescore_score=0.5, views=500)
    await _seed_published(db, tenant.id, suffix="c", rescore_score=0.9, views=900)
    await _seed_published(db, tenant.id, suffix="d", rescore_score=None, views=400)

    with bound_tenant(tenant.id):
        report = await compute_calibration(db, platform="youtube")
    assert len(report.rows) == 4
    assert report.n_paired == 3
    # Unrescored row has rescore_score=None.
    unrescored = [r for r in report.rows if r.rescore_score is None]
    assert len(unrescored) == 1


async def test_calibration_drops_jobs_with_no_views_from_pearson(db: Database) -> None:
    tenant = await TenantsRepo(db).create(name="A")
    await _seed_published(db, tenant.id, suffix="a", rescore_score=0.5, views=500)
    await _seed_published(db, tenant.id, suffix="b", rescore_score=0.7, views=None)

    with bound_tenant(tenant.id):
        report = await compute_calibration(db, platform="youtube")
    assert len(report.rows) == 2
    assert report.n_paired == 1
    # Pearson can't compute on n_paired=1 -> None.
    assert report.pearson_r is None


async def test_calibration_returns_none_for_under_3_paired(db: Database) -> None:
    """We refuse to compute Pearson on fewer than 3 paired samples."""
    tenant = await TenantsRepo(db).create(name="A")
    await _seed_published(db, tenant.id, suffix="a", rescore_score=0.5, views=500)
    await _seed_published(db, tenant.id, suffix="b", rescore_score=0.7, views=700)

    with bound_tenant(tenant.id):
        report = await compute_calibration(db, platform="youtube")
    assert report.n_paired == 2
    assert report.pearson_r is None


async def test_calibration_filters_by_platform(db: Database) -> None:
    """A YT calibration scan ignores TikTok jobs entirely."""
    tenant = await TenantsRepo(db).create(name="A")
    # Seed a TikTok job - must not count toward YouTube's report.
    await _seed_published(
        db, tenant.id, suffix="tt", rescore_score=0.5, views=500, platform="tiktok"
    )

    with bound_tenant(tenant.id):
        yt = await compute_calibration(db, platform="youtube")
        tt = await compute_calibration(db, platform="tiktok")
    assert yt.rows == []
    assert len(tt.rows) == 1
