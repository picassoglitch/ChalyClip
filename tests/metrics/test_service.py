"""Metrics-ingest worker — staleness gate, fetcher routing, audit rows."""

from __future__ import annotations

import datetime as _dt
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
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
    ConnectedAccount,
    PublishJob,
    StreamRow,
    VariantRow,
)
from chalybclip.metrics import NormalizedMetric, run_metrics_ingest
from chalybclip.tenancy import bound_tenant


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "metrics_svc.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


async def _seed_sent_job(
    db: Database,
    tenant_id: str,
    *,
    suffix: str,
    platform: str = "youtube",
    external_id: str = "ext_x",
) -> str:
    """Seed one tenant's chain ending in a `status='sent'` publish_job."""
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
                platform=platform,
                external_id="u",
                oauth_blob={"access_token": "tok"},
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
        # Flip to `sent` + give it the external_id the fetcher needs.
        conn = await db.connect()
        await conn.execute(
            "UPDATE publish_jobs SET status = 'sent', external_id = ? WHERE id = ?",
            (external_id, job.id),
        )
        await conn.commit()
    return job.id


async def _stub_fetcher(*responses: NormalizedMetric):
    """Build a fetcher that pops responses in order. Asserts coverage."""
    queue = list(responses)
    calls: list[tuple[str, str]] = []

    async def fake(
        job: PublishJob,
        account: ConnectedAccount,
        http: httpx.AsyncClient,
        access_token: str,
    ) -> NormalizedMetric:
        calls.append((job.id, job.platform))
        if not queue:
            raise AssertionError("no queued metric for fetch")
        return queue.pop(0)

    return fake, calls


async def test_ingest_writes_one_metric_row_per_sent_job(db: Database) -> None:
    tenant = await TenantsRepo(db).create(name="A")
    job_id = await _seed_sent_job(db, tenant.id, suffix="a")

    fake, calls = await _stub_fetcher(
        NormalizedMetric(views=1500, likes=42, raw_metadata={"k": "v"})
    )
    outcome = await run_metrics_ingest(
        tenant.id, db, fetchers={"youtube": fake}
    )
    assert outcome.fetched == 1
    assert outcome.failed == 0
    assert calls == [(job_id, "youtube")]

    with bound_tenant(tenant.id):
        latest = await PublishMetricsRepo(db).latest_for_job(job_id)
    assert latest is not None
    assert latest.views == 1500
    assert latest.raw_metadata == {"k": "v"}


async def test_ingest_skips_recently_fetched_jobs(db: Database) -> None:
    """A reading inside the refetch window short-circuits the fetcher."""
    tenant = await TenantsRepo(db).create(name="A")
    job_id = await _seed_sent_job(db, tenant.id, suffix="a")
    fixed_now = _dt.datetime(2026, 5, 7, 12, 0, 0, tzinfo=_dt.UTC)

    # Pre-record a metric 30 minutes ago.
    with bound_tenant(tenant.id):
        await PublishMetricsRepo(db).record(
            publish_job_id=job_id,
            platform="youtube",
            fetched_at=(fixed_now - _dt.timedelta(minutes=30)).isoformat(),
            views=100,
        )

    fake, calls = await _stub_fetcher(NormalizedMetric(views=999))
    outcome = await run_metrics_ingest(
        tenant.id,
        db,
        fetchers={"youtube": fake},
        refetch_after_s=3600,
        clock=lambda: fixed_now,
    )
    assert outcome.fetched == 0
    assert outcome.skipped_recent == 1
    assert calls == []


async def test_ingest_refetches_when_stale(db: Database) -> None:
    tenant = await TenantsRepo(db).create(name="A")
    job_id = await _seed_sent_job(db, tenant.id, suffix="a")
    fixed_now = _dt.datetime(2026, 5, 7, 12, 0, 0, tzinfo=_dt.UTC)

    # Pre-record from 2 hours ago - past the 1-hour refetch_after.
    with bound_tenant(tenant.id):
        await PublishMetricsRepo(db).record(
            publish_job_id=job_id,
            platform="youtube",
            fetched_at=(fixed_now - _dt.timedelta(hours=2)).isoformat(),
            views=100,
        )

    fake, calls = await _stub_fetcher(NormalizedMetric(views=300))
    outcome = await run_metrics_ingest(
        tenant.id,
        db,
        fetchers={"youtube": fake},
        refetch_after_s=3600,
        clock=lambda: fixed_now,
    )
    assert outcome.fetched == 1
    assert calls == [(job_id, "youtube")]


async def test_ingest_failed_fetcher_increments_failed(db: Database) -> None:
    tenant = await TenantsRepo(db).create(name="A")
    job_id = await _seed_sent_job(db, tenant.id, suffix="a")

    async def boom(*_args: object, **_kwargs: object) -> NormalizedMetric:
        raise RuntimeError("network down")

    outcome = await run_metrics_ingest(
        tenant.id, db, fetchers={"youtube": boom}
    )
    assert outcome.fetched == 0
    assert outcome.failed == 1

    # No metric row landed.
    with bound_tenant(tenant.id):
        latest = await PublishMetricsRepo(db).latest_for_job(job_id)
    assert latest is None


async def test_ingest_skips_jobs_with_no_token(db: Database) -> None:
    tenant = await TenantsRepo(db).create(name="A")
    job_id = await _seed_sent_job(db, tenant.id, suffix="a")
    # Strip the access token.
    conn = await db.connect()
    await conn.execute(
        "UPDATE connected_accounts SET oauth_blob_json = NULL WHERE tenant_id = ?",
        (tenant.id,),
    )
    await conn.commit()

    fake, calls = await _stub_fetcher(NormalizedMetric(views=1))
    outcome = await run_metrics_ingest(
        tenant.id, db, fetchers={"youtube": fake}
    )
    assert outcome.fetched == 0
    assert outcome.skipped_no_account == 1
    assert calls == []
    _ = job_id


async def test_ingest_only_drains_sent_jobs(db: Database) -> None:
    """Jobs that are still `pending` should NOT be polled."""
    tenant = await TenantsRepo(db).create(name="A")
    sent_job = await _seed_sent_job(db, tenant.id, suffix="sent", external_id="ok")
    pending_job = await _seed_sent_job(db, tenant.id, suffix="pend", external_id="px")
    # Flip pending_job back to status='pending'.
    conn = await db.connect()
    await conn.execute(
        "UPDATE publish_jobs SET status = 'pending' WHERE id = ?",
        (pending_job,),
    )
    await conn.commit()

    fake, calls = await _stub_fetcher(NormalizedMetric(views=1))
    await run_metrics_ingest(tenant.id, db, fetchers={"youtube": fake})
    assert calls == [(sent_job, "youtube")]
