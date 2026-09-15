"""Dashboard surfaces engagement outcomes + calibration table."""

from __future__ import annotations

import datetime as _dt

import httpx

from chalybclip.db import (
    CandidatesRepo,
    ClipsRepo,
    ConnectedAccountsRepo,
    Database,
    PersonasRepo,
    PublishJobsRepo,
    PublishMetricsRepo,
    StreamsRepo,
    VariantsRepo,
)
from chalybclip.db.models import (
    CandidateRow,
    ClipRow,
    StreamRow,
    VariantRow,
)
from chalybclip.tenancy import bound_tenant


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


async def _seed_clip_with_outcome(
    db: Database, tenant_id: str, *, suffix: str, views: int, platform: str = "youtube"
) -> str:
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
        await CandidatesRepo(db).update_rescore(
            f"cnd_{suffix}",
            rescore_score=0.85,
            rescore_reason="strong",
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
        # Set sent + external_url to mimic a posted clip.
        conn = await db.connect()
        await conn.execute(
            "UPDATE publish_jobs SET status = 'sent', external_id = ?, "
            "external_url = ? WHERE id = ?",
            (f"vid_{suffix}", f"https://example/{suffix}", job.id),
        )
        await conn.commit()
        await PublishMetricsRepo(db).record(
            publish_job_id=job.id,
            platform=platform,
            fetched_at=_now(),
            views=views,
            likes=views // 50,
            comments=views // 200,
        )
    return f"clp_{suffix}"


async def test_clip_page_renders_engagement_outcomes(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    tenant_id = tenants["alice"]["id"]
    clip_id = await _seed_clip_with_outcome(db, tenant_id, suffix="a", views=12_345)

    await client.post("/dashboard/login", data={"token": tenants["alice"]["token"]})
    r = await client.get(f"/dashboard/clips/{clip_id}")
    assert r.status_code == 200
    assert "Engagement outcomes" in r.text
    # Number is comma-formatted in the template.
    assert "12,345" in r.text
    # Link to calibration page is present.
    assert "/dashboard/calibration" in r.text


async def test_calibration_page_renders_per_platform_reports(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """The page renders even with 0 data — and shows our 3 default platforms."""
    await client.post("/dashboard/login", data={"token": tenants["alice"]["token"]})
    r = await client.get("/dashboard/calibration")
    assert r.status_code == 200
    assert "Vision-rescore calibration" in r.text
    # All three default platforms surface.
    assert "youtube" in r.text
    assert "tiktok" in r.text
    assert "buffer" in r.text


async def test_calibration_page_renders_pearson_when_paired_data_present(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """≥3 paired samples -> Pearson r is rendered + per-clip table appears."""
    tenant_id = tenants["alice"]["id"]
    for i in range(3):
        await _seed_clip_with_outcome(
            db, tenant_id, suffix=f"a{i}", views=(i + 1) * 100
        )
    await client.post("/dashboard/login", data={"token": tenants["alice"]["token"]})
    r = await client.get("/dashboard/calibration")
    assert r.status_code == 200
    assert "n_paired = 3" in r.text
    # Per-clip table renders.
    assert "youtube — per-clip outcomes" in r.text
