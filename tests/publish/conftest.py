"""Fixtures for publish-worker tests."""

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
from chalybclip.tenancy import bound_tenant


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "publish.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


@pytest_asyncio.fixture
async def seeded(db: Database) -> dict[str, str]:
    """Seed one tenant with a stream/candidate/clip/persona/variant/account
    + one pending publish_job. Returns ids the tests use to query state."""
    tenant = await TenantsRepo(db).create(name="Aldo Co")
    with bound_tenant(tenant.id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_a",
                tenant_id=tenant.id,
                vod_url="https://example.com/v",
                platform="kick",
                title="t",
                channel="c",
                duration_s=300.0,
                source_video_path="/tmp/x.mp4",
                source_audio_path="/tmp/x.wav",
                status="ingested",
                created_at=_now(),
            )
        )
        await CandidatesRepo(db).upsert_many(
            [
                CandidateRow(
                    id="cnd_a",
                    stream_id="str_a",
                    tenant_id=tenant.id,
                    ts=10.0,
                    score=0.9,
                    reason="voice",
                    evidence={},
                    created_at=_now(),
                )
            ]
        )
        await ClipsRepo(db).upsert_many(
            [
                ClipRow(
                    id="clp_a",
                    stream_id="str_a",
                    tenant_id=tenant.id,
                    candidate_id="cnd_a",
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
        await PersonasRepo(db).create(
            persona_id="aldo",
            name="Aldo",
            primary_language="es",
            target_languages=["es", "en"],
            voice_prompt="direct",
        )
        await VariantsRepo(db).replace_for_clip_persona(
            "clp_a",
            "aldo",
            [
                VariantRow(
                    id="var_a",
                    clip_id="clp_a",
                    tenant_id=tenant.id,
                    persona_id="aldo",
                    language="es",
                    caption="hola",
                    title_card_text="",
                    hashtags=["clip", "live"],
                    model="claude-haiku-4-5-20251001",
                    created_at=_now(),
                )
            ],
        )
        account = await ConnectedAccountsRepo(db).create(
            platform="buffer",
            external_id="buf_profile_xyz",
            display_name="Aldo TikTok",
            oauth_blob={"access_token": "btok_abc"},
        )
        job = await PublishJobsRepo(db).enqueue(
            clip_id="clp_a",
            variant_id="var_a",
            account_id=account.id,
            platform="buffer",
        )
    return {
        "tenant_id": tenant.id,
        "clip_id": "clp_a",
        "variant_id": "var_a",
        "account_id": account.id,
        "job_id": job.id,
    }
