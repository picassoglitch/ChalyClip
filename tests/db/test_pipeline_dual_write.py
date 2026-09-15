"""End-to-end tests for `process_vod` with dual-write through the SQLite repos.

We re-use the heavy stubs from the existing pipeline test module (no network,
no Whisper, no ffmpeg, no Anthropic) and add the DB layer on top: each test
seeds a tenant in a fresh SQLite file under tmp_path, runs the orchestrator
with `db_path=...`, and asserts the expected rows exist.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from chalybclip.db import (
    CandidatesRepo,
    ClipsRepo,
    Database,
    LLMCallsRepo,
    PersonasRepo,
    StreamsRepo,
    TenantsRepo,
    TranscriptsRepo,
    VariantsRepo,
    apply_migrations,
)
from chalybclip.pipeline import PipelineDeps, process_vod
from chalybclip.tenancy import bound_tenant

from tests.llm._fakes import FakeProvider  # type: ignore[import]
from tests.llm._fixtures import make_llm_config  # type: ignore[import]
from tests.pipeline.test_process_vod import (  # type: ignore[import]
    _hook_payload,
    _make_config,
    _make_personas,
    _stub_ffmpeg,
    _stub_ingest,
    _stub_whisper,
)


async def _seed_tenant(db_path: Path, tenant_id: str = "default") -> None:
    db = Database(db_path)
    try:
        await apply_migrations(db)
        await TenantsRepo(db).create(tenant_id=tenant_id, name="Test Tenant")
    finally:
        await db.close()


def _make_deps_no_router() -> PipelineDeps:
    """Use the real LLMRouter (with db wired in by the orchestrator) so the
    LLM-call DB row gets exercised. Tests still inject a FakeProvider via
    the default-provider-factory monkeypatch."""
    return PipelineDeps(
        config=_make_config(),
        llm_config=make_llm_config(retry_attempts=1, purpose="hook_generation"),
        personas=_make_personas(),
    )


def _patch_default_anthropic_provider(
    monkeypatch: pytest.MonkeyPatch, fake: FakeProvider
) -> None:
    """Force the real LLMRouter's default factory to return our FakeProvider."""
    from chalybclip.llm import router as router_module

    def factory(name, _config, _api_key):
        return fake if name == "anthropic" else None

    monkeypatch.setattr(router_module, "_default_provider_factory", factory)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def test_pipeline_writes_every_entity_to_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "test.db"
    asyncio.run(_seed_tenant(db_path))

    _stub_ingest(monkeypatch)
    _stub_whisper(monkeypatch)
    _stub_ffmpeg(monkeypatch)
    fake = FakeProvider("anthropic")
    fake.queue_success(_hook_payload())
    _patch_default_anthropic_provider(monkeypatch, fake)

    manifest = asyncio.run(
        process_vod(
            tenant_id="default",
            vod_url="https://kick.com/aldovillanueva/videos/abc",
            output_dir=tmp_path,
            persona_id="aldo_villanueva",
            language="es",
            n_variants=3,
            db_path=str(db_path),
            deps=_make_deps_no_router(),
        )
    )

    async def _verify() -> None:
        db = Database(db_path)
        try:
            with bound_tenant("default"):
                streams = await StreamsRepo(db).list_for_tenant()
                assert len(streams) == 1
                assert streams[0].id == manifest.stream.id

                tx = await TranscriptsRepo(db).get(manifest.stream.id)
                assert tx is not None
                assert tx.language == "es"

                cands = await CandidatesRepo(db).list_for_stream(manifest.stream.id)
                assert len(cands) == len(manifest.candidates) == 1
                assert cands[0].id.startswith("cnd_")

                clips = await ClipsRepo(db).list_for_stream(manifest.stream.id)
                assert len(clips) == 1
                assert clips[0].candidate_id == cands[0].id

                # Persona was upserted as a side effect of the variants step.
                p = await PersonasRepo(db).get("aldo_villanueva")
                assert p is not None
                assert p.primary_language == "es"

                # One deterministic stub variant per clip.
                variants = await VariantsRepo(db).list_for_clip(clips[0].id)
                assert len(variants) == 1
                assert all(v.id.startswith("var_") for v in variants)

                # The only LLM call left in the pipeline is the auto-hook.
                llm_rows = await LLMCallsRepo(db).list_for_tenant()
                assert len(llm_rows) == 1
                assert llm_rows[0].purpose == "hook_generation"
                assert llm_rows[0].cost_usd_micros > 0
        finally:
            await db.close()

    asyncio.run(_verify())


def test_pipeline_idempotent_on_dual_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running process_vod twice with the same stream_id leaves the DB with
    one row of each entity (no duplicates) and one extra LLM call (LLM rows
    are append-only)."""
    db_path = tmp_path / "test.db"
    asyncio.run(_seed_tenant(db_path))

    _stub_ingest(monkeypatch)
    _stub_whisper(monkeypatch)
    _stub_ffmpeg(monkeypatch)
    fake = FakeProvider("anthropic")
    fake.queue_success(_hook_payload())
    _patch_default_anthropic_provider(monkeypatch, fake)

    deps = _make_deps_no_router()

    first = asyncio.run(
        process_vod(
            tenant_id="default",
            vod_url="https://kick.com/aldovillanueva/videos/abc",
            output_dir=tmp_path,
            persona_id="aldo_villanueva",
            language="es",
            n_variants=3,
            db_path=str(db_path),
            deps=deps,
        )
    )
    # Second run with same stream_id: every step hits its filesystem cache
    # AND the DB upserts are no-ops (INSERT OR IGNORE for stream/candidates/
    # clips, ON CONFLICT DO UPDATE for transcript/persona).
    asyncio.run(
        process_vod(
            tenant_id="default",
            vod_url="https://kick.com/aldovillanueva/videos/abc",
            output_dir=tmp_path,
            persona_id="aldo_villanueva",
            stream_id=first.stream.id,
            language="es",
            n_variants=3,
            db_path=str(db_path),
            deps=deps,
        )
    )

    async def _verify() -> None:
        db = Database(db_path)
        try:
            with bound_tenant("default"):
                assert len(await StreamsRepo(db).list_for_tenant()) == 1
                cands = await CandidatesRepo(db).list_for_stream(first.stream.id)
                assert len(cands) == 1
                clips = await ClipsRepo(db).list_for_stream(first.stream.id)
                assert len(clips) == 1
                # Variants: cache hit on second run -> only one stub in DB.
                # (replace_for_clip_persona reset would happen on regenerate.)
                variants = await VariantsRepo(db).list_for_clip(clips[0].id)
                assert len(variants) == 1
                # LLM calls: filesystem cache hit on second run -> only ONE
                # call total (no extra LLM call on the resume).
                llm_rows = await LLMCallsRepo(db).list_for_tenant()
                assert len(llm_rows) == 1
        finally:
            await db.close()

    asyncio.run(_verify())


def test_pipeline_no_db_path_skips_db_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without `db_path`, the orchestrator stays filesystem-only — Phase 0 path."""
    _stub_ingest(monkeypatch)
    _stub_whisper(monkeypatch)
    _stub_ffmpeg(monkeypatch)
    fake = FakeProvider("anthropic")
    fake.queue_success(_hook_payload())
    _patch_default_anthropic_provider(monkeypatch, fake)

    manifest = asyncio.run(
        process_vod(
            tenant_id="default",
            vod_url="https://kick.com/aldovillanueva/videos/abc",
            output_dir=tmp_path,
            persona_id="aldo_villanueva",
            language="es",
            n_variants=3,
            deps=_make_deps_no_router(),
        )
    )
    # Filesystem artifacts present:
    assert (tmp_path / manifest.stream.id / "manifest.json").exists()
    # No SQLite file should have been created.
    assert not any(tmp_path.glob("*.db"))


def test_pipeline_unknown_tenant_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If db_path is set but the tenant doesn't exist, the FK violation
    surfaces during the first DB write."""
    db_path = tmp_path / "test.db"
    # Migrate but do NOT seed tenant.
    db = Database(db_path)
    asyncio.run(apply_migrations(db))
    asyncio.run(db.close())

    _stub_ingest(monkeypatch)
    _stub_whisper(monkeypatch)
    _stub_ffmpeg(monkeypatch)
    fake = FakeProvider("anthropic")
    fake.queue_success(_hook_payload())
    _patch_default_anthropic_provider(monkeypatch, fake)

    with pytest.raises(Exception, match="FOREIGN KEY|integrity|tenant"):
        asyncio.run(
            process_vod(
                tenant_id="nonexistent",
                vod_url="https://kick.com/c/videos/1",
                output_dir=tmp_path,
                persona_id="aldo_villanueva",
                language="es",
                n_variants=3,
                db_path=str(db_path),
                deps=_make_deps_no_router(),
            )
        )
