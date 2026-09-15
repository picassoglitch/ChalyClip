"""Verify the orchestrator emits the canonical events for a happy-path run."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from chalybclip.db import Database, EventsRepo, TenantsRepo, apply_migrations
from chalybclip.events import (
    CLIP_READY_FOR_REVIEW,
    STREAM_CREATED,
    STREAM_PROCESSED,
)
from chalybclip.pipeline import process_vod
from chalybclip.tenancy import bound_tenant

from tests.db.test_pipeline_dual_write import (  # type: ignore[import]
    _make_deps_no_router,
    _patch_default_anthropic_provider,
    _seed_tenant,
)
from tests.llm._fakes import FakeProvider  # type: ignore[import]
from tests.pipeline.test_process_vod import (  # type: ignore[import]
    _hook_payload,
    _stub_ffmpeg,
    _stub_ingest,
    _stub_whisper,
)


def _events_for(db: Database, tenant_id: str) -> list:
    async def go() -> list:
        with bound_tenant(tenant_id):
            return await EventsRepo(db).list_for_tenant()

    return asyncio.run(go())


def test_pipeline_emits_canonical_events(
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

    db = Database(db_path)
    try:
        events = _events_for(db, "default")
    finally:
        asyncio.run(db.close())

    # Canonical lifecycle events: 1 stream + 1 clip + 1 stream-processed.
    # The run also emits auxiliary rows (`pipeline.step.*`, `clip.cut.*`,
    # `clip.auto_corrected`, …) for the dashboard's progress poller — those
    # are additive and tested elsewhere, so assert only that each canonical
    # event fires exactly once.
    canonical_types = [STREAM_CREATED, CLIP_READY_FOR_REVIEW, STREAM_PROCESSED]
    canonical = [e.type for e in events if e.type in canonical_types]
    assert sorted(canonical) == sorted(canonical_types)

    by_type = {e.type: e for e in events if e.type in canonical_types}

    assert by_type[STREAM_CREATED].payload["stream_id"] == manifest.stream.id
    assert by_type[STREAM_CREATED].payload["platform"] == "kick"

    assert by_type[CLIP_READY_FOR_REVIEW].payload["stream_id"] == manifest.stream.id
    assert by_type[CLIP_READY_FOR_REVIEW].payload["persona_id"] == "aldo_villanueva"
    # Variants are a deterministic stub now: exactly one per clip.
    assert by_type[CLIP_READY_FOR_REVIEW].payload["variant_count"] == 1

    assert by_type[STREAM_PROCESSED].payload["stream_id"] == manifest.stream.id
    assert by_type[STREAM_PROCESSED].payload["clip_count"] == 1
    assert by_type[STREAM_PROCESSED].payload["llm_calls"] == 1
    assert by_type[STREAM_PROCESSED].payload["cost_usd_micros"] > 0


def test_pipeline_no_db_emits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a DB, emit() is a no-op everywhere — pipeline still works."""
    _stub_ingest(monkeypatch)
    _stub_whisper(monkeypatch)
    _stub_ffmpeg(monkeypatch)
    fake = FakeProvider("anthropic")
    fake.queue_success(_hook_payload())
    _patch_default_anthropic_provider(monkeypatch, fake)

    asyncio.run(
        process_vod(
            tenant_id="default",
            vod_url="https://kick.com/c/videos/1",
            output_dir=tmp_path,
            persona_id="aldo_villanueva",
            language="es",
            n_variants=3,
            deps=_make_deps_no_router(),
        )
    )
    # No DB created.
    assert not any(tmp_path.glob("*.db"))


def test_pipeline_emits_one_clip_event_per_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the run produces N clips, we get N clip.ready_for_review events.

    We verify the 1-clip case (the stub produces one trigger). The fan-out
    test (N>1) is exercised structurally by the same code path.
    """
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
            vod_url="https://kick.com/c/videos/1",
            output_dir=tmp_path,
            persona_id="aldo_villanueva",
            language="es",
            n_variants=3,
            db_path=str(db_path),
            deps=_make_deps_no_router(),
        )
    )
    db = Database(db_path)
    try:
        events = _events_for(db, "default")
    finally:
        asyncio.run(db.close())

    clip_events = [e for e in events if e.type == CLIP_READY_FOR_REVIEW]
    assert len(clip_events) == len(manifest.clip_entries)
