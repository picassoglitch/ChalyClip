"""Phase L.2 — auto-clip after a live stream ends.

When a streamer pushes through our RTMP relay and stops, MediaMTX calls
`/api/internal/live/ended`. On top of flipping the stream to 'live_ended',
we now auto-launch the clip pipeline on the recording so they get
publish-ready clips with zero dashboard interaction.

Covered here:
  - The atomic claim helper (idempotency guard vs duplicate webhooks).
  - `maybe_autoclip_after_live_end`: schedules on the happy path, is
    idempotent, skips when disabled, skips (and doesn't claim) with no
    persona.
  - `live_pipeline_runner`: ingests the recording like an upload, then
    runs the pipeline, then refreshes the balance.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest_asyncio

from chalybclip.api import _pipeline
from chalybclip.db import (
    Database,
    PersonasRepo,
    StreamsRepo,
    TenantsRepo,
    apply_migrations,
)
from chalybclip.db.models import StreamRow
from chalybclip.db.repos import _streams_repo_try_claim_for_processing
from chalybclip.tenancy import bound_tenant


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "live.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


async def _seed_live_ended(
    db: Database, *, tenant_id: str, stream_id: str, status: str = "live_ended"
) -> None:
    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=stream_id,
                tenant_id=tenant_id,
                vod_url=f"live://rtmp/{stream_id}",
                platform="live",
                title="Mi stream en vivo",
                channel=None,
                duration_s=0.0,
                source_video_path=f"/data/live/{stream_id}/source.mp4",
                source_audio_path=f"/data/live/{stream_id}/source.audio.wav",
                status=status,
                created_at=_now(),
            )
        )


async def _seed_persona(db: Database, tenant_id: str, persona_id: str) -> None:
    with bound_tenant(tenant_id):
        await PersonasRepo(db).upsert(
            persona_id=persona_id,
            name="Default",
            primary_language="es",
            target_languages=["es"],
            voice_prompt="habla claro",
        )


async def _get_row(db: Database, tenant_id: str, stream_id: str) -> StreamRow | None:
    with bound_tenant(tenant_id):
        return await StreamsRepo(db).get(stream_id)


def _stub_settings(db_file: Path, out_dir: Path, *, enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        db_path=str(db_file),
        default_output_dir=str(out_dir),
        live_auto_clip_enabled=enabled,
    )


# ---- claim helper ---------------------------------------------------------


async def test_claim_for_processing_atomic_and_idempotent(db: Database) -> None:
    """First claim wins (live_ended → processing); the second loses."""
    tenant = await TenantsRepo(db).create(name="Aldo")
    await _seed_live_ended(db, tenant_id=tenant.id, stream_id="str_c")

    assert await _streams_repo_try_claim_for_processing(db, stream_id="str_c") is True
    # Row is 'processing' now — a duplicate webhook can't re-claim it.
    assert await _streams_repo_try_claim_for_processing(db, stream_id="str_c") is False

    row = await _get_row(db, tenant.id, "str_c")
    assert row is not None and row.status == "processing"


async def test_claim_skips_when_not_live_ended(db: Database) -> None:
    """A stream still 'live' (or any other status) is not claimable."""
    tenant = await TenantsRepo(db).create(name="Aldo")
    await _seed_live_ended(db, tenant_id=tenant.id, stream_id="str_n", status="live")
    assert await _streams_repo_try_claim_for_processing(db, stream_id="str_n") is False


# ---- maybe_autoclip_after_live_end ----------------------------------------


async def test_autoclip_schedules_runner_when_enabled(
    db: Database, tmp_path: Path, monkeypatch
) -> None:
    tenant = await TenantsRepo(db).create(name="Aldo")
    await _seed_live_ended(db, tenant_id=tenant.id, stream_id="str_a")
    await _seed_persona(db, tenant.id, "per_a")
    monkeypatch.setattr(
        "chalybclip.settings.get_settings",
        lambda: _stub_settings(tmp_path / "live.db", tmp_path / "out"),
    )

    calls: list[dict] = []

    def collector(**kwargs: object) -> None:
        calls.append(dict(kwargs))

    row = await _get_row(db, tenant.id, "str_a")
    scheduled = await _pipeline.maybe_autoclip_after_live_end(
        db, row=row, schedule=collector
    )

    assert scheduled is True
    assert len(calls) == 1
    c = calls[0]
    assert c["tenant_id"] == tenant.id
    assert c["stream_id"] == "str_a"
    assert c["persona_id"] == "per_a"
    assert str(c["recording_path"]).endswith("source.mp4")
    assert c["title"] == "Mi stream en vivo"
    # The stream was claimed (idempotency marker).
    after = await _get_row(db, tenant.id, "str_a")
    assert after is not None and after.status == "processing"


async def test_autoclip_idempotent_second_call_skips(
    db: Database, tmp_path: Path, monkeypatch
) -> None:
    """A duplicate 'ended' webhook must not schedule a second pipeline."""
    tenant = await TenantsRepo(db).create(name="Aldo")
    await _seed_live_ended(db, tenant_id=tenant.id, stream_id="str_i")
    await _seed_persona(db, tenant.id, "per_i")
    monkeypatch.setattr(
        "chalybclip.settings.get_settings",
        lambda: _stub_settings(tmp_path / "live.db", tmp_path / "out"),
    )
    calls: list[dict] = []
    row = await _get_row(db, tenant.id, "str_i")

    first = await _pipeline.maybe_autoclip_after_live_end(
        db, row=row, schedule=lambda **kw: calls.append(dict(kw))
    )
    # Same (now-stale) row object, as a retried webhook would carry.
    second = await _pipeline.maybe_autoclip_after_live_end(
        db, row=row, schedule=lambda **kw: calls.append(dict(kw))
    )

    assert first is True
    assert second is False
    assert len(calls) == 1


async def test_autoclip_skips_when_disabled(
    db: Database, tmp_path: Path, monkeypatch
) -> None:
    """CHALYBCLIP_LIVE_AUTO_CLIP=false → recording-only; nothing scheduled,
    nothing claimed."""
    tenant = await TenantsRepo(db).create(name="Aldo")
    await _seed_live_ended(db, tenant_id=tenant.id, stream_id="str_d")
    await _seed_persona(db, tenant.id, "per_d")
    monkeypatch.setattr(
        "chalybclip.settings.get_settings",
        lambda: _stub_settings(tmp_path / "live.db", tmp_path / "out", enabled=False),
    )
    calls: list[dict] = []
    row = await _get_row(db, tenant.id, "str_d")

    scheduled = await _pipeline.maybe_autoclip_after_live_end(
        db, row=row, schedule=lambda **kw: calls.append(dict(kw))
    )

    assert scheduled is False
    assert calls == []
    after = await _get_row(db, tenant.id, "str_d")
    assert after is not None and after.status == "live_ended"  # not claimed


async def test_autoclip_skips_when_no_persona(
    db: Database, tmp_path: Path, monkeypatch
) -> None:
    """No persona → can't clip; don't claim, don't schedule (operator adds
    a persona + runs manually)."""
    tenant = await TenantsRepo(db).create(name="Aldo")
    await _seed_live_ended(db, tenant_id=tenant.id, stream_id="str_p")
    monkeypatch.setattr(
        "chalybclip.settings.get_settings",
        lambda: _stub_settings(tmp_path / "live.db", tmp_path / "out"),
    )
    calls: list[dict] = []
    row = await _get_row(db, tenant.id, "str_p")

    scheduled = await _pipeline.maybe_autoclip_after_live_end(
        db, row=row, schedule=lambda **kw: calls.append(dict(kw))
    )

    assert scheduled is False
    assert calls == []
    after = await _get_row(db, tenant.id, "str_p")
    assert after is not None and after.status == "live_ended"  # not claimed


# ---- _acquire_live_recording (R2 vs shared disk) --------------------------


class _FakeStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self.calls = 0

    async def fetch_latest(self, *, stream_id: str, dest_dir: Path) -> Path | None:
        self.calls += 1
        return self._path


async def test_acquire_uses_r2_store_when_configured(
    tmp_path: Path, monkeypatch
) -> None:
    """With object storage configured, the recording is pulled via the
    store (Path B) — no shared disk needed."""
    rec = tmp_path / "rec.mp4"
    rec.write_bytes(b"\x00" * 4096)
    store = _FakeStore(rec)
    monkeypatch.setattr(
        "chalybclip.integrations.storage.build_recording_store", lambda _s: store
    )
    monkeypatch.setattr("chalybclip.settings.get_settings", lambda: SimpleNamespace())

    got = await _pipeline._acquire_live_recording(
        stream_id="str_1", recording_path="/data/unused", work_dir=tmp_path / "in"
    )
    assert got == rec
    assert store.calls == 1


async def test_acquire_falls_back_to_disk_without_r2(
    tmp_path: Path, monkeypatch
) -> None:
    """No store configured → read the shared volume (Path A)."""
    d = tmp_path / "live" / "str_1"
    d.mkdir(parents=True)
    f = d / "source.mp4"
    f.write_bytes(b"\x00" * 4096)
    monkeypatch.setattr(
        "chalybclip.integrations.storage.build_recording_store", lambda _s: None
    )
    monkeypatch.setattr("chalybclip.settings.get_settings", lambda: SimpleNamespace())

    got = await _pipeline._acquire_live_recording(
        stream_id="str_1", recording_path=str(d / "source"), work_dir=tmp_path / "in"
    )
    assert got == f


# ---- live_pipeline_runner -------------------------------------------------


async def test_live_runner_ingests_then_processes(
    db: Database, tmp_path: Path, monkeypatch
) -> None:
    """The runner resolves the recording, ingests it like an upload, runs
    the pipeline, then refreshes the balance — in that order."""
    tenant = await TenantsRepo(db).create(name="Aldo")
    seq: list[str] = []

    async def fake_acquire(
        *, stream_id: str, recording_path: str, work_dir: Path
    ) -> Path:
        seq.append("resolve")
        return tmp_path / "rec.mp4"

    monkeypatch.setattr(_pipeline, "_acquire_live_recording", fake_acquire)

    fake_stream = SimpleNamespace(id="str_r", vod_url="live://rtmp/str_r")

    async def fake_ingest(**kwargs: object) -> object:
        seq.append("ingest")
        assert str(kwargs["stream_id"]) == "str_r"
        return fake_stream

    monkeypatch.setattr("chalybclip.ingest.ingest_uploaded", fake_ingest)

    def fake_stream_to_row(_stream: object, **_kw: object) -> StreamRow:
        return StreamRow(
            id="str_r",
            tenant_id=tenant.id,
            vod_url="upload://x",
            platform="upload",
            title="x",
            channel=None,
            duration_s=12.0,
            source_video_path=str(tmp_path / "out" / "str_r" / "source" / "video.mp4"),
            source_audio_path=str(tmp_path / "out" / "str_r" / "source" / "audio.wav"),
            status="ingested",
            created_at=_now(),
        )

    monkeypatch.setattr("chalybclip.db.adapters.stream_to_row", fake_stream_to_row)

    async def fake_process(**kwargs: object) -> None:
        seq.append("process")
        # Pipeline runs against the live stream id.
        assert str(kwargs["stream_id"]) == "str_r"

    monkeypatch.setattr("chalybclip.pipeline.process_vod", fake_process)

    async def fake_refresh(**_kwargs: object) -> None:
        seq.append("refresh")

    monkeypatch.setattr(_pipeline, "_refresh_balance_after_run", fake_refresh)
    monkeypatch.setattr(
        "chalybclip.settings.get_settings",
        lambda: _stub_settings(tmp_path / "live.db", tmp_path / "out"),
    )

    await _pipeline.live_pipeline_runner(
        tenant_id=tenant.id,
        stream_id="str_r",
        persona_id="per_r",
        recording_path=str(tmp_path / "rec.mp4"),
        output_dir=tmp_path / "out",
        title="Mi stream",
    )

    assert seq == ["resolve", "ingest", "process", "refresh"]
    # The promoted row is tagged live.
    after = await _get_row(db, tenant.id, "str_r")
    assert after is not None
    assert after.platform == "live"
    assert after.vod_url == "live://rtmp/str_r"
    # Flipped to a terminal status so the dashboard stops showing
    # 'Analyzing…' (the autoclip claim left it 'processing').
    assert after.status == "done"
