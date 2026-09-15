"""Task A1b — AssemblyAI cost metering through the budget governor.

Three things to pin:
  1. cost_micros_for() math matches the published pricing.
  2. transcribe() service records an LLMCallRow when the provider
     declares a cost rate AND a db is supplied. Skips the record
     when the provider is local (no marginal cost) or when no db.
  3. transcribe() refuses with BudgetExceeded when today's spend is
     already at-or-above the tenant's daily cap.

We use the real Database + repos for the LLMCallRow / TenantsRepo
paths and inject a FakeAssemblyAIProvider so we don't hit the wire.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import pytest_asyncio

from chalybclip.db import (
    Database,
    LLMCallsRepo,
    TenantsRepo,
    apply_migrations,
)
from chalybclip.errors import BudgetExceeded
from chalybclip.ingest import Stream
from chalybclip.tenancy import bound_tenant
from chalybclip.transcribe.models import Segment, Transcript, Word
from chalybclip.transcribe.providers import assemblyai
from chalybclip.transcribe.service import transcribe

import pytest


class _FakeAssemblyAI:
    """Drop-in for AssemblyAIProvider — declares the cost rate the
    service reads, returns a canned Transcript with a known duration
    so the cost math is deterministic."""

    def __init__(self, *, duration_s: float, speaker_labels: bool = True) -> None:
        self._duration_s = duration_s
        self._speaker_labels = speaker_labels
        # The service reads `_speech_model` to populate the `model`
        # column on the LLMCallRow when the provider doesn't expose
        # one directly. Keep this stub field so the fake stands in
        # for AssemblyAIProvider's downstream cost-row contract.
        self._speech_model = "universal-3-pro"

    @property
    def name(self) -> str:
        return "assemblyai-universal-3-pro"

    def cost_for_duration_micros(self, duration_s: float) -> int:
        return assemblyai.cost_micros_for(
            duration_s=duration_s, speaker_labels=self._speaker_labels,
        )

    async def transcribe(self, req: Any) -> Transcript:
        return Transcript(
            stream_id=req.stream_id,
            tenant_id=req.tenant_id,
            language="es",
            duration_s=self._duration_s,
            model=self.name,
            segments=[
                Segment(
                    ts=0.0, end_ts=1.0, text="hi",
                    speaker="A",
                    words=[Word(ts=0.0, end_ts=1.0, text="hi", prob=0.9)],
                ),
            ],
            speakers=["A"],
        )


@pytest_asyncio.fixture
async def db(tmp_path: Path):
    d = Database(tmp_path / "cost.db")
    await apply_migrations(d)
    yield d
    await d.close()


@pytest_asyncio.fixture
async def tenant(db: Database) -> str:
    t = await TenantsRepo(db).create(name="Cost Test")
    return t.id


def _make_stream(tmp_path: Path, tenant_id: str) -> Stream:
    """Construct a Stream pointing at a real audio file so the
    audio_path.exists() check passes."""
    stream_dir = tmp_path / "str_x"
    src = stream_dir / "source"
    src.mkdir(parents=True, exist_ok=True)
    audio = src / "audio.wav"
    audio.write_bytes(b"RIFFFAKEWAV")
    video = src / "video.mp4"
    video.write_bytes(b"\x00\x00")
    return Stream(
        id="str_x",
        tenant_id=tenant_id,
        vod_url="https://kick.com/x/videos/1",
        platform="kick",
        title="t",
        channel="c",
        duration_s=600.0,
        source_video_path=video,
        source_audio_path=audio,
    )


# ---- cost math ----


def test_cost_micros_base_only() -> None:
    """1 hour without diarization = $0.15 = 150_000 micros."""
    assert assemblyai.cost_micros_for(
        duration_s=3600.0, speaker_labels=False,
    ) == 150_000


def test_cost_micros_with_diarization() -> None:
    """1 hour with diarization = $0.17 = 170_000 micros."""
    assert assemblyai.cost_micros_for(
        duration_s=3600.0, speaker_labels=True,
    ) == 170_000


def test_cost_micros_short_audio() -> None:
    """12 seconds with diarization = ~566 micros (sanity)."""
    cost = assemblyai.cost_micros_for(
        duration_s=12.0, speaker_labels=True,
    )
    assert 500 <= cost <= 700


def test_cost_micros_clamps_negative_duration() -> None:
    assert assemblyai.cost_micros_for(
        duration_s=-5.0, speaker_labels=True,
    ) == 0


# ---- service-level recording ----


async def test_transcribe_records_cost_when_provider_has_rate(
    tmp_path: Path, db: Database, tenant: str,
) -> None:
    stream = _make_stream(tmp_path, tenant)
    provider = _FakeAssemblyAI(duration_s=3600.0)  # 1 hour

    await transcribe(
        tenant_id=tenant, stream=stream,
        provider=provider, db=db,
    )

    with bound_tenant(tenant):
        rows = await LLMCallsRepo(db).list_for_tenant(limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row.purpose == "transcribe"
    assert row.provider == "assemblyai-universal-3-pro"
    assert row.cost_usd_micros == 170_000  # base + diarization
    assert row.status == "ok"


async def test_transcribe_records_zero_when_no_provider_rate(
    tmp_path: Path, db: Database, tenant: str,
) -> None:
    """LocalWhisperProvider doesn't expose cost_rate_per_second_micros
    — the service should skip the record entirely (not write a 0-cost
    row that would clutter the audit log)."""
    class _FakeLocal:
        @property
        def name(self) -> str:
            return "local-whisper-tiny-cpu"

        async def transcribe(self, req: Any) -> Transcript:
            return Transcript(
                stream_id=req.stream_id,
                tenant_id=req.tenant_id,
                language="es",
                duration_s=600.0,
                model="tiny",
                segments=[],
            )

    stream = _make_stream(tmp_path, tenant)
    await transcribe(
        tenant_id=tenant, stream=stream,
        provider=_FakeLocal(), db=db,
    )
    with bound_tenant(tenant):
        rows = await LLMCallsRepo(db).list_for_tenant(limit=10)
    assert rows == []


async def test_transcribe_skips_record_when_no_db(
    tmp_path: Path, tenant: str,
) -> None:
    """CLI / test usage without a db gets the transcript but no cost
    record — no crash, no log entry, transcript still on disk."""
    stream = _make_stream(tmp_path, tenant)
    provider = _FakeAssemblyAI(duration_s=60.0)
    transcript = await transcribe(
        tenant_id=tenant, stream=stream,
        provider=provider,
        # db=None default
    )
    assert transcript.duration_s == 60.0
    assert (stream.source_audio_path.parent / "transcript.json").exists()


async def test_transcribe_refuses_when_budget_exceeded(
    tmp_path: Path, db: Database, tenant: str,
) -> None:
    """Tenant with a $0.10 daily cap who's already at $0.10 spent
    today gets BudgetExceeded before the provider call. The
    pre-existing llm_calls row stays; no new transcribe row lands."""
    # Cap = $0.10 = 100_000 micros
    await TenantsRepo(db).set_budget(
        tenant, daily_llm_budget_usd_micros=100_000,
    )
    # Pre-fill spend = 100_000 (already at cap).
    from chalybclip.db.models import LLMCallRow
    from chalybclip.ids import new_id
    with bound_tenant(tenant):
        await LLMCallsRepo(db).record(
            LLMCallRow(
                id=new_id("llmc"),
                tenant_id=tenant,
                purpose="caption",
                provider="anthropic",
                model="claude-opus-4-7",
                quality="standard",
                cost_usd_micros=100_000,
                ts=_dt.datetime.now(_dt.UTC).isoformat(),
            )
        )

    stream = _make_stream(tmp_path, tenant)
    provider = _FakeAssemblyAI(duration_s=600.0)

    with pytest.raises(BudgetExceeded):
        await transcribe(
            tenant_id=tenant, stream=stream,
            provider=provider, db=db,
        )

    # Only the pre-filled record exists — no transcribe row landed.
    with bound_tenant(tenant):
        rows = await LLMCallsRepo(db).list_for_tenant(limit=10)
    assert len(rows) == 1
    assert rows[0].purpose == "caption"
