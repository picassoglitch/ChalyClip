"""Vision-LLM rescore — guardrails + reordering + persistence."""

from __future__ import annotations

import datetime as _dt
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from chalybclip.db import (
    CandidatesRepo,
    Database,
    EventsRepo,
    StreamsRepo,
    TenantsRepo,
    apply_migrations,
)
from chalybclip.db.adapters import candidate_pk
from chalybclip.db.models import StreamRow
from chalybclip.detect import Candidate, rescore_candidates
from chalybclip.errors import LLMError
from chalybclip.governance import BudgetGovernor
from chalybclip.ingest.models import Stream
from chalybclip.llm import LLMRouter, MemoryFrameStore
from chalybclip.llm.config import ProviderConfig
from chalybclip.tenancy import bound_tenant
from tests.llm._fakes import FakeProvider  # type: ignore[import]
from tests.llm._fixtures import make_llm_config  # type: ignore[import]


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "rescore.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


def _factory(providers: dict[str, FakeProvider]):
    def _build(name: str, _config: ProviderConfig, _api_key: str) -> FakeProvider | None:
        return providers.get(name)

    return _build


def _make_stream(tmp_path: Path, tenant_id: str) -> Stream:
    """A Stream pointing at a placeholder video file (sample_frames is monkeypatched)."""
    video = tmp_path / "vod.mp4"
    video.write_bytes(b"\x00fakevideo")
    audio = tmp_path / "vod.wav"
    audio.write_bytes(b"\x00fakeaudio")
    return Stream(
        id="str_resc",
        tenant_id=tenant_id,
        vod_url="x",
        platform="kick",
        title="t",
        channel="c",
        duration_s=300.0,
        source_video_path=video,
        source_audio_path=audio,
    )


async def _seed_stream_row(db: Database, stream: Stream) -> None:
    with bound_tenant(stream.tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=stream.id,
                tenant_id=stream.tenant_id,
                vod_url=stream.vod_url,
                platform=stream.platform,
                title=stream.title,
                channel=stream.channel,
                duration_s=stream.duration_s,
                source_video_path=str(stream.source_video_path),
                source_audio_path=str(stream.source_audio_path),
                status="ingested",
                created_at=_now(),
            )
        )


async def _seed_candidates(
    db: Database, stream: Stream, candidates: list[Candidate]
) -> None:
    """Persist heuristic candidate rows so update_rescore can find them."""
    from chalybclip.db.models import CandidateRow

    rows = [
        CandidateRow(
            id=candidate_pk(stream.id, c),
            stream_id=stream.id,
            tenant_id=stream.tenant_id,
            ts=c.timestamp,
            score=c.score,
            reason=c.reason,
            evidence=c.evidence,
            created_at=_now(),
        )
        for c in candidates
    ]
    with bound_tenant(stream.tenant_id):
        await CandidatesRepo(db).upsert_many(rows)


def _stub_sample_frames(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace cv2-backed sampler with a deterministic stub. Returns the
    list it appends to so tests can assert the sampler was called."""
    calls: list[float] = []

    def fake_sample(video_path: Path, ts: float, n: int = 1, *, spread_s: float | None = None):
        calls.append(float(ts))
        return [f"frame@{ts:.2f}#{i}".encode() for i in range(n)]

    monkeypatch.setattr("chalybclip.detect.vision_rescore.sample_frames", fake_sample)
    return calls


def _verdict_payload(score: float, *, reason: str = "looks like a real reaction") -> dict:
    return {"score": score, "face_emotion": "shock", "reason": reason}


# ---------- Happy path: rescore reorders by verdict ----------


async def test_rescore_promotes_high_verdict_over_higher_heuristic(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two candidates: heuristic ranks A>B, but vision rescores B higher."""
    tenant = await TenantsRepo(db).create(name="Aldo")
    stream = _make_stream(tmp_path, tenant.id)
    await _seed_stream_row(db, stream)

    a = Candidate(timestamp=10.0, score=0.9, reason="voice", evidence={})
    b = Candidate(timestamp=20.0, score=0.6, reason="audio", evidence={})
    await _seed_candidates(db, stream, [a, b])

    fake = FakeProvider("anthropic")
    fake.queue_success(_verdict_payload(0.20))  # A scores low
    fake.queue_success(_verdict_payload(0.95))  # B scores high
    config = make_llm_config(purpose="vision_rescore", retry_attempts=1)
    router = LLMRouter(
        config=config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
        db=db,
    )
    _stub_sample_frames(monkeypatch)

    final, outcome = await rescore_candidates(
        tenant_id=tenant.id,
        stream=stream,
        candidates=[a, b],
        db=db,
        router=router,
        frame_store=MemoryFrameStore(),
        n_frames_per_candidate=2,
    )
    assert outcome.rescored == 2
    assert outcome.skipped_budget == 0
    assert outcome.fatal_errors == 0
    # B (rescore=0.95) leads A (rescore=0.20) post-rescore.
    assert [c.timestamp for c in final] == [20.0, 10.0]
    assert final[0].evidence["rescore"]["score"] == 0.95
    assert final[1].evidence["rescore"]["score"] == 0.20

    # DB rows reflect the verdicts.
    with bound_tenant(tenant.id):
        rows = await CandidatesRepo(db).list_for_stream(stream.id)
    by_ts = {r.ts: r for r in rows}
    assert by_ts[10.0].rescore_score == 0.20
    assert by_ts[20.0].rescore_score == 0.95
    assert by_ts[10.0].rescore_model is not None
    assert by_ts[10.0].rescore_reason is not None


# ---------- Budget exhaustion halts cleanly ----------


async def test_rescore_halts_on_budget_exhausted_keeps_earlier_verdicts(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tenant ceiling = $0; first call refused before the provider is hit."""
    tenant = await TenantsRepo(db).create(name="Aldo")
    await TenantsRepo(db).set_budget(
        tenant.id, daily_llm_budget_usd_micros=0
    )  # 0 USD = blocked immediately
    stream = _make_stream(tmp_path, tenant.id)
    await _seed_stream_row(db, stream)
    a = Candidate(timestamp=10.0, score=0.9, reason="voice", evidence={})
    await _seed_candidates(db, stream, [a])

    fake = FakeProvider("anthropic")
    fake.queue_success(_verdict_payload(0.5))
    config = make_llm_config(purpose="vision_rescore", retry_attempts=1)
    router = LLMRouter(
        config=config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
        db=db,
        governor=BudgetGovernor(db),
    )
    _stub_sample_frames(monkeypatch)

    final, outcome = await rescore_candidates(
        tenant_id=tenant.id,
        stream=stream,
        candidates=[a],
        db=db,
        router=router,
        governor=BudgetGovernor(db),
        frame_store=MemoryFrameStore(),
    )
    assert outcome.rescored == 0
    assert outcome.skipped_budget == 1
    # No verdict landed.
    assert "rescore" not in final[0].evidence
    # Provider was never asked.
    assert fake.calls == []
    # And `vision_rescore.budget_exhausted` event fired.
    with bound_tenant(tenant.id):
        events = await EventsRepo(db).list_for_tenant(type="vision_rescore.budget_exhausted")
    assert len(events) == 1


# ---------- Cooldown halts whole pass ----------


async def test_rescore_skips_when_in_cooldown(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tenant = await TenantsRepo(db).create(name="Aldo")
    stream = _make_stream(tmp_path, tenant.id)
    await _seed_stream_row(db, stream)
    a = Candidate(timestamp=10.0, score=0.9, reason="voice", evidence={})
    await _seed_candidates(db, stream, [a])

    fake = FakeProvider("anthropic")
    config = make_llm_config(purpose="vision_rescore", retry_attempts=1)
    router = LLMRouter(
        config=config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
        db=db,
    )
    fixed_now = _dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=_dt.UTC)
    governor = BudgetGovernor(
        db,
        clock=lambda: fixed_now,
        low_confidence_threshold=0.3,
        low_confidence_lookback=2,
        cooldown_s=300,
    )
    # Trip the cooldown.
    governor.record_rescore_verdict(tenant.id, 0.05)
    governor.record_rescore_verdict(tenant.id, 0.10)
    _stub_sample_frames(monkeypatch)

    _final, outcome = await rescore_candidates(
        tenant_id=tenant.id,
        stream=stream,
        candidates=[a],
        db=db,
        router=router,
        governor=governor,
        frame_store=MemoryFrameStore(),
    )
    assert outcome.rescored == 0
    assert outcome.skipped_cooldown == 1
    assert fake.calls == []
    with bound_tenant(tenant.id):
        events = await EventsRepo(db).list_for_tenant(type="vision_rescore.cooldown_active")
    assert len(events) == 1


# ---------- Provider error is non-fatal for the pass ----------


async def test_rescore_provider_error_is_non_fatal(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tenant = await TenantsRepo(db).create(name="Aldo")
    stream = _make_stream(tmp_path, tenant.id)
    await _seed_stream_row(db, stream)
    a = Candidate(timestamp=10.0, score=0.9, reason="voice", evidence={})
    b = Candidate(timestamp=20.0, score=0.5, reason="audio", evidence={})
    await _seed_candidates(db, stream, [a, b])

    fake = FakeProvider("anthropic")
    fake._responses.append(LLMError("provider 401 - unauthorized"))
    fake.queue_success(_verdict_payload(0.85))
    config = make_llm_config(purpose="vision_rescore", retry_attempts=1)
    router = LLMRouter(
        config=config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
        db=db,
    )
    _stub_sample_frames(monkeypatch)

    final, outcome = await rescore_candidates(
        tenant_id=tenant.id,
        stream=stream,
        candidates=[a, b],
        db=db,
        router=router,
        frame_store=MemoryFrameStore(),
    )
    # One verdict landed, one failed; the one that failed kept its heuristic
    # rank (no rescore.score key).
    assert outcome.rescored == 1
    assert outcome.fatal_errors == 1
    assert any(c.evidence.get("rescore", {}).get("score") == 0.85 for c in final)


# ---------- Frame cache hits skip re-decode ----------


async def test_rescore_uses_frame_cache_when_present(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tenant = await TenantsRepo(db).create(name="Aldo")
    stream = _make_stream(tmp_path, tenant.id)
    await _seed_stream_row(db, stream)
    a = Candidate(timestamp=10.0, score=0.9, reason="voice", evidence={})
    await _seed_candidates(db, stream, [a])

    fake = FakeProvider("anthropic")
    fake.queue_success(_verdict_payload(0.7))
    config = make_llm_config(purpose="vision_rescore", retry_attempts=1)
    router = LLMRouter(
        config=config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
        db=db,
    )

    # Pre-warm the store with the exact source-ts grid the rescorer samples.
    store = MemoryFrameStore()
    spread_half = 1.5 / 2.0  # _DEFAULT_FRAME_SPREAD_S in vision_rescore.py
    step = 1.5 / 2  # n=3 -> step = spread / (n-1)
    for i in range(3):
        offset = -spread_half + i * step
        store.put(stream.id, max(0.0, a.timestamp + offset), b"prewarm")

    sampler_calls = _stub_sample_frames(monkeypatch)

    await rescore_candidates(
        tenant_id=tenant.id,
        stream=stream,
        candidates=[a],
        db=db,
        router=router,
        frame_store=store,
        n_frames_per_candidate=3,
    )
    assert sampler_calls == [], "cache hits should skip re-decoding"
    # The fake provider received 3 prewarm bytes.
    call = fake.calls[0]
    assert call["n_images"] == 3
    assert all(img.data == b"prewarm" for img in call["images"])


# ---------- Argument validation ----------


async def test_rescore_tenant_mismatch_raises(db: Database, tmp_path: Path) -> None:
    tenant = await TenantsRepo(db).create(name="Aldo")
    stream = _make_stream(tmp_path, tenant.id)
    await _seed_stream_row(db, stream)
    fake = FakeProvider("anthropic")
    config = make_llm_config(purpose="vision_rescore", retry_attempts=1)
    router = LLMRouter(
        config=config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
        db=db,
    )
    with pytest.raises(Exception, match="tenant mismatch"):
        await rescore_candidates(
            tenant_id="other_tenant",
            stream=stream,
            candidates=[],
            db=db,
            router=router,
        )


async def test_rescore_empty_candidates_returns_clean_outcome(
    db: Database, tmp_path: Path
) -> None:
    tenant = await TenantsRepo(db).create(name="Aldo")
    stream = _make_stream(tmp_path, tenant.id)
    await _seed_stream_row(db, stream)
    fake = FakeProvider("anthropic")
    config = make_llm_config(purpose="vision_rescore", retry_attempts=1)
    router = LLMRouter(
        config=config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
        db=db,
    )
    final, outcome = await rescore_candidates(
        tenant_id=tenant.id,
        stream=stream,
        candidates=[],
        db=db,
        router=router,
    )
    assert final == []
    assert outcome.rescored == 0
