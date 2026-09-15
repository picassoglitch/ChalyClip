"""ClipBreakdown — the "why this clip?" math."""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from chalybclip.clip import clip_breakdown
from chalybclip.db import (
    CandidatesRepo,
    ClipsRepo,
    Database,
    StreamsRepo,
    TenantsRepo,
    TranscriptsRepo,
    VisualSignalsRepo,
    apply_migrations,
)
from chalybclip.db.models import (
    CandidateRow,
    ClipRow,
    StreamRow,
    TranscriptRow,
)
from chalybclip.errors import ClipError
from chalybclip.tenancy import bound_tenant
from chalybclip.vision import VisualSignal, VisualSignalTrack


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "bdown.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


async def _seed(db: Database) -> str:
    """Seed one tenant + stream + clip + candidate. Returns the tenant id."""
    tenant = await TenantsRepo(db).create(name="Aldo")
    with bound_tenant(tenant.id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_b",
                tenant_id=tenant.id,
                vod_url="x",
                platform="kick",
                title=None,
                channel=None,
                duration_s=120.0,
                source_video_path="/tmp/v",
                source_audio_path="/tmp/a",
                status="ingested",
                created_at=_now(),
            )
        )
        await CandidatesRepo(db).upsert_many(
            [
                CandidateRow(
                    id="cnd_b",
                    stream_id="str_b",
                    tenant_id=tenant.id,
                    ts=10.0,
                    score=0.6,
                    reason="audio",
                    evidence={},
                    created_at=_now(),
                )
            ]
        )
        await ClipsRepo(db).upsert_many(
            [
                ClipRow(
                    id="clp_b",
                    stream_id="str_b",
                    tenant_id=tenant.id,
                    candidate_id="cnd_b",
                    start_s=10.0,
                    end_s=20.0,
                    duration_s=10.0,
                    width=1080,
                    height=1920,
                    path="/tmp/c.mp4",
                    status="cut",
                    created_at=_now(),
                )
            ]
        )
    return tenant.id


async def test_breakdown_returns_nones_when_no_signals_or_transcript(
    db: Database,
) -> None:
    tenant_id = await _seed(db)
    with bound_tenant(tenant_id):
        b = await clip_breakdown(db, "clp_b")
    assert b.clip_id == "clp_b"
    assert b.duration_s == 10.0
    assert b.motion_score is None
    assert b.face_presence is None
    assert b.speaking_intensity is None
    assert b.reaction_confidence is None
    assert b.rescore_delta is None
    assert b.heuristic_reason == "audio"
    assert b.heuristic_score == 0.6


async def test_breakdown_computes_motion_and_face_from_visual_signals(
    db: Database,
) -> None:
    tenant_id = await _seed(db)
    # Seed visual_signals: 10s of per-second rows in [10, 20)
    signals = []
    for sec in range(10, 20):
        signals.append(
            VisualSignal(
                ts_offset_s=float(sec),
                scene_cut=False,
                face_emotion="neutral" if sec >= 13 else None,  # 7 of 10 with face
                motion_energy=0.10 + (sec - 10) * 0.05,
                text_changed=False,
            )
        )
    track = VisualSignalTrack(
        stream_id="str_b", tenant_id=tenant_id, signals=signals
    )
    with bound_tenant(tenant_id):
        await VisualSignalsRepo(db).replace_for_stream("str_b", track)
        b = await clip_breakdown(db, "clp_b")

    # Motion = avg of [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55] = 0.325
    assert b.motion_score is not None
    assert abs(b.motion_score - 0.325) < 1e-6
    # Face presence: 7 of 10 seconds.
    assert b.face_presence == pytest.approx(0.7)


async def test_breakdown_computes_speaking_intensity_from_transcript(
    db: Database,
) -> None:
    tenant_id = await _seed(db)
    # Build a transcript where 30 words fall inside the [10, 20) clip window.
    segments_payload = [
        {
            "start_ts": 0.0,
            "end_ts": 30.0,
            "text": "...",
            "words": [
                {"text": f"w{i}", "ts": 11.0 + i * 0.25, "end_ts": 11.1 + i * 0.25, "prob": 0.9}
                for i in range(30)
            ]
            + [
                # Out-of-window words that must NOT contribute to the count.
                {"text": "before", "ts": 5.0, "end_ts": 5.5, "prob": 0.9},
                {"text": "after", "ts": 25.0, "end_ts": 25.5, "prob": 0.9},
            ],
        }
    ]
    with bound_tenant(tenant_id):
        await TranscriptsRepo(db).upsert(
            TranscriptRow(
                stream_id="str_b",
                tenant_id=tenant_id,
                language="es",
                duration_s=120.0,
                model="medium",
                segments_json=json.dumps(segments_payload),
                created_at=_now(),
            )
        )
        b = await clip_breakdown(db, "clp_b")

    # 30 words / 10s = 3.0 words/sec
    assert b.speaking_intensity == pytest.approx(3.0)


async def test_breakdown_includes_rescore_when_present(db: Database) -> None:
    tenant_id = await _seed(db)
    with bound_tenant(tenant_id):
        await CandidatesRepo(db).update_rescore(
            "cnd_b",
            rescore_score=0.85,
            rescore_reason="strong shock onset visible at the anchor",
            rescore_model="claude-opus-4-7",
        )
        b = await clip_breakdown(db, "clp_b")

    assert b.reaction_confidence == 0.85
    assert b.rescore_delta == pytest.approx(0.85 - 0.6)
    assert b.rescore_reason and "shock" in b.rescore_reason


async def test_breakdown_unknown_clip_raises(db: Database) -> None:
    tenant = await TenantsRepo(db).create(name="Aldo")
    with bound_tenant(tenant.id), pytest.raises(ClipError, match="clip not found"):
        await clip_breakdown(db, "clp_does_not_exist")


