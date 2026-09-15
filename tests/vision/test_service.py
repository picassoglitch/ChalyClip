"""Tests for analyze_video — the orchestrator that folds the three
detector outputs into per-second VisualSignal rows.

The CV libraries are stubbed here so the test runs in milliseconds and
doesn't depend on synthetic-video timing quirks. The detector modules
themselves have their own tests (test_scene_detect, test_motion,
test_face_emotion).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from chalybclip.db import Database, TenantsRepo, VisualSignalsRepo, apply_migrations
from chalybclip.errors import DetectionError
from chalybclip.ingest import Stream
from chalybclip.tenancy import bound_tenant
from chalybclip.vision import (
    FaceFrame,
    MotionFrame,
    SceneCut,
    VisualSignalTrack,
    analyze_video,
    load_visual_signals,
)
from chalybclip.vision import service as vision_service


def _stub_detectors(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuts: list[SceneCut] | None = None,
    motions: list[MotionFrame] | None = None,
    faces: list[FaceFrame] | None = None,
) -> None:
    monkeypatch.setattr(vision_service, "detect_scene_cuts", lambda _p: cuts or [])
    monkeypatch.setattr(vision_service, "compute_motion_energy", lambda _p: motions or [])
    monkeypatch.setattr(vision_service, "detect_face_emotions", lambda _p: faces or [])


def _stream(tmp_path: Path, *, tenant_id: str = "ten_a") -> Stream:
    stream_id = "str_01TEST"
    stream_dir = tmp_path / stream_id / "source"
    stream_dir.mkdir(parents=True, exist_ok=True)
    video_path = stream_dir / "video.mp4"
    audio_path = stream_dir / "audio.wav"
    video_path.write_bytes(b"fake_video_bytes")
    audio_path.write_bytes(b"fake_audio_bytes")
    return Stream(
        id=stream_id,
        tenant_id=tenant_id,
        vod_url="https://kick.com/c/videos/1",
        platform="kick",
        title="t",
        channel="c",
        duration_s=5.0,
        source_video_path=video_path,
        source_audio_path=audio_path,
    )


def test_folds_three_detector_streams_into_per_second_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_detectors(
        monkeypatch,
        cuts=[SceneCut(ts=2.5)],
        motions=[
            MotionFrame(ts=0.0, energy=0.05),
            MotionFrame(ts=1.0, energy=0.04),
            MotionFrame(ts=2.0, energy=0.7),  # spike here
            MotionFrame(ts=3.0, energy=0.2),
        ],
        faces=[
            FaceFrame(ts=0.0, has_face=True, emotion="neutral", confidence=0.6),
            FaceFrame(ts=2.5, has_face=True, emotion="smile", confidence=0.9),
        ],
    )
    track = asyncio.run(
        analyze_video(
            tenant_id="ten_a",
            stream=_stream(tmp_path),
            output_dir=tmp_path,
        )
    )
    assert isinstance(track, VisualSignalTrack)
    # 5 s duration -> 5 rows
    assert len(track.signals) == 5

    by_sec = {s.ts_offset_s: s for s in track.signals}
    assert by_sec[2.0].scene_cut is True  # int(2.5) == 2
    assert by_sec[2.0].motion_energy == pytest.approx(0.7)
    assert by_sec[0.0].face_emotion == "neutral"
    assert by_sec[2.0].face_emotion == "smile"
    assert by_sec[4.0].motion_energy is None  # no detector output for sec 4


def test_persists_visual_signals_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_detectors(monkeypatch, cuts=[SceneCut(ts=1.0)])
    track = asyncio.run(
        analyze_video(
            tenant_id="ten_a",
            stream=_stream(tmp_path),
            output_dir=tmp_path,
        )
    )
    loaded = load_visual_signals(tmp_path / track.stream_id)
    assert loaded is not None
    assert loaded == track


def test_idempotent_when_json_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second call returns the cached track without re-invoking detectors."""
    calls = {"scene": 0, "motion": 0, "faces": 0}

    def cnt_scene(_p: Path) -> list[SceneCut]:
        calls["scene"] += 1
        return [SceneCut(ts=1.0)]

    def cnt_motion(_p: Path) -> list[MotionFrame]:
        calls["motion"] += 1
        return []

    def cnt_faces(_p: Path) -> list[FaceFrame]:
        calls["faces"] += 1
        return []

    monkeypatch.setattr(vision_service, "detect_scene_cuts", cnt_scene)
    monkeypatch.setattr(vision_service, "compute_motion_energy", cnt_motion)
    monkeypatch.setattr(vision_service, "detect_face_emotions", cnt_faces)

    stream = _stream(tmp_path)
    asyncio.run(analyze_video(tenant_id="ten_a", stream=stream, output_dir=tmp_path))
    asyncio.run(analyze_video(tenant_id="ten_a", stream=stream, output_dir=tmp_path))
    assert calls == {"scene": 1, "motion": 1, "faces": 1}


def test_force_reruns_detectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invocations = 0

    def cnt(_p: Path) -> list[SceneCut]:
        nonlocal invocations
        invocations += 1
        return []

    monkeypatch.setattr(vision_service, "detect_scene_cuts", cnt)
    monkeypatch.setattr(vision_service, "compute_motion_energy", lambda _p: [])
    monkeypatch.setattr(vision_service, "detect_face_emotions", lambda _p: [])

    stream = _stream(tmp_path)
    asyncio.run(analyze_video(tenant_id="ten_a", stream=stream, output_dir=tmp_path))
    asyncio.run(
        analyze_video(
            tenant_id="ten_a", stream=stream, output_dir=tmp_path, force=True
        )
    )
    assert invocations == 2


def test_writes_to_db_when_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_detectors(
        monkeypatch,
        cuts=[SceneCut(ts=1.0)],
        motions=[MotionFrame(ts=2.0, energy=0.5)],
        faces=[FaceFrame(ts=3.0, has_face=True, emotion="smile", confidence=0.9)],
    )

    async def go() -> list:
        db_path = tmp_path / "test.db"
        db = Database(db_path)
        await apply_migrations(db)
        await TenantsRepo(db).create(tenant_id="ten_a", name="A")
        try:
            with bound_tenant("ten_a"):
                stream = _stream(tmp_path)
                # Streams FK requires the row to exist.
                from chalybclip.db.adapters import stream_to_row
                from chalybclip.db.repos import StreamsRepo

                await StreamsRepo(db).upsert(stream_to_row(stream))
                await analyze_video(
                    tenant_id="ten_a",
                    stream=stream,
                    output_dir=tmp_path,
                    db=db,
                )
                return await VisualSignalsRepo(db).list_for_stream(stream.id)
        finally:
            await db.close()

    rows = asyncio.run(go())
    assert len(rows) == 5
    by_sec = {r["ts_offset_s"]: r for r in rows}
    assert by_sec[1.0]["scene_cut"] is True
    assert by_sec[2.0]["motion_energy"] == pytest.approx(0.5)
    assert by_sec[3.0]["face_emotion"] == "smile"


def test_db_rerun_replaces_prior_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running with --force should swap in the new track without
    duplicating rows on the stream."""
    _stub_detectors(monkeypatch, cuts=[SceneCut(ts=1.0)])

    async def go() -> int:
        db_path = tmp_path / "test.db"
        db = Database(db_path)
        await apply_migrations(db)
        await TenantsRepo(db).create(tenant_id="ten_a", name="A")
        try:
            with bound_tenant("ten_a"):
                stream = _stream(tmp_path)
                from chalybclip.db.adapters import stream_to_row
                from chalybclip.db.repos import StreamsRepo

                await StreamsRepo(db).upsert(stream_to_row(stream))
                await analyze_video(
                    tenant_id="ten_a",
                    stream=stream,
                    output_dir=tmp_path,
                    db=db,
                )
                # Re-run with different cuts and force=True.
                monkeypatch.setattr(
                    vision_service,
                    "detect_scene_cuts",
                    lambda _p: [SceneCut(ts=2.0)],
                )
                await analyze_video(
                    tenant_id="ten_a",
                    stream=stream,
                    output_dir=tmp_path,
                    db=db,
                    force=True,
                )
                rows = await VisualSignalsRepo(db).list_for_stream(stream.id)
                # Still 5 rows (one per second), and the cut moved from 1.0 -> 2.0.
                cut_secs = [r["ts_offset_s"] for r in rows if r["scene_cut"]]
                return cut_secs
        finally:
            await db.close()

    cut_secs = asyncio.run(go())
    assert cut_secs == [2.0]


def test_tenant_mismatch_raises(tmp_path: Path) -> None:
    with pytest.raises(DetectionError, match="tenant mismatch"):
        asyncio.run(
            analyze_video(
                tenant_id="ten_b",
                stream=_stream(tmp_path, tenant_id="ten_a"),
                output_dir=tmp_path,
            )
        )


def test_missing_video_raises(tmp_path: Path) -> None:
    stream = _stream(tmp_path)
    stream.source_video_path.unlink()
    with pytest.raises(DetectionError, match="video file missing"):
        asyncio.run(
            analyze_video(
                tenant_id="ten_a",
                stream=stream,
                output_dir=tmp_path,
            )
        )
