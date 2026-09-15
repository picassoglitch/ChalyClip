"""Local vision pipeline orchestrator.

`analyze_video` runs the three local detectors (scene cuts, motion
energy, face/emotion) and folds their per-event/per-frame outputs into
one `VisualSignal` row per second of stream timeline.

The output mirrors the `visual_signals` DB table 1:1; if a `Database`
is provided, rows land there too. JSON persistence at
`<stream_dir>/visual_signals.json` is always written for offline review
(matches the Phase 0/1 dual-write pattern).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from chalybclip.errors import DetectionError
from chalybclip.ingest import Stream

from .face_emotion import detect_face_emotions
from .models import (
    FaceFrame,
    MotionFrame,
    SceneCut,
    VisualSignal,
    VisualSignalTrack,
)
from .motion import compute_motion_energy
from .scene_detect import detect_scene_cuts

if TYPE_CHECKING:
    from chalybclip.db import Database


def visual_signals_path(stream_dir: Path) -> Path:
    return Path(stream_dir) / "visual_signals.json"


async def analyze_video(
    tenant_id: str,
    stream: Stream,
    *,
    output_dir: Path,
    db: Database | None = None,
    force: bool = False,
) -> VisualSignalTrack:
    """Run scene cut + motion + face/emotion on `stream.source_video_path`.

    Idempotent: if `<stream_dir>/visual_signals.json` exists, return the
    cached track unless `force=True`. Writes the JSON to disk and (when
    db is provided) the `visual_signals` table.
    """
    if tenant_id != stream.tenant_id:
        raise DetectionError(f"tenant mismatch: caller={tenant_id!r}, stream={stream.tenant_id!r}")
    if not stream.source_video_path.exists():
        raise DetectionError(f"video file missing: {stream.source_video_path}")

    stream_dir = Path(output_dir) / stream.id
    out_path = visual_signals_path(stream_dir)
    if not force and out_path.exists():
        return VisualSignalTrack.model_validate_json(out_path.read_text("utf-8"))

    # Each detector is CPU-bound; off-load to a thread so the event loop
    # stays responsive (e.g. for the FastAPI dashboard in Task 9).
    cuts, motions, faces = await asyncio.gather(
        asyncio.to_thread(detect_scene_cuts, stream.source_video_path),
        asyncio.to_thread(compute_motion_energy, stream.source_video_path),
        asyncio.to_thread(detect_face_emotions, stream.source_video_path),
    )

    signals = _fold_to_per_second(
        cuts=cuts,
        motions=motions,
        faces=faces,
        duration_s=stream.duration_s,
    )
    track = VisualSignalTrack(stream_id=stream.id, tenant_id=tenant_id, signals=signals)

    stream_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(track.model_dump_json(indent=2), encoding="utf-8")

    if db is not None:
        from chalybclip.db import VisualSignalsRepo

        await VisualSignalsRepo(db).replace_for_stream(stream.id, track)

    return track


def _fold_to_per_second(
    *,
    cuts: list[SceneCut],
    motions: list[MotionFrame],
    faces: list[FaceFrame],
    duration_s: float,
) -> list[VisualSignal]:
    """Collapse the three per-detector streams to one row per integer second."""
    n_seconds = max(1, round(duration_s))
    by_sec: dict[int, VisualSignal] = {
        sec: VisualSignal(ts_offset_s=float(sec)) for sec in range(n_seconds)
    }

    for cut in cuts:
        sec = int(cut.ts)
        if 0 <= sec < n_seconds:
            by_sec[sec] = by_sec[sec].model_copy(update={"scene_cut": True})

    # For each second, take the MAX motion energy (more representative of
    # action than the mean — a single high-motion frame is the signal).
    motion_per_sec: dict[int, float] = {}
    for m in motions:
        sec = int(m.ts)
        if not (0 <= sec < n_seconds):
            continue
        if sec not in motion_per_sec or m.energy > motion_per_sec[sec]:
            motion_per_sec[sec] = m.energy
    for sec, energy in motion_per_sec.items():
        by_sec[sec] = by_sec[sec].model_copy(update={"motion_energy": energy})

    # For each second pick the highest-confidence face label (or None).
    face_per_sec: dict[int, FaceFrame] = {}
    for f in faces:
        sec = int(f.ts)
        if not (0 <= sec < n_seconds) or not f.has_face:
            continue
        existing = face_per_sec.get(sec)
        if existing is None or f.confidence > existing.confidence:
            face_per_sec[sec] = f
    for sec, f in face_per_sec.items():
        by_sec[sec] = by_sec[sec].model_copy(update={"face_emotion": f.emotion})

    return [by_sec[sec] for sec in sorted(by_sec)]


def load_visual_signals(stream_dir: Path) -> VisualSignalTrack | None:
    """Read the saved track, or None when missing — silent degrade."""
    path = visual_signals_path(stream_dir)
    if not path.exists():
        return None
    return VisualSignalTrack.model_validate_json(path.read_text("utf-8"))
