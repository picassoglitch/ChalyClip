"""Clip cutting + 9:16 reformat via two ffmpeg invocations.

Pipeline per candidate:
    1. Fast cut         → `_cut.mp4` (`-ss` before `-i`, `-c copy`; may snap to keyframe).
    2. Smart-crop       → choose a face-centered 9:16 box on the source frame.
    3. Auto-thumbnail   → pick the sharpest face-bearing frame, save JPEG.
    4. Reformat 9:16    → `clip.mp4` (`crop=W:H:X:Y,scale=Wo:Ho`, libx264 + aac).

Caption burning is deliberately skipped in Phase 0/1 — see PHASE_0.md.

Layout:
    <output_dir>/<stream_id>/
        clips_manifest.json
        clips/
            <clip_id>/
                clip.mp4
                thumbnail.jpg
                metadata.json
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from chalybclip.config import ClipConfig
from chalybclip.detect import Candidate
from chalybclip.errors import ClipError
from chalybclip.events import (
    CLIP_CUT_COMPLETED,
    CLIP_CUT_STARTED,
    CLIP_CUT_SUBSTEP,
    emit,
)
from chalybclip.ids import new_id
from chalybclip.ingest import Stream
from chalybclip.logging import get_logger

if TYPE_CHECKING:
    from chalybclip.db import Database
    from chalybclip.detect.framing import FramingVerdict
    from chalybclip.transcribe import Transcript

from chalybclip.resources import heavy_slot

from .encoders import (
    ffmpeg_thread_args,
    mark_nvenc_runtime_failure,
    pick_video_encoder_args,
)
from .frame_pool import ClipFrameBatch, sample_clip_frames
from .models import Clip, ClipManifest, SmartCropBox
from .smart_crop import (
    compute_smart_crop_box,
    compute_smart_crop_box_from_frames,
    crop_box_to_ffmpeg_filter,
)
from .thumbnail import pick_thumbnail, pick_thumbnail_from_frames, save_thumbnail
from .thumbnail_brand import pick_brand_kit_handle, render_branded_thumbnails

_log = get_logger("chalybclip.clip")


def cut_window(
    *,
    timestamp: float,
    pre_roll_s: float,
    post_roll_s: float,
    stream_duration_s: float,
    trigger_kind: str = "forward",
    retroactive_lookback_s: float | None = None,
) -> tuple[float, float, float]:
    """Compute (start, end, duration) for one candidate, clamped to VOD bounds.

    Two modes:
      * `trigger_kind="forward"` (default) — symmetric window centered loosely
        around the timestamp: [ts - pre_roll, ts + post_roll]. Matches the
        original behavior. Use for audio spikes, viral picks, forward voice
        triggers ('clipea esto').
      * `trigger_kind="retroactive"` — clip extends BACKWARD from the
        timestamp: [ts - retroactive_lookback, ts]. The natural shape for
        post-hoc voice markers ('clipeaste eso' — the moment ended, then
        the streamer flagged it). Falls back to forward semantics if
        `retroactive_lookback_s` isn't provided.
    """
    if stream_duration_s <= 0:
        raise ClipError(f"non-positive stream duration: {stream_duration_s}")
    if trigger_kind == "retroactive" and retroactive_lookback_s is not None:
        start = max(0.0, timestamp - retroactive_lookback_s)
        end = min(stream_duration_s, timestamp)
        duration = max(0.0, end - start)
        return start, end, duration
    start = max(0.0, timestamp - pre_roll_s)
    naive_end = min(stream_duration_s, timestamp + post_roll_s)
    duration = max(0.0, naive_end - start)
    return start, start + duration, duration


async def cut_clips(
    tenant_id: str,
    stream: Stream,
    candidates: list[Candidate],
    output_dir: Path,
    *,
    config: ClipConfig | None = None,
    force: bool = False,
    brand_kits: list[object] | None = None,
    transcript: "Transcript | None" = None,
    db: "Database | None" = None,
) -> list[Clip]:
    """Cut + reformat one clip per candidate. Idempotent on `clips_manifest.json`.

    Args:
        tenant_id: Must match `stream.tenant_id` (CLAUDE.md hard rule #1).
        stream: Source stream produced by `ingest_vod`.
        candidates: Detected candidates produced by `detect_voice_triggers`.
        output_dir: Root output dir (`./out`); clips go to `<output_dir>/<stream.id>/clips/`.
        config: Cut/reformat parameters. Defaults to `ClipConfig()`.
        force: Re-cut every clip even when the manifest exists.
        brand_kits: Optional parallel-to-candidates list of resolved BrandKitRow
            (or None per candidate). When set, the renderer burns the kit's
            primary social handle into the top-left of each clip. Typed loosely
            as `list[object] | None` to avoid a clip → db type cycle; the
            renderer reads attributes via `getattr` and falls back to no overlay
            on any missing field. Voice-markers spec slice D.1.
        transcript: Optional Transcript — when present (and the new
            `ClipConfig.dynamic_windowing` flag is True), each clip's
            start/end snaps to sentence boundaries instead of using the
            fixed pre_roll/post_roll. Slice G.1. Pass `None` to keep the
            legacy static-window behavior (existing tests use this path).
        db: Optional Database — when provided, the cut step emits
            `clip.cut.started` / `clip.cut.substep` / `clip.cut.completed`
            events through the standard event log so the dashboard can
            surface per-clip substeps + X/N progress. Pass None to keep
            the silent CLI/test behavior (Task 1d).

    Task 1a — candidates run concurrently up to `cfg.cut_concurrency`
    (default 3). The event loop stays responsive; each candidate's
    ffmpeg + OpenCV work runs in a worker thread bounded by a semaphore.
    Output is ORDER-PRESERVING — `asyncio.gather` returns in submission
    order, so the manifest keeps the same candidate→clip mapping as the
    serial path. Identical clip files, identical manifest, just faster.
    """
    if tenant_id != stream.tenant_id:
        raise ClipError(f"tenant mismatch: caller={tenant_id!r}, stream={stream.tenant_id!r}")

    cfg = config or ClipConfig()
    output_dir = Path(output_dir).resolve()
    stream_dir = output_dir / stream.id
    clips_dir = stream_dir / "clips"
    manifest_path = stream_dir / "clips_manifest.json"
    clips_dir.mkdir(parents=True, exist_ok=True)

    if not force and manifest_path.exists():
        try:
            return ClipManifest.model_validate_json(
                manifest_path.read_text("utf-8")
            ).clips
        except Exception as e:  # corrupt cache must not wedge the stream
            # Truncated/garbled manifest (crash mid-write on an old
            # deploy, disk full): fall through and re-cut instead of
            # failing every later run of this step.
            _log.warning(
                "clip.manifest_cache_corrupt",
                stream_id=stream.id, path=str(manifest_path),
                error=str(e)[:200],
            )

    if not stream.source_video_path.exists():
        raise ClipError(f"source video missing: {stream.source_video_path}")

    total = len(candidates)
    concurrency = max(1, int(getattr(cfg, "cut_concurrency", 3)))
    sem = asyncio.Semaphore(concurrency)

    # Task 1d — thread-safe substep emitter. Worker threads call this
    # with (clip_id, candidate_idx, step) at each substep boundary; we
    # schedule the emit() coroutine back on the orchestrator's loop
    # without blocking the worker. Fire-and-forget — a dropped event
    # is observability, not correctness.
    loop = asyncio.get_running_loop()

    def on_substep_threadsafe(
        clip_id: str, candidate_idx: int, step: str
    ) -> None:
        if db is None:
            return
        payload = {
            "stream_id": stream.id,
            "clip_id": clip_id,
            "candidate_index": candidate_idx,
            "total_candidates": total,
            "step": step,
        }
        try:
            asyncio.run_coroutine_threadsafe(
                emit(db, CLIP_CUT_SUBSTEP, payload), loop
            )
        except Exception:  # noqa: BLE001 — never block the worker
            pass

    async def _one(idx: int, candidate: Candidate) -> Clip | None:
        kit = brand_kits[idx] if brand_kits and idx < len(brand_kits) else None
        async with sem:
            if db is not None:
                await emit(
                    db, CLIP_CUT_STARTED,
                    {
                        "stream_id": stream.id,
                        "candidate_index": idx,
                        "total_candidates": total,
                    },
                )
            # Process-wide governor — bounds concurrent heavy ops across the
            # cut fan-out, operator renders, and the poll pipeline so they
            # can't collectively exhaust the host's thread/PID/memory ceiling
            # (the EAGAIN cascade). The local `sem` above still caps this
            # call's own fan-out; this caps it against everything else.
            async with heavy_slot():
                clip = await asyncio.to_thread(
                    _cut_one_sync,
                    tenant_id=tenant_id,
                    stream=stream,
                    candidate=candidate,
                    kit=kit,
                    clips_dir=clips_dir,
                    cfg=cfg,
                    transcript=transcript,
                    idx=idx,
                    on_substep=on_substep_threadsafe,
                )
            if clip is not None and db is not None:
                await emit(
                    db, CLIP_CUT_COMPLETED,
                    {
                        "stream_id": stream.id,
                        "clip_id": clip.id,
                        "candidate_index": idx,
                        "total_candidates": total,
                    },
                )
            return clip

    results = await asyncio.gather(
        *(_one(i, c) for i, c in enumerate(candidates))
    )
    clips: list[Clip] = [c for c in results if c is not None]

    manifest = ClipManifest(stream_id=stream.id, tenant_id=tenant_id, clips=clips)
    # Atomic write: a crash mid-write must not leave truncated JSON on the
    # name the idempotency gate trusts.
    tmp_path = manifest_path.with_suffix(".json.tmp")
    tmp_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp_path, manifest_path)
    return clips


def _cut_one_sync(
    *,
    tenant_id: str,
    stream: Stream,
    candidate: Candidate,
    kit: object | None,
    clips_dir: Path,
    cfg: ClipConfig,
    transcript: "Transcript | None",
    idx: int,
    on_substep: Callable[[str, int, str], None] | None = None,
) -> Clip | None:
    """Run the full cut pipeline for ONE candidate. Sync — must be
    dispatched via `asyncio.to_thread`. Returns None when the chosen
    window has zero duration (anchor past stream end), the Clip
    otherwise.

    Mirrors the original `_cut_all` per-iteration body 1:1 with three
    additions vs the pre-Task-1 code:

      * Task 1c — one frame-pool decode is shared across smart_crop /
        framing / thumbnail instead of three independent OpenCV opens.
      * Task 1d — `on_substep(clip_id, idx, step)` is called at the
        sampling / cropping / encoding / branding boundaries so the
        orchestrator can emit progress events. Fire-and-forget;
        raised exceptions are swallowed.
      * Slice O.55 duration reconciliation preserved verbatim — after
        the reformat, ffprobe the file and trust it over the requested
        duration if drift > 50ms.
    """

    def _emit(clip_id: str, step: str) -> None:
        if on_substep is None:
            return
        try:
            on_substep(clip_id, idx, step)
        except Exception:  # noqa: BLE001 — observability never breaks cut
            pass

    ev = candidate.evidence or {}
    trigger_kind = str(ev.get("trigger_kind", "forward"))
    retro_lookback = ev.get("retroactive_lookback_s")
    retro_lookback_f = float(retro_lookback) if isinstance(retro_lookback, int | float) else None

    # Slice G.1 — dynamic windowing when a transcript is available
    # AND the operator hasn't disabled it. Falls back to the legacy
    # static cut_window() so existing tests + callers (without
    # transcript) keep producing the same boundaries they did before.
    if transcript is not None and getattr(cfg, "dynamic_windowing", True):
        from .windowing import plan_clip_window

        plan = plan_clip_window(
            candidate=candidate,
            transcript=transcript,
            stream_duration_s=stream.duration_s,
            fallback_pre_roll_s=cfg.pre_roll_s,
            fallback_post_roll_s=cfg.post_roll_s,
            max_clip_duration_s=getattr(cfg, "max_clip_duration_s", 60.0),
        )
        start, end, duration = plan.start_s, plan.end_s, plan.duration_s
        window_plan_evidence: dict[str, object] | None = {
            "kind": plan.kind,
            "reason": plan.reason,
            "duration_s": round(plan.duration_s, 3),
        }
    else:
        start, end, duration = cut_window(
            timestamp=candidate.timestamp,
            pre_roll_s=cfg.pre_roll_s,
            post_roll_s=cfg.post_roll_s,
            stream_duration_s=stream.duration_s,
            trigger_kind=trigger_kind,
            retroactive_lookback_s=retro_lookback_f,
        )
        window_plan_evidence = None

    if duration <= 0.0:
        return None

    clip_id = new_id("clp")
    clip_dir = clips_dir / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    intermediate = clip_dir / "_cut.mp4"
    final = clip_dir / "clip.mp4"

    # Task 1c — share one decode across smart_crop / framing /
    # thumbnail. `sample_clip_frames` returns None on any failure (no
    # opencv, undecodable codec, test stub video); each _safe_*
    # helper sees `batch=None` and falls back to its legacy per-pass
    # open. Behavior identical, three OpenCV seeks → one.
    _emit(clip_id, "sampling")
    batch = sample_clip_frames(
        stream.source_video_path, start_s=start, end_s=end,
    )

    # When the shared frame batch failed to decode, `sample_clip_frames`
    # already opened (and failed on) cv2 once. The legacy fallbacks below
    # would each re-open the SAME source — on a resource-starved host that
    # turns one failed open into three per candidate (smart_crop + framing
    # + thumbnail), hammering the exhausted resource. `allow_reopen=False`
    # makes each helper skip the redundant re-open and degrade gracefully
    # (centered crop / conservative framing / no thumbnail).
    _emit(clip_id, "cropping")
    smart_box = _safe_smart_crop(
        video_path=stream.source_video_path,
        start_s=start, end_s=end, batch=batch, allow_reopen=False,
    )
    # Slice G.3 — framing intelligence. Decides whether this clip
    # should ship as 9:16 static crop / 9:16 tracking crop /
    # full-screen horizontal / already-mobile recenter / rejected.
    framing_verdict = _safe_analyze_framing(
        video_path=stream.source_video_path,
        start_s=start, end_s=end, batch=batch, allow_reopen=False,
    )
    thumbnail_path, raw_jpeg = _safe_thumbnail(
        video_path=stream.source_video_path,
        start_s=start, end_s=end,
        clip_dir=clip_dir, batch=batch, allow_reopen=False,
    )

    _emit(clip_id, "encoding")
    try:
        _ffmpeg_fast_cut(
            video_path=stream.source_video_path,
            start_s=start,
            duration_s=duration,
            out_path=intermediate,
            cfg=cfg,
        )
        _ffmpeg_reformat_9_16(
            in_path=intermediate,
            out_path=final,
            cfg=cfg,
            smart_box=smart_box,
            brand_kit=kit,
            framing_verdict=framing_verdict,
        )
    finally:
        if intermediate.exists():
            intermediate.unlink()

    _emit(clip_id, "branding")
    branded = (
        _safe_branded_thumbnails(
            source_jpeg=raw_jpeg, clip_dir=clip_dir, brand_kit=kit
        )
        if raw_jpeg is not None
        else {}
    )

    # Slice G.1 — when the new dynamic-windowing path picked the
    # boundaries, stamp the plan's metadata onto the candidate's
    # evidence so it persists with the clip and the dashboard can
    # display "reaction band 10-22s; snapped end to sentence boundary".
    # Slice G.3 — same pattern for the framing verdict: persisted
    # under `evidence.framing` so the dashboard renders the chosen
    # export-output label and G.5 reads it to drive the renderer.
    updated_evidence: dict[str, object] = dict(candidate.evidence or {})
    if window_plan_evidence is not None:
        updated_evidence["window_plan"] = window_plan_evidence
    if framing_verdict is not None:
        updated_evidence["framing"] = _framing_evidence(framing_verdict)
    if updated_evidence != (candidate.evidence or {}):
        candidate = candidate.model_copy(update={"evidence": updated_evidence})

    # Slice O.55 — ffprobe the produced clip and replace duration
    # with the ACTUAL file duration if it drifted past 50ms from
    # the requested value. This guarantees DB <-> on-disk parity
    # and prevents "preview shows 40s, export is 35s" type bugs
    # downstream (the recorder uses this duration as canonical).
    actual_duration = _ffprobe_duration_s(final)
    if actual_duration is not None:
        drift = abs(actual_duration - duration)
        if drift > 0.05:
            _log.warning(
                "clip.cut.duration_drift",
                requested_s=round(duration, 3),
                actual_s=round(actual_duration, 3),
                drift_ms=int(drift * 1000),
                clip_id=clip_id,
            )
            # Trust the file. Keep start_s as-is; recompute end_s.
            duration = actual_duration
            end = start + duration

    clip = Clip(
        id=clip_id,
        tenant_id=tenant_id,
        stream_id=stream.id,
        candidate=candidate,
        start_s=start,
        end_s=end,
        duration_s=duration,
        width=cfg.output_width,
        height=cfg.output_height,
        path=final,
        smart_crop_box=smart_box,
        thumbnail_path=thumbnail_path,
        thumbnail_16x9_path=branded.get("16x9"),
        thumbnail_9x16_path=branded.get("9x16"),
        thumbnail_1x1_path=branded.get("1x1"),
    )
    (clip_dir / "metadata.json").write_text(clip.model_dump_json(indent=2), encoding="utf-8")
    return clip


def _ffprobe_duration_s(path: Path) -> float | None:
    """Return the duration in seconds, or None if ffprobe isn't available
    or the file can't be parsed. Best-effort — never raises."""
    import json as _json
    import shutil
    import subprocess

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        proc = subprocess.run(  # noqa: S603 — args list
            [
                ffprobe, "-v", "error", "-print_format", "json",
                "-show_format", str(path),
            ],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if proc.returncode != 0:
            return None
        info = _json.loads(proc.stdout or "{}")
        return float(info.get("format", {}).get("duration") or 0.0) or None
    except Exception:  # noqa: BLE001 — never break cut on probe failure
        return None


def _safe_smart_crop(
    *,
    video_path: Path,
    start_s: float,
    end_s: float,
    batch: ClipFrameBatch | None = None,
    allow_reopen: bool = True,
) -> SmartCropBox | None:
    """Run smart_crop, log + skip on failure (e.g. unreadable test stub video).

    Task 1c — when a shared `ClipFrameBatch` is supplied, the haar
    cascade runs against the pre-decoded frames instead of re-opening
    the source. Falls back to the legacy per-pass open whenever the
    batch is missing or its `_from_frames` path raises.

    `allow_reopen=False` suppresses the per-pass cv2 re-open when no batch
    is available — the cut pipeline sets this after `sample_clip_frames`
    has already tried (and failed) to open the source, so we don't hammer
    an exhausted resource. Returns None → caller uses a centered crop.
    """
    try:
        if batch is not None and batch.frames_bgr:
            return compute_smart_crop_box_from_frames(batch)
        if not allow_reopen:
            return None
        return compute_smart_crop_box(video_path, start_s=start_s, end_s=end_s)
    except ClipError as e:
        _log.warning("smart_crop.skipped", reason=str(e))
        return None
    except Exception as e:  # noqa: BLE001 — never break cut on a cv2 quirk
        _log.warning("smart_crop.skipped", reason=str(e))
        return None


def _safe_analyze_framing(
    *,
    video_path: Path,
    start_s: float,
    end_s: float,
    batch: ClipFrameBatch | None = None,
    allow_reopen: bool = True,
) -> "FramingVerdict | None":
    """Slice G.3 — run analyze_framing inside a guard so unreadable
    test-stub videos / missing opencv never break the cut step. The
    framing module's own internal guards already return a conservative
    fallback; this layer just adds a belt-and-suspenders catch.

    Task 1c — when `batch` is supplied, runs the framing classifier
    against the shared decoded frames instead of opening the source
    video again.

    `allow_reopen=False` suppresses the per-pass cv2 re-open when no batch
    is available (the cut pipeline already tried via `sample_clip_frames`).
    Returns None → caller ships without a framing verdict."""
    try:
        if batch is not None and batch.frames_bgr:
            from chalybclip.detect.framing import analyze_framing_from_frames
            return analyze_framing_from_frames(batch)
        if not allow_reopen:
            return None
        from chalybclip.detect.framing import analyze_framing
        return analyze_framing(
            video_path=video_path, start_s=start_s, end_s=end_s,
        )
    except Exception as e:  # noqa: BLE001 — never break cut
        _log.warning("framing.skipped", reason=str(e))
        return None


def _framing_evidence(verdict: "FramingVerdict") -> dict[str, object]:
    """Compact JSON-friendly representation of the framing verdict
    to stash under `candidate.evidence.framing`. The dashboard renders
    `recommended_output` / `confidence` / `reason` for the operator;
    G.5 reads `recommended_output` + `safe_crop_box` to drive export."""
    crop = verdict.safe_crop_box
    crop_dict: dict[str, float] | None = (
        {"x": crop.x, "y": crop.y, "w": crop.w, "h": crop.h}
        if crop is not None
        else None
    )
    return {
        "source_orientation": verdict.source_orientation,
        "recommended_output": verdict.recommended_output,
        "confidence": round(verdict.confidence, 3),
        "safe_crop_box": crop_dict,
        "subject_count": len(verdict.subject_boxes),
        "reason": verdict.reason,
    }


def _safe_thumbnail(
    *,
    video_path: Path,
    start_s: float,
    end_s: float,
    clip_dir: Path,
    batch: ClipFrameBatch | None = None,
    allow_reopen: bool = True,
) -> tuple[Path | None, bytes | None]:
    """Run pick_thumbnail + save_thumbnail; log + skip on failure.

    Returns `(path, raw_jpeg_bytes)` so the branded compositor can reuse
    the same decoded frame without another OpenCV pass over the source
    video. On failure both values are None and the caller skips the
    branded-variant step too.

    Task 1c — picks from the shared frame batch when available;
    otherwise falls through to the legacy per-pass open.

    `allow_reopen=False` suppresses the per-pass cv2 re-open when no batch
    is available (the cut pipeline already tried via `sample_clip_frames`).
    Returns `(None, None)` → caller ships without a thumbnail.
    """
    try:
        if batch is not None and batch.frames_bgr:
            jpeg, _ts, _bd = pick_thumbnail_from_frames(batch)
        elif not allow_reopen:
            return None, None
        else:
            jpeg, _ts, _bd = pick_thumbnail(video_path, start_s=start_s, end_s=end_s)
        return save_thumbnail(clip_dir, jpeg), jpeg
    except ClipError as e:
        _log.warning("thumbnail.skipped", reason=str(e))
        return None, None
    except Exception as e:  # noqa: BLE001 — never break cut
        _log.warning("thumbnail.skipped", reason=str(e))
        return None, None


def _safe_branded_thumbnails(
    *, source_jpeg: bytes, clip_dir: Path, brand_kit: object | None
) -> dict[str, Path]:
    """Composite the 16:9 / 9:16 / 1:1 brand-kit thumbnail variants.

    `brand_kit` is loosely typed as `object | None` to keep the clip
    module free of a hard dependency on `chalybclip.db` (same approach
    `_brand_kit_drawtext_filter` uses for the slice D.1 handle overlay).
    When `brand_kit` is None we fall back to a neutral grey scheme so
    the variants still render — the publisher upload still benefits
    from the aspect-correct thumbnails even without brand colors.

    Returns a `{aspect: path}` dict keyed by `"16x9"` / `"9x16"` / `"1x1"`.
    Missing keys mean that variant failed to render (per-variant try/
    except in the compositor)."""
    primary = (
        str(getattr(brand_kit, "primary_color", None) or "#1F2937")
        if brand_kit is not None
        else "#1F2937"
    )
    accent = (
        str(getattr(brand_kit, "accent_color", None) or "#F59E0B")
        if brand_kit is not None
        else "#F59E0B"
    )
    text = (
        str(getattr(brand_kit, "text_color", None) or "#FFFFFF")
        if brand_kit is not None
        else "#FFFFFF"
    )
    handle = pick_brand_kit_handle(
        handle_tiktok=getattr(brand_kit, "handle_tiktok", None),
        handle_youtube=getattr(brand_kit, "handle_youtube", None),
        handle_instagram=getattr(brand_kit, "handle_instagram", None),
        handle_kick=getattr(brand_kit, "handle_kick", None),
    )
    paths = render_branded_thumbnails(
        source_jpeg=source_jpeg,
        clip_dir=clip_dir,
        handle=handle,
        primary_color=primary,
        accent_color=accent,
        text_color=text,
    )
    out: dict[str, Path] = {}
    for p in paths:
        # render_branded_thumbnails names files thumb_<aspect>.jpg — peel
        # the aspect token out of the filename so we don't re-import the
        # private VARIANTS tuple just to map back.
        stem = p.stem  # 'thumb_16x9'
        if stem.startswith("thumb_"):
            out[stem.removeprefix("thumb_")] = p
    return out


def _ffmpeg_fast_cut(
    *,
    video_path: Path,
    start_s: float,
    duration_s: float,
    out_path: Path,
    cfg: ClipConfig | None = None,
) -> None:
    """Cut a precise window out of `video_path`.

    Slice O.55 — was stream-copy with `-ss` BEFORE `-i`. That uses
    "input seek" which is fast but keyframe-aligned: ffmpeg seeks to
    the nearest keyframe BEFORE start_s, then `-t` reads from there.
    On a stream where the GOP is 5-10s (typical screen-recording or
    h264 baseline output) this could produce output that's 0-5s
    longer than requested AND starts up to 5s early. The downstream
    DB stored the REQUESTED duration_s, so preview reported one
    length while the file actually had another — that's the root
    cause of "preview 40s, export 35s" type drift.

    Behavior: re-encode the cut so seek is sample-accurate.
      - `-ss start_s` BEFORE `-i` is still here for the fast scrubber
        (decoder skips to the keyframe before start_s)
      - `-accurate_seek` + re-encode → output starts exactly at start_s
      - The downstream reformat step encodes again (so this is one
        EXTRA encode on the cut, ~0.5s libx264 / sub-100ms NVENC on
        a typical 30s clip)

    Task 1b — when NVENC is available + preferred, BOTH this cut and
    the downstream reformat use h264_nvenc, so this extra pass is
    nearly free on a GPU host. The libx264 fallback still uses the
    O.55 baseline (`fast` + `crf 19`).

    `cfg=None` is the legacy-call path (kept for explicit-call
    callers without a config object); defaults match the O.55
    pre-Task-1 ladder so behavior is unchanged.
    """
    effective = cfg or ClipConfig()

    def _build(encoder_args: list[str]) -> list[str]:
        # Task 1b — encoder picker selects libx264 (fast/CRF 19) or
        # h264_nvenc (p5/CQ matched). The wrapper handles fallback
        # when NVENC fails at runtime (Railway CPU + libcuda missing).
        return [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            # Fast input seek to the keyframe BEFORE start_s. The
            # `-accurate_seek` flag (default in modern ffmpeg) combined
            # with the re-encode below means the actual output starts
            # exactly at start_s — the keyframe before is decoded into
            # the buffer but only frames at/after start_s land in output.
            "-ss",
            f"{start_s:.3f}",
            "-accurate_seek",
            "-i",
            str(video_path),
            "-t",
            f"{duration_s:.3f}",
            *encoder_args,
            "-pix_fmt",
            "yuv420p",
            # Audio: re-encode to AAC@48k since social platforms want
            # that uniformly, and accurate cutting requires re-encoding
            # the audio at the cut point anyway (sample-accurate trim).
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            str(out_path),
        ]

    _run_encode_with_fallback(
        cfg=effective,
        build_cmd=_build,
        what=f"accurate cut at {start_s:.3f}s",
    )


def _ffmpeg_reformat_9_16(
    *,
    in_path: Path,
    out_path: Path,
    cfg: ClipConfig,
    smart_box: SmartCropBox | None = None,
    brand_kit: object | None = None,
    framing_verdict: "FramingVerdict | None" = None,
) -> None:
    """Reformat the cut window into the export aspect ratio.

    Slice G.4c — branching on source orientation:

      * vertical / square source → 9:16 crop (smart_box if we have a face
        track, else center crop). This is the existing path.
      * horizontal source → DON'T crop. Scale to fit width-wise and pad
        the top/bottom with a heavily-blurred copy of the same frame so
        the operator gets a familiar "TikTok with blurred letterbox"
        look + zero content lost off the sides. Previously horizontal
        sources had 70% of the frame chopped off and the operator had to
        re-trim everything manually.

    When `brand_kit` is provided and the kit carries a primary social handle
    + we can resolve a font on this OS, append a `drawtext` filter that burns
    the handle into the top-left corner using the kit's accent color. Slice
    D.1 of the voice-markers spec. Logo burn-in lands in D.3 alongside the
    AI logo generator.

    Failures to resolve a font or read kit attributes are silent — the
    clip still renders, just without the overlay.
    """
    is_horizontal = (
        framing_verdict is not None
        and getattr(framing_verdict, "source_orientation", None) == "horizontal"
    )

    if is_horizontal:
        # Slice G.4c (refined) — black-letterbox with a modest 20% zoom.
        # Operator feedback after the initial blurred-letterbox ship:
        # the foreground was too small + the blurred bars drew the eye
        # away from the actual content. New behavior:
        #   - scale source 20% larger than "fit inside frame", so it
        #     fills more of the vertical
        #   - center-crop horizontally so the result is exactly 1080 wide
        #     (lose ~10% off each side; the operator chose this trade
        #     because their content is center-framed)
        #   - pad the remaining top/bottom with solid black
        # Single-stream filter — faster than the two-stream blur path.
        w, h = cfg.output_width, cfg.output_height
        # The 1.2 factor is the "20% zoom" the operator asked for.
        # `ZOOM=1.2` means foreground occupies 1.2× the fit-into-frame
        # size; sides spill past `w` and get center-cropped.
        zoom = 1.2
        # Compute the target foreground height for an h-tall frame +
        # zoom. force_original_aspect_ratio=decrease + this scale yields
        # a 16:9 source landing at (1080*1.2)=1296 wide, 729 tall — 10%
        # crop each side, ~38% vertical fill.
        fg_w = int(w * zoom)
        fg_h = int(h * zoom)
        vf = (
            f"scale={fg_w}:{fg_h}:force_original_aspect_ratio=decrease,"
            f"crop={w}:'min(ih,{h})':(iw-{w})/2:0,"
            f"pad={w}:{h}:0:({h}-ih)/2:black"
        )
    elif smart_box is not None:
        vf = crop_box_to_ffmpeg_filter(
            smart_box, output_w=cfg.output_width, output_h=cfg.output_height
        )
    else:
        vf = f"crop=ih*9/16:ih,scale={cfg.output_width}:{cfg.output_height}"

    overlay = _brand_kit_drawtext_filter(brand_kit, output_w=cfg.output_width)
    if overlay:
        vf = f"{vf},{overlay}"

    # The horizontal-source path uses multi-stream filter syntax
    # ([bg], [fg], overlay) which needs -filter_complex, not -vf.
    # Detect via the [ character — every multi-stream filter graph
    # mentions at least one labeled pad.
    filter_flag = "-filter_complex" if "[" in vf else "-vf"

    # Task 1b — encoder selection.
    # - If cfg.encoder is overridden away from "libx264", honor the
    #   operator's choice with the legacy preset/crf args (preserves
    #   the libx265 path used by callers + tests).
    # - Else if NVENC is on this host and the operator hasn't disabled
    #   it via `cfg.prefer_nvenc=False`, swap in h264_nvenc.
    # - Else libx264. No-NVENC hosts are byte-identical to pre-Task-1.
    # _run_encode_with_fallback transparently retries with libx264 if
    # NVENC blows up at runtime (Railway CPU + libcuda missing).
    def _build(encoder_args: list[str]) -> list[str]:
        return [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(in_path),
            filter_flag,
            vf,
            *encoder_args,
            "-c:a",
            "aac",
            str(out_path),
        ]

    _run_encode_with_fallback(
        cfg=cfg,
        build_cmd=_build,
        what=f"9:16 reformat -> {out_path.name}",
    )


def _brand_kit_drawtext_filter(
    brand_kit: object | None, *, output_w: int
) -> str | None:
    """Slice N.5 — DISABLED.

    Was: burned the brand_kit's @handle into the top-left corner of
    every clip.mp4 during the cut step (D.1-era branding intent).

    Why it's off now: the L.2 platform-sim chrome + M.14 official
    Kick banner overlay both render the operator's handle in better
    typography, at better positions, with platform-specific styling.
    The cut-step drawtext was redundant AND visually conflicting —
    operator-reported "second bad kick banner" was THIS burn-in.

    Clips cut from this point forward won't have the redundant
    drawtext. Existing clips that already shipped through the old
    cut path will still show it until they're re-cut.
    """
    _ = (brand_kit, output_w)  # unused — disabled per N.5
    return None


def _find_system_font() -> Path | None:
    """Resolve a usable TTF/OTF for ffmpeg drawtext across OSes.

    Cached after first hit. None when nothing matches — caller must
    tolerate it. The bundled-font path (ship our own Inter.ttf) is a
    slice E upgrade for cross-machine consistency.
    """
    global _CACHED_FONT
    if not isinstance(_CACHED_FONT, _UnsetFont):
        return _CACHED_FONT
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
        Path("/System/Library/Fonts/HelveticaNeue.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    ]
    for c in candidates:
        if c.exists():
            _CACHED_FONT = c
            return c
    _CACHED_FONT = None
    return None


class _UnsetFont:
    pass


_UNSET_FONT = _UnsetFont()
_CACHED_FONT: Path | None | _UnsetFont = _UNSET_FONT


def _run_encode_with_fallback(
    *,
    cfg: ClipConfig,
    build_cmd: "Callable[[list[str]], list[str]]",
    what: str,
) -> None:
    """Run an ffmpeg encode and retry with libx264 if NVENC blows up.

    Task 1b shipped NVENC selection but the probe was too optimistic:
    `h264_nvenc` shows up in `ffmpeg -encoders` whenever ffmpeg was
    BUILT with NVENC support, regardless of whether libcuda.so.1 is
    actually loadable at runtime. On Railway CPU hosts that produced
    "Cannot load libcuda.so.1" on the first cut.

    The probe in `encoders.py` is now more careful (it actually
    attempts a tiny encode), but this wrapper is the belt-and-braces:
    if pick_video_encoder_args returned NVENC args AND the encode
    fails for any reason, we flip the runtime-disable flag, log it,
    and re-run the same ffmpeg invocation with libx264 args. Any
    subsequent clip in the run sees `has_nvenc()=False` and skips
    NVENC outright.

    `build_cmd(encoder_args)` is the caller's cmd-builder that
    accepts a list of `-c:v ...` flags and splices them into the
    full argv. Keeps the encoder choice out of the call site so the
    surrounding ffmpeg structure stays declarative.
    """
    encoder_args = pick_video_encoder_args(cfg)
    used_nvenc = "h264_nvenc" in encoder_args
    cmd = build_cmd(encoder_args)
    try:
        _run_ffmpeg(cmd, what=what)
    except ClipError as e:
        if not used_nvenc:
            raise
        # NVENC was used and the encode failed. Permanently disable
        # NVENC for the rest of this process and retry once with the
        # libx264 fallback. If the fallback ALSO fails, that's a real
        # cut error and we let it surface.
        mark_nvenc_runtime_failure(reason=str(e))
        _log.warning(
            "clip.encode.nvenc_fallback",
            what=what, error=str(e)[:300],
        )
        fallback_args = [
            "-c:v", "libx264",
            "-preset", cfg.preset,
            "-crf", str(cfg.crf),
            *ffmpeg_thread_args(cfg),
        ]
        cmd = build_cmd(fallback_args)
        _run_ffmpeg(cmd, what=f"{what} (libx264 fallback)")


# Hard ceiling on one ffmpeg invocation. Clips are bounded (≤ a few
# minutes of source), so an encode past this is a wedged ffmpeg — kill it
# instead of pinning a cut_concurrency slot AND a process-wide heavy_slot
# forever, which would stall every later pipeline run.
_FFMPEG_TIMEOUT_S = 1800


def _run_ffmpeg(cmd: list[str], *, what: str) -> None:
    try:
        subprocess.run(
            cmd, check=True, capture_output=True, timeout=_FFMPEG_TIMEOUT_S
        )
    except FileNotFoundError as e:
        raise ClipError("ffmpeg binary not found on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise ClipError(
            f"ffmpeg {what} killed after {_FFMPEG_TIMEOUT_S}s (wedged encode)"
        ) from e
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", "replace") if e.stderr else ""
        raise ClipError(f"ffmpeg {what} failed: {stderr}") from e


def load_clips(stream_dir: Path) -> ClipManifest:
    """Read the saved clips manifest back from disk."""
    path = Path(stream_dir) / "clips_manifest.json"
    if not path.exists():
        raise ClipError(f"clips manifest not found at {path}")
    return ClipManifest.model_validate_json(path.read_text("utf-8"))
