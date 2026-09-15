"""Vision-LLM auto-thumbnail picker (P2 Task 5).

The Phase 1 picker uses sharpness x face-presence as a heuristic. The
vision-LLM picker reads composition, expression, and on-screen text -
the model returns the index of the strongest frame from a set of
candidates.

Falls back to the Phase 1 picker on any LLM error so a flaky vision
call never blocks a clip from being thumbnailed.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from chalybclip.errors import ClipError, LLMError
from chalybclip.llm import (
    FrameStore,
    LLMRouter,
    MultimodalImage,
    ThumbnailPickVerdict,
)
from chalybclip.llm.config import Quality

from .thumbnail import pick_thumbnail

_log = structlog.get_logger(__name__)
_PURPOSE = "vision_rescore"

_SYSTEM_PROMPT = (
    "You pick the single best thumbnail frame for a short-form clip.\n\n"
    "You see N candidate frames sampled across the clip. Pick the one most "
    "likely to make a viewer click. Strong picks usually have:\n"
    "  * a clear human face with readable expression\n"
    "  * sharp focus (not motion-blurred)\n"
    "  * a single subject (not too busy)\n\n"
    "Return the 0-based index of the frame you'd use plus a one-sentence reason."
)


async def pick_thumbnail_vision(
    *,
    tenant_id: str,
    stream_id: str,
    video_path: Path,
    start_s: float,
    end_s: float,
    router: LLMRouter,
    frame_store: FrameStore | None = None,
    sample_n: int = 6,
    quality: Quality | None = None,
) -> tuple[bytes, float, dict[str, float | str]]:
    """Return `(jpeg_bytes, source_ts, breakdown)`.

    Falls back to the Phase 1 sharpness x face-detect picker on any LLM
    error (provider 5xx, schema mismatch, budget exhausted, ...). The
    fallback's score breakdown is returned with no rescore_index/reason
    fields so callers can detect the fallback in the panel.
    """
    if end_s <= start_s:
        raise ClipError(f"end_s must be > start_s ({start_s} -> {end_s})")
    if sample_n < 1:
        raise ClipError(f"sample_n must be >= 1, got {sample_n}")
    if not video_path.exists():
        raise ClipError(f"video file missing: {video_path}")


    # Build the sample-time grid up front so we can map the LLM's chosen
    # index back to a source timestamp.
    if sample_n == 1:
        sample_times = [(start_s + end_s) / 2.0]
    else:
        step = (end_s - start_s) / (sample_n - 1)
        sample_times = [start_s + i * step for i in range(sample_n)]

    blobs = _gather_frames(
        stream_id=stream_id,
        video_path=video_path,
        sample_times=sample_times,
        frame_store=frame_store,
    )
    if not blobs:
        return _fallback(
            video_path=video_path,
            start_s=start_s,
            end_s=end_s,
            sample_n=sample_n,
            reason="no_frames",
        )

    images = [MultimodalImage(media_type="image/jpeg", data=b) for b in blobs]
    user = (
        f"You see {len(images)} candidate frames sampled evenly across "
        f"the clip's {end_s - start_s:.1f}s window. "
        f"Pick the strongest one (return index, 0..{len(images) - 1})."
    )
    try:
        verdict = await router.complete_multimodal(
            tenant_id=tenant_id,
            purpose=_PURPOSE,
            system=_SYSTEM_PROMPT,
            user=user,
            images=images,
            schema=ThumbnailPickVerdict,
            quality=quality,
        )
    except (LLMError, Exception) as e:
        _log.info("thumbnail_vision_fallback", error=str(e), reason="llm_error")
        return _fallback(
            video_path=video_path,
            start_s=start_s,
            end_s=end_s,
            sample_n=sample_n,
            reason="llm_error",
        )

    idx = max(0, min(len(images) - 1, verdict.index))
    chosen_blob = blobs[idx]
    chosen_ts = sample_times[idx]
    breakdown: dict[str, float | str] = {
        "score": 1.0,  # vision picker doesn't compute a numeric score
        "ts": float(chosen_ts),
        "rescore_index": float(idx),
        "rescore_reason": verdict.reason,
    }
    return chosen_blob, float(chosen_ts), breakdown


def _gather_frames(
    *,
    stream_id: str,
    video_path: Path,
    sample_times: list[float],
    frame_store: FrameStore | None,
) -> list[bytes]:
    blobs: list[bytes] = []
    missing_indices: list[int] = []
    if frame_store is not None:
        for idx, ts in enumerate(sample_times):
            cached = frame_store.get(stream_id, ts)
            if cached is not None:
                blobs.append(cached)
            else:
                blobs.append(b"")
                missing_indices.append(idx)
    else:
        blobs = [b""] * len(sample_times)
        missing_indices = list(range(len(sample_times)))

    if missing_indices:
        from chalybclip.vision.frame_sampler import sample_frames

        # Take the union; sampling at the clip center with spread = window
        # length gives us coverage equivalent to the per-ts grid (close
        # enough for thumbnails - vision picker compares the set).
        center = sum(sample_times) / len(sample_times)
        spread = max(0.0, sample_times[-1] - sample_times[0])
        try:
            sampled = sample_frames(
                video_path,
                ts=center,
                n=len(missing_indices),
                spread_s=spread if len(missing_indices) > 1 else 0.0,
            )
        except Exception:
            return []
        for idx, blob in zip(missing_indices, sampled, strict=False):
            blobs[idx] = blob
            if frame_store is not None:
                frame_store.put(stream_id, sample_times[idx], blob)
    return [b for b in blobs if b]


def _fallback(
    *,
    video_path: Path,
    start_s: float,
    end_s: float,
    sample_n: int,
    reason: str,
) -> tuple[bytes, float, dict[str, float | str]]:
    blob, ts, breakdown = pick_thumbnail(
        video_path, start_s=start_s, end_s=end_s, sample_n=sample_n
    )
    casted: dict[str, float | str] = dict(breakdown)
    casted["fallback_reason"] = reason
    return blob, ts, casted
