"""Vision-LLM smart-crop picker (P2 Task 5).

The Phase 1 picker uses Haar Cascade face detection — good enough when a
single face dominates the frame, weak when the clip is gameplay-heavy or
has multiple people. The vision-LLM picker reads compositional cues
(foreground objects, on-screen text, gaze direction) and returns a 9:16
crop box that keeps "the thing the viewer is supposed to see" centered.

When the LLM call fails (provider error, budget exhausted, decode hiccup),
the picker falls back to the Phase 1 face-detect heuristic. This keeps
free-tier streamers covered and means a single failed call never blocks
a clip from being cut.
"""

from __future__ import annotations

import base64
from pathlib import Path

import structlog

from chalybclip.errors import ClipError, LLMError
from chalybclip.llm import (
    CropBoxVerdict,
    FrameStore,
    LLMRouter,
    MultimodalImage,
)
from chalybclip.llm.config import Quality

from .models import SmartCropBox
from .smart_crop import compute_smart_crop_box

_log = structlog.get_logger(__name__)
_PURPOSE = "vision_rescore"  # Phase 2 reuses the rescore route
_ASPECT_W = 9
_ASPECT_H = 16

_SYSTEM_PROMPT = (
    "You pick a 9:16 vertical crop window for a moment in a livestream VOD.\n\n"
    "You see one frame from the source video. Decide what part of the frame "
    "would make the best 9:16 short. Prioritize:\n"
    "  1. The streamer's face if it's on camera.\n"
    "  2. The foreground subject of the action otherwise (game UI focal point, "
    "object being held, the person being argued with).\n"
    "  3. Avoid cropping through text overlays.\n\n"
    "Return the crop window as fractions of the source width:\n"
    "  * `x_frac` is the left edge, 0.0 = far left.\n"
    "  * `width_frac` is the crop width. Match a 9:16 aspect ratio for the "
    "frame's height; for a 16:9 source that's roughly 0.31.\n"
    "Keep `reason` to one short sentence."
)


async def compute_smart_crop_box_vision(
    *,
    tenant_id: str,
    stream_id: str,
    video_path: Path,
    start_s: float,
    end_s: float,
    router: LLMRouter,
    frame_store: FrameStore | None = None,
    quality: Quality | None = None,
) -> SmartCropBox:
    """Vision-LLM smart crop. Falls back to Phase 1 heuristic on any LLM error.

    The fallback is *quiet*: if the vision call dies for any reason
    (provider 5xx, budget, schema mismatch), we log + return the
    heuristic-derived box. From the caller's perspective there's never an
    "no crop available" outcome.
    """
    if end_s <= start_s:
        raise ClipError(f"end_s must be > start_s ({start_s} -> {end_s})")
    if not video_path.exists():
        raise ClipError(f"video file missing: {video_path}")

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ClipError(f"cv2 could not open {video_path}")
    try:
        source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if source_w <= 0 or source_h <= 0:
            raise ClipError(f"unusable video metadata for {video_path}")
    finally:
        cap.release()

    # If the source is already 9:16 or narrower, no LLM needed.
    crop_w = min(source_w, max(1, (source_h * _ASPECT_W) // _ASPECT_H))
    if crop_w >= source_w:
        return SmartCropBox(x=0, y=0, w=source_w, h=source_h)

    anchor_ts = (start_s + end_s) / 2.0
    image = _gather_anchor_frame(
        stream_id=stream_id,
        video_path=video_path,
        ts=anchor_ts,
        frame_store=frame_store,
    )
    if image is None:
        # No decodable frame -> let the heuristic try.
        return _fallback(video_path, start_s=start_s, end_s=end_s, reason="no_frame")

    user = (
        f"Source resolution: {source_w} x {source_h}.\n"
        f"Target crop aspect: 9:16 (so width should be ~{(source_h * _ASPECT_W) / (_ASPECT_H * source_w):.2f} of the source width).\n"
        f"Pick the crop window that keeps the most important subject centered."
    )
    try:
        verdict = await router.complete_multimodal(
            tenant_id=tenant_id,
            purpose=_PURPOSE,
            system=_SYSTEM_PROMPT,
            user=user,
            images=[image],
            schema=CropBoxVerdict,
            quality=quality,
        )
    except (LLMError, Exception) as e:
        _log.info("smart_crop_vision_fallback", error=str(e), reason="llm_error")
        return _fallback(video_path, start_s=start_s, end_s=end_s, reason="llm_error")

    return _verdict_to_box(verdict, source_w=source_w, source_h=source_h)


def _verdict_to_box(
    verdict: CropBoxVerdict, *, source_w: int, source_h: int
) -> SmartCropBox:
    """Translate fractional verdict to integer pixel coords + clamp to bounds.

    We only honor the model's *x position* - the width is always snapped to
    the canonical 9:16 width for this height. Letting the model also pick
    width opens the failure mode of a 0.5-width crop that breaks the aspect
    ratio downstream filters expect.
    """
    canonical_w = max(1, min(source_w, (source_h * _ASPECT_W) // _ASPECT_H))
    x = round(verdict.x_frac * source_w)
    x = max(0, min(source_w - canonical_w, x))
    return SmartCropBox(x=x, y=0, w=canonical_w, h=source_h)


def _fallback(
    video_path: Path, *, start_s: float, end_s: float, reason: str
) -> SmartCropBox:
    box = compute_smart_crop_box(video_path, start_s=start_s, end_s=end_s)
    _log.info("smart_crop_fallback", reason=reason, box=box.model_dump())
    return box


def _gather_anchor_frame(
    *,
    stream_id: str,
    video_path: Path,
    ts: float,
    frame_store: FrameStore | None,
) -> MultimodalImage | None:
    """Return one frame at `ts`, hitting the cache when present."""
    if frame_store is not None:
        cached = frame_store.get(stream_id, ts)
        if cached is not None:
            return MultimodalImage(media_type="image/jpeg", data=cached)

    from chalybclip.vision.frame_sampler import sample_frames

    try:
        blobs = sample_frames(video_path, ts=ts, n=1, spread_s=0.0)
    except Exception:
        return None
    if not blobs:
        return None
    blob = blobs[0]
    if frame_store is not None:
        frame_store.put(stream_id, ts, blob)
    return MultimodalImage(media_type="image/jpeg", data=blob)


# Silence unused-import for base64 (kept as a documentation hook for the
# byte-shape we'd encode if a future provider needed pre-encoded inputs).
_ = base64
