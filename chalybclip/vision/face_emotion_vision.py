"""Vision-LLM face emotion detector (P2 Task 6).

Phase 1 ships face-presence-only via Haar Cascade (`emotion="neutral"` when
a face is detected, `None` otherwise). Phase 2 plugs the vision LLM in:
the model gets one frame per sample tick and returns a real emotion label.

Falls back to the Phase 1 detector on any LLM error so the pipeline keeps
moving when the budget governor refuses or the provider is flapping.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from chalybclip.errors import DetectionError, LLMError
from chalybclip.llm import (
    FaceEmotionVerdict,
    FrameStore,
    LLMRouter,
    MultimodalImage,
)
from chalybclip.llm.config import Quality

from .face_emotion import detect_face_emotions
from .models import FaceEmotion, FaceFrame

_log = structlog.get_logger(__name__)
_PURPOSE = "vision_rescore"
_VALID_EMOTIONS: frozenset[str] = frozenset(
    {"neutral", "smile", "laugh", "shock", "anger", "sad"}
)

_SYSTEM_PROMPT = (
    "You're labeling the streamer's facial expression in one frame.\n\n"
    "If you can see a clear human face, set has_face=true and pick one of:\n"
    "  neutral | smile | laugh | shock | anger | sad\n\n"
    "If no face is visible (off-camera, back of head, full UI overlay), "
    "set has_face=false and emotion=null.\n\n"
    "Confidence (0.0-1.0) reflects how sure you are about both the face's "
    "presence AND the emotion label. Values below 0.5 imply the labeler "
    "should ignore this frame."
)


async def detect_face_emotions_vision(
    *,
    tenant_id: str,
    stream_id: str,
    video_path: Path,
    router: LLMRouter,
    frame_store: FrameStore | None = None,
    sample_rate_hz: float = 0.5,
    quality: Quality | None = None,
) -> list[FaceFrame]:
    """Return one `FaceFrame` per sampled frame using a vision LLM.

    Sample rate defaults to 0.5 Hz (one frame every 2s). Higher rates
    burn premium tokens for diminishing returns - the 1Hz vision pass
    already gets us a usable emotion track for clip selection.

    On any LLM error mid-pass, falls through to the Phase 1 Haar
    Cascade detector for the *whole* video. Frames already labeled by
    the LLM in the same call are discarded - we don't want a half-LLM,
    half-heuristic track.
    """
    if not video_path.exists():
        raise DetectionError(f"video file missing: {video_path}")

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise DetectionError(f"cv2 could not open {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            raise DetectionError(f"unknown fps for {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_s = total_frames / fps if total_frames else 0.0
    finally:
        cap.release()

    if duration_s <= 0:
        return []

    interval_s = 1.0 / sample_rate_hz
    n_samples = max(1, int(duration_s * sample_rate_hz))
    sample_times = [min(duration_s, i * interval_s) for i in range(n_samples)]

    frames: list[FaceFrame] = []
    for ts in sample_times:
        try:
            verdict = await _classify_one(
                tenant_id=tenant_id,
                stream_id=stream_id,
                video_path=video_path,
                ts=ts,
                router=router,
                frame_store=frame_store,
                quality=quality,
            )
        except (LLMError, Exception) as e:
            _log.info(
                "face_emotion_vision_fallback",
                error=str(e),
                reason="llm_error",
                samples_completed=len(frames),
            )
            return detect_face_emotions(video_path, sample_rate_hz=sample_rate_hz)

        if not verdict.has_face:
            frames.append(FaceFrame(ts=ts, has_face=False))
            continue
        emotion: FaceEmotion | None = None
        if verdict.emotion in _VALID_EMOTIONS:
            emotion = verdict.emotion  # type: ignore[assignment]
        frames.append(
            FaceFrame(
                ts=ts,
                has_face=True,
                emotion=emotion,
                confidence=float(verdict.confidence),
            )
        )
    return frames


async def _classify_one(
    *,
    tenant_id: str,
    stream_id: str,
    video_path: Path,
    ts: float,
    router: LLMRouter,
    frame_store: FrameStore | None,
    quality: Quality | None,
) -> FaceEmotionVerdict:
    image = _gather_frame(
        stream_id=stream_id,
        video_path=video_path,
        ts=ts,
        frame_store=frame_store,
    )
    if image is None:
        # No frame -> behave as "no face" rather than escalating; the
        # heuristic detector would also report no-face here.
        return FaceEmotionVerdict(has_face=False, emotion=None, confidence=0.0)
    return await router.complete_multimodal(
        tenant_id=tenant_id,
        purpose=_PURPOSE,
        system=_SYSTEM_PROMPT,
        user=f"Frame sampled at {ts:.1f}s into the source stream.",
        images=[image],
        schema=FaceEmotionVerdict,
        quality=quality,
    )


def _gather_frame(
    *,
    stream_id: str,
    video_path: Path,
    ts: float,
    frame_store: FrameStore | None,
) -> MultimodalImage | None:
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
