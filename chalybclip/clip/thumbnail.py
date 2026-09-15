"""Auto-thumbnail picker.

Samples frames evenly across the clip window, scores each with a
sharpness x face heuristic, and writes the highest-scoring JPEG to
`<clip_dir>/thumbnail.jpg`. The score is:

    sharpness_score * (1 + face_present)

where:
  * `sharpness_score` is `min(1, laplacian_variance / 500)` - Laplacian
    variance is the standard sharpness/blur metric. 500 is a reasonable
    "this looks sharp" threshold for 1080p frames.
  * `face_present` adds a 1.0 bonus when OpenCV's Haar Cascade finds at
    least one face in the frame, doubling the score for face-driven
    thumbnails.

Phase 2 will swap in a vision-LLM picker that also weighs composition
and on-screen text. The on-disk shape (`<clip_dir>/thumbnail.jpg`) and
the `clips.thumbnail_frame_path` column don't change.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from chalybclip.errors import ClipError

if TYPE_CHECKING:
    from .frame_pool import ClipFrameBatch

_DEFAULT_SHARPNESS_FLOOR = 500.0


def pick_thumbnail(
    video_path: Path,
    *,
    start_s: float,
    end_s: float,
    sample_n: int = 10,
) -> tuple[bytes, float, dict[str, float]]:
    """Return `(jpeg_bytes, source_ts, score_breakdown)` for the best frame.

    `score_breakdown` is a dict with `sharpness`, `has_face`, and `score`
    so callers (or tests) can inspect what won the pick.
    """
    if end_s <= start_s:
        raise ClipError(f"end_s must be > start_s ({start_s} -> {end_s})")
    if sample_n < 1:
        raise ClipError(f"sample_n must be >= 1, got {sample_n}")
    if not video_path.exists():
        raise ClipError(f"video file missing: {video_path}")

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ClipError(f"cv2 could not open {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or total_frames <= 0:
            raise ClipError(f"unusable video metadata for {video_path}")

        if sample_n == 1:
            sample_times = [(start_s + end_s) / 2.0]
        else:
            step = (end_s - start_s) / (sample_n - 1)
            sample_times = [start_s + i * step for i in range(sample_n)]

        from chalybclip.vision.haar import haar_face_detector

        # A missing/limited Haar detector treats every frame as face-absent
        # rather than raise; the picker still runs on sharpness.
        face_detector = haar_face_detector()

        best: tuple[float, bytes, float, dict[str, float]] | None = None
        for ts in sample_times:
            frame_idx = round(ts * fps)
            frame_idx = max(0, min(total_frames - 1, frame_idx))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            sharpness = min(1.0, lap_var / _DEFAULT_SHARPNESS_FLOOR)

            has_face = 0.0
            if face_detector is not None:
                faces = face_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )
                has_face = 1.0 if len(faces) > 0 else 0.0

            score = sharpness * (1.0 + has_face)
            if best is None or score > best[0]:
                ok, jpeg = cv2.imencode(".jpg", frame)
                if not ok:
                    continue
                breakdown = {
                    "sharpness": sharpness,
                    "has_face": has_face,
                    "score": score,
                    "ts": float(ts),
                }
                best = (score, bytes(jpeg.tobytes()), float(ts), breakdown)

        if best is None:
            raise ClipError(f"no decodable frames in clip window {start_s}-{end_s}")
        _, jpeg_bytes, source_ts, breakdown = best
        return jpeg_bytes, source_ts, breakdown
    finally:
        cap.release()


def pick_thumbnail_from_frames(
    batch: "ClipFrameBatch",
) -> tuple[bytes, float, dict[str, float]]:
    """Same contract as `pick_thumbnail` but reads decoded frames from
    a shared batch instead of re-opening the source video. Task 1c.

    Identical sharpness × face-presence heuristic, identical breakdown
    shape — the cut service swaps one for the other transparently.
    """
    if not batch.frames_bgr:
        raise ClipError("empty frame batch — cannot pick thumbnail")
    import cv2

    from chalybclip.vision.haar import haar_face_detector

    face_detector = haar_face_detector()

    best: tuple[float, bytes, float, dict[str, float]] | None = None
    for frame, ts in zip(batch.frames_bgr, batch.sample_times, strict=True):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness = min(1.0, lap_var / _DEFAULT_SHARPNESS_FLOOR)
        has_face = 0.0
        if face_detector is not None:
            faces = face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            has_face = 1.0 if len(faces) > 0 else 0.0
        score = sharpness * (1.0 + has_face)
        if best is None or score > best[0]:
            ok, jpeg = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            breakdown = {
                "sharpness": sharpness,
                "has_face": has_face,
                "score": score,
                "ts": float(ts),
            }
            best = (score, bytes(jpeg.tobytes()), float(ts), breakdown)

    if best is None:
        raise ClipError("no encodable frames in batch")
    _, jpeg_bytes, source_ts, breakdown = best
    return jpeg_bytes, source_ts, breakdown


def save_thumbnail(clip_dir: Path, jpeg_bytes: bytes) -> Path:
    """Persist the picked thumbnail to `<clip_dir>/thumbnail.jpg`."""
    path = Path(clip_dir) / "thumbnail.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jpeg_bytes)
    return path
