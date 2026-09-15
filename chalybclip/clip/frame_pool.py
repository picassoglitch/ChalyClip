"""Shared per-clip frame sampling — Task 1c.

The three OpenCV passes that ran independently before this module —
`compute_smart_crop_box`, `analyze_framing`, `pick_thumbnail` — each
opened the source VOD, seeked to N timestamps, and decoded N frames.
On a multi-gigabyte source video that's three full seek-cascades worth
of wasted I/O per clip.

This module opens the source ONCE per clip, samples N frames spread
across the window, and returns a `ClipFrameBatch` that downstream
consumers reuse. Each consumer has a matching `_from_frames` entry
point next to its legacy one (`compute_smart_crop_box_from_frames`,
`analyze_framing_from_frames`, `pick_thumbnail_from_frames`); the cut
service prefers the batch path and falls back to the per-pass path
when the batch can't be built (test stubs, unreadable codecs, no
OpenCV installed).

DEFAULT_SAMPLE_N is sized to the most demanding consumer:
`pick_thumbnail` defaults to 10. Smart-crop and framing only need 5–6
but reading the same 10 BGR ndarrays costs nothing extra — the haar
cascade pass that follows is independent of how many frames the source
had to decode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chalybclip.logging import get_logger

_log = get_logger("chalybclip.clip.frame_pool")

# Matches pick_thumbnail's default sample count. Cropping/framing
# stride down from this same pool.
DEFAULT_SAMPLE_N = 10


@dataclass(frozen=True)
class ClipFrameBatch:
    """Decoded frames + source metadata for one clip window.

    `frames_bgr` are raw OpenCV BGR ndarrays at the source resolution
    — consumers run cv2 ops on them directly. `sample_times` is the
    source-relative timestamp each frame was decoded at (NOT
    clip-relative); the thumbnail picker stamps this into its
    breakdown so the on-disk filename matches the source coord
    system, same as the legacy pick_thumbnail.

    `__len__` returns the actual decoded count, which may be less
    than the requested sample_n when the source ran out of frames
    near the edges.
    """

    frames_bgr: list[Any]
    sample_times: list[float]
    source_w: int
    source_h: int
    fps: float

    def __len__(self) -> int:
        return len(self.frames_bgr)


def sample_clip_frames(
    video_path: Path,
    *,
    start_s: float,
    end_s: float,
    sample_n: int = DEFAULT_SAMPLE_N,
) -> ClipFrameBatch | None:
    """Decode `sample_n` frames spread across [start_s, end_s].

    Returns None on any failure (unreadable codec, OpenCV missing,
    metadata garbage). Callers MUST handle the None return and fall
    back to their legacy per-pass entry point. The frame-pool
    optimization is a speedup, not a correctness change — a None
    result must produce the same clip output as the pre-pool code.

    Behavior matches the legacy seek pattern exactly so a batch and a
    fresh per-pass open of the same source land on the same frames:
    timestamps spread evenly across the window, indices rounded to
    the nearest source frame, clamped to [0, total-1].
    """
    if end_s <= start_s or sample_n < 1 or not Path(video_path).exists():
        return None
    try:
        import cv2
    except Exception:  # noqa: BLE001
        # OpenCV not installed (test environments, minimal CI images).
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or total <= 0 or w <= 0 or h <= 0:
            # Test-stub videos land here (placeholder bytes have no
            # decodable metadata). Returning None makes the caller
            # fall back to the legacy path which also returns None /
            # raises ClipError — wired-up tests get the same result.
            return None

        if sample_n == 1:
            sample_times = [(start_s + end_s) / 2.0]
        else:
            step = (end_s - start_s) / (sample_n - 1)
            sample_times = [start_s + i * step for i in range(sample_n)]

        frames: list[Any] = []
        kept_times: list[float] = []
        for ts in sample_times:
            idx = round(ts * fps)
            idx = max(0, min(total - 1, idx))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frames.append(frame)
            kept_times.append(float(ts))

        if not frames:
            return None
        return ClipFrameBatch(
            frames_bgr=frames,
            sample_times=kept_times,
            source_w=w,
            source_h=h,
            fps=fps,
        )
    finally:
        cap.release()


__all__ = ["ClipFrameBatch", "DEFAULT_SAMPLE_N", "sample_clip_frames"]
