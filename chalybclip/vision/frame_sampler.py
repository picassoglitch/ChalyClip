"""Sample N frames around a timestamp.

Used by Phase 1's analyze_video step (informational) and by Phase 2's
multimodal scoring (the LLM "sees" these frames). For Phase 1 we only
need to extract them on demand; the cache + DB index lands with the
multimodal router method in Task 8.
"""

from __future__ import annotations

from pathlib import Path

from chalybclip.errors import DetectionError


def sample_frames(
    video_path: Path,
    ts: float,
    n: int = 5,
    *,
    spread_s: float | None = None,
) -> list[bytes]:
    """Return `n` JPEG-encoded frames centered on `ts` seconds.

    Args:
        video_path: Path to a video readable by OpenCV.
        ts: Center timestamp in seconds.
        n: How many frames to return. Must be >= 1.
        spread_s: Total span (seconds) covered by the N frames. Defaults
            to 1.0 — so for n=5 the frames are at ts ± 0.5 s every 0.25 s.

    Returns the frames as JPEG bytes so callers can hand them to vision
    LLMs without a second round-trip to disk.
    """
    if n < 1:
        raise DetectionError(f"n must be >= 1, got {n}")
    if not video_path.exists():
        raise DetectionError(f"video file missing: {video_path}")

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise DetectionError(f"cv2 could not open {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or total_frames <= 0:
            raise DetectionError(f"unknown fps/length for {video_path}")
        duration_s = total_frames / fps

        effective_spread = spread_s if spread_s is not None else 1.0
        if n == 1:
            offsets = [0.0]
        else:
            step = effective_spread / (n - 1)
            offsets = [-(effective_spread / 2.0) + i * step for i in range(n)]
        target_times = [max(0.0, min(duration_s - (1.0 / fps), ts + offset)) for offset in offsets]

        frames: list[bytes] = []
        for target_ts in target_times:
            target_frame_idx = round(target_ts * fps)
            target_frame_idx = max(0, min(total_frames - 1, target_frame_idx))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            frames.append(bytes(encoded.tobytes()))
        return frames
    finally:
        cap.release()


def save_frames(stream_dir: Path, ts: float, frames: list[bytes]) -> list[Path]:
    """Persist sampled frames to `<stream_dir>/frames/<ts>_<idx>.jpg`."""
    out_dir = Path(stream_dir) / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx, blob in enumerate(frames):
        path = out_dir / f"{ts:09.3f}_{idx:02d}.jpg"
        path.write_bytes(blob)
        paths.append(path)
    return paths
