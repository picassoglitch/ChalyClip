"""OpenCV frame-diff motion energy.

Sample one frame per second, downscale to a small fixed size, and take
the mean of `|frame_t - frame_{t-1}|`. Higher value = more change between
adjacent samples = more action / camera movement / cut.

Output is normalized to [0, 1] (since 8-bit grayscale gives 0..255).
"""

from __future__ import annotations

from pathlib import Path

from chalybclip.errors import DetectionError

from .models import MotionFrame

_DOWNSCALE_W = 160
_DOWNSCALE_H = 90


def compute_motion_energy(video_path: Path, *, sample_rate_hz: float = 1.0) -> list[MotionFrame]:
    """Return one MotionFrame per sampled second (default = 1 fps)."""
    import cv2

    if not video_path.exists():
        raise DetectionError(f"video file missing: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise DetectionError(f"cv2 could not open {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            raise DetectionError(f"unknown fps for {video_path}")
        frame_skip = max(1, round(fps / sample_rate_hz))

        motions: list[MotionFrame] = []
        prev_small = None
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (_DOWNSCALE_W, _DOWNSCALE_H))
                if prev_small is not None:
                    diff = cv2.absdiff(small, prev_small)
                    energy = float(diff.mean()) / 255.0
                    ts = frame_idx / fps
                    motions.append(MotionFrame(ts=ts, energy=energy))
                prev_small = small
            frame_idx += 1
    finally:
        cap.release()

    return motions
