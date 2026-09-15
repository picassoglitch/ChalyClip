"""PySceneDetect wrapper.

PySceneDetect's `detect()` returns a list of (start, end) timecodes for
each scene. The cut points are the start timestamps of every scene
*except the first* — those are the boundaries we care about for clip
detection.
"""

from __future__ import annotations

from pathlib import Path

from chalybclip.errors import DetectionError

from .models import SceneCut


def detect_scene_cuts(video_path: Path, *, threshold: float = 27.0) -> list[SceneCut]:
    """Return scene-cut timestamps for `video_path`.

    Args:
        video_path: Path to a video file readable by ffmpeg / OpenCV.
        threshold: ContentDetector threshold. Lower = more sensitive
            (more cuts). 27 is PySceneDetect's default, tuned for
            typical content.
    """
    from scenedetect import ContentDetector, detect

    if not video_path.exists():
        raise DetectionError(f"video file missing: {video_path}")

    try:
        scenes = detect(str(video_path), ContentDetector(threshold=threshold))
    except Exception as e:
        raise DetectionError(f"PySceneDetect failed on {video_path}: {e}") from e

    # scenes is List[Tuple[FrameTimecode, FrameTimecode]]; the cut points
    # are the start timecode of each scene after the first.
    return [SceneCut(ts=float(scene[0].seconds)) for scene in scenes[1:]]
