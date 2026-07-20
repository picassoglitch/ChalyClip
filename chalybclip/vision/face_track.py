"""Dense per-frame subject tracking for the reframe pipeline.

The static smart-crop / framing passes sample 6-10 frames across a clip
and keep only the LARGEST face per frame — enough to pick one fixed crop,
but blind to WHERE the subject is at each moment. Active-speaker reframe
needs temporal resolution: a chosen subject center sampled densely enough
(~3 fps) that the renderer can pan a crop window to follow it.

This module opens the source window once, decodes it sequentially (cheap —
`grab()` skips the frames we don't sample), runs the shared Haar detector
on each sampled frame, and returns one chosen-subject center-x per sample.

Active-speaker selection (baseline)
-----------------------------------
With no audio-visual model wired yet, "the speaker" is approximated by a
size + temporal-persistence heuristic: among the faces on a frame, prefer
the one closest to the previously-tracked subject (lock-on / no flicker),
ignoring faces much smaller than the frame's largest (background extras).
On a single-face talking-head clip — the common streamer case — this is
simply "follow the face", which is exactly the win over a static average.

The real upgrade (Light-ASD / TalkNet lip-motion + audio correlation) is a
localized swap: `_choose_subject_cx` is the ONLY chooser, and it returns a
source-pixel center. Replace its body and the rest of the pipeline is
unchanged. Kept CPU-only + dependency-free (OpenCV Haar) on purpose so it
runs on the operator's box with zero new installs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nexoclip.logging import get_logger

_log = get_logger("nexoclip.vision.face_track")

# Detection runs on a downscaled copy — Haar cost scales with pixel count
# and a face is still comfortably detectable at ~480px wide. Centers map
# back to source pixels via the scale factor.
_DETECT_WIDTH = 480
# Hard cap on decoded samples so a long clip window can't explode decode
# time. At 3 fps this covers a 60s clip; longer clips just sample coarser.
_MAX_SAMPLES = 200
# A face smaller than this fraction of the frame's largest face is treated
# as a background extra, never the tracked subject.
_MIN_RELATIVE_FACE_AREA = 0.5


@dataclass(frozen=True)
class SubjectSamples:
    """Per-timestamp chosen-subject center over a clip window.

    `times_s` are CLIP-relative seconds; `centers_x` is the subject's
    center-x in SOURCE pixels, or None where no face was detected on that
    sample. The two lists are parallel and equal-length. `coverage` is the
    fraction of samples that found a face — the caller uses it to decide
    whether a reframe track is trustworthy or it should fall back.
    """

    times_s: list[float]
    centers_x: list[float | None]
    source_w: int
    source_h: int

    @property
    def coverage(self) -> float:
        if not self.centers_x:
            return 0.0
        hits = sum(1 for c in self.centers_x if c is not None)
        return hits / len(self.centers_x)


def sample_subject_track(
    video_path: Path,
    *,
    start_s: float,
    end_s: float,
    target_fps: float = 3.0,
) -> SubjectSamples | None:
    """Decode [start_s, end_s] at ~`target_fps` and return the chosen
    subject center per sample.

    Returns None on any failure (no OpenCV, unreadable codec, garbage
    metadata, no detector) — callers MUST treat None as "no track, use the
    static crop". Never raises: reframe is an enhancement, never a step
    that can fail a cut.
    """
    if end_s <= start_s or target_fps <= 0.0:
        return None
    if not Path(video_path).exists():
        return None
    try:
        import cv2
    except Exception:  # OpenCV optional in some environments
        return None

    from nexoclip.vision.haar import haar_face_detector

    detector = haar_face_detector()
    if detector is None:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or total <= 0 or sw <= 0 or sh <= 0:
            return None

        start_frame = max(0, min(total - 1, round(start_s * fps)))
        end_frame = max(start_frame, min(total - 1, round(end_s * fps)))
        stride = max(1, round(fps / target_fps))
        # If the window is long, coarsen the stride so we stay under the
        # sample cap rather than truncating the tail (a truncated track
        # would pan correctly then freeze halfway through the clip).
        span = end_frame - start_frame
        if span // stride + 1 > _MAX_SAMPLES:
            stride = max(stride, span // _MAX_SAMPLES + 1)

        scale = sw / float(_DETECT_WIDTH) if sw > _DETECT_WIDTH else 1.0
        detect_w = int(sw / scale)
        detect_h = int(sh / scale)
        min_face = max(24, detect_h // 12)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        times_s: list[float] = []
        centers_x: list[float | None] = []
        prev_cx: float | None = None
        offset = 0
        while start_frame + offset <= end_frame and len(times_s) < _MAX_SAMPLES:
            if offset % stride == 0:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                cx = _choose_subject_cx(
                    frame,
                    detector=detector,
                    prev_cx=prev_cx,
                    scale=scale,
                    detect_w=detect_w,
                    detect_h=detect_h,
                    min_face=min_face,
                    cv2=cv2,
                )
                times_s.append((start_frame + offset) / fps - start_s)
                centers_x.append(cx)
                if cx is not None:
                    prev_cx = cx
            else:
                if not cap.grab():
                    break
            offset += 1

        if not times_s:
            return None
        return SubjectSamples(
            times_s=times_s, centers_x=centers_x, source_w=sw, source_h=sh
        )
    except Exception as e:  # never break a cut on a cv2 quirk
        _log.warning("subject_track.failed", reason=str(e))
        return None
    finally:
        cap.release()


def _choose_subject_cx(
    frame: object,
    *,
    detector: object,
    prev_cx: float | None,
    scale: float,
    detect_w: int,
    detect_h: int,
    min_face: int,
    cv2: object,
) -> float | None:
    """Return the chosen subject's center-x in SOURCE pixels, or None.

    THE active-speaker chooser. Swap this body for an audio-visual ASD
    model (Light-ASD/TalkNet) to upgrade quality; the contract — one
    source-pixel center per frame, None when absent — stays the same.
    """
    small = cv2.resize(frame, (detect_w, detect_h))  # type: ignore[attr-defined]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)  # type: ignore[attr-defined]
    faces = detector.detectMultiScale(  # type: ignore[attr-defined]
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face, min_face)
    )
    if len(faces) == 0:
        return None

    # (x, y, w, h) in downscaled coords; area gates out background extras.
    max_area = max(int(w) * int(h) for _x, _y, w, h in faces)
    candidates = [
        f for f in faces if int(f[2]) * int(f[3]) >= _MIN_RELATIVE_FACE_AREA * max_area
    ]
    if not candidates:
        candidates = list(faces)

    if prev_cx is None:
        # No lock yet — take the largest face.
        x, _y, w, _h = max(candidates, key=lambda r: int(r[2]) * int(r[3]))
        return (int(x) + int(w) / 2.0) * scale

    # Lock-on: among the sizeable faces, pick the one whose center is
    # closest to where the subject was last frame. Keeps the crop stuck to
    # the speaker instead of snapping to whoever the detector found first.
    prev_small = prev_cx / scale

    def _dist(rect: object) -> float:
        cx_small = int(rect[0]) + int(rect[2]) / 2.0  # type: ignore[index]
        return abs(cx_small - prev_small)

    best = min(candidates, key=_dist)
    return (int(best[0]) + int(best[2]) / 2.0) * scale


__all__ = ["SubjectSamples", "sample_subject_track"]
