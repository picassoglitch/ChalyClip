"""Dense per-frame subject tracking for the reframe pipeline.

The static smart-crop / framing passes sample 6-10 frames across a clip
and keep only the LARGEST face per frame — enough to pick one fixed crop,
but blind to WHERE the subject is at each moment. Active-speaker reframe
needs temporal resolution: a chosen subject center sampled densely enough
(~3 fps) that the renderer can pan a crop window to follow it.

This module opens the source window once, decodes it sequentially (cheap —
`grab()` skips the frames we don't sample), runs the shared Haar detector
on each sampled frame, and returns one chosen-subject center-x per sample.

Active-speaker selection
------------------------
"The speaker" is chosen per sampled frame. On a SINGLE-face talking-head
clip — the common streamer case — there is nothing to disambiguate: the one
face is followed via a size + temporal-persistence lock (unchanged, so
those exports stay byte-identical to before).

When MULTIPLE faces share the frame (a co-host / reaction split), a
heuristic active-speaker detector picks who is talking: per face we measure
lip motion — the frame-to-frame pixel change in the lower third of the face
box — and, when the source has an audio track, how well that lip motion
syncs with the audio-energy envelope. The face whose mouth moves (in time
with the sound) wins. The pure scorer lives in
`nexoclip.vision.active_speaker`; this module only feeds it pixels + audio.

This is a HEURISTIC ASD (frame-diff lip motion + audio-energy sync), NOT a
learned model. The real upgrade (Light-ASD / TalkNet) is a localized swap:
feed `pick_active_speaker` a per-candidate speaking probability and the rest
of the pipeline — the dense decode loop, the SubjectSamples contract, the
downstream reframe track — is unchanged. Kept CPU-only + dependency-free
(OpenCV Haar + a stdlib ffmpeg/audio pass) on purpose so it runs on the
operator's box with zero new installs.
"""

from __future__ import annotations

import struct
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nexoclip.logging import get_logger
from nexoclip.vision.active_speaker import mouth_roi_box, pick_active_speaker

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

# --- Active-speaker (multi-face) tuning ---
# How many consecutive samples of per-face lip motion the scorer sees. ~8
# at 3 fps is ~2.5s — long enough to catch a speaking rhythm and audio sync,
# short enough to react when the mic changes hands.
_MOTION_WINDOW = 8
# Minimum mean mouth motion (8-bit pixel-diff units) for a face to count as
# "visibly speaking". Below this on EVERY face, the ASD abstains and the
# size/persistence lock decides — a still multi-face frame shouldn't jitter.
_MIN_MOUTH_MOTION = 1.5
# Associate this frame's faces to live tracks by nearest center-x, within a
# gate of this fraction of the (downscaled) frame width. Two real speakers
# sit well apart; this only has to absorb per-face detector jitter.
_ASSOC_GATE_FRAC = 0.2
# Drop a track that has gone unmatched for this many consecutive samples.
_TRACK_MAX_MISSES = 4

# --- Optional audio-energy envelope ---
# Low mono sample rate + coarse bins is plenty for an energy envelope (we
# correlate its shape with lip motion, we don't resynthesize it).
_AUDIO_SR = 8000
_AUDIO_BINS_PER_S = 25
# Cap the extraction subprocess so a wedged ffmpeg can't stall a render.
_AUDIO_TIMEOUT_S = 30.0


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

        tracker = _SubjectTracker(
            detector=detector,
            cv2=cv2,
            scale=scale,
            detect_w=detect_w,
            detect_h=detect_h,
            min_face=min_face,
            video_path=Path(video_path),
            start_s=start_s,
            duration_s=end_s - start_s,
        )

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        times_s: list[float] = []
        centers_x: list[float | None] = []
        offset = 0
        while start_frame + offset <= end_frame and len(times_s) < _MAX_SAMPLES:
            if offset % stride == 0:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                clip_t = (start_frame + offset) / fps - start_s
                cx = tracker.update(frame, clip_t=clip_t)
                times_s.append(clip_t)
                centers_x.append(cx)
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


@dataclass
class _Track:
    """A live per-face track through the multi-face section of a clip.

    `cx_small` is the last center-x in DOWNSCALED detection coords; `motion`
    is the per-sample mouth-motion history (aligned frame-for-frame with the
    audio window); `misses` counts consecutive unmatched samples so a face
    that has left frame can be expired.
    """

    cx_small: float
    motion: deque[float]
    misses: int = 0


class _SubjectTracker:
    """Stateful per-sample subject chooser for one clip window.

    Holds the cross-frame state the active-speaker heuristic needs — the
    previous sampled frame (for lip-motion frame-differencing), the live
    per-face tracks, the persistence lock, and a lazily-extracted
    audio-energy envelope — behind a single `update(frame) -> center-x`
    call so `sample_subject_track`'s decode loop stays a thin reader.

    Single-face clips (the common streamer case) never touch the ASD path:
    `update` short-circuits to the same size + persistence pick the static
    tracker always used, so those exports are byte-identical.
    """

    def __init__(
        self,
        *,
        detector: Any,
        cv2: Any,
        scale: float,
        detect_w: int,
        detect_h: int,
        min_face: int,
        video_path: Path,
        start_s: float,
        duration_s: float,
    ) -> None:
        self._detector = detector
        self._cv2 = cv2
        self._scale = scale
        self._detect_w = detect_w
        self._detect_h = detect_h
        self._min_face = min_face
        self._video_path = video_path
        self._start_s = start_s
        self._duration_s = duration_s
        self._prev_gray: Any | None = None
        self._prev_cx: float | None = None
        self._tracks: list[_Track] = []
        self._audio_env: list[float] | None = None
        self._audio_tried = False
        self._audio_win: deque[float] = deque(maxlen=_MOTION_WINDOW)

    def update(self, frame: Any, *, clip_t: float) -> float | None:
        """Chosen subject center-x in SOURCE pixels for this sample, or None.

        Detects faces once, then routes: 0-1 faces take the unchanged
        size/persistence lock; 2+ faces go through the lip-motion ASD, which
        itself falls back to that same lock when no mouth is visibly moving.
        Any per-frame failure degrades to the lock rather than raising —
        reframe must never break a cut.
        """
        cv2 = self._cv2
        try:
            small = cv2.resize(frame, (self._detect_w, self._detect_h))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            candidates = _gate_by_area(
                _detect_faces(gray, self._detector, self._min_face)
            )
            if len(candidates) >= 2:
                cx = self._choose_active_speaker(candidates, gray, clip_t)
            else:
                cx = _select_by_persistence(candidates, self._prev_cx, self._scale)
        except Exception as e:  # a cv2 hiccup on one frame must not be fatal
            _log.warning("subject_track.frame_failed", reason=str(e))
            self._prev_gray = None
            return None

        self._prev_gray = gray
        if cx is not None:
            self._prev_cx = cx
        return cx

    def _choose_active_speaker(
        self, candidates: list[Any], gray: Any, clip_t: float
    ) -> float | None:
        """Pick the talking face among 2+ candidates via lip motion + audio.

        Builds per-track mouth-motion windows (associated across frames by
        nearest center-x), scores them with the pure `pick_active_speaker`,
        and maps the winner back to source pixels. Abstention (no visible
        lip motion) falls back to the size/persistence lock.
        """
        motions = [self._mouth_motion(gray, rect) for rect in candidates]
        present = self._advance_tracks(candidates, motions)
        audio = self._push_audio(clip_t)
        windows = [list(track.motion) for track, _cx in present]
        idx = pick_active_speaker(
            windows, audio_energy=audio, min_motion=_MIN_MOUTH_MOTION
        )
        if idx is None:
            return _select_by_persistence(candidates, self._prev_cx, self._scale)
        return present[idx][1] * self._scale

    def _mouth_motion(self, gray: Any, rect: Any) -> float:
        """Mean abs frame-diff over this face's mouth ROI vs the last sample.

        0.0 when there is no previous frame or the ROI is degenerate — the
        scorer reads that as "no lip evidence".
        """
        prev = self._prev_gray
        if prev is None:
            return 0.0
        x0, y0, x1, y1 = mouth_roi_box(
            int(rect[0]),
            int(rect[1]),
            int(rect[2]),
            int(rect[3]),
            frame_w=self._detect_w,
            frame_h=self._detect_h,
        )
        if x1 <= x0 or y1 <= y0:
            return 0.0
        cur_roi = gray[y0:y1, x0:x1]
        prev_roi = prev[y0:y1, x0:x1]
        if cur_roi.shape != prev_roi.shape or cur_roi.size == 0:
            return 0.0
        return float(self._cv2.absdiff(cur_roi, prev_roi).mean())

    def _advance_tracks(
        self, candidates: list[Any], motions: list[float]
    ) -> list[tuple[_Track, float]]:
        """Match this frame's faces to live tracks by nearest center-x.

        Every surviving track advances by exactly one sample (matched faces
        push their motion, unmatched tracks push 0.0) so all track windows —
        and the audio window — stay the same length and time-aligned.
        Returns the (track, center-x) pairs PRESENT this frame, in candidate
        order, for scoring.
        """
        gate = self._detect_w * _ASSOC_GATE_FRAC
        old = self._tracks
        used: set[int] = set()
        present: list[tuple[_Track, float]] = []
        new_tracks: list[_Track] = []
        for rect, motion in zip(candidates, motions, strict=True):
            cx_small = int(rect[0]) + int(rect[2]) / 2.0
            best_j, best_d = -1, gate
            for j, track in enumerate(old):
                if j in used:
                    continue
                dist = abs(track.cx_small - cx_small)
                if dist <= best_d:
                    best_d, best_j = dist, j
            if best_j >= 0:
                track = old[best_j]
                track.cx_small = cx_small
                track.motion.append(motion)
                track.misses = 0
                used.add(best_j)
                present.append((track, cx_small))
            else:
                track = _Track(
                    cx_small=cx_small,
                    motion=deque([motion], maxlen=_MOTION_WINDOW),
                )
                new_tracks.append(track)
                present.append((track, cx_small))

        survivors = list(new_tracks)
        for j, track in enumerate(old):
            if j in used:
                survivors.append(track)
                continue
            track.motion.append(0.0)
            track.misses += 1
            if track.misses <= _TRACK_MAX_MISSES:
                survivors.append(track)
        self._tracks = survivors
        return present

    def _push_audio(self, clip_t: float) -> list[float] | None:
        """Advance the audio window by one sample; None when no audio track.

        The envelope is extracted once, lazily, on the first multi-face
        sample (single-face clips never pay for it). Failure is sticky and
        silent — the ASD then runs lip-motion-only.
        """
        if not self._audio_tried:
            self._audio_tried = True
            self._audio_env = _audio_energy_envelope(
                self._video_path, self._start_s, self._duration_s
            )
        if self._audio_env is None:
            return None
        self._audio_win.append(
            _sample_envelope(self._audio_env, clip_t, self._duration_s)
        )
        return list(self._audio_win)


def _detect_faces(gray: Any, detector: Any, min_face: int) -> Any:
    """Run the Haar detector on a downscaled grayscale frame."""
    return detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face, min_face)
    )


def _gate_by_area(faces: Any) -> list[Any]:
    """Drop background extras — faces far smaller than the frame's largest.

    Returns the sizeable faces (or all of them if the gate would empty the
    set), matching the original single-frame chooser exactly.
    """
    if len(faces) == 0:
        return []
    max_area = max(int(w) * int(h) for _x, _y, w, h in faces)
    gated = [
        f for f in faces if int(f[2]) * int(f[3]) >= _MIN_RELATIVE_FACE_AREA * max_area
    ]
    return gated if gated else list(faces)


def _select_by_persistence(
    candidates: list[Any], prev_cx: float | None, scale: float
) -> float | None:
    """Size + temporal-persistence subject center in SOURCE pixels, or None.

    The original heuristic, unchanged: no lock yet -> largest face; locked
    -> the sizeable face whose center is closest to last sample's, so the
    crop stays stuck to one subject instead of snapping between faces. Used
    for the single-face fast path and as the multi-face ASD fallback.
    """
    if not candidates:
        return None
    if prev_cx is None:
        x, _y, w, _h = max(candidates, key=lambda r: int(r[2]) * int(r[3]))
        return (int(x) + int(w) / 2.0) * scale
    prev_small = prev_cx / scale
    best = min(
        candidates, key=lambda r: abs((int(r[0]) + int(r[2]) / 2.0) - prev_small)
    )
    return (int(best[0]) + int(best[2]) / 2.0) * scale


def _audio_energy_envelope(
    video_path: Path, start_s: float, duration_s: float
) -> list[float] | None:
    """Best-effort per-bin RMS audio envelope over [start_s, start_s+dur].

    One ffmpeg subprocess pulls low-rate mono PCM for the clip window, which
    is reduced to `_AUDIO_BINS_PER_S` RMS bins per second. Pure stdlib
    (struct), no numpy. Returns None on ANY failure (no ffmpeg, no audio
    track, bad window, timeout) so the caller degrades to lip-motion-only —
    audio is a tie-breaker, never a requirement.
    """
    if duration_s <= 0.0:
        return None
    cmd = [
        "ffmpeg",
        "-ss", f"{max(0.0, start_s):.3f}",
        "-i", str(video_path),
        "-t", f"{duration_s:.3f}",
        "-ac", "1",
        "-ar", str(_AUDIO_SR),
        "-f", "s16le",
        "-loglevel", "error",
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, check=False, timeout=_AUDIO_TIMEOUT_S
        )
    except Exception as e:  # ffmpeg missing, timeout, OS error — all non-fatal
        _log.warning("subject_track.audio_failed", reason=str(e))
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    return _rms_bins(proc.stdout, sr=_AUDIO_SR, bins_per_s=_AUDIO_BINS_PER_S)


def _rms_bins(pcm: bytes, *, sr: int, bins_per_s: int) -> list[float] | None:
    """Reduce int16 mono PCM to normalized RMS bins in [0, 1]. None if empty."""
    n_samples = len(pcm) // 2  # int16 = 2 bytes
    if n_samples == 0:
        return None
    bin_size = max(1, sr // max(1, bins_per_s))
    n_bins = max(1, n_samples // bin_size)
    out: list[float] = []
    for i in range(n_bins):
        start = i * bin_size * 2
        end = min(len(pcm), start + bin_size * 2)
        chunk = pcm[start:end]
        m = len(chunk) // 2
        if m == 0:
            out.append(0.0)
            continue
        ints = struct.unpack(f"<{m}h", chunk[: m * 2])
        rms = float((sum(v * v for v in ints) / m) ** 0.5)
        out.append(rms / 32768.0)
    return out


def _sample_envelope(env: list[float], clip_t: float, duration_s: float) -> float:
    """Value of a clip-window envelope at CLIP-relative time `clip_t`."""
    if not env:
        return 0.0
    if duration_s <= 0.0:
        return env[0]
    frac = min(1.0, max(0.0, clip_t / duration_s))
    idx = min(len(env) - 1, int(frac * len(env)))
    return env[idx]

__all__ = ["SubjectSamples", "sample_subject_track"]
