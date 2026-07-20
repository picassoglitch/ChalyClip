"""Dynamic active-speaker reframe — raw subject centers → moving crop.

The static sibling (`smart_crop.py`) averages face centers into ONE crop
held for the whole clip. This module turns the DENSE per-frame centers
from `vision.face_track` into a smoothed, in-bounds path and renders it as
a single ffmpeg `crop` filter whose `x` position is a piecewise-linear
function of time — so the 9:16 column pans to keep the speaker framed.

Three stages, all pure Python (no OpenCV / ffmpeg here) so the maths is
unit-testable in isolation:

  1. fill gaps (hold the last known center through faceless frames)
  2. smooth (forward-backward EMA — gentle pan, no lag, no jitter)
  3. simplify (clamp in-bounds → Ramer-Douglas-Peucker → cap keyframes)

The renderer applies `setpts=PTS-STARTPTS` before the crop so `t` in the
expression is always clip-relative (0.0 = clip start), independent of how
the upstream cut stamped its timestamps.
"""

from __future__ import annotations

from pathlib import Path

from .models import ReframeKeyframe, ReframeTrack

# Smoothing / simplification tuning. These shape "how OpusClip-like" the
# pan feels: lower alpha = smoother/slower, higher epsilon = fewer
# keyframes (coarser follow). Defaults chosen for a calm, TV-like pan.
_EMA_ALPHA = 0.25
_RDP_EPSILON_PX = 8.0
_MAX_KEYFRAMES = 24
# Below this fraction of samples finding a face, a track is not trustworthy
# — a static crop is the safer export than a path built on guesses.
_MIN_COVERAGE = 0.5

_ASPECT_W = 9
_ASPECT_H = 16


def compute_reframe_track(
    video_path: Path,
    *,
    start_s: float,
    end_s: float,
    target_fps: float = 3.0,
) -> ReframeTrack | None:
    """End-to-end: sample the source window, build a smoothed track.

    Returns None whenever a dynamic reframe isn't warranted or possible
    (no OpenCV, too few faces, source already ≤9:16) — the caller then
    uses the existing static crop. Never raises.
    """
    from nexoclip.vision.face_track import sample_subject_track

    samples = sample_subject_track(
        video_path, start_s=start_s, end_s=end_s, target_fps=target_fps
    )
    if samples is None or samples.coverage < _MIN_COVERAGE:
        return None
    return build_reframe_track(
        centers_x=samples.centers_x,
        times_s=samples.times_s,
        source_w=samples.source_w,
        source_h=samples.source_h,
        duration_s=end_s - start_s,
    )


def build_reframe_track(
    *,
    centers_x: list[float | None],
    times_s: list[float],
    source_w: int,
    source_h: int,
    duration_s: float,
) -> ReframeTrack | None:
    """Turn parallel (time, center-x|None) samples into a ReframeTrack.

    Pure function — no I/O — so the smoothing + keyframe reduction is
    exercised directly in tests. Returns None when no subject was ever
    detected or the source is already ≤9:16 (nothing to pan).
    """
    if source_w <= 0 or source_h <= 0 or not times_s:
        return None

    crop_w = min(source_w, max(1, (source_h * _ASPECT_W) // _ASPECT_H))
    crop_h = source_h
    if crop_w >= source_w:
        return None  # already portrait/narrower — no horizontal room to pan

    filled = _fill_centers(centers_x)
    if filled is None:
        return None  # never saw a face

    smoothed = _forward_backward_ema(filled, _EMA_ALPHA)

    # Clamp each smoothed center to a legal crop-left-x so the column never
    # runs off the source edge — the expression is then clamp-free at runtime.
    half = crop_w / 2.0
    max_left = source_w - crop_w
    pts: list[tuple[float, int]] = []
    for t, c in zip(times_s, smoothed, strict=True):
        c = max(half, min(source_w - half, c))
        x_left = max(0, min(max_left, round(c - half)))
        t = max(0.0, min(duration_s, float(t)))
        pts.append((t, x_left))

    pts = _dedupe_by_time(pts)
    if pts and pts[0][0] > 0.0:
        pts.insert(0, (0.0, pts[0][1]))  # anchor the start at t=0

    if len(pts) < 2:
        x0 = pts[0][1] if pts else 0
        return ReframeTrack(
            source_w=source_w, source_h=source_h, w=crop_w, h=crop_h,
            keyframes=[ReframeKeyframe(t_s=0.0, x=x0)],
        )

    simplified = _rdp(pts, _RDP_EPSILON_PX)
    if len(simplified) > _MAX_KEYFRAMES:
        simplified = _decimate(simplified, _MAX_KEYFRAMES)

    keyframes = _strictly_increasing(
        [ReframeKeyframe(t_s=round(t, 3), x=x) for t, x in simplified]
    )
    return ReframeTrack(
        source_w=source_w, source_h=source_h, w=crop_w, h=crop_h,
        keyframes=keyframes,
    )


def reframe_track_to_ffmpeg_filter(
    track: ReframeTrack, *, output_w: int, output_h: int
) -> str:
    """Render the crop+scale `-vf` chain for a reframe track.

    A dynamic track becomes a time-driven crop (`x` is a piecewise-linear
    expression in `t`, single-quoted so its commas survive filtergraph
    parsing). A degenerate/near-static track collapses to a plain constant
    crop — byte-identical to the static path.
    """
    cw, ch = track.w, track.h
    kfs = track.keyframes
    if not track.is_dynamic or len(kfs) < 2:
        x0 = kfs[0].x if kfs else 0
        return f"crop=w={cw}:h={ch}:x={x0}:y=0,scale={output_w}:{output_h}"

    expr = _piecewise_x_expr(kfs)
    return (
        f"setpts=PTS-STARTPTS,"
        f"crop=w={cw}:h={ch}:x='{expr}':y=0,"
        f"scale={output_w}:{output_h}"
    )


# ---- Internals (pure maths) ----


def _fill_centers(centers: list[float | None]) -> list[float] | None:
    """Forward-fill then back-fill the None gaps. None if all missing."""
    if all(c is None for c in centers):
        return None
    first_known = next(c for c in centers if c is not None)
    out: list[float] = []
    last = first_known
    for c in centers:
        if c is not None:
            last = c
        out.append(last)
    return out


def _forward_backward_ema(vals: list[float], alpha: float) -> list[float]:
    """Two-pass EMA (forward + backward, averaged) — smooths without the
    lag a single forward pass introduces."""
    n = len(vals)
    if n == 0:
        return []
    fwd = [0.0] * n
    s = vals[0]
    for i in range(n):
        s = alpha * vals[i] + (1 - alpha) * s
        fwd[i] = s
    bwd = [0.0] * n
    s = vals[-1]
    for i in range(n - 1, -1, -1):
        s = alpha * vals[i] + (1 - alpha) * s
        bwd[i] = s
    return [(f + b) / 2.0 for f, b in zip(fwd, bwd, strict=True)]


def _dedupe_by_time(pts: list[tuple[float, int]]) -> list[tuple[float, int]]:
    """Drop samples that don't advance in time (keep the later x)."""
    out: list[tuple[float, int]] = []
    for t, x in pts:
        if out and t <= out[-1][0]:
            out[-1] = (out[-1][0], x)
        else:
            out.append((t, x))
    return out


def _rdp(pts: list[tuple[float, int]], eps: float) -> list[tuple[float, int]]:
    """Ramer-Douglas-Peucker with VERTICAL (x-vs-chord) distance — the
    right metric for a time series: keep a sample only when dropping it
    would move the interpolated crop by more than `eps` pixels."""
    if len(pts) < 3:
        return pts
    t0, x0 = pts[0]
    t1, x1 = pts[-1]
    span = t1 - t0
    dmax, idx = 0.0, 0
    for i in range(1, len(pts) - 1):
        t, x = pts[i]
        x_chord = x0 + (x1 - x0) * ((t - t0) / span) if span else x0
        d = abs(x - x_chord)
        if d > dmax:
            dmax, idx = d, i
    if dmax > eps:
        left = _rdp(pts[: idx + 1], eps)
        right = _rdp(pts[idx:], eps)
        return left[:-1] + right
    return [pts[0], pts[-1]]


def _decimate(pts: list[tuple[float, int]], n: int) -> list[tuple[float, int]]:
    """Uniformly thin to `n` points, always keeping both endpoints."""
    if len(pts) <= n:
        return pts
    step = (len(pts) - 1) / (n - 1)
    picked = [pts[round(i * step)] for i in range(n)]
    picked[-1] = pts[-1]
    return picked


def _strictly_increasing(kfs: list[ReframeKeyframe]) -> list[ReframeKeyframe]:
    """Drop any keyframe whose time didn't advance after rounding, so the
    interpolation never divides by a zero time delta."""
    out: list[ReframeKeyframe] = []
    for k in kfs:
        if out and k.t_s <= out[-1].t_s:
            continue
        out.append(k)
    return out


def _piecewise_x_expr(kfs: list[ReframeKeyframe]) -> str:
    """Nested-if piecewise-linear ffmpeg expression for x(t).

    Before the first keyframe holds x0; each segment linearly interpolates;
    after the last keyframe holds xn. Depth == keyframe count (bounded by
    `_MAX_KEYFRAMES`), well within ffmpeg's expression limits.
    """
    expr = str(kfs[-1].x)
    for i in range(len(kfs) - 2, -1, -1):
        a, b = kfs[i], kfs[i + 1]
        dt = b.t_s - a.t_s
        lin = f"({a.x}+({b.x}-{a.x})*(t-{a.t_s:.3f})/{dt:.3f})"
        expr = f"if(lt(t,{b.t_s:.3f}),{lin},{expr})"
    return f"if(lt(t,{kfs[0].t_s:.3f}),{kfs[0].x},{expr})"


__all__ = [
    "build_reframe_track",
    "compute_reframe_track",
    "reframe_track_to_ffmpeg_filter",
]
