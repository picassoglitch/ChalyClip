"""Tests for the dynamic active-speaker reframe track + ffmpeg filter.

All pure-Python — the smoothing, keyframe reduction, bounds clamping and
piecewise-linear expression are exercised without OpenCV or ffmpeg. The
generated crop expression is validated by EVALUATING its maths (a tiny
`if(...)`/`lt(...)` interpreter) so a wrong interpolation is a test
failure, not something only ffmpeg would notice.
"""

from __future__ import annotations

import itertools

from nexoclip.clip.models import ReframeKeyframe, ReframeTrack
from nexoclip.clip.reframe import (
    _MAX_KEYFRAMES,
    build_reframe_track,
    reframe_track_to_ffmpeg_filter,
)

# 1920x1080 → 9:16 column is 607px wide; legal left-x is [0, 1313].
_SW, _SH = 1920, 1080
_CROP_W = (_SH * 9) // 16  # 607
_MAX_LEFT = _SW - _CROP_W  # 1313


def _eval_ff_expr(expr: str, t: float) -> float:
    """Evaluate a generated ffmpeg x-expression at time `t`.

    Rewrites `if(`/`lt(` to Python callables and evals in a sandbox — the
    expression is machine-generated from our own keyframes, never external
    input.
    """
    py = expr.replace("if(", "_iff(").replace("lt(", "_lt(")
    return float(
        eval(  # trusted, self-generated expression
            py,
            {"__builtins__": {}},
            {
                "t": t,
                "_iff": lambda c, a, b: a if c else b,
                "_lt": lambda a, b: a < b,
            },
        )
    )


def _extract_expr(filter_str: str) -> str:
    """Pull the x='...' expression body out of a dynamic crop filter."""
    assert "x='" in filter_str
    return filter_str.split("x='", 1)[1].split("':y=0", 1)[0]


def _linear_pan_centers(n: int, x0: float, x1: float) -> tuple[list[float | None], list[float]]:
    centers = [x0 + (x1 - x0) * (i / (n - 1)) for i in range(n)]
    times = [i * (4.0 / (n - 1)) for i in range(n)]
    return list(centers), times


# ---- build_reframe_track ----


def test_constant_center_is_not_dynamic() -> None:
    """A subject that never moves collapses to a non-dynamic track (the
    caller then uses the cheaper static crop)."""
    centers = [960.0] * 12
    times = [i * 0.33 for i in range(12)]
    track = build_reframe_track(
        centers_x=centers, times_s=times, source_w=_SW, source_h=_SH, duration_s=4.0
    )
    assert track is not None
    assert track.is_dynamic is False


def test_linear_pan_produces_monotonic_in_bounds_track() -> None:
    centers, times = _linear_pan_centers(9, 400.0, 1500.0)
    track = build_reframe_track(
        centers_x=centers, times_s=times, source_w=_SW, source_h=_SH, duration_s=4.0
    )
    assert track is not None
    assert track.is_dynamic is True
    assert track.w == _CROP_W and track.h == _SH
    # Anchored at t=0, every anchor in-bounds, overall left-to-right travel.
    assert track.keyframes[0].t_s == 0.0
    xs = [k.x for k in track.keyframes]
    assert all(0 <= x <= _MAX_LEFT for x in xs)
    assert xs[0] < xs[-1]
    # Strictly increasing time (no zero-length segments for the expression).
    ts = [k.t_s for k in track.keyframes]
    assert all(b > a for a, b in itertools.pairwise(ts))


def test_centers_clamped_to_frame_bounds() -> None:
    """Centers hard against both edges never push the crop off-frame."""
    centers = [0.0, 0.0, float(_SW), float(_SW), 0.0]
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    track = build_reframe_track(
        centers_x=centers, times_s=times, source_w=_SW, source_h=_SH, duration_s=4.0
    )
    assert track is not None
    for k in track.keyframes:
        assert 0 <= k.x <= _MAX_LEFT


def test_all_missing_returns_none() -> None:
    track = build_reframe_track(
        centers_x=[None] * 6,
        times_s=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        source_w=_SW,
        source_h=_SH,
        duration_s=5.0,
    )
    assert track is None


def test_gaps_are_filled_not_fatal() -> None:
    """Faceless frames in the middle are held, not dropped."""
    centers: list[float | None] = [500.0, None, None, 1400.0, None, 1400.0]
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    track = build_reframe_track(
        centers_x=centers, times_s=times, source_w=_SW, source_h=_SH, duration_s=5.0
    )
    assert track is not None
    assert track.is_dynamic is True


def test_portrait_source_returns_none() -> None:
    """A source already ≤9:16 has no horizontal room to pan."""
    track = build_reframe_track(
        centers_x=[240.0, 240.0, 240.0],
        times_s=[0.0, 1.0, 2.0],
        source_w=480,
        source_h=1080,
        duration_s=2.0,
    )
    assert track is None


def test_keyframes_capped() -> None:
    """A wiggly signal never blows past the keyframe ceiling."""
    centers = [500.0 + (900.0 if i % 2 else 0.0) for i in range(120)]
    times = [i * (6.0 / 119) for i in range(120)]
    track = build_reframe_track(
        centers_x=centers, times_s=times, source_w=_SW, source_h=_SH, duration_s=6.0
    )
    assert track is not None
    assert 2 <= len(track.keyframes) <= _MAX_KEYFRAMES


# ---- reframe_track_to_ffmpeg_filter ----


def test_dynamic_filter_shape_and_maths() -> None:
    centers, times = _linear_pan_centers(9, 400.0, 1500.0)
    track = build_reframe_track(
        centers_x=centers, times_s=times, source_w=_SW, source_h=_SH, duration_s=4.0
    )
    assert track is not None and track.is_dynamic
    vf = reframe_track_to_ffmpeg_filter(track, output_w=1080, output_h=1920)

    # Time base reset so `t` is clip-relative; crop then scale.
    assert vf.startswith("setpts=PTS-STARTPTS,")
    assert f"crop=w={_CROP_W}:h={_SH}:x='" in vf
    assert vf.endswith(",scale=1080:1920")

    expr = _extract_expr(vf)
    # At each keyframe time the expression yields that keyframe's x.
    for k in track.keyframes:
        assert _eval_ff_expr(expr, k.t_s) == float(k.x)
    # Between two keyframes it interpolates linearly (monotone ramp here).
    a, b = track.keyframes[0], track.keyframes[1]
    mid = _eval_ff_expr(expr, (a.t_s + b.t_s) / 2.0)
    assert min(a.x, b.x) <= mid <= max(a.x, b.x)
    # The evaluated position never leaves the frame.
    for t in [0.0, 0.5, 1.7, 3.9, 4.0, 10.0]:
        assert 0 <= _eval_ff_expr(expr, t) <= _MAX_LEFT


def test_non_dynamic_track_renders_static_crop() -> None:
    """A single-anchor / near-constant track produces a plain constant
    crop — no expression, no setpts — identical in shape to the static
    path."""
    track = ReframeTrack(
        source_w=_SW, source_h=_SH, w=_CROP_W, h=_SH,
        keyframes=[ReframeKeyframe(t_s=0.0, x=656)],
    )
    assert track.is_dynamic is False
    vf = reframe_track_to_ffmpeg_filter(track, output_w=1080, output_h=1920)
    assert vf == f"crop=w={_CROP_W}:h={_SH}:x=656:y=0,scale=1080:1920"
    assert "setpts" not in vf and "'" not in vf


def test_is_dynamic_threshold() -> None:
    """Two anchors a hair apart are static; far apart are dynamic."""
    near = ReframeTrack(
        source_w=_SW, source_h=_SH, w=_CROP_W, h=_SH,
        keyframes=[ReframeKeyframe(t_s=0.0, x=600), ReframeKeyframe(t_s=2.0, x=610)],
    )
    far = ReframeTrack(
        source_w=_SW, source_h=_SH, w=_CROP_W, h=_SH,
        keyframes=[ReframeKeyframe(t_s=0.0, x=200), ReframeKeyframe(t_s=2.0, x=900)],
    )
    assert near.is_dynamic is False
    assert far.is_dynamic is True
