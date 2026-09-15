"""Tests for `cut_window` (start/end/duration clamping)."""

from __future__ import annotations

import pytest

from chalybclip.clip import cut_window
from chalybclip.errors import ClipError


def test_normal_window() -> None:
    start, end, duration = cut_window(
        timestamp=120.0, pre_roll_s=30.0, post_roll_s=15.0, stream_duration_s=600.0
    )
    assert start == pytest.approx(90.0)
    assert end == pytest.approx(135.0)
    assert duration == pytest.approx(45.0)


def test_clamps_start_to_zero_for_early_anchor() -> None:
    start, end, duration = cut_window(
        timestamp=10.0, pre_roll_s=30.0, post_roll_s=15.0, stream_duration_s=600.0
    )
    assert start == 0.0
    assert end == pytest.approx(25.0)
    assert duration == pytest.approx(25.0)


def test_clamps_end_to_stream_length() -> None:
    start, end, duration = cut_window(
        timestamp=595.0, pre_roll_s=30.0, post_roll_s=15.0, stream_duration_s=600.0
    )
    assert start == pytest.approx(565.0)
    assert end == pytest.approx(600.0)
    assert duration == pytest.approx(35.0)


def test_anchor_after_stream_end_yields_zero_duration() -> None:
    start, end, duration = cut_window(
        timestamp=700.0, pre_roll_s=30.0, post_roll_s=15.0, stream_duration_s=600.0
    )
    assert duration == 0.0
    assert start == end


def test_zero_stream_duration_raises() -> None:
    with pytest.raises(ClipError):
        cut_window(timestamp=10.0, pre_roll_s=30.0, post_roll_s=15.0, stream_duration_s=0.0)


# ---- retroactive trigger window ('clipeaste eso' → 60s back) ----


def test_retroactive_window_extends_backward() -> None:
    """When trigger_kind=retroactive, the clip ends AT the timestamp and
    starts retroactive_lookback_s earlier — symmetric pre/post roll is
    ignored. This is the natural shape for 'clipeaste eso': the moment
    ended, then the streamer flagged it after the fact."""
    start, end, duration = cut_window(
        timestamp=200.0,
        pre_roll_s=30.0,    # ignored on retroactive
        post_roll_s=15.0,   # ignored on retroactive
        stream_duration_s=600.0,
        trigger_kind="retroactive",
        retroactive_lookback_s=60.0,
    )
    assert start == pytest.approx(140.0)
    assert end == pytest.approx(200.0)
    assert duration == pytest.approx(60.0)


def test_retroactive_window_clamps_start_to_zero() -> None:
    """If the lookback would push us before t=0, clamp."""
    start, end, duration = cut_window(
        timestamp=20.0,
        pre_roll_s=30.0,
        post_roll_s=15.0,
        stream_duration_s=600.0,
        trigger_kind="retroactive",
        retroactive_lookback_s=60.0,
    )
    assert start == 0.0
    assert end == pytest.approx(20.0)
    assert duration == pytest.approx(20.0)


def test_retroactive_without_lookback_falls_back_to_forward() -> None:
    """Defensive: a retroactive flag without a lookback value uses the
    forward (pre/post roll) shape rather than producing a zero-length clip."""
    start, end, duration = cut_window(
        timestamp=120.0,
        pre_roll_s=30.0,
        post_roll_s=15.0,
        stream_duration_s=600.0,
        trigger_kind="retroactive",
        retroactive_lookback_s=None,
    )
    assert start == pytest.approx(90.0)
    assert end == pytest.approx(135.0)
    assert duration == pytest.approx(45.0)
