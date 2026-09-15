"""libx264 `-threads` cap (#3a) — keeps a single encode from
oversubscribing the host thread ceiling under concurrent load."""

from __future__ import annotations

import pytest

from chalybclip.clip import encoders
from chalybclip.clip.encoders import ffmpeg_thread_args, pick_video_encoder_args
from chalybclip.config import ClipConfig


def test_thread_args_default_is_two() -> None:
    assert ffmpeg_thread_args(ClipConfig()) == ["-threads", "2"]


def test_thread_args_zero_disables() -> None:
    # 0 => let ffmpeg keep threads=auto (no -threads emitted).
    assert ffmpeg_thread_args(ClipConfig(ffmpeg_threads=0)) == []


def test_libx264_args_carry_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(encoders, "has_nvenc", lambda: False)
    args = pick_video_encoder_args(ClipConfig(ffmpeg_threads=4))
    assert "libx264" in args
    assert args[-2:] == ["-threads", "4"]


def test_operator_override_encoder_carries_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(encoders, "has_nvenc", lambda: False)
    args = pick_video_encoder_args(ClipConfig(encoder="libx265", ffmpeg_threads=2))
    assert "libx265" in args
    assert args[-2:] == ["-threads", "2"]


def test_nvenc_path_has_no_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    # NVENC is GPU-bound; the CPU thread cap must not leak into its args.
    monkeypatch.setattr(encoders, "has_nvenc", lambda: True)
    args = pick_video_encoder_args(ClipConfig(prefer_nvenc=True, ffmpeg_threads=2))
    assert "h264_nvenc" in args
    assert "-threads" not in args
