"""Runtime NVENC fallback — when libcuda is missing or the GPU dies
mid-run, the cut step retries with libx264 instead of crashing.

Operator-reported on 06-03: Railway image has ffmpeg with NVENC
compiled in but no libcuda.so.1; the probe lied and the very first
clip cut blew up with 'Cannot load libcuda.so.1'. Two layers fix it:

  1. The probe in encoders.py now actually attempts an encode.
  2. This wrapper retries with libx264 if NVENC fails at runtime
     for any reason (sticky after first failure so subsequent clips
     in the same run skip NVENC outright).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chalybclip.clip import encoders
from chalybclip.clip import service as clip_service
from chalybclip.config import ClipConfig
from chalybclip.errors import ClipError


@pytest.fixture(autouse=True)
def _reset_nvenc_state() -> None:
    """Wipe both the probe cache AND the runtime-disable flag between
    tests so one test's failure doesn't leak into the next."""
    encoders.reset_nvenc_cache()
    yield
    encoders.reset_nvenc_cache()


def test_nvenc_failure_retries_with_libx264(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First _run_ffmpeg call fails simulating CUDA missing; the
    wrapper detects NVENC was used + retries with libx264. The second
    call records libx264 in argv."""
    monkeypatch.setattr(encoders, "has_nvenc", lambda: True)

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], *, what: str) -> None:
        calls.append(cmd)
        # Fail the first call (NVENC); succeed on the libx264 retry.
        if len(calls) == 1:
            raise ClipError(
                "ffmpeg accurate cut failed: [h264_nvenc] Cannot load "
                "libcuda.so.1"
            )

    monkeypatch.setattr(clip_service, "_run_ffmpeg", fake_run)

    def build(encoder_args: list[str]) -> list[str]:
        return ["ffmpeg", *encoder_args, "out.mp4"]

    clip_service._run_encode_with_fallback(
        cfg=ClipConfig(),
        build_cmd=build,
        what="test cut",
    )

    assert len(calls) == 2
    # First attempt used h264_nvenc
    assert "h264_nvenc" in calls[0]
    # Retry used libx264 with the config's preset + crf
    assert "libx264" in calls[1]


def test_runtime_failure_sticks_for_subsequent_encodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After one NVENC runtime failure, has_nvenc() returns False for
    the rest of the process. Subsequent clips skip NVENC outright
    instead of re-failing the same way."""
    monkeypatch.setattr(encoders, "_probe_nvenc", lambda: True)
    encoders.reset_nvenc_cache()
    assert encoders.has_nvenc() is True

    encoders.mark_nvenc_runtime_failure(reason="libcuda missing")
    assert encoders.has_nvenc() is False

    # pick_video_encoder_args reflects the sticky disable, even though
    # cfg.prefer_nvenc is still True.
    cfg = ClipConfig()
    args = encoders.pick_video_encoder_args(cfg)
    assert "libx264" in args
    assert "h264_nvenc" not in args


def test_fallback_does_not_run_when_nvenc_was_not_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When NVENC isn't selected (e.g. operator forced libx264), an
    encode failure surfaces as-is without a spurious retry."""
    monkeypatch.setattr(encoders, "has_nvenc", lambda: False)

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], *, what: str) -> None:
        calls.append(cmd)
        raise ClipError("ffmpeg failed for unrelated reason")

    monkeypatch.setattr(clip_service, "_run_ffmpeg", fake_run)

    def build(encoder_args: list[str]) -> list[str]:
        return ["ffmpeg", *encoder_args, "out.mp4"]

    with pytest.raises(ClipError, match="unrelated reason"):
        clip_service._run_encode_with_fallback(
            cfg=ClipConfig(),
            build_cmd=build,
            what="test cut",
        )
    # Exactly one call — no retry because libx264 was already chosen.
    assert len(calls) == 1


def test_libx264_fallback_failure_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If libx264 ALSO fails on the retry, we surface that error
    instead of masking it. The pipeline needs to know cuts are broken
    end-to-end, not just that NVENC didn't work."""
    monkeypatch.setattr(encoders, "has_nvenc", lambda: True)

    def fake_run(cmd: list[str], *, what: str) -> None:
        if "h264_nvenc" in cmd:
            raise ClipError("NVENC failed")
        raise ClipError("libx264 also failed: disk full")

    monkeypatch.setattr(clip_service, "_run_ffmpeg", fake_run)

    def build(encoder_args: list[str]) -> list[str]:
        return ["ffmpeg", *encoder_args, "out.mp4"]

    with pytest.raises(ClipError, match="libx264 also failed"):
        clip_service._run_encode_with_fallback(
            cfg=ClipConfig(),
            build_cmd=build,
            what="test cut",
        )
