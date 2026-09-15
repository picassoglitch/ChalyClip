"""Audio waveform extractor + cache (slice F.7 follow-up A)."""

from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from chalybclip.clip.waveform import (
    DEFAULT_N_BUCKETS,
    _bucket_peaks,
    cache_path_for_clip,
    compute_waveform,
    load_or_compute,
)


def _pack_int16(values: list[int]) -> bytes:
    """Pack a list of int16s as little-endian PCM bytes."""
    return struct.pack(f"<{len(values)}h", *values)


# ---- _bucket_peaks ----


def test_bucket_peaks_takes_max_abs_per_bucket() -> None:
    """Sample stream: [10, -200, 0, 0, 50, 100, -10, -300]
    → 4 buckets of size 2 → peaks: 200, 0, 100, 300 → normalized."""
    pcm = _pack_int16([10, -200, 0, 0, 50, 100, -10, -300])
    peaks = _bucket_peaks(pcm, n_samples=8, n_buckets=4)
    assert peaks == [
        200 / 32768.0,
        0 / 32768.0,
        100 / 32768.0,
        300 / 32768.0,
    ]


def test_bucket_peaks_handles_silent_audio() -> None:
    """All-zero PCM → all peaks = 0.0."""
    pcm = _pack_int16([0] * 64)
    peaks = _bucket_peaks(pcm, n_samples=64, n_buckets=8)
    assert peaks == [0.0] * 8


def test_bucket_peaks_normalized_to_0_1() -> None:
    """Max int16 = 32767 → peak = 32767/32768 ≈ 1.0 (still ≤ 1)."""
    pcm = _pack_int16([32767] * 4)
    peaks = _bucket_peaks(pcm, n_samples=4, n_buckets=2)
    assert all(0.0 <= p <= 1.0 for p in peaks)
    assert peaks[0] > 0.99


def test_bucket_peaks_short_pcm_pads_with_zeros() -> None:
    """If the audio is shorter than expected, trailing buckets get 0."""
    pcm = _pack_int16([1000, 2000])
    peaks = _bucket_peaks(pcm, n_samples=2, n_buckets=4)
    assert len(peaks) == 4
    assert peaks[0] > 0  # first bucket has data
    assert peaks[-1] == 0.0  # tail bucket is empty


# ---- compute_waveform (with mocked ffmpeg) ----


def test_compute_waveform_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compute_waveform(tmp_path / "nope.mp4")


def test_compute_waveform_raises_on_ffmpeg_failure(tmp_path: Path) -> None:
    fake = tmp_path / "x.mp4"
    fake.write_bytes(b"\x00\x00")

    def fake_run(cmd, capture_output, check):
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout=b"",
            stderr=b"Invalid data found",
        )

    with (
        patch("chalybclip.clip.waveform.subprocess.run", fake_run),
        pytest.raises(RuntimeError, match="ffmpeg failed"),
    ):
        compute_waveform(fake)


def test_compute_waveform_raises_on_empty_audio(tmp_path: Path) -> None:
    """ffmpeg succeeded but produced no PCM bytes → empty audio track."""
    fake = tmp_path / "x.mp4"
    fake.write_bytes(b"\x00")

    def fake_run(cmd, capture_output, check):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=b"", stderr=b""
        )

    with (
        patch("chalybclip.clip.waveform.subprocess.run", fake_run),
        pytest.raises(RuntimeError, match="empty audio track"),
    ):
        compute_waveform(fake)


def test_compute_waveform_returns_n_buckets_floats(tmp_path: Path) -> None:
    """Happy path: ffmpeg returns PCM, we get n_buckets normalized floats."""
    fake = tmp_path / "x.mp4"
    fake.write_bytes(b"\x00")
    pcm = _pack_int16([5000, -10000, 20000, -30000] * 100)

    def fake_run(cmd, capture_output, check):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=pcm, stderr=b""
        )

    with patch("chalybclip.clip.waveform.subprocess.run", fake_run):
        peaks = compute_waveform(fake, n_buckets=10)
    assert len(peaks) == 10
    assert all(0.0 <= p <= 1.0 for p in peaks)


# ---- load_or_compute (cache) ----


def test_load_or_compute_uses_cache_when_present(tmp_path: Path) -> None:
    """A previously-cached waveform.json is returned without invoking
    ffmpeg — important: this is what makes subsequent page loads
    (and the click-to-seek scrubber) instant."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    cache = cache_path_for_clip(clip)
    cache.write_text(json.dumps([0.1, 0.2, 0.3]), encoding="utf-8")

    def boom(*a, **k):
        raise AssertionError("ffmpeg should not have been called")

    with patch("chalybclip.clip.waveform.subprocess.run", boom):
        peaks = load_or_compute(clip)
    assert peaks == [0.1, 0.2, 0.3]


def test_load_or_compute_recomputes_when_cache_corrupt(tmp_path: Path) -> None:
    """If the cache JSON is malformed, fall through to ffmpeg + rewrite."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    cache = cache_path_for_clip(clip)
    cache.write_text("not even json", encoding="utf-8")

    pcm = _pack_int16([1000] * 100)

    def fake_run(cmd, capture_output, check):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=pcm, stderr=b""
        )

    with patch("chalybclip.clip.waveform.subprocess.run", fake_run):
        peaks = load_or_compute(clip, n_buckets=5)
    assert len(peaks) == 5
    # And the cache got rewritten in valid form.
    re_read = json.loads(cache.read_text("utf-8"))
    assert re_read == peaks


def test_load_or_compute_returns_empty_list_on_total_failure(tmp_path: Path) -> None:
    """Missing media + missing cache → []. UI renders a flat scrubber
    rather than 500ing."""
    clip = tmp_path / "clip.mp4"
    # Don't create the file → FileNotFoundError inside compute_waveform.
    peaks = load_or_compute(clip)
    assert peaks == []


def test_default_n_buckets_is_reasonable() -> None:
    """A regression guard so a typo doesn't accidentally explode the
    JSON cache size or shrink the waveform's resolution."""
    assert 64 <= DEFAULT_N_BUCKETS <= 1024


def test_cache_path_lives_next_to_clip(tmp_path: Path) -> None:
    clip = tmp_path / "stream" / "clips" / "clp_x" / "clip.mp4"
    assert cache_path_for_clip(clip) == clip.parent / "waveform.json"
