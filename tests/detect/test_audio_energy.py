"""Tests for the audio energy detector.

Synthetic WAVs are generated programmatically so the tests don't need
fixture files on disk — each test builds the exact loud / quiet pattern
it wants and asserts the detector's response.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from chalybclip.config import AudioEnergyConfig
from chalybclip.detect import detect_audio_energy
from chalybclip.errors import DetectionError
from chalybclip.ingest import Stream

_SAMPLE_RATE = 16000


def _write_wav(path: Path, segments: list[tuple[float, float]], *, seed: int = 0) -> None:
    """Write a 16-bit mono WAV from `[(duration_s, amplitude), ...]`.

    Amplitude is the linear scale (0..1) of white noise; 0.01 is "quiet
    background", 0.4+ is "loud spike". A fixed seed keeps the WAVs
    deterministic across test runs.
    """
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for duration_s, amplitude in segments:
        n = int(_SAMPLE_RATE * duration_s)
        if n <= 0:
            continue
        signal = rng.normal(0.0, amplitude, n)
        clipped = np.clip(signal * 32767.0, -32768.0, 32767.0).astype(np.int16)
        parts.append(clipped)
    full = np.concatenate(parts) if parts else np.array([], dtype=np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(full.tobytes())


def _stream(audio_path: Path, *, tenant_id: str = "ten_a") -> Stream:
    return Stream(
        id="str_01TEST",
        tenant_id=tenant_id,
        vod_url="https://kick.com/c/videos/1",
        platform="kick",
        title="t",
        channel="c",
        duration_s=600.0,
        source_video_path=audio_path.with_name("video.mp4"),
        source_audio_path=audio_path,
    )


def _config(**overrides: object) -> AudioEnergyConfig:
    base = {
        "enabled": True,
        "weight": 0.5,
        "frame_s": 0.5,
        "baseline_window_s": 5.0,
        "spike_ratio": 2.5,
        "sustain_s": 1.5,
    }
    base.update(overrides)
    return AudioEnergyConfig(**base)  # type: ignore[arg-type]


def test_disabled_detector_returns_empty(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    _write_wav(audio, [(1.0, 0.1)])
    cfg = AudioEnergyConfig(enabled=False)
    assert detect_audio_energy("ten_a", _stream(audio), cfg) == []


def test_quiet_audio_yields_no_candidates(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    _write_wav(audio, [(20.0, 0.02)])
    assert detect_audio_energy("ten_a", _stream(audio), _config()) == []


def test_loud_spike_after_quiet_baseline_fires(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    # 10 s quiet + 3 s loud + 5 s quiet
    _write_wav(audio, [(10.0, 0.02), (3.0, 0.4), (5.0, 0.02)])
    cands = detect_audio_energy("ten_a", _stream(audio), _config())
    assert len(cands) == 1
    c = cands[0]
    assert c.reason == "audio"
    # Spike starts in the first loud frame after the baseline kicks in.
    assert 9.0 <= c.timestamp <= 12.0
    assert c.evidence["ratio"] > 2.5
    assert c.evidence["rms"] > c.evidence["baseline_rms"]


def test_short_pop_below_sustain_does_not_fire(tmp_path: Path) -> None:
    """A 0.5 s blip is below sustain_s = 1.5 s, so it gets suppressed."""
    audio = tmp_path / "audio.wav"
    # 10 s quiet + 0.5 s loud (a one-frame pop) + 5 s quiet
    _write_wav(audio, [(10.0, 0.02), (0.5, 0.4), (5.0, 0.02)])
    cands = detect_audio_energy("ten_a", _stream(audio), _config())
    assert cands == []


def test_just_under_spike_ratio_does_not_fire(tmp_path: Path) -> None:
    """RMS that's only ~1.5x baseline doesn't cross spike_ratio = 2.5."""
    audio = tmp_path / "audio.wav"
    _write_wav(audio, [(10.0, 0.05), (3.0, 0.075), (5.0, 0.05)])
    cands = detect_audio_energy("ten_a", _stream(audio), _config())
    assert cands == []


def test_score_saturates_at_weight(tmp_path: Path) -> None:
    """An extreme spike caps at `weight` (no double-counting via ratio)."""
    audio = tmp_path / "audio.wav"
    _write_wav(audio, [(10.0, 0.01), (3.0, 0.6), (5.0, 0.01)])
    cands = detect_audio_energy("ten_a", _stream(audio), _config(weight=0.4))
    assert len(cands) == 1
    assert cands[0].score == pytest.approx(0.4)


def test_two_separated_spikes_yield_two_candidates(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    # 10 quiet + 3 loud + 30 quiet + 3 loud + 5 quiet
    _write_wav(
        audio,
        [
            (10.0, 0.02),
            (3.0, 0.4),
            (30.0, 0.02),
            (3.0, 0.4),
            (5.0, 0.02),
        ],
    )
    cands = detect_audio_energy("ten_a", _stream(audio), _config())
    assert len(cands) == 2
    # The two spikes should be ~33s apart on the timeline
    gap = cands[1].timestamp - cands[0].timestamp
    assert 25.0 < gap < 40.0


def test_tenant_mismatch_raises(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    _write_wav(audio, [(1.0, 0.1)])
    with pytest.raises(DetectionError, match="tenant mismatch"):
        detect_audio_energy("ten_b", _stream(audio, tenant_id="ten_a"), _config())


def test_missing_audio_raises(tmp_path: Path) -> None:
    audio = tmp_path / "missing.wav"
    with pytest.raises(DetectionError, match="audio file missing"):
        detect_audio_energy("ten_a", _stream(audio), _config())


def test_non_16bit_wav_raises(tmp_path: Path) -> None:
    audio = tmp_path / "8bit.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(audio), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)  # 8-bit -- not what ingest produces
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(b"\x00" * (_SAMPLE_RATE * 2))
    with pytest.raises(DetectionError, match="16-bit PCM"):
        detect_audio_energy("ten_a", _stream(audio), _config())


def test_empty_audio_returns_empty(tmp_path: Path) -> None:
    audio = tmp_path / "empty.wav"
    _write_wav(audio, [(0.0, 0.0)])
    assert detect_audio_energy("ten_a", _stream(audio), _config()) == []
