"""Audio energy detector.

Reads the 16 kHz mono WAV produced by ingest, computes RMS per
`frame_s` bin, then sweeps a rolling baseline. A spike fires when:
    * the current frame's RMS exceeds `spike_ratio * baseline`, AND
    * that condition holds for at least `sustain_s` consecutive frames
      (suppresses one-frame pops like dropped objects, audio glitches).

Uses Python's stdlib `wave` + numpy — no librosa, no scipy. The
audio is already 16 kHz mono PCM by ingest's contract; if the file
is something else, the detector raises rather than silently degrading.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from chalybclip.config import AudioEnergyConfig
from chalybclip.errors import DetectionError
from chalybclip.ingest import Stream

from .models import Candidate

_INT16_MAX = 32768.0


def detect_audio_energy(
    tenant_id: str,
    stream: Stream,
    config: AudioEnergyConfig,
) -> list[Candidate]:
    """Return audio-energy candidates for `stream`."""
    if not config.enabled:
        return []
    if tenant_id != stream.tenant_id:
        raise DetectionError(f"tenant mismatch: caller={tenant_id!r}, stream={stream.tenant_id!r}")
    if not stream.source_audio_path.exists():
        raise DetectionError(f"audio file missing: {stream.source_audio_path}")

    rms, frame_s = _read_rms_per_frame(stream.source_audio_path, config.frame_s)
    if rms.size == 0:
        return []

    baseline_frames = max(1, round(config.baseline_window_s / frame_s))
    sustain_frames = max(1, round(config.sustain_s / frame_s))

    return _scan_for_spikes(
        rms=rms,
        frame_s=frame_s,
        baseline_frames=baseline_frames,
        sustain_frames=sustain_frames,
        spike_ratio=config.spike_ratio,
        weight=config.weight,
    )


def _read_rms_per_frame(audio_path: Path, frame_s: float) -> tuple[np.ndarray, float]:
    """Read 16-bit PCM mono WAV, return (rms_per_frame, frame_s)."""
    try:
        with wave.open(str(audio_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames_samples = wf.getnframes()
            raw = wf.readframes(n_frames_samples)
    except wave.Error as e:
        raise DetectionError(f"failed to open WAV {audio_path}: {e}") from e

    if sample_width != 2:
        raise DetectionError(
            f"audio must be 16-bit PCM, got {sample_width * 8}-bit at {audio_path}"
        )

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / _INT16_MAX
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    samples_per_frame = round(framerate * frame_s)
    if samples_per_frame <= 0:
        raise DetectionError(f"frame_s too small for {framerate} Hz audio")

    n_full_frames = samples.shape[0] // samples_per_frame
    if n_full_frames == 0:
        return np.array([], dtype=np.float32), frame_s

    truncated = samples[: n_full_frames * samples_per_frame].reshape(
        n_full_frames, samples_per_frame
    )
    rms = np.sqrt(np.mean(truncated.astype(np.float64) ** 2, axis=1)).astype(np.float32)
    return rms, frame_s


def _scan_for_spikes(
    *,
    rms: np.ndarray,
    frame_s: float,
    baseline_frames: int,
    sustain_frames: int,
    spike_ratio: float,
    weight: float,
) -> list[Candidate]:
    """Single linear pass with a sliding-window mean for the baseline."""
    candidates: list[Candidate] = []
    consecutive_over = 0
    in_spike = False  # already emitted a candidate for this run-of-overs

    for i in range(rms.shape[0]):
        # baseline = mean RMS over [i - baseline_frames, i)
        start = max(0, i - baseline_frames)
        if start == i:
            # No history yet — can't decide a spike on the first frame.
            consecutive_over = 0
            in_spike = False
            continue
        baseline = float(rms[start:i].mean())
        current = float(rms[i])

        is_over = baseline > 0.0 and current > spike_ratio * baseline
        if is_over:
            consecutive_over += 1
            if consecutive_over == sustain_frames and not in_spike:
                in_spike = True
                spike_start_frame = i - sustain_frames + 1
                ratio = current / baseline
                candidates.append(
                    Candidate(
                        timestamp=spike_start_frame * frame_s,
                        score=weight * min(1.0, ratio / spike_ratio),
                        reason="audio",
                        evidence={
                            "rms": current,
                            "baseline_rms": baseline,
                            "ratio": ratio,
                            "frame_s": frame_s,
                            "sustain_s": sustain_frames * frame_s,
                        },
                    )
                )
        else:
            consecutive_over = 0
            in_spike = False

    return candidates
