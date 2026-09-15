"""Audio waveform extractor — normalized peak buckets for the editor scrubber.

Pipeline:
    clip.mp4 → ffmpeg -ac 1 -ar 8000 -f s16le → PCM bytes → bucketed peaks → JSON

The output is a list of `n_buckets` floats in [0, 1] representing the
peak absolute amplitude in each time bucket. The clip editor renders
these as a horizontal SVG waveform under the preview, with a playhead
cursor that follows `<video>.currentTime`.

The result is cached as `<clip_dir>/waveform.json` so subsequent page
loads (and the click-to-seek scrubber) read from disk instead of
re-running ffmpeg. Cache is invalidated by deleting the file — there's
no version field; if we change the bucket count or sample rate, ops
just `rm out/<stream>/clips/<clip>/waveform.json`.

stdlib-only (struct + subprocess) — no numpy dep.
"""

from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

import structlog

_log = structlog.get_logger(__name__)

# Sane bucket count — the SVG renders ~1.5px per bucket at typical
# preview widths, so 200 gives a nice dense waveform without bloating
# the JSON cache.
DEFAULT_N_BUCKETS = 200
# 8 kHz mono is enough resolution for a peak waveform (we're not
# trying to resynthesize the audio). Cuts ffmpeg work + PCM size.
TARGET_SAMPLE_RATE = 8000


def compute_waveform(
    media_path: Path, *, n_buckets: int = DEFAULT_N_BUCKETS
) -> list[float]:
    """Return `n_buckets` peak-normalized floats in [0, 1] over the audio
    track of `media_path` (mp4, wav, anything ffmpeg can decode).

    Raises:
        FileNotFoundError if the media is missing.
        RuntimeError on ffmpeg failure or empty audio track.
    """
    if not media_path.exists():
        raise FileNotFoundError(f"media file missing: {media_path}")
    cmd = [
        "ffmpeg",
        "-i", str(media_path),
        "-ac", "1",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-f", "s16le",
        "-loglevel", "error",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed extracting audio from {media_path}: "
            f"{proc.stderr.decode('utf-8', errors='replace')[:400]}"
        )
    pcm = proc.stdout
    n_samples = len(pcm) // 2  # int16 = 2 bytes
    if n_samples == 0:
        raise RuntimeError(f"empty audio track in {media_path}")
    return _bucket_peaks(pcm, n_samples=n_samples, n_buckets=n_buckets)


def _bucket_peaks(
    pcm: bytes, *, n_samples: int, n_buckets: int
) -> list[float]:
    """Walk the int16 PCM stream once, take the max abs per bucket."""
    bucket_size = max(1, n_samples // n_buckets)
    peaks: list[float] = []
    for i in range(n_buckets):
        # Byte offsets — each int16 sample is 2 bytes.
        start = i * bucket_size * 2
        end = min(len(pcm), start + bucket_size * 2)
        chunk = pcm[start:end]
        if len(chunk) < 2:
            peaks.append(0.0)
            continue
        n_in_chunk = len(chunk) // 2
        ints = struct.unpack(f"<{n_in_chunk}h", chunk[: n_in_chunk * 2])
        peak = max((abs(x) for x in ints), default=0)
        # int16 max abs is 32768 (one extra on the negative side, but
        # close enough for a normalized display).
        peaks.append(peak / 32768.0)
    return peaks


# ---- on-disk cache ----


def cache_path_for_clip(clip_path: Path) -> Path:
    """`<clip_dir>/waveform.json` — sits next to the clip MP4."""
    return clip_path.parent / "waveform.json"


def load_or_compute(
    clip_path: Path, *, n_buckets: int = DEFAULT_N_BUCKETS
) -> list[float]:
    """Read the cached waveform JSON if present, else compute + cache.

    Returns `[]` on any error (missing file, ffmpeg failure, etc.) —
    the UI degrades to "no waveform" gracefully rather than 500ing.
    """
    cache = cache_path_for_clip(clip_path)
    if cache.exists():
        try:
            data = json.loads(cache.read_text("utf-8"))
            if isinstance(data, list) and all(isinstance(x, int | float) for x in data):
                return [float(x) for x in data]
        except Exception as e:
            _log.warning("waveform.cache_read_failed", path=str(cache), error=str(e))

    try:
        peaks = compute_waveform(clip_path, n_buckets=n_buckets)
    except Exception as e:
        _log.warning("waveform.compute_failed", path=str(clip_path), error=str(e))
        return []

    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(peaks), encoding="utf-8")
    except OSError as e:
        _log.warning("waveform.cache_write_failed", path=str(cache), error=str(e))
    return peaks
