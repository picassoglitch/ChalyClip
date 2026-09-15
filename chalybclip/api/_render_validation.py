"""Render Migration R1-R3 — validate the produced MP4 BEFORE it can
be served to the operator.

The download path used to trust two weak signals:

  1. The recorder didn't raise → mark_render_ready.
     But ffmpeg can return exit code 0 and still produce a malformed
     MP4 (rare but real — missing moov atom, broken audio mux, weird
     timestamps). The operator then downloaded a file Windows refused
     to open with 0xC00D36C4 codec error.

  2. `clip_render_<res>.mp4` exists on disk → cache hit, serve it.
     But a partial / 0-byte / truncated file also passes `.exists()`.
     Combined with (1), an aborted hybrid run could leave a corrupt
     fragment that the next download click happily delivered.

This module provides:

  - `validate_rendered_mp4(path, expected_duration_s)` — strict gate
    used by the background runner before flipping state=ready. Spawns
    `ffprobe -show_streams -show_format`; requires ≥1 video stream,
    ≥1 audio stream (we always mux source audio), and a duration
    that's at least 80% of what was requested. Returns a small tuple
    (ok, reason). The runner deletes the file + marks failed on a
    False verdict — partial bytes never survive the gate.

  - `is_servable_cached_mp4(path)` — fast O(1) check used by the
    download / status endpoints before treating an on-disk file as
    a cache hit. No subprocess: size floor + ISO BMFF magic-byte
    signature ("ftyp" at offset 4). Designed to be cheap enough to
    run on every cache-hit decision without measurable latency.

Both helpers are deliberately separate. The strict ffprobe pass runs
ONCE at render time (the marginal ~50-150ms is free relative to a
full render). The cheap magic-byte check runs on EVERY download
request and gates the cache trust.
"""
from __future__ import annotations

import json as _json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Final

_log = logging.getLogger("chalybclip.render_validation")

# A real 1080p clip of even 5 seconds at the recorder's CRF settings
# weighs north of ~1 MB; the legacy seek-and-shoot produces a slightly
# larger file than the hybrid alpha path but both clear the 1 MB floor
# by a wide margin. Anything smaller is debris from a crashed encode,
# not a real export.
_MIN_SERVABLE_BYTES: Final = 1_000_000

# ISO BMFF major-brand discriminator. The first 4 bytes are the box
# size (big-endian uint32) and bytes 4-8 are the box type. The
# top-level box of every well-formed MP4 (faststart or otherwise)
# starts with `ftyp`. Without this, the file is not an MP4.
_FTYP_OFFSET: Final = 4
_FTYP_MAGIC: Final = b"ftyp"

# How much of the requested duration the produced file must cover for
# us to call the render trustworthy. 80% is the floor; in practice
# ffmpeg-encoded files land within 100ms of the request. We pick the
# lower bound deliberately loose so transient timestamp glitches
# don't kill an otherwise-correct render.
_MIN_DURATION_RATIO: Final = 0.80


def is_servable_cached_mp4(path: Path) -> bool:
    """Cheap is-this-file-actually-a-real-MP4 check.

    Used by the download / status endpoints before treating an
    on-disk file as a cache hit. No subprocess, no encoding work:

      - File exists.
      - Size is at least 1 MB (real 1080p clips are megabytes;
        anything smaller is debris from a crashed ffmpeg run).
      - Bytes 4-8 are b"ftyp" (the ISO BMFF major-brand marker
        every well-formed MP4 carries).

    Returns False on any failure — including read errors — so a
    quirky filesystem state never causes us to serve garbage.
    Caller decides whether to unlink + re-render or just 202.
    """
    try:
        if not path.exists():
            return False
        if path.stat().st_size < _MIN_SERVABLE_BYTES:
            return False
        with path.open("rb") as f:
            header = f.read(_FTYP_OFFSET + len(_FTYP_MAGIC))
        if len(header) < _FTYP_OFFSET + len(_FTYP_MAGIC):
            return False
        return header[_FTYP_OFFSET:_FTYP_OFFSET + len(_FTYP_MAGIC)] == _FTYP_MAGIC
    except OSError:
        # Filesystem flake — be conservative, treat as not-servable.
        return False


def validate_rendered_mp4(
    path: Path,
    *,
    expected_duration_s: float,
) -> tuple[bool, str | None]:
    """Strict gate run by the background renderer before marking ready.

    Spawns ffprobe and checks:

      - File is at least 1 MB (cheap pre-check; rejects 0-byte
        partials without paying for ffprobe).
      - ffprobe parses the container (exit 0).
      - There is at least one video stream.
      - There is at least one audio stream (the recorder always
        muxes source audio; missing audio means the mux failed).
      - Duration is at least 80% of the requested length (catches
        early-EOF / truncated containers).

    Returns (True, None) on success or (False, reason) on failure.
    Caller is expected to delete the file and mark the render
    failed when this returns False — that way the broken bytes
    never leak into the download endpoint's cache-hit branch.

    Best-effort on ffprobe-missing: if ffprobe isn't on PATH the
    check degrades to the cheap magic-byte gate. The render still
    gets accepted — we don't want to fail every Railway deploy
    where someone forgot to install ffmpeg-full — but we log a
    warning so the operator knows their validation is degraded.
    """
    if not path.exists():
        return (False, f"output file missing: {path.name}")

    try:
        size = path.stat().st_size
    except OSError as e:
        return (False, f"stat failed: {e}")

    if size < _MIN_SERVABLE_BYTES:
        return (
            False,
            (
                f"output is {size} bytes; expected at least "
                f"{_MIN_SERVABLE_BYTES} (likely a truncated / "
                "aborted encode)"
            ),
        )

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        # Degraded mode — fall back to magic-byte gate. Log once so
        # the operator gets paged before this becomes the next "why
        # didn't we catch the corrupt file" incident.
        _log.warning(
            "render_validation.ffprobe_missing path=%s "
            "size=%d — falling back to magic-byte gate only",
            path, size,
        )
        if is_servable_cached_mp4(path):
            return (True, None)
        return (False, "ffprobe missing AND file failed magic-byte check")

    try:
        proc = subprocess.run(  # noqa: S603 — args list
            [
                ffprobe, "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", str(path),
            ],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except subprocess.TimeoutExpired:
        return (False, "ffprobe timed out after 30s")
    except OSError as e:
        return (False, f"ffprobe spawn failed: {e}")

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()[:200]
        return (False, f"ffprobe rc={proc.returncode}: {stderr}")

    try:
        info = _json.loads(proc.stdout or "{}")
    except _json.JSONDecodeError as e:
        return (False, f"ffprobe output unparseable: {e}")

    streams = info.get("streams") or []
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if not video_streams:
        return (False, "no video stream in output")
    if not audio_streams:
        return (False, "no audio stream in output")

    try:
        actual_duration = float(
            (info.get("format") or {}).get("duration") or 0.0
        )
    except (TypeError, ValueError):
        actual_duration = 0.0

    if actual_duration <= 0:
        return (False, "duration in container is zero")

    if expected_duration_s > 0:
        ratio = actual_duration / expected_duration_s
        if ratio < _MIN_DURATION_RATIO:
            return (
                False,
                (
                    f"duration {actual_duration:.2f}s is only "
                    f"{ratio:.0%} of requested "
                    f"{expected_duration_s:.2f}s — likely truncated"
                ),
            )

    return (True, None)


__all__ = ["validate_rendered_mp4", "is_servable_cached_mp4"]
