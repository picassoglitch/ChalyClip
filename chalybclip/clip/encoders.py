"""Encoder selection — libx264 vs h264_nvenc (Task 1b).

Probes ffmpeg's encoder list once per process and caches the result.
Selection rules (in order):

  1. If `cfg.encoder` is set to anything other than `libx264`, the operator
     has explicitly chosen a codec — honor it with `cfg.preset`/`cfg.crf`.
  2. Else if `cfg.prefer_nvenc` is True AND `ffmpeg -encoders` lists
     `h264_nvenc`, use NVENC tuned to be visually equal to the libx264
     veryfast / CRF 23 baseline.
  3. Else use libx264 with the configured preset + CRF.

The NVENC arg set (`-preset p5 -rc vbr -cq 23 -b:v 0`) is the operator-
agreed calibration target. `cfg.nvenc_cq` overrides the constant-quality
target without changing the rest of the chain — useful for A/B-ing
quality without re-engineering this module.

Visual identity guard: NVENC's quality at p5/CQ 23 is close to libx264
veryfast CRF 23 on the social-video targets we ship (1080×1920, ≤60s),
but the platforms re-encode aggressively anyway. The escape hatch is
`prefer_nvenc=False` in config — restores the old path bit-for-bit.

Thread-safety: `has_nvenc()` is called from worker threads inside the
parallel cut pipeline; the cached probe is a single bool assignment so
torn reads are harmless.
"""

from __future__ import annotations

import shutil
import subprocess

from chalybclip.config import ClipConfig
from chalybclip.logging import get_logger

_log = get_logger("chalybclip.clip.encoders")

# Sentinel: None = not yet probed; True/False = probed result.
_NVENC_CACHE: bool | None = None
# Sticky failure flag — the cut pipeline flips this after a runtime
# NVENC error (libcuda missing, GPU died mid-run, driver crashed).
# Once True, has_nvenc() returns False for the rest of the process so
# subsequent clips don't burn time retrying NVENC.
_NVENC_RUNTIME_DISABLED: bool = False


def has_nvenc() -> bool:
    """Returns True iff a tiny NVENC encode actually succeeds.

    Cached per-process. Any failure mode (no ffmpeg on PATH, probe
    timeout, libcuda missing, encoder exits non-zero) resolves to
    False — we never fail the cut pipeline because the probe blew up.

    A runtime failure during real encoding (via `mark_nvenc_runtime_failure`)
    sticks for the rest of the process — once we see "Cannot load
    libcuda.so.1" once, all subsequent clips skip NVENC.
    """
    global _NVENC_CACHE
    if _NVENC_RUNTIME_DISABLED:
        return False
    if _NVENC_CACHE is None:
        _NVENC_CACHE = _probe_nvenc()
    return _NVENC_CACHE


def mark_nvenc_runtime_failure(reason: str = "") -> None:
    """Permanently disable NVENC for this process.

    Called by the cut pipeline when an ffmpeg invocation that USED
    NVENC fails. The fallback path retries with libx264; this flag
    keeps subsequent encodes from re-trying NVENC and re-failing the
    same way.
    """
    global _NVENC_RUNTIME_DISABLED
    if not _NVENC_RUNTIME_DISABLED:
        _log.warning(
            "nvenc.runtime_disabled",
            reason=reason[:200] if reason else "unspecified",
        )
    _NVENC_RUNTIME_DISABLED = True


def reset_nvenc_cache() -> None:
    """Test hook — discard the cached probe so the next call re-probes."""
    global _NVENC_CACHE, _NVENC_RUNTIME_DISABLED
    _NVENC_CACHE = None
    _NVENC_RUNTIME_DISABLED = False


def ffmpeg_thread_args(cfg: ClipConfig) -> list[str]:
    """`-threads N` for software (CPU) encodes, or `[]` when unset.

    Returns an empty slice when `cfg.ffmpeg_threads` is 0 so ffmpeg keeps
    its `threads=auto` behavior; otherwise pins the encoder to N threads to
    keep a single encode from oversubscribing the host's thread ceiling.
    """
    n = int(getattr(cfg, "ffmpeg_threads", 0) or 0)
    return ["-threads", str(n)] if n > 0 else []


def pick_video_encoder_args(cfg: ClipConfig) -> list[str]:
    """Return the ffmpeg `-c:v ...` argument sequence for the reformat
    step. Applies the selection rules in the module docstring.

    Output is a flat argv slice — splice it directly into the ffmpeg
    command between the filter and the audio codec args.
    """
    # Rule 1 — operator override.
    encoder = (cfg.encoder or "").strip() or "libx264"
    if encoder != "libx264":
        return [
            "-c:v", encoder, "-preset", cfg.preset, "-crf", str(cfg.crf),
            *ffmpeg_thread_args(cfg),
        ]

    # Rule 2 — NVENC if available + preferred. GPU-bound, so no -threads cap.
    if getattr(cfg, "prefer_nvenc", True) and has_nvenc():
        cq = int(getattr(cfg, "nvenc_cq", cfg.crf))
        return [
            "-c:v", "h264_nvenc",
            "-preset", "p5",
            "-rc", "vbr",
            "-cq", str(cq),
            "-b:v", "0",
        ]

    # Rule 3 — libx264 fallback.
    return [
        "-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf),
        *ffmpeg_thread_args(cfg),
    ]


# ---- internals ----


def _probe_nvenc() -> bool:
    """Verify NVENC is BOTH compiled in AND usable at runtime.

    The naive `ffmpeg -encoders | grep h264_nvenc` probe gives false
    positives on hosts where ffmpeg was built with nvenc support but
    `libcuda.so.1` isn't installed (typical for Railway / fly.io CPU
    images that pull a generic ffmpeg). The encoder lists fine; the
    first actual encode crashes with "Cannot load libcuda.so.1" and
    the cut step explodes.

    The robust check is to actually attempt a tiny 1-frame encode and
    inspect the exit code:

      ffmpeg -hide_banner -loglevel error
        -f lavfi -i nullsrc=s=64x64:d=1   # synthesized 1-frame source
        -c:v h264_nvenc -t 1
        -f null -                          # discard the output

    On a working host this completes in ~300ms. On a CPU-only host the
    NVENC encoder bails immediately with a libcuda error and exit != 0.
    Cached per-process so the cost is paid once at first cut.
    """
    if shutil.which("ffmpeg") is None:
        _log.info("nvenc.probe", available=False, reason="ffmpeg_not_on_path")
        return False
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-f", "lavfi",
                "-i", "nullsrc=s=64x64:d=1",
                "-c:v", "h264_nvenc",
                "-t", "1",
                "-f", "null",
                "-",
            ],
            capture_output=True, check=False, timeout=10.0, text=True,
        )
    except Exception as e:  # noqa: BLE001 — probe failure must not break cut
        _log.warning("nvenc.probe.failed", error=str(e))
        return False
    if proc.returncode == 0:
        _log.info("nvenc.probe", available=True)
        return True
    # Non-zero exit → NVENC unusable. Capture the first line of stderr
    # so production logs show WHY (libcuda missing, no GPU, driver
    # too old, etc.) instead of just "available=False".
    stderr_first_line = (proc.stderr or "").strip().splitlines()
    reason = stderr_first_line[0][:200] if stderr_first_line else "unknown"
    _log.info("nvenc.probe", available=False, reason=reason)
    return False


__all__ = [
    "ffmpeg_thread_args",
    "has_nvenc",
    "mark_nvenc_runtime_failure",
    "pick_video_encoder_args",
    "reset_nvenc_cache",
]
