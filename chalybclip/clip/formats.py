"""Clip output formats: vertical 9:16 (the cut), horizontal 16:9, square 1:1.

The cut step produces `clip.mp4` already reformatted to 9:16. The other two
formats need the full frame, so the cut also keeps the full-frame cut it
starts from as `clip_landscape.mp4` (the "master", see `save_master`) and
uploads it next to the clip in the bucket. A format's base video — the
frames the export render composites its overlays onto — is derived from
the master on demand, for the clip's CURRENT bounds (so manual/auto trims
and Deshacer are honoured), and cached beside the clip:

    clip_horizontal_<start_ms>_<end_ms>.mp4   1920x1080, letterboxed
    clip_square_<start_ms>_<end_ms>.mp4       1080x1080, smart-crop centered

The master covers the clip's pristine window — `original_start_s/end_s`
when the clip was trimmed, else `start_s/end_s` (the trim code keeps the
first pre-trim bounds there). When the master is gone (clip cut before this
existed) the stream's source video is used if it's still on disk; otherwise
the format isn't available for that clip.

The chosen format lives in the clip's `overlay_config["format"]`; the
render (`api/_clip_render.render_clip_in_background`) reads it, so download,
approve pre-render and publish all ship the chosen format with no change to
their cache paths.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import structlog

_log = structlog.get_logger(__name__)

FORMATS: tuple[str, ...] = ("vertical", "horizontal", "square")
MASTER_NAME = "clip_landscape.mp4"


def clip_format(clip: Any) -> str:
    """The format chosen for `clip` (vertical when unset/unknown)."""
    ov = getattr(clip, "overlay_config", None)
    fmt = ov.get("format") if isinstance(ov, dict) else None
    return fmt if isinstance(fmt, str) and fmt in FORMATS else "vertical"


def render_dims(fmt: str, width: int, height: int) -> tuple[int, int]:
    """Viewport for `fmt` given the vertical (width, height) of a quality
    preset: 1080x1920 → 1920x1080 horizontal, 1080x1080 square."""
    if fmt == "horizontal":
        return height, width
    if fmt == "square":
        return width, width
    return width, height


def master_path(clip_dir: Path) -> Path:
    return clip_dir / MASTER_NAME


def save_master(intermediate: Path, clip_dir: Path) -> None:
    """Keep the cut step's full-frame intermediate as the format master.
    Best-effort: a failure only means horizontal/square need the source."""
    try:
        if intermediate.exists():
            shutil.copy2(intermediate, master_path(clip_dir))
    except Exception as e:
        _log.warning("formats.save_master_failed", clip_dir=str(clip_dir), error=str(e)[:200])


def pristine_window(clip: Any) -> tuple[float, float]:
    """The window the master covers (see module docstring)."""
    start = getattr(clip, "original_start_s", None)
    end = getattr(clip, "original_end_s", None)
    if start is None or end is None:
        return float(clip.start_s), float(clip.end_s)
    return float(start), float(end)


def base_name(fmt: str, start_s: float, end_s: float) -> str:
    return f"clip_{fmt}_{round(start_s * 1000)}_{round(end_s * 1000)}.mp4"


def _video_filter(fmt: str, center_px: float | None) -> str:
    if fmt == "horizontal":
        return (
            "scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,format=yuv420p"
        )
    # Square: the largest square, centered on the smart-crop subject (the
    # box is in source pixels; the master is a same-resolution cut).
    cx = "iw/2" if center_px is None else f"{center_px:.1f}"
    return (
        "crop='min(iw,ih)':'min(iw,ih)':"
        f"'max(0,min(iw-min(iw,ih),{cx}-min(iw,ih)/2))':'(ih-min(iw,ih))/2',"
        "scale=1080:1080,setsar=1,format=yuv420p"
    )


def _encode(*, src: Path, out: Path, local_start_s: float, duration_s: float, vf: str) -> None:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    tmp = out.with_suffix(".tmp.mp4")
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, local_start_s):.3f}",
        "-i",
        str(src),
        "-t",
        f"{duration_s:.3f}",
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)


def crop_center_px(clip: Any) -> float | None:
    """Horizontal center of the smart-crop box, in source pixels."""
    box = getattr(clip, "smart_crop_box", None)
    if not isinstance(box, dict):
        return None
    try:
        x = float(box.get("x", 0.0))
        w = float(box.get("w", 0.0))
    except (TypeError, ValueError):
        return None
    if w <= 0 or x < 0:
        return None
    return x + w / 2


async def ensure_master(*, clip: Any, tenant_id: str) -> Path | None:
    """The local master, pulled back from the bucket when reclaimed."""
    clip_dir = Path(clip.path).parent
    master = master_path(clip_dir)
    if master.is_file():
        return master
    try:
        from chalybclip.integrations.storage import build_artifact_store, clip_landscape_key
        from chalybclip.settings import get_settings

        store = build_artifact_store(get_settings())
        if store is None:
            return None
        tmp = master.with_name(f"{master.name}.rehydrating")
        got = await store.download(key=clip_landscape_key(tenant_id, clip.id), dest=tmp)
        if got is None:
            tmp.unlink(missing_ok=True)
            return None
        tmp.replace(master)
        return master
    except Exception as e:
        _log.warning("formats.master_rehydrate_failed", clip_id=clip.id, error=str(e)[:200])
        return None


async def available_formats(*, clip: Any, tenant_id: str, source_video: Path | None) -> list[str]:
    """Which formats can be produced for `clip` right now."""
    out = ["vertical"]
    has_source = source_video is not None and source_video.is_file()
    has_master = master_path(Path(clip.path).parent).is_file()
    if not has_master and not has_source:
        try:
            from chalybclip.integrations.storage import build_artifact_store, clip_landscape_key
            from chalybclip.settings import get_settings

            store = build_artifact_store(get_settings())
            has_master = bool(
                store is not None and await store.exists(key=clip_landscape_key(tenant_id, clip.id))
            )
        except Exception:
            has_master = False
    if has_master or has_source:
        out += ["horizontal", "square"]
    return out


async def ensure_format_base(
    *, clip: Any, fmt: str, tenant_id: str, source_video: Path | None
) -> Path | None:
    """The base video for `fmt` at the clip's current bounds; None when it
    can't be produced (no master inside the window, no source)."""
    if fmt == "vertical":
        return Path(clip.path)
    if fmt not in FORMATS:
        return None
    clip_dir = Path(clip.path).parent
    start, end = float(clip.start_s), float(clip.end_s)
    out = clip_dir / base_name(fmt, start, end)
    if out.is_file() and out.stat().st_size > 0:
        return out
    lo, hi = pristine_window(clip)
    vf = _video_filter(fmt, crop_center_px(clip))
    src: Path | None = None
    local_start = 0.0
    if lo - 0.01 <= start and end <= hi + 0.01:
        master = await ensure_master(clip=clip, tenant_id=tenant_id)
        if master is not None:
            src, local_start = master, start - lo
    if src is None and source_video is not None and source_video.is_file():
        src, local_start = source_video, start
    if src is None:
        return None
    import asyncio

    await asyncio.to_thread(
        _encode, src=src, out=out, local_start_s=local_start, duration_s=end - start, vf=vf
    )
    # Bases for older bounds of this format are stale now.
    for old in clip_dir.glob(f"clip_{fmt}_*.mp4"):
        if old != out:
            old.unlink(missing_ok=True)
    return out


__all__ = [
    "FORMATS",
    "MASTER_NAME",
    "available_formats",
    "base_name",
    "clip_format",
    "ensure_format_base",
    "ensure_master",
    "master_path",
    "pristine_window",
    "render_dims",
    "save_master",
]
