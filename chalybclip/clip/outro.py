"""End-of-clip splash — the chalybclip outro card (TikTok/CapCut-style).

Every exported clip ends with a short branded end card: the bundled
`assets/outro.mp4` video concatenated after the clip. Free-tier tenants
always get it; paid tiers can switch it off per brand kit
(`brand_kits.show_chalybclip_outro`, default ON) — the caller resolves
that decision and only invokes this module when the outro should land.

Design:

  * `append_outro(...)` is the only public entry. It probes the produced
    MP4 + the bundled asset for fps / dimensions / audio presence, then
    concatenates the asset after the clip via a single ffmpeg
    `-filter_complex concat`. The asset is normalized (fps / scale+pad /
    SAR / pixel format) to the CLIP's geometry so the join is seamless
    regardless of how the asset itself was authored. The result replaces
    the input atomically (temp sibling → `os.replace`).

  * It is called at the points that write the published MP4 — the two
    recorders and the legacy overlay burn — right before each one's
    atomic publish. Baking it into the `.partial.mp4` there means every
    download / publish / serve path inherits the outro for free, exactly
    once, with no overlay/caption bleed (the outro is a separate segment
    appended after the overlay layer).

  * It NEVER raises. An outro that can't render must not break an
    export — on any failure the caller's original bytes are left
    untouched and we return False.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import structlog

_log = structlog.get_logger("chalybclip.clip.outro")

# Audio normalization target for the concat — every segment's audio is
# forced to this so the concat filter (which requires matching sample
# rate + channel layout + sample format across segments) is happy. fltp
# is libx264/aac's native planar float layout.
_AUDIO_SR = 44100
_AUDIO_LAYOUT = "stereo"
_AUDIO_FMT = "fltp"


def outro_asset_path() -> Path:
    """Absolute path to the bundled end-card video that ships with the
    app. Existence is the caller's concern (append_outro checks)."""
    return Path(__file__).parent / "assets" / "outro.mp4"


def _ffprobe_meta(path: Path) -> tuple[int, int, int, bool, float] | None:
    """Return (width, height, fps, has_audio, duration_s) for `path`, or
    None when ffprobe is unavailable or no video stream can be parsed.

    fps is rounded to the nearest integer from the avg_frame_rate
    fraction; has_audio is True when at least one audio stream exists;
    duration comes from the container format.
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    import json as _json

    try:
        proc = subprocess.run(  # noqa: S603 — args list, no shell
            [
                ffprobe, "-v", "error", "-print_format", "json",
                "-show_streams", "-show_format", str(path),
            ],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if proc.returncode != 0:
            return None
        info = _json.loads(proc.stdout or "{}") or {}
    except Exception:  # noqa: BLE001 — never break export on a probe
        return None

    streams = info.get("streams", [])
    width = height = 0
    fps = 30
    has_audio = False
    for s in streams:
        if s.get("codec_type") == "audio":
            has_audio = True
        elif s.get("codec_type") == "video":
            width = int(s.get("width") or 0)
            height = int(s.get("height") or 0)
            rate = str(s.get("avg_frame_rate") or s.get("r_frame_rate") or "")
            if "/" in rate:
                num, _, den = rate.partition("/")
                try:
                    n, d = float(num), float(den)
                    if d > 0 and n > 0:
                        fps = max(1, round(n / d))
                except ValueError:
                    pass
    if width <= 0 or height <= 0:
        return None
    try:
        duration = float(info.get("format", {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    return width, height, fps, has_audio, duration


def build_outro_command(
    *,
    source_path: Path,
    asset_path: Path,
    out_path: Path,
    width: int,
    height: int,
    fps: int,
    source_has_audio: bool,
    asset_has_audio: bool,
    source_duration_s: float,
    asset_duration_s: float,
) -> list[str]:
    """Build the ffmpeg argv that concatenates `asset_path` after
    `source_path`, writing the combined video to `out_path`.

    Pure — no I/O — so it can be unit-tested without ffmpeg. Both video
    segments are normalized to the CLIP's geometry (`width`×`height`,
    `fps`). The output carries audio whenever EITHER segment has it; a
    segment that lacks audio gets a silent track of its own duration
    synthesized via `anullsrc`, so the concat's audio stream is
    continuous across the join. When neither has audio the output is
    video-only.
    """
    # fps / scale-with-letterbox / SAR / pixel-format normalization,
    # applied to both segments so concat sees identical video params.
    v_norm = (
        f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p"
    )

    inputs: list[str] = ["-i", str(source_path), "-i", str(asset_path)]
    graph: list[str] = [f"[0:v]{v_norm}[v0]", f"[1:v]{v_norm}[v1]"]

    want_audio = source_has_audio or asset_has_audio
    if want_audio:
        afmt = (
            f"aformat=sample_fmts={_AUDIO_FMT}:sample_rates={_AUDIO_SR}"
            f":channel_layouts={_AUDIO_LAYOUT}"
        )
        anull = f"anullsrc=channel_layout={_AUDIO_LAYOUT}:sample_rate={_AUDIO_SR}"
        next_idx = 2  # next free input index for synthesized silence

        if source_has_audio:
            graph.append(f"[0:a]{afmt}[a0]")
        else:
            inputs += ["-f", "lavfi", "-t", f"{source_duration_s}", "-i", anull]
            graph.append(f"[{next_idx}:a]{afmt}[a0]")
            next_idx += 1

        if asset_has_audio:
            graph.append(f"[1:a]{afmt}[a1]")
        else:
            inputs += ["-f", "lavfi", "-t", f"{asset_duration_s}", "-i", anull]
            graph.append(f"[{next_idx}:a]{afmt}[a1]")
            next_idx += 1

        graph.append("[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]")
        maps = ["-map", "[v]", "-map", "[a]"]
        audio_args = ["-c:a", "aac", "-b:a", "128k"]
    else:
        graph.append("[v0][v1]concat=n=2:v=1:a=0[v]")
        maps = ["-map", "[v]"]
        audio_args = []

    cmd: list[str] = ["ffmpeg", "-y", "-loglevel", "error"]
    cmd += inputs
    cmd += ["-filter_complex", ";".join(graph)]
    cmd += maps
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    cmd += audio_args
    cmd += [str(out_path)]
    return cmd


def append_outro(video_path: Path) -> bool:
    """Append the bundled chalybclip end card to `video_path` in place.

    Probes the clip + the asset, concatenates, and atomically replaces
    `video_path` with the combined result. Returns True on success,
    False (leaving the original untouched) when ffmpeg / the asset is
    unavailable, the file can't be probed, or ffmpeg fails.

    NEVER raises — the export must survive a failed outro.
    """
    if not video_path.exists():
        return False
    if shutil.which("ffmpeg") is None:
        _log.warning("outro.no_ffmpeg", path=str(video_path))
        return False
    asset = outro_asset_path()
    if not asset.exists():
        _log.warning("outro.no_asset", asset=str(asset))
        return False

    clip_meta = _ffprobe_meta(video_path)
    asset_meta = _ffprobe_meta(asset)
    if clip_meta is None or asset_meta is None:
        _log.warning("outro.probe_failed", path=str(video_path))
        return False
    width, height, fps, clip_has_audio, clip_dur = clip_meta
    _aw, _ah, _afps, asset_has_audio, asset_dur = asset_meta

    tmp_path = video_path.with_suffix(".outro.mp4")
    cmd = build_outro_command(
        source_path=video_path,
        asset_path=asset,
        out_path=tmp_path,
        width=width,
        height=height,
        fps=fps,
        source_has_audio=clip_has_audio,
        asset_has_audio=asset_has_audio,
        source_duration_s=clip_dur,
        asset_duration_s=asset_dur,
    )
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0 or not tmp_path.exists() or tmp_path.stat().st_size == 0:
            err = (proc.stderr or b"").decode("utf-8", errors="replace")[-600:]
            _log.warning("outro.ffmpeg_failed", path=str(video_path), error=err.strip())
            tmp_path.unlink(missing_ok=True)
            return False
        os.replace(tmp_path, video_path)
        return True
    except Exception as e:  # noqa: BLE001 — never break export on the outro
        _log.warning("outro.exception", path=str(video_path), error=str(e))
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False


__all__ = ["append_outro", "build_outro_command", "outro_asset_path"]
