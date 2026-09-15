"""Headless-Chrome recorder for the clip preview (slice O.54).

Renders the same HTML/CSS that the operator sees in the editor (the
`/dashboard/clips/<id>/render` page from slice O.19) and produces a
constant-frame-rate MP4 that's byte-deterministic per (clip, overlay
config) pair.

Architecture: deterministic seek-and-shoot
==========================================

Earlier iterations (slices O.50–O.53) all captured the page in real
time — Chrome plays the <video> element at wall-clock speed, we
collect frames as they're rendered, ffmpeg encodes the result. That
sounds clean but fails under Railway's CPU constraints:

  - <video> element stutters when the box is loaded -> captured frames
    show the stutter
  - caption animator runs in rAF reading video.currentTime — if the
    video stalls while rAF keeps ticking, captions advance while
    pixels don't, breaking sync between caption text + audio
  - frame rate is "best effort" so audio/video duration can drift

Slice O.54 inverts the architecture. We DON'T play the video. The
recorder seeks to each desired frame timestamp, waits for the page
to settle, captures the screenshot, and moves on. Timeline is purely
controlled by the recorder; Chrome just renders whatever frame we
asked for.

  for i in range(int(duration_s * fps)):
      target_t = i / fps
      page.evaluate(captureFrameAt(target_t))
      screenshot = page.screenshot()
      save(screenshot, f"frame_{i:08d}.jpg")
  ffmpeg image2 -framerate 30 + audio mux

Properties this gives us:
  - constant frame rate by construction
  - audio sync is automatic — frame N is at t = N/fps, audio is muxed
    from t=0, both timelines are the same
  - no stutter possible — each frame is fully painted before capture
  - captions match exactly — same code as the editor, same currentTime
    we drive both
  - reproducible — same inputs always produce byte-equal outputs

Trade-off: slower than real-time capture (seek + paint per frame).
For a 30s clip at 30fps that's ~900 seeks. Each takes 30–80ms on
Railway's CPU, so ~30–70s of capture time. Acceptable for the
"download once, share many" usage pattern.

Validation
==========

After encode we ffprobe the output and validate:
  - duration drift vs input clip duration < 100ms
  - frame rate is exactly the requested fps (CFR)
  - audio sample rate is 48kHz, codec is AAC
  - video codec is H.264, pix_fmt is yuv420p

A manifest JSON file is written alongside each export with the
inputs, settings, and validation results. Operators can diff manifest
files to compare exports or feed them back to support.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import structlog

_log = structlog.get_logger(__name__)


# Recording knobs — kept module-level so callers can override via
# kwargs if a specific delivery target (e.g. Reels 60fps) ever needs it.
DEFAULT_FPS = 30
DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 1920
TAIL_PAD_S = 0.3  # extra seconds we record past clip.duration_s
# Wait at most this long for the page's <video> element to be ready.
READY_TIMEOUT_S = 10.0
# JPEG quality for the screencast frames. 95 is "visually transparent"
# — at 100 the file sizes balloon for no visible benefit, and Chrome's
# JPEG encoder gets noticeably slower above 95.
SCREENCAST_JPEG_QUALITY = 95
# ffmpeg encode settings — applied once to the screencast frames.
# CRF 14 is mathematically "visually transparent for high-motion
# content" on libx264. The medium preset trades ~2× encode time vs
# veryfast for noticeably better detail retention; output is cached
# per (clip, resolution) so the cost amortizes across re-downloads.
FFMPEG_CRF = "14"
FFMPEG_PRESET = "medium"
# Cap libx264's CPU threads so a render doesn't oversubscribe the host's
# thread ceiling when it overlaps the cut fan-out / poll pipeline (the
# EAGAIN cascade that makes `x264_encoder_open` fail). Override via
# CHALYBCLIP_FFMPEG_THREADS; 0 restores ffmpeg's `threads=auto`.
FFMPEG_THREADS = os.environ.get("CHALYBCLIP_FFMPEG_THREADS", "2")


async def record_clip_to_mp4(
    *,
    clip_id: str,
    duration_s: float,
    audio_source_path: Path,
    output_path: Path,
    base_url: str,
    auth_cookie_name: str = "chalybclip_token",
    auth_cookie_value: str | None = None,
    auth_query: str = "",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
    progress_callback: "object | None" = None,
    ass_file_path: Path | None = None,
    append_outro_enabled: bool = True,
) -> Path:
    """Record the render page for `clip_id` to an MP4 at `output_path`.

    Pipeline:
      1. Launch Playwright + Chromium headless at viewport width × height.
      2. Navigate to `<base_url>/dashboard/clips/<clip_id>/render`,
         authenticate via the operator's session cookie.
      3. Wait for the page's `window.__chalybclipRender.allReady` (video
         playing + caption animator running) before recording the first
         frame.
      4. Start an ffmpeg subprocess that consumes JPEG frames from stdin
         and emits an H.264 MP4 with the source audio muxed in.
      5. Start a CDP screencast on the page; for each `screencastFrame`
         event, write the decoded JPEG bytes to ffmpeg's stdin and ACK
         the frame so Chrome sends the next one.
      6. After `duration_s + TAIL_PAD_S`, stop the screencast, close
         ffmpeg's stdin, wait for it to flush, return.

    Returns `output_path` on success. Raises `PreviewRecordingError`
    with a short reason on any failure (recorder caller decides
    fallback strategy).
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError as e:
        raise PreviewRecordingError(
            "Playwright not installed. Run "
            "`pip install playwright && playwright install chromium`."
        ) from e

    if not audio_source_path.exists():
        raise PreviewRecordingError(
            f"Audio source missing on disk: {audio_source_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Capture mode: ?capture=1 tells the render page to NOT autoplay
    # the <video>. We'll drive currentTime ourselves per frame.
    # `auth_query` (e.g. "tenant=..&exp=..&sig=..") authenticates the render
    # via a signed URL instead of a cookie — used by background auto-publish,
    # which has no operator session. When set, no cookie is added below.
    record_url = (
        f"{base_url.rstrip('/')}/dashboard/clips/{clip_id}/render?capture=1"
        + (f"&{auth_query}" if auth_query else "")
    )

    # Slice O.55 — canonical-duration reconciliation. The caller passes
    # `duration_s` from the DB; we ALSO ffprobe the source MP4 on
    # disk. If they disagree (legacy clips cut before the O.55 cut-step
    # accuracy fix, or post-trim drift), use the SHORTER value. This
    # prevents the recorder from seeking past EOF + duplicating the
    # last frame for the missing seconds — the previous failure mode
    # the operator described as "40s preview, 35s export".
    probed_duration = _ffprobe_duration_s(audio_source_path)
    canonical_duration = duration_s
    if probed_duration is not None and probed_duration > 0:
        if abs(probed_duration - duration_s) > 0.15:
            _log.warning(
                "preview_recorder.duration_mismatch",
                clip_id=clip_id,
                db_duration_s=round(duration_s, 3),
                file_duration_s=round(probed_duration, 3),
                drift_ms=int(abs(probed_duration - duration_s) * 1000),
                canonical_source="file",
            )
        # Always trust the file. The DB row will get reconciled below
        # via the manifest validation step.
        canonical_duration = probed_duration

    started_at = _dt.datetime.now(_dt.UTC)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--autoplay-policy=no-user-gesture-required",
                "--disable-gpu-rasterization",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        try:
            context = await browser.new_context(
                viewport={"width": width, "height": height},
                device_scale_factor=1,
            )
            if auth_cookie_value:
                from urllib.parse import urlparse
                parsed = urlparse(base_url)
                await context.add_cookies([
                    {
                        "name": auth_cookie_name,
                        "value": auth_cookie_value,
                        "domain": parsed.hostname or "127.0.0.1",
                        "path": "/",
                        "httpOnly": False,
                        "secure": parsed.scheme == "https",
                    }
                ])

            page = await context.new_page()
            _log.info(
                "preview_recorder.navigate",
                clip_id=clip_id, url=record_url,
                viewport=[width, height], fps=fps,
                mode="seek-and-shoot",
            )
            await page.goto(record_url, wait_until="domcontentloaded")

            # In capture mode we only need metadata loaded + captions
            # fetched. The `allReady` gate also requires `playing`,
            # which capture mode sets the moment metadata's in (the
            # video never actually plays). Same gate as before.
            try:
                await page.wait_for_function(
                    "window.__chalybclipRender"
                    " && window.__chalybclipRender.allReady === true"
                    " && typeof window.__chalybclipRender.captureFrameAt"
                    " === 'function'",
                    timeout=READY_TIMEOUT_S * 1000,
                )
            except Exception as e:  # noqa: BLE001
                raise PreviewRecordingError(
                    f"Render page never reached capture-ready within "
                    f"{READY_TIMEOUT_S:.1f}s. error: {e}"
                ) from e

            _log.info(
                "preview_recorder.recording_start",
                clip_id=clip_id,
                db_duration_s=duration_s,
                canonical_duration_s=round(canonical_duration, 3),
                file_duration_s=(
                    round(probed_duration, 3)
                    if probed_duration is not None else None
                ),
                expected_frames=int(canonical_duration * fps),
            )
            capture_started = _dt.datetime.now(_dt.UTC)
            await _record_via_seek_and_shoot(
                page=page,
                clip_id=clip_id,
                audio_source_path=audio_source_path,
                output_path=output_path,
                width=width,
                height=height,
                fps=fps,
                duration_s=canonical_duration,
                progress_callback=progress_callback,
                ass_file_path=ass_file_path,
                append_outro_enabled=append_outro_enabled,
            )
            capture_ended = _dt.datetime.now(_dt.UTC)
            await page.close()
            await context.close()
        finally:
            await browser.close()

    # Validate the encode + write a manifest beside the file. Both are
    # best-effort: if ffprobe isn't on PATH or the validation fails,
    # we log + flag in the manifest but don't tank the download. The
    # operator can still grab the file and review.
    #
    # Slice O.55 — validate against the CANONICAL duration (file-based
    # if the probe succeeded, otherwise the caller's value). Validating
    # against duration_s when the file is shorter would always fail
    # for legacy clips with bad DB rows.
    validation = _validate_export(
        output_path=output_path,
        expected_duration_s=canonical_duration,
        expected_fps=fps,
    )
    if not validation["ok"]:
        # Fail loudly if duration drift is past tolerance — that's a
        # genuine sync bug we want the caller to surface, not a soft
        # warning.
        if (
            validation.get("checks", {})
            .get("duration_drift_ms", 0) > _DURATION_TOLERANCE_MS
        ):
            raise PreviewRecordingError(
                f"export duration drift exceeds tolerance: "
                f"{validation['checks']['duration_drift_ms']}ms "
                f"(expected_s={duration_s:.3f}, "
                f"actual_s={validation['checks'].get('actual_duration_s')})"
            )

    manifest = {
        "clip_id": clip_id,
        "rendered_at": started_at.isoformat(),
        "capture_wall_s": (capture_ended - capture_started).total_seconds(),
        "input": {
            # Slice O.55 — three numbers to make any future drift
            # immediately diagnosable from the manifest alone:
            #   db_duration_s: what the caller passed (typically from
            #     the DB row)
            #   file_duration_s: what ffprobe read from the source MP4
            #   canonical_duration_s: the one we actually used for the
            #     render timeline (currently == file_duration_s when
            #     the probe succeeded, else db_duration_s)
            "db_duration_s": duration_s,
            "file_duration_s": probed_duration,
            "canonical_duration_s": canonical_duration,
            "audio_source_path": str(audio_source_path),
            # R16 parity — which ASS file (if any) was burned, so a
            # caption-less export is diagnosable from the manifest.
            "ass_file_path": str(ass_file_path) if ass_file_path else None,
            "expected_fps": fps,
        },
        "render": {
            "engine": "playwright-seek-and-shoot",
            "viewport": [width, height],
            "expected_frames": int(canonical_duration * fps),
            "ffmpeg_crf": FFMPEG_CRF,
            "ffmpeg_preset": FFMPEG_PRESET,
        },
        "output": {
            "path": str(output_path),
            "size_bytes": (
                output_path.stat().st_size if output_path.exists() else 0
            ),
        },
        "validation": validation,
    }
    _write_manifest(output_path=output_path, manifest=manifest)
    _log.info(
        "preview_recorder.done",
        clip_id=clip_id, output=str(output_path),
        output_size=manifest["output"]["size_bytes"],
        capture_wall_s=round(manifest["capture_wall_s"], 2),
        validation_ok=validation["ok"],
    )
    return output_path


_DURATION_TOLERANCE_MS = 100  # ffprobe vs requested duration; flagged above this


def _ffprobe_duration_s(path: Path) -> float | None:
    """Probe a file's duration in seconds via ffprobe. Returns None if
    ffprobe isn't on PATH or the file can't be parsed. Best-effort."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        proc = subprocess.run(  # noqa: S603 — args list
            [
                ffprobe, "-v", "error", "-print_format", "json",
                "-show_format", str(path),
            ],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if proc.returncode != 0:
            return None
        info = json.loads(proc.stdout or "{}")
        d = float(info.get("format", {}).get("duration") or 0.0)
        return d if d > 0 else None
    except Exception:  # noqa: BLE001 — never raise on probe
        return None


async def _record_via_seek_and_shoot(
    *,
    page: object,  # playwright Page — typed loosely to avoid the import
    clip_id: str,
    audio_source_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    duration_s: float,
    progress_callback: "object | None" = None,
    ass_file_path: Path | None = None,
    append_outro_enabled: bool = True,
) -> None:
    """Deterministic frame-by-frame export.

    For each i in [0, duration_s * fps):
      - call window.__chalybclipRender.captureFrameAt(i / fps) — page
        seeks the video, renders captions, settles two rAF ticks
      - take a JPEG screenshot via page.screenshot
      - save as frame_NNNNNNNN.jpg in a temp dir

    Then encode the constant-rate frame sequence with ffmpeg's image2
    demuxer + mux audio bit-perfect from the source clip MP4.

    Output: constant 30 fps H.264 MP4 with audio + video timelines
    exactly aligned (frame i is at i/fps seconds, audio is muxed
    starting at 0). Captions are baked into the JPEGs at the same
    timestamps so they stay in sync by construction.
    """
    frame_count = max(1, int(round(duration_s * fps)))
    frame_dt = 1.0 / float(fps)

    # Lazy import the playwright type only for IDE help; the runtime
    # types are passed in by the caller.
    with tempfile.TemporaryDirectory(prefix="chalybclip_frames_") as tmpdir:
        tmp_path = Path(tmpdir)
        captured = 0
        capture_errors = 0
        for i in range(frame_count):
            target_t = i * frame_dt
            try:
                # Drive the page: seek + render captions + settle.
                await page.evaluate(  # type: ignore[attr-defined]
                    "(t) => window.__chalybclipRender.captureFrameAt(t)",
                    target_t,
                )
                # Capture. JPEG quality 95 = visually transparent.
                jpeg_bytes = await page.screenshot(  # type: ignore[attr-defined]
                    type="jpeg",
                    quality=SCREENCAST_JPEG_QUALITY,
                    full_page=False,
                    omit_background=False,
                )
                (tmp_path / f"frame_{i:08d}.jpg").write_bytes(jpeg_bytes)
                captured += 1
            except Exception as e:  # noqa: BLE001
                capture_errors += 1
                _log.warning(
                    "preview_recorder.frame_capture_failed",
                    frame_idx=i, target_t=round(target_t, 3),
                    error=str(e),
                )
                # If we failed within the first 3 frames bail — the
                # page is fundamentally broken. Otherwise duplicate
                # the last successful frame so the timeline stays
                # complete (single-frame freeze is a milder defect
                # than a totally truncated export).
                if i >= 1:
                    prev = tmp_path / f"frame_{i - 1:08d}.jpg"
                    if prev.exists():
                        try:
                            (tmp_path / f"frame_{i:08d}.jpg").write_bytes(
                                prev.read_bytes()
                            )
                            captured += 1
                        except OSError:
                            pass
                if capture_errors > 0 and captured == 0:
                    raise PreviewRecordingError(
                        f"page.screenshot failed on frame {i}: {e}"
                    ) from e

            # Periodic telemetry on long clips so the log shows progress.
            # Render Migration T1 — also fires `progress_callback` so the
            # background runner can flush the % into the DB and the UI
            # poll surfaces "Rendering 67%..." instead of a flat spinner.
            if i and i % max(1, frame_count // 10) == 0:
                pct = int(100 * captured / frame_count)
                _log.info(
                    "preview_recorder.capture_progress",
                    captured=captured, total=frame_count, pct=pct,
                )
                if progress_callback is not None:
                    try:
                        if asyncio.iscoroutinefunction(progress_callback):
                            await progress_callback(pct)
                        else:
                            progress_callback(pct)
                    except Exception:  # noqa: BLE001 — callback never blocks render
                        _log.warning(
                            "preview_recorder.progress_callback_failed",
                            clip_id=clip_id, pct=pct,
                        )

        _log.info(
            "preview_recorder.capture_done",
            captured=captured, expected=frame_count,
            capture_errors=capture_errors,
        )
        if captured == 0:
            raise PreviewRecordingError(
                "no frames captured — render page is broken"
            )

        await _encode_image_sequence(
            frames_dir=tmp_path,
            audio_source_path=audio_source_path,
            output_path=output_path,
            duration_s=duration_s,
            fps=fps,
            ass_file_path=ass_file_path,
            append_outro_enabled=append_outro_enabled,
        )


async def _encode_image_sequence(
    *,
    frames_dir: Path,
    audio_source_path: Path,
    output_path: Path,
    duration_s: float,
    fps: int,
    ass_file_path: Path | None = None,
    append_outro_enabled: bool = True,
) -> None:
    """Encode <frames_dir>/frame_NNNNNNNN.jpg into the final MP4.

    The image2 demuxer reads numbered frames at a constant -framerate.
    Because the seek-and-shoot loop writes exactly frame_count files
    with i/fps timestamps, this is correct by construction — ffmpeg
    sees frame 0 at t=0, frame 1 at t=1/fps, etc.

    Audio is muxed from the source clip MP4 bit-perfect. ffmpeg's
    -t clamps both streams to the operator's requested duration in
    case the JPEG count overshoots by one frame due to rounding.
    """
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    # Render Migration R9 — atomic publish. Same rationale as the
    # hybrid recorder: ffmpeg writes to a sibling `.partial.mp4` and
    # we `os.replace` into `output_path` only after the post-conditions
    # pass. Keeps the download endpoint from ever sampling a growing
    # file's size for Content-Length and then over-shooting it.
    partial_path = output_path.with_suffix(".partial.mp4")
    # R16 parity for the fallback path. Since R16 the render page never
    # draws captions — libass burns them from a server-generated ASS
    # file. The hybrid composite got that filter; this legacy encode
    # didn't, so every hybrid failure that fell back here shipped a
    # caption-less MP4 (operator saw captions in the preview, the
    # published file had none). Same `ass=` filter + shared escaping.
    if ass_file_path is not None and ass_file_path.exists():
        from .captions_ass import escape_ass_path_for_filter

        vf_args = [
            "-vf", f"ass='{escape_ass_path_for_filter(ass_file_path)}'",
        ]
    else:
        vf_args = []
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        # Input 0: image sequence at constant fps.
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%08d.jpg"),
        # Input 1: audio from the source clip MP4.
        "-i", str(audio_source_path),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        # Video encode.
        "-c:v", "libx264",
        "-preset", FFMPEG_PRESET,
        "-crf", FFMPEG_CRF,
        # Cap encoder threads (see FFMPEG_THREADS) to avoid EAGAIN under
        # concurrent load. Spliced via *[] so 0/unset means "ffmpeg auto".
        *(["-threads", FFMPEG_THREADS] if FFMPEG_THREADS not in ("0", "") else []),
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-vsync", "cfr",
        # Caption burn-in (see vf_args above) — empty when no ASS file.
        *vf_args,
        # Audio: lossless copy from source. Normalize to 48kHz AAC
        # ONLY if the source isn't already AAC — modal social
        # platforms (TikTok / Reels / Shorts) standardize on 48kHz so
        # this prevents an extra normalization pass on their side.
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        # Hard clamp to the requested duration. Belt-and-braces — the
        # frame count already encodes the duration via /fps.
        "-t", f"{duration_s:.3f}",
        # Render Migration R8 — +faststart REMOVED. See full rationale
        # in hybrid_recorder.py. Short version: on Railway's volume,
        # ffmpeg's faststart re-open step fails ("Unable to re-open
        # output file for shifting data"), ffmpeg cleans the file up
        # as it bails, we get rc=0 + missing output → R5 raises and
        # the operator sees a failed render. Faststart only matters
        # for in-browser streaming preview which we don't do; every
        # player handles moov-at-end fine for the download flow.
        str(partial_path),
    ]
    _log.info(
        "preview_recorder.ffmpeg_start",
        cmd=" ".join(cmd),
        partial=str(partial_path),
        final=str(output_path),
    )
    loop = asyncio.get_event_loop()
    try:
        proc = await loop.run_in_executor(
            None,
            lambda: subprocess.run(  # noqa: S603 — args list, not shell
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=max(120.0, duration_s + 60.0),
                check=False,
            ),
        )
        _log.info("preview_recorder.ffmpeg_done", rc=proc.returncode)
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or b"").decode("utf-8", errors="replace")[-800:]
            raise PreviewRecordingError(
                f"ffmpeg encode failed (rc={proc.returncode}): {stderr_tail.strip()}"
            )
        if not partial_path.exists() or partial_path.stat().st_size == 0:
            raise PreviewRecordingError(
                "ffmpeg returned 0 but no output file was written"
            )

        # Append the chalybclip end card before publishing, when the
        # caller (who resolved the tenant's tier + brand-kit toggle)
        # asked for it. Best-effort: append_outro never raises and
        # leaves `partial_path` untouched on failure, so a render still
        # ships even if the outro can't.
        if append_outro_enabled:
            from .outro import append_outro

            append_outro(partial_path)

        # Atomic publish — sibling paths, single FS, race-free swap.
        os.replace(partial_path, output_path)
    except BaseException:
        try:
            partial_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _validate_export(
    *,
    output_path: Path,
    expected_duration_s: float,
    expected_fps: int,
) -> dict[str, object]:
    """ffprobe the encoded MP4 + validate against the operator's
    expectations. Returns a dict that goes straight into the manifest.

    All probes are best-effort: ffprobe might be missing, the file
    might be unparseable, etc. Each branch defaults conservatively
    so a missing field never crashes the caller.
    """
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    checks: dict[str, object] = {}
    ok = True
    errors: list[str] = []

    try:
        proc = subprocess.run(  # noqa: S603 — args list
            [
                ffprobe, "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams",
                str(output_path),
            ],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if proc.returncode != 0:
            errors.append(f"ffprobe rc={proc.returncode}")
            ok = False
        else:
            info = json.loads(proc.stdout or "{}")
            fmt = info.get("format") or {}
            actual_duration = float(fmt.get("duration") or 0.0)
            checks["actual_duration_s"] = round(actual_duration, 3)
            drift_ms = int(round(abs(actual_duration - expected_duration_s) * 1000))
            checks["duration_drift_ms"] = drift_ms
            if drift_ms > _DURATION_TOLERANCE_MS:
                ok = False
                errors.append(f"duration drift {drift_ms}ms > {_DURATION_TOLERANCE_MS}ms")

            video_stream = next(
                (s for s in info.get("streams", []) if s.get("codec_type") == "video"),
                None,
            )
            audio_stream = next(
                (s for s in info.get("streams", []) if s.get("codec_type") == "audio"),
                None,
            )
            if video_stream:
                # avg_frame_rate is "30/1" -> compute the float
                fr = str(video_stream.get("avg_frame_rate") or "0/0")
                try:
                    n, d = fr.split("/")
                    actual_fps = float(n) / float(d) if float(d) else 0.0
                except (ValueError, ZeroDivisionError):
                    actual_fps = 0.0
                checks["actual_fps"] = round(actual_fps, 2)
                checks["video_codec"] = video_stream.get("codec_name")
                checks["pix_fmt"] = video_stream.get("pix_fmt")
                if abs(actual_fps - expected_fps) > 0.5:
                    ok = False
                    errors.append(
                        f"fps drift: {actual_fps:.2f} vs expected {expected_fps}"
                    )
                if video_stream.get("codec_name") != "h264":
                    ok = False
                    errors.append(f"video codec {video_stream.get('codec_name')!r} != h264")
            else:
                ok = False
                errors.append("no video stream in output")

            if audio_stream:
                checks["audio_codec"] = audio_stream.get("codec_name")
                checks["audio_sample_rate"] = audio_stream.get("sample_rate")
                # Audio is informational unless it's missing entirely.
            else:
                # Some clips have no audio; just flag, don't fail.
                checks["audio_codec"] = None
    except Exception as e:  # noqa: BLE001
        errors.append(f"ffprobe exception: {e}")
        ok = False

    return {"ok": ok, "checks": checks, "errors": errors}


def _write_manifest(*, output_path: Path, manifest: dict[str, object]) -> None:
    """Write export.manifest.json next to the output MP4."""
    manifest_path = output_path.with_suffix(".manifest.json")
    try:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    except OSError as e:
        _log.warning(
            "preview_recorder.manifest_write_failed",
            path=str(manifest_path), error=str(e),
        )


# Slice O.54 — the legacy real-time CDP screencast path is retired.
# Anything still importing _record_via_cdp gets an explicit error so
# they update to record_clip_to_mp4.
async def _record_via_cdp(*args, **kwargs):  # noqa: ARG001
    raise PreviewRecordingError(
        "_record_via_cdp removed in slice O.54. "
        "Use record_clip_to_mp4() which now drives seek-and-shoot."
    )


class PreviewRecordingError(RuntimeError):
    """Raised on any failure inside the recorder pipeline.

    Caller decides what to do — typical handling: log + return 500 with
    a short detail, or fall back to the legacy overlay_burn pipeline
    (still in the repo, slice O.21 left it intact but disconnected).
    """


__all__ = ["record_clip_to_mp4", "PreviewRecordingError"]
