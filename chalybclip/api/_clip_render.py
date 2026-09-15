"""Background clip-render runner — Render Migration T1.

The legacy download path awaited `record_clip_to_mp4` inline inside
the HTTP request handler. On Railway a 35s clip rendered for ~4.7
min (seek-and-shoot at ~3.8 fps for 1077 frames), well past the
request timeout. Operators saw a permanent "Preparing…" spinner;
the actual render had been SIGKILL'd by the proxy.

This module is the background path. The download endpoint now:

  1. If the rendered MP4 is on disk → serve it (cache hit).
  2. Else flip the clip row to render_state='rendering' atomically
     (the SQL guards against double-dispatch).
  3. Schedule `render_clip_in_background(...)` as a FastAPI
     BackgroundTask.
  4. Return JSON status so the UI's polling Download button can
     show progress + swap to "Download MP4" when ready.

The runner itself:

  1. Calls record_clip_to_mp4 with a `progress_callback` that
     writes back to clips.render_progress_pct so the UI doesn't
     need to read events directly.
  2. On success: marks state='ready', progress=100.
  3. On failure: marks state='failed' with the truncated error.

Failures are absorbed here — we never propagate into FastAPI's
BackgroundTask runner because that just logs a traceback and the
operator sees nothing. The UI's status poll reads
render_state='failed' + render_error and surfaces a clear retry
button.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from chalybclip.api._render_validation import validate_rendered_mp4

if TYPE_CHECKING:
    pass

_log = logging.getLogger("chalybclip.clip_render")

# How often the background task is allowed to write progress updates
# back to the DB. Capping at 1/s prevents a 30-fps capture from
# hammering SQLite with thousands of UPDATE statements.
_PROGRESS_UPDATE_DEBOUNCE_S = 1.0


async def _build_ass_for_clip(
    *,
    db: object,
    tenant_id: str,
    clip_id: str,
    output_path: Path,
    width: int,
    height: int,
) -> Path | None:
    """Generate an ASS subtitle file for the clip's captions.

    Reads the transcript via TranscriptsRepo, slices to the clip's
    window via `captions_for_clip` (same code path the editor's
    /captions.json endpoint uses, so the burn matches the preview
    byte-for-byte), then writes the ASS file sibling to the cache
    MP4 as `clip_render_<res>.ass`. Returns the path on success,
    None when there's nothing to burn (no transcript / disabled /
    empty result) — caller treats None as "skip the libass step".

    Pulls captions config off the clip's overlay_config so a
    clip-level disable, position pick, or highlight-color override
    flows through. Defaults match the operator-visible editor
    preview when no per-clip override is set.
    """
    from chalybclip.clip.captions_ass import generate_ass
    from chalybclip.clip import captions_for_clip, lines_to_json
    from chalybclip.db import ClipsRepo, TranscriptsRepo
    from chalybclip.tenancy import bound_tenant

    with bound_tenant(tenant_id):
        clip = await ClipsRepo(db).get(clip_id)
        if clip is None:
            return None
        ov = clip.overlay_config or {}
        captions_cfg = ov.get("captions") if isinstance(ov.get("captions"), dict) else {}
        if not captions_cfg.get("enabled", True):
            return None

        transcript = await TranscriptsRepo(db).get(clip.stream_id)
        if transcript is None:
            return None

        lines = captions_for_clip(
            transcript.segments_json or "",
            clip_start_s=clip.start_s,
            clip_end_s=clip.end_s,
        )
        if not lines:
            return None

        # captions_for_clip returns Pydantic / dataclass objects; the
        # ASS generator needs plain dict. lines_to_json gives us the
        # exact serializable shape the /captions.json endpoint exposes
        # — same shape the editor renders from, so the ASS burn and
        # the editor preview stay in lockstep.
        lines_json = lines_to_json(lines)

    highlight = captions_cfg.get("highlight_color") or "#ffd84a"
    position = captions_cfg.get("position") or "lower_third"
    font_size_key = captions_cfg.get("font_size") or "medium"
    # Match the CSS scale * the base size. CSS base is 4.231cqw on a
    # 1080-px-wide container = ~46 px. Scale to absolute pixels and
    # scale by the operator's size pick.
    base_size = int(width * 0.0423)
    scale = {"small": 0.78, "medium": 1.0, "large": 1.24, "xl": 1.5}.get(
        font_size_key, 1.0,
    )
    font_size = max(16, int(base_size * scale))

    ass_path = output_path.with_suffix(".ass")
    generate_ass(
        lines=lines_json,
        width=width,
        height=height,
        output_path=ass_path,
        highlight_color_hex=highlight,
        position=position,
        font_size=font_size,
    )
    _log.info(
        "render_clip.ass_built clip=%s lines=%d ass=%s "
        "highlight=%s position=%s font_size=%d",
        clip_id, len(lines_json), ass_path, highlight, position, font_size,
    )
    return ass_path


def _safe_unlink(path: Path) -> None:
    """Best-effort delete used by the failure paths so a partial /
    corrupt file is never left for the next download click to serve
    as a cache hit. We swallow OSError because the only thing worse
    than a partial file is failing the failure path."""
    try:
        path.unlink(missing_ok=True)
    except OSError as e:  # noqa: BLE001 — cleanup never raises
        _log.warning(
            "render_clip.unlink_failed path=%s error=%s", path, e,
        )


async def render_clip_in_background(
    *,
    clip_id: str,
    tenant_id: str,
    duration_s: float,
    audio_source_path: Path,
    output_path: Path,
    base_url: str,
    auth_cookie_value: str | None,
    width: int,
    height: int,
    db_path: str,
    auth_query: str = "",
) -> None:
    """Run the headless-Chromium recorder and update clip.render_state.

    This is the function the download endpoint schedules via
    BackgroundTasks. It opens its own DB connection (the request's
    connection closes as soon as the response is sent) so the state
    updates land even after the HTTP layer is gone.

    Failure surfacing: any exception here is caught + recorded on the
    clip row as render_state='failed' + render_error. The UI poll
    surfaces it. We never re-raise — FastAPI's BackgroundTask runner
    would just log a traceback and the operator would see a stuck
    spinner.
    """
    from chalybclip.clip.preview_recorder import (
        PreviewRecordingError,
        record_clip_to_mp4,
    )
    from chalybclip.db import ClipsRepo, Database
    from chalybclip.resources import heavy_slot
    from chalybclip.tenancy import bound_tenant

    db = Database(db_path)
    await db.connect()
    # Render Migration R5 — outer guard. ANY unhandled exception below
    # leaves state stuck at 'rendering' → operator's UI polls forever
    # with no way to retry. Wrap the whole body so a stray crash (in
    # validation, mark_render_ready DB write, anywhere) still ends up
    # at mark_render_failed before this coroutine returns.
    crashed_with: BaseException | None = None
    try:
        # Throttled progress callback. The recorder calls this many
        # times per second; we debounce to 1 Hz so SQLite writes don't
        # contend with everything else.
        last_pct_written: dict = {"pct": -1, "ts": 0.0}

        async def _on_progress(pct: int) -> None:
            import time as _time
            now = _time.monotonic()
            if (
                pct == last_pct_written["pct"]
                or now - last_pct_written["ts"] < _PROGRESS_UPDATE_DEBOUNCE_S
            ):
                return
            last_pct_written["pct"] = pct
            last_pct_written["ts"] = now
            try:
                with bound_tenant(tenant_id):
                    await ClipsRepo(db).update_render_progress(
                        clip_id, pct=pct,
                    )
            except Exception:  # noqa: BLE001 — observability never blocks
                _log.warning(
                    "render_clip.progress_write_failed clip=%s pct=%s",
                    clip_id, pct,
                )

        # Render Migration T2 — try the hybrid ffmpeg + overlay-alpha
        # path first (~10-30s for a typical 35s clip). On any failure
        # we fall back to the legacy seek-and-shoot recorder for one
        # retry (~5min) — the operator still gets their MP4, just
        # slower. The fallback path is the same code T1 backgrounded;
        # the failure-recovery semantics are unchanged.
        from chalybclip.clip.hybrid_recorder import (
            HybridRecordingError, record_clip_hybrid,
        )

        # Render Migration R16 — generate the ASS subtitle file once,
        # pass it to whichever recorder runs (hybrid first, legacy
        # fallback). The page no longer renders captions in any form;
        # libass burns them after the alpha composite. If captions
        # are disabled on this clip, or the transcript has no
        # word-level data overlapping the window, we pass None and
        # the composite skips the ass=filter chain.
        ass_file_path: Path | None = None
        try:
            ass_file_path = await _build_ass_for_clip(
                db=db,
                tenant_id=tenant_id,
                clip_id=clip_id,
                output_path=output_path,
                width=width,
                height=height,
            )
        except Exception as e:  # noqa: BLE001 — ASS is enrichment, not gating
            _log.warning(
                "render_clip.ass_build_failed clip=%s error=%s",
                clip_id, str(e)[:300],
            )

        # Resolve the end-card outro decision ONCE and pass it to
        # whichever recorder runs. Free tier always gets it; paid tiers
        # respect their brand kit's show_chalybclip_outro toggle. Failure
        # defaults to ON (free-like) so a lookup hiccup never silently
        # strips the credit.
        append_outro_enabled = True
        try:
            from chalybclip.branding import outro_enabled_for_clip
            with bound_tenant(tenant_id):
                _clip = await ClipsRepo(db).get(clip_id)
            append_outro_enabled = await outro_enabled_for_clip(
                db,
                tenant_id=tenant_id,
                stream_id=(_clip.stream_id if _clip else ""),
            )
        except Exception:  # noqa: BLE001 — best-effort; default to on
            append_outro_enabled = True

        hybrid_failed_with: str | None = None
        try:
            # Process-wide governor — a headless-Chromium render plus its
            # ffmpeg composite is the single heaviest op in the process.
            # Acquiring a slot keeps it from peaking alongside the cut
            # fan-out / poll pipeline and tipping the host into EAGAIN.
            async with heavy_slot():
                await record_clip_hybrid(
                    clip_id=clip_id,
                    duration_s=duration_s,
                    audio_source_path=audio_source_path,
                    output_path=output_path,
                    base_url=base_url,
                    auth_cookie_value=auth_cookie_value,
                    auth_query=auth_query,
                    width=width,
                    height=height,
                    progress_callback=_on_progress,
                    ass_file_path=ass_file_path,
                    append_outro_enabled=append_outro_enabled,
                )
        except HybridRecordingError as e:
            hybrid_failed_with = str(e)
            _log.warning(
                "render_clip.hybrid_failed_falling_back clip=%s error=%s",
                clip_id, str(e)[:300],
            )
        except Exception as e:  # noqa: BLE001
            # Render Migration R5 — catch ANY exception the hybrid
            # recorder leaked. Previously only HybridRecordingError
            # was caught, so a FileNotFoundError (Railway's silent
            # ffmpeg disk-full failure landed on `output_path.stat()`
            # in the success-log line) propagated all the way out
            # of render_clip_in_background → state stuck at
            # 'rendering' → UI polled /render-status forever with
            # no way to retry. Treat any non-HybridRecordingError
            # exception as a hybrid failure → fall through to the
            # legacy seek-and-shoot path, same as if it had raised
            # HybridRecordingError properly.
            hybrid_failed_with = f"{type(e).__name__}: {e}"
            _log.exception(
                "render_clip.hybrid_unexpected_exception_falling_back "
                "clip=%s",
                clip_id,
            )
            # The hybrid path may have left a partial composite file
            # on disk; wipe before legacy gets a chance to skip-because-
            # exists or otherwise interact with stale bytes.
            _safe_unlink(output_path)

        if hybrid_failed_with is not None:
            # Reset progress so the legacy path's 0-100% animation
            # doesn't read as "going backwards" after the hybrid bailed.
            try:
                with bound_tenant(tenant_id):
                    await ClipsRepo(db).update_render_progress(
                        clip_id, pct=0,
                    )
            except Exception:  # noqa: BLE001
                pass

        try:
            if hybrid_failed_with is None:
                # Hybrid succeeded — output is on disk, skip legacy.
                pass
            else:
                async with heavy_slot():
                    await record_clip_to_mp4(
                        clip_id=clip_id,
                        duration_s=duration_s,
                        audio_source_path=audio_source_path,
                        output_path=output_path,
                        base_url=base_url,
                        auth_cookie_value=auth_cookie_value,
                        auth_query=auth_query,
                        width=width,
                        height=height,
                        progress_callback=_on_progress,
                        # R16 parity — since the render page stopped
                        # drawing captions, the legacy fallback must
                        # burn the same ASS file the hybrid path got;
                        # without it a hybrid failure shipped a
                        # caption-less MP4.
                        ass_file_path=ass_file_path,
                        append_outro_enabled=append_outro_enabled,
                    )
        except PreviewRecordingError as e:
            # Render Migration R2 — wipe any partial bytes the recorder
            # left on disk BEFORE marking failed. Otherwise the download
            # endpoint's rendered.exists() branch serves the fragment
            # ahead of the 409 + retry hint.
            _safe_unlink(output_path)
            with bound_tenant(tenant_id):
                await ClipsRepo(db).mark_render_failed(
                    clip_id, error=f"recorder: {e}",
                )
            _log.warning(
                "render_clip.failed clip=%s error=%s", clip_id, str(e)[:300],
            )
            return
        except Exception as e:  # noqa: BLE001 — catch-all so state lands
            _safe_unlink(output_path)
            with bound_tenant(tenant_id):
                await ClipsRepo(db).mark_render_failed(
                    clip_id, error=f"{type(e).__name__}: {e}",
                )
            _log.exception(
                "render_clip.crashed clip=%s", clip_id,
            )
            return

        # Render Migration R1 — validate the produced MP4 BEFORE
        # marking ready. ffmpeg can return exit 0 and still emit a
        # malformed container (missing moov atom, broken mux, weird
        # timestamps). The download path served those as cache hits
        # → operator got a file Windows refused to open
        # (0xC00D36C4). Validation now gates the ready flip: a file
        # that doesn't parse / has no video / has no audio / is
        # truncated → gets unlinked + the render is marked failed
        # with a clear reason. Partial bytes never survive into the
        # download endpoint's cache-hit branch.
        ok, reason = validate_rendered_mp4(
            output_path,
            expected_duration_s=duration_s,
        )
        if not ok:
            _safe_unlink(output_path)
            with bound_tenant(tenant_id):
                await ClipsRepo(db).mark_render_failed(
                    clip_id,
                    error=f"output failed validation: {reason}",
                )
            _log.warning(
                "render_clip.validation_failed clip=%s reason=%s",
                clip_id, (reason or "")[:200],
            )
            return

        with bound_tenant(tenant_id):
            await ClipsRepo(db).mark_render_ready(clip_id)
        _log.info(
            "render_clip.ready clip=%s output=%s renderer=%s "
            "bytes=%d duration_s=%.2f",
            clip_id,
            output_path,
            "legacy" if hybrid_failed_with else "hybrid",
            output_path.stat().st_size if output_path.exists() else 0,
            duration_s,
        )
    except BaseException as e:  # noqa: BLE001
        # R5 outer guard. We caught nested exceptions for each step
        # above, but ANY uncaught path (validation crash, DB write
        # crash, import error, etc.) lands here so state always
        # exits 'rendering' before the coroutine returns. We swallow
        # rather than re-raise — FastAPI's BackgroundTask runner
        # would log + drop, leaving the operator with a stuck
        # spinner and no error message. Only re-raise for
        # KeyboardInterrupt / SystemExit / asyncio.CancelledError
        # so the worker can still shut down cleanly.
        crashed_with = e
        import asyncio as _asyncio
        if isinstance(e, (KeyboardInterrupt, SystemExit, _asyncio.CancelledError)):
            raise
        _safe_unlink(output_path)
        try:
            with bound_tenant(tenant_id):
                await ClipsRepo(db).mark_render_failed(
                    clip_id,
                    error=f"runner crashed: {type(e).__name__}: {e}",
                )
        except BaseException:  # noqa: BLE001 — even the DB write may fail
            _log.exception(
                "render_clip.outer_guard_mark_failed_crashed clip=%s",
                clip_id,
            )
        _log.exception(
            "render_clip.outer_guard_caught clip=%s", clip_id,
        )
    finally:
        await db.close()
        if crashed_with is not None:
            _log.warning(
                "render_clip.finished_via_outer_guard clip=%s",
                clip_id,
            )


async def ensure_clip_rendered(
    *,
    db: object,
    clip: object,
    tenant_id: str,
    base_url: str,
    auth_cookie_value: str | None,
    db_path: str,
    width: int = 1080,
    height: int = 1920,
    output_name: str = "clip_render_1080.mp4",
    max_wait_s: float = 240.0,
    auth_query: str = "",
) -> Path:
    """Guarantee the burned-in publish-preset MP4 exists, then return it.

    This is the shared guarantee behind "publish the same video you
    download". The download endpoint already lazy-renders this exact
    file (overlays + libass captions burned in via the headless
    recorder); the publish path calls this so Zernio fetches that same
    rendered file rather than the raw, unedited source.

    Behaviour:
      1. Valid cached MP4 on disk → return it (the common case once
         approve pre-renders).
      2. A background render already in flight (e.g. from approve) →
         wait it out (up to `max_wait_s`) instead of starting a
         duplicate recorder against the same output path.
      3. Otherwise render synchronously, inline, using the operator's
         session cookie (the /render page is auth-gated).

    Raises RuntimeError when a servable render can't be produced — the
    caller turns that into an operator-readable error. We NEVER fall
    back to the raw source here; shipping the unedited clip is exactly
    the bug this function exists to close.
    """
    from chalybclip.api._render_validation import is_servable_cached_mp4
    from chalybclip.db import ClipsRepo
    from chalybclip.tenancy import bound_tenant

    clip_id = getattr(clip, "id")
    original = Path(getattr(clip, "path"))
    output = original.parent / output_name

    # 1. Warm cache — instant once approve has pre-rendered.
    if is_servable_cached_mp4(output):
        return output

    repo = ClipsRepo(db)

    # 2. Wait out an in-flight background render (approve schedules one)
    #    rather than racing a second recorder against the same file.
    waited = 0.0
    while waited < max_wait_s:
        with bound_tenant(tenant_id):
            fresh = await repo.get(clip_id)
        state = getattr(fresh, "render_state", "idle") if fresh else "idle"
        if is_servable_cached_mp4(output):
            return output
        if state != "rendering":
            break
        await asyncio.sleep(1.0)
        waited += 1.0

    if is_servable_cached_mp4(output):
        return output

    # 3. Cold (idle / failed / wait exhausted) — render now, inline.
    #    mark_render_started is atomic on (state != 'rendering') so a
    #    concurrent publish doesn't double-dispatch.
    if not original.exists():
        # Phase 2a — the local copy may have been reclaimed (retention,
        # or the pipeline ran on a worker). The render needs real local
        # bytes; pull the bucket copy back before failing the publish.
        from chalybclip.api._artifacts import rehydrate_clip

        if not await rehydrate_clip(
            tenant_id=tenant_id, clip_id=clip_id, clip_path=original
        ):
            raise RuntimeError(
                f"source clip file missing from disk: {original}"
            )
    with bound_tenant(tenant_id):
        await repo.mark_render_started(clip_id)
    await render_clip_in_background(
        clip_id=clip_id,
        tenant_id=tenant_id,
        duration_s=float(getattr(clip, "duration_s")),
        audio_source_path=original,
        output_path=output,
        base_url=base_url,
        auth_cookie_value=auth_cookie_value,
        width=width,
        height=height,
        db_path=db_path,
        auth_query=auth_query,
    )
    if is_servable_cached_mp4(output):
        return output

    # render_clip_in_background absorbs its own failures onto the clip
    # row; surface the recorded reason if we have one.
    with bound_tenant(tenant_id):
        fresh = await repo.get(clip_id)
    reason = getattr(fresh, "render_error", None) if fresh else None
    raise RuntimeError(reason or "clip render did not produce a servable MP4")


__all__ = ["render_clip_in_background", "ensure_clip_rendered"]
