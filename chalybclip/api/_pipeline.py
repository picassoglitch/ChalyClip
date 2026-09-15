"""PipelineKickoff bundle + default runner — re-export shim.

The canonical home for these types is `chalybclip.jobs` (slice F.8).
Existing imports `from chalybclip.api._pipeline import PipelineKickoff`
keep working via the re-exports below; new code should import from
`chalybclip.jobs` directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from chalybclip.jobs import PipelineKickoff, PipelineRunner
from chalybclip.settings import resolve_db_target

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from chalybclip.db import Database

_log = logging.getLogger("chalybclip.pipeline.runner")


async def default_pipeline_runner(kickoff: PipelineKickoff) -> None:
    """Real `process_vod` invocation, lazily imported so tests that don't
    kick off the pipeline don't pay for Whisper / CV / yt-dlp module-load
    cost.

    Forwards the configured `db_path` so `_step()` can write
    pipeline.step.{start,done,failed} rows to the events table — without
    this, the dashboard's progress card has no events to read and stays
    permanently stuck on 'all six steps pending'.

    ERROR SURFACING (slice NX.5):
        process_vod raises in two distinct ways:
          (a) Inside a `_step(...)` context — that context manager already
              emits `pipeline.step.failed` with the error. The progress
              endpoint reads those events and renders the step in red.
          (b) BEFORE entering any step (e.g. persona-not-found at
              pipeline.py:316, db_path bad, etc). No event gets emitted,
              the progress endpoint sees all-pending, the UI spins
              forever, the user has no idea what happened.

        We catch (b) here and write a `pipeline.failed` top-level event
        scoped to the stream_id. The progress endpoint flags this as a
        terminal failure even when no step has been started. Then we
        re-raise so FastAPI's BackgroundTasks logs the full traceback
        for the operator.
    """
    from chalybclip.jobs.active import pipeline_active
    from chalybclip.pipeline import process_vod
    from chalybclip.settings import get_settings

    db_path = resolve_db_target(get_settings())

    try:
        # Registered as in-flight so the disk reclaimers (watchdog,
        # ingest preflight, sweep backstop) never delete this stream's
        # source mid-run — long transcribe/LLM stretches only READ it,
        # so mtime-based activity checks can't see the run.
        with pipeline_active(kickoff.stream.id):
            await process_vod(
                tenant_id=kickoff.tenant_id,
                vod_url=kickoff.stream.vod_url,
                output_dir=kickoff.output_dir,
                persona_id=kickoff.persona_id,
                stream_id=kickoff.stream.id,
                language=kickoff.language,
                db_path=db_path,
            )
    except Exception as e:
        # Best-effort: emit a top-level failure event so the dashboard's
        # progress card surfaces the error instead of spinning. We catch
        # broadly because we can't anticipate every internal exception type
        # — VariantError, IngestError, ValidationError, RuntimeError,
        # KeyError on bad config, etc.
        try:
            await _emit_top_level_failure(
                db_path=db_path,
                tenant_id=kickoff.tenant_id,
                stream_id=kickoff.stream.id,
                error=e,
            )
        except Exception:
            # If even the event write fails, fall through to re-raise. The
            # original error matters more than the dashboard UX.
            _log.exception(
                "pipeline.failed event write failed for stream=%s",
                kickoff.stream.id,
            )
        # Reclaim the heavy source VOD on FAILURE too — not just on success.
        # The source MP4 + audio is 5-50 GB per multi-hour stream. Without
        # this, a failed long-VOD ingest sits on disk until the 7-day
        # retention sweep; a burst of failures (e.g. expired cookies, a slow
        # disk) fills the volume and every later download then truncates /
        # no-spaces — the self-reinforcing disk-exhaustion spiral behind a
        # wall of "failed" streams. A retry re-downloads from the row, so
        # dropping the source here is safe. Gated on the same flag as the
        # success-path cleanup; cleanup failure must never mask the original.
        if get_settings().delete_source_on_completion:
            try:
                from chalybclip.retention import reclaim_stream_source

                freed = reclaim_stream_source(
                    stream_dir=kickoff.output_dir / kickoff.stream.id,
                    source_video_path=kickoff.stream.source_video_path,
                    source_audio_path=kickoff.stream.source_audio_path,
                )
                _log.info(
                    "pipeline.failed.source_reclaimed stream=%s bytes_freed=%d",
                    kickoff.stream.id, freed,
                )
            except Exception:
                _log.exception(
                    "source reclaim after failure failed for stream=%s",
                    kickoff.stream.id,
                )
        # Always re-raise so FastAPI logs the full traceback for the
        # operator to debug from Railway logs.
        raise

    # Token T2 — run succeeded (a failure re-raises above). Charge the
    # per-run base fee + force a live Chalyb balance refresh so the token
    # chip reflects what this run just consumed without the operator
    # clicking it.
    await _refresh_balance_after_run(
        db_path=db_path,
        tenant_id=kickoff.tenant_id,
        stream_id=kickoff.stream.id,
    )


# Grace delay before the post-run balance pull. The pipeline's final LLM /
# transcription usage reports are fire-and-forget (3s timeout each); this
# lets the last ones land at Chalyb so the fetched balance reflects the
# full run, not a mid-deduction snapshot.
_BALANCE_REFRESH_GRACE_S = 4.0


async def _refresh_balance_after_run(
    *, db_path: str, tenant_id: str, stream_id: str | None = None,
) -> None:
    """Token T2/T3 — after a successful run: (1) charge the per-run base
    fee, (2) pull the live Chalyb balance into the cache so the chip is
    fresh without a manual click.

    Best-effort: the base charge is awaited (so it lands before we read
    the balance), then a short grace delay lets the run's fire-and-forget
    provider reports land, then ONE live fetch. Every error is swallowed —
    the run already succeeded and the 30s chip poll is the fallback."""
    import asyncio

    from chalybclip.db import Database
    from chalybclip.integrations.chalyb.balance import fetch_balance_now

    try:
        db = Database(db_path)
        await db.connect()
        try:
            # (1) Per-run base charge — covers server/render/storage
            #     overhead so a near-free-API run still draws down quota.
            if stream_id:
                await _charge_run_base_fee(db, tenant_id, stream_id)
            # (2) Let the run's fire-and-forget provider reports land.
            await asyncio.sleep(_BALANCE_REFRESH_GRACE_S)
            # (3) Live balance fetch → cache → chip.
            await fetch_balance_now(db, tenant_id=tenant_id)
        finally:
            await db.close()
    except Exception:  # noqa: BLE001 — observability, never affects the run
        _log.warning(
            "post-run balance refresh failed for tenant=%s", tenant_id,
        )


async def _charge_run_base_fee(
    db: "Database", tenant_id: str, stream_id: str
) -> None:
    """Token T3 — report the per-run base charge (engine.base) to Chalyb.

    A flat fee covering server time / render / storage that the raw API
    cost doesn't capture. Idempotent: source_id is base_<stream_id>, so a
    re-run of the same stream is a no-op upstream. Amount is configurable
    via CHALYBCLIP_PIPELINE_BASE_CHARGE_USD_MICROS (0 disables). Awaited (not
    fire-and-forget) so the balance fetch right after reflects it."""
    import datetime as _dt

    from chalybclip.integrations.chalyb.reporter import report_usage
    from chalybclip.settings import get_settings

    base = int(getattr(get_settings(), "pipeline_base_charge_usd_micros", 0) or 0)
    if base <= 0:
        return
    try:
        await report_usage(
            db,
            tenant_id=tenant_id,
            kind="engine.base",
            amount=1,
            cost_usd_micros=base,
            source_id=f"base_{stream_id}",
            occurred_at_iso=_dt.datetime.now(_dt.UTC).isoformat(),
            provider="chalybclip",
            operation="pipeline_run",
        )
    except Exception:  # noqa: BLE001 — never affects the run
        _log.warning(
            "base-charge report failed for tenant=%s stream=%s",
            tenant_id, stream_id,
        )


async def _emit_top_level_failure(
    *,
    db_path: str,
    tenant_id: str,
    stream_id: str,
    error: Exception,
) -> None:
    """Write a `pipeline.failed` event to the tenant's events table.

    Different from `pipeline.step.failed` (which is scoped to one of the six
    pipeline steps): this fires when the runner's outermost try/except
    catches an exception that escaped the per-step contexts entirely.
    The progress endpoint treats this as a terminal pipeline failure even
    if no step is in 'running' state.
    """
    from chalybclip.db import Database
    from chalybclip.events import emit
    from chalybclip.tenancy import bound_tenant

    # process_vod already opens + closes its own DB session; that one is
    # gone by the time we get here. Open a fresh one just for this write.
    db = Database(db_path)
    await db.connect()
    with bound_tenant(tenant_id):
        await emit(
            db,
            "pipeline.failed",
            {
                "stream_id": stream_id,
                "error_type": type(error).__name__,
                # Keep the message short — the full traceback is in the
                # Railway log. The UI only needs enough text to display.
                "error": str(error)[:500],
            },
        )


async def upload_pipeline_runner(
    *,
    tenant_id: str,
    stream_id: str,
    persona_id: str,
    tmp_path: "Path",
    output_dir: "Path",
    title: str | None,
    language: str | None = None,
) -> None:
    """Run the full pipeline for an uploaded file in the background.

    The HTTP upload endpoint stashes the request body to a tempfile
    inline (unavoidable — the bytes have to arrive before we can
    respond) and then schedules THIS runner so the operator gets a
    303 to the live progress page immediately. Audio extraction +
    transcribe + detect + cut all happen here, asynchronously.

    Phases inside this runner:

      1. `ingest_uploaded()` — moves the tempfile to the canonical
         <output_dir>/<stream_id>/source/video.mp4, extracts audio,
         writes stream.json.
      2. UPSERT the StreamRow with the real values now that we know
         the duration + on-disk paths.
      3. `process_vod()` — picks up the cached stream.json (skipping
         ingest_vod's URL path entirely) and runs the rest of the
         pipeline.

    Failure surfacing matches default_pipeline_runner: any exception
    that escapes the per-step contexts gets a `pipeline.failed` event
    so the dashboard's progress card explains the failure instead of
    spinning forever. We then re-raise so FastAPI logs the traceback.
    """
    from chalybclip.db import Database, StreamsRepo
    from chalybclip.db.adapters import stream_to_row
    from chalybclip.ingest import ingest_uploaded
    from chalybclip.pipeline import process_vod
    from chalybclip.settings import get_settings
    from chalybclip.tenancy import bound_tenant

    settings = get_settings()
    db_path = resolve_db_target(settings)

    try:
        # Phase 1 — finish ingest (move file + extract audio + persist
        # stream.json). Idempotent on stream.json: if the operator
        # somehow re-triggers this run, the second call returns the
        # cached Stream without re-extracting.
        stream = await ingest_uploaded(
            tenant_id=tenant_id,
            source_path=tmp_path,
            output_dir=output_dir,
            stream_id=stream_id,
            title=title,
        )

        # Phase 2 — promote the placeholder StreamRow we inserted in
        # the endpoint to the real values (duration_s, title from
        # ffprobe, etc.). The pipeline's StreamsRepo.upsert further
        # along would also do this, but doing it here means the
        # dashboard's progress page shows real metadata as soon as
        # ingest finishes instead of waiting for transcribe to start.
        db = Database(db_path)
        await db.connect()
        try:
            with bound_tenant(tenant_id):
                await StreamsRepo(db).upsert(stream_to_row(stream))
                # upsert is INSERT OR IGNORE; backfill the real length onto
                # a row a live webhook may have created with duration 0.
                await StreamsRepo(db).set_duration(
                    stream.id, float(getattr(stream, "duration_s", 0) or 0)
                )
        finally:
            await db.close()

        # Phase 3 — run the full pipeline. process_vod sees the
        # cached stream.json and skips ingest_vod entirely (the
        # upload:// pseudo-URL never has to hit yt-dlp). Registered as
        # in-flight so the disk reclaimers never eat the source mid-run.
        from chalybclip.jobs.active import pipeline_active

        with pipeline_active(stream.id):
            await process_vod(
                tenant_id=tenant_id,
                vod_url=stream.vod_url,
                output_dir=output_dir,
                persona_id=persona_id,
                stream_id=stream.id,
                language=language,
                db_path=db_path,
            )
    except Exception as e:
        try:
            await _emit_top_level_failure(
                db_path=db_path,
                tenant_id=tenant_id,
                stream_id=stream_id,
                error=e,
            )
        except Exception:
            _log.exception(
                "upload pipeline.failed event write failed for stream=%s",
                stream_id,
            )
        raise

    # Token T2 — run succeeded; charge the base fee + refresh the balance.
    await _refresh_balance_after_run(
        db_path=db_path, tenant_id=tenant_id, stream_id=stream_id,
    )


# ---- Phase L.2 — auto-clip a live stream once it ends ---------------------


def _resolve_recording_file(recording_path: str) -> "Path | None":
    """Locate the finished MediaMTX recording on disk.

    The live webhook stored `source_video_path` as MediaMTX's record
    target (e.g. `/data/live/<stream_id>/source`, possibly without an
    extension). MediaMTX's actual output is deployment-config-dependent,
    so we resolve defensively:
      1. the path as-is (if it already names a file),
      2. `<path>.mp4`,
      3. otherwise the newest non-trivial `*.mp4` in the same directory.

    Returns None if nothing usable is on disk yet. Size floor (>1 KiB)
    rejects a just-created empty file MediaMTX hasn't written to.
    """
    from pathlib import Path as _Path

    p = _Path(str(recording_path))
    candidates: list[_Path] = []
    if p.suffix:
        candidates.append(p)
    else:
        candidates.append(p.with_suffix(".mp4"))
        candidates.append(_Path(str(p) + ".mp4"))
    for c in candidates:
        try:
            if c.is_file() and c.stat().st_size > 1024:
                return c
        except OSError:
            pass
    parent = p.parent
    try:
        if parent.is_dir():
            mp4s = [
                f for f in parent.glob("*.mp4")
                if f.is_file() and f.stat().st_size > 1024
            ]
            if mp4s:
                return max(mp4s, key=lambda f: f.stat().st_mtime)
    except OSError:
        pass
    return None


# How long the live runner waits for the recording to become available
# after the 'ended' webhook. Generous on purpose: in the Path-B (R2) flow
# the upload of a multi-hour recording can lag the 'ended' call by minutes,
# and this runs in the background. The manual "Run pipeline" button is the
# fallback if the window expires. 60 x 5s = 5 min.
_LIVE_RECORDING_WAIT_ATTEMPTS = 60
_LIVE_RECORDING_WAIT_DELAY_S = 5.0


async def _acquire_live_recording(
    *, stream_id: str, recording_path: str, work_dir: "Path"
) -> "Path | None":
    """Get the finished live recording onto local disk, polling until it
    shows up (or the wait budget runs out).

    Two source modes, chosen by config:
      - Path B (object storage configured): poll R2 under
        `<prefix>/<stream_id>/` until the live-ingest service's upload
        appears, download it, return the local path. This is what lets the
        MediaMTX service run as a separate Railway service — no shared disk.
      - Path A (no R2): poll the shared `/data` volume for the file MediaMTX
        wrote in-place.

    Polling (not a single stat) matters either way: MediaMTX may still be
    flushing, or the R2 upload may still be in flight, when 'ended' fires.
    Returns None if nothing usable arrives within the budget.
    """
    import asyncio

    from chalybclip.integrations.storage import build_recording_store
    from chalybclip.settings import get_settings

    store = build_recording_store(get_settings())
    for _ in range(_LIVE_RECORDING_WAIT_ATTEMPTS):
        if store is not None:
            got = await store.fetch_latest(stream_id=stream_id, dest_dir=work_dir)
            if got is not None:
                return got
        else:
            local = _resolve_recording_file(recording_path)
            if local is not None:
                return local
        await asyncio.sleep(_LIVE_RECORDING_WAIT_DELAY_S)
    # One last attempt after the final sleep.
    if store is not None:
        return await store.fetch_latest(stream_id=stream_id, dest_dir=work_dir)
    return _resolve_recording_file(recording_path)


async def live_pipeline_runner(
    *,
    tenant_id: str,
    stream_id: str,
    persona_id: str,
    recording_path: str,
    output_dir: "Path",
    title: str | None = None,
    language: str | None = None,
) -> None:
    """Phase L.2 — run the full clip pipeline on a finished live recording.

    Triggered automatically by the MediaMTX `live/ended` webhook (see
    `maybe_autoclip_after_live_end`). The recording is treated exactly like
    an upload: `ingest_uploaded` moves it into the canonical
    `<output_dir>/<stream_id>/source/` layout, extracts audio, ffprobes the
    real duration, and writes `stream.json` — after which `process_vod`
    reuses that cache and skips any download. So transcribe → detect → cut
    → score all run identically to the upload path.

    Failure surfacing mirrors the other runners: anything that escapes gets
    a `pipeline.failed` event (so the live dashboard explains it instead of
    leaving the stream stuck on 'processing'), then re-raises for the log.
    """
    from chalybclip.db import Database, StreamsRepo
    from chalybclip.db.adapters import stream_to_row
    from chalybclip.ingest import ingest_uploaded
    from chalybclip.pipeline import process_vod
    from chalybclip.settings import get_settings
    from chalybclip.tenancy import bound_tenant

    db_path = resolve_db_target(get_settings())

    try:
        # 1. Acquire the recording locally — pull from R2 (Path B) or read
        #    the shared volume (Path A), polling until it shows up.
        work_dir = output_dir / stream_id / "_incoming"
        source_file = await _acquire_live_recording(
            stream_id=stream_id,
            recording_path=recording_path,
            work_dir=work_dir,
        )
        if source_file is None:
            raise RuntimeError(
                f"live recording for {stream_id} did not become available "
                f"(R2 prefix live/{stream_id}/ or disk {recording_path}); "
                "the upload may have failed or recordPath differs"
            )

        # 2. Ingest the recording like an upload (audio + duration +
        #    stream.json + canonical layout). Idempotent on stream_id.
        stream = await ingest_uploaded(
            tenant_id=tenant_id,
            source_path=source_file,
            output_dir=output_dir,
            stream_id=stream_id,
            title=title or f"Live {stream_id[:12]}",
        )

        # 3. Promote the StreamRow with the real duration/paths, but keep
        #    it tagged as a live stream (platform + live:// url) so the
        #    dashboard still groups it with live runs.
        db = Database(db_path)
        await db.connect()
        try:
            with bound_tenant(tenant_id):
                row = stream_to_row(stream).model_copy(
                    update={
                        "platform": "live",
                        "vod_url": f"live://rtmp/{stream_id}",
                    }
                )
                await StreamsRepo(db).upsert(row)
                # upsert is INSERT OR IGNORE; backfill the real length onto
                # the live row the ChalyOBS webhook created with duration 0.
                await StreamsRepo(db).set_duration(
                    stream.id, float(getattr(stream, "duration_s", 0) or 0)
                )
        finally:
            await db.close()

        # 4. Run the pipeline on the cached stream.json (no re-ingest).
        # Registered as in-flight so the disk reclaimers never eat the
        # recording mid-run.
        from chalybclip.jobs.active import pipeline_active

        with pipeline_active(stream.id):
            await process_vod(
                tenant_id=tenant_id,
                vod_url=stream.vod_url,
                output_dir=output_dir,
                persona_id=persona_id,
                stream_id=stream.id,
                language=language,
                db_path=db_path,
            )
    except Exception as e:
        try:
            await _emit_top_level_failure(
                db_path=db_path,
                tenant_id=tenant_id,
                stream_id=stream_id,
                error=e,
            )
        except Exception:
            _log.exception(
                "live pipeline.failed event write failed for stream=%s",
                stream_id,
            )
        raise

    # Run succeeded — flip the stream to a terminal status. The autoclip
    # claim set it to 'processing'; nothing else resets a live stream, so
    # without this the dashboard shows 'Analyzing…' forever even with clips
    # on disk. Best-effort: the clips already exist, so a failed status
    # write isn't worth failing the run.
    try:
        from chalybclip.db import Database
        from chalybclip.db.repos import _streams_repo_set_status

        sdb = Database(db_path)
        await sdb.connect()
        try:
            await _streams_repo_set_status(sdb, stream_id=stream_id, status="done")
        finally:
            await sdb.close()
    except Exception:  # noqa: BLE001 — cosmetic status only
        _log.warning("live: could not mark stream done for %s", stream_id)

    # Token T2/T3 — run succeeded; charge the base fee + refresh balance.
    await _refresh_balance_after_run(
        db_path=db_path, tenant_id=tenant_id, stream_id=stream_id,
    )


async def maybe_autoclip_after_live_end(
    db: "Database",
    *,
    row: object,
    schedule: "Callable[..., None]",
) -> bool:
    """Phase L.2 — decide whether to auto-launch the clip pipeline after a
    live stream ends, and if so, claim the stream + call `schedule(...)`.

    Separated from the webhook handler so it's testable without HTTP: the
    caller passes a `schedule` callback (the webhook hands in one that does
    `background_tasks.add_task(live_pipeline_runner, **kwargs)`; tests pass
    a collector).

    Skips (returns False) when:
      - the row is missing or not in 'live_ended' (safety),
      - auto-clip is disabled (`CHALYBCLIP_LIVE_AUTO_CLIP=false`),
      - the tenant has no persona (emits a `pipeline.failed` so the operator
        sees why; they can add one + run manually),
      - another webhook already claimed the stream (idempotency).

    Returns True iff it claimed the stream and scheduled the runner.
    """
    from pathlib import Path

    from chalybclip.db import PersonasRepo
    from chalybclip.db.repos import _streams_repo_try_claim_for_processing
    from chalybclip.settings import get_settings
    from chalybclip.tenancy import bound_tenant

    if row is None:
        return False
    status = getattr(row, "status", None)
    if status != "live_ended":
        return False

    settings = get_settings()
    if not getattr(settings, "live_auto_clip_enabled", True):
        return False

    tenant_id = getattr(row, "tenant_id", "")
    stream_id = getattr(row, "id", "")
    if not (tenant_id and stream_id):
        return False

    # Resolve the persona to clip with. Live ingest doesn't capture one at
    # push time, so we use the tenant's first persona (same default the
    # /dashboard/start page applies).
    with bound_tenant(tenant_id):
        personas = await PersonasRepo(db).list_for_tenant()
    if not personas:
        try:
            await _emit_top_level_failure(
                db_path=resolve_db_target(settings),
                tenant_id=tenant_id,
                stream_id=stream_id,
                error=RuntimeError(
                    "live stream ended but the tenant has no persona — "
                    "configure one to auto-clip, or run the pipeline manually"
                ),
            )
        except Exception:  # noqa: BLE001
            _log.warning("live autoclip: no-persona event failed for %s", stream_id)
        return False
    persona_id = personas[0].id

    # Atomic claim — idempotent against duplicate 'ended' webhooks.
    if not await _streams_repo_try_claim_for_processing(db, stream_id=stream_id):
        return False

    schedule(
        tenant_id=tenant_id,
        stream_id=stream_id,
        persona_id=persona_id,
        recording_path=str(getattr(row, "source_video_path", "") or ""),
        output_dir=Path(settings.default_output_dir),
        title=getattr(row, "title", None),
        language=None,
    )
    return True


__all__ = [
    "PipelineKickoff",
    "PipelineRunner",
    "default_pipeline_runner",
    "live_pipeline_runner",
    "maybe_autoclip_after_live_end",
    "upload_pipeline_runner",
]
