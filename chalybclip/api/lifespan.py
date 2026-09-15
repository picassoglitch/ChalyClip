"""FastAPI lifespan — auto-drains for the metrics + webhook workers.

When `enable_background_drains=True` (default) the API server kicks the
background loops at boot:

  * webhook dispatch      every 30s   per active tenant
  * metrics ingest        every 1h    per active tenant
  * retention sweep       every 24h   all tenants (runs shortly after boot)

The legacy publish_jobs drain was removed (Etapa A): publishing now goes
through Zernio, not the per-platform worker. `publish_interval_s` is kept
on the signatures for back-compat but no longer starts a loop.

The retention loop is what actually enforces the per-tenant retention
windows. `sweep_retention` was previously only reachable via the CLI
(`chalybclip retention sweep`), so unless an operator SSH'd in and ran it by
hand, nothing on disk was ever reclaimed and the volume grew without bound.
The loop runs once shortly after boot (so a fresh deploy reclaims space
promptly even if redeploys happen more often than once a day) and then on
the 24h cadence.

Each loop iterates `TenantsRepo.list_all()` and runs the drain serially
for each tenant. Per-tenant errors are logged and swallowed so a single
broken integration doesn't kill the loop. Shutdown cancels + awaits the
tasks before returning.

Tests call `create_app(enable_background_drains=False)` so the loops
never start. The drain functions themselves are exercised by their own
test suites.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import structlog
from fastapi import FastAPI

from chalybclip.db import Database, TenantsRepo

_log = structlog.get_logger(__name__)

# Default cadences. Tests + ops can override via `LifespanIntervals` if
# tighter timing matters.
_DEFAULT_PUBLISH_INTERVAL_S = 60.0
_DEFAULT_WEBHOOK_INTERVAL_S = 30.0
_DEFAULT_METRICS_INTERVAL_S = 3600.0
# Channel polling is comparatively expensive (a yt-dlp listing per watch
# plus a full pipeline run per new VOD) — keep it slow.
_DEFAULT_CHANNEL_POLL_INTERVAL_S = 900.0
# Retention only needs to run daily — aged-out artifacts don't get more
# aged-out by sweeping more often, and the scan touches every tenant.
_DEFAULT_RETENTION_INTERVAL_S = 86400.0
# Delay before the FIRST retention sweep after boot. Long enough to let
# migrations + warm-up finish and not contend with startup I/O, short
# enough that a deploy reclaims disk within minutes rather than a day.
_DEFAULT_RETENTION_INITIAL_DELAY_S = 120.0
# Pipeline-recovery cadence. The in-process dispatcher drops background
# tasks on a restart, freezing URL ingests at `pending`; this loop
# re-dispatches them. The FIRST pass runs shortly after boot so a deploy
# that orphaned a job heals within a minute, not on the next interval.
_DEFAULT_RECOVERY_INTERVAL_S = 600.0
_DEFAULT_RECOVERY_INITIAL_DELAY_S = 45.0
# Disk watchdog — far more frequent than the daily retention sweep because
# its job is to keep the volume from FILLING (a size budget), not to age out
# old artifacts (a time budget). Each tick sheds the oldest raw sources when
# free space is below the high-water mark. Cheap when there's nothing to do.
_DEFAULT_DISK_WATCHDOG_INTERVAL_S = 900.0
_DEFAULT_DISK_WATCHDOG_INITIAL_DELAY_S = 60.0


# Delay before the one-shot zernio-events recovery sweep. Short: its whole
# point is healing events whose background task died in the LAST process
# (deploy/crash mid-handling), so it should run right after boot settles.
_ZERNIO_PENDING_SWEEP_DELAY_S = 15.0


async def _zernio_pending_sweep(db: Database) -> None:
    """One-shot crash-recovery pass over unprocessed zernio_events.

    A Zernio webhook is persisted first and processed by a background
    task; a restart between the two leaves the row `processed=0` forever
    (`process_pending` existed for exactly this but was never wired in).
    Runs once shortly after boot — the sweep is idempotent, so overlapping
    with live webhook handling is safe.
    """
    from chalybclip.integrations.zernio.events import process_pending

    await asyncio.sleep(_ZERNIO_PENDING_SWEEP_DELAY_S)
    total = 0
    try:
        while True:
            n = await process_pending(db, limit=100)
            total += n
            if n < 100:
                break
    except Exception:
        _log.exception("zernio_pending_sweep_failed", processed_before_error=total)
        return
    if total:
        _log.info("zernio_pending_sweep_done", processed=total)


async def _webhook_loop(db: Database, interval_s: float) -> None:
    """Drain webhook subscriptions for every tenant on a loop."""
    from chalybclip.webhooks import run_webhook_dispatch

    while True:
        try:
            await asyncio.sleep(interval_s)
            tenants = await TenantsRepo(db).list_all()
            for tenant in tenants:
                try:
                    await run_webhook_dispatch(tenant.id, db)
                except Exception as e:
                    _log.warning(
                        "webhook_drain_failed",
                        tenant_id=tenant.id,
                        error=str(e),
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.warning("webhook_loop_iteration_failed", error=str(e))


async def _metrics_loop(db: Database, interval_s: float) -> None:
    """Pull engagement metrics for every tenant on a loop."""
    from chalybclip.metrics import run_metrics_ingest

    while True:
        try:
            await asyncio.sleep(interval_s)
            tenants = await TenantsRepo(db).list_all()
            for tenant in tenants:
                try:
                    await run_metrics_ingest(tenant.id, db)
                except Exception as e:
                    _log.warning(
                        "metrics_drain_failed",
                        tenant_id=tenant.id,
                        error=str(e),
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.warning("metrics_loop_iteration_failed", error=str(e))


async def _channel_poll_loop(
    db: Database,
    dispatcher: object,
    output_dir: Path,
    interval_s: float,
) -> None:
    """Poll every channel watch for new VODs and auto-ingest them."""
    from chalybclip.channels import make_channel_ingest_callback, poll_channel_watches
    from chalybclip.jobs import JobDispatcher

    assert isinstance(dispatcher, JobDispatcher)
    ingest_callback = make_channel_ingest_callback(
        db, dispatcher, output_dir=output_dir
    )
    while True:
        try:
            await asyncio.sleep(interval_s)
            await poll_channel_watches(
                db, ingest_callback=ingest_callback, respect_schedule=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.warning("channel_poll_loop_iteration_failed", error=str(e))


async def _retention_loop(
    db: Database,
    output_dir: Path,
    interval_s: float,
    initial_delay_s: float = _DEFAULT_RETENTION_INITIAL_DELAY_S,
) -> None:
    """Sweep per-tenant retention windows on a loop, hard-deleting aged
    artifacts (raw VODs, old clips, stale transcripts).

    Runs once after `initial_delay_s` (so a fresh deploy reclaims disk
    promptly even when redeploys are more frequent than the interval),
    then every `interval_s`. `sweep_retention` already logs per-tenant
    outcomes and swallows per-tenant errors, so a single bad tenant won't
    stall the loop.
    """
    from chalybclip.retention import reclaim_orphan_dirs, sweep_retention

    first = True
    while True:
        try:
            await asyncio.sleep(initial_delay_s if first else interval_s)
            first = False
            await sweep_retention(db, output_dir=output_dir)
            # Orphan pass AFTER the row-driven sweep: dirs whose rows are
            # gone (torn down by the sweep, lost in the SQLite->Postgres
            # cutover, deleted by hand) are invisible to every row-driven
            # reclaimer and would ratchet the volume up forever.
            await reclaim_orphan_dirs(db, output_dir=output_dir)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.warning("retention_loop_iteration_failed", error=str(e))


async def _recovery_loop(
    db: Database,
    dispatcher: object,
    output_dir: Path,
    interval_s: float,
    initial_delay_s: float = _DEFAULT_RECOVERY_INITIAL_DELAY_S,
) -> None:
    """Re-dispatch orphaned URL-ingest pipelines on a loop.

    The in-process dispatcher loses its background task when the process
    restarts (deploy, OOM, crash), leaving the URL-job stream frozen at
    `status="pending"` with nothing to resume it. This loop finds those
    dead-in-flight streams and re-runs them — `process_vod` is idempotent,
    so a re-dispatch is safe and the intake self-heals across restarts.

    Runs once after `initial_delay_s` (so a deploy that orphaned a job
    heals promptly) then every `interval_s`. `recover_orphaned_pipelines`
    logs + swallows per-tenant errors; `task_registry` keeps strong refs
    to the in-flight recovery tasks so they aren't GC'd mid-run.
    """
    from chalybclip.jobs import JobDispatcher
    from chalybclip.recovery import recover_orphaned_pipelines

    assert isinstance(dispatcher, JobDispatcher)
    task_registry: set[asyncio.Task[None]] = set()
    first = True
    while True:
        try:
            await asyncio.sleep(initial_delay_s if first else interval_s)
            first = False
            await recover_orphaned_pipelines(
                db,
                dispatcher=dispatcher,
                output_dir=output_dir,
                task_registry=task_registry,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.warning("recovery_loop_iteration_failed", error=str(e))


async def _disk_watchdog_loop(
    db: Database,
    output_dir: Path,
    interval_s: float,
    initial_delay_s: float = _DEFAULT_DISK_WATCHDOG_INITIAL_DELAY_S,
) -> None:
    """Keep the output volume under its disk budget by shedding the oldest
    raw sources whenever free space drops below the high-water mark.

    This is the proactive guard that stops the volume from filling at all —
    independent of the daily retention sweep (which only ages out artifacts)
    and the per-ingest preflight (which only fires when something downloads).
    Sources only; user clips/transcripts are never touched here. No-ops when
    `target_free_disk_bytes` is 0.
    """
    from chalybclip.retention import reclaim_sources_until_free
    from chalybclip.settings import get_settings

    first = True
    while True:
        try:
            await asyncio.sleep(initial_delay_s if first else interval_s)
            first = False
            target = int(getattr(get_settings(), "target_free_disk_bytes", 0) or 0)
            if target > 0:
                await reclaim_sources_until_free(
                    db, output_dir=output_dir, target_free_bytes=target,
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.warning("disk_watchdog_loop_iteration_failed", error=str(e))


@asynccontextmanager
async def background_drains_lifespan(
    app: FastAPI,
    *,
    publish_interval_s: float = _DEFAULT_PUBLISH_INTERVAL_S,
    webhook_interval_s: float = _DEFAULT_WEBHOOK_INTERVAL_S,
    metrics_interval_s: float = _DEFAULT_METRICS_INTERVAL_S,
    channel_poll_interval_s: float = _DEFAULT_CHANNEL_POLL_INTERVAL_S,
    retention_interval_s: float = _DEFAULT_RETENTION_INTERVAL_S,
    recovery_interval_s: float = _DEFAULT_RECOVERY_INTERVAL_S,
    disk_watchdog_interval_s: float = _DEFAULT_DISK_WATCHDOG_INTERVAL_S,
) -> AsyncIterator[None]:
    """Start the background loops on boot; cancel + await on shutdown."""
    db: Database = app.state.db
    started_at = _dt.datetime.now(_dt.UTC).isoformat()
    _log.info(
        "lifespan_starting",
        started_at=started_at,
        publish_interval_s=publish_interval_s,
        webhook_interval_s=webhook_interval_s,
        metrics_interval_s=metrics_interval_s,
        channel_poll_interval_s=channel_poll_interval_s,
        retention_interval_s=retention_interval_s,
        recovery_interval_s=recovery_interval_s,
    )
    from chalybclip.settings import get_settings

    output_dir = Path(get_settings().default_output_dir)
    tasks = [
        asyncio.create_task(
            _webhook_loop(db, webhook_interval_s), name="chalybclip-webhook-loop"
        ),
        asyncio.create_task(
            _metrics_loop(db, metrics_interval_s), name="chalybclip-metrics-loop"
        ),
        asyncio.create_task(
            _retention_loop(db, output_dir, retention_interval_s),
            name="chalybclip-retention-loop",
        ),
        asyncio.create_task(
            _disk_watchdog_loop(db, output_dir, disk_watchdog_interval_s),
            name="chalybclip-disk-watchdog-loop",
        ),
        asyncio.create_task(
            _zernio_pending_sweep(db), name="chalybclip-zernio-pending-sweep"
        ),
    ]
    # The channel-poll + recovery loops need the pipeline dispatcher + output
    # dir; only start them when the app wired a dispatcher (the dashboard app
    # does).
    dispatcher = getattr(app.state, "job_dispatcher", None)
    if dispatcher is not None:
        tasks.append(
            asyncio.create_task(
                _channel_poll_loop(
                    db, dispatcher, output_dir, channel_poll_interval_s
                ),
                name="chalybclip-channel-poll-loop",
            )
        )
        tasks.append(
            asyncio.create_task(
                _recovery_loop(db, dispatcher, output_dir, recovery_interval_s),
                name="chalybclip-recovery-loop",
            )
        )
    try:
        yield
    finally:
        _log.info("lifespan_shutting_down")
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError, Exception):
                await task
