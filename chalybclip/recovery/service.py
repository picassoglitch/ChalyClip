"""Pipeline recovery sweeper — re-dispatches orphaned ingest jobs.

WHY THIS EXISTS
    The pipeline runs as in-process FastAPI BackgroundTasks via
    `InProcessJobDispatcher`, which is explicitly NOT durable: if the
    process dies mid-run (deploy, OOM, crash), the background task is
    lost. The URL-ingest path (`POST /dashboard/url-jobs`) inserts a
    placeholder stream row (`status="pending"`, `duration_s=0`) and
    defers the ENTIRE pipeline — including the download — to that task.
    So a restart between the 303 response and the task running freezes
    the row at `pending`/0s forever.

    Nothing recovered these: the dashboard's "abandoned" detector only
    fires for a step that is `running` with a stale `started_at`, and an
    orphan that died before emitting any `pipeline.step.start` has
    all-pending steps — indistinguishable from a fresh submission. So it
    shows "Downloading…" indefinitely with no re-run affordance.

WHAT THIS DOES
    On a loop (started by the API lifespan), find streams that are
    in-flight-but-dead and re-dispatch them through the same dispatcher.
    `process_vod` is idempotent (CLAUDE.md rule 4) — a re-download reuses
    the cached `stream.json`, transcripts upsert, candidate/clip inserts
    dedupe — so re-running is safe and self-healing across restarts.

SAFETY — what we DON'T touch
    * Live streams + uploads: only `http(s)` VOD URLs are recovered. The
      live (`live://`) and upload (`upload://`) paths have their own
      runners and their source bytes may be gone, so re-running the
      generic `process_vod` on them is wrong. URL ingest is the actual
      orphan class anyway.
    * Completed runs: a `stream.processed` event is terminal → skipped.
    * Surfaced failures: a `pipeline.failed` event already shows the
      operator an error + a manual re-run button → skipped.
    * Legitimately-running long steps: Whisper on a multi-hour VOD can
      sit between `pipeline.step.start` and `.done` with NO intermediate
      event for ~2h. So a stream that HAS started a step is only treated
      as dead after `IN_FLIGHT_SILENCE_S` of total event silence. A
      never-started stream (no `step.start` at all) is eligible after the
      much shorter `NEVER_STARTED_GRACE_S` — nothing is running, and a
      healthy dispatch emits `ingest.start` within seconds.
    * Thrash: each re-dispatch writes a `stream.recovery_dispatched`
      event; after `MAX_ATTEMPTS` we emit one give-up `pipeline.failed`
      (which then makes the stream terminal) instead of looping forever.
    * Ancient rows: streams older than `MAX_AGE_S` are left for retention
      — silently auto-running a two-week-old VOD is not what the operator
      wants.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from chalybclip.db import (
    Database,
    EventsRepo,
    PersonasRepo,
    StreamsRepo,
    TenantsRepo,
)
from chalybclip.events.log import STREAM_PROCESSED
from chalybclip.tenancy import bound_tenant

if TYPE_CHECKING:
    from chalybclip.db.models import Event, StreamRow
    from chalybclip.jobs import JobDispatcher

_log = structlog.get_logger(__name__)

# Event types we key on. STREAM_PROCESSED ("stream.processed") is the
# terminal-success signal the pipeline emits at the end; the rest are
# literal strings emitted by the per-step context + failure surfacing
# (no shared constants exist for them yet).
_PIPELINE_FAILED = "pipeline.failed"
_STEP_START = "pipeline.step.start"
RECOVERY_DISPATCHED = "stream.recovery_dispatched"

# A never-started orphan is eligible this long after creation. Short
# because nothing is running — a healthy dispatch emits ingest.start
# within seconds, so silence this long means the task never ran (or died
# instantly on a restart).
NEVER_STARTED_GRACE_S = 15 * 60

# A stream that HAS started a step is only declared dead after this much
# total event silence. Sized above the longest plausible silent step
# (Whisper on a multi-hour VOD) so we never kill a slow-but-alive run.
IN_FLIGHT_SILENCE_S = 2 * 3600

# Don't resurrect streams older than this — retention will reclaim them,
# and auto-running a weeks-old VOD is surprising rather than helpful.
MAX_AGE_S = 14 * 24 * 3600

# After this many auto-redispatches that still didn't reach a terminal
# state, give up and surface a failure instead of looping forever.
MAX_ATTEMPTS = 3

# Cap re-dispatches per sweep so a long outage doesn't unleash a thundering
# herd of concurrent heavy pipelines. The rest are picked up next sweep.
MAX_PER_SWEEP = 3


@dataclass(frozen=True)
class OrphanDecision:
    """One stream's classification. `give_up` means it has exhausted its
    recovery attempts and should be failed rather than retried again."""

    stream_id: str
    created_at: str
    give_up: bool


def _parse_ts(value: str) -> _dt.datetime | None:
    """Parse an ISO-8601 event/stream timestamp to an aware UTC datetime.

    Returns None on an unparseable value (treated as "no signal" by the
    caller). Naive timestamps are assumed UTC — every writer in this repo
    stamps UTC."""
    try:
        dt = _dt.datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.UTC)
    return dt


def classify_streams(
    streams: list[StreamRow],
    events: list[Event],
    *,
    now: _dt.datetime,
    active_ids: frozenset[str] = frozenset(),
) -> list[OrphanDecision]:
    """Pure decision core — pick which streams are orphaned (dead in-flight)
    and whether each should be retried or given up on.

    Side-effect free + deterministic given `now`, so the threshold logic is
    unit-testable without a database. The lifespan loop feeds it the live
    rows; `recover_orphaned_pipelines` turns decisions into dispatches.

    `active_ids` is the `jobs.active` registry — streams with a run queued
    or executing in THIS process. They are excluded outright: a run parked
    behind the concurrency semaphore for hours (two multi-hour VODs hold
    both slots all afternoon) emits no events, which is indistinguishable
    from a lost task by timing alone. Without this, the sweeper stacks
    duplicate dispatches onto the queue and then writes a bogus
    RecoveryExhausted `pipeline.failed` over a perfectly healthy run.
    After a restart the registry is empty, so true orphans still recover.
    """
    # Bucket events by their stream_id payload once — the events table is
    # per-tenant and small (a handful of types per stream).
    by_stream: dict[str, list[Event]] = {}
    for ev in events:
        sid = ev.payload.get("stream_id")
        if isinstance(sid, str):
            by_stream.setdefault(sid, []).append(ev)

    decisions: list[OrphanDecision] = []
    for s in streams:
        if s.id in active_ids:
            continue  # queued or running in this process — not an orphan
        # Only http(s) VOD ingests — live:// and upload:// have their own
        # runners and possibly-gone source bytes (see module docstring).
        if getattr(s, "is_live", False) or not s.vod_url.lower().startswith("http"):
            continue

        created = _parse_ts(s.created_at)
        if created is None:
            continue
        age_s = (now - created).total_seconds()
        if age_s > MAX_AGE_S:
            continue  # too old — leave for retention

        evs = by_stream.get(s.id, [])
        types = {e.type for e in evs}
        if STREAM_PROCESSED in types:
            continue  # completed successfully
        if _PIPELINE_FAILED in types:
            continue  # already surfaced as failed (manual re-run available)

        # Total event silence: a never-started orphan's only event is
        # `stream.created` (~= created_at), so silence ~= age there.
        event_ts = [t for t in (_parse_ts(e.ts) for e in evs) if t is not None]
        last_activity = max(event_ts) if event_ts else created
        silence_s = (now - last_activity).total_seconds()

        if _STEP_START in types:
            # Started a step then went silent — only dead past the long
            # ceiling so we never kill a slow Whisper.
            eligible = silence_s >= IN_FLIGHT_SILENCE_S
        else:
            # Never emitted a single step → the task never ran.
            eligible = age_s >= NEVER_STARTED_GRACE_S and silence_s >= NEVER_STARTED_GRACE_S
        if not eligible:
            continue

        attempts = sum(1 for e in evs if e.type == RECOVERY_DISPATCHED)
        decisions.append(
            OrphanDecision(
                stream_id=s.id,
                created_at=s.created_at,
                give_up=attempts >= MAX_ATTEMPTS,
            )
        )
    return decisions


async def recover_orphaned_pipelines(
    db: Database,
    *,
    dispatcher: JobDispatcher,
    output_dir: Path,
    now: _dt.datetime | None = None,
    max_per_sweep: int = MAX_PER_SWEEP,
    task_registry: set[asyncio.Task[None]] | None = None,
) -> list[str]:
    """Sweep every tenant for orphaned URL-ingest pipelines and re-dispatch
    them. Returns the stream ids re-dispatched this sweep.

    Per tenant: load the streams + recent events, classify, then for each
    orphan (oldest first, capped at `max_per_sweep`) resolve the persona,
    emit a `stream.recovery_dispatched` audit event, and schedule the
    pipeline via the dispatcher. Streams that have exhausted their attempts
    get one give-up `pipeline.failed` so they stop being reconsidered and
    the operator sees an explicit error.

    Per-tenant errors are logged and swallowed so one bad tenant can't stall
    the loop. Dispatch is fire-and-forget (the pipeline runs concurrently);
    `task_registry` holds strong refs so the tasks aren't GC'd mid-run.
    """
    now = now or _dt.datetime.now(_dt.UTC)
    recovered: list[str] = []

    tenants = await TenantsRepo(db).list_all()
    for tenant in tenants:
        try:
            with bound_tenant(tenant.id):
                recovered.extend(
                    await _recover_one_tenant(
                        db,
                        tenant_id=tenant.id,
                        dispatcher=dispatcher,
                        output_dir=output_dir,
                        now=now,
                        max_per_sweep=max_per_sweep,
                        task_registry=task_registry,
                    )
                )
        except Exception as e:  # one tenant must not stall the loop
            _log.warning("recovery.tenant_failed", tenant_id=tenant.id, error=str(e))
    if recovered:
        _log.info("recovery.swept", count=len(recovered), stream_ids=recovered)
    return recovered


async def _recover_one_tenant(
    db: Database,
    *,
    tenant_id: str,
    dispatcher: JobDispatcher,
    output_dir: Path,
    now: _dt.datetime,
    max_per_sweep: int,
    task_registry: set[asyncio.Task[None]] | None,
) -> list[str]:
    """Per-tenant body of the sweep. Tenant is already bound."""
    from chalybclip.jobs.active import active_stream_ids

    streams = await StreamsRepo(db).list_for_tenant()
    # Generous limit: must reach the `stream.processed` / step events of
    # any stream within MAX_AGE_S so terminal runs aren't re-dispatched.
    events = await EventsRepo(db).list_for_tenant(limit=5000)
    decisions = classify_streams(
        streams, events, now=now, active_ids=active_stream_ids()
    )
    if not decisions:
        return []

    # Oldest first so a backlog drains in submission order.
    decisions.sort(key=lambda d: d.created_at)

    give_ups = [d for d in decisions if d.give_up]
    retries = [d for d in decisions if not d.give_up][:max_per_sweep]

    for d in give_ups:
        await EventsRepo(db).emit(
            type=_PIPELINE_FAILED,
            payload={
                "stream_id": d.stream_id,
                "error_type": "RecoveryExhausted",
                "error": (
                    f"Auto-recovery re-dispatched this run {MAX_ATTEMPTS} times "
                    "and it never reached a terminal state. Giving up — re-run "
                    "it manually to retry."
                ),
            },
        )
        _log.warning("recovery.gave_up", tenant_id=tenant_id, stream_id=d.stream_id)

    if not retries:
        return []

    # The placeholder URL-job inserts no persona on the row, so resolve the
    # same default the submit path used: the tenant's first persona.
    personas = await PersonasRepo(db).list_for_tenant()
    if not personas:
        _log.warning("recovery.no_persona", tenant_id=tenant_id)
        return []
    persona_id = personas[0].id

    streams_by_id = {s.id: s for s in streams}
    recovered: list[str] = []
    for d in retries:
        row = streams_by_id.get(d.stream_id)
        if row is None:
            continue
        await EventsRepo(db).emit(
            type=RECOVERY_DISPATCHED,
            payload={"stream_id": d.stream_id, "vod_url": row.vod_url},
        )
        await _dispatch_recovery(
            dispatcher=dispatcher,
            tenant_id=tenant_id,
            row=row,
            persona_id=persona_id,
            output_dir=output_dir,
            task_registry=task_registry,
        )
        recovered.append(d.stream_id)
        _log.info(
            "recovery.dispatched",
            tenant_id=tenant_id,
            stream_id=d.stream_id,
            vod_url=row.vod_url,
        )
    return recovered


async def _dispatch_recovery(
    *,
    dispatcher: JobDispatcher,
    tenant_id: str,
    row: StreamRow,
    persona_id: str,
    output_dir: Path,
    task_registry: set[asyncio.Task[None]] | None,
) -> None:
    """Build a PipelineKickoff from the stream row and schedule it.

    Mirrors `dashboard.url_job_create`'s stub: `default_pipeline_runner`
    only reads `kickoff.stream.{id,vod_url}` — the source paths are the
    canonical placeholders `process_vod`'s ingest will materialize. We
    schedule via `create_task` so the (potentially long) pipeline runs
    concurrently instead of blocking the sweep.
    """
    from chalybclip.ingest import Stream as StreamModel
    from chalybclip.jobs import PipelineKickoff

    source_dir = output_dir / row.id / "source"
    stub = StreamModel(
        id=row.id,
        tenant_id=tenant_id,
        vod_url=row.vod_url,
        platform=row.platform,
        duration_s=0.0,
        source_video_path=source_dir / "video.mp4",
        source_audio_path=source_dir / "audio.wav",
    )
    kickoff = PipelineKickoff(
        tenant_id=tenant_id,
        stream=stub,
        persona_id=persona_id,
        output_dir=output_dir,
    )
    # background_tasks=None → the in-process dispatcher awaits the runner;
    # wrap in a task so this sweep doesn't block on the full pipeline.
    task = asyncio.create_task(
        dispatcher.dispatch_pipeline(kickoff),
        name=f"chalybclip-recovery-{row.id}",
    )
    if task_registry is not None:
        task_registry.add(task)
        task.add_done_callback(task_registry.discard)
