"""Admission + metering for pipeline runs (the consumption contract).

Every heavy run goes through three moments, all here so each kickoff path
(upload routes, URL ingest, channel poll, live autoclip, recovery, rerun)
gets identical behaviour:

  1. `admit_job` — ask the hub before accepting bytes / starting work.
     Records the run in `usage_jobs` under a fresh external_job_id.
  2. `readmit_job` — once the media is probed, re-admit the SAME job id
     with the real `source_minutes` (and the real upload size). A refusal
     here cancels the reservation and raises.
  3. `metered_run` — wraps the run itself: binds the reservation so every
     usage event carries it, heartbeats long runs, and in `finally` meters
     compute.seconds for the lane, settles the reservation
     (succeeded / failed / cancelled) and drains the outbox.

`AdmissionRefused` (integrations.chalyb.admission) carries the HTTP status
+ Spanish message a route returns; background paths turn it into a
`pipeline.failed` event via `fail_stream_for_refusal`.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import math
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chalybclip.db import Database
from chalybclip.db.usage_repos import (
    UsageJobsRepo,
    set_stream_storage_bytes,
    tenant_storage_bytes,
)
from chalybclip.ids import new_id
from chalybclip.integrations.chalyb.admission import (
    AdmissionRefused,
    AdmitRequest,
    admit,
    heartbeat,
)
from chalybclip.integrations.chalyb.outbox import (
    bind_reservation,
    drain_until_idle,
    enqueue_settle,
    enqueue_usage,
    reset_reservation,
)
from chalybclip.settings import get_settings

_log = logging.getLogger("chalybclip.jobs.usage")

_MIB = 1024 * 1024

# compute.seconds cost per lane (contract: Cloud Run us-central1 list
# prices). standard = 4 vCPU / 8 GiB worker, boost = 8 vCPU / 32 GiB job.
COMPUTE_MICROS_PER_S: dict[str, int] = {"standard": 88, "boost": 208}

# Admission estimate inputs. Wall time is assumed ~0.5x the source length
# (transcribe is remote; detect + cut dominate); a source of unknown
# length (pre-probe) is estimated as 10 minutes so the first admit doesn't
# refuse a short clip on a 1-hour guess — the post-probe re-admit corrects it.
_EST_WALL_PER_SOURCE_S = 0.5
_UNKNOWN_SOURCE_MINUTES = 10.0
# Billable tokens per USD micro (contract: blended $4 / 1M tokens).
_MICROS_PER_TOKEN = 4

OPERATION_PIPELINE = "clips.pipeline"


def estimate_tokens(*, source_minutes: float, lane: str = "standard") -> int:
    """Best estimate of the billable tokens a run will draw: LLM +
    transcription (estimator), compute for the lane, and the base fee."""
    from chalybclip.llm.estimator import estimate_pipeline_run

    minutes = source_minutes if source_minutes > 0 else _UNKNOWN_SOURCE_MINUTES
    est = estimate_pipeline_run(duration_seconds=minutes * 60.0)
    compute_um = int(
        minutes * 60.0 * _EST_WALL_PER_SOURCE_S * COMPUTE_MICROS_PER_S.get(lane, 88)
    )
    base_um = int(getattr(get_settings(), "pipeline_base_charge_usd_micros", 0) or 0)
    total_um = est.total_cost_usd_micros + compute_um + base_um
    return max(1, math.ceil(total_um / _MICROS_PER_TOKEN))


@dataclass(frozen=True)
class JobAdmission:
    """An admitted run — what a kickoff carries into the runner."""

    job_id: str
    tenant_id: str
    stream_id: str
    reservation_id: str | None
    lane: str = "standard"
    boost: bool | None = None
    limits: dict[str, Any] = field(default_factory=dict)
    local: bool = False

    @property
    def max_upload_bytes(self) -> int | None:
        mb = self.limits.get("max_upload_mb")
        try:
            return int(float(mb) * _MIB) if mb is not None and float(mb) > 0 else None
        except (TypeError, ValueError):
            return None


async def admit_job(
    db: Database,
    *,
    tenant_id: str,
    stream_id: str,
    upload_bytes: int = 0,
    source_minutes: float = 0.0,
    boost: bool | None = None,
    operation: str = OPERATION_PIPELINE,
    job_class: str = "job",
    job_id: str | None = None,
) -> JobAdmission:
    """Admit one run. Raises `AdmissionRefused` (refused, or hub down)."""
    job_id = job_id or new_id("ujob")
    upload_mb = max(0, int(upload_bytes)) / _MIB
    stored_mb = await _stored_mb(db, tenant_id)
    result = await admit(
        db,
        tenant_id=tenant_id,
        request=AdmitRequest(
            external_job_id=job_id,
            job_class=job_class,
            operation=operation,
            est_tokens=estimate_tokens(source_minutes=source_minutes),
            upload_mb=upload_mb,
            source_minutes=source_minutes,
            storage_mb_after=stored_mb + upload_mb,
            boost=boost,
            ttl_seconds=int(get_settings().usage_reservation_ttl_s),
        ),
    )
    await UsageJobsRepo(db).upsert(
        job_id=job_id,
        tenant_id=tenant_id,
        stream_id=stream_id,
        operation=operation,
        job_class=job_class,
        reservation_id=result.reservation_id,
        lane=result.lane,
        boost=boost,
        upload_mb=upload_mb,
        source_minutes=source_minutes,
    )
    return JobAdmission(
        job_id=job_id,
        tenant_id=tenant_id,
        stream_id=stream_id,
        reservation_id=result.reservation_id,
        lane=result.lane,
        boost=boost,
        limits=result.limits,
        local=result.local,
    )


_KEEP: Any = object()


async def readmit_job(
    db: Database,
    *,
    job_id: str | None,
    source_minutes: float | None = None,
    upload_bytes: int | None = None,
    boost: bool | None = _KEEP,
    stream_id: str | None = None,
) -> JobAdmission | None:
    """Re-check an admitted job with its real numbers (same job id → the
    hub updates the same reservation). On refusal the reservation is
    cancelled before `AdmissionRefused` propagates. Returns None for a run
    with no admission record (pre-contract kickoff)."""
    if not job_id:
        return None
    repo = UsageJobsRepo(db)
    row = await repo.get(job_id)
    if row is None:
        return None
    minutes = row.source_minutes if source_minutes is None else max(0.0, source_minutes)
    upload_mb = row.upload_mb if upload_bytes is None else max(0, int(upload_bytes)) / _MIB
    new_boost = row.boost if boost is _KEEP else boost
    try:
        stored_mb = await _stored_mb(db, row.tenant_id)
        result = await admit(
            db,
            tenant_id=row.tenant_id,
            request=AdmitRequest(
                external_job_id=row.id,
                job_class=row.job_class,
                operation=row.operation,
                est_tokens=estimate_tokens(source_minutes=minutes, lane=row.lane),
                upload_mb=upload_mb,
                source_minutes=minutes,
                # The upload is already part of what the tenant will hold.
                storage_mb_after=stored_mb + upload_mb,
                boost=new_boost,
                ttl_seconds=int(get_settings().usage_reservation_ttl_s),
            ),
        )
    except AdmissionRefused:
        await enqueue_settle(
            db, tenant_id=row.tenant_id, reservation_id=row.reservation_id,
            outcome="cancelled",
        )
        await repo.set_status(row.id, "refused")
        raise
    reservation_id = result.reservation_id or row.reservation_id
    stream_id = stream_id or row.stream_id
    await repo.upsert(
        job_id=row.id,
        tenant_id=row.tenant_id,
        stream_id=stream_id,
        operation=row.operation,
        job_class=row.job_class,
        reservation_id=reservation_id,
        lane=result.lane if not result.local else row.lane,
        boost=new_boost,
        upload_mb=upload_mb,
        source_minutes=minutes,
    )
    if not result.local and result.lane != row.lane:
        _log.info(
            "re-admit changed lane %s→%s for job=%s (run already placed; kept %s)",
            row.lane, result.lane, row.id, row.lane,
        )
    return JobAdmission(
        job_id=row.id,
        tenant_id=row.tenant_id,
        stream_id=stream_id,
        reservation_id=reservation_id,
        lane=result.lane if not result.local else row.lane,
        boost=new_boost,
        limits=result.limits,
        local=result.local,
    )


async def cancel_admission(db: Database, admission: JobAdmission | None) -> None:
    """Release an admission whose run never started (upload rejected,
    dispatch deduped, …). Never raises."""
    if admission is None:
        return
    try:
        await enqueue_settle(
            db, tenant_id=admission.tenant_id,
            reservation_id=admission.reservation_id, outcome="cancelled",
        )
        await UsageJobsRepo(db).set_status(admission.job_id, "cancelled")
    except Exception:  # noqa: BLE001
        _log.exception("cancel_admission failed · job=%s", admission.job_id)


async def settle_job(
    db: Database, *, tenant_id: str, job_id: str | None,
    reservation_id: str | None, outcome: str,
) -> None:
    """Terminal settle by ids (dispatcher-side failures). Never raises."""
    try:
        await enqueue_settle(
            db, tenant_id=tenant_id, reservation_id=reservation_id, outcome=outcome
        )
        if job_id:
            await UsageJobsRepo(db).set_status(job_id, outcome)
    except Exception:  # noqa: BLE001
        _log.exception("settle_job failed · job=%s", job_id)


async def _stored_mb(db: Database, tenant_id: str) -> float:
    try:
        return await tenant_storage_bytes(db, tenant_id) / _MIB
    except Exception:  # noqa: BLE001 — a missing figure shouldn't block a run
        _log.warning("storage lookup failed · tenant=%s", tenant_id)
        return 0.0


def dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue
    except OSError:
        return 0
    return total


# ---- metering -------------------------------------------------------------


@dataclass
class MeteredRun:
    """Handle yielded by `metered_run` — lets the run report what it is."""

    stream_id: str
    job_id: str | None
    reservation_id: str | None
    lane: str

    async def readmit(self, db: Database, *, source_minutes: float) -> None:
        """Post-probe re-admit from inside a run (URL ingests learn the
        duration only after download). Raises AdmissionRefused."""
        await readmit_job(db, job_id=self.job_id, source_minutes=source_minutes)


@asynccontextmanager
async def metered_run(
    db_path: str,
    *,
    tenant_id: str,
    stream_id: str,
    job_id: str | None,
    reservation_id: str | None,
    lane: str = "standard",
    stream_dir: Path | None = None,
) -> AsyncIterator[MeteredRun]:
    """Meter + settle one pipeline run, in every exit path.

    On exit: one `compute.seconds` event (provider gcp, the lane's rate),
    the reservation settled succeeded / failed / cancelled, the stream's
    storage measured (success only), then a bounded outbox drain so a
    worker that scales to zero right after doesn't sit on its events."""
    lane = lane if lane in COMPUTE_MICROS_PER_S else "standard"
    token = bind_reservation(reservation_id)
    hb_task: asyncio.Task[None] | None = None
    if reservation_id:
        hb_task = asyncio.create_task(_heartbeat_loop(reservation_id))
    started = time.monotonic()
    # This attempt's own id: the same usage job can run more than once (a
    # dispatch retried after a lost response, a boost start that timed out
    # but ran, then fell back to the standard worker), and each attempt
    # really used its compute.
    attempt_id = new_id("run")
    outcome = "failed"
    handle = MeteredRun(
        stream_id=stream_id, job_id=job_id, reservation_id=reservation_id, lane=lane
    )
    try:
        yield handle
        outcome = "succeeded"
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except AdmissionRefused:
        # Mid-run refusal (post-probe re-admit) already cancelled the
        # reservation; settling again is a no-op, but say what happened.
        outcome = "cancelled"
        raise
    finally:
        if hb_task is not None:
            hb_task.cancel()
        elapsed = max(0.0, time.monotonic() - started)
        reset_reservation(token)
        # Shielded: a cancelled run must still get metered + settled.
        await asyncio.shield(
            _finish_run(
                db_path, tenant_id=tenant_id, stream_id=stream_id, job_id=job_id,
                reservation_id=reservation_id, lane=lane, elapsed_s=elapsed,
                outcome=outcome, stream_dir=stream_dir, attempt_id=attempt_id,
            )
        )


async def _heartbeat_loop(reservation_id: str) -> None:
    interval = float(get_settings().usage_heartbeat_interval_s or 1800.0)
    while True:
        await asyncio.sleep(interval)
        await heartbeat(reservation_id)


async def _finish_run(
    db_path: str,
    *,
    tenant_id: str,
    stream_id: str,
    job_id: str | None,
    reservation_id: str | None,
    lane: str,
    elapsed_s: float,
    outcome: str,
    stream_dir: Path | None,
    attempt_id: str | None = None,
) -> None:
    try:
        db = Database(db_path)
        await db.connect()
    except Exception:  # noqa: BLE001
        _log.exception("metering: DB open failed · stream=%s", stream_id)
        return
    try:
        seconds = max(1, math.ceil(elapsed_s))
        await enqueue_usage(
            db,
            tenant_id=tenant_id,
            kind="compute.seconds",
            amount=seconds,
            cost_usd_micros=seconds * COMPUTE_MICROS_PER_S[lane],
            # One compute event per run ATTEMPT. Keyed on the job id alone,
            # a second attempt of the same job collided with the first
            # (outbox + hub both dedupe on source_id) and went unbilled.
            source_id=(
                f"compute_{job_id}_{attempt_id or new_id('run')}"
                if job_id else f"compute_{attempt_id or new_id('run')}"
            ),
            occurred_at_iso=_dt.datetime.now(_dt.UTC).isoformat(),
            provider="gcp",
            operation=OPERATION_PIPELINE,
            reservation_id=reservation_id,
            metadata={"lane": lane, "stream_id": stream_id, "outcome": outcome},
            kick=False,
        )
        if outcome == "succeeded" and stream_dir is not None:
            size = await asyncio.to_thread(dir_size_bytes, stream_dir)
            await set_stream_storage_bytes(
                db, tenant_id=tenant_id, stream_id=stream_id, storage_bytes=size
            )
        await enqueue_settle(
            db, tenant_id=tenant_id, reservation_id=reservation_id, outcome=outcome,
            kick=False,
        )
        if job_id:
            await UsageJobsRepo(db).set_status(job_id, outcome)
        await drain_until_idle(db, timeout_s=30.0)
    except Exception:  # noqa: BLE001 — metering must never mask the run's outcome
        _log.exception("metering: finish failed · stream=%s job=%s", stream_id, job_id)
    finally:
        await db.close()


async def fail_stream_for_refusal(
    db: Database, *, tenant_id: str, stream_id: str, refusal: AdmissionRefused
) -> None:
    """Background kickoffs (channel poll, live autoclip, recovery) have no
    HTTP response to carry a refusal: surface it on the stream's progress
    card as `pipeline.failed` with the user-facing message. Never raises."""
    from chalybclip.events import emit
    from chalybclip.tenancy import bound_tenant

    try:
        with bound_tenant(tenant_id):
            await emit(
                db,
                "pipeline.failed",
                {
                    "stream_id": stream_id,
                    "error_type": "AdmissionRefused",
                    "reason": refusal.reason,
                    "error": refusal.user_message,
                },
            )
    except Exception:  # noqa: BLE001
        _log.exception("refusal event write failed · stream=%s", stream_id)


async def settle_kickoff(kickoff: Any, outcome: str) -> None:
    """Settle an admitted kickoff from outside its runner — a dispatch
    that was deduped (cancelled) or died before the runner started
    (failed). No-op for kickoffs without a reservation. Never raises."""
    reservation_id = getattr(kickoff, "reservation_id", None)
    if not reservation_id:
        return
    from chalybclip.settings import resolve_db_target

    try:
        db = Database(resolve_db_target(get_settings()))
        await db.connect()
        try:
            await settle_job(
                db, tenant_id=kickoff.tenant_id,
                job_id=getattr(kickoff, "usage_job_id", None),
                reservation_id=reservation_id, outcome=outcome,
            )
        finally:
            await db.close()
    except Exception:  # noqa: BLE001
        _log.exception("settle_kickoff failed · stream=%s", kickoff.stream.id)
