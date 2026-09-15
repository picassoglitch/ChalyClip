"""PC worker — the full VOD pipeline on operator-owned hardware.

The self-hosted twin of `infra/modal_pipeline_app.py`, speaking the exact
HTTP contract `ModalJobDispatcher` already implements, so Railway needs
ZERO code changes to use it — just point
`CHALYBCLIP_MODAL_PIPELINE_ENDPOINT_URL` at this app's (tunneled) URL:

    POST /            {auth_token, tenant_id, persona_id, language, stream}
        → 303 See Other, Location: /jobs/{job_id}
    GET  /jobs/{id}   → 303 (self) while running     [poll_until_terminal]
                      → 200 {stream_id, status: done|failed, error,
                             n_clips, elapsed_s} when finished

Why hold jobs behind the redirect-poll instead of the open POST: pipeline
runs take minutes-to-hours and tunnels/proxies kill idle requests long
before that; the poll loop is the protocol the dispatcher already speaks
for Modal.

The worker runs `default_pipeline_runner` — the same coroutine the web box
and the Modal app use — against the shared Postgres (step events, rows,
cost tracking, failure surfacing) and R2 (artifact offload), so the
dashboard can't tell where a run happened. On this box the expensive parts
are free: faster-whisper on the local GPU (CHALYBCLIP_TRANSCRIBE_PROVIDER=
local) and hooks/detection against the local open-LLM runtime.

Run it:  `chalybclip worker --port 8100`  (see the CLI command), then front
it with a tunnel (Cloudflare Tunnel / Tailscale Funnel) so Railway can
reach it.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse

from chalybclip.jobs.base import PipelineKickoff, PipelineRunner
from chalybclip.settings import get_settings

_log = structlog.get_logger(__name__)

# Completed-job results are kept in memory for the dispatcher to collect;
# cap the ledger so a long-lived worker doesn't grow unbounded.
_MAX_FINISHED_JOBS = 200


@dataclass
class _Job:
    stream_id: str
    started_at: float
    task: asyncio.Task[None] | None = None
    result: dict[str, Any] | None = None  # None while running


@dataclass
class _JobLedger:
    jobs: dict[str, _Job] = field(default_factory=dict)
    counter: int = 0

    def new_id(self, stream_id: str) -> str:
        self.counter += 1
        return f"job_{stream_id}_{self.counter}"

    def active_stream_ids(self) -> set[str]:
        return {j.stream_id for j in self.jobs.values() if j.result is None}

    def prune(self) -> None:
        finished = [jid for jid, j in self.jobs.items() if j.result is not None]
        for jid in finished[: max(0, len(finished) - _MAX_FINISHED_JOBS)]:
            self.jobs.pop(jid, None)


def _preflight_error() -> str | None:
    """Refuse runs whose results would be invisible to the dashboard.

    Same two hard requirements as the Modal worker: shared Postgres (else
    rows/events land in a local SQLite the web box never sees) and the R2
    bucket (else clips exist only on this box's disk)."""
    if not (
        os.environ.get("DATABASE_URL") or os.environ.get("CHALYBCLIP_DATABASE_URL") or ""
    ).strip():
        return (
            "worker misconfigured: DATABASE_URL is not set — the run's "
            "rows/events would be written to a worker-local SQLite file "
            "the dashboard never sees"
        )
    if not (os.environ.get("CHALYBCLIP_OBJECT_STORAGE_BUCKET") or "").strip():
        return (
            "worker misconfigured: CHALYBCLIP_OBJECT_STORAGE_BUCKET is not "
            "set — cut clips would exist only on this machine"
        )
    return None


def _expected_token() -> str:
    return (
        os.environ.get("CHALYBCLIP_WORKER_TOKEN")
        or os.environ.get("CHALYBCLIP_MODAL_TOKEN")
        or ""
    ).strip()


def _count_clips_from_manifest(stream_dir: Path) -> int | None:
    try:
        manifest = json.loads(
            (stream_dir / "manifest.json").read_text(encoding="utf-8")
        )
        clips = manifest.get("clip_entries")
        return len(clips) if isinstance(clips, list) else None
    except Exception:
        return None


def create_worker_app(*, runner: PipelineRunner | None = None) -> FastAPI:
    """Build the worker app. `runner` is injectable for tests; the default
    is the production `default_pipeline_runner`."""
    # Settings binds database_url to the un-prefixed DATABASE_URL alias, so a
    # worker configured only with CHALYBCLIP_DATABASE_URL would pass preflight
    # yet resolve_db_target() would silently fall back to a worker-local
    # SQLite. Normalize before anything instantiates Settings.
    if not (os.environ.get("DATABASE_URL") or "").strip():
        _prefixed_db_url = (os.environ.get("CHALYBCLIP_DATABASE_URL") or "").strip()
        if _prefixed_db_url:
            os.environ["DATABASE_URL"] = _prefixed_db_url
            get_settings.cache_clear()
    app = FastAPI(title="chalybclip-pc-worker", docs_url=None, redoc_url=None)
    ledger = _JobLedger()
    # One box, one GPU: run pipelines one at a time by default (a second
    # concurrent run starves both — CV passes, whisper and ffmpeg all fight
    # for the same cores). Excess jobs queue on the semaphore; their poll
    # URLs keep 303-ing, which the dispatcher's poll loop already tolerates
    # for hours. CHALYBCLIP_WORKER_MAX_JOBS raises the cap on beefier boxes.
    max_jobs = max(1, int(os.environ.get("CHALYBCLIP_WORKER_MAX_JOBS", "1") or 1))
    run_slots = asyncio.Semaphore(max_jobs)

    async def _default_runner(kickoff: PipelineKickoff) -> None:
        from chalybclip.api._pipeline import default_pipeline_runner

        await default_pipeline_runner(kickoff)

    run = runner or _default_runner

    async def _execute(job_id: str, kickoff: PipelineKickoff) -> None:
        job = ledger.jobs[job_id]
        started = job.started_at
        stream_id = kickoff.stream.id
        try:
            async with run_slots:
                _log.info(
                    "worker.slot_acquired", stream_id=stream_id, job_id=job_id,
                    waited_s=round(time.time() - started, 1),
                )
                await run(kickoff)
        except Exception as e:
            # The runner already emitted pipeline.failed to the shared DB;
            # a failed body (HTTP 200 at the poll) tells the dispatcher to
            # log rather than double-emit — same contract as Modal.
            _log.error(
                "worker.run_failed", stream_id=stream_id,
                error=f"{type(e).__name__}: {e}",
            )
            job.result = {
                "stream_id": stream_id,
                "status": "failed",
                "error": f"{type(e).__name__}: {str(e)[:500]}",
                "elapsed_s": time.time() - started,
            }
            return
        n_clips = _count_clips_from_manifest(kickoff.output_dir / stream_id)
        _log.info(
            "worker.run_done", stream_id=stream_id, n_clips=n_clips,
            elapsed_s=round(time.time() - started, 1),
        )
        job.result = {
            "stream_id": stream_id,
            "status": "done",
            "error": None,
            "n_clips": n_clips,
            "elapsed_s": time.time() - started,
        }

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse(
            {
                "ok": _preflight_error() is None,
                "active_jobs": len(ledger.active_stream_ids()),
            }
        )

    @app.post("/", response_model=None)
    async def submit(request: Request) -> Response:
        payload = await request.json()
        expected = _expected_token()
        if not expected or (payload or {}).get("auth_token", "") != expected:
            raise HTTPException(status_code=401, detail="invalid bearer token")

        preflight = _preflight_error()
        if preflight is not None:
            raise HTTPException(status_code=500, detail=preflight)

        tenant_id = str((payload or {}).get("tenant_id") or "").strip()
        persona_id = str((payload or {}).get("persona_id") or "").strip()
        language = (payload or {}).get("language") or None
        stream_raw = (payload or {}).get("stream") or {}
        if not tenant_id or not persona_id or not stream_raw:
            raise HTTPException(
                status_code=400,
                detail="tenant_id, persona_id and stream are required",
            )

        from chalybclip.ingest import Stream
        from chalybclip.settings import get_settings

        try:
            stream = Stream.model_validate(stream_raw)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"stream payload invalid: {e}"
            ) from e

        # One run per stream at a time — the dispatcher dedupes on its side
        # too, but a retried POST after a lost response must not double-run.
        for job_id, job in ledger.jobs.items():
            if job.stream_id == stream.id and job.result is None:
                _log.info(
                    "worker.dedup_redirect", stream_id=stream.id, job_id=job_id
                )
                return RedirectResponse(
                    url=f"/jobs/{job_id}?t={expected}", status_code=303
                )

        output_dir = Path(get_settings().default_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        kickoff = PipelineKickoff(
            tenant_id=tenant_id,
            stream=stream,
            persona_id=persona_id,
            output_dir=output_dir,
            language=language,
        )

        job_id = ledger.new_id(stream.id)
        ledger.jobs[job_id] = _Job(stream_id=stream.id, started_at=time.time())
        task = asyncio.create_task(_execute(job_id, kickoff))
        ledger.jobs[job_id].task = task
        ledger.prune()
        _log.info(
            "worker.run_start", stream_id=stream.id, job_id=job_id,
            tenant_id=tenant_id, persona_id=persona_id,
        )
        return RedirectResponse(url=f"/jobs/{job_id}?t={expected}", status_code=303)

    @app.post("/v1/{path:path}", response_model=None)
    async def llm_proxy(path: str, request: Request) -> Response:
        """Bearer-authenticated pass-through to the local open-LLM runtime.

        Exposing Ollama raw through the tunnel would hand the GPU to anyone
        holding the URL; this route makes the worker token the gate. The
        web box sets CHALYBCLIP_OPENLLM_BASE_URL=https://<tunnel>/v1 and
        OPENLLM_API_KEY=<worker token> — the OpenAI-compatible provider
        already sends it as an Authorization: Bearer header."""
        import httpx

        expected = _expected_token()
        auth = request.headers.get("authorization", "")
        if not expected or auth != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="invalid bearer token")
        upstream = (
            os.environ.get("CHALYBCLIP_LOCAL_LLM_URL") or "http://127.0.0.1:11434/v1"
        ).rstrip("/")
        body = await request.body()
        async with httpx.AsyncClient(timeout=600.0) as client:
            r = await client.post(
                f"{upstream}/{path}",
                content=body,
                headers={"Content-Type": "application/json"},
            )
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        )

    @app.get("/jobs/{job_id}", response_model=None)
    async def poll(job_id: str, t: str = "") -> Response:
        # The poll URL carries the token as a query param because the
        # dispatcher's poll loop GETs the Location verbatim (no headers).
        expected = _expected_token()
        if not expected or t != expected:
            raise HTTPException(status_code=401, detail="invalid poll token")
        job = ledger.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        if job.result is None:
            # Still running — redirect to self; the dispatcher's poll loop
            # sleeps between GETs.
            return RedirectResponse(
                url=f"/jobs/{job_id}?t={expected}", status_code=303
            )
        return JSONResponse(job.result)

    return app
