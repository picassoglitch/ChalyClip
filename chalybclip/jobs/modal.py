"""ModalJobDispatcher — runs the pipeline on the Modal worker app (Phase 2b).

The web box stops executing `process_vod` in-process and instead POSTs the
`PipelineKickoff` to `infra/modal_pipeline_app.py`'s web endpoint (same
bearer-token + 303-poll protocol the whisper provider uses). The worker
runs the SAME `default_pipeline_runner` against the shared Postgres + R2
bucket, so step events, failure surfacing, cost tracking and artifact
offload behave exactly like an in-process run — the dashboard can't tell
the difference, except the web box's CPU stays flat.

`jobs.active` semantics are preserved: the stream registers at dispatch
time and unregisters when the remote run reaches a terminal state, so the
recovery sweeper's and disk reclaimers' in-flight checks keep working. If
THIS process dies mid-poll the registry resets (it's in-memory — same as
in-process), while the Modal run keeps going and keeps writing step
events; the recovery sweeper's event-silence rules take over from there.

Sources that exist only on this box (`upload://`, `live://` pseudo-URLs)
can't run remotely — they route to the wrapped in-process fallback.
Concurrency for remote runs is bounded on the Modal side
(`max_containers`), not by queueing here.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from chalybclip.errors import ChalybClipError
from chalybclip.integrations.modal_http import (
    ModalPollDeadlineError,
    poll_until_terminal,
)

from .active import active_stream_ids, register, unregister
from .base import JobDispatcher, PipelineKickoff

if TYPE_CHECKING:
    from fastapi import BackgroundTasks

_log = structlog.get_logger(__name__)

# Per-request HTTP timeout. Modal answers the initial POST with the 303
# polling redirect within seconds-to-a-couple-minutes; each poll GET is
# instant. The multi-hour RUN duration is bounded by the poll deadline
# (`modal_pipeline_timeout_s`), not by this.
_REQUEST_TIMEOUT_S = 300.0

# Poll cadence. Pipeline runs take minutes-to-hours, so a slower tick than
# whisper's 5s keeps a 6h run at ~1400 polls instead of ~4300.
_POLL_INTERVAL_S = 15.0


class ModalJobDispatcher(JobDispatcher):
    """Dispatches pipeline runs to the Modal worker app.

    Constructor raises `ChalybClipError` when the endpoint/token aren't
    configured — `create_app`'s defensive boot catches that and falls
    back to in-process, so a half-configured deploy still serves.
    """

    def __init__(
        self,
        *,
        endpoint_url: str | None = None,
        bearer_token: str | None = None,
        timeout_s: float | None = None,
        poll_interval_s: float = _POLL_INTERVAL_S,
        fallback: JobDispatcher | None = None,
    ) -> None:
        from chalybclip.settings import get_settings

        settings = get_settings()
        self._endpoint_url = (
            endpoint_url
            or getattr(settings, "modal_pipeline_endpoint_url", None)
            or ""
        ).strip().rstrip("/")
        self._bearer_token = (
            bearer_token or getattr(settings, "modal_token", None) or ""
        ).strip()
        if not self._endpoint_url or not self._bearer_token:
            raise ChalybClipError(
                "ModalJobDispatcher misconfigured: set "
                "CHALYBCLIP_MODAL_PIPELINE_ENDPOINT_URL (from `modal deploy "
                "infra/modal_pipeline_app.py`) and CHALYBCLIP_MODAL_TOKEN, "
                "or set CHALYBCLIP_JOB_DISPATCHER=in_process."
            )
        self._timeout_s = float(
            timeout_s
            if timeout_s is not None
            else getattr(settings, "modal_pipeline_timeout_s", 21600.0)
        )
        self._poll_interval_s = poll_interval_s
        self._fallback = fallback
        # Keep strong refs to the poller tasks — a bare create_task result
        # that goes out of scope can be garbage-collected mid-flight.
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def name(self) -> str:
        return "modal"

    async def drain(self) -> None:
        """Await every in-flight poller (tests + graceful shutdown)."""
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def dispatch_pipeline(
        self,
        kickoff: PipelineKickoff,
        *,
        background_tasks: BackgroundTasks | None = None,
    ) -> None:
        stream_id = kickoff.stream.id
        vod_url = str(getattr(kickoff.stream, "vod_url", "") or "")

        # upload:// and live:// sources exist only on this box's disk —
        # the worker can't ingest them. Run those in-process (they're the
        # minority; URL VODs carry the measured CPU load).
        if not vod_url.startswith(("http://", "https://")):
            if self._fallback is not None:
                _log.info(
                    "jobs.modal.fallback_local",
                    stream_id=stream_id,
                    reason="non-http source can't be ingested remotely",
                )
                await self._fallback.dispatch_pipeline(
                    kickoff, background_tasks=background_tasks
                )
                return
            raise ChalybClipError(
                f"ModalJobDispatcher can't run non-http source {vod_url!r} "
                "remotely and no in-process fallback is wired."
            )

        # Same dedup as in-process: one run per stream at a time, whether
        # it's queued locally or executing remotely.
        if stream_id in active_stream_ids():
            _log.info(
                "jobs.dispatch.skipped_in_flight",
                stream_id=stream_id,
                reason="a pipeline run for this stream is already queued or "
                       "running (locally or on the worker)",
            )
            return
        register(stream_id)
        task = asyncio.create_task(self._run_remote(kickoff))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_remote(self, kickoff: PipelineKickoff) -> None:
        """POST the kickoff to the worker + poll to a terminal state.

        Holds the `jobs.active` registration for the whole remote run.
        Failure surfacing is split by what we actually know:
          * dispatch/HTTP errors (Modal down, bad token, function crash
            before the runner's own try/except) → the run never produced
            its own failure event, so WE emit `pipeline.failed`;
          * a terminal body with status="failed" → the worker's runner
            already emitted the event; just log;
          * poll deadline → the run may well still be alive and writing
            events, so do NOT mark it failed — log and stop tracking.
        """
        stream_id = kickoff.stream.id
        try:
            payload: dict[str, Any] = {
                "auth_token": self._bearer_token,
                "tenant_id": kickoff.tenant_id,
                "persona_id": kickoff.persona_id,
                "language": kickoff.language,
                "stream": kickoff.stream.model_dump(mode="json"),
            }
            _log.info(
                "jobs.modal.dispatch",
                stream_id=stream_id,
                tenant_id=kickoff.tenant_id,
                endpoint_host=(
                    self._endpoint_url.split("/")[2]
                    if "//" in self._endpoint_url else "?"
                ),
            )
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_S) as client:
                resp = await client.post(self._endpoint_url, json=payload)
                resp = await poll_until_terminal(
                    client=client,
                    initial=resp,
                    deadline_s=self._timeout_s,
                    poll_interval_s=self._poll_interval_s,
                    label="modal pipeline",
                    ref=stream_id,
                )
                resp.raise_for_status()
                result: dict[str, Any] = resp.json()
            status = str(result.get("status") or "?")
            _log.info(
                "jobs.modal.terminal",
                stream_id=stream_id,
                status=status,
                n_clips=result.get("n_clips"),
                error=(str(result.get("error") or "")[:200] or None),
            )
        except ModalPollDeadlineError as e:
            # Lost tracking, not (necessarily) a lost run — the worker
            # keeps writing step events to the shared DB and the recovery
            # sweeper's silence rules own it from here.
            _log.warning(
                "jobs.modal.poll_deadline",
                stream_id=stream_id,
                error=str(e),
            )
        except httpx.HTTPStatusError as e:
            body = (e.response.text or "")[:300]
            _log.error(
                "jobs.modal.dispatch_failed",
                stream_id=stream_id,
                status=e.response.status_code,
                body=body,
            )
            await self._emit_failure(
                kickoff,
                error=(
                    f"modal pipeline returned HTTP "
                    f"{e.response.status_code}: {body}"
                ),
            )
        except Exception as e:
            _log.error(
                "jobs.modal.dispatch_failed",
                stream_id=stream_id,
                error=str(e),
            )
            await self._emit_failure(
                kickoff, error=f"modal pipeline dispatch error: {e}"
            )
        finally:
            unregister(stream_id)

    async def _emit_failure(
        self, kickoff: PipelineKickoff, *, error: str
    ) -> None:
        """Best-effort `pipeline.failed` event so the dashboard's progress
        card explains the dead run instead of spinning. Mirrors the shape
        `default_pipeline_runner` writes for in-process failures."""
        try:
            from chalybclip.db import Database
            from chalybclip.events import emit
            from chalybclip.settings import get_settings, resolve_db_target
            from chalybclip.tenancy import bound_tenant

            db = Database(resolve_db_target(get_settings()))
            await db.connect()
            try:
                with bound_tenant(kickoff.tenant_id):
                    await emit(
                        db,
                        "pipeline.failed",
                        {
                            "stream_id": kickoff.stream.id,
                            "error_type": "ModalDispatchError",
                            "error": error[:500],
                        },
                    )
            finally:
                await db.close()
        except Exception:  # observability only — never raise past here
            _log.exception(
                "jobs.modal.failure_event_write_failed",
                stream_id=kickoff.stream.id,
            )
