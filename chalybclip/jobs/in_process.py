"""InProcessJobDispatcher — runs the pipeline on this host via
FastAPI's BackgroundTasks. Default for local dev + single-host deploys.

NOT durable: if the dashboard process dies mid-run, the job is lost.
The pipeline emits step-event rows along the way so the dashboard can
detect "abandoned" runs and show the operator a re-run button — that's
the current best we can do without a real queue.

Concurrency IS bounded: a single global semaphore gates how many heavy
pipelines (download + Whisper + ffmpeg render) execute at once, so a
redeploy-recovery backlog or a channel-poll burst can't stampede the box.
Excess launches queue on the semaphore and run as slots free.

For the chalyb deployment, replace this with `ModalJobDispatcher`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

from .active import active_stream_ids, register, unregister
from .base import JobDispatcher, PipelineKickoff, PipelineRunner

if TYPE_CHECKING:
    from fastapi import BackgroundTasks

_log = structlog.get_logger(__name__)


class InProcessJobDispatcher(JobDispatcher):
    """Wraps a PipelineRunner so the API layer doesn't reach into
    `app.state.pipeline_runner` directly anymore.

    `max_concurrency` bounds simultaneous pipeline execution across EVERY
    launch path (API, channel-poll, recovery sweep) — they all funnel through
    `dispatch_pipeline`. Defaults to `Settings.max_concurrent_pipelines`.
    """

    def __init__(
        self, runner: PipelineRunner, *, max_concurrency: int | None = None
    ) -> None:
        self._runner = runner
        if max_concurrency is None:
            from chalybclip.settings import get_settings

            max_concurrency = int(
                getattr(get_settings(), "max_concurrent_pipelines", 2) or 2
            )
        self._sem = asyncio.Semaphore(max(1, max_concurrency))

    @property
    def name(self) -> str:
        return "in-process"

    async def _guarded_run(self, kickoff: PipelineKickoff) -> None:
        """Execute the runner under the global concurrency semaphore. A
        launch over the cap simply waits here for a slot — it holds a cheap
        coroutine, not download/render resources."""
        async with self._sem:
            await self._runner(kickoff)

    async def _run_registered(self, kickoff: PipelineKickoff) -> None:
        """Run under the semaphore, releasing the `jobs.active` registration
        taken at dispatch time. The registration deliberately spans the wait
        for a slot: a run queued for hours behind a multi-hour VOD is in
        flight, not orphaned, and the recovery sweeper / disk reclaimers key
        on that registry."""
        try:
            await self._guarded_run(kickoff)
        finally:
            unregister(kickoff.stream.id)

    async def dispatch_pipeline(
        self,
        kickoff: PipelineKickoff,
        *,
        background_tasks: BackgroundTasks | None = None,
    ) -> None:
        stream_id = kickoff.stream.id
        # One run per stream at a time. A second launch while one is queued
        # or running (double-clicked rerun, recovery sweep racing the
        # channel poller) would execute CONCURRENTLY once slots free —
        # two yt-dlp processes writing the same source/*.part is exactly
        # the fragment corruption the ingest self-heal exists to clean up.
        # Steps are idempotent between runs, not against a parallel twin.
        if stream_id in active_stream_ids():
            _log.info(
                "jobs.dispatch.skipped_in_flight",
                stream_id=stream_id,
                reason="a pipeline run for this stream is already queued or "
                       "running in this process",
            )
            return
        register(stream_id)
        if background_tasks is not None:
            # Normal API path — defer until after the response is sent. The
            # semaphore still applies once the background task runs.
            background_tasks.add_task(self._run_registered, kickoff)
            return
        # CLI / recovery / test path — run inline (callers wrap in a task).
        await self._run_registered(kickoff)
