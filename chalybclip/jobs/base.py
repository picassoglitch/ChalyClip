"""JobDispatcher Protocol + PipelineKickoff dataclass.

`PipelineKickoff` is the request envelope every dispatcher accepts.
It used to live in `chalybclip.api._pipeline`; moved here so non-API
callers (CLI, future workers, tests) can share the same shape.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fastapi import BackgroundTasks

    from chalybclip.ingest import Stream


@dataclass(frozen=True)
class PipelineKickoff:
    """Inputs handed to a JobDispatcher after ingest. The dispatcher
    decides where the rest of the pipeline (transcribe → detect → cut →
    variants) actually runs — same host, Modal, or an SQS-backed worker.
    """

    tenant_id: str
    stream: Stream
    persona_id: str
    output_dir: Path
    language: str | None = None


# The "actual runner" type — a coroutine that does the real work.
# Today: `process_vod` from chalybclip.pipeline. Tomorrow: a Modal
# `app.function` invocation, or an SQS publish.
PipelineRunner = Callable[[PipelineKickoff], Awaitable[None]]


@runtime_checkable
class JobDispatcher(Protocol):
    """Dispatches one `PipelineKickoff` to wherever the pipeline runs.

    Implementations are responsible for:
      - durability (or lack of) — in-process drops on crash; cloud
        impls should persist before returning
      - kicking off the work without blocking the API response
      - returning quickly (no awaiting the actual pipeline run)
    """

    async def dispatch_pipeline(
        self,
        kickoff: PipelineKickoff,
        *,
        background_tasks: BackgroundTasks | None = None,
    ) -> None:
        """Schedule the pipeline. Returns when the work is *queued*,
        not when it finishes.

        `background_tasks` is the FastAPI helper; in-process dispatchers
        use it to defer work until after the HTTP response is sent.
        Cloud dispatchers ignore it (they enqueue + return immediately).
        """
        ...

    @property
    def name(self) -> str:
        """Short identifier for logs + telemetry, e.g. "in-process" or "modal"."""
        ...
