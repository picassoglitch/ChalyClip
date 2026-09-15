"""Pipeline job dispatch — slice F.8.

The API layer used to call `app.state.pipeline_runner` directly with
FastAPI's `BackgroundTasks`. That's fine for single-host dev, but
when the chalyb deployment lands we'll want pipeline runs to fan
out to Modal (serverless GPU) instead.

`JobDispatcher` is the abstraction that lets us flip targets with a
config setting. Today: `InProcessJobDispatcher`. F.10+: `ModalJobDispatcher`.

The shipping path-by-stage:
    Stage 1 (local dev, single host)     → in_process
    Stage 2 (chalyb deploy on Vercel)   → modal (Vercel can't run GPU)
    Stage 3 (AWS at scale)               → sqs    (queue + ECS workers)
"""

from __future__ import annotations

from .active import active_stream_ids, pipeline_active
from .base import JobDispatcher, PipelineKickoff, PipelineRunner
from .factory import get_dispatcher
from .in_process import InProcessJobDispatcher
from .modal import ModalJobDispatcher

__all__ = [
    "InProcessJobDispatcher",
    "JobDispatcher",
    "ModalJobDispatcher",
    "PipelineKickoff",
    "PipelineRunner",
    "active_stream_ids",
    "get_dispatcher",
    "pipeline_active",
]
