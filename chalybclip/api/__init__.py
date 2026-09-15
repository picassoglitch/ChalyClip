"""FastAPI surface for ChalybClip.

Phase 1 ships:
    * Bearer-token auth (api_tokens table, sha256-hashed at rest).
    * Tenant-scoped routes for streams, candidates, clips, personas,
      publish jobs, llm calls.
    * One process_vod kickoff endpoint (the actual pipeline runs as a
      FastAPI BackgroundTask in-process; Phase 3 swaps in a worker pool).

The HTMX dashboard (Task 10) mounts on top of this same app and reuses
every dependency here.
"""

from ._pipeline import PipelineKickoff, PipelineRunner
from .app import create_app

__all__ = ["PipelineKickoff", "PipelineRunner", "create_app"]
