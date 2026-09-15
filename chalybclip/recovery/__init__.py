"""Pipeline recovery — re-dispatch orphaned in-process ingest jobs.

The in-process dispatcher drops background tasks on a process restart,
freezing URL-ingest streams at `pending`. This module's sweeper finds
those orphans and re-runs them (idempotently) so intakes survive deploys
and crashes. Started by the API lifespan; see `service` for the rules.
"""

from __future__ import annotations

from .service import (
    RECOVERY_DISPATCHED,
    OrphanDecision,
    classify_streams,
    recover_orphaned_pipelines,
)

__all__ = [
    "RECOVERY_DISPATCHED",
    "OrphanDecision",
    "classify_streams",
    "recover_orphaned_pipelines",
]
