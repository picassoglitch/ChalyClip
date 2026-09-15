"""`emit()` — the canonical way to record one state transition.

Every state change in the pipeline lands as one row in the `events` table.
Phase 1 just writes the rows; Phase 2 will dispatch them to webhook
subscribers from the same source.

Behavior:
  * `db is None` → no-op. Phase 0-style invocations (no DB) silently skip.
  * Tenant comes from `current_tenant_id()` — never passed by the caller.
    If no tenant is bound, the write fails inside the repo and the failure
    is swallowed (events are observability, not correctness).
  * Failures are logged via structlog at WARNING but never propagate; an
    LLM call or pipeline step must not break because the audit row failed.

Event-type strings are stable across phases. Reserved types reflected here
appear in `__init__.py` as constants so callers don't typo them. Phase 1
emits `stream.created`, `clip.ready_for_review`, `stream.processed`,
`llm.fallback`, `llm.exhausted`. Phase 2+ adds `clip.approved`,
`clip.published`, `clip.failed`, `publish_job.failed`.
"""

from __future__ import annotations

from typing import Any

from chalybclip.db import Database, EventsRepo
from chalybclip.logging import get_logger

# Canonical event types Phase 1 emits.
STREAM_CREATED = "stream.created"
STREAM_PROCESSED = "stream.processed"
CLIP_READY_FOR_REVIEW = "clip.ready_for_review"
LLM_FALLBACK = "llm.fallback"
LLM_EXHAUSTED = "llm.exhausted"

# Task 1d — per-candidate cut progress. The dashboard turns these
# into a substep label ("Sampling… / Cropping… / Encoding… /
# Branding…") plus an X/N counter so the operator stops staring
# at an opaque "Cut (running…)" for minutes.
CLIP_CUT_STARTED = "clip.cut.started"
CLIP_CUT_SUBSTEP = "clip.cut.substep"
CLIP_CUT_COMPLETED = "clip.cut.completed"

# Task 2a — VOD ingest substeps. Splits the opaque "ingesting" row
# into download + audio-extract phases with wall-clock + throughput
# in the payload. The dashboard surfaces this as
# "Downloading… X MB at Y Mbps via yt-dlp" → "Extracting audio…"
# so the operator stops wondering whether a 12-minute "ingest" is
# stuck or just chewing through a 4 GB source.
STREAM_DOWNLOAD_STARTED = "stream.download.started"
STREAM_DOWNLOAD_COMPLETED = "stream.download.completed"
STREAM_AUDIO_EXTRACTED = "stream.audio_extracted"

_log = get_logger("chalybclip.events")


async def emit(
    db: Database | None,
    type_: str,
    payload: dict[str, Any] | None = None,
) -> None:
    """Append one event row. Best-effort: never raises into the caller.

    Args:
        db: The Database, or None for filesystem-only runs (no-op).
        type_: One of the canonical type constants in this module.
        payload: Optional structured fields (clip_id, etc.) — JSON-serialized
            into `events.payload_json`. Keep it small and primitive-typed.
    """
    if db is None:
        return
    try:
        await EventsRepo(db).emit(type=type_, payload=payload)
    except Exception as e:
        # Mirror the LLMRouter pattern: events are observability, not
        # correctness, so a write failure must not propagate.
        _log.warning("event.emit.failed", event_type=type_, error=str(e))
