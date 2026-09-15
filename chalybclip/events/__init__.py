"""Event log — append-only `events` table writes for every state transition.

Phase 1 just persists the events; webhook fan-out to subscribers lands in
Phase 2. The event_type strings are stable across phases — once a type is
emitted it can't be renamed without breaking historical replay.
"""

from .log import (
    CLIP_CUT_COMPLETED,
    CLIP_CUT_STARTED,
    CLIP_CUT_SUBSTEP,
    CLIP_READY_FOR_REVIEW,
    LLM_EXHAUSTED,
    LLM_FALLBACK,
    STREAM_AUDIO_EXTRACTED,
    STREAM_CREATED,
    STREAM_DOWNLOAD_COMPLETED,
    STREAM_DOWNLOAD_STARTED,
    STREAM_PROCESSED,
    emit,
)

__all__ = [
    "CLIP_CUT_COMPLETED",
    "CLIP_CUT_STARTED",
    "CLIP_CUT_SUBSTEP",
    "CLIP_READY_FOR_REVIEW",
    "LLM_EXHAUSTED",
    "LLM_FALLBACK",
    "STREAM_AUDIO_EXTRACTED",
    "STREAM_CREATED",
    "STREAM_DOWNLOAD_COMPLETED",
    "STREAM_DOWNLOAD_STARTED",
    "STREAM_PROCESSED",
    "emit",
]
