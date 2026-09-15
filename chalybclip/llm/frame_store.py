"""FrameStore protocol — pluggable backend for the multimodal frame cache.

Phase 2 ships only the in-memory `MemoryFrameStore` (the existing
`FrameCache` rebadged behind the protocol). Phase 3 will add an S3-backed
implementation - same surface, the worker pool reads from object storage
instead of process-local memory.

Callers (vision rescore, vision smart crop / thumb pickers) accept the
protocol; they never need to know which backend they're talking to.
"""

from __future__ import annotations

from typing import Protocol

from .frame_cache import FrameCache


class FrameStore(Protocol):
    """Minimum surface every frame-cache backend must expose."""

    def get(self, stream_id: str, ts: float) -> bytes | None: ...

    def put(self, stream_id: str, ts: float, blob: bytes) -> None: ...

    def clear(self) -> None: ...

    def __len__(self) -> int: ...


# `FrameCache` already implements every method on the protocol; we re-export
# it here under the protocol-oriented name so new code uses the cleaner
# "MemoryFrameStore" reading order. `FrameCache` stays as-is for the existing
# imports.
MemoryFrameStore = FrameCache
