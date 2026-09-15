"""In-memory frame cache for multimodal LLM calls.

Vision-aware variant generation samples a few frames per clip and hands
them to the multimodal router method. When two personas generate variants
for the same clip - or when a retry re-issues the call - we don't want
to spin OpenCV up again to re-decode the same JPEGs from disk.

The cache is keyed by `(stream_id, ts)` where `ts` is the *source-stream*
timestamp (not the clip-relative offset) so cache hits work across calls
that happen to land on the same source moment.

Phase 1 keeps this in-process. Phase 3's worker pool will need a shared
cache (Redis or similar) - the surface stays the same, only the storage
swaps.
"""

from __future__ import annotations

from collections import OrderedDict

# Round timestamps to milliseconds so floating-point drift between samplers
# doesn't punch holes in the cache.
_TS_PRECISION = 3


class FrameCache:
    """LRU-bounded in-memory cache of JPEG-encoded frame bytes."""

    def __init__(self, *, max_entries: int = 256) -> None:
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")
        self._max_entries = max_entries
        self._cache: OrderedDict[tuple[str, float], bytes] = OrderedDict()

    def get(self, stream_id: str, ts: float) -> bytes | None:
        key = (stream_id, round(ts, _TS_PRECISION))
        blob = self._cache.get(key)
        if blob is not None:
            self._cache.move_to_end(key)
        return blob

    def put(self, stream_id: str, ts: float, blob: bytes) -> None:
        key = (stream_id, round(ts, _TS_PRECISION))
        self._cache[key] = blob
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, item: tuple[str, float]) -> bool:
        stream_id, ts = item
        return (stream_id, round(ts, _TS_PRECISION)) in self._cache
