"""FrameStore protocol conformance + MemoryFrameStore drop-in for FrameCache."""

from __future__ import annotations

from chalybclip.llm import FrameCache, FrameStore, MemoryFrameStore


def test_memory_frame_store_is_alias_for_frame_cache() -> None:
    """The Phase-2 protocol-oriented name and the Phase-1 class are the same type."""
    assert MemoryFrameStore is FrameCache


def test_frame_cache_satisfies_protocol() -> None:
    """Static-ish: a FrameCache instance is assignable to FrameStore."""
    store: FrameStore = FrameCache()
    store.put("str_x", 1.0, b"jpeg")
    assert store.get("str_x", 1.0) == b"jpeg"
    assert store.get("str_x", 2.0) is None
    assert len(store) == 1
    store.clear()
    assert len(store) == 0


def test_protocol_signatures_round_trip() -> None:
    """A handcrafted FrameStore impl works through the same surface."""

    class _Counting:
        def __init__(self) -> None:
            self._d: dict[tuple[str, float], bytes] = {}
            self.gets = 0
            self.puts = 0

        def get(self, stream_id: str, ts: float) -> bytes | None:
            self.gets += 1
            return self._d.get((stream_id, ts))

        def put(self, stream_id: str, ts: float, blob: bytes) -> None:
            self.puts += 1
            self._d[(stream_id, ts)] = blob

        def clear(self) -> None:
            self._d.clear()

        def __len__(self) -> int:
            return len(self._d)

    store: FrameStore = _Counting()
    store.put("s", 1.0, b"x")
    assert store.get("s", 1.0) == b"x"
    assert store.get("s", 2.0) is None
    assert len(store) == 1
