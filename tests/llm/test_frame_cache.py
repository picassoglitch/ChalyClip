"""Tests for the in-memory frame cache used by multimodal calls."""

from __future__ import annotations

import pytest

from chalybclip.llm import FrameCache


def test_get_returns_none_on_miss() -> None:
    cache = FrameCache()
    assert cache.get("str_xyz", 1.0) is None


def test_put_then_get_round_trips() -> None:
    cache = FrameCache()
    cache.put("str_xyz", 5.0, b"\xff\xd8\xffjpeg-bytes")
    assert cache.get("str_xyz", 5.0) == b"\xff\xd8\xffjpeg-bytes"
    assert len(cache) == 1


def test_distinct_streams_dont_collide() -> None:
    cache = FrameCache()
    cache.put("str_a", 1.0, b"a")
    cache.put("str_b", 1.0, b"b")
    assert cache.get("str_a", 1.0) == b"a"
    assert cache.get("str_b", 1.0) == b"b"
    assert len(cache) == 2


def test_ts_rounds_to_millisecond_precision() -> None:
    """Floating-point fuzz at the microsecond level shouldn't miss."""
    cache = FrameCache()
    cache.put("str_x", 1.234, b"frame")
    # Same value -> hit.
    assert cache.get("str_x", 1.234) == b"frame"
    # Sub-millisecond drift on the lookup -> still hits the same bucket.
    assert cache.get("str_x", 1.234001) == b"frame"
    # A different millisecond bucket -> miss.
    assert cache.get("str_x", 1.235) is None


def test_lru_eviction_kicks_out_oldest() -> None:
    cache = FrameCache(max_entries=3)
    cache.put("s", 1.0, b"a")
    cache.put("s", 2.0, b"b")
    cache.put("s", 3.0, b"c")
    # Touch (1.0) so it becomes the most-recent.
    assert cache.get("s", 1.0) == b"a"
    # Now (2.0) is the oldest; inserting a 4th entry should evict it.
    cache.put("s", 4.0, b"d")
    assert cache.get("s", 2.0) is None
    assert cache.get("s", 1.0) == b"a"
    assert cache.get("s", 3.0) == b"c"
    assert cache.get("s", 4.0) == b"d"
    assert len(cache) == 3


def test_clear_drops_all_entries() -> None:
    cache = FrameCache()
    cache.put("s", 0.0, b"x")
    cache.put("s", 1.0, b"y")
    cache.clear()
    assert len(cache) == 0
    assert cache.get("s", 0.0) is None


def test_max_entries_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_entries"):
        FrameCache(max_entries=0)


def test_contains_uses_rounded_ts() -> None:
    cache = FrameCache()
    cache.put("s", 7.0, b"f")
    assert ("s", 7.0) in cache
    assert ("s", 8.0) not in cache
