"""Process-wide heavy-op governor (chalybclip.resources)."""

from __future__ import annotations

import asyncio

import pytest

from chalybclip import resources
from chalybclip.resources import heavy_slot, heavy_slots


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHALYBCLIP_HEAVY_SLOTS", "5")
    assert heavy_slots() == 5


def test_invalid_env_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHALYBCLIP_HEAVY_SLOTS", "not-an-int")
    assert heavy_slots() >= 2


def test_zero_or_negative_env_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHALYBCLIP_HEAVY_SLOTS", "0")
    assert heavy_slots() >= 2


def test_default_has_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHALYBCLIP_HEAVY_SLOTS", raising=False)
    assert heavy_slots() >= 2


def test_governor_caps_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    """With 2 slots and 8 contending workers, no more than 2 run at once."""
    monkeypatch.setenv("CHALYBCLIP_HEAVY_SLOTS", "2")
    resources.reset_for_testing()

    # Single-threaded asyncio — no await sits between the increment and the
    # peak read, so plain counters are race-free.
    state = {"active": 0, "peak": 0}

    async def worker() -> None:
        async with heavy_slot():
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            await asyncio.sleep(0.02)
            state["active"] -= 1

    async def run() -> None:
        await asyncio.gather(*(worker() for _ in range(8)))

    asyncio.run(run())
    assert state["peak"] == 2


def test_reusable_across_event_loops(monkeypatch: pytest.MonkeyPatch) -> None:
    """A module-level semaphore would bind to the first loop and raise on
    reuse; the per-loop cache must let successive asyncio.run calls work."""
    monkeypatch.setenv("CHALYBCLIP_HEAVY_SLOTS", "1")
    resources.reset_for_testing()

    async def once() -> bool:
        async with heavy_slot():
            return True

    assert asyncio.run(once()) is True
    assert asyncio.run(once()) is True
