"""Process-wide governor for heavy OS-level work (ffmpeg, OpenCV decode,
headless Chromium).

Without a cap, three subsystems peak independently inside one process:
the clip-cut fan-out (`clip.service.cut_clips`), operator-triggered clip
renders (`api._clip_render.render_clip_in_background`), and the channel-poll
pipeline that runs the cut step inline. When they overlap they exhaust the
container's thread/PID/memory ceiling, and every subsequent fork /
pthread_create fails with EAGAIN — surfacing as decoder-open failures,
`x264_encoder_open` errors, and "can't start new thread". This module bounds
the number of concurrent heavy operations so a spike degrades into queueing
instead of a cascade of EAGAIN failures.

Sizing: `CHALYBCLIP_HEAVY_SLOTS` overrides; otherwise half the CPU count
(floor `_SLOTS_FLOOR`). Heavy ops are memory-bound on our containers (each
ffmpeg / Chromium is hundreds of MB), so the default errs low.

The semaphore is created lazily per running event loop so the module can be
imported once and reused across the many short-lived loops that tests spin
up with `asyncio.run` — asyncio primitives bind to the first loop that
touches them and raise if reused across loops.
"""

from __future__ import annotations

import asyncio
import os
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from chalybclip.logging import get_logger

_log = get_logger("chalybclip.resources")

_SLOTS_FLOOR = 2
_ENV_VAR = "CHALYBCLIP_HEAVY_SLOTS"

# Per-loop semaphores, keyed weakly on the loop so finished test loops get
# collected and each fresh loop gets a fresh semaphore.
_sems: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, asyncio.Semaphore
] = weakref.WeakKeyDictionary()


def heavy_slots() -> int:
    """Resolve the concurrent-heavy-op cap for this process.

    `CHALYBCLIP_HEAVY_SLOTS` (>=1) wins; otherwise half the CPU count with a
    floor of `_SLOTS_FLOOR`. Read fresh each time a loop's semaphore is
    created so tests can pin it via the environment.
    """
    raw = os.environ.get(_ENV_VAR)
    if raw:
        try:
            n = int(raw)
        except ValueError:
            n = 0
        if n >= 1:
            return n
    return max(_SLOTS_FLOOR, (os.cpu_count() or _SLOTS_FLOOR) // 2)


def _semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _sems.get(loop)
    if sem is None:
        n = heavy_slots()
        sem = asyncio.Semaphore(n)
        _sems[loop] = sem
        _log.info("resources.governor_init", slots=n)
    return sem


@asynccontextmanager
async def heavy_slot() -> AsyncIterator[None]:
    """Hold one heavy-op slot for the duration of the block.

    Every CPU/IO-heavy subprocess or browser launch (an ffmpeg encode, an
    OpenCV decode batch, a headless-Chromium render) should run inside this
    so the process can't oversubscribe the host. A blocked acquire simply
    awaits a free slot — it never fails or times out.
    """
    sem = _semaphore()
    async with sem:
        yield


def reset_for_testing() -> None:
    """Drop cached semaphores so the next acquire re-reads the environment.

    Tests that tweak `CHALYBCLIP_HEAVY_SLOTS` within a single event loop call
    this; the common case (one `asyncio.run` per test) gets a fresh loop and
    therefore a fresh semaphore without needing it.
    """
    _sems.clear()


__all__ = ["heavy_slot", "heavy_slots", "reset_for_testing"]
