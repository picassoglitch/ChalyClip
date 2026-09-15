"""In-process registry of streams with a pipeline run in flight.

The disk reclaimers — the watchdog loop, the per-ingest preflight, and
the retention sweep's source backstop — must never delete a source out
from under a RUNNING pipeline. A multi-hour VOD spends long stretches
(AssemblyAI polling, LLM passes) without writing to `source/`, so the
mtime-based "recently active" check alone can't see the run and would
reclaim it mid-flight, failing the run at the next step that reads the
file.

Authoritative for the current single-host deployment: every launch path
(API, upload, live, channel poll, recovery sweep) funnels through the
runners in `chalybclip.api._pipeline`, and every reclaimer runs in the
same process. A future multi-worker split (Modal) moves this to a
shared store, same as the auto-program locks did (migration 055).
Ref-counted so overlapping runs on the same stream (rare, but the
dispatcher doesn't forbid them) don't unregister each other early.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager

_active: Counter[str] = Counter()


def active_stream_ids() -> frozenset[str]:
    """Stream ids with at least one pipeline run in flight right now."""
    return frozenset(sid for sid, n in _active.items() if n > 0)


def register(stream_id: str) -> None:
    """Mark one in-flight run for `stream_id`. Pair with `unregister`.

    Split out of `pipeline_active` so the dispatcher can register at
    DISPATCH time (covering the wait for a semaphore slot, which can be
    hours behind a multi-hour VOD) rather than only once the runner
    executes — the recovery sweeper must see queued runs as in flight or
    it re-dispatches them as "orphans".
    """
    _active[stream_id] += 1


def unregister(stream_id: str) -> None:
    """Release one in-flight run for `stream_id`."""
    _active[stream_id] -= 1
    if _active[stream_id] <= 0:
        del _active[stream_id]


@contextmanager
def pipeline_active(stream_id: str) -> Iterator[None]:
    """Mark `stream_id`'s pipeline as in flight for the duration."""
    register(stream_id)
    try:
        yield
    finally:
        unregister(stream_id)
