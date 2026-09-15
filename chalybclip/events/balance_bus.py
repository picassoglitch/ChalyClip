"""In-process pub/sub for token-balance changes → SSE push.

The token-balance chip used to poll /dashboard/_balance/chip every 30s. That
burned a full request (auth middleware + DB read + template render) per client
every 30s, forever, even when nothing changed. This bus replaces the poll with
a push: the ONE place the cached balance changes (TenantsRepo.set_balance_cache)
publishes here, and the /dashboard/_balance/stream SSE endpoint relays it to the
browser, which then swaps the chip — so a request fires only on an actual change.

Scope / assumptions:
  - In-process only. ChalyClip runs as a single uvicorn worker (Whisper runs
    in-process; see run.py / Dockerfile CMD), so every request — including the
    usage reporter that updates the balance and the SSE connection — shares one
    event loop. If this ever scales to multiple workers/instances, swap this for
    a Redis pub/sub (or a DB-trigger fan-out); the publish/subscribe surface
    below is intentionally tiny so that swap stays local.
  - Best-effort. A full subscriber queue drops the event rather than blocking
    the DB write — the next change (or the SSE reconnect) reconciles the chip.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict


class BalanceBus:
    """Fan-out of balance-change payloads to per-tenant subscribers."""

    def __init__(self) -> None:
        # tenant_id -> set of subscriber queues (one per open SSE connection).
        self._subs: dict[str, set[asyncio.Queue[dict]]] = defaultdict(set)

    def subscribe(self, tenant_id: str) -> asyncio.Queue[dict]:
        q: asyncio.Queue[dict] = asyncio.Queue(maxsize=8)
        self._subs[tenant_id].add(q)
        return q

    def unsubscribe(self, tenant_id: str, q: asyncio.Queue[dict]) -> None:
        subs = self._subs.get(tenant_id)
        if not subs:
            return
        subs.discard(q)
        if not subs:
            self._subs.pop(tenant_id, None)

    def publish(self, tenant_id: str, payload: dict) -> None:
        """Notify every open SSE connection for this tenant. Non-blocking and
        exception-safe — a balance write must never fail because a chip
        subscriber is slow or gone."""
        for q in list(self._subs.get(tenant_id, ())):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                # Subscriber is backed up; drop — the chip reconciles on the
                # next event / reconnect. Never block the caller.
                pass


# Module-level singleton — shared across the whole process.
balance_bus = BalanceBus()
