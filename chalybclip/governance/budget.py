"""BudgetGovernor — the single point that says "yes, this call is allowed".

Three independent enforcement axes:

1. **Daily LLM spend ceiling.** Reads `llm_calls.cost_usd_micros` summed
   over today (UTC) for the tenant; compares against
   `tenants.daily_llm_budget_usd_micros`. NULL budget = unlimited.

2. **Daily publish quota.** Reads `publish_jobs` row count for today,
   optionally filtered by platform; compares against
   `tenants.daily_publish_limit`. NULL = unlimited.

3. **Rescore concurrency + cooldown.** Per-tenant `asyncio.Semaphore`
   sized from `tenants.rescore_concurrency_cap` caps how many vision-
   rescore calls can be in flight at once. A rolling window of the last K
   rescore verdicts triggers a cooldown when all K are below
   `low_confidence_threshold` — until `cooldown_s` elapses, new rescore
   requests are refused.

The governor is process-local (semaphores, verdict deque) and DB-backed
(daily counts). A second worker process sees the same daily totals from
the DB but maintains its own semaphores + verdict windows; that's fine
because the cap is per-process anyway and the cooldown is conservative.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from chalybclip.db import Database, LLMCallsRepo, PublishJobsRepo, TenantsRepo
from chalybclip.errors import BudgetExceeded, ChalybClipError, CooldownActive, QuotaExceeded
from chalybclip.tenancy import bound_tenant

# Default cooldown configuration. Operators can override per `BudgetGovernor`.
_DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.30
_DEFAULT_LOW_CONFIDENCE_LOOKBACK = 3
_DEFAULT_COOLDOWN_S = 300.0  # 5 minutes


@dataclass(frozen=True)
class _CooldownState:
    """Per-tenant rolling-window state for the cooldown heuristic."""

    until_ts: _dt.datetime | None  # None = no active cooldown


def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC)


class BudgetGovernor:
    """Single point that gates every external-cost call."""

    def __init__(
        self,
        db: Database,
        *,
        clock: Callable[[], _dt.datetime] | None = None,
        low_confidence_threshold: float = _DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        low_confidence_lookback: int = _DEFAULT_LOW_CONFIDENCE_LOOKBACK,
        cooldown_s: float = _DEFAULT_COOLDOWN_S,
    ) -> None:
        if low_confidence_lookback < 1:
            raise ChalybClipError(
                f"low_confidence_lookback must be >= 1, got {low_confidence_lookback}"
            )
        self._db = db
        self._clock = clock or _utc_now
        self._low_confidence_threshold = low_confidence_threshold
        self._low_confidence_lookback = low_confidence_lookback
        self._cooldown_s = cooldown_s
        # Per-tenant semaphores keyed by tenant_id. Sized lazily from
        # tenants.rescore_concurrency_cap on first acquire.
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        # Lazy-init lock so 5 concurrent callers don't each construct their
        # own Semaphore and defeat the cap.
        self._semaphore_init_lock = asyncio.Lock()
        # Per-tenant rolling window of recent rescore scores.
        self._recent_verdicts: dict[str, deque[float]] = {}
        # Per-tenant cooldown end timestamps.
        self._cooldowns: dict[str, _CooldownState] = {}

    # ------------------------------------------------------------------
    # LLM spend
    # ------------------------------------------------------------------

    async def check_llm_spend(self, tenant_id: str) -> None:
        """Refuse if today's spend already meets the daily ceiling.

        Pre-call check. We don't try to project the next call's cost because
        we don't know its token count yet; instead we refuse once today's
        running total is at-or-above the cap. The call that pushed us over
        runs, but every subsequent call gets blocked until midnight UTC.
        """
        cap = await self._tenant_llm_budget_micros(tenant_id)
        if cap is None:
            return
        with bound_tenant(tenant_id):
            spent = await LLMCallsRepo(self._db).total_spend_today_micros()
        if spent >= cap:
            raise BudgetExceeded(
                f"tenant {tenant_id!r} hit daily LLM budget: "
                f"{spent} usd_micros >= cap={cap} usd_micros"
            )

    async def _tenant_llm_budget_micros(self, tenant_id: str) -> int | None:
        tenant = await TenantsRepo(self._db).get(tenant_id)
        if tenant is None:
            raise ChalybClipError(f"tenant not found: {tenant_id}")
        return tenant.daily_llm_budget_usd_micros

    # ------------------------------------------------------------------
    # Publish quota
    # ------------------------------------------------------------------

    async def check_publish_quota(
        self, tenant_id: str, *, platform: str | None = None
    ) -> None:
        """Refuse if today's publish_jobs count already meets the cap.

        `platform=None` checks the cross-platform total; passing a value
        scopes it (a future use case if per-platform quotas are added).
        """
        tenant = await TenantsRepo(self._db).get(tenant_id)
        if tenant is None:
            raise ChalybClipError(f"tenant not found: {tenant_id}")
        cap = tenant.daily_publish_limit
        if cap is None:
            return
        with bound_tenant(tenant_id):
            count = await PublishJobsRepo(self._db).count_for_tenant_today(
                platform=platform
            )
        if count >= cap:
            scope = f"platform={platform!r} " if platform else ""
            raise QuotaExceeded(
                f"tenant {tenant_id!r} hit daily publish quota {scope}"
                f"({count} jobs >= cap={cap})"
            )

    # ------------------------------------------------------------------
    # Rescore concurrency
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def acquire_rescore_slot(self, tenant_id: str) -> AsyncIterator[None]:
        """Hold a per-tenant rescore concurrency slot for the duration of a block.

        Sized from `tenants.rescore_concurrency_cap` on first call (cached
        per-tenant). 5 callers against cap=2 means 2 hold and 3 wait.
        """
        sem = await self._semaphore_for(tenant_id)
        await sem.acquire()
        try:
            yield
        finally:
            sem.release()

    async def _semaphore_for(self, tenant_id: str) -> asyncio.Semaphore:
        existing = self._semaphores.get(tenant_id)
        if existing is not None:
            return existing
        # Serialize lazy-init so concurrent first-callers all share the same
        # Semaphore instance (otherwise each would construct a fresh one and
        # the cap would leak by N-1).
        async with self._semaphore_init_lock:
            cached = self._semaphores.get(tenant_id)
            if cached is not None:
                return cached
            tenant = await TenantsRepo(self._db).get(tenant_id)
            if tenant is None:
                raise ChalybClipError(f"tenant not found: {tenant_id}")
            cap = max(1, tenant.rescore_concurrency_cap)
            sem = asyncio.Semaphore(cap)
            self._semaphores[tenant_id] = sem
            return sem

    # ------------------------------------------------------------------
    # Cooldown
    # ------------------------------------------------------------------

    async def check_rescore_cooldown(self, tenant_id: str) -> None:
        """Refuse new rescores while the per-tenant cooldown is active."""
        state = self._cooldowns.get(tenant_id)
        if state is None or state.until_ts is None:
            return
        now = self._clock()
        if now < state.until_ts:
            seconds_left = (state.until_ts - now).total_seconds()
            raise CooldownActive(
                f"tenant {tenant_id!r} is in rescore cooldown for "
                f"{seconds_left:.0f}s more (low-confidence streak)"
            )
        # Cooldown has elapsed; clear it so the next failure starts fresh.
        self._cooldowns.pop(tenant_id, None)

    def record_rescore_verdict(self, tenant_id: str, score: float) -> None:
        """Append a verdict to the rolling window. Triggers cooldown when
        the last K verdicts are all below the low-confidence threshold."""
        window = self._recent_verdicts.setdefault(
            tenant_id, deque(maxlen=self._low_confidence_lookback)
        )
        window.append(score)
        if (
            len(window) >= self._low_confidence_lookback
            and all(s < self._low_confidence_threshold for s in window)
        ):
            until = self._clock() + _dt.timedelta(seconds=self._cooldown_s)
            self._cooldowns[tenant_id] = _CooldownState(until_ts=until)
            # Clear the window so we don't re-trip the cooldown the next
            # call without a fresh streak of failures.
            window.clear()

    # ------------------------------------------------------------------
    # Test seams
    # ------------------------------------------------------------------

    def _peek_cooldown(self, tenant_id: str) -> _dt.datetime | None:
        """Return the cooldown end ts (or None). Tests use this for assertions."""
        state = self._cooldowns.get(tenant_id)
        return state.until_ts if state else None
