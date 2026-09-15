"""Budget governance + concurrency caps + cooldown enforcement.

Phase 2 Task 1 lock-down. Every external-cost path (LLM calls, native
publish dispatches) consults a `BudgetGovernor` before issuing the call.
The governor reads daily totals from `llm_calls` and `publish_jobs`, holds
per-tenant `asyncio.Semaphore`s sized from `tenants.rescore_concurrency_cap`,
and tracks low-confidence rescore verdicts in memory to trigger cooldowns.

Failure modes raise typed errors: `BudgetExceeded`, `QuotaExceeded`,
`CooldownActive`. None of them propagate to user-facing 500s; callers catch
them, emit `llm.budget_exhausted` / `publish.quota_exceeded` events, and
halt the affected work cleanly.
"""

from .budget import BudgetGovernor

__all__ = ["BudgetGovernor"]
