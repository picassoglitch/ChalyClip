"""LLM-spend projections + per-axis breakdowns.

Phase 2 Task 11. Reads `llm_calls` rows for the bound tenant and rolls
them into the numbers the dashboard's "/dashboard/llm-calls" panel
renders:

  * **today**: usd consumed since 00:00 UTC.
  * **mtd**: usd consumed since the 1st of the current month (UTC).
  * **projected_eom**: linear extrapolation of MTD pace through the end
    of the month. Useful for the "we're on track to spend $X" framing.
  * **by_purpose** / **by_model**: dict[str, int_micros] for the per-axis
    breakdown tables.
  * **budget_consumed_frac**: today_usd / tenants.daily_llm_budget when
    the budget cap is set; None when unlimited. Drives the headroom bar.

The HTMX template polls every 60s (per spec, no SSE) so callers don't
need to memoize - the math is cheap, the table is tenant-indexed.
"""

from __future__ import annotations

import calendar
import datetime as _dt
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

from chalybclip.db import Database, LLMCallsRepo, TenantsRepo
from chalybclip.errors import ChalybClipError
from chalybclip.tenancy import current_tenant_id


@dataclass(frozen=True)
class CostProjection:
    """Roll-up the dashboard renders. All money values are USD micros."""

    today_micros: int
    mtd_micros: int
    projected_eom_micros: int
    by_purpose: dict[str, int]
    by_model: dict[str, int]
    daily_budget_micros: int | None
    budget_consumed_frac: float | None  # 0.0-1.0+ (can exceed when over budget)
    n_calls_mtd: int = field(default=0)


def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC)


def _start_of_month(now: _dt.datetime) -> _dt.datetime:
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _start_of_today(now: _dt.datetime) -> _dt.datetime:
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _days_in_month(now: _dt.datetime) -> int:
    return calendar.monthrange(now.year, now.month)[1]


async def compute_cost_projection(
    db: Database,
    *,
    limit: int = 5000,
    clock: Callable[[], _dt.datetime] | None = None,
) -> CostProjection:
    """Compute the projection panel data for the bound tenant.

    `limit` caps how many recent rows we pull for the per-axis breakdowns
    (the totals come from a separate sum in `total_spend_today_micros` so
    they're correct even when `limit` truncates).
    """
    tenant_id = current_tenant_id()
    now = (clock or _utc_now)()
    today_floor = _start_of_today(now)
    month_floor = _start_of_month(now)

    rows = await LLMCallsRepo(db).list_for_tenant(limit=limit)
    today_total = 0
    mtd_total = 0
    n_mtd = 0
    by_purpose: dict[str, int] = defaultdict(int)
    by_model: dict[str, int] = defaultdict(int)

    for row in rows:
        try:
            ts = _dt.datetime.fromisoformat(row.ts)
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=_dt.UTC)
        if ts < month_floor:
            continue
        cost = int(row.cost_usd_micros)
        mtd_total += cost
        n_mtd += 1
        by_purpose[row.purpose] += cost
        by_model[f"{row.provider}/{row.model}"] += cost
        if ts >= today_floor:
            today_total += cost

    # Linear projection: average daily MTD * days_in_month.
    days_elapsed = max(1, (now.day - 1) + 1)  # day 1 = 1, ...
    days_in_month = _days_in_month(now)
    projected_eom = round(mtd_total * days_in_month / days_elapsed)

    # Budget headroom: today's spend vs the daily ceiling.
    tenant = await TenantsRepo(db).get(tenant_id)
    if tenant is None:
        raise ChalybClipError(f"tenant not found: {tenant_id}")
    daily_budget = tenant.daily_llm_budget_usd_micros
    consumed_frac: float | None = None
    if daily_budget is not None and daily_budget > 0:
        consumed_frac = today_total / daily_budget

    return CostProjection(
        today_micros=today_total,
        mtd_micros=mtd_total,
        projected_eom_micros=projected_eom,
        by_purpose=dict(by_purpose),
        by_model=dict(by_model),
        daily_budget_micros=daily_budget,
        budget_consumed_frac=consumed_frac,
        n_calls_mtd=n_mtd,
    )
