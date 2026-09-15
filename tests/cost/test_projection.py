"""CostProjection — MTD totals, EOM extrapolation, headroom math."""

from __future__ import annotations

import datetime as _dt
from collections.abc import AsyncIterator
from pathlib import Path

import pytest_asyncio

from chalybclip.cost import compute_cost_projection
from chalybclip.db import (
    Database,
    LLMCallsRepo,
    TenantsRepo,
    apply_migrations,
)
from chalybclip.db.models import LLMCallRow
from chalybclip.tenancy import bound_tenant


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "cost.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


async def _record(
    db: Database,
    tenant_id: str,
    *,
    micros: int,
    ts: str,
    purpose: str = "variant_generation",
    provider: str = "anthropic",
    model: str = "claude-haiku-4-5-20251001",
) -> None:
    with bound_tenant(tenant_id):
        await LLMCallsRepo(db).record(
            LLMCallRow(
                id=f"llm_{ts}_{micros}",
                tenant_id=tenant_id,
                purpose=purpose,
                provider=provider,
                model=model,
                quality="standard",
                input_tokens=10,
                output_tokens=5,
                cost_usd_micros=micros,
                status="ok",
                error=None,
                attempts=1,
                ts=ts,
            )
        )


async def test_projection_groups_by_purpose_and_model(db: Database) -> None:
    """Today's spend rolled up correctly across two purposes + two models."""
    tenant = await TenantsRepo(db).create(name="A")
    fixed_now = _dt.datetime(2026, 5, 7, 14, 30, 0, tzinfo=_dt.UTC)

    today_iso_morning = "2026-05-07T08:00:00+00:00"
    today_iso_now = "2026-05-07T14:00:00+00:00"
    earlier_in_month = "2026-05-03T10:00:00+00:00"
    last_month = "2026-04-30T23:00:00+00:00"

    # Today: 1000 + 2500
    await _record(db, tenant.id, micros=1000, ts=today_iso_morning,
                  purpose="variant_generation", model="claude-haiku-4-5-20251001")
    await _record(db, tenant.id, micros=2500, ts=today_iso_now,
                  purpose="vision_rescore", model="claude-opus-4-7")
    # Earlier this month
    await _record(db, tenant.id, micros=4000, ts=earlier_in_month,
                  purpose="variant_generation", model="claude-haiku-4-5-20251001")
    # Last month — must NOT count.
    await _record(db, tenant.id, micros=99_999, ts=last_month)

    with bound_tenant(tenant.id):
        proj = await compute_cost_projection(db, clock=lambda: fixed_now)

    assert proj.today_micros == 3500
    assert proj.mtd_micros == 7500
    assert proj.n_calls_mtd == 3
    # by_purpose: variant_generation = 5000, vision_rescore = 2500
    assert proj.by_purpose["variant_generation"] == 5000
    assert proj.by_purpose["vision_rescore"] == 2500
    # by_model: haiku = 5000, opus = 2500
    assert proj.by_model["anthropic/claude-haiku-4-5-20251001"] == 5000
    assert proj.by_model["anthropic/claude-opus-4-7"] == 2500


async def test_projection_extrapolates_to_eom(db: Database) -> None:
    tenant = await TenantsRepo(db).create(name="A")
    # Day 7 of May (31 days). MTD spend = $1.00 = 1_000_000 micros.
    fixed_now = _dt.datetime(2026, 5, 7, 12, 0, 0, tzinfo=_dt.UTC)
    await _record(
        db,
        tenant.id,
        micros=1_000_000,
        ts="2026-05-04T12:00:00+00:00",
    )
    with bound_tenant(tenant.id):
        proj = await compute_cost_projection(db, clock=lambda: fixed_now)
    # Linear: 1_000_000 * 31 / 7 = 4_428_571
    assert proj.projected_eom_micros == round(1_000_000 * 31 / 7)


async def test_projection_budget_headroom_when_set(db: Database) -> None:
    tenant = await TenantsRepo(db).create(name="A")
    await TenantsRepo(db).set_budget(
        tenant.id, daily_llm_budget_usd_micros=10_000_000
    )
    fixed_now = _dt.datetime(2026, 5, 7, 14, 0, 0, tzinfo=_dt.UTC)
    await _record(
        db, tenant.id, micros=2_500_000, ts="2026-05-07T08:00:00+00:00"
    )
    with bound_tenant(tenant.id):
        proj = await compute_cost_projection(db, clock=lambda: fixed_now)
    assert proj.daily_budget_micros == 10_000_000
    assert proj.budget_consumed_frac is not None
    assert abs(proj.budget_consumed_frac - 0.25) < 1e-6


async def test_projection_budget_frac_none_when_unlimited(db: Database) -> None:
    tenant = await TenantsRepo(db).create(name="A")
    fixed_now = _dt.datetime(2026, 5, 7, 14, 0, 0, tzinfo=_dt.UTC)
    await _record(
        db, tenant.id, micros=5_000_000, ts="2026-05-07T08:00:00+00:00"
    )
    with bound_tenant(tenant.id):
        proj = await compute_cost_projection(db, clock=lambda: fixed_now)
    assert proj.daily_budget_micros is None
    assert proj.budget_consumed_frac is None


