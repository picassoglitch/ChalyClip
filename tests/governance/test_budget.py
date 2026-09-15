"""BudgetGovernor — daily LLM ceiling, publish quota, concurrency cap, cooldown.

The governor is process-local for semaphores + verdict windows and DB-backed
for daily totals. Tests exercise each axis in isolation; router + publisher
integration is covered separately.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from chalybclip.db import (
    ApiTokensRepo,
    Database,
    LLMCallsRepo,
    PublishJobsRepo,
    TenantsRepo,
    apply_migrations,
)
from chalybclip.db.models import (
    CandidateRow,
    ClipRow,
    LLMCallRow,
    StreamRow,
    VariantRow,
)
from chalybclip.errors import BudgetExceeded, CooldownActive, ChalybClipError, QuotaExceeded
from chalybclip.governance import BudgetGovernor
from chalybclip.tenancy import bound_tenant


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "gov.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


async def _seed_tenant(
    db: Database,
    name: str = "Aldo",
    *,
    daily_llm_budget_usd_micros: int | None = None,
    daily_publish_limit: int | None = None,
    rescore_concurrency_cap: int | None = None,
) -> str:
    repo = TenantsRepo(db)
    t = await repo.create(name=name)
    kwargs: dict[str, int | None] = {}
    if daily_llm_budget_usd_micros is not None:
        kwargs["daily_llm_budget_usd_micros"] = daily_llm_budget_usd_micros
    if daily_publish_limit is not None:
        kwargs["daily_publish_limit"] = daily_publish_limit
    if rescore_concurrency_cap is not None:
        kwargs["rescore_concurrency_cap"] = rescore_concurrency_cap
    if kwargs:
        await repo.set_budget(t.id, **kwargs)  # type: ignore[arg-type]
    return t.id


async def _record_llm_spend(db: Database, tenant_id: str, micros: int) -> None:
    with bound_tenant(tenant_id):
        await LLMCallsRepo(db).record(
            LLMCallRow(
                id=f"llm_{micros}",
                tenant_id=tenant_id,
                purpose="variant_generation",
                provider="anthropic",
                model="m",
                quality="standard",
                input_tokens=10,
                output_tokens=5,
                cost_usd_micros=micros,
                status="ok",
                error=None,
                attempts=1,
                ts=_now(),
            )
        )


# ---------- LLM spend ceiling ----------


async def test_check_llm_spend_unlimited_when_cap_null(db: Database) -> None:
    tenant_id = await _seed_tenant(db)  # no budget set -> NULL = unlimited
    gov = BudgetGovernor(db)
    # Should not raise no matter how much we've spent.
    await _record_llm_spend(db, tenant_id, 999_999_999)
    await gov.check_llm_spend(tenant_id)


async def test_check_llm_spend_under_cap_passes(db: Database) -> None:
    tenant_id = await _seed_tenant(db, daily_llm_budget_usd_micros=5_000_000)
    await _record_llm_spend(db, tenant_id, 1_000_000)
    gov = BudgetGovernor(db)
    await gov.check_llm_spend(tenant_id)  # 1M < 5M ceiling -> ok


async def test_check_llm_spend_at_cap_raises(db: Database) -> None:
    """Once today's total >= cap, the next call refuses."""
    tenant_id = await _seed_tenant(db, daily_llm_budget_usd_micros=5_000_000)
    await _record_llm_spend(db, tenant_id, 5_000_000)
    gov = BudgetGovernor(db)
    with pytest.raises(BudgetExceeded, match="hit daily LLM budget"):
        await gov.check_llm_spend(tenant_id)


async def test_check_llm_spend_unknown_tenant_raises(db: Database) -> None:
    gov = BudgetGovernor(db)
    with pytest.raises(ChalybClipError, match="tenant not found"):
        await gov.check_llm_spend("ten_does_not_exist")


# ---------- Publish quota ----------


async def _seed_clip_chain(db: Database, tenant_id: str) -> tuple[str, str, str]:
    """Minimal scaffolding for a publish_job FK chain."""
    from chalybclip.db import (
        CandidatesRepo,
        ClipsRepo,
        ConnectedAccountsRepo,
        PersonasRepo,
        StreamsRepo,
        VariantsRepo,
    )

    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_q",
                tenant_id=tenant_id,
                vod_url="x",
                platform="kick",
                title=None,
                channel=None,
                duration_s=60.0,
                source_video_path="/tmp/v",
                source_audio_path="/tmp/a",
                status="ingested",
                created_at=_now(),
            )
        )
        await CandidatesRepo(db).upsert_many(
            [
                CandidateRow(
                    id="cnd_q",
                    stream_id="str_q",
                    tenant_id=tenant_id,
                    ts=10.0,
                    score=0.5,
                    reason="voice",
                    evidence={},
                    created_at=_now(),
                )
            ]
        )
        await ClipsRepo(db).upsert_many(
            [
                ClipRow(
                    id="clp_q",
                    stream_id="str_q",
                    tenant_id=tenant_id,
                    candidate_id="cnd_q",
                    start_s=0.0,
                    end_s=10.0,
                    duration_s=10.0,
                    width=1080,
                    height=1920,
                    path="/tmp/c.mp4",
                    status="cut",
                    created_at=_now(),
                )
            ]
        )
        await PersonasRepo(db).create(
            persona_id="p1",
            name="P",
            primary_language="es",
            target_languages=["es"],
            voice_prompt="v",
        )
        await VariantsRepo(db).replace_for_clip_persona(
            "clp_q",
            "p1",
            [
                VariantRow(
                    id="var_q",
                    clip_id="clp_q",
                    tenant_id=tenant_id,
                    persona_id="p1",
                    language="es",
                    caption="c",
                    title_card_text="",
                    hashtags=[],
                    model=None,
                    created_at=_now(),
                )
            ],
        )
        acct = await ConnectedAccountsRepo(db).create(
            platform="tiktok", external_id="u"
        )
    return "clp_q", "var_q", acct.id


async def test_check_publish_quota_unlimited_when_null(db: Database) -> None:
    tenant_id = await _seed_tenant(db)  # no quota
    gov = BudgetGovernor(db)
    await gov.check_publish_quota(tenant_id)  # ok regardless of count


async def test_check_publish_quota_at_cap_raises(db: Database) -> None:
    tenant_id = await _seed_tenant(db, daily_publish_limit=2)
    clip_id, variant_id, account_id = await _seed_clip_chain(db, tenant_id)
    with bound_tenant(tenant_id):
        repo = PublishJobsRepo(db)
        for _ in range(2):
            await repo.enqueue(
                clip_id=clip_id,
                variant_id=variant_id,
                account_id=account_id,
                platform="tiktok",
            )
    gov = BudgetGovernor(db)
    with pytest.raises(QuotaExceeded, match="daily publish quota"):
        await gov.check_publish_quota(tenant_id)


async def test_check_publish_quota_per_platform_scope(db: Database) -> None:
    """quota=2; one tiktok job + one yt job -> tiktok still under, yt still under."""
    tenant_id = await _seed_tenant(db, daily_publish_limit=2)
    clip_id, variant_id, account_id = await _seed_clip_chain(db, tenant_id)
    with bound_tenant(tenant_id):
        repo = PublishJobsRepo(db)
        await repo.enqueue(
            clip_id=clip_id, variant_id=variant_id,
            account_id=account_id, platform="tiktok",
        )
        await repo.enqueue(
            clip_id=clip_id, variant_id=variant_id,
            account_id=account_id, platform="youtube",
        )
    gov = BudgetGovernor(db)
    # Cross-platform total = 2 = cap -> refused.
    with pytest.raises(QuotaExceeded):
        await gov.check_publish_quota(tenant_id)
    # tiktok scope = 1 < 2 -> still ok.
    await gov.check_publish_quota(tenant_id, platform="tiktok")
    await gov.check_publish_quota(tenant_id, platform="youtube")


# ---------- Rescore concurrency cap ----------


async def test_concurrency_cap_blocks_extra_callers(db: Database) -> None:
    """5 concurrent rescores against cap=2 means 3 wait until others release."""
    tenant_id = await _seed_tenant(db, rescore_concurrency_cap=2)
    gov = BudgetGovernor(db)

    in_flight = 0
    max_in_flight = 0
    waiting = asyncio.Event()
    release = asyncio.Event()

    async def worker() -> None:
        nonlocal in_flight, max_in_flight
        async with gov.acquire_rescore_slot(tenant_id):
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            if in_flight == 2:
                waiting.set()
            await release.wait()
            in_flight -= 1

    tasks = [asyncio.create_task(worker()) for _ in range(5)]
    # Give the first 2 time to grab the semaphore.
    await asyncio.wait_for(waiting.wait(), timeout=1.0)
    assert max_in_flight == 2, f"cap leaked: max_in_flight={max_in_flight}"
    # Three are still waiting.
    pending = [t for t in tasks if not t.done()]
    assert len(pending) == 5  # nobody finishes until release fires
    release.set()
    await asyncio.gather(*tasks)
    # After everyone runs the cap still held - max never went above 2.
    assert max_in_flight == 2


async def test_concurrency_cap_unknown_tenant_raises(db: Database) -> None:
    gov = BudgetGovernor(db)
    with pytest.raises(ChalybClipError, match="tenant not found"):
        async with gov.acquire_rescore_slot("ten_nope"):
            pass


# ---------- Cooldown ----------


async def test_cooldown_triggers_after_low_confidence_streak(db: Database) -> None:
    tenant_id = await _seed_tenant(db)
    fixed_now = _dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=_dt.UTC)
    gov = BudgetGovernor(
        db,
        clock=lambda: fixed_now,
        low_confidence_threshold=0.3,
        low_confidence_lookback=3,
        cooldown_s=300,
    )
    # Three low verdicts in a row - trips the cooldown.
    gov.record_rescore_verdict(tenant_id, 0.10)
    gov.record_rescore_verdict(tenant_id, 0.20)
    await gov.check_rescore_cooldown(tenant_id)  # still OK before the 3rd
    gov.record_rescore_verdict(tenant_id, 0.05)
    with pytest.raises(CooldownActive, match="rescore cooldown"):
        await gov.check_rescore_cooldown(tenant_id)


async def test_cooldown_clears_after_window(db: Database) -> None:
    tenant_id = await _seed_tenant(db)
    t0 = _dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=_dt.UTC)
    current = [t0]
    gov = BudgetGovernor(
        db,
        clock=lambda: current[0],
        low_confidence_threshold=0.3,
        low_confidence_lookback=2,
        cooldown_s=60,
    )
    gov.record_rescore_verdict(tenant_id, 0.1)
    gov.record_rescore_verdict(tenant_id, 0.1)
    with pytest.raises(CooldownActive):
        await gov.check_rescore_cooldown(tenant_id)
    # Advance past the cooldown.
    current[0] = t0 + _dt.timedelta(seconds=120)
    await gov.check_rescore_cooldown(tenant_id)  # cleared


async def test_cooldown_resets_on_high_verdict_in_window(db: Database) -> None:
    """If even one of the last K verdicts is above threshold, no cooldown."""
    tenant_id = await _seed_tenant(db)
    fixed_now = _dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=_dt.UTC)
    gov = BudgetGovernor(
        db,
        clock=lambda: fixed_now,
        low_confidence_threshold=0.3,
        low_confidence_lookback=3,
        cooldown_s=60,
    )
    gov.record_rescore_verdict(tenant_id, 0.1)
    gov.record_rescore_verdict(tenant_id, 0.5)  # breaks the streak
    gov.record_rescore_verdict(tenant_id, 0.1)
    # Last 3 are [0.1, 0.5, 0.1]; not all below 0.3 -> no cooldown.
    await gov.check_rescore_cooldown(tenant_id)
    assert gov._peek_cooldown(tenant_id) is None


# ---------- Smoke checks ----------


async def test_governor_rejects_invalid_lookback(db: Database) -> None:
    with pytest.raises(ChalybClipError, match="lookback must be"):
        BudgetGovernor(db, low_confidence_lookback=0)


async def test_api_tokens_repo_works_for_test_setup(db: Database) -> None:
    """Sanity: governance tests can still seed tenants + api tokens."""
    t = await TenantsRepo(db).create(name="Aldo")
    with bound_tenant(t.id):
        # mint a token via helpers used in api fixtures
        from chalybclip.tenancy import hash_token, mint_token

        raw, _ = mint_token()
        await ApiTokensRepo(db).create(hash_=hash_token(raw), scope="full")
