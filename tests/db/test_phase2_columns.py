"""P2 Task 0 lock-down tests: schema 002 columns + repo helpers round-trip.

Each new column or method gets one focused assertion. The point is that
the storage shape is locked at v2 - any future schema change must be a
new migration, not a silent column tweak.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from chalybclip.db import (
    CandidatesRepo,
    ConnectedAccountsRepo,
    Database,
    LLMCallsRepo,
    PublishJobsRepo,
    StreamsRepo,
    TenantsRepo,
    WebhookSubscriptionsRepo,
)
from chalybclip.db.models import (
    CandidateRow,
    ClipRow,
    LLMCallRow,
    StreamRow,
)
from chalybclip.errors import TenancyError
from chalybclip.tenancy import bound_tenant


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


# ---------- tenants budget knobs ----------


async def test_tenant_default_budget_is_unlimited(migrated_db: Database) -> None:
    t = await TenantsRepo(migrated_db).create(name="Aldo")
    assert t.daily_llm_budget_usd_micros is None
    assert t.daily_publish_limit is None
    assert t.rescore_concurrency_cap == 4


async def test_tenant_set_budget_persists(migrated_db: Database) -> None:
    repo = TenantsRepo(migrated_db)
    t = await repo.create(name="Aldo")
    updated = await repo.set_budget(
        t.id,
        daily_llm_budget_usd_micros=5_000_000,  # $5
        daily_publish_limit=50,
        rescore_concurrency_cap=8,
    )
    assert updated.daily_llm_budget_usd_micros == 5_000_000
    assert updated.daily_publish_limit == 50
    assert updated.rescore_concurrency_cap == 8

    # Round-trips through .get(...).
    again = await repo.get(t.id)
    assert again is not None
    assert again.daily_llm_budget_usd_micros == 5_000_000


async def test_tenant_set_budget_partial_leaves_others_alone(migrated_db: Database) -> None:
    repo = TenantsRepo(migrated_db)
    t = await repo.create(name="Aldo")
    await repo.set_budget(t.id, daily_publish_limit=100)
    refreshed = await repo.get(t.id)
    assert refreshed is not None
    assert refreshed.daily_publish_limit == 100
    assert refreshed.daily_llm_budget_usd_micros is None
    assert refreshed.rescore_concurrency_cap == 4  # default preserved


# ---------- connected_accounts: oauth + status ----------


async def test_connected_account_round_trips_oauth_fields(migrated_db: Database) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="Aldo")
    with bound_tenant(tenant.id):
        acct = await ConnectedAccountsRepo(migrated_db).create(
            platform="tiktok",
            external_id="user_42",
            display_name="Aldo",
            refresh_token="rt_abc",
            expires_at="2026-12-31T00:00:00+00:00",
            scopes=["video.upload", "video.list"],
        )
        assert acct.refresh_token == "rt_abc"
        assert acct.scopes == ["video.upload", "video.list"]
        assert acct.status == "active"
        assert acct.expires_at == "2026-12-31T00:00:00+00:00"


async def test_connected_account_update_oauth_partial(migrated_db: Database) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="Aldo")
    with bound_tenant(tenant.id):
        repo = ConnectedAccountsRepo(migrated_db)
        acct = await repo.create(
            platform="tiktok", external_id="u1", refresh_token="old_rt"
        )
        updated = await repo.update_oauth(
            acct.id,
            refresh_token="new_rt",
            expires_at="2027-01-01T00:00:00+00:00",
        )
        assert updated.refresh_token == "new_rt"
        assert updated.expires_at == "2027-01-01T00:00:00+00:00"
        # Untouched fields preserved.
        assert updated.platform == "tiktok"
        assert updated.status == "active"


async def test_connected_account_mark_status(migrated_db: Database) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="Aldo")
    with bound_tenant(tenant.id):
        repo = ConnectedAccountsRepo(migrated_db)
        acct = await repo.create(platform="tiktok", external_id="u1")
        flipped = await repo.mark_status(acct.id, "auth_failed")
        assert flipped.status == "auth_failed"


# ---------- candidates: rescore verdict ----------


async def test_candidate_rescore_columns_default_none(migrated_db: Database) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="Aldo")
    with bound_tenant(tenant.id):
        await StreamsRepo(migrated_db).upsert(
            StreamRow(
                id="str_a",
                tenant_id=tenant.id,
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
        repo = CandidatesRepo(migrated_db)
        await repo.upsert_many(
            [
                CandidateRow(
                    id="cnd_a",
                    stream_id="str_a",
                    tenant_id=tenant.id,
                    ts=10.0,
                    score=0.5,
                    reason="voice",
                    evidence={},
                    created_at=_now(),
                )
            ]
        )
        rows = await repo.list_for_stream("str_a")
        assert len(rows) == 1
        assert rows[0].rescore_score is None
        assert rows[0].rescore_reason is None
        assert rows[0].rescore_model is None


async def test_candidate_update_rescore_persists(migrated_db: Database) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="Aldo")
    with bound_tenant(tenant.id):
        await StreamsRepo(migrated_db).upsert(
            StreamRow(
                id="str_a",
                tenant_id=tenant.id,
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
        repo = CandidatesRepo(migrated_db)
        await repo.upsert_many(
            [
                CandidateRow(
                    id="cnd_a",
                    stream_id="str_a",
                    tenant_id=tenant.id,
                    ts=10.0,
                    score=0.5,
                    reason="voice",
                    evidence={},
                    created_at=_now(),
                )
            ]
        )
        updated = await repo.update_rescore(
            "cnd_a",
            rescore_score=0.85,
            rescore_reason="strong shock-face onset at ts=10.5",
            rescore_model="claude-opus-4-7",
        )
        assert updated.rescore_score == 0.85
        assert updated.rescore_model == "claude-opus-4-7"
        assert "shock-face" in (updated.rescore_reason or "")


# ---------- publish_jobs: external_url + platform_metadata ----------


async def _seed_clip_for_publish(db: Database, tenant_id: str) -> str:
    """Minimum scaffolding so a publish_job can FK into clips."""
    from chalybclip.db import ClipsRepo, PersonasRepo, VariantsRepo
    from chalybclip.db.models import VariantRow

    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_p",
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
                    id="cnd_p",
                    stream_id="str_p",
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
                    id="clp_p",
                    stream_id="str_p",
                    tenant_id=tenant_id,
                    candidate_id="cnd_p",
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
            "clp_p",
            "p1",
            [
                VariantRow(
                    id="var_p",
                    clip_id="clp_p",
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
    return "clp_p"


async def test_publish_job_enqueues_with_phase2_columns_default_null(
    migrated_db: Database,
) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="Aldo")
    clip_id = await _seed_clip_for_publish(migrated_db, tenant.id)
    with bound_tenant(tenant.id):
        # Need an account_id; just create one inline.
        acct = await ConnectedAccountsRepo(migrated_db).create(
            platform="tiktok", external_id="u"
        )
        job = await PublishJobsRepo(migrated_db).enqueue(
            clip_id=clip_id,
            variant_id="var_p",
            account_id=acct.id,
            platform="tiktok",
        )
        assert job.external_url is None
        assert job.platform_metadata is None
        assert job.status == "pending"


async def test_publish_jobs_count_for_tenant_today(migrated_db: Database) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="Aldo")
    clip_id = await _seed_clip_for_publish(migrated_db, tenant.id)
    with bound_tenant(tenant.id):
        acct = await ConnectedAccountsRepo(migrated_db).create(
            platform="tiktok", external_id="u"
        )
        repo = PublishJobsRepo(migrated_db)
        for _ in range(3):
            await repo.enqueue(
                clip_id=clip_id, variant_id="var_p",
                account_id=acct.id, platform="tiktok",
            )
        # Cross-platform total + per-platform scope.
        assert await repo.count_for_tenant_today() == 3
        assert await repo.count_for_tenant_today(platform="tiktok") == 3
        assert await repo.count_for_tenant_today(platform="youtube") == 0


# ---------- llm_calls today-spend roll-up ----------


async def test_llm_calls_total_spend_today_micros(migrated_db: Database) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="Aldo")
    repo = LLMCallsRepo(migrated_db)
    with bound_tenant(tenant.id):
        # Today's calls.
        for cost in (1_000, 2_500, 750):
            await repo.record(
                LLMCallRow(
                    id=f"llm_{cost}",
                    tenant_id=tenant.id,
                    purpose="variant_generation",
                    provider="anthropic",
                    model="claude-haiku-4-5-20251001",
                    quality="standard",
                    input_tokens=100,
                    output_tokens=50,
                    cost_usd_micros=cost,
                    status="ok",
                    error=None,
                    attempts=1,
                    ts=_now(),
                )
            )
        # An old call should NOT count.
        await repo.record(
            LLMCallRow(
                id="llm_old",
                tenant_id=tenant.id,
                purpose="variant_generation",
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                quality="standard",
                input_tokens=10,
                output_tokens=5,
                cost_usd_micros=99_999,
                status="ok",
                error=None,
                attempts=1,
                ts="2020-01-01T12:00:00+00:00",
            )
        )
        total = await repo.total_spend_today_micros()
        assert total == 1_000 + 2_500 + 750


# ---------- webhook_subscriptions CRUD ----------


async def test_webhook_create_then_get_round_trips(migrated_db: Database) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="Aldo")
    with bound_tenant(tenant.id):
        repo = WebhookSubscriptionsRepo(migrated_db)
        sub = await repo.create(
            url="https://example.com/hook",
            types=["clip.published", "clip.approved"],
            secret="s" * 32,
        )
        assert sub.id.startswith("whk_")
        assert sub.types == ["clip.published", "clip.approved"]
        assert sub.status == "active"
        assert sub.failure_count == 0

        loaded = await repo.get(sub.id)
        assert loaded is not None
        assert loaded.url == "https://example.com/hook"


async def test_webhook_list_filters_by_status(migrated_db: Database) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="Aldo")
    with bound_tenant(tenant.id):
        repo = WebhookSubscriptionsRepo(migrated_db)
        await repo.create(url="https://a/", types=["x"], secret="s")
        await repo.create(url="https://b/", types=["y"], secret="s")
        all_active = await repo.list_for_tenant(status="active")
        assert len(all_active) == 2
        assert {s.url for s in all_active} == {"https://a/", "https://b/"}


async def test_webhook_record_dispatch_and_failure(migrated_db: Database) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="Aldo")
    with bound_tenant(tenant.id):
        repo = WebhookSubscriptionsRepo(migrated_db)
        sub = await repo.create(url="https://a/", types=["x"], secret="s")

        n = await repo.record_failure(sub.id)
        assert n == 1
        n = await repo.record_failure(sub.id)
        assert n == 2

        # Successful dispatch resets failure_count and updates last_dispatch_ts.
        ts = _now()
        await repo.record_dispatch(sub.id, ts=ts)
        refreshed = await repo.get(sub.id)
        assert refreshed is not None
        assert refreshed.failure_count == 0
        assert refreshed.last_dispatch_ts == ts


async def test_webhook_delete_returns_true_only_on_hit(migrated_db: Database) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="Aldo")
    with bound_tenant(tenant.id):
        repo = WebhookSubscriptionsRepo(migrated_db)
        sub = await repo.create(url="https://a/", types=["x"], secret="s")
        assert await repo.delete(sub.id) is True
        assert await repo.delete(sub.id) is False  # already gone


async def test_webhook_cross_tenant_isolation(migrated_db: Database) -> None:
    """Bob's repo can't read or delete Alice's subscription."""
    alice = await TenantsRepo(migrated_db).create(name="Alice")
    bob = await TenantsRepo(migrated_db).create(name="Bob")
    with bound_tenant(alice.id):
        sub = await WebhookSubscriptionsRepo(migrated_db).create(
            url="https://alice/", types=["x"], secret="s"
        )
    with bound_tenant(bob.id):
        bob_repo = WebhookSubscriptionsRepo(migrated_db)
        assert await bob_repo.get(sub.id) is None
        assert await bob_repo.list_for_tenant() == []
        assert await bob_repo.delete(sub.id) is False


# ---------- bound_tenant required for tenant-scoped repos ----------


async def test_set_budget_works_without_bound_tenant(migrated_db: Database) -> None:
    """TenantsRepo.set_budget is admin-grade - no tenant context needed."""
    t = await TenantsRepo(migrated_db).create(name="Aldo")
    # No bound_tenant() wrapper.
    updated = await TenantsRepo(migrated_db).set_budget(
        t.id, daily_llm_budget_usd_micros=1_000_000
    )
    assert updated.daily_llm_budget_usd_micros == 1_000_000


async def test_webhook_repo_requires_bound_tenant(migrated_db: Database) -> None:
    with pytest.raises(TenancyError, match="no tenant bound"):
        await WebhookSubscriptionsRepo(migrated_db).create(
            url="https://a/", types=["x"], secret="s"
        )
