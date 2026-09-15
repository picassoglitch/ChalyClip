"""Token T3 — per-stream cost attribution on llm_calls.

Pins: stream_id round-trips on a recorded row, and cost_for_stream
aggregates tokens + USD cost across providers for ONE stream while
ignoring other streams / other tenants / failed calls.
"""

from __future__ import annotations

import datetime as _dt

from chalybclip.db import Database, LLMCallsRepo, TenantsRepo
from chalybclip.db.models import LLMCallRow
from chalybclip.tenancy import bound_tenant


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


def _row(
    tenant_id: str, *, stream_id: str | None, provider: str,
    input_t: int, output_t: int, cost: int, status: str = "ok",
    rid: str | None = None,
) -> LLMCallRow:
    import uuid
    return LLMCallRow(
        id=rid or f"llm_{uuid.uuid4().hex[:10]}",
        tenant_id=tenant_id, purpose="variants", provider=provider,
        model="m", quality="standard", input_tokens=input_t,
        output_tokens=output_t, cost_usd_micros=cost, status=status,
        attempts=1, ts=_now(), stream_id=stream_id,
    )


async def test_stream_id_round_trips(migrated_db: Database) -> None:
    t = await TenantsRepo(migrated_db).create(name="A")
    with bound_tenant(t.id):
        repo = LLMCallsRepo(migrated_db)
        await repo.record(_row(t.id, stream_id="str_1", provider="anthropic",
                               input_t=10, output_t=5, cost=100))
        rows = await repo.list_for_tenant()
    assert rows[0].stream_id == "str_1"


async def test_cost_for_stream_sums_across_providers(
    migrated_db: Database,
) -> None:
    """Claude + transcription cost for one stream roll up together, with
    a per-provider breakdown."""
    t = await TenantsRepo(migrated_db).create(name="A")
    with bound_tenant(t.id):
        repo = LLMCallsRepo(migrated_db)
        # Two Claude calls + one transcription, all on str_1.
        await repo.record(_row(t.id, stream_id="str_1", provider="anthropic",
                               input_t=20_000, output_t=17_000, cost=111_000))
        await repo.record(_row(t.id, stream_id="str_1", provider="anthropic",
                               input_t=5_000, output_t=3_000, cost=24_000))
        await repo.record(_row(t.id, stream_id="str_1", provider="assemblyai",
                               input_t=0, output_t=0, cost=43_000))
        # A different stream — must NOT be included.
        await repo.record(_row(t.id, stream_id="str_OTHER", provider="anthropic",
                               input_t=99_000, output_t=0, cost=999_000))
        spend = await repo.cost_for_stream("str_1")

    assert spend.stream_id == "str_1"
    assert spend.total_tokens == 45_000        # 37k + 8k Claude (assemblyai=0)
    assert spend.total_cost_usd_micros == 178_000  # 111k + 24k + 43k
    assert spend.total_calls == 3
    assert abs(spend.cost_usd - 0.178) < 1e-9
    by = {p.provider: p for p in spend.by_provider}
    assert by["anthropic"].cost_usd_micros == 135_000
    assert by["anthropic"].calls == 2
    assert by["assemblyai"].cost_usd_micros == 43_000


async def test_cost_for_stream_excludes_failed_calls(
    migrated_db: Database,
) -> None:
    t = await TenantsRepo(migrated_db).create(name="A")
    with bound_tenant(t.id):
        repo = LLMCallsRepo(migrated_db)
        await repo.record(_row(t.id, stream_id="str_1", provider="anthropic",
                               input_t=1_000, output_t=0, cost=5_000))
        await repo.record(_row(t.id, stream_id="str_1", provider="anthropic",
                               input_t=1_000, output_t=0, cost=5_000,
                               status="error"))
        spend = await repo.cost_for_stream("str_1")
    # Only the ok call counts.
    assert spend.total_cost_usd_micros == 5_000
    assert spend.total_calls == 1


async def test_cost_for_stream_isolated_per_tenant(
    migrated_db: Database,
) -> None:
    alice = await TenantsRepo(migrated_db).create(name="Alice")
    bob = await TenantsRepo(migrated_db).create(name="Bob")
    with bound_tenant(alice.id):
        await LLMCallsRepo(migrated_db).record(
            _row(alice.id, stream_id="str_x", provider="anthropic",
                 input_t=1_000, output_t=0, cost=9_000))
    # Bob asks for the same stream_id — sees nothing (tenant-scoped).
    with bound_tenant(bob.id):
        spend = await LLMCallsRepo(migrated_db).cost_for_stream("str_x")
    assert spend.total_cost_usd_micros == 0
    assert spend.total_calls == 0


async def test_empty_stream_returns_zeros(migrated_db: Database) -> None:
    t = await TenantsRepo(migrated_db).create(name="A")
    with bound_tenant(t.id):
        spend = await LLMCallsRepo(migrated_db).cost_for_stream("str_none")
    assert spend.total_tokens == 0
    assert spend.total_cost_usd_micros == 0
    assert spend.by_provider == []
