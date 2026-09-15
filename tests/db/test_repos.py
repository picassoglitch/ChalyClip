"""Tests for the repository layer — the *primary* lock-down deliverable.

Every test here is checking one of:
  * round-trip insert/get works for a given table
  * cross-tenant access returns None / raises (defense-in-depth at the SQL boundary)
  * foreign-key constraints are enforced (no silent dangling rows)
"""

from __future__ import annotations

import datetime as _dt

import pytest

from chalybclip.db import (
    ApiTokensRepo,
    Database,
    EventsRepo,
    LLMCallsRepo,
    PersonasRepo,
    StreamsRepo,
    TenantsRepo,
    UsersRepo,
)
from chalybclip.db.models import LLMCallRow, StreamRow
from chalybclip.errors import TenancyError
from chalybclip.tenancy import bound_tenant, hash_token, mint_token

from .._db_backend import INTEGRITY_ERRORS


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


# ---------- Tenants (the one table not filtered by tenant_id) ----------


async def test_tenants_create_and_get(migrated_db: Database) -> None:
    repo = TenantsRepo(migrated_db)
    t = await repo.create(tenant_id="ten_a", name="Alice Co")
    assert t.id == "ten_a"
    assert t.name == "Alice Co"
    fetched = await repo.get("ten_a")
    assert fetched == t


async def test_tenants_list_all(migrated_db: Database) -> None:
    repo = TenantsRepo(migrated_db)
    await repo.create(tenant_id="ten_a", name="Alice")
    await repo.create(tenant_id="ten_b", name="Bob")
    listed = await repo.list_all()
    assert {t.id for t in listed} == {"ten_a", "ten_b"}


async def test_tenants_get_missing_returns_none(migrated_db: Database) -> None:
    assert await TenantsRepo(migrated_db).get("ten_nope") is None


# ---------- Users ----------


async def _seed_two_tenants(db: Database) -> None:
    repo = TenantsRepo(db)
    await repo.create(tenant_id="ten_a", name="A")
    await repo.create(tenant_id="ten_b", name="B")


async def test_users_round_trip(migrated_db: Database) -> None:
    await _seed_two_tenants(migrated_db)
    repo = UsersRepo(migrated_db)
    with bound_tenant("ten_a"):
        u = await repo.create(email="alice@a.com")
        assert u.tenant_id == "ten_a"
        fetched = await repo.get(u.id)
        assert fetched == u


async def test_users_cross_tenant_get_returns_none(migrated_db: Database) -> None:
    await _seed_two_tenants(migrated_db)
    repo = UsersRepo(migrated_db)
    with bound_tenant("ten_a"):
        u = await repo.create(email="alice@a.com")
    with bound_tenant("ten_b"):
        # Tenant B asking for A's user ID gets nothing.
        assert await repo.get(u.id) is None
        # Listing for B returns nothing.
        assert await repo.list_for_tenant() == []


async def test_users_create_requires_bound_tenant(migrated_db: Database) -> None:
    repo = UsersRepo(migrated_db)
    with pytest.raises(TenancyError, match="no tenant bound"):
        await repo.create(email="x@y.com")


# ---------- API tokens ----------


async def test_api_token_round_trip(migrated_db: Database) -> None:
    await _seed_two_tenants(migrated_db)
    repo = ApiTokensRepo(migrated_db)
    raw, hashed = mint_token()
    with bound_tenant("ten_a"):
        row = await repo.create(hash_=hashed, scope="full")
        assert row.tenant_id == "ten_a"
        assert row.hash == hashed
        # Raw token is NEVER stored.
        assert raw not in row.hash


async def test_api_token_lookup_by_hash_is_unscoped(migrated_db: Database) -> None:
    """The auth path: lookup_by_hash returns the row regardless of bound tenant."""
    await _seed_two_tenants(migrated_db)
    repo = ApiTokensRepo(migrated_db)
    raw, hashed = mint_token()
    with bound_tenant("ten_a"):
        await repo.create(hash_=hashed, scope="full")
    # No tenant bound — auth lookup must still find the row.
    found = await repo.lookup_by_hash(hashed)
    assert found is not None
    assert found.tenant_id == "ten_a"


async def test_api_token_lookup_by_hash_uses_hash_not_raw(migrated_db: Database) -> None:
    """The raw token isn't the equality key; only the hash is."""
    await _seed_two_tenants(migrated_db)
    repo = ApiTokensRepo(migrated_db)
    raw, hashed = mint_token()
    with bound_tenant("ten_a"):
        await repo.create(hash_=hashed)
    # Looking up with the raw token (not its hash) finds nothing.
    assert await repo.lookup_by_hash(raw) is None
    # Looking up with the correct hash works.
    assert (await repo.lookup_by_hash(hashed)) is not None


async def test_api_token_lookup_updates_last_used_at(migrated_db: Database) -> None:
    await _seed_two_tenants(migrated_db)
    repo = ApiTokensRepo(migrated_db)
    raw, hashed = mint_token()
    with bound_tenant("ten_a"):
        row = await repo.create(hash_=hashed)
    assert row.last_used_at is None
    await repo.lookup_by_hash(hashed)
    with bound_tenant("ten_a"):
        rows = await repo.list_for_tenant()
    assert rows[0].last_used_at is not None


async def test_hash_token_does_not_round_trip_to_raw() -> None:
    raw = "tok_01ABCDEFGHJKMNPQRSTVWXYZ12"
    hashed = hash_token(raw)
    assert hashed != raw
    assert len(hashed) == 64  # sha256 hex


# ---------- Personas ----------


async def test_personas_round_trip(migrated_db: Database) -> None:
    await _seed_two_tenants(migrated_db)
    repo = PersonasRepo(migrated_db)
    with bound_tenant("ten_a"):
        p = await repo.create(
            persona_id="aldo_villanueva",
            name="Aldo",
            primary_language="es",
            target_languages=["es", "en"],
            voice_prompt="Direct.",
            routing_tags=["mindset"],
        )
        assert p.target_languages == ["es", "en"]
        assert p.routing_tags == ["mindset"]
        fetched = await repo.get("aldo_villanueva")
        assert fetched == p


async def test_personas_cross_tenant_get_returns_none(migrated_db: Database) -> None:
    await _seed_two_tenants(migrated_db)
    repo = PersonasRepo(migrated_db)
    with bound_tenant("ten_a"):
        await repo.create(
            persona_id="x",
            name="X",
            primary_language="es",
            target_languages=["es"],
            voice_prompt="...",
        )
    with bound_tenant("ten_b"):
        assert await repo.get("x") is None
        assert await repo.list_for_tenant() == []


# ---------- Streams ----------


def _stream_row(*, stream_id: str, tenant_id: str) -> StreamRow:
    return StreamRow(
        id=stream_id,
        tenant_id=tenant_id,
        vod_url="https://kick.com/c/videos/1",
        platform="kick",
        title="t",
        channel="c",
        duration_s=600.0,
        source_video_path=f"/out/{stream_id}/source/video.mp4",
        source_audio_path=f"/out/{stream_id}/source/audio.wav",
        status="ingested",
        created_at=_now(),
    )


async def test_streams_upsert_is_idempotent(migrated_db: Database) -> None:
    await _seed_two_tenants(migrated_db)
    repo = StreamsRepo(migrated_db)
    with bound_tenant("ten_a"):
        first = await repo.upsert(_stream_row(stream_id="str_1", tenant_id="ten_a"))
        second = await repo.upsert(_stream_row(stream_id="str_1", tenant_id="ten_a"))
        assert first == second


async def test_streams_cross_tenant_get_returns_none(migrated_db: Database) -> None:
    await _seed_two_tenants(migrated_db)
    repo = StreamsRepo(migrated_db)
    with bound_tenant("ten_a"):
        await repo.upsert(_stream_row(stream_id="str_1", tenant_id="ten_a"))
    with bound_tenant("ten_b"):
        assert await repo.get("str_1") is None
        assert await repo.list_for_tenant() == []


async def test_streams_upsert_rejects_mismatched_tenant(migrated_db: Database) -> None:
    """Insert a row whose tenant_id doesn't match the bound tenant -> raise."""
    await _seed_two_tenants(migrated_db)
    repo = StreamsRepo(migrated_db)
    row = _stream_row(stream_id="str_1", tenant_id="ten_b")
    with bound_tenant("ten_a"):
        with pytest.raises(TenancyError, match="!="):
            await repo.upsert(row)


async def test_streams_unbound_tenant_raises(migrated_db: Database) -> None:
    await _seed_two_tenants(migrated_db)
    repo = StreamsRepo(migrated_db)
    with pytest.raises(TenancyError):
        await repo.get("anything")


# ---------- LLM calls (audit log) ----------


def _llm_row(*, tenant_id: str = "ten_a") -> LLMCallRow:
    return LLMCallRow(
        id="llm_1",
        tenant_id=tenant_id,
        purpose="variant_generation",
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        quality="standard",
        input_tokens=100,
        output_tokens=50,
        cost_usd_micros=280,
        ts=_now(),
    )


async def test_llm_calls_record_and_list(migrated_db: Database) -> None:
    await _seed_two_tenants(migrated_db)
    repo = LLMCallsRepo(migrated_db)
    with bound_tenant("ten_a"):
        await repo.record(_llm_row())
        rows = await repo.list_for_tenant()
        assert len(rows) == 1
        assert rows[0].cost_usd_micros == 280


async def test_llm_calls_cross_tenant_isolated(migrated_db: Database) -> None:
    await _seed_two_tenants(migrated_db)
    repo = LLMCallsRepo(migrated_db)
    with bound_tenant("ten_a"):
        await repo.record(_llm_row())
    with bound_tenant("ten_b"):
        assert await repo.list_for_tenant() == []


async def test_llm_calls_record_rejects_mismatched_tenant(migrated_db: Database) -> None:
    await _seed_two_tenants(migrated_db)
    repo = LLMCallsRepo(migrated_db)
    row = _llm_row(tenant_id="ten_b")
    with bound_tenant("ten_a"):
        with pytest.raises(TenancyError):
            await repo.record(row)


# ---------- Events ----------


async def test_events_emit_and_list(migrated_db: Database) -> None:
    await _seed_two_tenants(migrated_db)
    repo = EventsRepo(migrated_db)
    with bound_tenant("ten_a"):
        e1 = await repo.emit(type="stream.created", payload={"stream_id": "s1"})
        e2 = await repo.emit(type="clip.ready_for_review")
        assert e1.tenant_id == "ten_a"
        assert e1.payload == {"stream_id": "s1"}
        assert e2.payload == {}
        events = await repo.list_for_tenant()
        assert {e.type for e in events} == {
            "stream.created",
            "clip.ready_for_review",
        }


async def test_events_filter_by_type(migrated_db: Database) -> None:
    await _seed_two_tenants(migrated_db)
    repo = EventsRepo(migrated_db)
    with bound_tenant("ten_a"):
        await repo.emit(type="x")
        await repo.emit(type="y")
        await repo.emit(type="x")
        xs = await repo.list_for_tenant(type="x")
        assert len(xs) == 2 and all(e.type == "x" for e in xs)


async def test_events_cross_tenant_isolated(migrated_db: Database) -> None:
    await _seed_two_tenants(migrated_db)
    repo = EventsRepo(migrated_db)
    with bound_tenant("ten_a"):
        await repo.emit(type="x")
    with bound_tenant("ten_b"):
        assert await repo.list_for_tenant() == []


# ---------- Clips: status counts ----------


async def test_clips_count_for_tenant_with_status(migrated_db: Database) -> None:
    """The un-capped COUNT companion to list_for_tenant_with_status —
    labels the Auto-programar button with the REAL backlog size instead
    of the 60-row display page. Tenant-scoped like everything else."""
    from chalybclip.db import ClipsRepo
    from chalybclip.db.models import ClipRow

    await _seed_two_tenants(migrated_db)
    with bound_tenant("ten_a"):
        await StreamsRepo(migrated_db).upsert(
            StreamRow(
                id="str_c", tenant_id="ten_a", vod_url="https://kick.com/x",
                platform="kick", title="t", channel="c", duration_s=60.0,
                source_video_path="/tmp/v", source_audio_path="/tmp/a",
                status="ingested", created_at=_now(),
            )
        )
        await ClipsRepo(migrated_db).upsert_many(
            [
                ClipRow(
                    id=f"clp_{i}", stream_id="str_c", tenant_id="ten_a",
                    start_s=0.0, end_s=10.0, duration_s=10.0,
                    width=1080, height=1920, path=f"/tmp/{i}.mp4",
                    status=status, created_at=_now(),
                )
                for i, status in enumerate(["approved", "approved", "published"])
            ]
        )
        repo = ClipsRepo(migrated_db)
        assert await repo.count_for_tenant_with_status(["approved"]) == 2
        assert await repo.count_for_tenant_with_status(["approved", "published"]) == 3
        assert await repo.count_for_tenant_with_status([]) == 0
    with bound_tenant("ten_b"):
        assert await ClipsRepo(migrated_db).count_for_tenant_with_status(
            ["approved"]
        ) == 0  # tenant-isolated


# ---------- Foreign keys + cascades ----------


async def test_inserting_user_with_unknown_tenant_raises_fk(
    migrated_db: Database,
) -> None:
    """Direct SQL: attempting to create a child of a non-existent tenant fails."""
    conn = await migrated_db.connect()
    with pytest.raises(INTEGRITY_ERRORS):
        await conn.execute(
            "INSERT INTO users (id, tenant_id, email, role, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("usr_x", "ten_ghost", "x@y.com", "owner", _now()),
        )
        await conn.commit()


async def test_deleting_tenant_cascades_to_streams(migrated_db: Database) -> None:
    """tenants→streams is CASCADE on delete (data goes); api_tokens is RESTRICT.

    Restrict means we can't actually delete a tenant that still has tokens.
    Test the cascade indirectly: insert a stream, then drop a tenant that has
    no tokens — streams disappear too.
    """
    await _seed_two_tenants(migrated_db)
    repo = StreamsRepo(migrated_db)
    with bound_tenant("ten_a"):
        await repo.upsert(_stream_row(stream_id="str_1", tenant_id="ten_a"))
    conn = await migrated_db.connect()
    await conn.execute("DELETE FROM tenants WHERE id = 'ten_a'")
    await conn.commit()
    cur = await conn.execute(
        "SELECT COUNT(*) FROM streams WHERE tenant_id = 'ten_a'"
    )
    assert (await cur.fetchone())[0] == 0


async def test_deleting_tenant_with_tokens_is_restricted(
    migrated_db: Database,
) -> None:
    """tenants -> api_tokens is RESTRICT: can't drop a tenant with tokens."""
    await _seed_two_tenants(migrated_db)
    _, hashed = mint_token()
    with bound_tenant("ten_a"):
        await ApiTokensRepo(migrated_db).create(hash_=hashed)
    conn = await migrated_db.connect()
    with pytest.raises(INTEGRITY_ERRORS):
        await conn.execute("DELETE FROM tenants WHERE id = 'ten_a'")
        await conn.commit()


# ---------- StreamsRepo.reconcile_metadata (URL-job placeholder backfill) ----


def _placeholder_stream_row(*, stream_id: str, tenant_id: str) -> StreamRow:
    """Mimics the URL-job placeholder: pending, 0s, no title/channel."""
    return StreamRow(
        id=stream_id,
        tenant_id=tenant_id,
        vod_url="https://www.youtube.com/watch?v=abc",
        platform="youtube",
        title=None,
        channel=None,
        duration_s=0.0,
        source_video_path=f"/out/{stream_id}/source/video.mp4",
        source_audio_path=f"/out/{stream_id}/source/audio.wav",
        status="pending",
        created_at=_now(),
    )


async def test_reconcile_metadata_fills_placeholder(migrated_db: Database) -> None:
    await _seed_two_tenants(migrated_db)
    repo = StreamsRepo(migrated_db)
    with bound_tenant("ten_a"):
        await repo.upsert(_placeholder_stream_row(stream_id="str_1", tenant_id="ten_a"))
        await repo.reconcile_metadata(
            "str_1", title="Real Title", channel="real_channel", duration_s=1924.0
        )
        row = await repo.get("str_1")
    assert row is not None
    assert row.title == "Real Title"
    assert row.channel == "real_channel"
    assert row.duration_s == 1924.0
    assert row.status == "ingested"  # pending -> ingested


async def test_reconcile_metadata_does_not_clobber_real_values(
    migrated_db: Database,
) -> None:
    await _seed_two_tenants(migrated_db)
    repo = StreamsRepo(migrated_db)
    with bound_tenant("ten_a"):
        # A fully-populated, non-placeholder row (e.g. channel/live path).
        await repo.upsert(_stream_row(stream_id="str_1", tenant_id="ten_a"))
        await repo.reconcile_metadata(
            "str_1", title="OTHER", channel="other", duration_s=99.0
        )
        row = await repo.get("str_1")
    assert row is not None
    # _stream_row sets title="t", channel="c", duration 600, status ingested.
    assert row.title == "t"
    assert row.channel == "c"
    assert row.duration_s == 600.0
    assert row.status == "ingested"
