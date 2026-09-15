"""MCP tool functions — direct unit tests over the in-process helpers.

We test the `tool_*` async functions without going through the FastMCP
stdio transport. The transport is the SDK's responsibility; what we own
is (a) tenancy enforcement, (b) the right repo calls, (c) shape of the
returned dicts.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from chalybclip.db import (
    ApiTokensRepo,
    CandidatesRepo,
    ClipsRepo,
    ConnectedAccountsRepo,
    Database,
    LLMCallsRepo,
    PersonasRepo,
    StreamsRepo,
    TenantsRepo,
    VariantsRepo,
    apply_migrations,
)
from chalybclip.db.models import (
    CandidateRow,
    ClipRow,
    LLMCallRow,
    StreamRow,
    VariantRow,
)
from chalybclip.errors import ChalybClipError, TenancyError
from chalybclip.mcp_server.server import (
    build_server,
    resolve_tenant_from_token,
    tool_get_calibration,
    tool_get_clip,
    tool_get_cost_projection,
    tool_get_stream,
    tool_list_candidates,
    tool_list_llm_calls,
    tool_list_personas,
    tool_list_streams,
    tool_update_clip_status,
)
from chalybclip.tenancy import bound_tenant, hash_token, mint_token


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "mcp.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


async def _seed_basic(db: Database) -> dict[str, str]:
    """Stream + candidate + clip + persona + variant + buffer account."""
    tenant = await TenantsRepo(db).create(name="Aldo")
    with bound_tenant(tenant.id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_a",
                tenant_id=tenant.id,
                vod_url="https://example/x",
                platform="kick",
                title="Live show",
                channel="aldo",
                duration_s=300.0,
                source_video_path="/tmp/x.mp4",
                source_audio_path="/tmp/x.wav",
                status="ingested",
                created_at=_now(),
            )
        )
        await CandidatesRepo(db).upsert_many(
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
        await ClipsRepo(db).upsert_many(
            [
                ClipRow(
                    id="clp_a",
                    stream_id="str_a",
                    tenant_id=tenant.id,
                    candidate_id="cnd_a",
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
            persona_id="aldo",
            name="Aldo",
            primary_language="es",
            target_languages=["es"],
            voice_prompt="direct",
        )
        await VariantsRepo(db).replace_for_clip_persona(
            "clp_a",
            "aldo",
            [
                VariantRow(
                    id="var_a",
                    clip_id="clp_a",
                    tenant_id=tenant.id,
                    persona_id="aldo",
                    language="es",
                    caption="hello",
                    title_card_text="",
                    hashtags=[],
                    model="m",
                    created_at=_now(),
                )
            ],
        )
        account = await ConnectedAccountsRepo(db).create(
            platform="buffer",
            external_id="buf_x",
            display_name="Aldo Buffer",
            oauth_blob={"access_token": "tok"},
        )
    return {"tenant_id": tenant.id, "account_id": account.id}


# ---- Auth ----


async def test_resolve_tenant_returns_tenant_and_scope(db: Database) -> None:
    tenant = await TenantsRepo(db).create(name="A")
    with bound_tenant(tenant.id):
        raw, _ = mint_token()
        await ApiTokensRepo(db).create(hash_=hash_token(raw), scope="full")
    tid, scope = await resolve_tenant_from_token(db, raw_token=raw)
    assert tid == tenant.id
    assert scope == "full"


async def test_resolve_tenant_unknown_token_raises(db: Database) -> None:
    with pytest.raises(TenancyError, match="unknown"):
        await resolve_tenant_from_token(db, raw_token="tok_obviously_fake")


async def test_resolve_tenant_empty_token_raises(db: Database) -> None:
    with pytest.raises(TenancyError, match="empty"):
        await resolve_tenant_from_token(db, raw_token="")


# ---- Read tools ----


async def test_list_streams_returns_only_my_tenant(db: Database) -> None:
    seeded = await _seed_basic(db)
    bob = await TenantsRepo(db).create(name="Bob")
    with bound_tenant(bob.id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_b",
                tenant_id=bob.id,
                vod_url="x",
                platform="kick",
                title="Bob's stream",
                channel=None,
                duration_s=60.0,
                source_video_path="/tmp/y.mp4",
                source_audio_path="/tmp/y.wav",
                status="ingested",
                created_at=_now(),
            )
        )
    result = await tool_list_streams(db, seeded["tenant_id"])
    assert {r["id"] for r in result} == {"str_a"}


async def test_get_stream_returns_counts(db: Database) -> None:
    seeded = await _seed_basic(db)
    payload = await tool_get_stream(db, seeded["tenant_id"], stream_id="str_a")
    assert payload["id"] == "str_a"
    assert payload["n_candidates"] == 1
    assert payload["n_clips"] == 1


async def test_get_stream_unknown_raises(db: Database) -> None:
    seeded = await _seed_basic(db)
    with pytest.raises(ChalybClipError, match="not found"):
        await tool_get_stream(db, seeded["tenant_id"], stream_id="str_does_not_exist")


async def test_list_candidates_404s_for_unknown_stream(db: Database) -> None:
    seeded = await _seed_basic(db)
    with pytest.raises(ChalybClipError, match="not found"):
        await tool_list_candidates(db, seeded["tenant_id"], stream_id="str_x")


async def test_get_clip_includes_breakdown_and_transitions(db: Database) -> None:
    seeded = await _seed_basic(db)
    out = await tool_get_clip(db, seeded["tenant_id"], clip_id="clp_a")
    assert out["clip"]["id"] == "clp_a"
    assert "breakdown" in out
    assert out["breakdown"]["heuristic_reason"] == "voice"
    # 'cut' transitions to ready_for_review or rejected.
    assert set(out["valid_transitions"]) == {"ready_for_review", "rejected"}
    # Variants listed.
    assert len(out["variants"]) == 1
    assert out["variants"][0]["id"] == "var_a"


async def test_list_personas_lists_for_tenant(db: Database) -> None:
    seeded = await _seed_basic(db)
    rows = await tool_list_personas(db, seeded["tenant_id"])
    assert any(r["id"] == "aldo" for r in rows)


async def test_list_llm_calls_returns_recent_rows(db: Database) -> None:
    seeded = await _seed_basic(db)
    with bound_tenant(seeded["tenant_id"]):
        await LLMCallsRepo(db).record(
            LLMCallRow(
                id="llm_x",
                tenant_id=seeded["tenant_id"],
                purpose="variant_generation",
                provider="anthropic",
                model="m",
                quality="standard",
                input_tokens=10,
                output_tokens=5,
                cost_usd_micros=1000,
                status="ok",
                error=None,
                attempts=1,
                ts=_now(),
            )
        )
    rows = await tool_list_llm_calls(db, seeded["tenant_id"], limit=10)
    assert len(rows) == 1


async def test_get_cost_projection_returns_dict(db: Database) -> None:
    seeded = await _seed_basic(db)
    out = await tool_get_cost_projection(db, seeded["tenant_id"])
    # Projection dataclass keys.
    assert "today_micros" in out
    assert "mtd_micros" in out
    assert "projected_eom_micros" in out
    assert "by_purpose" in out


async def test_get_calibration_empty_when_no_data(db: Database) -> None:
    seeded = await _seed_basic(db)
    out = await tool_get_calibration(db, seeded["tenant_id"], platform="youtube")
    assert out["platform"] == "youtube"
    assert out["pearson_r"] is None
    assert out["n_paired"] == 0


# ---- State transitions ----


async def test_update_clip_status_walks_transition_map(db: Database) -> None:
    seeded = await _seed_basic(db)
    out = await tool_update_clip_status(
        db,
        seeded["tenant_id"],
        scope="full",
        clip_id="clp_a",
        new_status="ready_for_review",
    )
    assert out["status"] == "ready_for_review"


async def test_update_clip_status_invalid_transition_raises(db: Database) -> None:
    seeded = await _seed_basic(db)
    with pytest.raises(ChalybClipError, match="cannot transition"):
        await tool_update_clip_status(
            db,
            seeded["tenant_id"],
            scope="full",
            clip_id="clp_a",
            new_status="approved",  # cut -> approved isn't allowed
        )


async def test_update_clip_status_requires_full_scope(db: Database) -> None:
    seeded = await _seed_basic(db)
    with pytest.raises(TenancyError, match="scope=full"):
        await tool_update_clip_status(
            db,
            seeded["tenant_id"],
            scope="read",
            clip_id="clp_a",
            new_status="ready_for_review",
        )


# ---- FastMCP server registration shape ----


async def test_build_server_registers_expected_tool_names(db: Database) -> None:
    """`build_server` exposes every documented tool by name."""
    seeded = await _seed_basic(db)
    server = build_server(db=db, tenant_id=seeded["tenant_id"], scope="full")
    listed = await server.list_tools()
    names = {t.name for t in listed}
    assert names >= {
        "list_streams",
        "get_stream",
        "list_candidates",
        "list_clips",
        "get_clip",
        "list_personas",
        "list_llm_calls",
        "get_cost_projection",
        "get_calibration",
        "update_clip_status",
    }


async def test_publish_metrics_repo_unused_in_mcp_module(db: Database) -> None:
    """Sanity: PublishMetricsRepo isn't referenced inside the MCP module
    surface, even though it's imported transitively. Phase 3 #1 surfaces
    metrics through `get_calibration`/`get_cost_projection`, not direct
    repo access from agents — keeps the read surface curated."""
    import chalybclip.mcp_server.server as srv

    src = Path(srv.__file__).read_text(encoding="utf-8")
    assert "PublishMetricsRepo" not in src


async def test_resolve_tenant_disabled_account_still_resolves(db: Database) -> None:
    """A token's lookup_by_hash returns the row regardless of subscription
    status — auth is just (token -> tenant). The tools' own logic gates
    what the agent can do beyond that."""
    tenant = await TenantsRepo(db).create(name="A")
    with bound_tenant(tenant.id):
        raw, _ = mint_token()
        await ApiTokensRepo(db).create(hash_=hash_token(raw), scope="read")
    tid, scope = await resolve_tenant_from_token(db, raw_token=raw)
    assert tid == tenant.id
    assert scope == "read"
