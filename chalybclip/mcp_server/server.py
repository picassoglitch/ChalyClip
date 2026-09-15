"""FastMCP server definition + tool registry.

Each tool is an `async def` that accepts plain JSON-shaped arguments and
returns a JSON-shaped dict (or list-of-dicts). FastMCP auto-generates the
JSON Schema from type hints + docstrings.

The server runs over stdio by default — that's the transport Claude Code
and Cursor expect for local MCP servers.

Tools split:
    Reads: list_streams, get_stream, list_candidates, list_clips,
           get_clip (with breakdown), list_personas, list_llm_calls,
           get_calibration, get_cost_projection.
    State transitions (full-scope only): update_clip_status.

Persona / connected-account writes stay out of MCP — those are dashboard
flows (the operator sets up tenancy + integrations once; agents act
inside that context).
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from mcp.server.fastmcp import FastMCP

from chalybclip.clip import clip_breakdown
from chalybclip.cost import compute_cost_projection
from chalybclip.db import (
    ApiTokensRepo,
    CandidatesRepo,
    ClipsRepo,
    Database,
    EventsRepo,
    LLMCallsRepo,
    PersonasRepo,
    StreamsRepo,
    VariantsRepo,
    apply_migrations,
)
from chalybclip.errors import ChalybClipError, TenancyError
from chalybclip.metrics import compute_calibration
from chalybclip.tenancy import bound_tenant, hash_token

_log = structlog.get_logger(__name__)

# clip status transition map mirrors what the REST PATCH /clips/{id} accepts;
# we duplicate the (small) table here so this module stays decoupled from the
# api/routers/clips.py import (which would drag fastapi into the mcp surface).
_VALID_CLIP_TRANSITIONS: dict[str, set[str]] = {
    "cut": {"ready_for_review", "rejected"},
    "ready_for_review": {"approved", "rejected"},
    "approved": {"rejected", "published"},
    "rejected": {"ready_for_review"},
    "published": set(),
}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


async def resolve_tenant_from_token(db: Database, *, raw_token: str) -> tuple[str, str]:
    """Resolve `(tenant_id, scope)` from a raw API token. Raises on failure.

    The caller passes the raw token they received once at issuance; we
    sha256 it and look up `api_tokens.hash`. No cross-tenant context
    leaks — rejected tokens fail fast at server boot.
    """
    if not raw_token:
        raise TenancyError("empty bearer token")
    try:
        token_hash = hash_token(raw_token.strip())
    except Exception as e:
        raise TenancyError(f"invalid token format: {e}") from e
    row = await ApiTokensRepo(db).lookup_by_hash(token_hash)
    if row is None:
        raise TenancyError("unknown token")
    return row.tenant_id, row.scope


# ---------------------------------------------------------------------------
# Tool implementations (plain async helpers — easy to unit-test directly)
# ---------------------------------------------------------------------------


async def tool_list_streams(db: Database, tenant_id: str) -> list[dict[str, Any]]:
    """Return all streams for the bound tenant (newest first)."""
    with bound_tenant(tenant_id):
        rows = await StreamsRepo(db).list_for_tenant()
    return [r.model_dump() for r in rows]


async def tool_get_stream(
    db: Database, tenant_id: str, *, stream_id: str
) -> dict[str, Any]:
    """Return one stream + counts (candidates, clips). 404 raises ChalybClipError."""
    with bound_tenant(tenant_id):
        stream = await StreamsRepo(db).get(stream_id)
        if stream is None:
            raise ChalybClipError(f"stream not found: {stream_id}")
        candidates = await CandidatesRepo(db).list_for_stream(stream_id)
        clips = await ClipsRepo(db).list_for_stream(stream_id)
    return {
        **stream.model_dump(),
        "n_candidates": len(candidates),
        "n_clips": len(clips),
    }


async def tool_list_candidates(
    db: Database, tenant_id: str, *, stream_id: str
) -> list[dict[str, Any]]:
    """Return every candidate for a stream, with rescore columns when present."""
    with bound_tenant(tenant_id):
        if await StreamsRepo(db).get(stream_id) is None:
            raise ChalybClipError(f"stream not found: {stream_id}")
        rows = await CandidatesRepo(db).list_for_stream(stream_id)
    return [r.model_dump() for r in rows]


async def tool_list_clips(
    db: Database, tenant_id: str, *, stream_id: str
) -> list[dict[str, Any]]:
    """Return every clip for a stream."""
    with bound_tenant(tenant_id):
        if await StreamsRepo(db).get(stream_id) is None:
            raise ChalybClipError(f"stream not found: {stream_id}")
        rows = await ClipsRepo(db).list_for_stream(stream_id)
    return [r.model_dump() for r in rows]


async def tool_get_clip(
    db: Database, tenant_id: str, *, clip_id: str
) -> dict[str, Any]:
    """Return clip detail + variants + the confidence-breakdown panel data."""
    with bound_tenant(tenant_id):
        clip = await ClipsRepo(db).get(clip_id)
        if clip is None:
            raise ChalybClipError(f"clip not found: {clip_id}")
        variants = await VariantsRepo(db).list_for_clip(clip_id)
        breakdown = await clip_breakdown(db, clip_id)
    return {
        "clip": clip.model_dump(),
        "variants": [v.model_dump() for v in variants],
        "breakdown": dataclasses.asdict(breakdown),
        "valid_transitions": sorted(
            _VALID_CLIP_TRANSITIONS.get(clip.status, set())
        ),
    }


async def tool_list_personas(db: Database, tenant_id: str) -> list[dict[str, Any]]:
    with bound_tenant(tenant_id):
        rows = await PersonasRepo(db).list_for_tenant()
    return [r.model_dump() for r in rows]


async def tool_list_llm_calls(
    db: Database, tenant_id: str, *, limit: int = 50
) -> list[dict[str, Any]]:
    """Recent LLM calls — useful for cost monitoring from an agent."""
    with bound_tenant(tenant_id):
        rows = await LLMCallsRepo(db).list_for_tenant(limit=max(1, min(limit, 1000)))
    return [r.model_dump() for r in rows]


async def tool_get_cost_projection(
    db: Database, tenant_id: str
) -> dict[str, Any]:
    """Today / MTD / projected EOM USD totals + per-purpose / per-model breakdowns."""
    with bound_tenant(tenant_id):
        proj = await compute_cost_projection(db)
    return dataclasses.asdict(proj)


async def tool_get_calibration(
    db: Database, tenant_id: str, *, platform: str, since_days: int = 30
) -> dict[str, Any]:
    """Vision-rescore calibration report for one platform.

    `pearson_r` is None when fewer than 3 paired (rescore_score, views)
    samples exist or either axis has zero variance.
    """
    with bound_tenant(tenant_id):
        report = await compute_calibration(
            db, platform=platform, since_days=since_days
        )
    return {
        "platform": report.platform,
        "pearson_r": report.pearson_r,
        "n_paired": report.n_paired,
        "rows": [dataclasses.asdict(r) for r in report.rows],
    }


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


async def tool_update_clip_status(
    db: Database,
    tenant_id: str,
    scope: str,
    *,
    clip_id: str,
    new_status: str,
) -> dict[str, Any]:
    """Move a clip through its review lifecycle.

    Mirrors the REST `PATCH /clips/{id}` transition map exactly:
      cut             -> ready_for_review | rejected
      ready_for_review -> approved | rejected
      approved        -> rejected | published
      rejected        -> ready_for_review
      published       -> (terminal)

    Requires `scope='full'`.
    """
    if scope != "full":
        raise TenancyError(f"requires scope=full, token has scope={scope!r}")
    with bound_tenant(tenant_id):
        clip = await ClipsRepo(db).get(clip_id)
        if clip is None:
            raise ChalybClipError(f"clip not found: {clip_id}")
        allowed = _VALID_CLIP_TRANSITIONS.get(clip.status, set())
        if new_status not in allowed:
            raise ChalybClipError(
                f"cannot transition clip from {clip.status!r} to {new_status!r}; "
                f"allowed: {sorted(allowed)}"
            )
        conn = await db.connect()
        await conn.execute(
            "UPDATE clips SET status = ? WHERE id = ? AND tenant_id = ?",
            (new_status, clip_id, tenant_id),
        )
        await conn.commit()
        await EventsRepo(db).emit(
            type=f"clip.{new_status}",
            payload={"clip_id": clip_id, "from": clip.status, "to": new_status},
        )
        refreshed = await ClipsRepo(db).get(clip_id)
    assert refreshed is not None
    return refreshed.model_dump()


# ---------------------------------------------------------------------------
# FastMCP wiring
# ---------------------------------------------------------------------------


def build_server(
    *,
    db: Database,
    tenant_id: str,
    scope: str,
    name: str = "chalybclip",
) -> FastMCP:
    """Return a FastMCP server with all ChalybClip tools registered.

    Tests can build a server, hand-call its registered tools through the
    in-process tool functions exported above, and skip the stdio bridge.
    Production callers go through `run_stdio_server(...)`.
    """
    server = FastMCP(name=name)

    # Reads ----------------------------------------------------------------

    @server.tool(name="list_streams", description="List every stream for this tenant.")
    async def _list_streams() -> list[dict[str, Any]]:
        return await tool_list_streams(db, tenant_id)

    @server.tool(name="get_stream", description="Return one stream + candidate / clip counts.")
    async def _get_stream(stream_id: str) -> dict[str, Any]:
        return await tool_get_stream(db, tenant_id, stream_id=stream_id)

    @server.tool(name="list_candidates", description="Return every candidate for a stream.")
    async def _list_candidates(stream_id: str) -> list[dict[str, Any]]:
        return await tool_list_candidates(db, tenant_id, stream_id=stream_id)

    @server.tool(name="list_clips", description="Return every clip for a stream.")
    async def _list_clips(stream_id: str) -> list[dict[str, Any]]:
        return await tool_list_clips(db, tenant_id, stream_id=stream_id)

    @server.tool(
        name="get_clip",
        description=(
            "Return clip detail + variants + the confidence-breakdown panel "
            "(motion / face presence / speaking intensity / reaction "
            "confidence / rescore delta) + the list of allowed status "
            "transitions from the clip's current state."
        ),
    )
    async def _get_clip(clip_id: str) -> dict[str, Any]:
        return await tool_get_clip(db, tenant_id, clip_id=clip_id)

    @server.tool(name="list_personas", description="List every persona for this tenant.")
    async def _list_personas() -> list[dict[str, Any]]:
        return await tool_list_personas(db, tenant_id)

    @server.tool(name="list_llm_calls", description="Recent LLM calls (cost monitoring).")
    async def _list_llm_calls(limit: int = 50) -> list[dict[str, Any]]:
        return await tool_list_llm_calls(db, tenant_id, limit=limit)

    @server.tool(
        name="get_cost_projection",
        description=(
            "Today / MTD / projected EOM USD totals + per-purpose and "
            "per-model breakdowns + budget headroom fraction (or null when "
            "no daily LLM budget cap is set)."
        ),
    )
    async def _get_cost_projection() -> dict[str, Any]:
        return await tool_get_cost_projection(db, tenant_id)

    @server.tool(
        name="get_calibration",
        description=(
            "Vision-rescore calibration for one platform: Pearson r over "
            "(rescore_score, views) for jobs in the last `since_days` days. "
            "pearson_r is null when fewer than 3 paired samples exist."
        ),
    )
    async def _get_calibration(
        platform: str, since_days: int = 30
    ) -> dict[str, Any]:
        return await tool_get_calibration(
            db, tenant_id, platform=platform, since_days=since_days
        )

    # State transitions ----------------------------------------------------

    @server.tool(
        name="update_clip_status",
        description=(
            "Move a clip through the review lifecycle. Allowed transitions: "
            "cut->{ready_for_review,rejected}; ready_for_review->{approved,"
            "rejected}; approved->{rejected,published}; rejected->"
            "ready_for_review. Requires scope=full."
        ),
    )
    async def _update_clip_status(clip_id: str, new_status: str) -> dict[str, Any]:
        return await tool_update_clip_status(
            db, tenant_id, scope, clip_id=clip_id, new_status=new_status
        )

    return server


# ---------------------------------------------------------------------------
# Stdio runner (production entry point)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _open_db_for_token(db_path: Path | str, raw_token: str):  # type: ignore[no-untyped-def]
    """Open + migrate the DB, resolve the token to a tenant, yield the trio."""
    db = Database(db_path)
    try:
        await apply_migrations(db)
        tenant_id, scope = await resolve_tenant_from_token(db, raw_token=raw_token)
        yield db, tenant_id, scope
    finally:
        await db.close()


def run_stdio_server(
    *,
    db_path: Path | str,
    raw_token: str | None = None,
    token_env: str = "CHALYBCLIP_API_TOKEN",
) -> None:
    """Boot the stdio MCP server.

    Reads the token from `raw_token` if supplied, else from the env var
    `CHALYBCLIP_API_TOKEN`. Fails fast on missing / invalid token so the
    agent harness sees the error before any tool call lands.

    Blocks until the parent process closes stdin (the MCP shutdown signal).
    """
    token = raw_token or os.environ.get(token_env, "").strip()
    if not token:
        raise TenancyError(
            f"no API token; pass --token or set {token_env}"
        )

    async def _main() -> None:
        async with _open_db_for_token(db_path, token) as (db, tenant_id, scope):
            _log.info(
                "mcp_server_starting",
                tenant_id=tenant_id,
                scope=scope,
                db_path=str(db_path),
            )
            server = build_server(db=db, tenant_id=tenant_id, scope=scope)
            await server.run_stdio_async()

    asyncio.run(_main())
