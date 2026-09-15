"""Shared fixtures for API tests.

Each test gets a fresh DB seeded with two tenants (`alice` and `bob`),
each with one full-scope token. The `client` fixture builds a FastAPI
app over that DB and returns an httpx AsyncClient configured for ASGI
transport (no real network).

The pipeline runner is stubbed so `POST /streams` doesn't try to spin
Whisper/CV up.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest_asyncio

from chalybclip.api import create_app
from chalybclip.db import (
    ApiTokensRepo,
    Database,
    TenantsRepo,
)
from chalybclip.tenancy import bound_tenant, hash_token, mint_token

from .._db_backend import migrated_database


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    """Migrated DB per test — Postgres when CHALYBCLIP_TEST_PG_DSN is set
    (truncate-isolated), else a fresh SQLite file. The "api.db" name matches
    the db_path some background-task tests hand the renderer."""
    async for d in migrated_database(tmp_path, name="api.db"):
        yield d


@pytest_asyncio.fixture
async def tenants(db: Database) -> dict[str, dict[str, str]]:
    """Seed two tenants with one full-scope token each.

    Returns `{"alice": {"id": "ten_...", "token": "tok_..."}, "bob": {...}}`
    so individual tests can grab the bearer they need.
    """
    out: dict[str, dict[str, str]] = {}
    for handle, name in (("alice", "Alice Co"), ("bob", "Bob LLC")):
        tenant = await TenantsRepo(db).create(name=name)
        with bound_tenant(tenant.id):
            raw, _hash = mint_token()
            await ApiTokensRepo(db).create(hash_=hash_token(raw), scope="full")
        out[handle] = {"id": tenant.id, "token": raw}
    return out


async def _noop_pipeline(_kickoff: object) -> None:
    """Default pipeline stub for API tests - the kickoff is verified separately."""


@pytest_asyncio.fixture
async def client(db: Database) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(db=db, pipeline_runner=_noop_pipeline)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as c:
        yield c


def auth(token: str) -> dict[str, str]:
    """Build the Authorization header for a bearer token."""
    return {"Authorization": f"Bearer {token}"}
