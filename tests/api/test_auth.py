"""Auth middleware: bearer required, unknown tokens rejected, scope enforced."""

from __future__ import annotations

import httpx

from chalybclip.db import ApiTokensRepo, Database
from chalybclip.tenancy import bound_tenant, hash_token, mint_token

from .conftest import auth


async def test_healthz_is_public(client: httpx.AsyncClient) -> None:
    """`/healthz` doesn't require auth so monitoring can hit it."""
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_missing_bearer_is_401(client: httpx.AsyncClient) -> None:
    r = await client.get("/streams")
    assert r.status_code == 401
    assert r.headers.get("www-authenticate") == "Bearer"


async def test_unknown_token_is_401(client: httpx.AsyncClient) -> None:
    r = await client.get("/streams", headers=auth("tok_obviously_fake"))
    assert r.status_code == 401


async def test_known_token_grants_tenant_access(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    r = await client.get("/streams", headers=auth(tenants["alice"]["token"]))
    assert r.status_code == 200
    assert r.json() == []


async def test_read_scope_blocks_writes(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    """A read-only token can list streams but can't create personas."""
    tenant_id = tenants["alice"]["id"]
    with bound_tenant(tenant_id):
        raw, _ = mint_token()
        await ApiTokensRepo(db).create(hash_=hash_token(raw), scope="read")

    # Read works.
    r = await client.get("/personas", headers=auth(raw))
    assert r.status_code == 200

    # Write does not.
    r = await client.post(
        "/personas",
        json={
            "id": "p1",
            "name": "Test",
            "primary_language": "en",
            "voice_prompt": "v",
        },
        headers=auth(raw),
    )
    assert r.status_code == 403
    assert "scope=full" in r.json()["detail"]
