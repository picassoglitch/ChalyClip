"""create_profile_for_tenant tests (Zernio).

Pins the create-and-persist behavior: calls Zernio's POST /profiles,
stores the returned server `_id` + name on the tenant row.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from chalybclip.db import Database, TenantsRepo
from chalybclip.db.migrations import apply_migrations
from chalybclip.integrations.zernio import ZernioClient, ZernioError, create_profile_for_tenant
from chalybclip.tenancy import bound_tenant

_BASE = "https://zernio.com/api/v1"


@pytest.fixture
async def seeded_db(tmp_path: Path):
    db = Database(tmp_path / "zernio_profiles.db")
    await db.connect()
    await apply_migrations(db)
    tenant_id = "ten_01KS0TEST123"
    with bound_tenant(tenant_id):
        await TenantsRepo(db).create(tenant_id=tenant_id, name="Test")
    yield db, tenant_id
    await db.close()


@pytest.mark.asyncio
async def test_create_profile_persists_id_and_name(seeded_db) -> None:
    db, tenant_id = seeded_db
    body = {"profile": {"_id": "prof_abc123", "name": "My Brand"}}
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post(f"{_BASE}/profiles").mock(
                return_value=httpx.Response(201, json=body)
            )
            client = ZernioClient(api_key="sk_x", http=http)
            with bound_tenant(tenant_id):
                profile = await create_profile_for_tenant(
                    db=db,
                    tenant_id=tenant_id,
                    client=client,
                    name="My Brand",
                    description="Testing",
                )
    assert profile.profile_id == "prof_abc123"
    assert profile.name == "My Brand"
    # Request carried name + description.
    import json as _json
    sent = _json.loads(route.calls.last.request.content.decode())
    assert sent["name"] == "My Brand"
    assert sent["description"] == "Testing"
    # Persisted on the tenant row (id + name).
    with bound_tenant(tenant_id):
        tenant = await TenantsRepo(db).get(tenant_id)
    assert tenant is not None
    assert tenant.zernio_profile_id == "prof_abc123"
    assert tenant.zernio_profile_name == "My Brand"


@pytest.mark.asyncio
async def test_create_profile_propagates_zernio_error(seeded_db) -> None:
    db, tenant_id = seeded_db
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.post(f"{_BASE}/profiles").mock(
                return_value=httpx.Response(402, json={"error": "plan limit"})
            )
            client = ZernioClient(api_key="sk_x", http=http)
            with bound_tenant(tenant_id), pytest.raises(ZernioError) as ei:
                await create_profile_for_tenant(
                    db=db, tenant_id=tenant_id, client=client, name="X",
                )
    assert ei.value.status_code == 402
    # Tenant row stays unset.
    with bound_tenant(tenant_id):
        tenant = await TenantsRepo(db).get(tenant_id)
    assert tenant.zernio_profile_id is None
    assert tenant.zernio_profile_name is None


@pytest.mark.asyncio
async def test_missing_tenant_raises(seeded_db) -> None:
    db, _ = seeded_db
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=False):
            client = ZernioClient(api_key="sk_x", http=http)
            with (
                pytest.raises(ZernioError, match="tenant not found"),
                bound_tenant("ten_nope"),
            ):
                await create_profile_for_tenant(
                    db=db, tenant_id="ten_nope", client=client, name="X",
                )


@pytest.mark.asyncio
async def test_unlink_clears_id_and_name(seeded_db) -> None:
    """set_zernio_profile(profile_id=None) clears BOTH columns."""
    db, tenant_id = seeded_db
    repo = TenantsRepo(db)
    with bound_tenant(tenant_id):
        await repo.set_zernio_profile(
            tenant_id, profile_id="prof_x", profile_name="Name",
        )
        await repo.set_zernio_profile(tenant_id, profile_id=None)
        tenant = await repo.get(tenant_id)
    assert tenant.zernio_profile_id is None
    assert tenant.zernio_profile_name is None
