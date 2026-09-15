"""Auto-provision a Zernio profile so the Publish Center tabs appear
without a manual step (find-or-create, idempotent)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
import respx

from chalybclip.db import Database, TenantsRepo, apply_migrations
from chalybclip.integrations.zernio.client import ZernioClient
from chalybclip.integrations.zernio.profiles import ensure_zernio_profile_for_tenant

_ZBASE = "https://zernio.com/api/v1"


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "ensure.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


def _client(http: httpx.AsyncClient) -> ZernioClient:
    return ZernioClient(api_key="sk_test", http=http)


@pytest.mark.asyncio
async def test_returns_existing_link_without_zernio_call(db: Database) -> None:
    t = await TenantsRepo(db).create(name="Has Profile")
    await TenantsRepo(db).set_zernio_profile(t.id, profile_id="prof_already")
    async with httpx.AsyncClient() as http:
        # No respx mock → any Zernio call would error. None should happen.
        pid = await ensure_zernio_profile_for_tenant(
            db=db, tenant_id=t.id, client=_client(http),
        )
    assert pid == "prof_already"


@pytest.mark.asyncio
async def test_creates_when_none_exists(db: Database) -> None:
    t = await TenantsRepo(db).create(name="No Profile")
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            mock.get(f"{_ZBASE}/profiles").mock(
                return_value=httpx.Response(200, json={"profiles": []})
            )
            create = mock.post(f"{_ZBASE}/profiles").mock(
                return_value=httpx.Response(
                    201, json={"profile": {"_id": "prof_new", "name": f"ChalyClip {t.id}"}}
                )
            )
            pid = await ensure_zernio_profile_for_tenant(
                db=db, tenant_id=t.id, client=_client(http),
            )
    assert pid == "prof_new"
    assert create.called
    tenant = await TenantsRepo(db).get(t.id)
    assert tenant is not None and tenant.zernio_profile_id == "prof_new"


@pytest.mark.asyncio
async def test_relinks_existing_named_profile_without_creating(db: Database) -> None:
    """A profile named `ChalyClip <tenant_id>` already on Zernio (e.g. a
    prior auto-create whose local link was lost) is re-linked, not
    duplicated."""
    t = await TenantsRepo(db).create(name="Lost Link")
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=False) as mock:
            mock.get(f"{_ZBASE}/profiles").mock(
                return_value=httpx.Response(
                    200,
                    json={"profiles": [
                        {"_id": "prof_other", "name": "Some other brand"},
                        {"_id": "prof_mine", "name": f"ChalyClip {t.id}"},
                    ]},
                )
            )
            post = mock.post(f"{_ZBASE}/profiles").mock(
                return_value=httpx.Response(201, json={"profile": {"_id": "x"}})
            )
            pid = await ensure_zernio_profile_for_tenant(
                db=db, tenant_id=t.id, client=_client(http),
            )
    assert pid == "prof_mine"
    assert not post.called  # re-linked, not created


@pytest.mark.asyncio
async def test_zernio_error_returns_none_no_crash(db: Database) -> None:
    t = await TenantsRepo(db).create(name="Zernio Down")
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.get(f"{_ZBASE}/profiles").mock(
                return_value=httpx.Response(502, text="upstream down")
            )
            pid = await ensure_zernio_profile_for_tenant(
                db=db, tenant_id=t.id, client=_client(http),
            )
    assert pid is None  # best-effort; caller falls back to the manual form
    tenant = await TenantsRepo(db).get(t.id)
    assert tenant is not None and tenant.zernio_profile_id is None
