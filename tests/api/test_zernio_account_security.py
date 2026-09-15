"""Tenant-ownership gates on the account surfaces.

The Zernio API key is company-wide, so DELETE /accounts/{id} and the
profile-claim flow would happily act on ANOTHER tenant's resources —
these tests pin that both routes verify ownership BEFORE any vendor
mutation, 404ing (not 403) so route/resource existence isn't advertised.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
import pytest_asyncio
import respx

from chalybclip.db import Database, TenantsRepo
from chalybclip.integrations.chalyb.service import sync_tenant_tier
from chalybclip.settings import get_settings

from .conftest import auth

_ZBASE = "https://zernio.com/api/v1"


@pytest.fixture
def zernio_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("CHALYBCLIP_ZERNIO_API_KEY", "sk_test_acctsec")
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


@pytest_asyncio.fixture
async def alice(
    db: Database, tenants: dict[str, dict[str, str]]
) -> dict[str, str]:
    tid = tenants["alice"]["id"]
    await sync_tenant_tier(db, tenant_id=tid, tier="all_access")
    await TenantsRepo(db).set_zernio_profile(
        tid, profile_id="prof_alice", profile_name="Alice",
    )
    return tenants["alice"]


# ---- disconnect ----


@pytest.mark.asyncio
async def test_disconnect_foreign_account_is_404_and_never_reaches_zernio(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    """An account id NOT on this tenant's profile → 404 before the vendor
    DELETE fires (only the ownership-resolving GET /accounts is mocked; a
    DELETE would hit respx's unmocked-route error)."""
    with respx.mock() as mock:
        mock.get(f"{_ZBASE}/accounts").mock(
            return_value=httpx.Response(
                200, json={"accounts": [{"platform": "tiktok", "_id": "acct_mine"}]}
            )
        )
        resp = await client.post(
            "/dashboard/publish/zernio/accounts/acct_of_bob/disconnect",
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_disconnect_own_account_still_works(
    zernio_env: None, client: httpx.AsyncClient, alice: dict[str, str]
) -> None:
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{_ZBASE}/accounts").mock(
            return_value=httpx.Response(
                200, json={"accounts": [{"platform": "tiktok", "_id": "acct_mine"}]}
            )
        )
        mock.delete(f"{_ZBASE}/accounts/acct_mine").mock(
            return_value=httpx.Response(200, json={"success": True})
        )
        resp = await client.post(
            "/dashboard/publish/zernio/accounts/acct_mine/disconnect",
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


# ---- claim ----


@pytest.mark.asyncio
async def test_claim_profile_bound_to_other_tenant_is_404(
    zernio_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
    tenants: dict[str, dict[str, str]],
) -> None:
    """A profile another tenant row already owns can't be claimed — 404
    with the SAME message as an unknown profileId (probing can't tell
    "someone else's" from "doesn't exist"), and no Zernio lookup fires
    (nothing is mocked — a call would error)."""
    await TenantsRepo(db).set_zernio_profile(
        tenants["bob"]["id"], profile_id="prof_bob", profile_name="Bob",
    )
    with respx.mock():
        resp = await client.post(
            "/dashboard/publish/zernio/accounts/claim",
            data={"profile_id": "prof_bob"},
            headers=auth(alice["token"]),
        )
    assert resp.status_code == 404
    assert "No connected accounts found" in resp.json()["detail"]
    # Bob's binding is untouched.
    bob_row = await TenantsRepo(db).get(tenants["bob"]["id"])
    assert bob_row is not None and bob_row.zernio_profile_id == "prof_bob"


@pytest.mark.asyncio
async def test_claim_own_or_unbound_profile_still_works(
    zernio_env: None,
    client: httpx.AsyncClient,
    db: Database,
    alice: dict[str, str],
) -> None:
    """The legitimate flows survive the gate: re-claiming your OWN bound
    profile, and claiming an unbound one (post-DB-wipe recovery)."""
    for profile in ("prof_alice", "prof_unbound"):
        with respx.mock(assert_all_called=True) as mock:
            mock.get(f"{_ZBASE}/accounts").mock(
                return_value=httpx.Response(
                    200,
                    json={"accounts": [{"platform": "tiktok", "_id": "acct_1"}]},
                )
            )
            resp = await client.post(
                "/dashboard/publish/zernio/accounts/claim",
                data={"profile_id": profile},
                headers=auth(alice["token"]),
            )
        assert resp.status_code == 303, profile
        assert f"claimed={profile}" in resp.headers["location"]
