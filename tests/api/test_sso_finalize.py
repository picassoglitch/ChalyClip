"""GET /auth/sso — post-login redirect + external_user_id reconciliation.

Pins the funnel contract with chalyb:
  - a fresh SSO login lands on /dashboard/start (the "New clip" hero),
    matching the `next=/dashboard/start` chalyb sends on every launch URL
  - `next` is honored only as a same-origin relative path — /auth/sso must
    not be usable as an open redirect
  - the tenant's external_user_id is synced to the signed payload's user_id
    on EVERY login (backfill on NULL *and* overwrite on mismatch), because a
    stale link makes balance fetches read the wrong Chalyb ledger.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

from chalybclip.db import Database, TenantsRepo
from chalybclip.integrations.chalyb.sso import sign_sso_token
from chalybclip.settings import get_settings

_SECRET = "sso_shared_secret_value"


@pytest.fixture
def sso_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Strict-mode SSO: set the shared secret + bust the settings cache."""
    monkeypatch.setenv("CHALYB_SSO_SECRET", _SECRET)
    get_settings.cache_clear()
    try:
        yield _SECRET
    finally:
        get_settings.cache_clear()


def _token_for(tenant_id: str, *, user_id: str = "chalyb_user_1") -> str:
    return sign_sso_token(
        user_id=user_id,
        email="alice@example.com",
        tenant_id=tenant_id,
        secret=_SECRET,
    )


async def test_sso_lands_on_dashboard_start_by_default(
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
    sso_env: str,
) -> None:
    token = _token_for(tenants["alice"]["id"])
    r = await client.get(f"/auth/sso?token={token}", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/dashboard/start"
    assert "chalybclip_token" in r.headers.get("set-cookie", "")


async def test_sso_honors_relative_next(
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
    sso_env: str,
) -> None:
    token = _token_for(tenants["alice"]["id"])
    r = await client.get(
        f"/auth/sso?token={token}&next=/dashboard/streams",
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/dashboard/streams"


async def test_sso_auto_links_zernio_profile(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    sso_env: str,
) -> None:
    """When the token carries a zernio_profile_id and the tenant has
    none, SSO auto-links it — so the Publish Center tabs appear without
    pasting a profileId."""
    tid = tenants["alice"]["id"]
    token = sign_sso_token(
        user_id="chalyb_user_1", email="alice@example.com", tenant_id=tid,
        secret=_SECRET, zernio_profile_id="prof_from_chalyb",
    )
    r = await client.get(f"/auth/sso?token={token}", follow_redirects=False)
    assert r.status_code == 303
    tenant = await TenantsRepo(db).get(tid)
    assert tenant is not None
    assert tenant.zernio_profile_id == "prof_from_chalyb"


async def test_sso_does_not_clobber_existing_profile(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    sso_env: str,
) -> None:
    """A manual link/unlink the operator made in the dashboard wins —
    SSO only fills an EMPTY profile, never overwrites."""
    tid = tenants["alice"]["id"]
    await TenantsRepo(db).set_zernio_profile(tid, profile_id="prof_manual")
    token = sign_sso_token(
        user_id="chalyb_user_1", email="alice@example.com", tenant_id=tid,
        secret=_SECRET, zernio_profile_id="prof_from_chalyb",
    )
    await client.get(f"/auth/sso?token={token}", follow_redirects=False)
    tenant = await TenantsRepo(db).get(tid)
    assert tenant is not None
    assert tenant.zernio_profile_id == "prof_manual"  # not clobbered


async def test_sso_without_profile_id_is_backcompat(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    sso_env: str,
) -> None:
    """Older tokens without zernio_profile_id still validate + log in."""
    token = sign_sso_token(
        user_id="chalyb_user_1", email="alice@example.com",
        tenant_id=tenants["alice"]["id"], secret=_SECRET,
    )
    r = await client.get(f"/auth/sso?token={token}", follow_redirects=False)
    assert r.status_code == 303


@pytest.mark.parametrize(
    "evil_next",
    [
        "https://evil.example/phish",
        "//evil.example/phish",
        "/\\evil.example/phish",
        "",
    ],
)
async def test_sso_rejects_non_relative_next(
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
    sso_env: str,
    evil_next: str,
) -> None:
    token = _token_for(tenants["alice"]["id"])
    r = await client.get(
        f"/auth/sso?token={token}&next={evil_next}",
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/dashboard/start"


async def test_sso_backfills_missing_external_user_id(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    sso_env: str,
) -> None:
    tenant_id = tenants["alice"]["id"]
    before = await TenantsRepo(db).get(tenant_id)
    assert before is not None and not before.external_user_id

    token = _token_for(tenant_id, user_id="chalyb_user_fresh")
    r = await client.get(f"/auth/sso?token={token}", follow_redirects=False)
    assert r.status_code == 303

    after = await TenantsRepo(db).get(tenant_id)
    assert after is not None
    assert after.external_user_id == "chalyb_user_fresh"


async def test_sso_overwrites_stale_external_user_id(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    sso_env: str,
) -> None:
    """A tenant linked to the WRONG platform user reads someone else's
    token ledger. The signed payload is chalyb's word on ownership, so a
    mismatch is corrected on login — not preserved."""
    tenant_id = tenants["alice"]["id"]
    await TenantsRepo(db).set_external_user_id(tenant_id, "chalyb_user_stale")

    token = _token_for(tenant_id, user_id="chalyb_user_current")
    r = await client.get(f"/auth/sso?token={token}", follow_redirects=False)
    assert r.status_code == 303

    after = await TenantsRepo(db).get(tenant_id)
    assert after is not None
    assert after.external_user_id == "chalyb_user_current"
