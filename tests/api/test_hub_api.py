"""/api/hub/v1 — the hub's server-to-server API (foundation).

Pins:
  * only `Authorization: Bearer <hub launch token>` authenticates a hub path:
    no cookie, no api_token, no forged/expired/unsigned token;
  * tenant resolution matches /auth/sso (auto-provision + default persona,
    tier sync, external_user_id reconciliation, Zernio auto-link);
  * hub paths answer errors in the shared format, every other path keeps
    FastAPI's default bodies.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Iterator

import httpx
import pytest
from fastapi import HTTPException

from chalybclip.api.hub_api import from_admission, from_http_exception
from chalybclip.db import Database, PersonasRepo, TenantsRepo
from chalybclip.integrations.chalyb.admission import AdmissionRefused
from chalybclip.integrations.chalyb.sso import sign_sso_token
from chalybclip.settings import get_settings
from chalybclip.tenancy import bound_tenant

from .conftest import auth

_SECRET = "hub_shared_secret_value"


@pytest.fixture
def hub_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setenv("CHALYB_SSO_SECRET", _SECRET)
    get_settings.cache_clear()
    try:
        yield _SECRET
    finally:
        get_settings.cache_clear()


def _token(tenant_id: str, *, secret: str = _SECRET, **kw: object) -> str:
    return sign_sso_token(
        user_id=str(kw.pop("user_id", "chalyb_user_1")),
        email=str(kw.pop("email", "alice@example.com")),
        tenant_id=tenant_id,
        secret=secret,
        **kw,  # type: ignore[arg-type]
    )


def _hub(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _assert_hub_error(r: httpx.Response, status: int, code: str) -> dict:
    assert r.status_code == status, r.text
    body = r.json()
    assert body["ok"] is False
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]
    return body["error"]


# ── auth ────────────────────────────────────────────────────────────────


async def test_valid_token_resolves_tenant(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]], hub_env: str
) -> None:
    alice = tenants["alice"]["id"]
    r = await client.get("/api/hub/v1/me", headers=_hub(_token(alice)))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tenant_id"] == alice
    assert body["user_id"] == "chalyb_user_1"
    assert body["api_version"] == 1


async def test_missing_token_is_401(
    client: httpx.AsyncClient, hub_env: str
) -> None:
    r = await client.get("/api/hub/v1/me")
    err = _assert_hub_error(r, 401, "unauthorized")
    assert err["reason"] == "missing_token"
    assert r.headers.get("www-authenticate") == "Bearer"


async def test_session_cookie_is_not_accepted(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]], hub_env: str
) -> None:
    """A browser session (cookie) must never reach a hub path."""
    client.cookies.set("chalybclip_token", tenants["alice"]["token"])
    try:
        r = await client.get("/api/hub/v1/me")
    finally:
        client.cookies.clear()
    _assert_hub_error(r, 401, "unauthorized")


async def test_api_token_bearer_is_not_accepted(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]], hub_env: str
) -> None:
    r = await client.get("/api/hub/v1/me", headers=auth(tenants["alice"]["token"]))
    err = _assert_hub_error(r, 401, "unauthorized")
    assert err["reason"] == "invalid_token"


async def test_wrong_secret_is_401(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]], hub_env: str
) -> None:
    tok = _token(tenants["alice"]["id"], secret="not-the-secret")
    _assert_hub_error(await client.get("/api/hub/v1/me", headers=_hub(tok)), 401, "unauthorized")


async def test_tampered_payload_is_401(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]], hub_env: str
) -> None:
    """Swap the tenant inside a validly-signed token → signature breaks."""
    tok = _token(tenants["alice"]["id"])
    payload_b64, sig = tok.split(".", 1)
    raw = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4)))
    raw["tenant_id"] = tenants["bob"]["id"]
    forged = base64.urlsafe_b64encode(json.dumps(raw).encode()).decode().rstrip("=")
    r = await client.get("/api/hub/v1/me", headers=_hub(f"{forged}.{sig}"))
    _assert_hub_error(r, 401, "unauthorized")


async def test_expired_token_is_401(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]], hub_env: str
) -> None:
    tok = _token(tenants["alice"]["id"], ttl_seconds=-5)
    _assert_hub_error(await client.get("/api/hub/v1/me", headers=_hub(tok)), 401, "unauthorized")


async def test_garbage_token_is_401(client: httpx.AsyncClient, hub_env: str) -> None:
    for junk in ("x", "a.b", "....", "Bearer"):
        r = await client.get("/api/hub/v1/me", headers=_hub(junk))
        _assert_hub_error(r, 401, "unauthorized")
    # Non-ASCII bytes in the header (decoded as latin-1 by the server).
    r = await client.get(
        "/api/hub/v1/me", headers={"Authorization": "Bearer é.é".encode("latin-1")}
    )
    _assert_hub_error(r, 401, "unauthorized")


async def test_secret_not_configured_is_503(
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHALYB_SSO_SECRET", "")
    get_settings.cache_clear()
    try:
        tok = _token(tenants["alice"]["id"])
        r = await client.get("/api/hub/v1/me", headers=_hub(tok))
    finally:
        get_settings.cache_clear()
    err = _assert_hub_error(r, 503, "unavailable")
    assert err["retryable"] is True


async def test_each_token_only_sees_its_tenant(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]], hub_env: str
) -> None:
    for handle in ("alice", "bob"):
        tid = tenants[handle]["id"]
        r = await client.get("/api/hub/v1/me", headers=_hub(_token(tid)))
        assert r.json()["tenant_id"] == tid


# ── tenant resolution (same as /auth/sso) ───────────────────────────────


async def test_unknown_tenant_is_auto_provisioned(
    client: httpx.AsyncClient, db: Database, hub_env: str
) -> None:
    tid = "ten_hub_new_0001"
    assert await TenantsRepo(db).get(tid) is None
    r = await client.get(
        "/api/hub/v1/me",
        headers=_hub(_token(tid, user_id="u_new", email="new@example.com", tier="pro")),
    )
    assert r.status_code == 200, r.text
    tenant = await TenantsRepo(db).get(tid)
    assert tenant is not None
    assert tenant.name == "new@example.com"
    assert tenant.external_user_id == "u_new"
    assert tenant.tier == "pro"
    assert r.json()["tier"] == "pro"
    with bound_tenant(tid):
        assert await PersonasRepo(db).get("default") is not None
    # Second call is a no-op on the same tenant.
    r2 = await client.get("/api/hub/v1/me", headers=_hub(_token(tid, user_id="u_new")))
    assert r2.status_code == 200
    assert r2.json()["tenant_id"] == tid


async def test_claims_are_synced_like_sso(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    hub_env: str,
) -> None:
    alice = tenants["alice"]["id"]
    await TenantsRepo(db).set_external_user_id(alice, "stale_user")
    tok = _token(alice, user_id="real_user", tier="all_access", zernio_profile_id="zp_123")
    r = await client.get("/api/hub/v1/me", headers=_hub(tok))
    assert r.status_code == 200
    t = await TenantsRepo(db).get(alice)
    assert t is not None
    assert t.external_user_id == "real_user"
    assert t.tier == "all_access"
    assert t.zernio_profile_id == "zp_123"
    # An existing Zernio link is never clobbered.
    tok2 = _token(alice, user_id="real_user", zernio_profile_id="zp_other")
    await client.get("/api/hub/v1/me", headers=_hub(tok2))
    t2 = await TenantsRepo(db).get(alice)
    assert t2 is not None and t2.zernio_profile_id == "zp_123"


async def test_sso_login_still_provisions(
    client: httpx.AsyncClient, db: Database, hub_env: str
) -> None:
    """The /auth/sso refactor onto the shared helpers keeps its behaviour."""
    tid = "ten_sso_new_0001"
    r = await client.get(f"/auth/sso?token={_token(tid)}", follow_redirects=False)
    assert r.status_code == 303
    assert await TenantsRepo(db).get(tid) is not None
    with bound_tenant(tid):
        assert await PersonasRepo(db).get("default") is not None


# ── error format ────────────────────────────────────────────────────────


async def test_unknown_hub_route_uses_hub_format(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]], hub_env: str
) -> None:
    r = await client.get(
        "/api/hub/v1/does-not-exist", headers=_hub(_token(tenants["alice"]["id"]))
    )
    _assert_hub_error(r, 404, "not_found")


async def test_non_hub_routes_keep_default_errors(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    r = await client.get("/streams/nope", headers=auth(tenants["alice"]["token"]))
    assert r.status_code == 404
    assert "detail" in r.json()
    assert "error" not in r.json()


def test_http_exception_mapping() -> None:
    paywall = HTTPException(402, detail={"reason": "paywall_paid", "message": "Plan Pro"})
    e = from_http_exception(paywall)
    assert (e.code, e.status_code, e.reason, e.message) == (
        "needs_plan", 402, "paywall_paid", "Plan Pro",
    )
    assert from_http_exception(HTTPException(402, detail="sin tokens")).code == "no_credits"
    assert from_http_exception(HTTPException(413, detail="grande")).code == "too_large"
    assert from_http_exception(HTTPException(429, detail="lim")).code == "rate_limited"
    assert from_http_exception(HTTPException(404, detail="x")).code == "not_found"
    assert from_http_exception(HTTPException(409, detail="x")).code == "conflict"
    assert from_http_exception(HTTPException(422, detail="x")).code == "bad_request"
    inactive = HTTPException(403, detail={"reason": "tenant_inactive", "message": "pausada"})
    e = from_http_exception(inactive)
    assert (e.code, e.reason) == ("forbidden", "tenant_inactive")
    # Internal details in a 404/500 string detail are never echoed.
    assert from_http_exception(HTTPException(404, detail="row 42 missing")).message != (
        "row 42 missing"
    )


def test_admission_mapping() -> None:
    for reason, code, status in (
        ("no_tokens", "no_credits", 402),
        ("upload_too_large", "too_large", 413),
        ("minutes_cap", "rate_limited", 429),
        ("concurrency", "rate_limited", 429),
    ):
        e = from_admission(AdmissionRefused(reason))
        assert (e.code, e.status_code, e.reason) == (code, status, reason)
        assert e.message


async def test_token_ttl_is_enforced_server_side(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]], hub_env: str
) -> None:
    """A token that expired one second ago is refused (no leeway)."""
    tok = sign_sso_token(
        user_id="u", email="a@b.co", tenant_id=tenants["alice"]["id"],
        secret=_SECRET, ttl_seconds=-1,
    )
    assert int(time.time()) > 0
    _assert_hub_error(await client.get("/api/hub/v1/me", headers=_hub(tok)), 401, "unauthorized")
