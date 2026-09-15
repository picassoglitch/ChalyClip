"""GET /api/admin/sso-diag — operator diagnostic for the
'Strict SSO verify failed: bad signature' + provisioning-mismatch
symptoms.

The endpoint returns non-reversible fingerprints of the loaded SSO
secret + admin token so the operator can compare both sides without
revealing the raw value, and (given ?token=) runs the real strict
verify against the loaded secret to report the EXACT failure reason.

These tests pin:
  - auth (admin token via header OR ?key=, rejection on wrong/no key)
  - fingerprint equality semantics (same secret → same fingerprint)
  - trailing-whitespace detection (the silent 'bad signature' cause)
  - token decode + strict_verify verdicts: PASS / bad signature /
    token expired
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import httpx
import pytest

from chalybclip.integrations.chalyb.sso import sign_sso_token
from chalybclip.settings import get_settings

_ADMIN = "admintok_correct_value"
_SECRET = "sso_shared_secret_value"


def _fp(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


@pytest.fixture
def sso_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, str]]:
    """Set the admin token + SSO secret in the environment and bust the
    lru_cache so the handler's get_settings() sees them. Clears the
    cache again on teardown so the test values don't leak into the next
    test."""
    monkeypatch.setenv("CHALYB_ADMIN_TOKEN", _ADMIN)
    monkeypatch.setenv("CHALYB_SSO_SECRET", _SECRET)
    get_settings.cache_clear()
    try:
        yield {"admin": _ADMIN, "secret": _SECRET}
    finally:
        get_settings.cache_clear()


# ---- auth ----


async def test_sso_diag_rejects_missing_key(
    client: httpx.AsyncClient,
    sso_env: dict[str, str],
) -> None:
    r = await client.get("/api/admin/sso-diag")
    assert r.status_code == 401


async def test_sso_diag_rejects_wrong_key(
    client: httpx.AsyncClient,
    sso_env: dict[str, str],
) -> None:
    r = await client.get("/api/admin/sso-diag?key=not_the_admin_token")
    assert r.status_code == 403


async def test_sso_diag_accepts_key_query_param(
    client: httpx.AsyncClient,
    sso_env: dict[str, str],
) -> None:
    r = await client.get(f"/api/admin/sso-diag?key={_ADMIN}")
    assert r.status_code == 200


async def test_sso_diag_accepts_bearer_header(
    client: httpx.AsyncClient,
    sso_env: dict[str, str],
) -> None:
    r = await client.get(
        "/api/admin/sso-diag",
        headers={"Authorization": f"Bearer {_ADMIN}"},
    )
    assert r.status_code == 200


# ---- fingerprints ----


async def test_sso_diag_fingerprint_matches_known_hash(
    client: httpx.AsyncClient,
    sso_env: dict[str, str],
) -> None:
    """The reported fingerprint is exactly SHA256(secret)[:12], so the
    operator can compute the same on the Chalyb side."""
    r = await client.get(f"/api/admin/sso-diag?key={_ADMIN}")
    body = r.json()
    assert body["sso_secret"]["configured"] is True
    assert body["sso_secret"]["fingerprint"] == _fp(_SECRET)
    assert body["sso_secret"]["length"] == len(_SECRET)
    assert body["sso_secret"]["has_surrounding_whitespace"] is False
    # admin token is also fingerprinted (provisioning uses it).
    assert body["admin_token"]["fingerprint"] == _fp(_ADMIN)


async def test_sso_diag_detects_trailing_newline(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trailing newline on the secret (the classic copy-paste cause of
    'bad signature') must be flagged: has_surrounding_whitespace True and
    fingerprint != stripped_fingerprint."""
    secret_with_nl = _SECRET + "\n"
    monkeypatch.setenv("CHALYB_ADMIN_TOKEN", _ADMIN)
    monkeypatch.setenv("CHALYB_SSO_SECRET", secret_with_nl)
    get_settings.cache_clear()
    try:
        r = await client.get(f"/api/admin/sso-diag?key={_ADMIN}")
        body = r.json()
        assert body["sso_secret"]["has_surrounding_whitespace"] is True
        assert (
            body["sso_secret"]["fingerprint"]
            != body["sso_secret"]["stripped_fingerprint"]
        )
        # The stripped fingerprint equals the clean secret's fingerprint —
        # so the operator can see "if I strip the newline, THIS is what
        # matches the other side".
        assert body["sso_secret"]["stripped_fingerprint"] == _fp(_SECRET)
    finally:
        get_settings.cache_clear()


# ---- Zernio key (same wrong-variable trap) ----


async def test_sso_diag_zernio_flags_wrong_variable(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The classic Zernio 401: the real key is in the bare
    ZERNIO_API_KEY (Zernio's own SDK env name) while the app reads
    only the CHALYBCLIP_-prefixed var. The diag must surface both env
    fingerprints + what the app actually loaded so the operator sees
    the mismatch at a glance."""
    real_key = "real_zernio_key_value"
    stale_key = "stale_or_placeholder_value"
    monkeypatch.setenv("CHALYB_ADMIN_TOKEN", _ADMIN)
    # App reads the CHALYBCLIP_-prefixed var — set it to the STALE value.
    monkeypatch.setenv("CHALYBCLIP_ZERNIO_API_KEY", stale_key)
    # Operator put the REAL key in the unprefixed var (ignored by app).
    monkeypatch.setenv("ZERNIO_API_KEY", real_key)
    get_settings.cache_clear()
    try:
        r = await client.get(f"/api/admin/sso-diag?key={_ADMIN}")
        zn = r.json()["zernio"]
        # The app loaded the CHALYBCLIP_-prefixed (stale) value.
        assert zn["loaded_by_app"]["fingerprint"] == _fp(stale_key)
        assert zn["env_CHALYBCLIP_ZERNIO_API_KEY"]["fingerprint"] == _fp(
            stale_key
        )
        assert zn["env_ZERNIO_API_KEY"]["fingerprint"] == _fp(real_key)
        # The whole point: the two env vars DISAGREE → operator put the
        # key in the wrong name.
        assert (
            zn["env_CHALYBCLIP_ZERNIO_API_KEY"]["fingerprint"]
            != zn["env_ZERNIO_API_KEY"]["fingerprint"]
        )
    finally:
        get_settings.cache_clear()


# ---- token decode + strict verify ----


async def test_sso_diag_token_signed_with_same_secret_passes(
    client: httpx.AsyncClient,
    sso_env: dict[str, str],
) -> None:
    """A token signed with the loaded secret → strict_verify PASS and the
    decoded payload surfaces tenant_id + non-expired status."""
    token = sign_sso_token(
        user_id="auth0|abc",
        email="aldo@example.com",
        tenant_id="ten_real",
        secret=_SECRET,
        ttl_seconds=300,
    )
    r = await client.get(f"/api/admin/sso-diag?key={_ADMIN}&token={token}")
    body = r.json()
    assert body["token"]["strict_verify"] == "PASS"
    assert body["token"]["payload"]["tenant_id"] == "ten_real"
    assert body["token"]["payload"]["is_expired"] is False


async def test_sso_diag_token_signed_with_different_secret_fails_bad_sig(
    client: httpx.AsyncClient,
    sso_env: dict[str, str],
) -> None:
    """The actual operator scenario: the SSO link was signed by Chalyb
    with a DIFFERENT secret than ChalybClip loaded. strict_verify must
    report 'bad signature' — but the payload still decodes (unsigned) so
    the operator sees the tenant_id + that the link itself is well-formed
    and not expired. That distinguishes 'wrong key' from 'stale link'."""
    token = sign_sso_token(
        user_id="auth0|abc",
        email="aldo@example.com",
        tenant_id="ten_real",
        secret="a_completely_different_secret",
        ttl_seconds=300,
    )
    r = await client.get(f"/api/admin/sso-diag?key={_ADMIN}&token={token}")
    body = r.json()
    assert body["token"]["strict_verify"] == "FAIL: bad signature"
    # Payload still decodes — proves the link is well-formed, not expired,
    # so the ONLY problem is the signing key.
    assert body["token"]["payload"]["tenant_id"] == "ten_real"
    assert body["token"]["payload"]["is_expired"] is False


async def test_sso_diag_expired_token_with_correct_secret_reports_expired(
    client: httpx.AsyncClient,
    sso_env: dict[str, str],
) -> None:
    """A token signed with the CORRECT secret but expired → strict_verify
    'token expired', NOT 'bad signature'. This tells the operator the
    secrets are aligned and the real problem is a stale link / clock —
    a completely different fix path."""
    token = sign_sso_token(
        user_id="auth0|abc",
        email="aldo@example.com",
        tenant_id="ten_real",
        secret=_SECRET,
        ttl_seconds=-10,  # already expired
    )
    r = await client.get(f"/api/admin/sso-diag?key={_ADMIN}&token={token}")
    body = r.json()
    assert body["token"]["strict_verify"] == "FAIL: token expired"
