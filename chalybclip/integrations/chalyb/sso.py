"""HMAC-SHA256 SSO token signing + verification.

Token shape (matches what Chalyb's `chalybclip.ts` produces):

    {base64url_payload}.{base64url_sig}

`payload` is a JSON object:

    {
      "user_id": "auth0|...",      # Chalyb's Supabase user id
      "email": "alexis@gmail.com",
      "tenant_id": "ten_...",      # our tenant id, returned at provisioning
      "exp": 1700000000            # unix seconds (5 min TTL by default)
    }

`sig` = HMAC-SHA256(payload_base64url, CHALYBCLIP_CHALYB_SSO_SECRET), then
base64url-encoded. Pure stdlib — no JWT dependency to drag in for a
two-party trust relationship.

Why not JWT: we control both ends, no key rotation drama, no header/algo
agility needed. One HMAC call each side, done.
"""

from __future__ import annotations

import base64
import hmac
import json
import time
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, Field


class SsoTokenError(Exception):
    """Raised when a token fails verification. Generic on purpose — the
    handler turns this into a single user-facing 'session inválida' page
    rather than telling an attacker *why* their forgery didn't work."""


class SsoTokenPayload(BaseModel):
    """The verified payload after a successful `verify_sso_token` call.

    `tier` is optional for back-compat — older tokens minted before the tier
    propagation slice still validate. When present, the SSO route syncs the
    tenant's tier column to this value so a Pro/All-Access user lands with
    the correct tier from login one. Values: 'free' | 'pro' | 'all_access'.
    """

    user_id: str = Field(min_length=1, max_length=255)
    email: str = Field(min_length=3, max_length=320)
    tenant_id: str = Field(min_length=1, max_length=64)
    tier: str | None = Field(default=None, max_length=32)
    # Optional. When Chalyb knows the user's Zernio profileId (it
    # provisions the Zernio profile per user), it includes it here so
    # ChalyClip auto-links it on login — the streamer never has to paste
    # a profileId into the Publish Center. Back-compat: older tokens
    # without it still validate.
    zernio_profile_id: str | None = Field(default=None, max_length=64)
    exp: int = Field(gt=0)


# Default token lifetime when we MINT one (Chalyb is the minter — this
# constant is kept here for symmetry / tests). 5 minutes is enough time
# for the user to land + the browser to redirect, short enough that an
# intercepted token is mostly useless.
DEFAULT_TTL_SECONDS = 300


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    # base64 expects padding to a multiple of 4; urlsafe_b64encode strips it.
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def sign_sso_token(
    *,
    user_id: str,
    email: str,
    tenant_id: str,
    secret: str,
    tier: str | None = None,
    zernio_profile_id: str | None = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> str:
    """Mint a token. Used by tests + as a reference for the Chalyb side."""
    if not secret:
        raise SsoTokenError("missing secret")
    payload: dict[str, str | int] = {
        "user_id": user_id,
        "email": email,
        "tenant_id": tenant_id,
        "exp": int(time.time()) + ttl_seconds,
    }
    if tier:
        payload["tier"] = tier
    if zernio_profile_id:
        payload["zernio_profile_id"] = zernio_profile_id
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(secret.encode("utf-8"), payload_b64.encode("ascii"), sha256).digest()
    return f"{payload_b64}.{_b64url_encode(sig)}"


def verify_sso_token(
    token: str,
    *,
    secret: str,
    now: int | None = None,
    leeway_seconds: int = 0,
) -> SsoTokenPayload:
    """Validate a signed token. Returns the payload or raises SsoTokenError.

    `leeway_seconds` accommodates clock skew between Chalyb and ChalyClip.
    Default 0 (strict) is fine when both are deployed; bump to 5-10 in
    distributed setups if needed.
    """
    if not secret:
        raise SsoTokenError("chalyb_sso_secret not configured on this ChalyClip instance")
    if not token:
        raise SsoTokenError("empty token")

    try:
        payload_b64, sig_b64 = token.split(".", maxsplit=1)
    except ValueError:
        raise SsoTokenError("malformed token") from None

    expected_sig = hmac.new(
        secret.encode("utf-8"), payload_b64.encode("ascii"), sha256
    ).digest()
    try:
        actual_sig = _b64url_decode(sig_b64)
    except Exception as e:  # noqa: BLE001 — base64 lib raises a few different kinds
        raise SsoTokenError("bad signature encoding") from e

    # Constant-time compare so timing leaks don't disclose how-close-we-got.
    if not hmac.compare_digest(expected_sig, actual_sig):
        raise SsoTokenError("bad signature")

    try:
        payload_bytes = _b64url_decode(payload_b64)
        raw = json.loads(payload_bytes)
    except Exception as e:  # noqa: BLE001 — JSON / b64 / unicode all collapsed
        raise SsoTokenError("bad payload encoding") from e

    try:
        payload = SsoTokenPayload.model_validate(raw)
    except Exception as e:  # noqa: BLE001 — pydantic raises ValidationError
        raise SsoTokenError("payload missing required fields") from e

    current = now if now is not None else int(time.time())
    if payload.exp + leeway_seconds < current:
        raise SsoTokenError("token expired")

    return payload


# Re-export so callers can build their own structured-equality checks
# without poking at the BaseModel directly.
Algorithm = Literal["HS256"]
