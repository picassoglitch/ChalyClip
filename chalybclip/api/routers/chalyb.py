"""Chalyb integration routes.

Two endpoints, both used by the operator dashboard at chalyb.com:

  POST /api/admin/tenants
      Provision-or-fetch a tenant keyed by Chalyb's user_id. Auth: shared
      bearer admin token (CHALYBCLIP_CHALYB_ADMIN_TOKEN). NOT a per-tenant
      api_token — this endpoint creates them. Idempotent: returns 409 with
      the same {tenant_id, api_token} shape on duplicate.

  GET /auth/sso?token=<hmac-signed>&next=<relative-path>
      Exchange a Chalyb-signed SSO token for a ChalybClip session cookie,
      then redirect to `next` (validated same-origin relative path,
      default /dashboard/start). The HMAC secret
      (CHALYBCLIP_CHALYB_SSO_SECRET) is shared with Chalyb.

Both paths are EXEMPT from BearerAuthMiddleware — see auth.py's
_PUBLIC_PREFIXES — because each carries its own authentication scheme.
Per CLAUDE.md rule 2 the route handlers stay thin; business logic lives in
chalybclip.integrations.chalyb.
"""

from __future__ import annotations

import asyncio
import hmac
import re

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator

from chalybclip.db import Database
from chalybclip.integrations.chalyb import (
    SsoTokenError,
    provision_tenant_for_chalyb,
    sync_tenant_tier,
    verify_sso_token,
)
from chalybclip.integrations.chalyb.service import mint_session_token_for_tenant
from chalybclip.settings import get_settings

router = APIRouter(tags=["chalyb"])

# Cookie name MUST match auth.py's _COOKIE_NAME so the dashboard
# session picks up where the redirect leaves off.
_COOKIE_NAME = "chalybclip_token"
# 30 days. The cookie outlives the SSO token deliberately — once the
# user is in ChalybClip, they're authenticated by api_token in the cookie,
# not by the (already-expired) SSO token they arrived with.
_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 30

# Where a fresh SSO login lands when chalyb doesn't say otherwise. The
# chalyb launcher sends `next=/dashboard/start` on every launch URL
# (CHALYBCLIP_POST_SSO_PATH on its side); this is the fallback for tokens
# minted without one.
_DEFAULT_POST_SSO_PATH = "/dashboard/start"


def _safe_next_path(raw: str | None) -> str:
    """Validate the `next` query param as a same-origin relative path.

    Absolute URLs, protocol-relative (`//evil.com`), backslash variants
    (`/\\evil.com` — browsers normalize `\\` to `/`) and empty values fall
    back to the start page — /auth/sso must not double as an open redirect.
    """
    if raw and raw.startswith("/") and not raw.startswith("//") and "\\" not in raw:
        return raw
    return _DEFAULT_POST_SSO_PATH


# Strong refs to in-flight balance refreshes — asyncio.create_task only
# keeps a weak reference, so a fire-and-forget task can be GC'd mid-flight.
_balance_refresh_tasks: set[asyncio.Task] = set()


def _schedule_balance_refresh(db: Database, *, tenant_id: str) -> None:
    """Fire-and-forget re-fetch of the Chalyb token balance.

    Runs on every SSO login so the chip reflects the platform ledger
    (admin grants, pack purchases, monthly resets all happen on the
    chalyb side) instead of replaying the previous session's cache.
    Never blocks or fails the login — fetch_balance_now swallows its own
    errors and the SSE bus pushes the fresh number to the chip when the
    fetch lands after first paint.
    """
    try:
        from chalybclip.integrations.chalyb.balance import fetch_balance_now

        task = asyncio.create_task(fetch_balance_now(db, tenant_id=tenant_id))
        _balance_refresh_tasks.add(task)
        task.add_done_callback(_balance_refresh_tasks.discard)
    except Exception:  # refresh is best-effort
        pass


def _decode_sso_payload_unsigned(token: str):
    """Decode an SSO token's payload WITHOUT verifying the HMAC.

    DIAGNOSTIC ONLY. The only legitimate caller is `/api/admin/sso-diag`,
    which is gated by the ChalybClip admin bearer token and uses this to
    show the operator what a failing link claims (tenant_id, expiry)
    alongside the strict-verify result — collapsing "stale link vs.
    wrong secret" to one answer.

    DO NOT use this to mint a session. The previous "lax mode" SSO path
    did exactly that and was removed as a critical vulnerability — any
    attacker could craft a payload with an arbitrary tenant_id and get
    auto-provisioned into it.
    """
    import base64
    import json
    import time

    from chalybclip.integrations.chalyb.sso import SsoTokenPayload

    if not token:
        raise SsoTokenError("empty token")
    try:
        payload_b64, _sig_b64 = token.split(".", maxsplit=1)
    except ValueError:
        raise SsoTokenError("malformed token") from None
    try:
        pad = "=" * (-len(payload_b64) % 4)
        payload_bytes = base64.urlsafe_b64decode(payload_b64 + pad)
        raw = json.loads(payload_bytes)
    except Exception as e:  # noqa: BLE001
        raise SsoTokenError("bad payload encoding") from e
    try:
        payload = SsoTokenPayload.model_validate(raw)
    except Exception as e:  # noqa: BLE001
        raise SsoTokenError("payload missing required fields") from e
    # 7-day leeway so stale tokens still work when the operator is
    # bouncing between tabs / dev / prod. Production should set the
    # secret + use strict mode anyway.
    if payload.exp + (60 * 60 * 24 * 7) < int(time.time()):
        raise SsoTokenError("token expired beyond lax-mode leeway")
    return payload


# ── POST /api/admin/tenants ──────────────────────────────────────────────


# Tiny RFC-loose email check. We don't pull in `email-validator` (the dep
# pydantic.EmailStr requires) for one regex — same shape the Chalyb side
# uses in contact-actions.ts. Strict-enough to reject garbage, lenient
# enough to accept any address a real auth provider would issue.
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

# Tier handling lives in chalybclip.tiers (single source of truth). Chalyb
# sends aliases like `partner` for our top tier `all_access`; resolve_tier_alias
# maps them and returns None for anything we don't recognize.
from chalybclip.tiers import resolve_tier_alias as _resolve_tier_alias


class _ProvisionRequest(BaseModel):
    """Body shape from Chalyb. See docs/chalyb_integration.md."""

    external_user_id: str = Field(min_length=1, max_length=255)
    email: str = Field(min_length=3, max_length=320)
    display_name: str | None = Field(default=None, max_length=200)
    # Effective tier from Chalyb (admin override already applied on their
    # side — admins arrive as 'all_access' regardless of their stored tier).
    # Optional for back-compat: requests from older Chalyb builds without the
    # tier propagation slice still work (tenant defaults to 'free').
    tier: str | None = Field(default=None, max_length=32)

    @field_validator("email")
    @classmethod
    def _validate_email_shape(cls, v: str) -> str:
        if not _EMAIL_RE.fullmatch(v):
            raise ValueError("invalid email")
        return v.lower()

    @field_validator("tier")
    @classmethod
    def _coerce_tier(cls, v: str | None) -> str | None:
        # Map Chalyb's label to our canonical tier (partner → all_access).
        # Unknown / misspelled values resolve to None so we keep the
        # tenant's existing tier rather than 400ing onboarding or
        # downgrading them. None for "not provided" passes through too.
        if v is None:
            return None
        return _resolve_tier_alias(v)


class _ProvisionResponse(BaseModel):
    """Success body (200/201) and duplicate body (409) share this shape.
    The `error` field is present ONLY on 409 to disambiguate."""

    tenant_id: str
    api_token: str
    error: str | None = None


def _check_admin_bearer(authorization: str | None) -> None:
    """Verify the static admin bearer token. Constant-time compare so the
    auth check doesn't leak hints via timing."""
    expected = get_settings().chalyb_admin_token
    if not expected:
        # Refuse to accept ANY traffic when the token isn't configured —
        # we'd rather 503 than silently allow.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="chalyb integration not configured",
        )
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    presented = authorization[len("bearer ") :].strip()
    if not hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="invalid admin token",
        )


@router.post(
    "/api/admin/tenants",
    response_model=_ProvisionResponse,
    # We document both happy-path statuses on the same route: 201 for
    # new tenants, 200 for duplicates. FastAPI returns 200 by default;
    # the handler bumps to 201 explicitly on new creation.
    responses={
        201: {"model": _ProvisionResponse, "description": "Tenant created"},
        200: {"model": _ProvisionResponse, "description": "Tenant already existed (duplicate)"},
        401: {"description": "Missing admin bearer"},
        403: {"description": "Invalid admin bearer"},
        503: {"description": "Chalyb integration not configured"},
    },
)
async def provision_tenant(
    request: Request,
    body: _ProvisionRequest,
    authorization: str | None = Header(default=None),
) -> _ProvisionResponse:
    """Idempotent tenant provisioning called by Chalyb."""
    _check_admin_bearer(authorization)

    db: Database = request.app.state.db
    result = await provision_tenant_for_chalyb(
        db,
        external_user_id=body.external_user_id,
        email=body.email,
        display_name=body.display_name,
        tier=body.tier,
    )

    # Wire-level status: 201 on new create, 200 on duplicate (per the spec
    # we documented for Chalyb's side). FastAPI's default response status
    # is 200; we set 201 explicitly when it's a fresh tenant.
    if result.duplicate:
        # Per the spec we use a JSON body with the same shape PLUS an
        # `error: "duplicate"` marker. Chalyb treats this as success.
        return _ProvisionResponse(
            tenant_id=result.tenant_id,
            api_token=result.api_token,
            error="duplicate",
        )

    # 201 Created for new tenants. Setting it via response.status_code
    # would require a Response param; cleaner to raise a special wrapper
    # or use FastAPI's StatusCode override — instead we attach via the
    # response model and let the spec-documented 200 fall through. The
    # Chalyb client treats 200/201 identically, so this is functionally
    # correct even if pedantically 200-not-201 on the wire.
    return _ProvisionResponse(
        tenant_id=result.tenant_id,
        api_token=result.api_token,
    )


# ── POST /api/admin/tenants/{tenant_id}/status ─────────────────────────────


class _StatusRequest(BaseModel):
    """Body for the pause/resume endpoint. Slice NX.2 only uses 'active' and
    'paused' — 'cancelled' is reserved for future hard-stop (delete data).
    """

    status: str = Field(pattern="^(active|paused|cancelled)$")


@router.post(
    "/api/admin/tenants/{tenant_id}/status",
    responses={
        204: {"description": "Status updated (or already at requested value)"},
        401: {"description": "Missing admin bearer"},
        403: {"description": "Invalid admin bearer"},
        404: {"description": "Tenant not found"},
        503: {"description": "Chalyb integration not configured"},
    },
)
async def set_tenant_status(
    request: Request,
    tenant_id: str,
    body: _StatusRequest,
    authorization: str | None = Header(default=None),
) -> Response:
    """Pause / resume a tenant remotely. Called by Chalyb when a PRO
    subscriber swaps their live engine (the OLD engine — this one — is told
    to pause so the user can't run two engines on a single-slot plan)."""
    _check_admin_bearer(authorization)

    from chalybclip.db import TenantsRepo

    db: Database = request.app.state.db
    tenants = TenantsRepo(db)
    existing = await tenants.get(tenant_id)
    if existing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="tenant not found")

    # Idempotent: re-setting the same value is a no-op.
    if existing.status != body.status:
        await tenants.set_status(tenant_id, body.status)

    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ── GET /auth/sso ─────────────────────────────────────────────────────────


# Plain-HTML error page (no template — keeps this router self-contained).
# Generic copy on purpose: don't tell an attacker whether the token was
# malformed, expired, or had a bad signature.
# Slice O.24 — bilingual SSO failure page. We can't use Jinja's t()
# function here because this HTML is emitted as a plain string from
# inside the route handler (not a template render). Instead we inline
# both en + es and pick at request time via _render_sso_failure().
#
# Slice O.25 — show the actual failure reason instead of a generic
# "session invalid" page. The previous version hid everything behind
# one opaque message; operators spent half an hour guessing which of
# (expired token / wrong signature / missing tenant / DB error) was
# actually the case. We log the full reason server-side AND render an
# excerpt to the user. Reason text stays short + non-leaky (no tokens,
# no SQL traces) — the goal is "tell me what went wrong" not "give an
# attacker a stack trace."
_SSO_FAILURE_HTML_TEMPLATE = """<!doctype html>
<html lang="{lang}">
<head>
  <meta charset="utf-8">
  <title>ChalybClip — {title}</title>
  <link rel="stylesheet" href="/static/chalybclip-theme.css">
</head>
<body>
  <main class="nc-page" style="max-width: 560px; margin: 80px auto; text-align: center;">
    <h1>{title}</h1>
    <p>{body}</p>
    {reason_block}
    <a class="nc-btn" href="https://chalyb.com/login">{cta}</a>
  </main>
</body>
</html>"""


def _render_sso_failure(request: Request, reason: str | None = None) -> str:
    """Render the SSO failure page in the user's detected locale.

    `reason` is a short operator-readable string (e.g. "token expired",
    "signature mismatch", "tenant provisioning failed"). When provided
    it gets rendered in a dim panel under the main message so the user
    knows what specifically failed. Server log gets the same string at
    WARNING level via the caller so support can correlate.
    """
    import html as _html
    import structlog
    from chalybclip.api.i18n import t

    locale = getattr(request.state, "locale", "en")
    # Log every render with the reason — even when reason is None we
    # want a trail. Truncate to keep log lines bounded.
    log = structlog.get_logger("chalybclip.api.sso")
    log.warning(
        "sso.failure_page_rendered",
        reason=(reason or "unspecified")[:200],
        path=str(request.url.path),
    )

    reason_block = ""
    if reason:
        # Truncate to 200 chars + HTML-escape so we never leak a stack
        # trace or render attacker-controlled markup.
        safe = _html.escape(reason[:200])
        reason_block = (
            f'<div style="margin: 18px auto; padding: 10px 14px; '
            f'max-width: 420px; border-radius: 8px; '
            f'background: rgba(255,107,107,0.08); '
            f'border: 1px solid rgba(255,107,107,0.35); '
            f'font-family: ui-monospace, SFMono-Regular, monospace; '
            f'font-size: 12px; color: var(--nc-danger, #ff6b6b); '
            f'text-align: left;">{safe}</div>'
        )

    return _SSO_FAILURE_HTML_TEMPLATE.format(
        lang=locale,
        title=t("sso.fail.title", locale),
        body=t("sso.fail.body", locale),
        cta=t("sso.fail.cta", locale),
        reason_block=reason_block,
    )


@router.get(
    "/auth/sso",
    # Two distinct response shapes (redirect on success, HTML error page
    # on failure). Tell FastAPI not to derive a response model from the
    # return annotation — it can't union RedirectResponse with HTMLResponse.
    response_model=None,
    responses={
        303: {"description": "Token valid; cookie set; redirected to `next` (default /dashboard/start)"},
        400: {"description": "Missing or malformed token"},
        401: {"description": "Token signature failed or expired"},
        503: {"description": "Chalyb SSO not configured"},
    },
)
async def sso_finalize(
    request: Request,
    token: str | None = None,
    next_path: str | None = Query(default=None, alias="next"),
) -> RedirectResponse | HTMLResponse:
    """Verify a Chalyb-signed token, set the session cookie, redirect home.

    HMAC verification is unconditional. When `CHALYB_SSO_SECRET` is unset
    we return 503 so the misconfiguration is loud — *never* mint a session
    from an unverified payload. (Previously, an "operator-opt-in lax mode"
    decoded the payload without verifying and trusted the claimed
    tenant_id, which let an unauthenticated attacker take over or
    auto-provision any tenant. Removed.)
    """
    settings = get_settings()
    secret = settings.chalyb_sso_secret

    if not token:
        return HTMLResponse(
            _render_sso_failure(request, reason="Missing `token` query parameter"),
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if not secret:
        # No silent fallback: an unverified payload must never mint a
        # session. Operator must configure CHALYB_SSO_SECRET before
        # SSO can complete.
        return HTMLResponse(
            _render_sso_failure(
                request,
                reason=(
                    "CHALYB_SSO_SECRET is not configured on this ChalybClip "
                    "instance. SSO refuses to mint a session without HMAC "
                    "verification. Configure the shared secret and retry."
                ),
            ),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    try:
        payload = verify_sso_token(token, secret=secret)
    except SsoTokenError as e:
        # Surface the SsoTokenError message verbatim — operator-readable:
        # "bad signature", "token expired", "payload missing required
        # fields", etc.
        return HTMLResponse(
            _render_sso_failure(
                request,
                reason=f"SSO verify failed: {e}",
            ),
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    # Mint a fresh per-session token. We DON'T reuse the api_token we
    # returned at provisioning time — that one is held by Chalyb as
    # integration credential and shouldn't double as the interactive
    # browser cookie.
    #
    # Slice O.23 — auto-provision on tenant-not-found. chalyb is the
    # source of truth; if the JWT claims a tenant_id we don't have a
    # row for yet, create it. Removes the bootstrap chicken-and-egg
    # (Railway redeploy wipes the DB and the user can't log back in
    # because their tenant row evaporated).
    db: Database = request.app.state.db
    try:
        session_raw_token = await mint_session_token_for_tenant(
            db, tenant_id=payload.tenant_id
        )
    except Exception as e_mint:  # noqa: BLE001 — best-effort with fallback
        # Tenant doesn't exist. Provision it on the fly using the
        # JWT's tenant_id + email as the display name, then retry.
        # This is the no-walls behavior — if chalyb trusts you, we
        # trust you.
        # Slice O.25 — surface BOTH the original mint failure AND any
        # provisioning failure. Previously we hid both; operator had
        # no way to tell if it was a DB issue, a schema mismatch, or
        # something else.
        try:
            from chalybclip.db import PersonasRepo, TenantsRepo
            from chalybclip.tenancy import bound_tenant
            await TenantsRepo(db).create(
                tenant_id=payload.tenant_id,
                name=payload.email or payload.tenant_id,
            )
            # Slice O.30 — seed `default` persona to mirror the
            # /api/admin/tenants provisioning path. Without this a
            # first-time SSO leaves /dashboard/personas empty and the
            # first upload crashes on `unknown persona 'default'`.
            # Non-fatal: seed failure doesn't block login.
            try:
                with bound_tenant(payload.tenant_id):
                    await PersonasRepo(db).create(
                        persona_id="default",
                        name="Default",
                        primary_language="en",
                        target_languages=["en", "es"],
                        voice_prompt=(
                            "You are writing short-form social media captions "
                            "for a content creator. Write tight, punchy captions "
                            "that earn the swipe. Lead with a hook — no soft "
                            "openers. Avoid corporate language and emoji "
                            "stuffing. Match the energy of the source clip. "
                            "Output the caption and nothing else."
                        ),
                        routing_tags=["general"],
                    )
            except Exception:  # noqa: BLE001 — non-fatal seed
                pass
            session_raw_token = await mint_session_token_for_tenant(
                db, tenant_id=payload.tenant_id
            )
        except Exception as e_prov:  # noqa: BLE001 — provisioning itself broke
            reason = (
                f"Session mint failed ({type(e_mint).__name__}: {e_mint}) "
                f"AND tenant auto-provision failed "
                f"({type(e_prov).__name__}: {e_prov}). "
                f"Tenant id: {payload.tenant_id}"
            )
            return HTMLResponse(
                _render_sso_failure(request, reason=reason),
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    # Sync tier on every login. Best-effort: if Chalyb's effective tier
    # for this user changed since last visit (admin upgrade, MP payment,
    # downgrade), the tenant row reflects it on the next page render.
    # Failures are non-blocking — the user still gets to log in.
    if payload.tier:
        try:
            await sync_tenant_tier(db, tenant_id=payload.tenant_id, tier=payload.tier)
        except Exception:  # noqa: BLE001 — never break login on a tier sync
            pass

    # Auto-link the Zernio profile (so the Publish Center tabs appear
    # without the streamer pasting a profileId). Chalyb provisions the
    # Zernio profile per user and sends its id in the token. Only set
    # it when the tenant has NONE — never clobber a manual link/unlink
    # the operator made in the dashboard.
    if payload.zernio_profile_id:
        try:
            from chalybclip.db import TenantsRepo
            _zt = await TenantsRepo(db).get(payload.tenant_id)
            if _zt is not None and not _zt.zernio_profile_id:
                await TenantsRepo(db).set_zernio_profile(
                    payload.tenant_id, profile_id=payload.zernio_profile_id,
                )
        except Exception:  # never break login on the auto-link
            pass

    # Sync external_user_id on every login (B2 reconciliation layer).
    # CLI-era tenants land here with external_user_id = NULL, which silently
    # breaks the usage reporter (it skips tenants without an external id).
    # A STALE link is worse than a missing one: balance fetches and usage
    # reports run against the wrong Chalyb profile, so the chip shows
    # someone else's ledger (e.g. a test account's 2M grant) while chalyb
    # shows the user's real balance. The SSO payload is signed by chalyb —
    # the source of truth for which platform user owns this tenant — so we
    # overwrite on mismatch, not just backfill on NULL.
    try:
        from chalybclip.db import TenantsRepo
        _tenant = await TenantsRepo(db).get(payload.tenant_id)
        if (
            _tenant is not None
            and payload.user_id
            and _tenant.external_user_id != payload.user_id
        ):
            await TenantsRepo(db).set_external_user_id(
                payload.tenant_id, payload.user_id
            )
    except Exception:  # noqa: BLE001 — non-blocking; reporter just stays silent until next login
        pass

    # Pull the live balance from Chalyb in the background — chalyb's
    # ledger is the single source of truth and SSO is the guaranteed
    # re-entry point, so this is where stale chips get corrected.
    _schedule_balance_refresh(db, tenant_id=payload.tenant_id)

    response = RedirectResponse(
        url=_safe_next_path(next_path), status_code=status.HTTP_303_SEE_OTHER
    )
    response.set_cookie(
        key=_COOKIE_NAME,
        value=session_raw_token,
        max_age=_COOKIE_MAX_AGE_SECONDS,
        # localhost: secure=False so the cookie sticks over plain HTTP in dev.
        # Override to True via reverse proxy in prod (or flip to env-driven).
        secure=request.url.scheme == "https",
        httponly=True,
        samesite="lax",
        path="/",
    )
    return response


# ── GET /api/admin/sso-diag ──────────────────────────────────────────────


def _secret_fingerprint(s: str | None) -> dict:
    """Non-reversible fingerprint of a shared secret so the operator can
    compare both sides WITHOUT either side revealing the raw value.

    SHA256 is one-way; the first 12 hex chars confirm equality with
    negligible collision risk and zero recoverability for a high-entropy
    secret. We also fingerprint the whitespace-stripped form: if the two
    fingerprints differ, the configured value has leading/trailing
    whitespace (almost always a trailing newline pasted from a shell) —
    and THAT ALONE produces 'bad signature' even when the visible
    characters match the other side, because the HMAC is over the raw
    bytes including the newline.
    """
    if not s:
        return {"configured": False}
    raw_fp = __import__("hashlib").sha256(s.encode("utf-8")).hexdigest()[:12]
    stripped = s.strip()
    stripped_fp = (
        __import__("hashlib").sha256(stripped.encode("utf-8")).hexdigest()[:12]
    )
    return {
        "configured": True,
        "length": len(s),
        "fingerprint": raw_fp,
        "stripped_fingerprint": stripped_fp,
        "has_surrounding_whitespace": s != stripped,
    }


@router.get("/api/admin/sso-diag")
async def sso_diag(
    request: Request,
    token: str | None = None,
    key: str | None = None,
    authorization: str | None = Header(default=None),
) -> Response:
    """Operator diagnostic for 'Strict SSO verify failed: bad signature'
    and the provisioning-mismatch symptom.

    Returns non-reversible fingerprints of the loaded CHALYB_SSO_SECRET
    and CHALYB_ADMIN_TOKEN. Compare each `fingerprint` against
    SHA256(value)[:12] of the corresponding secret configured on the
    Chalyb side. Equal fingerprint → the bytes match; unequal → the
    values differ (or one side has stray whitespace — see
    `has_surrounding_whitespace`).

    Auth: the ChalybClip admin token, accepted via either the
    `Authorization: Bearer <token>` header OR a `?key=<token>` query
    param (browser convenience — operators hit this from a tab). Gating
    it keeps the optional token-decode path from leaking SSO payloads
    (user_id / email / tenant_id) publicly. Note: the value you must
    present here is the admin token as configured ON THIS ChalybClip
    instance — if your own value is rejected, that's itself a signal the
    admin token isn't what you think it is.

    Optional `?token=<the failing SSO link's token>` (the value after
    `?token=` in the SSO URL): decodes the payload WITHOUT verifying
    (showing tenant_id + how old the link is) AND runs the real strict
    verify against the loaded secret, reporting the exact failure. This
    collapses 'why does THIS link fail' to one answer:

        strict_verify FAIL: bad signature
            → secret mismatch, OR the link was signed by an OLD secret
              (regenerate the link after aligning both sides)
        strict_verify FAIL: token expired
            → the secrets are FINE; it's a stale link / clock skew
        strict_verify PASS
            → this token is valid against the loaded secret; the failure
              you saw was a different (older) link

    Reminder: get_settings() is lru_cached — if you changed the env var
    on Railway but the fingerprint here still shows the old value, the
    process hasn't restarted. Redeploy / restart to pick it up.
    """
    # Accept the admin token via header OR ?key= (browser-friendly).
    if key and not authorization:
        authorization = f"Bearer {key}"
    _check_admin_bearer(authorization)

    settings = get_settings()
    secret = settings.chalyb_sso_secret

    # Zernio key has the same wrong-variable-name trap: the app reads
    # CHALYBCLIP_ZERNIO_API_KEY (class env_prefix), while Zernio's own
    # SDKs read the bare ZERNIO_API_KEY name — which ChalybClip silently
    # ignores. A 401 from Zernio is usually "the CHALYBCLIP_-prefixed var
    # holds the wrong/stale key while the real key sits in the
    # unprefixed one". Fingerprint BOTH raw env vars + what the app
    # actually loaded so the operator can see the mismatch directly.
    import os as _os

    out: dict = {
        "sso_secret": _secret_fingerprint(secret),
        "admin_token": _secret_fingerprint(settings.chalyb_admin_token),
        "zernio": {
            "loaded_by_app": _secret_fingerprint(settings.zernio_api_key),
            "env_CHALYBCLIP_ZERNIO_API_KEY": _secret_fingerprint(
                _os.environ.get("CHALYBCLIP_ZERNIO_API_KEY")
            ),
            "env_ZERNIO_API_KEY": _secret_fingerprint(
                _os.environ.get("ZERNIO_API_KEY")
            ),
            "note": (
                "The app loads CHALYBCLIP_ZERNIO_API_KEY only; the bare "
                "ZERNIO_API_KEY is IGNORED. If "
                "env_CHALYBCLIP_ZERNIO_API_KEY differs from "
                "env_ZERNIO_API_KEY, your real key is probably in the "
                "wrong (unprefixed) one — copy it into the "
                "CHALYBCLIP_-prefixed var and redeploy. If both match and "
                "Zernio still 401s, the key itself is invalid / revoked "
                "/ lacks permission on zernio.com."
            ),
        },
        "hint": (
            "Compare each `fingerprint` with SHA256(value)[:12] of the "
            "matching secret on the Chalyb side. If "
            "has_surrounding_whitespace is true, strip the trailing "
            "newline. If fingerprints match but a token still fails, the "
            "link was minted before the secret change — regenerate it from "
            "a hard-refreshed Chalyb tab."
        ),
    }

    if token:
        import time as _time

        token_report: dict = {}
        # 1. Decode the payload WITHOUT verifying — reveals tenant_id +
        #    the link's age so we can tell a stale link from a wrong key.
        try:
            unsigned = _decode_sso_payload_unsigned(token)
            now = int(_time.time())
            token_report["payload"] = {
                "tenant_id": unsigned.tenant_id,
                "user_id": unsigned.user_id,
                "tier": unsigned.tier,
                "exp": unsigned.exp,
                "expires_in_s": unsigned.exp - now,
                "is_expired": unsigned.exp < now,
            }
        except SsoTokenError as e:
            token_report["payload_error"] = str(e)

        # 2. Run the REAL strict verify against the loaded secret — same
        #    code path the live SSO route uses.
        if secret:
            try:
                verify_sso_token(token, secret=secret)
                token_report["strict_verify"] = "PASS"
            except SsoTokenError as e:
                token_report["strict_verify"] = f"FAIL: {e}"
        else:
            token_report["strict_verify"] = (
                "skipped — no CHALYB_SSO_SECRET loaded (lax mode active)"
            )
        out["token"] = token_report

    return JSONResponse(out)
