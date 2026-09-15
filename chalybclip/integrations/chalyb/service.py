"""Provisioning service — idempotent tenant creation keyed by Chalyb user id.

Called from POST /api/admin/tenants. The handler stays thin: parse + auth, then
call `provision_tenant_for_chalyb`, then shape the response.

Flow:
  1. Look up existing tenant by `external_user_id` (Chalyb's user_id).
     - Found: mint a fresh api_token bound to it. Return 409-equivalent
       (the route handler maps duplicate=True to HTTP 409 with the same
       shape so Chalyb can persist the values either way).
  2. Not found: create tenant + first user (the email), then mint api_token.

Tokens shown to the caller are RAW — we only store sha256(raw). The Chalyb
side persists the raw token in `engine_subscriptions.external_credentials`
so it can later forward it (when iframe/proxy modes ship). For the current
SSO redirect flow we look the token up again by `tenant_id` to set the
cookie — see sso_finalize() below.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from chalybclip.db import (
    ApiTokensRepo,
    Database,
    TenantsRepo,
    UsersRepo,
)
from chalybclip.errors import ChalybClipError
from chalybclip.tenancy import bound_tenant, mint_token

_log = logging.getLogger("chalybclip.chalyb.service")


class ChalybServiceError(ChalybClipError):
    """Service-layer failure during Chalyb integration calls."""


@dataclass(frozen=True)
class ChalybProvisionResult:
    """Return shape for `provision_tenant_for_chalyb`.

    `duplicate=True` means we found an existing tenant for this
    external_user_id; the route handler maps this to HTTP 409 per the
    contract in docs/chalyb_integration.md. The Chalyb side treats 409
    as success and persists the returned ids.
    """

    tenant_id: str
    api_token: str
    duplicate: bool


async def provision_tenant_for_chalyb(
    db: Database,
    *,
    external_user_id: str,
    email: str,
    display_name: str | None,
    tier: str | None = None,
) -> ChalybProvisionResult:
    """Create-or-fetch a tenant for a Chalyb user. Always returns a fresh
    api_token (we mint a new one on every duplicate hit so admin re-grants
    behave like 'rotate'). The previous token stays valid until manually
    revoked from the dashboard.

    `tier` (optional) overrides the default 'free' on creation AND syncs on
    duplicate. Lets Chalyb propagate Pro / All-Access from day one so the
    dashboard doesn't render the watermark warning chip for paying users."""

    if not external_user_id:
        raise ChalybServiceError("external_user_id required")
    if not email or "@" not in email:
        raise ChalybServiceError("invalid email")

    tenants = TenantsRepo(db)

    existing = await tenants.find_by_external_user_id(external_user_id)
    if existing is not None:
        # Mint a fresh token bound to the existing tenant.
        raw, hash_ = mint_token()
        with bound_tenant(existing.id):
            await ApiTokensRepo(db).create(hash_=hash_, scope="full")
        # Sync tier on duplicate — handles upgrade / downgrade since last call.
        if tier and tier != existing.tier:
            await _set_tier(db, existing.id, tier)
        return ChalybProvisionResult(
            tenant_id=existing.id,
            api_token=raw,
            duplicate=True,
        )

    # ── Self-healing email fallback (B2 reconciliation layer) ────────────
    # No tenant claims this external_user_id, but a CLI-era tenant may
    # already exist for the same human under a different identity (the
    # email). Pre-integration tenants ALL look like this — they were
    # created by `chalybclip tenants add` before Chalyb sent any
    # external_user_id at all.
    #
    # Claim the orphan tenant, backfill external_user_id, sync tier,
    # mint a fresh token. The caller (Chalyb) gets back the SAME
    # tenant_id it would have created, so the engine_subscriptions row
    # on the Chalyb side stays consistent.
    #
    # Returning duplicate=True signals "we recognized this already"
    # which Chalyb maps to 409 (idempotent success).
    orphan = await tenants.find_by_user_email(email)
    if orphan is not None:
        _log.info(
            "claiming orphan tenant by email · tenant=%s email=%s external_user_id=%s",
            orphan.id, email, external_user_id,
        )
        await tenants.set_external_user_id(orphan.id, external_user_id)
        if tier and tier != orphan.tier:
            await _set_tier(db, orphan.id, tier)
        raw, hash_ = mint_token()
        with bound_tenant(orphan.id):
            await ApiTokensRepo(db).create(hash_=hash_, scope="full")
        return ChalybProvisionResult(
            tenant_id=orphan.id,
            api_token=raw,
            duplicate=True,
        )

    # New tenant. Name defaults to display_name → email local-part. Per
    # CLAUDE.md the tenant name is human-display, not an identifier.
    tenant_name = (display_name or email.split("@", maxsplit=1)[0]).strip() or email
    tenant = await tenants.create(name=tenant_name)

    # Persist the cross-system link so the next provisioning call hits the
    # duplicate path. set_external_user_id is a separate UPDATE so the
    # create() signature stays minimal (tenant_id is the FK in many tables;
    # the external link is metadata, not an identifier).
    await tenants.set_external_user_id(tenant.id, external_user_id)

    # Apply tier override if Chalyb sent one. Default in the column is 'free'
    # so we only UPDATE for upgrades.
    if tier and tier != "free":
        await _set_tier(db, tenant.id, tier)

    # Create the first user + first api_token under the new tenant.
    raw, hash_ = mint_token()
    with bound_tenant(tenant.id):
        await UsersRepo(db).create(email=email, role="owner")
        await ApiTokensRepo(db).create(hash_=hash_, scope="full")

    return ChalybProvisionResult(
        tenant_id=tenant.id,
        api_token=raw,
        duplicate=False,
    )


async def sync_tenant_tier(db: Database, *, tenant_id: str, tier: str) -> None:
    """Idempotent tier write. Called from /auth/sso on every login so the
    user's tier stays in sync with what Chalyb thinks it should be. No-op if
    the tier didn't change.

    Normalizes the incoming label first: Chalyb sends aliases like
    `partner` for our top tier `all_access`. An UNRECOGNIZED label
    (typo / brand-new tier we haven't mapped) resolves to None and we
    SKIP the write — better to keep the tenant's current tier than to
    silently overwrite it with garbage. The pre-normalization version
    wrote `partner` verbatim, so partner tenants ended up with a tier
    string no gate recognized → treated as free everywhere.
    """
    from chalybclip.tiers import resolve_tier_alias

    canonical = resolve_tier_alias(tier)
    if canonical is None:
        return
    tenant = await TenantsRepo(db).get(tenant_id)
    if tenant is None or tenant.tier == canonical:
        return
    await _set_tier(db, tenant_id, canonical)


async def _set_tier(db: Database, tenant_id: str, tier: str) -> None:
    """Write `tenants.tier` directly. Done as a raw UPDATE because there's no
    dedicated repo method for it yet and we don't want to expand TenantsRepo's
    surface for a single Chalyb-driven field. If a future slice generalizes
    tier management to other paths (CLI, admin UI, billing webhook), promote
    this to TenantsRepo.set_tier."""
    conn = await db.connect()
    await conn.execute(
        "UPDATE tenants SET tier = ? WHERE id = ?",
        (tier, tenant_id),
    )
    await conn.commit()


async def mint_session_token_for_tenant(
    db: Database,
    *,
    tenant_id: str,
) -> str:
    """Issue a short-lived bearer token for an SSO-initiated session.

    Called from GET /auth/sso AFTER the HMAC token from Chalyb is verified.
    We don't reuse the provisioning api_token because:
      (a) it has indefinite lifetime — bad for cookies
      (b) it's stored by Chalyb as integration credential and shouldn't
          double as the user's interactive session

    For now we mint a fresh `full`-scope token and set it as the cookie.
    Later we'll add a `session` scope + a short expiry on the row when the
    token_scope literal accepts it.
    """
    tenant = await TenantsRepo(db).get(tenant_id)
    if tenant is None:
        raise ChalybServiceError(f"tenant not found: {tenant_id}")

    raw, hash_ = mint_token()
    with bound_tenant(tenant_id):
        await ApiTokensRepo(db).create(hash_=hash_, scope="full")
    return raw
