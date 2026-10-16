"""Tenant resolution from a verified Chalyb launch token.

ONE implementation shared by the two places a Chalyb-signed token logs a
user in:

  * GET /auth/sso — the browser hand-off (sets the session cookie);
  * /api/hub/v1/* — the hub's server-to-server calls, where the same token
    rides in `Authorization: Bearer` on every request.

Both must land the user on the same tenant with the same side effects, so
the logic lives here rather than in either route:

  * auto-provision a tenant that doesn't exist yet (+ seed the `default`
    persona, mirroring /api/admin/tenants);
  * sync the tier, auto-link the Zernio profile, and reconcile
    `external_user_id` from the signed claims.

Every sync is idempotent and only writes when the stored value differs, so
calling this on each hub request costs one read in the steady state.
"""

from __future__ import annotations

import contextlib

import structlog

from chalybclip.db import Database

from .service import sync_tenant_tier
from .sso import SsoTokenPayload

_log = structlog.get_logger("chalybclip.integrations.chalyb.sso_tenant")

_DEFAULT_PERSONA_PROMPT = (
    "You are writing short-form social media captions "
    "for a content creator. Write tight, punchy captions "
    "that earn the swipe. Lead with a hook — no soft "
    "openers. Avoid corporate language and emoji "
    "stuffing. Match the energy of the source clip. "
    "Output the caption and nothing else."
)


async def provision_sso_tenant(db: Database, payload: SsoTokenPayload) -> None:
    """Create the tenant the token names, then seed its `default` persona.

    Slice O.23 — chalyb is the source of truth; if the signed token claims
    a tenant_id we don't have a row for yet, create it. Raises when the
    tenant row can't be created (the caller reports it); the persona seed
    is best-effort (Slice O.30 — without it the first upload crashes on
    `unknown persona 'default'`, but login must not fail over it).
    """
    from chalybclip.db import PersonasRepo, TenantsRepo
    from chalybclip.tenancy import bound_tenant

    await TenantsRepo(db).create(
        tenant_id=payload.tenant_id,
        name=payload.email or payload.tenant_id,
    )
    try:
        with bound_tenant(payload.tenant_id):
            await PersonasRepo(db).create(
                persona_id="default",
                name="Default",
                primary_language="en",
                target_languages=["en", "es"],
                voice_prompt=_DEFAULT_PERSONA_PROMPT,
                routing_tags=["general"],
            )
    except Exception:  # non-fatal seed
        pass


async def sync_sso_claims(db: Database, payload: SsoTokenPayload) -> None:
    """Apply the signed claims to an existing tenant. Never raises.

    * tier — Chalyb's effective tier (admin upgrade, payment, downgrade).
    * zernio_profile_id — only when the tenant has NONE (never clobber a
      manual link/unlink made in the dashboard).
    * external_user_id — overwritten on mismatch, not just backfilled: a
      stale link makes balance fetches and usage reports read someone
      else's Chalyb ledger.
    """
    from chalybclip.db import TenantsRepo

    if payload.tier:
        try:
            await sync_tenant_tier(db, tenant_id=payload.tenant_id, tier=payload.tier)
        except Exception:  # never break login on a tier sync
            _log.warning("sso_tenant.tier_sync_failed", tenant_id=payload.tenant_id)

    try:
        tenant = await TenantsRepo(db).get(payload.tenant_id)
    except Exception:
        return
    if tenant is None:
        return

    if payload.zernio_profile_id and not tenant.zernio_profile_id:
        with contextlib.suppress(Exception):  # never break login on the auto-link
            await TenantsRepo(db).set_zernio_profile(
                payload.tenant_id, profile_id=payload.zernio_profile_id,
            )

    if payload.user_id and tenant.external_user_id != payload.user_id:
        # Best-effort: the usage reporter stays silent until the next login.
        with contextlib.suppress(Exception):
            await TenantsRepo(db).set_external_user_id(payload.tenant_id, payload.user_id)


async def ensure_sso_tenant(db: Database, payload: SsoTokenPayload) -> bool:
    """Make sure the token's tenant exists and reflects its claims.

    Returns True when the tenant was created by this call. Raises only when
    the tenant is missing AND can't be provisioned.
    """
    from chalybclip.db import TenantsRepo

    created = False
    if await TenantsRepo(db).get(payload.tenant_id) is None:
        try:
            await provision_sso_tenant(db, payload)
            created = True
        except Exception:
            # A concurrent request may have created it between our read and
            # the insert — that's success, anything else is a real failure.
            if await TenantsRepo(db).get(payload.tenant_id) is None:
                raise
    await sync_sso_claims(db, payload)
    return created


__all__ = ["ensure_sso_tenant", "provision_sso_tenant", "sync_sso_claims"]
