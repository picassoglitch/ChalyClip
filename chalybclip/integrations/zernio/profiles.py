"""Tenant ↔ Zernio profile creation + persistence.

`create_profile_for_tenant(...)` is the entry point used by the
dashboard's inline "Create profile" action. It calls Zernio's
`POST /profiles` (a real server-side create that returns a generated
`_id` like `prof_abc123`), then persists that id + the operator-chosen
name on the tenant row.

This replaces an earlier shortcut that fabricated the profileId from
the tenant id without ever calling Zernio — Zernio requires the
profile to exist before you can connect accounts or publish under it.

Single-profile model: each tenant has at most one Zernio profile
(`tenant.zernio_profile_id`). Creating again overwrites the binding
(use after /unlink to start fresh); connecting accounts to the new
profile is a separate step.
"""
from __future__ import annotations

import logging

from chalybclip.db import Database, TenantsRepo
from chalybclip.integrations.zernio.client import (
    ZernioClient,
    ZernioError,
    ZernioProfile,
)

_log = logging.getLogger("chalybclip.integrations.zernio.profiles")


async def create_profile_for_tenant(
    *,
    db: Database,
    tenant_id: str,
    client: ZernioClient,
    name: str,
    description: str | None = None,
) -> ZernioProfile:
    """Create a Zernio profile for this tenant and persist the binding.

    Raises `ZernioError` if the tenant row is missing or Zernio rejects
    the create. The returned profile's `profile_id` is what connect +
    publish use as `profileId`.
    """
    repo = TenantsRepo(db)
    tenant = await repo.get(tenant_id)
    if tenant is None:
        raise ZernioError(f"tenant not found: {tenant_id}")

    profile = await client.create_profile(name=name, description=description)
    await repo.set_zernio_profile(
        tenant_id,
        profile_id=profile.profile_id,
        profile_name=profile.name,
    )
    _log.info(
        "zernio.profile_created tenant=%s profile_id=%s name=%s",
        tenant_id, profile.profile_id, profile.name,
    )
    return profile


def _auto_profile_name(tenant_id: str) -> str:
    """Deterministic name for an auto-provisioned profile, so the
    ensure step can find-or-create idempotently (survives DB resets and
    concurrent first-loads without duplicating)."""
    return f"ChalybClip {tenant_id}"


async def ensure_zernio_profile_for_tenant(
    *,
    db: Database,
    tenant_id: str,
    client: ZernioClient,
) -> str | None:
    """Guarantee the tenant has a Zernio profile, creating one if not,
    and return its profileId. Idempotent + safe:

      1. tenant already linked → return the stored id (no Zernio call).
      2. else look for an existing profile named `ChalybClip <tenant_id>`
         (a prior auto-create whose local link was lost) → re-link it.
      3. else create one with that deterministic name → link it.

    This is what makes the Publish Center tabs appear automatically —
    every gate is on tenants.zernio_profile_id. It does NOT touch a
    profile the operator manually created/connected accounts under
    (different name); that path stays the manual "Link existing"
    claim. Best-effort: returns None on any Zernio error (the caller
    falls back to the manual create/link form). Never raises.
    """
    repo = TenantsRepo(db)
    tenant = await repo.get(tenant_id)
    if tenant is None:
        return None
    if tenant.zernio_profile_id:
        return tenant.zernio_profile_id

    wanted = _auto_profile_name(tenant_id)
    try:
        for existing in await client.list_profiles():
            if existing.name == wanted:
                await repo.set_zernio_profile(
                    tenant_id, profile_id=existing.profile_id,
                    profile_name=existing.name,
                )
                _log.info(
                    "zernio.profile_relinked tenant=%s profile_id=%s",
                    tenant_id, existing.profile_id,
                )
                return existing.profile_id
        profile = await client.create_profile(name=wanted)
    except ZernioError as e:
        _log.warning(
            "zernio.profile_autocreate_failed tenant=%s err=%s", tenant_id, e,
        )
        return None
    await repo.set_zernio_profile(
        tenant_id, profile_id=profile.profile_id, profile_name=profile.name,
    )
    _log.info(
        "zernio.profile_autocreated tenant=%s profile_id=%s",
        tenant_id, profile.profile_id,
    )
    return profile.profile_id


__all__ = ["create_profile_for_tenant", "ensure_zernio_profile_for_tenant"]
