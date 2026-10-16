"""Hub API core router — /api/hub/v1 (shared foundation).

Auth, tenant binding and the error format live in `chalybclip.api.hub_api`.
Feature routers sit next to this one and are owned separately:

  * hub.py          — core: identity + everything outside editor/publish
  * hub_editor.py   — the clip editor (/clips/{id} …)
  * hub_publish.py  — Zernio publishing (/publish/zernio …)

Each builds its router with `hub_router()` so they share the prefix and the
tenant binding; app.py includes all three.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request

from chalybclip.api.hub_api import hub_router, hub_tenant

router = hub_router(tags=["hub"])

HUB_API_VERSION = 1


@router.get("/me")
async def hub_me(request: Request, tenant_id: str = hub_tenant()) -> dict[str, Any]:
    """Who the hub is acting as, as this engine sees it. The hub calls this
    to confirm the link works (and the tenant exists) before rendering."""
    balance = getattr(request.state, "token_balance", None)
    return {
        "api_version": HUB_API_VERSION,
        "tenant_id": tenant_id,
        "user_id": request.state.hub_user_id,
        "tier": getattr(request.state, "tenant_tier", "free"),
        "status": getattr(request.state, "tenant_status", "active"),
        "is_admin": bool(getattr(request.state, "is_admin", False)),
        "balance": (
            {
                "remaining": balance["remaining"],
                "unlimited": balance["unlimited"],
                "monthly_used": balance["monthly_used"],
                "at": str(balance["at"]),
            }
            if balance
            else None
        ),
    }
