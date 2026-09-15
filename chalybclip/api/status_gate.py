"""Tenant status gating — refuse work for paused tenants.

Slice NX.2: Chalyb can pause a tenant remotely (PRO subscribers swap their
live engine; the OLD engine, that's us, gets told to pause). When paused, we:
  - Refuse to start new pipeline jobs (ingest → transcribe → publish)
  - Refuse to publish queued clips
  - Refuse to fire LLM calls
  - Still serve the dashboard, but with a clear banner explaining why
    everything's frozen.

Read-only routes (dashboard browsing, viewing existing clips) stay open so
the user can still see what they made before the pause + come back when
re-activated. This file gives handlers a dependency-style check they can
use to block ONLY the work-doing routes.

Usage in a route handler:

    from fastapi import Depends
    from chalybclip.api.status_gate import require_active_tenant

    @router.post("/streams/start", dependencies=[Depends(require_active_tenant)])
    async def start_stream(...):
        ...

When the tenant is paused, FastAPI returns 423 Locked with a JSON body
explaining the reason. 423 is the canonical HTTP code for "resource is
locked / temporarily unavailable" — clearer signal than 403 (which means
'forbidden permanently') or 503 (which means 'server-side broken').
"""

from __future__ import annotations

from fastapi import HTTPException, Request, status


def require_active_tenant(request: Request) -> None:
    """Refuse the request if the bound tenant is paused or cancelled.

    Relies on `request.state.tenant_status` being set by BearerAuthMiddleware.
    If unset (somehow this dep got applied to a public route), default to
    'active' so we don't accidentally lock out unauthenticated traffic.
    """
    tenant_status: str = getattr(request.state, "tenant_status", "active") or "active"
    if tenant_status == "active":
        return
    # Distinguish paused (recoverable — they can swap back via Chalyb) from
    # cancelled (terminal — they need to re-subscribe).
    if tenant_status == "paused":
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail={
                "reason": "tenant_paused",
                "message": (
                    "Tu cuenta ChalyClip está en pausa. Esto pasa cuando "
                    "activas otro engine (ChalybStream, ChalybTrade, etc.) en "
                    "Chalyb con un plan Pro. Vuelve a Chalyb y selecciona "
                    "ChalyClip como tu engine en vivo para reanudar."
                ),
            },
        )
    # 'cancelled' or any other future state.
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "reason": "tenant_inactive",
            "message": f"Tu cuenta está en estado '{tenant_status}'. Contacta soporte.",
        },
    )


# Tiers that can publish to external platforms (TikTok, YouTube, IG Reels).
# Free can still ingest VODs, generate clips, and DOWNLOAD them — only
# outbound publishing is paywalled. The watermark applies separately at the
# clip-export layer (see brand_kits.show_chalybclip_credit + tier check there).
#
# Tier sets live in chalybclip.tiers (single source of truth). request.state
# .tenant_tier is already normalized to a canonical tier by the auth
# middleware, so these gates compare directly.
from chalybclip.tiers import PAID_TIERS as _PAID_TIERS
from chalybclip.tiers import TOP_TIERS as _TOP_TIERS


def require_paid_tier(request: Request) -> None:
    """Block free-tier tenants from any paid perk. Free can still create +
    download clips locally with a watermark.

    Returns 402 Payment Required — the canonical HTTP code for "you need to
    pay to use this feature". The JSON detail gives the client a paywall hint
    + a deep link back to Chalyb's subscription page.
    """
    tier: str = getattr(request.state, "tenant_tier", "free") or "free"
    if tier in _PAID_TIERS:
        return
    raise HTTPException(
        status_code=status.HTTP_402_PAYMENT_REQUIRED,
        detail={
            "reason": "paywall_paid",
            "current_tier": tier,
            "message": (
                "Esta función requiere el plan Pro o All-Access. Puedes "
                "descargar el clip con la marca de agua de ChalyClip, o subir "
                "tu plan."
            ),
            "upgrade_url": "https://chalyb.com/app/subscription",
        },
    )


def require_top_tier(request: Request) -> None:
    """Block everything below the TOP tier (all_access).

    Gates the legacy native publish-jobs paths (clips.py /publish +
    dashboard.py Buffer matrix). The Zernio publish surface is gated by
    require_paid_tier instead — `pro` CAN publish there, capped at one
    connected account (tiers.zernio_account_limit); all_access (VIP) is
    unlimited. The top tier includes Chalyb's `partner` alias,
    normalized to all_access at the auth read choke point.

    Returns 402 Payment Required with a paywall hint.
    """
    tier: str = getattr(request.state, "tenant_tier", "free") or "free"
    if tier in _TOP_TIERS:
        return
    raise HTTPException(
        status_code=status.HTTP_402_PAYMENT_REQUIRED,
        detail={
            "reason": "paywall_publish",
            "current_tier": tier,
            "message": (
                "Publicar en TikTok/YouTube/Instagram requiere el plan "
                "All-Access. Puedes descargar el clip y publicarlo "
                "manualmente, o subir tu plan."
            ),
            "upgrade_url": "https://chalyb.com/app/subscription",
        },
    )
