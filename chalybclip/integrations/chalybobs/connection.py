"""Read + write the ChalyOBS↔ChalyClip connection flag (source of truth in
ChalyOBS). Best-effort: network/config failures return safe defaults so the
Live page never 500s on a ChalyOBS hiccup."""

from __future__ import annotations

import os

import httpx


def _base() -> str | None:
    v = os.environ.get("CHALYBOBS_INTERNAL_URL", "https://chalybobs.chalyb.com")
    return v.rstrip("/") if v else None


def _secret() -> str | None:
    # Same shared internal bearer ChalyOBS checks as CHALYBOBS_RELAY_SECRET.
    return (os.environ.get("CHALYBCLIP_INTERNAL_SIGNING_SECRET") or "").strip() or None


def is_configured() -> bool:
    return bool(_base() and _secret())


async def get_connection(external_user_id: str) -> bool:
    """Is the ChalyClip connection on for this tenant? False on any error."""
    base, secret = _base(), _secret()
    if not (base and secret and external_user_id):
        return False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{base}/api/internal/connection",
                params={"tenant": external_user_id},
                headers={"Authorization": f"Bearer {secret}"},
            )
        if resp.status_code != 200:
            return False
        return bool(resp.json().get("enabled"))
    except Exception:  # noqa: BLE001 — best-effort
        return False


async def set_connection(external_user_id: str, enabled: bool) -> bool:
    """Flip the connection. Returns the new state (or the requested value
    optimistically if the call fails — the next page load reconciles)."""
    base, secret = _base(), _secret()
    if not (base and secret and external_user_id):
        return enabled
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{base}/api/internal/connection",
                json={"tenant": external_user_id, "enabled": enabled},
                headers={"Authorization": f"Bearer {secret}"},
            )
        if resp.status_code == 200:
            return bool(resp.json().get("enabled"))
    except Exception:  # noqa: BLE001 — best-effort
        pass
    return enabled
