"""On-demand balance fetcher.

The reporter (reporter.py) updates the balance cache as a side-effect of
LLM calls — but a user who hasn't run any AI work yet has no cache, so the
nav chip shows the muted "— tokens" placeholder. This module exposes an
explicit fetch so the chip can be clicked to populate the cache without
waiting for the next LLM call.

GET {CHALYB_BASE_URL}/api/engines/chalybclip/usage/balance?external_user_id=<id>
  Auth: Bearer CHALYB_ADMIN_TOKEN
  Returns: { balance: { remaining, unlimited, monthlyUsed, periodStart, ... } }

Always best-effort: timeout 5s, swallow on error. The chip can show stale
data rather than break the dashboard render.
"""

from __future__ import annotations

import datetime as _dt
import logging
import time
from dataclasses import dataclass, field

import httpx

from chalybclip.db import Database, TenantsRepo
from chalybclip.settings import get_settings

_log = logging.getLogger("chalybclip.chalyb.balance")

_FETCH_TIMEOUT_S = 5.0


@dataclass
class DiagPingResult:
    """Rich outcome for the diag page's ping. Captures the exact failure
    mode (HTTP status + body excerpt, exception type, or skip reason) so
    the operator never has to grep Railway logs to figure out which leg
    of the chain broke.

    `outcome` values:
      'ok'            — 2xx response, balance updated.
      'skip_no_url'   — CHALYB_BASE_URL unset.
      'skip_no_token' — CHALYB_ADMIN_TOKEN unset.
      'skip_no_external_id' — tenant.external_user_id is NULL.
      'http_error'    — response.status_code >= 400. status + body filled.
      'timeout'       — httpx.TimeoutException.
      'network'       — any other connect/SSL/DNS exception.
      'parse_error'   — 2xx but response wasn't JSON or shape was wrong.
    """

    outcome: str
    url: str | None = None
    status_code: int | None = None
    elapsed_ms: int | None = None
    body_excerpt: str | None = None
    exception_type: str | None = None
    exception_message: str | None = None
    balance_remaining: int | None = None
    balance_unlimited: bool | None = None
    balance_monthly_used: int | None = None
    notes: list[str] = field(default_factory=list)


async def fetch_balance_now_verbose(
    db: Database, *, tenant_id: str
) -> DiagPingResult:
    """Same as fetch_balance_now but returns a structured DiagPingResult.

    Used by the diag page so the operator can see the exact HTTP status,
    response body excerpt, and timing. Also updates the balance cache on
    success (so calling this from the diag still populates the chip).

    Built as a separate function from fetch_balance_now so the existing
    hot-path callers (chip refresh, reporter cache hydration) keep their
    cheap True/False return type and we don't pay the dataclass cost on
    every LLM call.
    """
    settings = get_settings()
    base = settings.chalyb_base_url
    token = settings.chalyb_admin_token

    if not base:
        return DiagPingResult(outcome="skip_no_url")
    if not token:
        return DiagPingResult(outcome="skip_no_token")

    try:
        tenant = await TenantsRepo(db).get(tenant_id)
    except Exception as e:  # noqa: BLE001
        return DiagPingResult(
            outcome="network",
            exception_type=type(e).__name__,
            exception_message=str(e)[:300],
        )
    if tenant is None or not tenant.external_user_id:
        return DiagPingResult(outcome="skip_no_external_id")

    url = (
        f"{base.rstrip('/')}/api/engines/chalybclip/usage/balance"
        f"?external_user_id={tenant.external_user_id}"
    )
    headers = {"Authorization": f"Bearer {token}"}

    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S) as client:
            response = await client.get(url, headers=headers)
    except httpx.TimeoutException as e:
        return DiagPingResult(
            outcome="timeout",
            url=url,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            exception_type=type(e).__name__,
            exception_message=str(e)[:300] or f"timed out after {_FETCH_TIMEOUT_S}s",
        )
    except Exception as e:  # noqa: BLE001
        return DiagPingResult(
            outcome="network",
            url=url,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            exception_type=type(e).__name__,
            exception_message=str(e)[:300],
        )
    elapsed_ms = int((time.monotonic() - started) * 1000)

    if response.status_code >= 400:
        return DiagPingResult(
            outcome="http_error",
            url=url,
            status_code=response.status_code,
            elapsed_ms=elapsed_ms,
            body_excerpt=response.text[:500],
            notes=_interpret_http_status(response.status_code),
        )

    try:
        balance = response.json().get("balance") or {}
    except Exception as e:  # noqa: BLE001
        return DiagPingResult(
            outcome="parse_error",
            url=url,
            status_code=response.status_code,
            elapsed_ms=elapsed_ms,
            body_excerpt=response.text[:500],
            exception_type=type(e).__name__,
            exception_message=str(e)[:300],
        )

    # Mirror fetch_balance_now's cache-update side effect on success so the
    # operator's "click chip → see diag" path STILL populates the cache
    # when the integration is actually healthy.
    now_iso = _dt.datetime.now(_dt.UTC).isoformat()
    cache_note: list[str] = []
    try:
        await TenantsRepo(db).set_balance_cache(
            tenant_id,
            remaining=int(balance.get("remaining", 0)),
            unlimited=bool(balance.get("unlimited", False)),
            monthly_used=int(balance.get("monthlyUsed", 0)),
            at_iso=now_iso,
        )
    except Exception as e:  # noqa: BLE001
        # Don't downgrade the outcome — the request itself was fine.
        cache_note.append(f"cache write failed: {type(e).__name__}: {e}")

    return DiagPingResult(
        outcome="ok",
        url=url,
        status_code=response.status_code,
        elapsed_ms=elapsed_ms,
        balance_remaining=int(balance.get("remaining", 0)),
        balance_unlimited=bool(balance.get("unlimited", False)),
        balance_monthly_used=int(balance.get("monthlyUsed", 0)),
        notes=cache_note,
    )


def _interpret_http_status(status: int) -> list[str]:
    """Translate common HTTP rejection codes from Chalyb into hints the
    operator can act on. Order matters — first relevant hint wins."""
    notes: list[str] = []
    if status == 401:
        notes.append(
            "401 unauthorized: el header Authorization no llegó o tiene formato malo."
        )
    elif status == 403:
        notes.append(
            "403 forbidden: el token bearer NO coincide con CHALYBCLIP_ADMIN_TOKEN "
            "del lado Chalyb. Verifica que sean idénticos byte por byte (cuidado "
            "con newlines / espacios al pegar)."
        )
    elif status == 404:
        notes.append(
            "404 not found: probablemente el external_user_id no existe en la "
            "tabla profiles de Supabase. Verifica que el id apunte a un user "
            "real (puedes correr: SELECT id FROM auth.users WHERE email='...')."
        )
    elif status == 429:
        notes.append(
            "429 rate limited: Chalyb rechazó el ping por throttle. "
            "Reintenta en unos minutos."
        )
    elif 500 <= status < 600:
        notes.append(
            "5xx desde Chalyb: el server tuvo un error interno. "
            "Probablemente algo del lado Vercel — revisa logs de Chalyb."
        )
    return notes


async def fetch_balance_now(db: Database, *, tenant_id: str) -> bool:
    """Force-fetch the current Chalyb balance for this tenant + update cache.

    Returns True if cache was updated, False otherwise. Reasons for False
    show up in the log so the operator can debug the silent path.
    """
    settings = get_settings()
    base = settings.chalyb_base_url
    token = settings.chalyb_admin_token

    if not base:
        _log.info("fetch skipped: CHALYB_BASE_URL unset · tenant=%s", tenant_id)
        return False
    if not token:
        _log.warning("fetch skipped: CHALYB_ADMIN_TOKEN unset · tenant=%s", tenant_id)
        return False

    try:
        tenant = await TenantsRepo(db).get(tenant_id)
    except Exception:
        _log.exception("fetch failed: tenant lookup error · tenant=%s", tenant_id)
        return False
    if tenant is None or not tenant.external_user_id:
        _log.info(
            "fetch skipped: tenant has no external_user_id · tenant=%s", tenant_id
        )
        return False

    url = (
        f"{base.rstrip('/')}/api/engines/chalybclip/usage/balance"
        f"?external_user_id={tenant.external_user_id}"
    )
    headers = {"Authorization": f"Bearer {token}"}

    try:
        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S) as client:
            response = await client.get(url, headers=headers)
    except httpx.TimeoutException:
        _log.warning("fetch failed: timeout · tenant=%s url=%s", tenant_id, url)
        return False
    except Exception:  # noqa: BLE001
        _log.exception("fetch failed: unexpected error · tenant=%s", tenant_id)
        return False

    if response.status_code >= 400:
        _log.warning(
            "fetch rejected by Chalyb: %d · tenant=%s body=%s",
            response.status_code, tenant_id, response.text[:300],
        )
        return False

    try:
        balance = response.json().get("balance") or {}
    except Exception:
        _log.warning("fetch ok but JSON parse failed · tenant=%s", tenant_id)
        return False

    now_iso = _dt.datetime.now(_dt.UTC).isoformat()
    try:
        await TenantsRepo(db).set_balance_cache(
            tenant_id,
            remaining=int(balance.get("remaining", 0)),
            unlimited=bool(balance.get("unlimited", False)),
            monthly_used=int(balance.get("monthlyUsed", 0)),
            at_iso=now_iso,
        )
    except Exception:
        _log.exception("fetch ok but cache update failed · tenant=%s", tenant_id)
        return False

    _log.info(
        "balance fetched · tenant=%s remaining=%s unlimited=%s",
        tenant_id,
        balance.get("remaining"),
        balance.get("unlimited"),
    )
    return True
