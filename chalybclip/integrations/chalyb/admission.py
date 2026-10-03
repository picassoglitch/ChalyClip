"""Hub admission client — POST /usage/admit, /usage/settle (heartbeat).

Contract: chalyb docs/engines/consumption-contract.md. The hub is the only
place that knows a user's tier, balance and caps; ChalyClip asks it before
spending anything heavy and does no work when it says no.

    admit(...)     → AdmitResult (allowed) or raises AdmissionRefused
    heartbeat(...) → pushes a long run's reservation expiry out

Terminal settles go through the outbox (`outbox.enqueue_settle`) so they
survive a crash; only the heartbeat is sent directly — a late heartbeat is
worthless, so there's nothing to make durable.

Failure policy is FAIL-CLOSED: if CHALYB_BASE_URL is set and the hub can't
be reached (timeout, network, 5xx, auth/config 4xx), the job is refused
with a "try again" message instead of running unmetered. The one exception
is CHALYB_BASE_URL unset — standalone/local dev, where there's no hub to
ask — which admits with a warning log. A tenant that was never linked to
Chalyb (no external_user_id: CLI/operator-created, no end user can create
one) is also admitted with a warning, since there's no user to bill.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from chalybclip.db import Database, TenantsRepo
from chalybclip.errors import ChalybClipError
from chalybclip.settings import get_settings

_log = logging.getLogger("chalybclip.chalyb.admission")

# Admission sits on the request path (upload/kickoff), so keep it tight:
# a hub that can't answer in ~10s is treated as down (fail-closed).
_ADMIT_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
# One quick retry on a transport error / 5xx before refusing.
_ADMIT_RETRY_DELAY_S = 0.5
_HEARTBEAT_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


# reason → (HTTP status, user-facing Spanish message). Statuses: 413 for
# size, 402 for money, 429 for caps/concurrency, 503 when the hub is down.
REFUSAL_MESSAGES: dict[str, tuple[int, str]] = {
    "upload_too_large": (
        413,
        "El archivo supera el tamaño máximo de tu plan. Sube un video más "
        "ligero o mejora tu plan.",
    ),
    "video_too_long": (
        413,
        "El video es más largo de lo que permite tu plan. Recórtalo o mejora "
        "tu plan para procesar videos más largos.",
    ),
    "storage_full": (
        413,
        "Tu almacenamiento está lleno. Borra videos antiguos o mejora tu plan "
        "para seguir subiendo.",
    ),
    "minutes_cap": (
        429,
        "Llegaste al límite de minutos procesados de este mes. Se renueva el "
        "próximo ciclo, o puedes mejorar tu plan.",
    ),
    "jobs_cap": (
        429,
        "Llegaste al límite de trabajos de este mes. Se renueva el próximo "
        "ciclo, o puedes mejorar tu plan.",
    ),
    "concurrency": (
        429,
        "Ya tienes el máximo de trabajos corriendo al mismo tiempo. Espera a "
        "que termine alguno e inténtalo de nuevo.",
    ),
    "streams_cap": (
        429,
        "Tu plan no incluye más transmisiones en vivo este mes.",
    ),
    "no_tokens": (
        402,
        "No tienes tokens suficientes para procesar este video. Recarga tu "
        "saldo o mejora tu plan.",
    ),
    "boost_unavailable": (
        402,
        "Tu plan no puede usar el carril Boost. Desactívalo e inténtalo de "
        "nuevo, o mejora tu plan.",
    ),
    "already_settled": (
        409,
        "Este trabajo ya se cerró. Vuelve a lanzarlo para procesarlo de nuevo.",
    ),
    "hub_unavailable": (
        503,
        "No pudimos verificar tu saldo con Chalyb en este momento. Inténtalo "
        "de nuevo en unos minutos.",
    ),
}
_UNKNOWN_REFUSAL = (
    429,
    "Tu plan no permite procesar este video en este momento.",
)


class AdmissionRefused(ChalybClipError):
    """The hub said no (or couldn't be asked). Carries what a route needs
    to answer the user: an HTTP status and a Spanish message."""

    def __init__(self, reason: str, *, detail: dict[str, Any] | None = None):
        status, message = REFUSAL_MESSAGES.get(reason, _UNKNOWN_REFUSAL)
        super().__init__(message)
        self.reason = reason
        self.status_code = status
        self.user_message = message
        self.detail = detail or {}


@dataclass(frozen=True)
class AdmitRequest:
    external_job_id: str
    job_class: str = "job"  # "job" | "stream"
    operation: str = "clips.pipeline"
    est_tokens: int = 0
    upload_mb: float = 0.0
    source_minutes: float = 0.0
    storage_mb_after: float = 0.0
    boost: bool | None = None
    ttl_seconds: int = 10800


@dataclass(frozen=True)
class AdmitResult:
    allowed: bool
    reservation_id: str | None = None
    lane: str = "standard"
    boost_fee_tokens: int = 0
    limits: dict[str, Any] = field(default_factory=dict)
    balance: dict[str, Any] = field(default_factory=dict)
    # True when no hub was asked (standalone dev / unlinked tenant): the
    # run is unmetered and has no reservation to settle.
    local: bool = False

    @property
    def max_upload_bytes(self) -> int | None:
        """The tier's upload ceiling in bytes, if the hub sent one."""
        mb = self.limits.get("max_upload_mb")
        try:
            return int(float(mb) * 1024 * 1024) if mb is not None and float(mb) > 0 else None
        except (TypeError, ValueError):
            return None


def _local_result() -> AdmitResult:
    return AdmitResult(allowed=True, local=True)


async def admit(
    db: Database,
    *,
    tenant_id: str,
    request: AdmitRequest,
    client: httpx.AsyncClient | None = None,
) -> AdmitResult:
    """Ask the hub whether this job may run. Returns the admission or
    raises `AdmissionRefused` (refusal OR hub unreachable — fail-closed)."""
    settings = get_settings()
    base = settings.chalyb_base_url
    token = settings.chalyb_admin_token
    if not base:
        _log.warning(
            "admission skipped (CHALYB_BASE_URL unset — local dev, unmetered) · "
            "tenant=%s job=%s", tenant_id, request.external_job_id,
        )
        return _local_result()
    if not token:
        _log.error("admission refused: CHALYB_ADMIN_TOKEN unset · tenant=%s", tenant_id)
        raise AdmissionRefused("hub_unavailable", detail={"error": "admin token unset"})

    tenant = await TenantsRepo(db).get(tenant_id)
    if tenant is None:
        raise AdmissionRefused("hub_unavailable", detail={"error": "tenant not found"})
    if not tenant.external_user_id:
        _log.warning(
            "admission skipped: tenant not linked to Chalyb (no external_user_id) · "
            "tenant=%s job=%s", tenant_id, request.external_job_id,
        )
        return _local_result()

    body: dict[str, Any] = {
        "external_user_id": tenant.external_user_id,
        "external_job_id": request.external_job_id,
        "class": request.job_class,
        "operation": request.operation,
        "est_tokens": max(0, int(request.est_tokens)),
        "upload_mb": round(max(0.0, float(request.upload_mb)), 2),
        "source_minutes": round(max(0.0, float(request.source_minutes)), 2),
        "storage_mb_after": round(max(0.0, float(request.storage_mb_after)), 2),
        "boost": request.boost,
        "ttl_seconds": int(request.ttl_seconds),
    }
    url = f"{base.rstrip('/')}/api/engines/chalybclip/usage/admit"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    own_client = client is None
    http = client or httpx.AsyncClient(timeout=_ADMIT_TIMEOUT)
    try:
        resp: httpx.Response | None = None
        error = ""
        for attempt in (1, 2):
            if attempt > 1:
                await asyncio.sleep(_ADMIT_RETRY_DELAY_S)
            try:
                resp = await http.post(url, json=body, headers=headers)
            except httpx.TimeoutException:
                resp, error = None, "timeout"
                continue
            except Exception as e:  # noqa: BLE001 — network/TLS/DNS
                resp, error = None, f"network error: {type(e).__name__}"
                continue
            if resp.status_code >= 500:
                error = f"HTTP {resp.status_code}"
                continue
            break
    finally:
        if own_client:
            await http.aclose()

    if resp is None or resp.status_code >= 500:
        _log.error(
            "admission refused: hub unreachable (%s) · tenant=%s job=%s",
            error, tenant_id, request.external_job_id,
        )
        raise AdmissionRefused("hub_unavailable", detail={"error": error})
    if resp.status_code >= 400:
        # 401/403/404/422: our request or config is wrong — still fail
        # closed, loudly, so the operator sees it.
        _log.error(
            "admission refused: hub rejected the request HTTP %d · tenant=%s body=%s",
            resp.status_code, tenant_id, (resp.text or "")[:300],
        )
        raise AdmissionRefused(
            "hub_unavailable", detail={"error": f"HTTP {resp.status_code}"}
        )
    try:
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        raise AdmissionRefused("hub_unavailable", detail={"error": "bad JSON"}) from e

    if not data.get("allowed"):
        reason = str(data.get("reason") or "unknown")
        _log.info(
            "admission refused: %s · tenant=%s job=%s", reason, tenant_id,
            request.external_job_id,
        )
        raise AdmissionRefused(reason, detail=data)

    result = AdmitResult(
        allowed=True,
        reservation_id=data.get("reservation_id"),
        lane=str(data.get("lane") or "standard"),
        boost_fee_tokens=int(data.get("boost_fee_tokens") or 0),
        limits=dict(data.get("limits") or {}),
        balance=dict(data.get("balance") or {}),
    )
    _log.info(
        "admitted · tenant=%s job=%s reservation=%s lane=%s est_tokens=%d",
        tenant_id, request.external_job_id, result.reservation_id, result.lane,
        body["est_tokens"],
    )
    return result


async def heartbeat(
    reservation_id: str | None, *, client: httpx.AsyncClient | None = None
) -> bool:
    """Extend a running job's reservation by its TTL. Best-effort."""
    settings = get_settings()
    if not reservation_id or not settings.chalyb_base_url or not settings.chalyb_admin_token:
        return False
    url = f"{settings.chalyb_base_url.rstrip('/')}/api/engines/chalybclip/usage/settle"
    headers = {"Authorization": f"Bearer {settings.chalyb_admin_token}"}
    own_client = client is None
    http = client or httpx.AsyncClient(timeout=_HEARTBEAT_TIMEOUT)
    try:
        resp = await http.post(
            url, json={"reservation_id": reservation_id, "outcome": "heartbeat"},
            headers=headers,
        )
        ok = resp.status_code < 300
        if not ok:
            _log.warning("heartbeat HTTP %d · reservation=%s", resp.status_code, reservation_id)
        return ok
    except Exception as e:  # noqa: BLE001
        _log.warning("heartbeat failed (%s) · reservation=%s", type(e).__name__, reservation_id)
        return False
    finally:
        if own_client:
            await http.aclose()
