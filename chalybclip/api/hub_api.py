"""Shared plumbing for the hub's JSON API: /api/hub/v1/*.

The Chalyb hub (www.chalyb.com) renders every Clips screen itself and talks
to this engine server-to-server. Each request carries
`Authorization: Bearer <launch token>` — the SAME HMAC-SHA256 token the hub
signs for GET /auth/sso ({user_id, email, tenant_id, tier, exp}, secret
CHALYB_SSO_SECRET), minted fresh per request with a short TTL.

This module owns:

  * `authenticate_hub_request` — called by `BearerAuthMiddleware` for hub
    paths. Verifies the token and resolves the tenant exactly like
    /auth/sso (auto-provision included, see integrations/chalyb/sso_tenant).
    Cookies and api_tokens are NOT accepted on hub paths: only the hub can
    call them, and a browser can't be tricked into sending a hub token.
  * the shared error format, installed app-wide but only applied to hub
    paths (every other route keeps FastAPI's default bodies):

        {"ok": false,
         "error": {"code": "<code>", "message": "<es copy>",
                   "reason": "<engine reason>|null", "retryable": bool}}

    `code` is one of HUB_ERROR_CODES; the hub maps it onto its normalized
    tool errors. 402/413/429 refusals from the consumption contract keep
    their admission `reason` (no_tokens, minutes_cap, upload_too_large…).
  * `hub_router()` — the APIRouter factory every hub router uses (core,
    editor, publish), so they share the prefix and the tenant binding.

Successful responses are plain JSON (no envelope).
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from chalybclip.db import Database
from chalybclip.integrations.chalyb.admission import AdmissionRefused

HUB_PREFIX = "/api/hub/v1"

_log = structlog.get_logger("chalybclip.api.hub")

# code → default HTTP status. Keep in sync with the hub's
# src/lib/tools/adapters/clips/errors.ts.
HUB_ERROR_CODES: dict[str, int] = {
    "bad_request": 400,
    "unauthorized": 401,
    "no_credits": 402,
    "needs_plan": 403,
    "forbidden": 403,
    "not_found": 404,
    "conflict": 409,
    "too_large": 413,
    "rate_limited": 429,
    "internal": 500,
    "unavailable": 503,
}

_DEFAULT_MESSAGES: dict[str, str] = {
    "bad_request": "La solicitud no es válida.",
    "unauthorized": "Tu sesión no es válida. Vuelve a entrar.",
    "no_credits": "No tienes tokens suficientes. Recarga tu saldo o mejora tu plan.",
    "needs_plan": "Tu plan no incluye esta función.",
    "forbidden": "No tienes permiso para hacer esto.",
    "not_found": "No encontramos lo que buscas.",
    "conflict": "Esto cambió mientras tanto. Recarga e inténtalo de nuevo.",
    "too_large": "El archivo supera el tamaño máximo de tu plan.",
    "rate_limited": "Llegaste a un límite de tu plan. Inténtalo más tarde.",
    "internal": "Algo falló de nuestro lado. Inténtalo de nuevo.",
    "unavailable": "Clips no está disponible en este momento. Inténtalo en un rato.",
}

_RETRYABLE = frozenset({"internal", "unavailable", "rate_limited"})


def is_hub_path(path: str) -> bool:
    return path == HUB_PREFIX or path.startswith(HUB_PREFIX + "/")


class HubError(Exception):
    """Raise from a hub route to answer with the shared error format."""

    def __init__(
        self,
        code: str,
        message: str | None = None,
        *,
        status_code: int | None = None,
        reason: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        if code not in HUB_ERROR_CODES:
            raise ValueError(f"unknown hub error code {code!r}")
        super().__init__(message or code)
        self.code = code
        self.message = message or _DEFAULT_MESSAGES[code]
        self.status_code = status_code or HUB_ERROR_CODES[code]
        self.reason = reason
        self.retryable = (code in _RETRYABLE) if retryable is None else retryable


def hub_error_response(
    code: str,
    message: str | None = None,
    *,
    status_code: int | None = None,
    reason: str | None = None,
    retryable: bool | None = None,
) -> JSONResponse:
    err = HubError(
        code, message, status_code=status_code, reason=reason, retryable=retryable
    )
    return _render(err)


def _render(err: HubError) -> JSONResponse:
    headers = {"WWW-Authenticate": "Bearer"} if err.status_code == 401 else None
    return JSONResponse(
        status_code=err.status_code,
        content={
            "ok": False,
            "error": {
                "code": err.code,
                "message": err.message,
                "reason": err.reason,
                "retryable": err.retryable,
            },
        },
        headers=headers,
    )


def _code_for_status(status_code: int, reason: str | None) -> str:
    if status_code == 402:
        # The tier paywalls (status_gate.require_paid_tier / require_top_tier)
        # are a plan question, not a balance one.
        return "needs_plan" if (reason or "").startswith("paywall") else "no_credits"
    if status_code == 422:
        return "bad_request"
    if status_code == 423:
        # status_gate.require_active_tenant: the tenant is paused/cancelled.
        return "forbidden"
    for code, st in HUB_ERROR_CODES.items():
        if st == status_code and code not in ("needs_plan",):
            return code
    if 400 <= status_code < 500:
        return "bad_request"
    return "internal"


def from_http_exception(exc: StarletteHTTPException) -> HubError:
    """Map an HTTPException raised by reused engine code (status gates,
    admission's refusal_http, dashboard helpers) onto the hub format."""
    detail: Any = exc.detail
    reason: str | None = None
    message: str | None = None
    if isinstance(detail, dict):
        reason = detail.get("reason") if isinstance(detail.get("reason"), str) else None
        msg = detail.get("message")
        message = msg if isinstance(msg, str) else None
    elif isinstance(detail, str) and exc.status_code in (402, 403, 413, 429, 503):
        # Those carry Spanish user copy (admission refusals, gates).
        message = detail
    code = _code_for_status(exc.status_code, reason)
    # Unmapped statuses (e.g. 405, 410) fall back to the code's own status.
    status_code = exc.status_code if exc.status_code >= 400 else HUB_ERROR_CODES[code]
    return HubError(code, message, status_code=status_code, reason=reason)


def from_admission(exc: AdmissionRefused) -> HubError:
    code = _code_for_status(exc.status_code, exc.reason)
    return HubError(code, exc.user_message, status_code=exc.status_code, reason=exc.reason)


def install_hub_error_handlers(app: FastAPI) -> None:
    """Hub paths answer in the shared format; every other path keeps
    FastAPI's default behaviour byte-for-byte."""

    async def _hub_error(_request: Request, exc: Exception) -> Response:
        assert isinstance(exc, HubError)
        return _render(exc)

    async def _http(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, StarletteHTTPException)
        if is_hub_path(request.url.path):
            return _render(from_http_exception(exc))
        return await http_exception_handler(request, exc)

    async def _validation(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, RequestValidationError)
        if is_hub_path(request.url.path):
            fields = [
                ".".join(str(p) for p in e.get("loc", ()) if p != "body")
                for e in exc.errors()
            ]
            msg = "Revisa estos campos: " + ", ".join(f for f in fields if f) if fields else None
            return _render(HubError("bad_request", msg, status_code=422))
        return await request_validation_exception_handler(request, exc)

    async def _admission(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, AdmissionRefused)
        if is_hub_path(request.url.path):
            return _render(from_admission(exc))
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.user_message})

    async def _unhandled(request: Request, exc: Exception) -> Response:
        if is_hub_path(request.url.path):
            _log.exception("hub.unhandled", path=request.url.path, error=type(exc).__name__)
            return _render(HubError("internal"))
        return PlainTextResponse("Internal Server Error", status_code=500)

    app.add_exception_handler(HubError, _hub_error)
    app.add_exception_handler(StarletteHTTPException, _http)
    app.add_exception_handler(RequestValidationError, _validation)
    app.add_exception_handler(AdmissionRefused, _admission)
    app.add_exception_handler(Exception, _unhandled)


async def authenticate_hub_request(request: Request, db: Database) -> Response | None:
    """Verify the hub's Bearer launch token and bind the tenant on
    `request.state`. Returns an error response to send, or None on success.

    Accepts ONLY `Authorization: Bearer <hmac token>`: no cookie (a hub path
    must not be callable from a browser session) and no api_token row.
    """
    from chalybclip.integrations.chalyb import SsoTokenError, verify_sso_token
    from chalybclip.integrations.chalyb.sso_tenant import ensure_sso_tenant
    from chalybclip.settings import get_settings

    secret = get_settings().chalyb_sso_secret
    if not secret:
        _log.warning("hub.auth.secret_missing")
        return hub_error_response("unavailable", reason="sso_secret_missing")

    header = request.headers.get("authorization", "")
    token = header[len("bearer ") :].strip() if header.lower().startswith("bearer ") else ""
    if not token:
        return hub_error_response("unauthorized", reason="missing_token")

    try:
        payload = verify_sso_token(token, secret=secret)
    except (SsoTokenError, ValueError) as e:
        # ValueError covers UnicodeEncodeError on a non-ASCII token.
        _log.info("hub.auth.rejected", why=str(e)[:80])
        return hub_error_response("unauthorized", reason="invalid_token")

    try:
        await ensure_sso_tenant(db, payload)
    except Exception as e:
        _log.error(
            "hub.auth.provision_failed",
            tenant_id=payload.tenant_id,
            error=f"{type(e).__name__}: {e}"[:200],
        )
        return hub_error_response("internal", reason="tenant_provision_failed")

    request.state.tenant_id = payload.tenant_id
    request.state.token_scope = "full"
    request.state.hub_user_id = payload.user_id
    request.state.hub_email = payload.email
    return None


async def _bind_hub_tenant(request: Request):  # type: ignore[no-untyped-def]
    """Same contextvar binding as deps.tenant_binder, but refuses any
    request that didn't come through `authenticate_hub_request`."""
    from chalybclip.tenancy import bound_tenant

    tenant_id = getattr(request.state, "tenant_id", None)
    if tenant_id is None or getattr(request.state, "hub_user_id", None) is None:
        raise HubError("unauthorized")
    with bound_tenant(tenant_id):
        yield tenant_id


def hub_tenant() -> Any:
    """`tenant_id: str = hub_tenant()` in a hub route signature."""
    return Depends(_bind_hub_tenant)


def hub_router(*, tags: list[str] | None = None) -> APIRouter:
    """APIRouter for /api/hub/v1 with the tenant binding applied to every
    route. Routes that need the tenant id declare `tenant_id = hub_tenant()`."""
    return APIRouter(
        prefix=HUB_PREFIX,
        tags=tags or ["hub"],
        dependencies=[Depends(_bind_hub_tenant)],
    )


__all__ = [
    "HUB_ERROR_CODES",
    "HUB_PREFIX",
    "HubError",
    "authenticate_hub_request",
    "hub_error_response",
    "hub_router",
    "hub_tenant",
    "install_hub_error_handlers",
    "is_hub_path",
]
