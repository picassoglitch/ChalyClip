"""Send the engine's user pages to their Chalyb hub screens (cutover).

Every Clips screen now lives in the hub (www.chalyb.com/app/clips/…). With
CHALYBCLIP_HUB_REDIRECT=1 a browser GET of a known dashboard page answers 303
to the matching hub page. Default off: flipping the hub's
TOOL_HUB_MODE_CHALYBCLIP=a first, then this, is the cutover; unsetting it is
the rollback.

Only the page routes in `_ROUTES` move. Everything else keeps working
exactly as today: media/download/render endpoints, operator pages
(/dashboard/_health*, /dashboard/_diag*), POSTs, /auth/sso, webhooks,
/overlay/*, /api/*. The editor and publishing workstreams add their pages
with `hub_route()`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

_ID = r"(?P<id>[A-Za-z0-9_-]{1,64})"

# (engine path regex, hub path template). First match wins.
_ROUTES: list[tuple[re.Pattern[str], str]] = []


def hub_route(pattern: str, target: str) -> None:
    """Map an engine dashboard path (regex, full match, `{id}` = an id
    segment) to a hub path (`{id}` substituted)."""
    _ROUTES.append((re.compile(pattern.replace("{id}", _ID)), target))


for _p, _t in (
    (r"/dashboard/?", "/app/clips"),
    (r"/dashboard/start", "/app/clips"),
    (r"/dashboard/url-jobs/new", "/app/clips/nuevo"),
    (r"/dashboard/streams", "/app/clips/videos"),
    (r"/dashboard/streams/{id}", "/app/clips/videos/{id}"),
    (r"/dashboard/live", "/app/clips/en-vivo"),
    (r"/dashboard/sources", "/app/clips/canales"),
    (r"/dashboard/brand-kits", "/app/clips/ajustes/estilos"),
    (r"/dashboard/brand-kits/new", "/app/clips/ajustes/estilos/nuevo"),
    (r"/dashboard/brand-kits/{id}", "/app/clips/ajustes/estilos/{id}"),
    (r"/dashboard/personas", "/app/clips/ajustes/personas"),
    (r"/dashboard/personas/{id}/edit", "/app/clips/ajustes/personas"),
    (r"/dashboard/calibration", "/app/clips/ajustes/calibracion"),
    (r"/dashboard/llm-calls", "/app/clips/ajustes/consumo"),
    (r"/dashboard/settings/llm", "/app/clips/ajustes/motor"),
    (r"/dashboard/settings/site", "/app/clips/ajustes/sitio"),
):
    hub_route(_p, _t)


def hub_redirect_enabled() -> bool:
    return os.environ.get("CHALYBCLIP_HUB_REDIRECT", "").strip().lower() in ("1", "true", "on")


def hub_target(path: str) -> str | None:
    """The hub path for an engine page path, or None to serve it here."""
    for pattern, target in _ROUTES:
        m = pattern.fullmatch(path)
        if m:
            return target.replace("{id}", m.groupdict().get("id") or "")
    return None


class HubRedirectMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.method == "GET" and hub_redirect_enabled():
            target = hub_target(request.url.path)
            if target is not None:
                base = (
                    os.environ.get("CHALYB_BASE_URL", "").strip() or "https://www.chalyb.com"
                ).rstrip("/")
                return RedirectResponse(url=f"{base}{target}", status_code=303)
        return await call_next(request)


__all__ = ["HubRedirectMiddleware", "hub_redirect_enabled", "hub_route", "hub_target"]
