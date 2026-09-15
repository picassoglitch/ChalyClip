"""HTMX-rendered dashboard.

Server-rendered Jinja2 + HTMX. Same auth path as the JSON API but reads
the token from the `chalybclip_token` cookie (set by `POST /dashboard/login`)
in addition to the `Authorization` header. The bearer middleware in
`auth.py` does the cookie fallback - this router just renders the HTML.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Mapping
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.templating import Jinja2Templates

from chalybclip.db import (
    BrandKitsRepo,
    CandidatesRepo,
    ChannelWatchesRepo,
    ClipsRepo,
    ConnectedAccountsRepo,
    Database,
    EventsRepo,
    LLMCallsRepo,
    PersonasRepo,
    PlatformSettingsRepo,
    StreamsRepo,
    TenantsRepo,
    VariantsRepo,
    ZernioPublishesRepo,
)
from chalybclip.db.models import CustomTriggerPhrases
from chalybclip.errors import ChalybClipError
from chalybclip.settings import resolve_db_target

from .._pipeline import PipelineKickoff
from ..deps import get_db, require_full_scope, tenant_binder
from ..status_gate import require_active_tenant
from .clips import _VALID_STATUS_TRANSITIONS

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
# Slice O.24 — i18n globals (`t`, `locale`) for dashboard templates.
# Same registry as the landing page; pages opt in by calling
# `{{ t('nav.publish') }}` etc.
from ..i18n import install_globals as _install_i18n  # noqa: E402
_install_i18n(templates)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

_COOKIE_NAME = "chalybclip_token"


@router.get("", include_in_schema=False)
@router.get("/", include_in_schema=False)
async def dashboard_root() -> Response:
    """Slice I.3 follow-up — operators kept hitting /dashboard/ in the
    address bar and getting a bare 404. There's no actual dashboard
    HOME page; the /dashboard/start hero ("New clip") is the canonical
    landing — same place SSO logins arrive — so redirect there. The
    bearer-auth middleware bounces unauthenticated hits to chalyb."""
    return RedirectResponse(url="/dashboard/start", status_code=303)


@router.get("/_balance/chip", response_class=HTMLResponse, include_in_schema=False)
async def balance_chip(request: Request) -> Response:
    """Slice O.47 — HTMX-polled token chip. Slice O.56 — hardened.

    Returns just the chip's HTML fragment so the nav can swap it in
    place on a poll (every 30s). No live Chalyb fetch; reads the
    cached values populated by the outbound usage reporter.

    Slice O.56 — three changes from the original O.47:
      - No tenant_binder dependency. The middleware already populates
        request.state.token_balance for authenticated requests; this
        route only needs to read it. Removing the binder eliminates
        a class of 500s where the binder's HTTPException(401) bubbled
        as a vanilla 500 (HTMX swap target was 500, not the bounce-to-
        login the binder intended for full-page requests).
      - Wraps the template render in a try/except so any future
        template bug returns a safe fallback fragment instead of
        500-spamming the nav every 30s.
      - Returns 200 with a muted placeholder chip if the user isn't
        authenticated (HTMX poll keeps spinning silently rather than
        re-flashing the login bounce on every tick).
    """
    bal = getattr(request.state, "token_balance", None)
    if not getattr(request.state, "tenant_id", None):
        # Not authenticated. HTMX is polling; just send a placeholder
        # so the nav stays consistent rather than 401-spamming.
        return HTMLResponse(
            '<a class="nc-chip nc-chip--muted" '
            'id="nc-balance-chip" '
            'style="text-decoration:none;">'
            '<i class="ti ti-coin" aria-hidden="true"></i>'
            '<span>—</span></a>',
            status_code=200,
        )
    # The `unhashable type: 'dict'` spam was the old Starlette
    # `TemplateResponse(name, context)` signature breaking under
    # Starlette 1.0, which moved `request` to the first positional
    # arg. The dict-shaped context ended up bound to `name`, and
    # Jinja hashed it for its template cache. Calling with the new
    # signature kills the error. `_coerce_balance_to_scalars` stays
    # as defense-in-depth so a future schema drift can't make the
    # 30s poll spam logs again.
    safe_bal = _coerce_balance_to_scalars(bal)
    try:
        return templates.TemplateResponse(
            request, "_balance_chip.html", {"_bal": safe_bal}
        )
    except Exception as e:  # noqa: BLE001
        from structlog import get_logger
        get_logger(__name__).warning(
            "balance_chip.render_failed", error=str(e),
            bal_type=type(bal).__name__,
        )
        # Static fallback so the nav doesn't display a broken chip.
        return HTMLResponse(
            '<a href="/dashboard/_balance/refresh" '
            'class="nc-chip nc-chip--muted" id="nc-balance-chip" '
            'title="Token balance unavailable. Click to retry." '
            'style="text-decoration:none;">'
            '<i class="ti ti-coin" aria-hidden="true"></i>'
            '<span>— tokens · retry</span></a>',
            status_code=200,
        )


def _coerce_balance_to_scalars(bal: object) -> dict | None:
    """Return a normalized balance dict with all values guaranteed
    to be scalar types the template can safely format / compare.

    The middleware should always populate the standard 4-key shape
    (remaining: int|None, unlimited: bool, monthly_used: int|None,
    at: str|None), but production saw a stream of
    `balance_chip.render_failed: unhashable type: 'dict'` errors
    that suggest something — schema drift, JSON nested decode,
    Jinja extension — was hashing somewhere. Easier to scrub upfront
    than chase the root cause across the dashboard.

    Returns None when bal is None (the template handles this).
    Returns the scrubbed dict otherwise — non-scalar values become
    None (for `at`) or 0 (for numeric fields), bool gets a strict
    bool() coerce so a stray dict-shape collapses to False.
    """
    if bal is None:
        return None
    if not isinstance(bal, dict):
        # Unexpected — token_balance is supposed to be a plain dict
        # set by auth middleware. Coerce to None so the template
        # falls through to the "no balance" branch instead of trying
        # to .get() on it.
        return None

    def _to_int(value: object) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        # str/None/dict/list/etc → fall back to 0
        return 0

    def _to_bool(value: object) -> bool:
        # bool() on a dict returns True if non-empty — not what we
        # want for a boolean flag. Explicit isinstance check first.
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return bool(value)
        return False

    def _to_str_or_none(value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        # Anything weirder (dict, list, datetime that didn't serialize)
        # → None so the template doesn't try to render it.
        return None

    # Token T1 — report_ok is tri-state (True/False/None); preserve it
    # as-is rather than coercing through _to_bool (which would turn None
    # into False and make a never-reported tenant look like a failure).
    _report_ok = bal.get("report_ok")
    if _report_ok is not None and not isinstance(_report_ok, bool):
        _report_ok = bool(_report_ok)

    return {
        "remaining": _to_int(bal.get("remaining")),
        "unlimited": _to_bool(bal.get("unlimited")),
        "monthly_used": _to_int(bal.get("monthly_used")),
        "at": _to_str_or_none(bal.get("at")),
        "report_ok": _report_ok,
        "report_error": _to_str_or_none(bal.get("report_error")),
    }


# How old the cached Chalyb balance may be before an SSE connect kicks a
# background re-fetch. Admin grants, pack purchases and monthly resets all
# happen on the chalyb side without telling us, so a cookie re-entry that
# never passes through /auth/sso would otherwise show last session's number.
_BALANCE_CACHE_TTL_S = 300.0
# Per-tenant monotonic timestamp of the last refresh attempt — throttles the
# fetch to once per TTL window even when several tabs open streams at once.
_balance_refresh_last_attempt: dict[str, float] = {}
# Strong refs so fire-and-forget tasks aren't GC'd mid-flight.
_balance_refresh_tasks: set[asyncio.Task] = set()


def _maybe_refresh_stale_balance(
    db: Database, *, tenant_id: str, balance: object
) -> None:
    """Kick a background Chalyb balance fetch when the cache is stale.

    Called once per balance-stream connect (≈ once per full page load).
    The fresh number lands via TenantsRepo.set_balance_cache → balance_bus,
    i.e. through the very SSE stream whose connect triggered the fetch.
    Best-effort: never raises, never blocks the stream.
    """
    import datetime as _dt
    import time as _time

    now = _time.monotonic()
    last = _balance_refresh_last_attempt.get(tenant_id)
    if last is not None and (now - last) < _BALANCE_CACHE_TTL_S:
        return

    stale = True
    at = balance.get("at") if isinstance(balance, dict) else None
    if isinstance(at, str) and at:
        try:
            cached_at = _dt.datetime.fromisoformat(at)
            if cached_at.tzinfo is None:
                cached_at = cached_at.replace(tzinfo=_dt.UTC)
            age_s = (_dt.datetime.now(_dt.UTC) - cached_at).total_seconds()
            stale = age_s >= _BALANCE_CACHE_TTL_S
        except ValueError:
            stale = True
    if not stale:
        return

    _balance_refresh_last_attempt[tenant_id] = now
    try:
        from chalybclip.integrations.chalyb.balance import fetch_balance_now

        task = asyncio.create_task(fetch_balance_now(db, tenant_id=tenant_id))
        _balance_refresh_tasks.add(task)
        task.add_done_callback(_balance_refresh_tasks.discard)
    except Exception:  # refresh is best-effort
        pass


@router.get("/_balance/stream", include_in_schema=False)
async def balance_stream(request: Request) -> Response:
    """SSE push for the token-balance chip — replaces the old 30s HTMX poll.

    The browser opens ONE EventSource here (see base.html). The server emits a
    `balance` event only when TenantsRepo.set_balance_cache() actually changes
    the cached numbers (published via balance_bus). The chip then re-fetches
    /_balance/chip once — so a request fires on change, not on a timer.

    No tenant_binder (same as /_balance/chip): the middleware already populated
    request.state; the binder would 401-bounce and break the stream. Idle
    connections cost a ~25s keepalive comment — no DB, no render.
    """
    from chalybclip.events.balance_bus import balance_bus

    tenant_id = getattr(request.state, "tenant_id", None)

    # Stale-cache repair: the chip renders from the cached columns, but the
    # ledger lives on chalyb. One throttled background fetch per page load
    # keeps the two in agreement; the result is pushed down this stream.
    if tenant_id:
        _maybe_refresh_stale_balance(
            request.app.state.db,
            tenant_id=tenant_id,
            balance=getattr(request.state, "token_balance", None),
        )

    async def gen():
        # Unauthenticated: send one comment and close — EventSource will retry,
        # and once the session cookie lands the reconnect binds a tenant.
        if not tenant_id:
            yield ": no-session\n\n"
            return
        q = balance_bus.subscribe(tenant_id)
        try:
            yield ": connected\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=25.0)
                    yield f"event: balance\ndata: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"  # keepalive through proxies (Railway)
        finally:
            balance_bus.unsubscribe(tenant_id, q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering so events flush
        },
    )


@router.get("/_balance/refresh", include_in_schema=False)
async def refresh_balance(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Force-fetch the current Chalyb token balance + update the local cache.

    Triggered by clicking the balance chip in the nav. Useful when:
      - The tenant just signed in and hasn't run any LLM calls yet (cache is
        empty so the chip shows '— tokens'.
      - The operator wants to confirm the cross-engine balance reflects a
        recent top-up purchase that happened on another device.

    Behavior:
      - Success → 303 back to the referer (chip re-renders with fresh data).
      - Failure (env vars unset, tenant has no external_user_id, network
        error, etc.) → 303 to /dashboard/_diag/chalyb so the user sees
        exactly what's broken instead of staring at the same '— tokens'
        chip wondering if the click did anything.
    """
    from chalybclip.integrations.chalyb.balance import fetch_balance_now

    ok = await fetch_balance_now(db, tenant_id=tenant_id)
    if not ok:
        # Tell the user what's actually wrong instead of silently bouncing.
        return RedirectResponse(url="/dashboard/_diag/chalyb", status_code=303)
    # Bounce back to wherever the user clicked from. Falls back to streams
    # list if no Referer (direct hit, curl, etc). Validate against the
    # whitelisted dashboard prefix so this can't be used as an open
    # redirect — an attacker page that links here would otherwise be
    # able to redirect the user back to itself for phishing. Mirrors
    # the safe_redirect pattern used by /connected-accounts/create
    # (lines 5432-5437 below).
    referer = request.headers.get("referer") or ""
    safe_target = referer if referer.startswith("/dashboard/") else "/dashboard/streams"
    return RedirectResponse(url=safe_target, status_code=303)


@router.get("/_diag/usage-probe", include_in_schema=False)
async def usage_probe(
    request: Request,
    kind: str = "transcription.seconds",
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """One-shot diagnostic: POST a tiny TEST usage event to Chalyb's
    /usage endpoint and return its VERBATIM status + response body.

    This is the fastest way to read WHY a usage report 400s without
    digging Railway logs or waiting for a pipeline run: open
    /dashboard/_diag/usage-probe?kind=transcription.seconds (or
    ?kind=llm.tokens) in the browser and read `chalyb_response_body`.

    The event is intentionally tiny (amount=1, cost_usd_micros=1) and
    uses a FIXED source_id, so Chalyb's UNIQUE(engine, source_id)
    idempotency makes repeated probes a no-op after the first success.
    It does NOT touch the report-status cache (the red chip), so probing
    is side-effect-free for the operator-facing state.
    """
    # Operator/admin diagnostic — never expose to creators (it surfaces
    # raw cost + upstream internals). 404 to keep it invisible.
    if not getattr(request.state, "is_admin", False):
        raise HTTPException(status_code=404, detail="not found")

    import datetime as _dt

    import httpx as _httpx

    from chalybclip.settings import get_settings

    settings = get_settings()
    base = (settings.chalyb_base_url or "").rstrip("/")
    token = settings.chalyb_admin_token or ""
    tenant = await TenantsRepo(db).get(tenant_id)
    ext = tenant.external_user_id if tenant else None

    if not base or not token:
        return JSONResponse(
            {"error": "CHALYB_BASE_URL or CHALYB_ADMIN_TOKEN not set"},
            status_code=503,
        )
    if not ext:
        return JSONResponse(
            {"error": "this tenant has no external_user_id (not linked to Chalyb)"},
            status_code=409,
        )

    event = {
        "provider": "assemblyai" if kind == "transcription.seconds" else "anthropic",
        "kind": kind,
        "amount": 1,
        "cost_usd_micros": 1,
        "source_id": f"probe_{kind}",
        "occurred_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "operation": "diag_probe",
    }
    payload = {"external_user_id": ext, "events": [event]}
    url = f"{base}/api/engines/chalybclip/usage"

    try:
        async with _httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url, json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            {"sent_to": url, "sent_payload": payload,
             "error": f"{type(e).__name__}: {e}"},
            status_code=502,
        )

    # A genuine 2xx is a real successful POST to /usage — so it's an
    # honest health confirmation. Clear any stale failure status (the red
    # chip) on success, so a fixed integration doesn't keep nagging until
    # the next pipeline run. We ONLY record SUCCESS here, never failure: a
    # probe-specific hiccup must not falsely red the chip.
    if resp.status_code < 400:
        try:
            from chalybclip.tenancy import bound_tenant as _bt
            with _bt(tenant_id):
                await TenantsRepo(db).set_usage_report_status(
                    tenant_id, ok=True,
                    at_iso=_dt.datetime.now(_dt.UTC).isoformat(),
                )
        except Exception:  # noqa: BLE001 — best-effort
            pass

    return JSONResponse({
        "sent_to": url,
        "sent_payload": payload,
        "http_status": resp.status_code,
        "chalyb_response_body": (resp.text or "")[:2000],
        "verdict": (
            "OK — Chalyb accepts this payload (cleared the sync warning)"
            if resp.status_code < 400
            else "REJECTED — see chalyb_response_body for the exact reason; "
                 "fix the /usage validation on the Chalyb side"
        ),
    })


@router.get("/_diag/chalyb", response_class=HTMLResponse, include_in_schema=False)
async def diag_chalyb(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Diagnostic page — surfaces every state bit that the balance pipeline
    depends on, rendered as a readable dashboard panel so we don't need to
    grep Railway logs to debug a missing token chip.

    Defensive: each fetch is wrapped in its own try/except so a broken
    upstream (e.g. CHALYB_BASE_URL pointing at a 404) shows up as an
    inline error on the panel rather than a generic 500. The whole route
    catches at the outer level too — the response should ALWAYS be HTML
    even if the integration is in a bad state.

    The env-var section shows the LITERAL value of CHALYB_BASE_URL (a URL
    is safe to display) so you can spot typos or whitespace in the Railway
    env var without copying the deploy logs. Token values stay
    presence-only (booleans) — never echo a secret back to the browser."""
    # Operator/admin diagnostic — never expose to creators. 404 (not 403)
    # so the page's existence stays invisible to non-admins.
    if not getattr(request.state, "is_admin", False):
        raise HTTPException(status_code=404, detail="not found")

    import traceback

    from chalybclip.integrations.chalyb.balance import (
        fetch_balance_now_verbose,
    )
    from chalybclip.settings import get_settings

    # Defaults so we can always render SOMETHING.
    base_url: str | None = None
    has_admin_token = False
    has_sso_secret = False
    external_user_id: str | None = None
    tenant_tier: str | None = None
    tenant_status: str | None = None
    cached_at: str | None = None
    cached_remaining: int | None = None
    cached_unlimited: int | None = None
    live_ok = False
    diag_error: str | None = None

    try:
        settings = get_settings()
        # Show the URL literal so the user can spot misspellings / trailing
        # whitespace / wrong port. Pydantic auto-strips outer whitespace,
        # so what shows up here is what the runtime actually uses.
        base_url = settings.chalyb_base_url or None
        has_admin_token = bool(settings.chalyb_admin_token)
        has_sso_secret = bool(settings.chalyb_sso_secret)
    except Exception as e:  # noqa: BLE001
        diag_error = f"settings load failed: {e!r}"

    # Token T1 — the OUTBOUND usage-report status. Distinct from the live
    # ping below (which tests the balance GET): this is whether the last
    # /usage POST succeeded. The red "sync ⚠" chip is driven by exactly
    # these fields, so surfacing them here answers "why is the chip red".
    report_ok: int | None = None
    report_error: str | None = None
    report_at: str | None = None

    try:
        tenant = await TenantsRepo(db).get(tenant_id)
        if tenant is not None:
            external_user_id = tenant.external_user_id
            tenant_tier = tenant.tier
            tenant_status = tenant.status
            cached_remaining = tenant.cached_balance_remaining
            cached_unlimited = tenant.cached_balance_unlimited
            cached_at = tenant.cached_balance_at
            report_ok = tenant.last_usage_report_ok
            report_error = tenant.last_usage_report_error
            report_at = tenant.last_usage_report_at
    except Exception as e:  # noqa: BLE001
        diag_error = (diag_error or "") + f" · tenant lookup failed: {e!r}"

    # Live ping. Returns a DiagPingResult with the full HTTP response info
    # (status code, body excerpt, exception type) so the diag page can
    # render the exact failure mode instead of just "ping failed". Updates
    # the cache as a side-effect on 'ok' so this page can ALSO function
    # as a manual refresh button.
    ping = None
    try:
        ping = await fetch_balance_now_verbose(db, tenant_id=tenant_id)
    except Exception as e:  # noqa: BLE001
        diag_error = (diag_error or "") + f" · ping crashed: {e!r}"
    live_ok = ping is not None and ping.outcome == "ok"

    # Re-read tenant after the ping so the rendered cache numbers reflect
    # any update that ping just produced (if it did).
    try:
        tenant_after = await TenantsRepo(db).get(tenant_id)
        if tenant_after is not None:
            external_user_id = tenant_after.external_user_id
            cached_remaining = tenant_after.cached_balance_remaining
            cached_unlimited = tenant_after.cached_balance_unlimited
            cached_at = tenant_after.cached_balance_at
    except Exception as e:  # noqa: BLE001
        diag_error = (diag_error or "") + f" · post-ping read failed: {e!r}"

    interpret = _interpret_diag(
        base_url=base_url,
        has_admin_token=has_admin_token,
        external_user_id=external_user_id,
        live_ok=live_ok,
    )

    try:
        return templates.TemplateResponse(
            request,
            "_diag_chalyb.html",
            {
                "tenant_id": tenant_id,
                "base_url": base_url,
                "has_admin_token": has_admin_token,
                "has_sso_secret": has_sso_secret,
                "external_user_id": external_user_id,
                "tenant_tier": tenant_tier,
                "tenant_status": tenant_status,
                "cached_remaining": cached_remaining,
                "cached_unlimited": cached_unlimited,
                "cached_at": cached_at,
                "report_ok": report_ok,
                "report_error": report_error,
                "report_at": report_at,
                "live_ok": live_ok,
                "ping": ping,
                "interpret": interpret,
                "diag_error": diag_error,
            },
        )
    except Exception as e:  # noqa: BLE001
        # Absolute last-resort: if the template itself crashes, emit a
        # plain-text response with the traceback so the operator still
        # sees SOMETHING useful instead of FastAPI's 500 page.
        tb = traceback.format_exc()
        return HTMLResponse(
            content=(
                "<pre style='padding:24px;font-family:monospace;color:#e54;'>"
                "diag template crashed:\n\n"
                f"{tb}\n\n"
                f"diag_error: {diag_error}</pre>"
            ),
            status_code=500,
        )


def _interpret_diag(
    *,
    base_url: str | None,
    has_admin_token: bool,
    external_user_id: str | None,
    live_ok: bool,
) -> dict[str, str]:
    """Translate the raw diag bits into one human-readable status + next step.
    Returns a {severity, headline, action} dict so the template can color the
    panel appropriately."""
    if not base_url:
        return {
            "severity": "danger",
            "headline": "CHALYB_BASE_URL no está configurada en Railway",
            "action": (
                "Abre el dashboard de Railway → este servicio → Variables, "
                "agrega CHALYB_BASE_URL=https://chalyb.com, y redeploya. "
                "Sin esto ChalybClip no sabe a dónde reportar usage."
            ),
        }
    if not has_admin_token:
        return {
            "severity": "danger",
            "headline": "CHALYB_ADMIN_TOKEN no está configurada en Railway",
            "action": (
                "Debe coincidir EXACTAMENTE con CHALYBCLIP_ADMIN_TOKEN del lado "
                "Chalyb (Vercel env vars). Si no son iguales, Chalyb "
                "responde 401 a cada reporte de usage."
            ),
        }
    if not external_user_id:
        return {
            "severity": "warn",
            "headline": "Este tenant no está vinculado a un usuario de Chalyb",
            "action": (
                "external_user_id está vacío. Probablemente el tenant se creó "
                "via CLI antes de la integración. Re-provisiona desde Chalyb "
                "(POST /api/admin/tenants) o actualiza la columna en SQLite "
                "manualmente para apuntarlo al user_id del usuario en Supabase."
            ),
        }
    if not live_ok:
        return {
            "severity": "warn",
            "headline": "Las env vars están bien pero el ping a Chalyb falló",
            "action": (
                "▼ Scroll abajo a la sección 'Ping en vivo a Chalyb' — "
                "verás el HTTP status code exacto + el body de respuesta + "
                "una pista específica según el código. Es la primera cosa "
                "que tienes que mirar en esta página. (Causas más comunes: "
                "tokens no coinciden byte-por-byte entre ambos sides, "
                "external_user_id no existe en la tabla profiles de Supabase, "
                "o chalyb.com está caído.)"
            ),
        }
    return {
        "severity": "ok",
        "headline": "Todo en verde — el chip debería estar reportando balance",
        "action": (
            "Si el chip aún muestra '— tokens', haz click una vez más para "
            "forzar el refresh manual. Si vuelve aquí, hay un edge case que "
            "no estoy detectando — revisa Railway logs."
        ),
    }


def _split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


async def _merged_personas(db: Database) -> list[object]:
    """Return personas the user can pick from: DB rows union YAML defaults.

    The pipeline only persists a persona to the DB during the variants step
    (so a freshly-uploaded stream whose pipeline hasn't reached step 5 yet
    will see an empty PersonasRepo, even though `config/personas.yaml`
    defines them). Surface both sources here so the dashboard never shows
    an empty dropdown when the YAML file has options. DB rows win on id
    conflict — they reflect any in-dashboard edits.
    """
    from chalybclip.variants import load_personas

    db_personas = await PersonasRepo(db).list_for_tenant()
    seen_ids = {p.id for p in db_personas}
    out: list[object] = list(db_personas)
    try:
        yaml_personas = load_personas()
    except Exception:
        yaml_personas = {}
    for pid, p in yaml_personas.items():
        if pid in seen_ids:
            continue
        # Pico's <select> just needs `.id`, `.name`, `.primary_language`. The
        # YAML Persona has all three with the same names — pass-through is fine.
        out.append(p)
    return out


# ---------- Logout ----------
#
# Slice O.23 — Login removed entirely. chalyb is the only gatekeeper.
# No GET /login (the page is gone), no POST /login (no token form).
# Access in: GET /auth/sso?token=<jwt-from-chalyb> sets a session
# cookie + redirects to /dashboard/start. Anyone hitting any
# /dashboard/* page without that cookie gets bounced to chalyb's
# login URL by the auth middleware. Roles/tiers come straight from
# the SSO token's `tier` claim, synced into the tenant row on each
# login (see chalyb.sso_finalize).


@router.post("/logout")
async def logout() -> Response:
    """Clear our session cookie + bounce to chalyb. No in-house login
    page exists anymore for the redirect to land on."""
    from chalybclip.settings import get_settings
    target = (get_settings().chalyb_login_url or "https://chalyb.com/login").strip()
    response = RedirectResponse(url=target, status_code=303)
    response.delete_cookie(_COOKIE_NAME)
    return response


# ---------- Diarize health (admin) ----------
#
# Slice O.38 — operator-facing diagnostic. The pipeline frequently marks
# diarization as "skipped" with a friendly note ("speaker labels off —
# pyannote not available on this server"), which is correct UX for an
# end-user but useless for the admin trying to fix the underlying cause.
# This route probes every fail-mode in process and reports the raw
# truth: HF_TOKEN presence, pyannote / torchaudio / speechbrain
# importability + versions, the DiarizationConfig the pipeline will use,
# and (when --probe-load=1 is passed) attempts the real pyannote Pipeline
# load — the same call the worker subprocess runs. Admin-gated.


@router.get("/_health/diarize", response_class=HTMLResponse, include_in_schema=False)
async def diarize_health(
    request: Request,
    probe_load: int = 0,
) -> Response:
    """Admin-only diarize diagnostic. Surfaces real, untranslated errors."""
    if not getattr(request.state, "is_admin", False):
        raise HTTPException(status_code=404, detail="not found")

    import importlib
    import os as _os

    from chalybclip.config import load_config as _load_pipeline_config

    def _probe_import(modname: str) -> dict[str, object]:
        try:
            mod = importlib.import_module(modname)
            return {
                "ok": True,
                "version": getattr(mod, "__version__", "?"),
                "path": getattr(mod, "__file__", "?"),
                "error": None,
            }
        except Exception as e:  # noqa: BLE001 — we want every failure
            return {
                "ok": False,
                "version": None,
                "path": None,
                "error": f"{type(e).__name__}: {e}",
            }

    pyannote = _probe_import("pyannote.audio")
    torchaudio = _probe_import("torchaudio")
    torch = _probe_import("torch")
    speechbrain = _probe_import("speechbrain")

    # torchaudio.AudioMetaData is the specific attribute pyannote 3.3.x
    # consults at import time — version skew flips this to False even
    # when both packages import successfully.
    torchaudio_audiometadata = False
    if torchaudio["ok"]:
        try:
            import torchaudio as _ta
            torchaudio_audiometadata = hasattr(_ta, "AudioMetaData")
        except Exception:  # noqa: BLE001
            torchaudio_audiometadata = False

    hf_token = _os.environ.get("HF_TOKEN", "").strip()
    hf_present = bool(hf_token)
    hf_prefix = hf_token[:6] if hf_token else ""

    cfg = _load_pipeline_config()
    diarize_cfg = {
        "enabled": cfg.diarization.enabled,
        "model": cfg.diarization.model,
        "device": cfg.diarization.device,
    }

    pipeline_load: dict[str, object] | None = None
    if probe_load == 1 and pyannote["ok"] and hf_present:
        try:
            from pyannote.audio import Pipeline  # type: ignore[import-not-found]
            pl = Pipeline.from_pretrained(
                diarize_cfg["model"], use_auth_token=hf_token
            )
            pipeline_load = {
                "ok": True,
                "summary": str(type(pl).__name__),
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            pipeline_load = {
                "ok": False,
                "summary": None,
                "error": f"{type(e).__name__}: {e}",
            }

    ctx = {
        "pyannote": pyannote,
        "torchaudio": torchaudio,
        "torchaudio_audiometadata": torchaudio_audiometadata,
        "torch": torch,
        "speechbrain": speechbrain,
        "hf_present": hf_present,
        "hf_prefix": hf_prefix,
        "diarize_cfg": diarize_cfg,
        "pipeline_load": pipeline_load,
        "probe_load_requested": bool(probe_load),
    }
    return templates.TemplateResponse(request, "diarize_health.html", ctx)


# ---------- Pipeline queue (admin) ----------
#
# Slice O.43 — operator visibility into "how saturated is the box?".
# Walks the events log across all tenants, pairs step.started with the
# corresponding step.completed by stream_id + step name, and surfaces
# the orphans as "currently running steps". Long-running outliers (a
# step open for > 30 min) flag separately so the operator sees if the
# CPU Whisper is choking.
#
# Why this is the right signal: streams don't have an explicit
# "queued" state today — uploads kick off pipelines in FastAPI's
# BackgroundTasks, which run as fast as the box can serve them. The
# event log is the only place that tells us "step X started at T,
# never reported done". With 10 users and long VODs we'll see the
# transcribe step pile up — that's the trigger for the Modal move.


@router.get("/_health/queue", response_class=HTMLResponse, include_in_schema=False)
async def queue_health(
    request: Request,
    db: Database = Depends(get_db),
) -> Response:
    """Admin pipeline-queue diagnostic. Surfaces all currently-running
    steps across every tenant + the longest waiters."""
    if not getattr(request.state, "is_admin", False):
        raise HTTPException(status_code=404, detail="not found")

    import datetime as _dt

    # Pull the last 7 days of step events from the events log (no
    # tenant filter — admin view). 7d window is generous: anything
    # older than that is almost certainly a crashed pipeline whose
    # next manual re-run will refresh the events.
    # `ts` is stored as an ISO-8601 UTC string, so a lexicographic >=
    # against a Python-computed cutoff is correct and backend-agnostic
    # (avoids SQLite-only `datetime('now', '-7 days')`, which Postgres
    # doesn't have).
    cutoff = (_dt.datetime.now(_dt.UTC) - _dt.timedelta(days=7)).isoformat()
    conn = await db.connect()
    cur = await conn.execute(
        "SELECT tenant_id, type, payload_json, ts FROM events "
        "WHERE type IN ('pipeline.step.started', 'pipeline.step.completed', "
        "'pipeline.step.failed') "
        "AND ts >= ? "
        "ORDER BY ts ASC",
        (cutoff,),
    )
    rows = await cur.fetchall()

    # Build (stream_id, step) → list of (event_type, ts) so we can
    # pair started with completed/failed. Whichever wins last sets the
    # final state. If only a started is seen, the step is open.
    open_steps: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        d = dict(row)
        try:
            payload = json.loads(d.get("payload_json") or "{}")
        except Exception:  # noqa: BLE001
            payload = {}
        stream_id = payload.get("stream_id")
        step = payload.get("step")
        if not stream_id or not step:
            continue
        key = (str(stream_id), str(step))
        if d["type"] == "pipeline.step.started":
            open_steps[key] = {
                "tenant_id": d["tenant_id"],
                "stream_id": stream_id,
                "step": step,
                "started_at": d["ts"],
            }
        else:
            # completed or failed — close the open step.
            open_steps.pop(key, None)

    # Sort longest-running first.
    now = _dt.datetime.now(_dt.UTC)
    running: list[dict[str, object]] = []
    for info in open_steps.values():
        try:
            started = _dt.datetime.fromisoformat(str(info["started_at"]))
            if started.tzinfo is None:
                started = started.replace(tzinfo=_dt.UTC)
            elapsed_s = (now - started).total_seconds()
        except Exception:  # noqa: BLE001
            elapsed_s = 0.0
        running.append({**info, "elapsed_s": elapsed_s})
    running.sort(key=lambda r: r["elapsed_s"], reverse=True)

    # Per-step counts.
    step_counts: dict[str, int] = {}
    for r in running:
        s = str(r["step"])
        step_counts[s] = step_counts.get(s, 0) + 1

    # Health bucket
    bucket = "ok"
    if len(running) >= 5 or any(r["elapsed_s"] > 1800 for r in running):
        bucket = "warn"
    if len(running) >= 10 or any(r["elapsed_s"] > 3600 for r in running):
        bucket = "danger"

    return templates.TemplateResponse(
        request,
        "queue_health.html",
        {
            "running": running,
            "step_counts": step_counts,
            "total_running": len(running),
            "longest_running_s": running[0]["elapsed_s"] if running else 0,
            "bucket": bucket,
        },
    )


@router.get("/_health/logs", response_class=HTMLResponse, include_in_schema=False)
async def logs_health(
    request: Request,
    level: str = "all",
    q: str = "",
) -> Response:
    """Admin log tracker. Recent WARNING+ records from the in-process
    ring buffer (structlog + stdlib capture — see chalybclip/logbuffer.py),
    filterable by level and free-text. Same operator-only tier as
    /_health/queue: non-admins get a 404, not a 403, so the page's
    existence isn't advertised."""
    if not getattr(request.state, "is_admin", False):
        raise HTTPException(status_code=404, detail="not found")

    from chalybclip.logbuffer import get_log_buffer

    buffer = get_log_buffer()
    entries = buffer.snapshot()

    if level == "error":
        entries = [e for e in entries if e.level in ("error", "critical")]
    elif level == "warning":
        entries = [e for e in entries if e.level == "warning"]

    needle = q.strip().lower()
    if needle:
        entries = [
            e
            for e in entries
            if needle in e.event.lower()
            or needle in e.logger.lower()
            or any(
                needle in k.lower() or needle in v.lower()
                for k, v in e.fields.items()
            )
        ]

    counts = buffer.counts()
    # Header chip: errors in the last hour → danger; any errors in 24h
    # or warnings in the last hour → warn; else ok.
    if counts["errors_1h"] > 0:
        bucket = "danger"
    elif counts["errors_24h"] > 0 or counts["warnings_1h"] > 0:
        bucket = "warn"
    else:
        bucket = "ok"

    return templates.TemplateResponse(
        request,
        "logs_health.html",
        {
            "entries": entries[:200],
            "shown": min(len(entries), 200),
            "matched": len(entries),
            "counts": counts,
            "bucket": bucket,
            "level": level,
            "q": q,
            "buffer_started_at": buffer.started_at,
        },
    )


# ---------- Streams ----------


@router.get("/streams", response_class=HTMLResponse)
async def streams_list(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    from chalybclip.ingest import is_ffmpeg_available
    from chalybclip.jobs.active import active_stream_ids

    streams = await StreamsRepo(db).list_for_tenant()
    personas = await _merged_personas(db)
    return templates.TemplateResponse(
        request,
        "streams_list.html",
        {
            "tenant_id": tenant_id,
            "streams": streams,
            "personas": personas,
            "ffmpeg_ok": is_ffmpeg_available(),
            # Streams with a run queued/executing right now — the status
            # column lags (stays 'ingested'/'failed' until the run ends),
            # so the Estado chip keys on this to show 'procesando' live.
            "active_ids": active_stream_ids(),
        },
    )


@router.post(
    "/streams",
    dependencies=[Depends(require_full_scope), Depends(require_active_tenant)],
)
async def streams_create(
    request: Request,
    background_tasks: BackgroundTasks,
    vod_url: str = Form(...),
    persona_id: str = Form(...),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    from chalybclip.db.adapters import stream_to_row
    from chalybclip.ingest import ingest_vod
    from chalybclip.settings import get_settings

    output_dir = Path(get_settings().default_output_dir)
    try:
        stream = await ingest_vod(
            vod_url=vod_url, tenant_id=tenant_id, output_dir=output_dir
        )
    except ChalybClipError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    row = await StreamsRepo(db).upsert(stream_to_row(stream))
    await EventsRepo(db).emit(type="stream.created", payload={"stream_id": row.id})
    dispatcher = request.app.state.job_dispatcher
    await dispatcher.dispatch_pipeline(
        PipelineKickoff(
            tenant_id=tenant_id,
            stream=stream,
            persona_id=persona_id,
            output_dir=output_dir,
        ),
        background_tasks=background_tasks,
    )
    return RedirectResponse(url=f"/dashboard/streams/{row.id}", status_code=303)


@router.post(
    "/streams/upload",
    dependencies=[Depends(require_full_scope), Depends(require_active_tenant)],
)
async def streams_upload(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    persona_id: str = Form(...),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Ingest an operator-uploaded video file — mirrors the URL-job flow.

    Previously this endpoint blocked the request on:
      1. Receiving the upload bytes (unavoidable; HTTP request body)
      2. ingest_uploaded() — file move + audio extract + ffprobe
         (~30-60 s for a 1-hour video on Railway CPU)
      3. Stream row upsert + event emit
      4. Pipeline dispatch
    …then redirected. For a 500 MB upload the operator stared at a
    spinner for minutes before they could see the live progress page.

    Now (matches /dashboard/url-jobs):
      1. Receive bytes (unavoidable)
      2. Mint stream_id, insert placeholder StreamRow (status="pending",
         duration_s=0, canonical paths the runner will write to)
      3. Emit stream.created
      4. Schedule `upload_pipeline_runner` as a background task —
         audio extract + transcribe + detect + cut all happen there
      5. 303 to /dashboard/streams/{id} immediately

    The live progress page reads the same event surface every other
    source uses (stream.download.* + stream.audio_extracted +
    pipeline.step.* + clip.cut.substep).
    """
    from chalybclip.api._pipeline import upload_pipeline_runner
    from chalybclip.api.routers.streams import _stash_upload_to_tmp
    from chalybclip.db.models import StreamRow
    from chalybclip.ids import new_id
    from chalybclip.ingest import is_ffmpeg_available
    from chalybclip.settings import get_settings

    if not is_ffmpeg_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "ffmpeg is not installed on the server. On Windows: "
                "`winget install --id=Gyan.FFmpeg -e` then reopen PowerShell "
                "and restart the dashboard."
            ),
        )

    output_dir = Path(get_settings().default_output_dir)
    tmp_path = await _stash_upload_to_tmp(file, output_dir)

    # Mint stream_id up-front so the placeholder row + eventual
    # stream.json + the redirect URL all agree.
    stream_id = new_id("str")
    stream_dir = output_dir / stream_id
    source_dir = stream_dir / "source"
    placeholder_video = source_dir / "video.mp4"
    placeholder_audio = source_dir / "audio.wav"
    pseudo_url = f"upload://{file.filename or 'upload.mp4'}"

    placeholder = StreamRow(
        id=stream_id,
        tenant_id=tenant_id,
        vod_url=pseudo_url,
        platform="upload",
        title=file.filename,
        channel=None,
        duration_s=0.0,
        source_video_path=str(placeholder_video),
        source_audio_path=str(placeholder_audio),
        status="pending",
        created_at=_now_iso(),
    )
    await StreamsRepo(db).upsert(placeholder)
    await EventsRepo(db).emit(
        type="stream.created",
        payload={
            "stream_id": stream_id,
            "vod_url": pseudo_url,
            "platform": "upload",
            "duration_s": 0.0,
            # `kind=upload_job` flags this row as the new background-
            # ingest upload path so the dashboard can differentiate it
            # from legacy inline-ingest uploads in event history.
            "kind": "upload_job",
        },
    )

    # Hand off to the background runner. Audio extract + transcribe +
    # detect + cut all happen there; the operator hits the progress
    # page instantly.
    background_tasks.add_task(
        upload_pipeline_runner,
        tenant_id=tenant_id,
        stream_id=stream_id,
        persona_id=persona_id,
        tmp_path=tmp_path,
        output_dir=output_dir,
        title=file.filename,
    )
    return RedirectResponse(
        url=f"/dashboard/streams/{stream_id}", status_code=303,
    )


# ---- Task 3 — Drop-a-URL self-serve ingest ----
#
# A single-field URL entry point that runs the FULL pipeline (ingest →
# transcribe → detect → cut) as a background task. The existing
# `/dashboard/streams` form blocks the request on ingest_vod (which can
# take 10+ minutes on long Twitch / Kick VODs) and requires the operator
# to pick a persona by hand. Task 3 changes both:
#
#  - persona is auto-resolved to the tenant's first persona (alphabetical
#    by created_at). If none exist the form shows a friendly "make a
#    persona first" alert and disables submit.
#  - ingest runs IN BACKGROUND. The endpoint mints the stream_id up
#    front, inserts a placeholder StreamRow with the URL + platform, and
#    redirects immediately to the existing stream detail page. That page
#    polls `/dashboard/streams/{id}/progress`, which surfaces the new
#    `stream.download.started` / `.completed` / `.audio_extracted` events
#    from Task 2a + the `pipeline.step.*` rows + `clip.cut.substep` rows.
#    Clips list lazily — they appear in the existing stream-detail UI
#    as the cut step finishes each one.
#
# All tenancy / budget / scope guards reuse the dependencies the
# existing /dashboard/streams flow uses (`require_full_scope`,
# `require_active_tenant`). No new bypass.


@router.get("/start", response_class=HTMLResponse)
async def start_page(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Unified entry-point page — surface every ingest method (paste URL,
    upload, Drive watch, live RTMP) as cards on one screen so operators
    aren't hunting through nav menus to start a job.

    Persona is auto-resolved to the tenant's first persona; per-run
    override stays on the legacy /dashboard/streams advanced form for
    operators who need it.
    """
    from chalybclip.ingest import is_ffmpeg_available
    from chalybclip.settings import get_settings

    personas = await PersonasRepo(db).list_for_tenant()
    default_persona = personas[0] if personas else None
    s = get_settings()
    live_configured = bool((getattr(s, "live_rtmp_base_url", "") or "").strip())
    # Cookie auth for Kick / age-gated YouTube. When the operator has set
    # either knob (cookies.txt path or browser pass-through), the setup hint
    # under the URL box is just noise — suppress it.
    cookies_configured = bool(
        (getattr(s, "cookies_file", "") or "").strip()
        or (getattr(s, "cookies_from_browser", "") or "").strip()
    )

    return templates.TemplateResponse(
        request,
        "start_page.html",
        {
            "personas": personas,
            "default_persona": default_persona,
            "ffmpeg_ok": is_ffmpeg_available(),
            "live_configured": live_configured,
            "cookies_configured": cookies_configured,
            # Real RTMP server URL for the live ingest panel's "Server" row.
            # The per-tenant stream key is still TODO(ingest) — the template
            # shows a masked placeholder until that endpoint is wired.
            "live_rtmp_base_url": (getattr(s, "live_rtmp_base_url", "") or "").strip(),
        },
    )


@router.get("/url-jobs/new", response_class=HTMLResponse)
async def url_job_form(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Backward-compat — bookmarked /dashboard/url-jobs/new redirects to
    the unified /dashboard/start page (Task UX 9). The old single-field
    template has been deleted."""
    return RedirectResponse(url="/dashboard/start", status_code=303)


@router.post(
    "/url-jobs",
    dependencies=[Depends(require_full_scope), Depends(require_active_tenant)],
)
async def url_job_create(
    request: Request,
    background_tasks: BackgroundTasks,
    vod_url: str = Form(...),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Create a URL ingest job (Task 3a).

    The full pipeline (ingest → transcribe → detect → cut) runs as a
    background task. The response redirects to the existing stream detail
    page which polls progress + lists clips as they finish.
    """
    from chalybclip.db.models import StreamRow
    from chalybclip.ids import new_id
    from chalybclip.ingest import detect_platform
    from chalybclip.settings import get_settings

    # Backend URL validation — mirrors the client-side URL check on the
    # /dashboard/start page so a curl or scripted submission gets the same
    # answer as the form.
    clean = (vod_url or "").strip()
    platform = detect_platform(clean)
    if not clean or platform == "unknown":
        raise HTTPException(
            status_code=400,
            detail=(
                "Only kick.com, twitch.tv, youtube.com, and youtu.be URLs "
                "are supported."
            ),
        )

    # Auto-resolve persona — Task 3 spec: "zero operator involvement".
    # Pick the first persona for this tenant. If none exist the form
    # already shows an alert + disables submit, but a direct POST still
    # needs an actionable error.
    personas = await PersonasRepo(db).list_for_tenant()
    if not personas:
        raise HTTPException(
            status_code=400,
            detail=(
                "No personas configured for this tenant. Create one at "
                "/dashboard/personas first."
            ),
        )
    persona = personas[0]

    output_dir = Path(get_settings().default_output_dir)
    stream_id = new_id("str")
    stream_dir = output_dir / stream_id
    source_dir = stream_dir / "source"
    placeholder_video = source_dir / "video.mp4"
    placeholder_audio = source_dir / "audio.wav"

    # Insert a placeholder StreamRow so the redirect target page renders
    # immediately (with "Downloading…" progress) instead of 404'ing while
    # ingest runs in background. ingest_vod inside process_vod is
    # idempotent on stream_id — it'll write stream.json and the pipeline
    # will UPSERT the StreamRow with real metadata (duration, title, etc)
    # once download lands. The placeholder paths get overwritten with
    # identical values, so the upsert is a no-op for those fields.
    # `status="pending"` flags the row as in-flight for any UI / repo
    # query that wants to differentiate fresh URL jobs from completed
    # ingests. The pipeline's StreamsRepo.upsert overwrites this with
    # "ingested" once download lands.
    placeholder = StreamRow(
        id=stream_id,
        tenant_id=tenant_id,
        vod_url=clean,
        platform=platform,
        title=None,
        channel=None,
        duration_s=0.0,
        source_video_path=str(placeholder_video),
        source_audio_path=str(placeholder_audio),
        status="pending",
        created_at=_now_iso(),
    )
    await StreamsRepo(db).upsert(placeholder)
    await EventsRepo(db).emit(
        type="stream.created",
        payload={
            "stream_id": stream_id,
            "vod_url": clean,
            "platform": platform,
            "duration_s": 0.0,
            # `kind=url_job` lets the dashboard differentiate Task 3
            # entries from the legacy POST /dashboard/streams flow when
            # filtering event history.
            "kind": "url_job",
        },
    )

    # Stub Stream for the kickoff envelope. `default_pipeline_runner`
    # only reads `kickoff.stream.id` + `kickoff.stream.vod_url` — the
    # paths are placeholders that process_vod's ingest will materialize.
    from chalybclip.ingest import Stream as StreamModel
    stub = StreamModel(
        id=stream_id,
        tenant_id=tenant_id,
        vod_url=clean,
        platform=platform,
        duration_s=0.0,
        source_video_path=placeholder_video,
        source_audio_path=placeholder_audio,
    )
    dispatcher = request.app.state.job_dispatcher
    await dispatcher.dispatch_pipeline(
        PipelineKickoff(
            tenant_id=tenant_id,
            stream=stub,
            persona_id=persona.id,
            output_dir=output_dir,
        ),
        background_tasks=background_tasks,
    )

    # Redirect to the existing stream detail page. It already polls
    # /progress (Task 1d substeps + Task 2a download events show up
    # there) and lists clips as they're cut.
    return RedirectResponse(
        url=f"/dashboard/streams/{stream_id}", status_code=303,
    )


def _now_iso() -> str:
    """ISO-8601 UTC timestamp for placeholder stream rows."""
    import datetime as _dt
    return _dt.datetime.now(_dt.UTC).isoformat()


@router.get("/streams/{stream_id}/progress", response_class=HTMLResponse)
async def stream_progress(
    request: Request,
    stream_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """HTMX poll target — renders the live pipeline-progress card.

    Reads `pipeline.step.{start,done,failed}` events for this stream and
    surfaces what's running right now plus what's already finished. The
    parent stream-detail page polls this every 3s while the pipeline is
    in flight.
    """
    from chalybclip.db import EventsRepo

    stream = await StreamsRepo(db).get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="stream not found")

    candidates = await CandidatesRepo(db).list_for_stream(stream_id)
    clips = await ClipsRepo(db).list_for_stream(stream_id)

    # Pull every step event we've written for any stream in this tenant
    # (the events table is per-tenant; we filter by stream_id in the
    # payload). The volume is small — six steps × N streams.
    all_events = await EventsRepo(db).list_for_tenant(limit=500)
    step_events = [
        e
        for e in all_events
        if e.type.startswith("pipeline.step.")
        and e.payload.get("stream_id") == stream_id
    ]

    # Roll up by step name -> latest known status. Step order is fixed.
    # Slice O.47 — `variants` removed from the user-visible list. The
    # step still runs in the pipeline (creates one stub row per clip so
    # publish works) but it's not a creator-facing concept anymore.
    step_order = [
        "ingest",
        "analyze_video",
        "diarize",
        "transcribe",
        "detect",
        "cut",
    ]
    step_state: dict[str, dict[str, object]] = {
        name: {
            "name": name,
            "status": "pending",
            "duration_s": None,
            "elapsed_s": None,
            "error": None,
            # Slice F.7-G — surface "skipped" + a one-line reason so
            # near-zero durations (cached ingest, disabled visual,
            # missing HF_TOKEN diarization) don't render as the
            # confusing "0.0s" that the user thought meant "broken".
            "skipped": False,
            "note": "",
        }
        for name in step_order
    }
    # Walk events oldest -> newest so the latest status wins.
    for ev in sorted(step_events, key=lambda e: e.ts):
        step_name = str(ev.payload.get("step", ""))
        if step_name not in step_state:
            continue
        if ev.type == "pipeline.step.start":
            step_state[step_name]["status"] = "running"
            step_state[step_name]["started_at"] = ev.ts
        elif ev.type == "pipeline.step.done":
            step_state[step_name]["status"] = "done"
            step_state[step_name]["duration_s"] = ev.payload.get("duration_s")
            step_state[step_name]["skipped"] = bool(ev.payload.get("skipped"))
            step_state[step_name]["note"] = str(ev.payload.get("note") or "")
        elif ev.type == "pipeline.step.failed":
            step_state[step_name]["status"] = "failed"
            step_state[step_name]["error"] = ev.payload.get("error")
            step_state[step_name]["duration_s"] = ev.payload.get("duration_s")

    # Compute elapsed-time-so-far for the currently-running step. Gives the
    # user a "yes, this is taking a while" signal instead of an indeterminate
    # spinner that says nothing about progress.
    #
    # If elapsed exceeds ABANDONED_THRESHOLD_S the step is reclassified as
    # 'abandoned' — almost always means the dashboard was restarted while
    # the background task was in-flight (Ctrl+C'd, crashed, etc.). The
    # actual pipeline thread is dead and the run will never finish on its
    # own. Show that explicitly so the user knows to click 'Run pipeline'
    # rather than wait forever on a fake 'running' state.
    #
    # UNLESS a run for this stream is registered in jobs.active — then the
    # run is alive in this process (executing, or queued behind the
    # concurrency semaphore) and the wall clock proves nothing: the
    # pipeline's own per-step ceilings scale with VOD duration
    # (analyze_video gets duration * 4 — about 20h on a 5-hour VOD), so a
    # healthy long run WILL sit in one step past any fixed threshold. Flagging it
    # abandoned contradicts the hero (which trusts the same registry) and
    # offers a Córrelo that would stack a duplicate run on top of the
    # live one. Registry says in flight → keep it 'running', keep polling;
    # the runner's own timeout/failure handling is what ends a stuck step.
    import datetime as _dt

    from chalybclip.jobs.active import active_stream_ids

    run_in_flight = stream_id in active_stream_ids()

    ABANDONED_THRESHOLD_S = 60 * 60  # 1 hour — generous even for hour+ VODs
    now = _dt.datetime.now(_dt.UTC)
    for s in step_state.values():
        if s["status"] == "running" and "started_at" in s:
            try:
                started = _dt.datetime.fromisoformat(str(s["started_at"]))
                elapsed = (now - started).total_seconds()
                s["elapsed_s"] = elapsed
                if elapsed > ABANDONED_THRESHOLD_S and not run_in_flight:
                    s["status"] = "abandoned"
            except ValueError:
                pass

    steps = [step_state[n] for n in step_order]

    # Slice NX.5 — top-level pipeline failure surfacing. If a `pipeline.failed`
    # event exists for this stream, the runner caught an exception OUTSIDE
    # any individual step (typically: bad persona_id, missing config, db
    # open error). The per-step events are all 'pending' in this case, so
    # without this check the UI would spin forever. We pull the most-recent
    # such event and:
    #   1) Flip the first pending step to a synthetic 'failed' state so the
    #      progress row renders red with the error message.
    #   2) Set has_failed=True so the parent template stops polling.
    pipeline_failed_events = [
        e
        for e in all_events
        if e.type == "pipeline.failed" and e.payload.get("stream_id") == stream_id
    ]
    pipeline_failure: dict[str, object] | None = None
    if pipeline_failed_events:
        latest = max(pipeline_failed_events, key=lambda e: e.ts)
        pipeline_failure = {
            "error": str(latest.payload.get("error") or "pipeline aborted"),
            "error_type": str(latest.payload.get("error_type") or "Exception"),
        }
        # Mark the first pending step as failed so the progress card has
        # SOMETHING to point at. Walk in order so the user sees the failure
        # at the earliest stage that didn't get to run.
        for s in steps:
            if s["status"] == "pending":
                s["status"] = "failed"
                s["error"] = pipeline_failure["error"]
                break

    is_running = any(s["status"] == "running" for s in steps) or (
        all(s["status"] == "pending" for s in steps) and pipeline_failure is None
    )
    is_done = all(s["status"] == "done" for s in steps)
    has_failed = any(s["status"] == "failed" for s in steps)
    is_abandoned = any(s["status"] == "abandoned" for s in steps)

    return templates.TemplateResponse(
        request,
        "_stream_progress.html",
        {
            "stream": stream,
            "steps": steps,
            "is_running": is_running,
            "is_done": is_done,
            "has_failed": has_failed,
            "is_abandoned": is_abandoned,
            "pipeline_failure": pipeline_failure,
            "candidate_count": len(candidates),
            "clip_count": len(clips),
        },
    )


# ---- stream_detail view-model (creator-facing redesign) --------------

# Map a detector `reason` + evidence into human, trust-building bullets.
# Creators don't know what "audio_energy 0.182" means — they DO understand
# "Audio energy spike — laughter / crowd reaction".
_REASON_HUMAN: dict[str, str] = {
    "voice": "Voice trigger — a clip phrase was spoken",
    "audio": "Audio energy spike — laughter / crowd / emphasis",
    "chat": "Chat heat — the audience reacted in real time",
    "visual": "Visual moment — scene change / on-screen action",
    "viral": "AI flagged a viral-shaped moment",
}


def _humanize_candidate(candidate: object | None) -> list[str]:
    """Turn a candidate's reason + evidence into a short list of
    human-readable signals for the clip card's 'AI Analysis' block."""
    if candidate is None:
        return []
    out: list[str] = []
    reason = (getattr(candidate, "reason", "") or "").lower()
    out.append(_REASON_HUMAN.get(reason, "AI-detected highlight"))
    ev = getattr(candidate, "evidence", None) or {}
    if isinstance(ev, dict):
        phrase = ev.get("phrase") or ev.get("transcript_snippet")
        if isinstance(phrase, str) and phrase.strip():
            snippet = phrase.strip()
            if len(snippet) > 80:
                snippet = snippet[:77] + "…"
            out.append(f'Said: "{snippet}"')
        speaker = ev.get("speaker_label")
        if isinstance(speaker, str) and speaker.strip():
            out.append(f"Speaker: {speaker.strip()}")
        # An LLM rescore confirmation reads as a strong trust signal.
        if ev.get("rescore"):
            out.append("Confirmed by a second AI pass")
    return out


def _viral_score(clip: object, candidate: object | None) -> int | None:
    """0-100 'viral score' for a clip. Prefers the publishability score
    (set during clip review); falls back to the detector's confidence
    (candidate.score, 0-1) so un-reviewed clips still show a number."""
    pub = getattr(clip, "publishability_score", None)
    if isinstance(pub, int) and pub > 0:
        return pub
    if candidate is not None:
        raw = getattr(candidate, "rescore_score", None)
        if raw is None:
            raw = getattr(candidate, "score", None)
        if isinstance(raw, (int, float)) and raw > 0:
            return max(1, min(100, round(float(raw) * 100)))
    return None


def _perf_label(score: int | None) -> str:
    if score is None:
        return "—"
    if score >= 80:
        return "High"
    if score >= 60:
        return "Medium"
    return "Building"


def _stream_overview(stream: object, clips: list, candidates: list) -> dict:
    """Creator-facing view model: outcome-first summary, per-clip viral
    scores + humanized AI signals, a visual timeline, and a 'best pick'.
    Keeps all the derivation in one place so the template stays dumb."""
    # Derive the outcome state. Trust the CLIPS over the stream.status
    # column: a stream can finish the pipeline (clips on disk) while its
    # status row is still "upload"/"pending" (the status update is best-
    # effort and not always written). If clips exist and we're not
    # actively running or failed, the run is DONE — show the outcome, not
    # a misleading "Analyzing your video…".
    status = (getattr(stream, "status", "") or "").lower()
    has_clips = len(clips) > 0
    # CLIPS win: a finished run has clips on disk even if the status column
    # is stale. Live streams in particular are left at 'processing' by the
    # auto-clip claim, so checking running-states first would show
    # 'Analyzing…' forever. Order: complete (clips/terminal) → failed →
    # running → pending.
    #
    # 'ingested' is NOT terminal: it means the download landed — the
    # pipeline may be queued, running, killed by a redeploy, or never
    # dispatched. With zero clips it must not render "Ya quedó listo · 0
    # clips" (a dead end); 'pending' shows the progress panel, whose
    # abandoned-run detector explains what happened and offers Córrelo.
    if has_clips or status in ("done", "ready"):
        status_kind = "complete"
    elif status == "failed":
        status_kind = "failed"
    elif status in ("running", "processing", "ingesting"):
        status_kind = "running"
    else:
        status_kind = "pending"

    cand_by_id = {getattr(c, "id", None): c for c in candidates}
    duration = float(getattr(stream, "duration_s", 0) or 0) or 1.0

    enriched: list[dict] = []
    for clip in clips:
        cand = cand_by_id.get(getattr(clip, "candidate_id", None))
        score = _viral_score(clip, cand)
        start = float(getattr(clip, "start_s", 0) or 0)
        end = float(getattr(clip, "end_s", 0) or 0)
        enriched.append(
            {
                "clip": clip,
                "candidate": cand,
                "viral_score": score,
                "signals": _humanize_candidate(cand),
                "timeline_left_pct": max(0.0, min(100.0, start / duration * 100)),
                "timeline_width_pct": max(
                    1.5, min(100.0, (end - start) / duration * 100)
                ),
            }
        )

    # Best pick — highest viral score (None scores sort last).
    scored = [e for e in enriched if e["viral_score"] is not None]
    best = max(scored, key=lambda e: e["viral_score"]) if scored else None

    ready = [
        c for c in clips
        if (getattr(c, "status", "") in ("approved", "published"))
    ]
    rejected = [c for c in clips if getattr(c, "status", "") == "rejected"]

    # Time-saved heuristic: manually finding, cutting, captioning +
    # reformatting one short-form clip is ~8 min of editing. Honest
    # order-of-magnitude, not a precise claim.
    time_saved_min = len(clips) * 8

    return {
        "status_kind": status_kind,
        "total_clips": len(clips),
        "ready_count": len(ready),
        "rejected_count": len(rejected),
        "moments_detected": len(candidates),
        "enriched_clips": enriched,
        "best": best,
        "best_score": best["viral_score"] if best else None,
        "perf_label": _perf_label(best["viral_score"] if best else None),
        "time_saved_min": time_saved_min,
        "duration_s": duration,
        # Static for now — could read brand_kit.target_platform later.
        "suggested_platforms": ["TikTok", "Instagram Reels", "YouTube Shorts"],
    }


@router.get("/streams/{stream_id}", response_class=HTMLResponse)
async def stream_detail(
    request: Request,
    stream_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    stream = await StreamsRepo(db).get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="stream not found")
    candidates = await CandidatesRepo(db).list_for_stream(stream_id)
    clips = await ClipsRepo(db).list_for_stream(stream_id)
    personas = await _merged_personas(db)
    overview = _stream_overview(stream, clips, candidates)

    # Right after Córrelo the status COLUMN still says 'ingested' / 'failed'
    # — it only flips at the end of the run — so the page rendered as if
    # nothing had happened and users re-clicked or gave up. The jobs.active
    # registry knows a run for this stream is queued or executing in this
    # process; trust it over the stale column so the analyzing hero + live
    # progress panel render. (Clips-bearing pages stay 'complete': the
    # existing clips are worth showing while a re-run cooks.)
    if overview["status_kind"] in ("pending", "failed"):
        from chalybclip.jobs.active import active_stream_ids

        if stream_id in active_stream_ids():
            overview["status_kind"] = "running"

    # Token T3 — ACTUAL spend for this run (Claude + transcription + any
    # provider, tagged with this stream_id). Drives the estimate-vs-actual
    # panel. Best-effort: a query hiccup shouldn't 500 the page.
    actual_spend = None
    try:
        from chalybclip.tenancy import bound_tenant as _bt
        with _bt(tenant_id):
            actual_spend = await LLMCallsRepo(db).cost_for_stream(stream_id)
    except Exception:  # noqa: BLE001
        actual_spend = None

    # Pre-run estimate. Uses stream duration + a heuristic for clip count
    # to produce an upper-bound token count, then divides by the user's
    # remaining balance to suggest how many more times they could run.
    # See chalybclip/llm/estimator.py for the math + the per-phase
    # multipliers.
    from chalybclip.llm.estimator import (
        estimate_pipeline_run,
        estimate_runs_per_balance,
        format_tokens,
    )

    # We don't know how many personas the user will pick at run time, so
    # estimate for the single-persona case (most common) AND surface what
    # the multiplier looks like when they pick more — the template renders
    # both as "1 persona: ~X tokens · cada persona extra: +Y".
    estimate_single = estimate_pipeline_run(
        duration_seconds=getattr(stream, "duration_s", None),
        persona_count=1,
    )
    # Variant tokens are the only persona-multiplied phase. Extract just
    # that line so the UI can compute "+Y per extra persona".
    variant_tokens_one_persona = next(
        (line.tokens for line in estimate_single.lines if line.phase == "Variantes"),
        0,
    )

    # Balance for the "runs available" hint. The middleware populates
    # request.state.token_balance already; we read it here for the
    # template. Admins (unlimited) get None back.
    token_balance = getattr(request.state, "token_balance", None)
    remaining = None
    if token_balance is not None and not getattr(token_balance, "unlimited", False):
        remaining = getattr(token_balance, "remaining", None)
    runs_available = estimate_runs_per_balance(
        remaining_tokens=remaining,
        duration_seconds=getattr(stream, "duration_s", None),
        persona_count=1,
    )

    # Token T5 — express this run's cost to the CLIENT in TOKENS, never USD.
    # Creators see only what their balance lost; raw USD is internal economics
    # (admin-only). We mirror Chalyb's deduction exactly (MICROS_PER_TOKEN=4,
    # floor 1 per provider) and fold in the per-run base charge so the number
    # here matches the balance chip's drop.
    from math import ceil as _ceil

    from chalybclip.settings import get_settings as _get_settings

    micros_per_token = 4  # MUST match Chalyb getTokenBalance's MICROS_PER_TOKEN
    _base_micros = int(
        getattr(_get_settings(), "pipeline_base_charge_usd_micros", 0) or 0
    )
    base_charge_tokens = _ceil(_base_micros / micros_per_token) if _base_micros > 0 else 0

    # Actual token draw (what the balance actually lost): per provider, LLM
    # deducts its native token count, metered providers deduct cost→tokens;
    # floor 1 each; plus the base charge.
    run_tokens_used: int | None = None
    run_tokens_by_source: list[dict[str, object]] = []
    if actual_spend is not None and actual_spend.total_calls:
        _sum = 0
        for p in actual_spend.by_provider:
            t = p.tokens if p.tokens > 0 else _ceil(p.cost_usd_micros / micros_per_token)
            t = max(1, int(t))
            run_tokens_by_source.append(
                {"label": p.provider, "tokens": t, "tokens_fmt": f"{t:,}"}
            )
            _sum += t
        if base_charge_tokens > 0:
            run_tokens_by_source.append(
                {
                    "label": "procesamiento base",
                    "tokens": base_charge_tokens,
                    "tokens_fmt": f"{base_charge_tokens:,}",
                }
            )
            _sum += base_charge_tokens
        run_tokens_used = _sum

    # Estimated token draw (upper bound, same units): LLM token estimate +
    # transcription cost→tokens + base charge.
    est_tokens_total = (
        int(estimate_single.total_tokens)
        + (
            _ceil(estimate_single.transcription_cost_usd_micros / micros_per_token)
            if estimate_single.transcription_cost_usd_micros > 0
            else 0
        )
        + base_charge_tokens
    )

    # Clips still awaiting approval — drives the "Aprobar todos" bulk button.
    pending_approval_count = sum(
        1 for c in clips if c.status in ("cut", "ready_for_review")
    )

    # The retention loop reclaims the source video once the stream is fully
    # processed (delete-source-on-completion) — by design, clips don't need
    # it anymore. Tell the template so it renders a friendly "liberado" note
    # instead of a broken black <video> that 404s against /source.
    source_available = False
    with contextlib.suppress(OSError):
        source_available = Path(stream.source_video_path).exists()

    return templates.TemplateResponse(
        request,
        "stream_detail.html",
        {
            "stream": stream,
            "candidates": candidates,
            "clips": clips,
            "source_available": source_available,
            "pending_approval_count": pending_approval_count,
            "overview": overview,
            "actual_spend": actual_spend,
            "personas": personas,
            "estimate": estimate_single,
            "estimate_total_fmt": format_tokens(estimate_single.total_tokens),
            "estimate_per_extra_persona_fmt": format_tokens(variant_tokens_one_persona),
            "runs_available": runs_available,
            # Token T5 — client-facing run cost in TOKENS (USD is admin-only).
            "run_tokens_used": run_tokens_used,
            "run_tokens_used_fmt": (
                format_tokens(run_tokens_used) if run_tokens_used else None
            ),
            "run_tokens_by_source": run_tokens_by_source,
            "est_tokens_total": est_tokens_total,
            "est_tokens_total_fmt": format_tokens(est_tokens_total),
            "base_charge_tokens": base_charge_tokens,
            # Only admins see raw USD economics; creators never do.
            "show_usd_cost": bool(getattr(request.state, "is_admin", False)),
            "is_unlimited": (
                token_balance is not None
                and getattr(token_balance, "unlimited", False)
            ),
        },
    )


@router.post(
    "/streams/{stream_id}/rerun",
    dependencies=[Depends(require_full_scope), Depends(require_active_tenant)],
)
async def streams_rerun(
    request: Request,
    stream_id: str,
    background_tasks: BackgroundTasks,
    persona_id: str = Form(...),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Re-trigger the pipeline for an already-ingested stream.

    Useful when (a) the original background task died with a server restart,
    (b) the user uploaded before the step-event tracking landed, or (c) they
    just want to rebuild clips after editing config. Each pipeline step is
    idempotent on its own output, so this is safe to call repeatedly.
    """
    from chalybclip.ingest import load_stream
    from chalybclip.settings import get_settings
    from chalybclip.variants import load_personas

    output_dir = Path(get_settings().default_output_dir)
    stream_row = await StreamsRepo(db).get(stream_id)
    if stream_row is None:
        raise HTTPException(status_code=404, detail="stream not found")

    # If the picked persona is YAML-only, upsert it into the DB now so the
    # variants step's FK constraint to `personas` holds. (The pipeline does
    # this itself at step 5, but we want the persona row to exist before
    # then so the dashboard's persona pages see it consistently.)
    db_personas = {p.id: p for p in await PersonasRepo(db).list_for_tenant()}
    if persona_id not in db_personas:
        try:
            yaml_personas = load_personas()
        except Exception:
            yaml_personas = {}
        if persona_id not in yaml_personas:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"unknown persona {persona_id!r}; not in DB or "
                    f"config/personas.yaml"
                ),
            )
        p = yaml_personas[persona_id]
        await PersonasRepo(db).upsert(
            persona_id=p.id,
            name=p.name,
            primary_language=p.primary_language,
            target_languages=p.target_languages,
            voice_prompt=p.voice_prompt,
            routing_tags=p.routing_tags,
        )
    # Load the on-disk Stream model so PipelineKickoff can carry its full
    # metadata (the in-DB row is a different shape).
    try:
        stream = load_stream(output_dir / stream_id)
    except Exception as e:
        # No on-disk artifacts — the original ingest never completed (e.g. a
        # transient YouTube "confirm you're not a bot" gate failed the
        # download). For a URL source we re-attempt the ingest from the stored
        # VOD URL: build a stub Stream and let the pipeline's first step
        # re-download. The bot-gate is usually transient, so a retry typically
        # lands. Uploads / live have no re-fetchable source, so those still
        # need to be re-added.
        refetchable = bool(stream_row.vod_url) and stream_row.platform not in (
            "upload", "live", "live_ended",
        )
        if not refetchable:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Stream {stream_id!r} has no on-disk artifacts at "
                    f"{output_dir / stream_id} and no re-fetchable source — "
                    f"re-add it to retry."
                ),
            ) from e
        from chalybclip.ingest import Stream as StreamModel

        stream = StreamModel(
            id=stream_id,
            tenant_id=tenant_id,
            vod_url=stream_row.vod_url,
            platform=stream_row.platform,  # type: ignore[arg-type]
            duration_s=stream_row.duration_s or 0.0,
            source_video_path=Path(stream_row.source_video_path),
            source_audio_path=Path(stream_row.source_audio_path),
        )
        # Reset a prior 'failed'/'pending' row so the pipeline's
        # reconcile_metadata flips it back to 'ingested' on a successful
        # re-download (reconcile only advances a row that's still 'pending').
        await StreamsRepo(db).set_status(stream_id, status="pending")

    # Slice F.7-G — pass the persona's primary_language into the runner
    # so Whisper transcribes in the right language. The pipeline default
    # used to be a hardcoded "es" fallback, which silently produced
    # garbage transcripts on English/Portuguese/etc clips. Now: the
    # persona's language wins; if none, faster-whisper auto-detects.
    persona_language: str | None = None
    persona_row = await PersonasRepo(db).get(persona_id)
    if persona_row is not None and persona_row.primary_language:
        persona_language = persona_row.primary_language

    dispatcher = request.app.state.job_dispatcher
    await dispatcher.dispatch_pipeline(
        PipelineKickoff(
            tenant_id=tenant_id,
            stream=stream,
            persona_id=persona_id,
            output_dir=output_dir,
            language=persona_language,
        ),
        background_tasks=background_tasks,
    )
    await EventsRepo(db).emit(
        type="stream.rerun_requested", payload={"stream_id": stream_id}
    )
    return RedirectResponse(url=f"/dashboard/streams/{stream_id}", status_code=303)


# ---------- Clips ----------


@router.get("/clips/{clip_id}", response_class=HTMLResponse)
async def clip_detail(
    request: Request,
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    from chalybclip.branding import (
        caption_style_or_default,
        platform_choices,
        platform_target_choices,
        preset_choices,
        style_choices,
        zones_for_platform,
        resolve_brand_kit_for_candidate,
    )
    from chalybclip.clip import (
        clip_breakdown,
        compute_ai_scores,
        compute_publishability,
        director_lines,
    )
    from chalybclip.db import (
        CandidatesRepo,
        PublishJobsRepo,
        PublishMetricsRepo,
    )

    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    # Slice O.46 — variants picker UI removed. We still read the
    # stub variant row that the pipeline creates per clip so the
    # publish flow (which consumes variant.caption) has something to
    # pass to TikTok / Reels / etc. No more enrichment / emotional
    # labels / retention bars — that was the panel we deleted.
    variants = await VariantsRepo(db).list_for_clip(clip_id)
    accounts = await ConnectedAccountsRepo(db).list_for_tenant()
    valid_transitions = sorted(_VALID_STATUS_TRANSITIONS.get(clip.status, set()))
    breakdown = await clip_breakdown(db, clip_id)
    ai_scores = compute_ai_scores(breakdown)

    # Slice G.6 — auto-correction summary written by the pipeline when the
    # clip was cut (trim + overlay fixes applied while the VOD still existed).
    # The editor only *reports* what was corrected — there's no "fix it"
    # button anymore. None when the clip predates G.6 or had nothing to fix.
    from chalybclip.clip import load_auto_corrections

    auto_corrections = None
    try:
        auto_corrections = load_auto_corrections(Path(clip.path).parent)
    except Exception:  # noqa: BLE001 — a missing/bad sidecar just hides the panel
        auto_corrections = None

    # Slice I.3 — publishability verdict + AI Director narration.
    # The verdict reads BOTH the breakdown AND the operator's overlay
    # config (current banner / hook / caption choices) so it scores
    # what's about to ship, not the raw clip. Safe-zone target falls
    # back to TikTok (the strictest mainstream chrome).
    safe_zone_target = "tiktok"
    if isinstance(clip.overlay_config, dict):
        szp = clip.overlay_config.get("safe_zone_platform")
        if isinstance(szp, str) and szp:
            safe_zone_target = szp
    publishability = compute_publishability(
        breakdown=breakdown,
        overlay_config=(
            clip.overlay_config if isinstance(clip.overlay_config, dict) else {}
        ),
        safe_zone_platform=safe_zone_target,
    )
    director = director_lines(breakdown=breakdown, verdict=publishability)

    # Slice G.2 — cache the verdict on the clip row so the inbox + streams
    # grid can render a status chip without recomputing. Best-effort:
    # a write failure here must not block the editor render.
    if (
        clip.publishability_score != int(publishability.score)
        or clip.publishability_status != publishability.status
    ):
        try:
            await ClipsRepo(db).set_publishability(
                clip_id,
                score=int(publishability.score),
                status=publishability.status,
            )
        except Exception:  # noqa: BLE001 — never block render on cache write
            pass

    # Resolve the brand kit + caption style so the editor can render
    # the live preview against the right defaults when overlay_config
    # is empty (or partially populated).
    speaker_label: str | None = None
    # Slice K.7 — surface trigger-phrase detection on the verdict card.
    # The operator's #1 confusion was "did the AI actually detect the
    # phrase I said?". Pull the matched phrase / snippet / kind out of
    # the candidate evidence so the template can render a "Detected
    # trigger: 'clip it'" line.
    candidate_trigger: dict[str, object] | None = None
    # Slice L.4b — face-aware hook positioning. Derive a coarse
    # "face_zone" hint from the G.3 framing verdict's safe_crop_box so
    # the editor's main hook + captions can dodge the face instead of
    # statically anchoring at 18% (and sometimes covering it). Only
    # three coarse zones — the precision isn't there to justify five —
    # plus None when framing wasn't run on this clip (degrades to the
    # L.4 static defaults).
    face_zone: str | None = None
    if clip.candidate_id:
        for cand in await CandidatesRepo(db).list_for_stream(clip.stream_id):
            if cand.id == clip.candidate_id:
                ev = cand.evidence or {}
                lbl = ev.get("speaker_label")
                if isinstance(lbl, str):
                    speaker_label = lbl
                if isinstance(ev, dict):
                    phrase = ev.get("phrase")
                    snippet = ev.get("transcript_snippet")
                    kind = ev.get("trigger_kind")
                    if isinstance(phrase, str) and phrase:
                        candidate_trigger = {
                            "phrase": phrase,
                            "snippet": snippet if isinstance(snippet, str) else "",
                            "kind": kind if isinstance(kind, str) else "forward",
                            "language": (
                                ev.get("language")
                                if isinstance(ev.get("language"), str)
                                else None
                            ),
                            "confidence": (
                                float(ev.get("confidence"))
                                if isinstance(ev.get("confidence"), int | float)
                                else None
                            ),
                        }
                    # Slice L.4b — extract face_zone from framing.
                    # The framing evidence persists `safe_crop_box`
                    # (the chosen 9:16 crop region within the source).
                    # `crop_box.y` near 0 means the crop is anchored at
                    # the TOP of the source = face appears HIGH in the
                    # 9:16 output. Map to three zones:
                    #   top    → face in upper third → hook conflicts
                    #   mid    → face mid-frame → L.4 default is fine
                    #   lower  → face near bottom → hook is fully clear
                    framing = ev.get("framing")
                    if isinstance(framing, dict):
                        crop = framing.get("safe_crop_box")
                        if isinstance(crop, dict):
                            cy = crop.get("y")
                            if isinstance(cy, int | float):
                                cy_f = float(cy)
                                if cy_f < 0.10:
                                    face_zone = "top"
                                elif cy_f < 0.35:
                                    face_zone = "mid"
                                else:
                                    face_zone = "lower"
                break
    brand_kit = await resolve_brand_kit_for_candidate(
        db, stream_id=clip.stream_id, speaker_label=speaker_label
    )
    caption_style = caption_style_or_default(
        brand_kit.caption_style if brand_kit is not None else None
    )

    # Slice L.6 — viral-intelligence decision engine. Pure function
    # over the breakdown + framing + platform inputs we already have.
    # Outputs:
    #   - intel_lines: ✨ phrases for the verdict card
    #   - caption_position: cascades into the "Auto Viral" picker
    #   - needs_hook / needs_subtitles / secondary_hook_recommended:
    #     drive editor visibility hints
    # The decision runs every request — it's a pure function with no
    # I/O so per-request cost is negligible (microseconds).
    from chalybclip.clip.ai_decisions import decide as _ai_decide

    # Resolve operator's target platform (per-clip override wins over
    # brand-kit default).
    _ai_target_platform: str | None = None
    if isinstance(clip.overlay_config, dict):
        _tp = clip.overlay_config.get("target_platform")
        if isinstance(_tp, str) and _tp:
            _ai_target_platform = _tp
    if not _ai_target_platform and brand_kit is not None:
        _bk_tp = getattr(brand_kit, "target_platform", None)
        if isinstance(_bk_tp, str) and _bk_tp:
            _ai_target_platform = _bk_tp

    ai_decisions = _ai_decide(
        face_presence=breakdown.face_presence,
        speaking_intensity=breakdown.speaking_intensity,
        reaction_confidence=breakdown.reaction_confidence,
        heuristic_reason=breakdown.heuristic_reason,
        duration_s=clip.duration_s,
        face_zone=face_zone,
        target_platform=_ai_target_platform,
    )

    # Slice F.7-G — branding stickiness across clips.
    # When THIS clip has no overlay_config yet, prefill from the most-
    # recently-edited sibling clip on the same stream. The operator
    # types "kick.com/aldovillanueva" on clip 1, opens clip 2, and
    # the URL is already filled in. The brand_kit covers the
    # cross-stream case (different VODs from the same channel); this
    # covers the in-stream case where the brand_kit hasn't been
    # populated yet (e.g. first-time user, didn't visit Brand Kits).
    inherited_overlay: dict[str, object] | None = None
    if not clip.overlay_config:
        siblings = await ClipsRepo(db).list_for_stream(clip.stream_id)
        # Walk newest-first by created_at; .list_for_stream returns in
        # insertion order — sort defensively.
        siblings_sorted = sorted(
            (c for c in siblings if c.id != clip.id and c.overlay_config),
            key=lambda c: c.created_at,
            reverse=True,
        )
        if siblings_sorted:
            inherited = siblings_sorted[0].overlay_config
            if isinstance(inherited, dict):
                inherited_overlay = inherited

    # Phase 3: surface engagement outcomes per published job. One row per
    # publish_job with the latest metric reading next to the platform +
    # external URL. Dashboard shows "not yet measured" rows for jobs that
    # the metrics worker hasn't touched yet.
    publish_jobs = await PublishJobsRepo(db).list_for_clip(clip_id)
    metrics_repo = PublishMetricsRepo(db)
    outcomes: list[dict[str, object]] = []
    for job in publish_jobs:
        latest = await metrics_repo.latest_for_job(job.id)
        outcomes.append(
            {
                "job": job,
                "metric": latest,
            }
        )

    return templates.TemplateResponse(
        request,
        "clip_detail.html",
        {
            "clip": clip,
            "variants": variants,
            "accounts": accounts,
            "valid_transitions": valid_transitions,
            "breakdown": breakdown,
            "ai_scores": ai_scores,
            # Slice I.3 — Publishability verdict + AI Director narrative.
            "publishability": publishability,
            "director_lines": director,
            # Slice G.6 — what the pipeline auto-corrected at cut time
            # (trim + overlay fixes). Drives the passive "Corregido
            # automáticamente" panel that replaced the manual fix button.
            "auto_corrections": auto_corrections,
            # Slice K.7 — surface the voice-trigger match that fired
            # this candidate. None when the clip wasn't voice-triggered
            # (e.g. chat-heat-only candidates).
            "candidate_trigger": candidate_trigger,
            # Slice L.4b — coarse "where is the face" hint derived from
            # G.3 framing. Template wires it onto the preview as
            # data-face-zone="top|mid|lower" so the hook + captions
            # avoid the face. None when framing wasn't run.
            "face_zone": face_zone,
            # Slice L.6 — AI auto-decisions (hook needed / subtitles
            # needed / caption position / face vs gameplay priority /
            # reaction-wins-over-captions). Drives the ✨ intel lines
            # on the verdict card AND the "Auto Viral" cascade in the
            # editor JS.
            "ai_decisions": ai_decisions,
            "outcomes": outcomes,
            "brand_kit": brand_kit,
            "caption_style": caption_style,
            "caption_preset_choices": preset_choices(),
            # Slice I.1 — Clip Style preset cards (Repost Page Viral /
            # Clean Creator / Gaming Chaos / Documentary / Minimal Native).
            "clip_style_choices": style_choices(),
            # Slice K.5 — Target-platform chip row (TikTok / Reels /
            # Shorts / Kick / All). Picking one auto-applies every
            # downstream editor setting via PLATFORM_PRESETS.
            "platform_target_choices": platform_target_choices(),
            # Slice I.2 — platform overlay simulation + safe zones.
            # `platform_choices` populates the simulation dropdown;
            # `platform_zone_specs` is the full zone catalog, indexed
            # by platform id, so the editor JS can draw dashed zones
            # without a round-trip and run collision warnings client
            # side. The server-side `detect_collisions` is reserved
            # for headless validation (export checklist in I.3).
            "platform_choices": platform_choices(),
            "platform_zone_specs": {
                p: [
                    {
                        "kind": z.kind,
                        "x": z.x, "y": z.y, "w": z.w, "h": z.h,
                        "reason": z.reason,
                    }
                    for z in zones_for_platform(p)
                ]
                for p, _label in platform_choices()
            },
            # Slice F.7-G — passed through so the template's form-
            # prefill block can fall back to the previous clip's
            # overlay when this clip's overlay_config is still null.
            "inherited_overlay": inherited_overlay,
        },
    )


# ---- Clip overlay editor (slice F.6) ----


def _parse_overlay_form(
    *,
    title_text: str,
    banner_enabled: str,
    banner_platform: str,
    banner_url: str,
    banner_color: str,
    banner_show_context: str = "",
    banner_show_safezones: str = "",
    captions_enabled: str,
    captions_preset: str,
    captions_highlight_color: str,
    captions_position: str = "",
    captions_font_size: str = "",
    captions_animation: str = "",
    captions_lead_ms: int = 120,
    comments_show: str = "",
    comments_fake_likes: int = 8,
    # Slice I.1 — clip style + banner variant + top hook.
    clip_style: str = "",
    banner_variant: str = "",
    banner_live_badge: str = "",
    top_hook_enabled: str = "",
    top_hook_text: str = "",
    top_hook_style: str = "",
    # Slice I.2 — platform overlay simulation + safe-zone target +
    # preview mode. All three are EDITOR-only state — they never feed
    # the burn, but they DO persist to the brand_kit so the operator's
    # preferred simulation target survives across clips.
    platform_overlay_preview: str = "",
    safe_zone_platform: str = "",
    preview_mode: str = "",
    # Slice K.5 — target-platform auto-config. ONE chip at the top of
    # the editor (tiktok / reels / shorts / kick / all) that the JS
    # uses to fan-out every other field (safe_zone, captions,
    # banner_variant, clip_style…). We just need to remember *which*
    # platform the operator picked so the next clip opens to the
    # same chip.
    target_platform: str = "",
) -> dict[str, object]:
    """Coerce the editor form's flat key/value submission into the
    nested overlay_config shape the renderer reads.

    Empty / blank string fields collapse to None at the top level so
    the renderer falls back to brand-kit defaults — i.e. the editor
    is additive, not destructive."""

    def _bool(v: str) -> bool:
        return v.lower() in ("1", "true", "on", "yes")

    return {
        "title_text": title_text.strip() or None,
        # Slice I.1 — clip style preset (Repost Page Viral / etc).
        "clip_style": (clip_style or "").strip().lower() or None,
        "banner": {
            "enabled": _bool(banner_enabled),
            "platform": (banner_platform or "kick").strip().lower(),
            "url": banner_url.strip() or None,
            "color": banner_color.strip() or None,
            # Slice F.7 social-context overlay toggle. Decorative
            # only — the renderer doesn't burn the LIVE badge / chat
            # into the MP4. The HTML preview shows them so the
            # operator can see what the published clip looks like
            # inside the platform UI.
            "show_context": _bool(banner_show_context),
            # Slice F.7-F platform safe-zone guides. Editor-only —
            # never affects the burned MP4. Tells the operator which
            # regions of the frame get covered by TikTok/Reels chrome.
            "show_safezones": _bool(banner_show_safezones),
            # Slice I.1 — Kick banner variant (repost_page / classic /
            # green_block / minimal_url) + LIVE NOW pill toggle.
            "variant": (banner_variant or "").strip().lower() or None,
            "live_badge": _bool(banner_live_badge),
        },
        # Slice I.1 — top hook box (white rounded headline above face).
        "top_hook": {
            "enabled": _bool(top_hook_enabled),
            "text": top_hook_text.strip(),
            "style": (top_hook_style or "white_rounded").strip().lower(),
        },
        # Slice I.2 — editor-only platform sim + safe-zone target.
        "platform_overlay_preview":
            (platform_overlay_preview or "").strip().lower() or None,
        "safe_zone_platform": (safe_zone_platform or "").strip().lower() or None,
        "preview_mode": (preview_mode or "").strip().lower() or None,
        # Slice K.5 — picked target platform (tiktok/reels/shorts/kick/all).
        # JS uses this to fan-out the rest of the editor; we persist it so
        # the next clip remembers the operator's last pick.
        "target_platform": (target_platform or "").strip().lower() or None,
        "captions": {
            "enabled": _bool(captions_enabled),
            "preset": captions_preset.strip() or None,
            "highlight_color": captions_highlight_color.strip() or None,
            # Slice F.7-H — read-ahead + visual knobs. Each is None when
            # blank so the renderer/template falls back to the brand-kit
            # caption_style defaults (no overlapping sources of truth).
            "position": (captions_position.strip().lower() or None)
                if captions_position in (
                    "upper_third", "centered", "lower_third", "bottom"
                ) else None,
            "font_size": (captions_font_size.strip().lower() or None)
                if captions_font_size in ("small", "medium", "large", "xl") else None,
            "animation": (captions_animation.strip().lower() or None)
                if captions_animation in ("pop", "slide", "typewriter", "fade") else None,
            # Clamp 0..500ms — beyond 500ms the read-ahead is so far
            # ahead it stops being "anticipation" and starts being
            # "spoiler"; pre-roll captions are no longer in sync.
            "lead_ms": max(0, min(500, int(captions_lead_ms))),
        },
        "comments": {
            "show_overlay": _bool(comments_show),
            "fake_likes": max(0, int(comments_fake_likes)),
        },
    }


# ---- Branding persistence helpers (slice F.7-G) ----
#
# On every clip-overlay save the operator's branding choices (platform
# handle, banner color, caption preset + highlight color) get mirrored
# back into the tenant's default brand_kit. The next clip the operator
# opens then prefills from the brand_kit fallback path in clip_detail.html.
#
# Why a *brand-kit* write rather than a sticky-defaults table:
#   - the kit ALREADY drives the renderer's fallback path when
#     overlay_config_json is null (see clip_detail.html ~232).
#     mirroring there means there's a single source of truth.
#   - operators editing brand kits in /dashboard/brand_kits see the
#     latest "live" handle / color without an extra migration screen.
#
# We auto-create a brand_kit with `name="Default"` and `is_default=1`
# when none exists, so first-time users don't need a kit-create UI
# detour before their second clip inherits their first clip's URL.


def _platform_handle_field(platform: str) -> str | None:
    """Map the dropdown platform → brand_kit handle column. Returns
    `None` for platforms we don't have a column for (so the caller
    doesn't fabricate a write for an unknown column)."""
    return {
        "kick": "handle_kick",
        "tiktok": "handle_tiktok",
        "youtube": "handle_youtube",
        "instagram": "handle_instagram",
    }.get(platform.lower())


async def _persist_branding_to_brand_kit(
    db: Database,
    cfg: Mapping[str, object],
) -> None:
    """Mirror the operator's editor choices back to the tenant's
    default brand_kit. Best-effort — a missing field or repo failure
    must not break the overlay save (publishers don't depend on this).
    """
    from chalybclip.branding import caption_style_or_default

    repo = BrandKitsRepo(db)
    try:
        kit = await repo.get_default()
    except Exception:  # noqa: BLE001 — best-effort
        return

    banner = cfg.get("banner") or {}
    captions = cfg.get("captions") or {}
    top_hook = cfg.get("top_hook") or {}
    if not isinstance(banner, dict) or not isinstance(captions, dict):
        return
    if not isinstance(top_hook, dict):
        top_hook = {}

    # Slice I.1 — clip style preset (Repost Page Viral / etc) +
    # banner variant + top hook are all user-level prefs.
    clip_style_raw = cfg.get("clip_style")
    clip_style = (
        str(clip_style_raw).strip().lower() if isinstance(clip_style_raw, str) else ""
    )
    # Slice K.5 — target platform (tiktok/reels/shorts/kick/all). One
    # chip drives the whole editor; remembered as the operator's
    # default so the next clip auto-applies the same preset bundle.
    target_platform_raw = cfg.get("target_platform")
    target_platform_val = (
        str(target_platform_raw).strip().lower()
        if isinstance(target_platform_raw, str)
        else ""
    )
    banner_variant_raw = banner.get("variant")
    banner_variant = (
        str(banner_variant_raw).strip().lower()
        if isinstance(banner_variant_raw, str)
        else ""
    )
    banner_live_badge_val = (
        bool(banner.get("live_badge")) if "live_badge" in banner else None
    )
    top_hook_enabled_val = (
        bool(top_hook.get("enabled")) if "enabled" in top_hook else None
    )
    top_hook_style_raw = top_hook.get("style")
    top_hook_style = (
        str(top_hook_style_raw).strip().lower()
        if isinstance(top_hook_style_raw, str)
        else ""
    )

    platform_raw = banner.get("platform")
    platform = str(platform_raw).lower() if isinstance(platform_raw, str) else ""
    url_raw = banner.get("url")
    url = str(url_raw).strip() if isinstance(url_raw, str) else ""
    color_raw = banner.get("color")
    color = str(color_raw).strip() if isinstance(color_raw, str) else ""
    preset_raw = captions.get("preset")
    preset = str(preset_raw).strip() if isinstance(preset_raw, str) else ""
    hilite_raw = captions.get("highlight_color")
    hilite = str(hilite_raw).strip() if isinstance(hilite_raw, str) else ""
    # Slice H.1 — caption knobs from F.7-H + banner toggles now also
    # persist to the brand_kit so they survive across clips.
    position_raw = captions.get("position")
    position = str(position_raw).strip() if isinstance(position_raw, str) else ""
    font_size_raw = captions.get("font_size")
    font_size = str(font_size_raw).strip() if isinstance(font_size_raw, str) else ""
    animation_raw = captions.get("animation")
    animation = str(animation_raw).strip() if isinstance(animation_raw, str) else ""
    lead_ms_raw = captions.get("lead_ms")
    lead_ms_val: int | None = None
    if isinstance(lead_ms_raw, int | float):
        lead_ms_val = int(lead_ms_raw)
    elif isinstance(lead_ms_raw, str) and lead_ms_raw.strip().lstrip("-").isdigit():
        lead_ms_val = int(lead_ms_raw)
    banner_enabled_val = bool(banner.get("enabled")) if "enabled" in banner else None
    banner_show_context_val = (
        bool(banner.get("show_context")) if "show_context" in banner else None
    )
    banner_show_safezones_val = (
        bool(banner.get("show_safezones")) if "show_safezones" in banner else None
    )

    # Build the caption_style dict — now carries ALL four F.7-H knobs
    # plus the legacy preset/highlight_color. The renderer + preview
    # JS both read from this same dict via caption_style_or_default.
    caption_style_patch: dict[str, object] | None = None
    if preset or hilite or position or font_size or animation or lead_ms_val is not None:
        base = (
            caption_style_or_default(kit.caption_style).model_dump()
            if kit is not None
            else caption_style_or_default(None).model_dump()
        )
        if preset:
            base["preset_id"] = preset
        if hilite:
            base["highlight_color"] = hilite
        if position:
            base["position"] = position
        if font_size:
            base["font_size"] = font_size
        if animation:
            base["animation"] = animation
        if lead_ms_val is not None:
            base["lead_ms"] = lead_ms_val
        caption_style_patch = base

    handle_field = _platform_handle_field(platform)
    handle_kick = url if handle_field == "handle_kick" and url else None
    handle_tiktok = url if handle_field == "handle_tiktok" and url else None
    handle_youtube = url if handle_field == "handle_youtube" and url else None
    handle_instagram = url if handle_field == "handle_instagram" and url else None

    if kit is None:
        # Auto-create the tenant's default kit on first save so
        # subsequent clips inherit the chosen branding even if the
        # operator never visits /dashboard/brand_kits.
        try:
            new_kit = await repo.create(
                name="Default",
                primary_color=color or "#53FC18",
                accent_color="#FFD700",
                is_default=True,
                caption_style=caption_style_patch,
                handle_kick=handle_kick,
                handle_tiktok=handle_tiktok,
                handle_youtube=handle_youtube,
                handle_instagram=handle_instagram,
            )
        except Exception:  # noqa: BLE001
            return
        # The new toggle/platform fields aren't in `create()`'s signature
        # (they're update-only). Apply them on the fresh row immediately.
        if any(v is not None for v in [
            platform or None, banner_enabled_val,
            banner_show_context_val, banner_show_safezones_val,
            clip_style or None, banner_variant or None,
            banner_live_badge_val, top_hook_enabled_val, top_hook_style or None,
            target_platform_val or None,
        ]):
            try:
                await repo.update(
                    new_kit.id,
                    default_platform=platform or None,
                    banner_enabled_default=banner_enabled_val,
                    banner_show_context_default=banner_show_context_val,
                    banner_show_safezones_default=banner_show_safezones_val,
                    # Slice I.1.
                    clip_style=clip_style or None,
                    bottom_banner_style=banner_variant or None,
                    banner_live_badge_default=banner_live_badge_val,
                    top_hook_enabled_default=top_hook_enabled_val,
                    top_hook_style_default=top_hook_style or None,
                    # Slice K.5.
                    target_platform=target_platform_val or None,
                )
            except Exception:  # noqa: BLE001
                pass
        return

    # Existing default kit — partial update of just the fields the
    # operator changed in the editor. None-valued args are ignored by
    # BrandKitsRepo.update so we only touch what was provided.
    has_change = any([
        color, handle_kick, handle_tiktok, handle_youtube, handle_instagram,
        caption_style_patch, platform,
        banner_enabled_val is not None,
        banner_show_context_val is not None,
        banner_show_safezones_val is not None,
        clip_style, banner_variant,
        banner_live_badge_val is not None,
        top_hook_enabled_val is not None,
        top_hook_style,
        target_platform_val,
    ])
    if not has_change:
        return
    try:
        await repo.update(
            kit.id,
            primary_color=color or None,
            handle_kick=handle_kick,
            handle_tiktok=handle_tiktok,
            handle_youtube=handle_youtube,
            handle_instagram=handle_instagram,
            caption_style=caption_style_patch,
            default_platform=platform or None,
            banner_enabled_default=banner_enabled_val,
            banner_show_context_default=banner_show_context_val,
            banner_show_safezones_default=banner_show_safezones_val,
            # Slice I.1.
            clip_style=clip_style or None,
            bottom_banner_style=banner_variant or None,
            banner_live_badge_default=banner_live_badge_val,
            top_hook_enabled_default=top_hook_enabled_val,
            top_hook_style_default=top_hook_style or None,
            # Slice K.5.
            target_platform=target_platform_val or None,
        )
    except Exception:  # noqa: BLE001
        pass


@router.post(
    "/me/brand-kit-prefs",
    dependencies=[Depends(require_full_scope)],
)
async def me_brand_kit_prefs(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Slice H.1 — auto-save endpoint for the editor's right panel.

    The clip-editor JS debounce-calls this on every form change so the
    operator's setup (URL, banner toggle, caption position / font size
    / animation / lead-time, etc.) lands in the tenant's default
    brand_kit immediately. The brand_kit becomes the canonical source
    of truth for user-level setup; every new clip's editor reads from
    it on render.

    Body is a single-field JSON patch:

        { "field": "banner_url",  "value": "aldovillanueva" }
        { "field": "banner_enabled", "value": true }
        { "field": "captions_lead_ms", "value": 150 }

    Returns `{"ok": true}` on success. Failures return a 400 so the JS
    can surface the error inline without breaking the page.

    SINGLE source of truth: this endpoint and the existing
    `_persist_branding_to_brand_kit` (which fires on Save Draft /
    Finalize) both write to the same brand_kit columns — no risk of
    drift because the brand_kit row is the only writer destination.
    """
    import json as _json

    body_bytes = await request.body()
    try:
        body = _json.loads(body_bytes or b"{}")
    except _json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid JSON") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    field = str(body.get("field") or "").strip()
    value: object = body.get("value")
    if not field:
        raise HTTPException(status_code=400, detail="missing 'field'")

    # Wrap the single-field patch into the same `cfg` shape that
    # `_persist_branding_to_brand_kit` already knows how to consume.
    # That keeps the brand-kit-write logic in exactly one place.
    cfg: dict[str, object] = {"banner": {}, "captions": {}, "top_hook": {}}
    banner_block: dict[str, object] = cfg["banner"]   # type: ignore[assignment]
    captions_block: dict[str, object] = cfg["captions"]  # type: ignore[assignment]
    top_hook_block: dict[str, object] = cfg["top_hook"]  # type: ignore[assignment]

    _BANNER_FIELDS = {
        "banner_enabled": ("enabled", bool),
        "banner_platform": ("platform", str),
        "banner_url": ("url", str),
        "banner_color": ("color", str),
        "banner_show_context": ("show_context", bool),
        "banner_show_safezones": ("show_safezones", bool),
        # Slice I.1.
        "banner_variant": ("variant", str),
        "banner_live_badge": ("live_badge", bool),
    }
    _CAPTION_FIELDS = {
        "captions_preset": ("preset", str),
        "captions_highlight_color": ("highlight_color", str),
        "captions_position": ("position", str),
        "captions_font_size": ("font_size", str),
        "captions_animation": ("animation", str),
        "captions_lead_ms": ("lead_ms", int),
    }
    # Slice I.1 — top hook + clip_style live at the top level / their
    # own block. clip_style is a flat top-level key on the cfg dict.
    _TOP_HOOK_FIELDS = {
        "top_hook_enabled": ("enabled", bool),
        "top_hook_text": ("text", str),
        "top_hook_style": ("style", str),
    }

    # `clip_style` is the only top-level brand-kit-bound editor field.
    # The I.2 editor-only fields (platform_overlay_preview / safe_zone_
    # platform / preview_mode) are persisted client-side via localStorage
    # rather than written to the brand_kit — they're pure editor state
    # and don't belong in the user's branding record. They still ride
    # the Save Draft path into clips.overlay_config_json so a per-clip
    # override is honored when set, but day-to-day persistence is JS.
    # Slice K.5 — `target_platform` is also brand-kit-bound: it's the
    # operator's "where I post" sticky default, so the next clip opens
    # to the same chip pre-selected and every downstream field
    # auto-applies from PLATFORM_PRESETS.
    if field == "clip_style":
        cfg["clip_style"] = _coerce(value, str)
    elif field == "target_platform":
        cfg["target_platform"] = _coerce(value, str)
    elif field in _BANNER_FIELDS:
        key, kind = _BANNER_FIELDS[field]
        banner_block[key] = _coerce(value, kind)
    elif field in _CAPTION_FIELDS:
        key, kind = _CAPTION_FIELDS[field]
        captions_block[key] = _coerce(value, kind)
    elif field in _TOP_HOOK_FIELDS:
        key, kind = _TOP_HOOK_FIELDS[field]
        top_hook_block[key] = _coerce(value, kind)
    else:
        raise HTTPException(status_code=400, detail=f"unknown field {field!r}")

    await _persist_branding_to_brand_kit(db, cfg)
    return Response(
        content='{"ok": true}',
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


def _coerce(value: object, kind: type) -> object:
    """Best-effort JSON value → target type. Falsy strings collapse to
    None so an empty input clears a setting instead of writing ''."""
    if kind is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "on", "yes")
        return bool(value)
    if kind is int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int | float):
            return int(value)
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value)
        return 0
    if kind is str:
        return str(value).strip() if value is not None else ""
    return value


@router.post(
    "/clips/{clip_id}/overlay",
    dependencies=[Depends(require_full_scope)],
)
async def clip_overlay_save(
    request: Request,
    clip_id: str,
    title_text: str = Form(""),
    banner_enabled: str = Form(""),
    banner_platform: str = Form("kick"),
    banner_url: str = Form(""),
    banner_color: str = Form(""),
    banner_show_context: str = Form(""),
    banner_show_safezones: str = Form(""),
    captions_enabled: str = Form(""),
    captions_preset: str = Form(""),
    captions_highlight_color: str = Form(""),
    captions_position: str = Form(""),
    captions_font_size: str = Form(""),
    captions_animation: str = Form(""),
    captions_lead_ms: int = Form(120),
    # Slice I.1 — clip style + Kick banner variant + top hook.
    clip_style: str = Form(""),
    banner_variant: str = Form(""),
    banner_live_badge: str = Form(""),
    top_hook_enabled: str = Form(""),
    top_hook_text: str = Form(""),
    top_hook_style: str = Form(""),
    # Slice I.2 — editor-only platform simulation + safe-zone target.
    platform_overlay_preview: str = Form(""),
    safe_zone_platform: str = Form(""),
    preview_mode: str = Form(""),
    # Slice K.5 — target-platform auto-config (one chip → all settings).
    target_platform: str = Form(""),
    comments_show: str = Form(""),
    comments_fake_likes: int = Form(0),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Save (only) the per-clip overlay config. Idempotent — re-POSTing
    overwrites. Used by the "Save draft" button on the clip editor."""
    if await ClipsRepo(db).get(clip_id) is None:
        raise HTTPException(status_code=404, detail="clip not found")
    cfg = _parse_overlay_form(
        title_text=title_text,
        banner_enabled=banner_enabled,
        banner_platform=banner_platform,
        banner_url=banner_url,
        banner_color=banner_color,
        banner_show_context=banner_show_context,
        banner_show_safezones=banner_show_safezones,
        captions_enabled=captions_enabled,
        captions_preset=captions_preset,
        captions_highlight_color=captions_highlight_color,
        captions_position=captions_position,
        captions_font_size=captions_font_size,
        captions_animation=captions_animation,
        captions_lead_ms=captions_lead_ms,
        clip_style=clip_style,
        banner_variant=banner_variant,
        banner_live_badge=banner_live_badge,
        top_hook_enabled=top_hook_enabled,
        top_hook_text=top_hook_text,
        top_hook_style=top_hook_style,
        platform_overlay_preview=platform_overlay_preview,
        safe_zone_platform=safe_zone_platform,
        preview_mode=preview_mode,
        target_platform=target_platform,
        comments_show=comments_show,
        comments_fake_likes=comments_fake_likes,
    )
    repo = ClipsRepo(db)
    clip_before = await repo.get(clip_id)
    await repo.set_overlay_config(clip_id, overlay_config=cfg)
    # Slice F.7-G — mirror branding choices to the tenant brand_kit
    # so the operator doesn't re-type URL / color on every clip.
    await _persist_branding_to_brand_kit(db, cfg)

    # Slice O.17 — burn-on-save REMOVED. The previous O.12 behavior
    # re-encoded clip_final.mp4 on every editor save, which was both
    # wasteful (operator might save 10× while iterating) and useless
    # (any subsequent save invalidates the prior burn anyway). The
    # download / publish path already lazy-regenerates clip_final.mp4
    # if it's missing — so saves become DB-only writes, and the burn
    # happens exactly when an MP4 is actually about to leave the
    # system.
    #
    # We MUST nuke any cached export here — otherwise the download
    # endpoint sees a file on disk + skips the lazy regen + serves
    # an MP4 produced with the OLD overlay config. Both the new
    # Playwright-rendered cache AND the legacy ffmpeg burn cache get
    # invalidated.
    if clip_before is not None:
        try:
            clip_dir = Path(clip_before.path).parent
            # Slice J.2a — invalidate ALL resolution caches, not just the
            # base name. clip_render.mp4 (legacy) + per-resolution variants
            # all need to go so the next download lazy-regenerates with
            # the current overlay config.
            stale_names = [
                "clip_render.mp4",
                "clip_render_1080.mp4",
                "clip_render_2k.mp4",
                "clip_render_4k.mp4",
                "clip_final.mp4",
            ]
            for name in stale_names:
                p = clip_dir / name
                if p.exists():
                    p.unlink()
        except Exception:  # noqa: BLE001 — invalidation is best-effort
            pass
        # Render Migration T1 — reset the render state along with the
        # cache files so the UI's polling Download button isn't stuck
        # reading 'ready' against a now-deleted file.
        try:
            await ClipsRepo(db).reset_render_state(clip_id)
        except Exception:  # noqa: BLE001 — best-effort
            pass
    return RedirectResponse(url=f"/dashboard/clips/{clip_id}", status_code=303)


@router.post(
    "/clips/{clip_id}/finalize",
    dependencies=[Depends(require_full_scope)],
)
async def clip_overlay_finalize(
    request: Request,
    background_tasks: BackgroundTasks,
    clip_id: str,
    title_text: str = Form(""),
    banner_enabled: str = Form(""),
    banner_platform: str = Form("kick"),
    banner_url: str = Form(""),
    banner_color: str = Form(""),
    banner_show_context: str = Form(""),
    banner_show_safezones: str = Form(""),
    captions_enabled: str = Form(""),
    captions_preset: str = Form(""),
    captions_highlight_color: str = Form(""),
    captions_position: str = Form(""),
    captions_font_size: str = Form(""),
    captions_animation: str = Form(""),
    captions_lead_ms: int = Form(120),
    # Slice I.1 — clip style + Kick banner variant + top hook.
    clip_style: str = Form(""),
    banner_variant: str = Form(""),
    banner_live_badge: str = Form(""),
    top_hook_enabled: str = Form(""),
    top_hook_text: str = Form(""),
    top_hook_style: str = Form(""),
    # Slice I.2 — editor-only platform simulation + safe-zone target.
    platform_overlay_preview: str = Form(""),
    safe_zone_platform: str = Form(""),
    preview_mode: str = Form(""),
    # Slice K.5 — target-platform auto-config (one chip → all settings).
    target_platform: str = Form(""),
    comments_show: str = Form(""),
    comments_fake_likes: int = Form(0),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """The 'Complete' button — saves the overlay config AND transitions
    the clip from `cut` / `ready_for_review` → `approved` (the existing
    pre-publish standby state).

    Rejects when the current status doesn't allow the transition (e.g.
    a `published` clip). The dashboard hides the button in those cases
    so the 409 only fires on a stale page."""
    repo = ClipsRepo(db)
    clip = await repo.get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")

    allowed = _VALID_STATUS_TRANSITIONS.get(clip.status, set())
    # If we're not already at `approved` and can't transition there
    # directly, walk one step (`cut` -> `ready_for_review` -> `approved`).
    if clip.status == "approved" or "approved" in allowed:
        target = "approved"
    elif clip.status == "cut" and "ready_for_review" in allowed:
        # `cut` -> `ready_for_review` first, then `approved`.
        await repo.update_status(clip_id, status="ready_for_review")
        target = "approved"
    else:
        raise HTTPException(
            status_code=409,
            detail=f"can't finalize from status {clip.status!r}",
        )

    cfg = _parse_overlay_form(
        title_text=title_text,
        banner_enabled=banner_enabled,
        banner_platform=banner_platform,
        banner_url=banner_url,
        banner_color=banner_color,
        banner_show_context=banner_show_context,
        banner_show_safezones=banner_show_safezones,
        captions_enabled=captions_enabled,
        captions_preset=captions_preset,
        captions_highlight_color=captions_highlight_color,
        captions_position=captions_position,
        captions_font_size=captions_font_size,
        captions_animation=captions_animation,
        captions_lead_ms=captions_lead_ms,
        clip_style=clip_style,
        banner_variant=banner_variant,
        banner_live_badge=banner_live_badge,
        top_hook_enabled=top_hook_enabled,
        top_hook_text=top_hook_text,
        top_hook_style=top_hook_style,
        platform_overlay_preview=platform_overlay_preview,
        safe_zone_platform=safe_zone_platform,
        preview_mode=preview_mode,
        target_platform=target_platform,
        comments_show=comments_show,
        comments_fake_likes=comments_fake_likes,
    )
    await repo.set_overlay_config(clip_id, overlay_config=cfg)
    # Slice F.7-G — see note in clip_overlay_save: persists branding
    # to the tenant brand_kit so the *next* clip prefills it.
    await _persist_branding_to_brand_kit(db, cfg)
    if target != clip.status:
        await repo.update_status(clip_id, status=target)

    # Slice O.37 — Approve no longer blocks on ffmpeg burn. O.17 took
    # the burn out of the save path; finalize was still doing a
    # synchronous re-encode, which is what made "Continue & close"
    # spin for 30-90s. The download path lazy-regenerates a fresh
    # MP4 (Playwright recorder, fallback ffmpeg burn) from the saved
    # overlay config, so we just nuke any stale cache and bounce.
    try:
        clip_dir = Path(clip.path).parent
        # Slice J.2a — same multi-resolution invalidation as the save path.
        stale_names = [
            "clip_render.mp4",
            "clip_render_1080.mp4",
            "clip_render_2k.mp4",
            "clip_render_4k.mp4",
            "clip_final.mp4",
        ]
        for name in stale_names:
            p = clip_dir / name
            if p.exists():
                p.unlink()
    except Exception:  # noqa: BLE001 — invalidation is best-effort
        pass

    # Pre-render the 1080 publish preset in the background now that the
    # cache is clear. Two payoffs:
    #   - the operator's next Download is a cache hit (no 30-90s wait);
    #   - a subsequent Publish finds the edited MP4 already on disk, so
    #     what ships to social == what they downloaded == the preview.
    # Best-effort: a pre-render failure must never block approval — the
    # download/publish paths still lazy-render on demand. Uses the
    # operator's session cookie to drive the auth-gated /render page.
    try:
        from chalybclip.settings import get_settings
        _settings = get_settings()
        _original = Path(clip.path)
        if _original.exists():
            _rendered_1080 = _original.parent / "clip_render_1080.mp4"
            _cookie_val = request.cookies.get("chalybclip_token", "") or None
            _explicit_base = (_settings.public_url or "").strip()
            if _explicit_base and _explicit_base != "http://localhost:8000":
                _base_url = _explicit_base
            else:
                _scheme = (
                    request.headers.get("x-forwarded-proto")
                    or request.url.scheme
                    or "https"
                )
                _host = request.headers.get("host") or request.url.netloc
                _base_url = f"{_scheme}://{_host}"
            # Atomic flip to 'rendering' so the download endpoint's poll
            # shows progress instead of re-dispatching a duplicate task.
            await repo.mark_render_started(clip_id)
            from chalybclip.api._clip_render import render_clip_in_background
            background_tasks.add_task(
                render_clip_in_background,
                clip_id=clip_id,
                tenant_id=tenant_id,
                duration_s=float(clip.duration_s),
                audio_source_path=_original,
                output_path=_rendered_1080,
                base_url=_base_url,
                auth_cookie_value=_cookie_val,
                width=1080,
                height=1920,
                db_path=resolve_db_target(_settings),
            )
    except Exception as e:  # noqa: BLE001 — pre-render must never block approval
        from structlog import get_logger
        get_logger(__name__).warning(
            "clip.finalize.prerender_failed",
            clip_id=clip_id,
            reason=str(e),
        )

    await EventsRepo(db).emit(
        type="clip.finalized",
        payload={
            "clip_id": clip_id,
            "to_status": target,
            "burn_outcome": "deferred_to_download",
        },
    )

    # Auto-publish ("Piloto automático"): if the tenant turned it on with
    # mode=on_approve, the approve we just did auto-enqueues the clip to
    # Zernio with its burned-in render. Best-effort — never blocks approval
    # (the helper swallows its own errors). Routed through the Zernio publish
    # core, NOT the retired publish_jobs worker.
    try:
        from chalybclip.api.routers.zernio import maybe_autopublish_on_approve

        await maybe_autopublish_on_approve(
            request=request, db=db, tenant_id=tenant_id, clip_id=clip_id
        )
    except Exception as e:  # auto-publish must never block the approve flow
        from structlog import get_logger

        get_logger(__name__).warning(
            "clip.finalize.autopublish_hook_failed", clip_id=clip_id, reason=str(e)
        )

    # Slice N.2 — Approve & continue. After finalize, walk to the
    # NEXT clip on the same stream that's still in the editor queue
    # (status in `cut` / `ready_for_review`). If none remain, fall
    # back to the stream detail page so the operator sees they're
    # done. The "next clip" definition: earliest created_at among
    # remaining draft clips on this stream, excluding the one we
    # just approved.
    siblings = await repo.list_for_stream(clip.stream_id)
    next_clip = None
    for c in sorted(siblings, key=lambda x: x.created_at):
        if c.id == clip_id:
            continue
        if c.status in ("cut", "ready_for_review"):
            next_clip = c
            break

    if next_clip is not None:
        return RedirectResponse(
            url=f"/dashboard/clips/{next_clip.id}?from_approve=1",
            status_code=303,
        )
    return RedirectResponse(
        url=f"/dashboard/streams/{clip.stream_id}?queue_done=1",
        status_code=303,
    )


@router.post(
    "/clips/{clip_id}/revert-trim",
    dependencies=[Depends(require_full_scope)],
)
async def clip_revert_trim(
    request: Request,
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Slice G.4b — restore the clip to its original bounds.

    No-op when the clip has never been trimmed. Otherwise re-cuts
    the source video at the originals.
    """
    from chalybclip.clip.trim import revert_trim

    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    try:
        result = await revert_trim(db, clip_id=clip_id)
        await EventsRepo(db).emit(
            type="clip.trim_reverted",
            payload={"clip_id": clip_id, **result},
        )
        flash = result.get("outcome", "reverted")
    except Exception as e:  # noqa: BLE001
        from structlog import get_logger
        get_logger(__name__).warning(
            "clip.revert_trim.failed", clip_id=clip_id, reason=str(e)
        )
        flash = "error"
    return RedirectResponse(
        url=f"/dashboard/clips/{clip_id}?trim={flash}", status_code=303
    )


@router.post(
    "/clips/{clip_id}/reject",
    dependencies=[Depends(require_full_scope)],
)
async def clip_reject(
    request: Request,
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Slice F.7-H — one-click reject + close editor.

    Forces the clip into `rejected` (any valid in-bound transition is
    accepted; we don't fight the operator's intent), emits the
    `clip.rejected` event for the audit trail, and redirects the
    browser back to the stream page so the editor closes immediately.
    The stream-page clip card shows the REJECTED stamp via its
    `data-status="rejected"` overlay.
    """
    repo = ClipsRepo(db)
    clip = await repo.get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    if clip.status != "rejected":
        # Skip the strict transition map — reject from ANY state is a
        # destination, not an intermediate step. The transitions table
        # only allows `cut -> rejected` and `ready_for_review -> rejected`
        # today; we extend that for the operator's one-click flow.
        conn = await db.connect()
        await conn.execute(
            "UPDATE clips SET status = ? WHERE id = ? AND tenant_id = ?",
            ("rejected", clip_id, tenant_id),
        )
        await conn.commit()
        await EventsRepo(db).emit(
            type="clip.rejected",
            payload={"clip_id": clip_id, "from": clip.status, "to": "rejected"},
        )
    return RedirectResponse(
        url=f"/dashboard/streams/{clip.stream_id}", status_code=303
    )


async def _burn_overlays_for_clip(
    *,
    db: Database,
    clip_id: str,
    overlay_config: dict[str, object],
) -> str:
    """Run the overlay burn for one clip, returning a short outcome
    string suitable for the clip.finalized event payload:

      - "skipped_no_overlays"  — nothing was enabled (no title, no
                                 banner, captions off OR no transcript)
      - "skipped_clip_missing" — the source MP4 isn't on disk
      - "burned"               — clip_final.mp4 written successfully
      - "failed:<short reason>" — ffmpeg returned non-zero; the
                                  original clip.mp4 is untouched and
                                  publishers will fall back to it

    NEVER raises — the finalize endpoint shouldn't fail just because
    a render variant didn't pan out. The operator can re-finalize to
    retry. The full ffmpeg stderr is logged regardless.
    """
    import asyncio
    import json as _json

    import structlog

    from chalybclip.clip import burn_overlays
    from chalybclip.db import TenantsRepo, TranscriptsRepo

    log = structlog.get_logger("chalybclip.api.dashboard")

    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        return "skipped_clip_missing"
    source_path = Path(clip.path)
    if not source_path.exists():
        log.warning("burn.source_missing", clip_id=clip_id, path=str(source_path))
        return "skipped_clip_missing"
    target_path = source_path.parent / "clip_final.mp4"

    transcript = await TranscriptsRepo(db).get(clip.stream_id)
    segments: list[object] = []
    if transcript is not None:
        try:
            segments = _json.loads(transcript.segments_json) or []
        except (TypeError, _json.JSONDecodeError):
            segments = []

    # Slice O.1 — resolve whether to burn the chalybclip.com watermark.
    # Free tier always; pro+ respects the brand_kit toggle. Best-effort
    # — we never block a burn on missing tier/kit data.
    render_wm = True
    try:
        tenant = await TenantsRepo(db).get(clip.tenant_id)
        tier = (tenant.tier if tenant else "free") or "free"
        if tier != "free":
            from chalybclip.branding import resolve_brand_kit_for_candidate
            kit = await resolve_brand_kit_for_candidate(
                db, stream_id=clip.stream_id, speaker_label=None
            )
            if kit is not None and not getattr(kit, "show_chalybclip_credit", True):
                render_wm = False
    except Exception:  # noqa: BLE001 — best-effort
        render_wm = True

    try:
        # ffmpeg is sync + CPU-bound; offload to a thread so the
        # event loop isn't blocked while a 30s clip re-encodes.
        burned = await asyncio.to_thread(
            burn_overlays,
            source_path=source_path,
            target_path=target_path,
            overlay_config=overlay_config,
            transcript_segments=segments,
            clip_start_s=clip.start_s,
            clip_end_s=clip.end_s,
            output_w=clip.width,
            output_h=clip.height,
            render_watermark=render_wm,
        )
    except Exception as e:  # noqa: BLE001 — we want the catch-all
        log.warning("burn.failed", clip_id=clip_id, error=str(e))
        # Reason capped at 80 chars so the events table doesn't get
        # ffmpeg essays.
        reason = str(e)[:80].replace("\n", " ")
        return f"failed:{reason}"

    # Append the chalybclip end card to the burned fallback artifact so
    # the legacy-burn path matches the recorder path (which appends its
    # own outro before publishing). Free tier always; paid tiers respect
    # the brand-kit toggle. Best-effort — append_outro never raises and
    # leaves clip_final.mp4 untouched on failure.
    if burned:
        from chalybclip.branding import outro_enabled_for_clip
        from chalybclip.clip.outro import append_outro

        if await outro_enabled_for_clip(
            db, tenant_id=clip.tenant_id, stream_id=clip.stream_id
        ):
            await asyncio.to_thread(append_outro, target_path)
    return "burned" if burned else "skipped_no_overlays"


@router.post(
    "/clips/{clip_id}/generate-hooks",
    dependencies=[Depends(require_full_scope)],
)
async def clip_generate_hooks(
    request: Request,
    clip_id: str,
    tone: str = Form("default"),
    n: int = Form(5),
    persona_id: str = Form(""),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Generate `n` viral-hook title candidates for a clip and return
    them as JSON for the editor's hook-picker UI to render inline.

    Driven by a tone preset (default / aggressive / gen_z / corporate /
    curious) so the operator can sweep approaches without re-typing
    the prompt. Each call is a single Anthropic request — cost-tracked
    by the LLMRouter under purpose='hook_generation'.

    Returns:
        JSON: {"hooks": [{"text": "..."}, ...], "tone": "...", "n": N}
        on success.
        502 on provider failure with the LLM error inline.
    """
    import json as _json

    from chalybclip.db import CandidatesRepo, PersonasRepo, TranscriptsRepo
    from chalybclip.llm import LLMRouter, load_llm_config
    from chalybclip.settings import get_settings
    from chalybclip.variants import generate_hooks
    from chalybclip.variants.personas import (
        get_persona as get_yaml_persona,
        load_personas as load_yaml_personas,
    )

    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")

    # Pick the persona: explicit form arg > first-DB-persona > YAML fallback.
    persona_voice = ""
    persona_language = "en"
    if persona_id:
        db_persona = await PersonasRepo(db).get(persona_id)
        if db_persona is not None:
            persona_voice = db_persona.voice_prompt
            persona_language = db_persona.primary_language
        else:
            try:
                yp = get_yaml_persona(persona_id)
                persona_voice = yp.voice_prompt
                persona_language = yp.primary_language
            except Exception:
                pass
    if not persona_voice:
        # No explicit persona — grab any persona the tenant has, else
        # the first YAML persona, else generate with empty voice (the
        # service tolerates it via "(no persona voice provided)").
        db_personas = await PersonasRepo(db).list_for_tenant()
        if db_personas:
            persona_voice = db_personas[0].voice_prompt
            persona_language = db_personas[0].primary_language
        else:
            try:
                yps = list(load_yaml_personas().values())
                if yps:
                    persona_voice = yps[0].voice_prompt
                    persona_language = yps[0].primary_language
            except Exception:
                pass

    # Pull a transcript snippet — the candidate evidence is the
    # cheapest source. Falls back to scanning the candidates table for
    # this clip when the candidate row carries a longer snippet.
    snippet = ""
    if clip.candidate_id:
        for cand in await CandidatesRepo(db).list_for_stream(clip.stream_id):
            if cand.id == clip.candidate_id:
                ev = cand.evidence or {}
                snippet = (
                    str(ev.get("transcript_snippet") or ev.get("phrase") or "")
                ).strip()
                break
    if not snippet:
        # Fall back to the words from the transcript that overlap
        # the clip window. Cheap heuristic — we slice the JSON segments.
        try:
            tx = await TranscriptsRepo(db).get(clip.stream_id)
            if tx is not None:
                segs = _json.loads(tx.segments_json)
                if isinstance(segs, list):
                    pieces = []
                    for s in segs:
                        if not isinstance(s, dict):
                            continue
                        s_start = s.get("start") or 0
                        s_end = s.get("end") or 0
                        text = (s.get("text") or "").strip()
                        if text and s_end >= clip.start_s and s_start <= clip.end_s:
                            pieces.append(text)
                    snippet = " ".join(pieces).strip()[:600]
        except Exception:
            snippet = ""

    output_dir = Path(get_settings().default_output_dir)
    call_log_path = output_dir / "llm_calls_hooks.jsonl"
    router_ = LLMRouter(
        config=load_llm_config(),
        call_log_path=call_log_path,
        db=db,
    )

    # Coerce + clamp form inputs.
    tone_id = tone.strip().lower() or "default"
    if tone_id not in ("default", "aggressive", "gen_z", "corporate", "curious"):
        tone_id = "default"
    try:
        n_int = int(n)
    except (TypeError, ValueError):
        n_int = 5
    n_int = max(1, min(10, n_int))

    source = "llm"
    try:
        hooks = await generate_hooks(
            tenant_id=tenant_id,
            persona_voice=persona_voice,
            persona_language=persona_language,
            transcript_snippet=snippet,
            tone=tone_id,  # type: ignore[arg-type]
            n=n_int,
            router=router_,
        )
        hook_texts = [h.text for h in hooks]
    except Exception:
        # LLM down / out of credits — the button must still return usable
        # titles. Build them deterministically from the transcript snippet
        # and the stream title instead of 502-ing the editor.
        from chalybclip.variants import deterministic_hook_candidates

        stream = await StreamsRepo(db).get(clip.stream_id)
        hook_texts = deterministic_hook_candidates(
            transcript_snippet=snippet,
            stream_title=(stream.title or "") if stream else "",
            language=persona_language,
            seed=clip_id,
            n=n_int,
        )
        source = "deterministic"

    return Response(
        content=_json.dumps(
            {
                "hooks": [{"text": t} for t in hook_texts],
                "tone": tone_id,
                "n": n_int,
                "source": source,
            }
        ),
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


@router.post(
    "/streams/{stream_id}/clips/approve-all",
    dependencies=[Depends(require_full_scope)],
)
async def stream_approve_all_clips(
    stream_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Bulk-approve every clip on a stream still awaiting review
    (cut / ready_for_review → approved), so the operator doesn't click
    through 40 clips one at a time.

    Status-flip ONLY — it deliberately does NOT publish or render. Firing a
    render per clip here would spawn dozens of Chromium/ffmpeg processes at
    once and exhaust the box. After this, the Publish Center's
    "Auto-programar" schedules the now-approved clips (sequentially, capped).
    """
    repo = ClipsRepo(db)
    clips = await repo.list_for_stream(stream_id)
    approved = 0
    for clip in clips:
        if clip.status in ("cut", "ready_for_review"):
            await repo.update_status(clip.id, status="approved")
            approved += 1
    await EventsRepo(db).emit(
        type="clips.bulk_approved",
        payload={"stream_id": stream_id, "count": approved},
    )
    return RedirectResponse(
        url=f"/dashboard/streams/{stream_id}?approved={approved}#clips",
        status_code=303,
    )


@router.get("/streams/{stream_id}/download-approved")
async def stream_download_approved(
    stream_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Slice O.1 — bulk extract: zip every approved/published clip
    on this stream into a single download. Operator's "Download all"
    flow at the top of the Clips section.

    Builds the zip in-memory (clips are short, the entire bundle
    rarely exceeds a few hundred MB). For a stream with hundreds of
    finalized clips this would warrant streaming-zip-on-disk, but
    that's a J.2 follow-up — short-form clip pipelines max out
    around 20-40 clips per stream.
    """
    import io
    import zipfile

    repo = ClipsRepo(db)
    clips = await repo.list_for_stream(stream_id)
    if not clips:
        raise HTTPException(status_code=404, detail="no clips on this stream")
    # Pick the ones the operator would actually want to ship —
    # `approved` (ready to publish) + `published` (already shipped
    # but operator wants a backup copy).
    ready = [c for c in clips if c.status in ("approved", "published")]
    if not ready:
        raise HTTPException(
            status_code=404,
            detail="no approved clips yet — approve some first",
        )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        # Lower compresslevel = MP4s are already compressed; we only
        # want zip's container behavior, not double-compression CPU.
        for i, c in enumerate(ready, 1):
            original = Path(c.path)
            final = original.parent / "clip_final.mp4"
            src = final if final.exists() else original
            if not src.exists():
                continue
            # Index-prefixed name so the order in the zip mirrors the
            # order on the stream page.
            zf.write(src, arcname=f"{i:02d}_chalybclip_{c.id}.mp4")

    buf.seek(0)
    body = buf.read()
    fname = f"chalybclip_stream_{stream_id}.zip"
    return Response(
        content=body,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{fname}"',
            "Content-Length": str(len(body)),
            "Cache-Control": "no-store",
        },
    )


# Slice J.2a — resolution tiers. Free + pro are 1080p only; all_access
# unlocks 2K + 4K with a per-export extra cost reported back to chalyb
# for billing. Keys are URL-safe; values are (w, h, multiplier-on-base-cost).
# The multiplier is what we add to the LLM-call cost report so the
# operator's plan absorbs the extra GPU time the larger viewport eats.
_EXPORT_RESOLUTIONS: dict[str, tuple[int, int, float]] = {
    "1080": (1080, 1920, 1.0),
    "2k": (1440, 2560, 1.8),
    "4k": (2160, 3840, 3.6),
}

# Render Migration R6 — a render that hasn't completed in this many
# seconds is treated as a zombie (background task crashed before it
# could mark_render_failed). The /render-status endpoint surfaces it
# as failed with a retry hint so the operator isn't stuck. We pick
# 15 min as a wide-but-safe ceiling: the slowest legacy seek-and-shoot
# path tops out around ~5-7 min for a 35s clip; 4K + future longer
# clips still have room. If an operator legitimately renders a longer
# clip and hits this, the next click on Retry triggers a fresh
# background task and continues.
_ZOMBIE_RENDERING_TIMEOUT_S: int = 15 * 60
_TIER_RESOLUTIONS: dict[str, set[str]] = {
    "free": {"1080"},
    "pro": {"1080"},
    "all_access": {"1080", "2k", "4k"},
}


def _resolve_export_resolution(tier: str, choice: str) -> tuple[str, int, int]:
    """Map (tier, requested) -> (resolved_key, width, height).

    Falls back to 1080 if the tier can't access the requested resolution.
    Caller is responsible for surfacing that fact to the operator.
    """
    requested = (choice or "1080").strip().lower()
    if requested not in _EXPORT_RESOLUTIONS:
        requested = "1080"
    allowed = _TIER_RESOLUTIONS.get(tier or "free", {"1080"})
    if requested not in allowed:
        requested = "1080"
    w, h, _mult = _EXPORT_RESOLUTIONS[requested]
    return requested, w, h


@router.get("/clips/{clip_id}/render-status")
async def clip_render_status(
    request: Request,
    clip_id: str,
    quality: str = "1080",
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Render Migration T1 — JSON status feed for the polling UI.

    The Download button hits this every ~1 s and renders a progress
    bar / "Download MP4" link / "Failed — retry" affordance based on
    the response shape:

      {state: "idle",     progress_pct: 0,   error: null}  → operator hasn't clicked yet
      {state: "rendering",progress_pct: 42,  error: null}  → background task in flight
      {state: "ready",    progress_pct: 100, error: null,  → cache file exists; UI swaps
       download_url: "..."}                                 to a direct download link
      {state: "failed",   progress_pct: ...,error: "..."}  → UI shows retry button

    Falls through to "ready" when the cache file is on disk but the
    DB column never got promoted (e.g. legacy clips rendered pre-T1
    or sweeper hasn't run). That keeps backward compat with any
    already-rendered MP4 the operator's deploy might have.
    """
    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    original = Path(clip.path)
    tenant_tier = getattr(request.state, "tenant_tier", "free") or "free"
    resolved_key, _w, _h = _resolve_export_resolution(tenant_tier, quality)
    rendered = original.parent / f"clip_render_{resolved_key}.mp4"

    # Cache hit takes precedence over the DB state — covers the legacy
    # clip that was rendered before this migration ran.
    #
    # Render Migration R3 — `.exists()` alone isn't enough: a partial
    # / 0-byte / truncated file from a crashed encode also passes. We
    # delegate to is_servable_cached_mp4 which adds a size floor + ISO
    # BMFF magic-byte check.
    #
    # Render Migration R9 — but DO NOT evict while state='rendering'.
    # /render-status is polled ~1/s during a render; the in-progress
    # ffmpeg output is small (< 1MB) for most of the encode, so
    # is_servable_cached_mp4 returns False. The OLD code then
    # unlinked the file — on Linux that orphans the inode (ffmpeg's
    # fd is still open so the encode "succeeds" with rc=0, but the
    # directory entry is gone), producing the exact symptom the
    # operator hit: "ffmpeg rc=0 but output missing". R3's race
    # guard already protects the download endpoint; this lands the
    # equivalent guard on the polling endpoint that was actually
    # destroying the file.
    from chalybclip.api._render_validation import is_servable_cached_mp4
    if rendered.exists():
        if is_servable_cached_mp4(rendered):
            return JSONResponse(
                {
                    "state": "ready",
                    "progress_pct": 100,
                    "error": None,
                    "download_url": (
                        f"/dashboard/clips/{clip_id}/download?quality={quality}"
                    ),
                }
            )
        # File exists but failed the sanity gate. Same two cases as
        # the download endpoint:
        #   - state == 'rendering' → ffmpeg is mid-write. Don't
        #     touch the file. Just surface current render progress
        #     and let the next poll re-check.
        #   - state != 'rendering' → debris from a pre-R5 crashed
        #     render. Unlink so the next download click triggers a
        #     clean render instead of seeing the same debris again.
        if clip.render_state != "rendering":
            try:
                rendered.unlink(missing_ok=True)
            except OSError:
                pass

    # Render Migration R6 — zombie detection. The background runner
    # SHOULD mark_render_failed before returning, but R5 + the outer
    # guard in render_clip_in_background only fix that going forward;
    # any clip whose background task crashed BEFORE the guard rolled
    # out (or got killed by Railway / OOM / process restart) is stuck
    # with state='rendering' and nobody to flip it. The UI then polls
    # this endpoint forever (the logs show > 100 polls in a single
    # minute) and the operator has no way to retry — the Download
    # button is wired to "is the state stuck"-anchored progress bar.
    #
    # If state has been 'rendering' for >= ZOMBIE_RENDERING_TIMEOUT_S
    # AND no cache file appeared (we'd have returned ready above),
    # we treat the render as dead. We don't fight the race: we DON'T
    # mark it failed here (that's a DB write on every status poll;
    # the next click on Download already calls mark_render_started
    # atomically so the operator can clear it themselves). We just
    # tell the UI what the state ACTUALLY is so it can surface a
    # retry button.
    if clip.render_state == "rendering" and clip.render_started_at:
        try:
            from datetime import UTC, datetime as _dt
            started = _dt.fromisoformat(clip.render_started_at)
            if started.tzinfo is None:
                started = started.replace(tzinfo=UTC)
            age_s = (_dt.now(UTC) - started).total_seconds()
            if age_s > _ZOMBIE_RENDERING_TIMEOUT_S:
                # Mark failed once so the retry path works without a
                # second click. Idempotent — if the background task
                # is somehow still alive and races us, it'll either
                # mark_render_ready or mark_render_failed too, and
                # the operator's next click handles it.
                try:
                    await ClipsRepo(db).mark_render_failed(
                        clip_id,
                        error=(
                            f"render timed out after {int(age_s)}s "
                            "without completing (background task "
                            "likely crashed)"
                        ),
                    )
                except Exception:  # noqa: BLE001 — best-effort
                    pass
                return JSONResponse(
                    {
                        "state": "failed",
                        "progress_pct": int(clip.render_progress_pct or 0),
                        "error": (
                            f"render timed out after {int(age_s)}s "
                            "without completing (background task "
                            "likely crashed) — click Retry"
                        ),
                        "retry_url": (
                            f"/dashboard/clips/{clip_id}/render-retry"
                            f"?quality={quality}"
                        ),
                    }
                )
        except (ValueError, TypeError):
            # Bad timestamp → ignore the zombie check, surface
            # current state.
            pass

    return JSONResponse(
        {
            "state": clip.render_state or "idle",
            "progress_pct": int(clip.render_progress_pct or 0),
            "error": clip.render_error,
            # No download_url until state="ready" — UI keeps the button
            # disabled.
        }
    )


@router.post("/clips/{clip_id}/render-retry")
async def clip_render_retry(
    request: Request,
    clip_id: str,
    quality: str = "1080",
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Render Migration T1 — UI "Retry" button on a failed render.

    Resets clip.render_state='idle' AND unlinks any cached render at
    the requested resolution so the next Download click forces a
    fresh background render. We don't enqueue the render here; the
    download endpoint is the single place that schedules
    BackgroundTasks. Operator clicks Retry → cache wiped + state
    goes idle → they click Download again → background render
    starts with the latest pipeline.

    Render Migration R17 — added the cache unlink. Without it, the
    cache-hit branch in clip_download fires before the state check
    and the operator gets the stale rendered file every time,
    regardless of how many Retry clicks land. R16's libass pipeline
    (and any future render-pipeline change) only ships if the cache
    actually misses.
    """
    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    original = Path(clip.path)
    # Wipe the cached MP4 for the requested resolution AND its sibling
    # debug / ass files so the next render starts clean.
    resolved_key, _w, _h = _resolve_export_resolution(
        getattr(request.state, "tenant_tier", "free") or "free",
        quality,
    )
    cache_path = original.parent / f"clip_render_{resolved_key}.mp4"
    sidecars = [
        cache_path,
        cache_path.with_suffix(".partial.mp4"),
        cache_path.with_suffix(".ass"),
    ]
    for p in sidecars:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
    await ClipsRepo(db).reset_render_state(clip_id)
    return JSONResponse(
        {
            "ok": True,
            "state": "idle",
            "unlinked": [str(p.name) for p in sidecars if not p.exists()],
        }
    )


@router.get("/clips/{clip_id}/download")
async def clip_download(
    request: Request,
    background_tasks: BackgroundTasks,
    clip_id: str,
    quality: str = "1080",
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Slice O.21 — download the clip's headless-Chrome rendered MP4.
    Slice J.2a — `quality` query param picks the export resolution
    (1080 / 2k / 4k). Free + pro tiers are capped at 1080; all_access
    can pick higher and pays the extra GPU time via the outbound
    usage report.

    Render Migration T1 — was inline; now backgrounded.
      - Cache hit (clip_render_<res>.mp4 on disk) → 200 FileResponse,
        immediate download.
      - Cache miss + state='idle' → flip state to 'rendering',
        schedule background runner, return 202 JSON status.
      - Cache miss + state='rendering' → 202 JSON status (the UI
        polls /render-status every ~1 s for progress).
      - Cache miss + state='failed' → 409 JSON status with error
        message + retry hint (operator hits POST /render-retry to
        reset to 'idle' then re-clicks Download).

    The exported MP4 is produced by Playwright recording the
    `/clips/<id>/render` page at the chosen viewport, then muxing the
    source clip's audio track lossless. By construction the output is
    pixel-identical to the editor preview — same browser engine
    renders both, just sized to the picked resolution.

    Cache key includes the resolution so 1080 and 4K versions of the
    same clip live as separate files — switching resolutions doesn't
    invalidate the other's cache.
    """
    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    original = Path(clip.path)

    # Resolve the export resolution against the tenant's tier.
    tenant_tier = getattr(request.state, "tenant_tier", "free") or "free"
    resolved_key, target_w, target_h = _resolve_export_resolution(
        tenant_tier, quality
    )
    # Cache file per-resolution so flipping between 1080 / 4K doesn't
    # require a re-record on every switch.
    rendered = original.parent / f"clip_render_{resolved_key}.mp4"

    # Cache hit — serve immediately, ignore the DB state column.
    #
    # Render Migration R3 — `.exists()` alone isn't enough: a partial
    # / truncated / 0-byte file from a crashed encode also passes
    # `.exists()`. Operators were getting these as Windows-codec-error
    # downloads (0xC00D36C4). is_servable_cached_mp4 adds a size
    # floor + ISO BMFF magic-byte gate. If the cache file fails the
    # gate, we unlink + fall through into the render-scheduling path
    # so the next response is either a clean cache miss (kicks off
    # background) or the 409 failed-state hint.
    from chalybclip.api._render_validation import is_servable_cached_mp4
    if rendered.exists():
        if is_servable_cached_mp4(rendered):
            pretty = f"chalybclip_{clip_id}.mp4"
            return FileResponse(
                path=rendered,
                media_type="video/mp4",
                filename=pretty,
                headers={
                    "Content-Disposition": f'attachment; filename="{pretty}"',
                },
            )
        # File exists but failed the sanity gate. Two cases:
        #
        #   a. state == 'rendering' — ffmpeg is mid-write. The file
        #      is currently small (< 1 MB) but growing; magic bytes
        #      were written first so a future check will pass. Do
        #      NOT evict — unlinking under Linux would yank the
        #      directory entry from underneath the running ffmpeg
        #      and orphan the inode; on Windows it would fail with
        #      a sharing violation. Just return 202 and let the UI
        #      keep polling.
        #
        #   b. state != 'rendering' — debris from a crashed render
        #      (the R1 + R2 fixes should prevent this going forward;
        #      this branch covers pre-fix debris already on disk).
        #      Log, unlink, reset state, fall through to the
        #      schedule-new-render path.
        if clip.render_state == "rendering":
            return JSONResponse(
                status_code=202,
                content={
                    "state": "rendering",
                    "progress_pct": int(clip.render_progress_pct or 0),
                    "error": None,
                    "status_url": (
                        f"/dashboard/clips/{clip_id}/render-status"
                        f"?quality={quality}"
                    ),
                },
            )

        import structlog
        try:
            cache_bytes = rendered.stat().st_size
        except OSError:
            cache_bytes = -1
        structlog.get_logger("chalybclip.api.dashboard").warning(
            "clip_download.corrupt_cache_evicted",
            clip_id=clip_id,
            path=str(rendered),
            bytes=cache_bytes,
            prior_state=clip.render_state,
        )
        try:
            rendered.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            await ClipsRepo(db).reset_render_state(clip_id)
            # Mirror the reset onto the local Pydantic row so the
            # failed/rendering branches below don't read the stale
            # value and short-circuit before the schedule-new-render
            # path runs. ClipRow's mutability is default (not frozen).
            clip.render_state = "idle"
            clip.render_progress_pct = 0
            clip.render_error = None
        except Exception:  # noqa: BLE001 — best-effort
            pass

    # Failed render — surface the error + tell UI to show retry button.
    if clip.render_state == "failed":
        return JSONResponse(
            status_code=409,
            content={
                "state": "failed",
                "progress_pct": int(clip.render_progress_pct or 0),
                "error": clip.render_error or "render failed (no details)",
                "retry_url": f"/dashboard/clips/{clip_id}/render-retry?quality={quality}",
            },
        )

    # Render already in flight — return current status.
    #
    # Render Migration R6 — but if render has been 'rendering' for
    # longer than _ZOMBIE_RENDERING_TIMEOUT_S, the background task
    # almost certainly crashed and nobody marked failed. Auto-reset
    # to idle so this very click can schedule a fresh background
    # task (the code below) instead of returning 202 + hopeless
    # status poll. This is the recovery path for the two stuck
    # clips already on the operator's deploy.
    if clip.render_state == "rendering":
        is_zombie = False
        if clip.render_started_at:
            try:
                from datetime import UTC, datetime as _dt
                started = _dt.fromisoformat(clip.render_started_at)
                if started.tzinfo is None:
                    started = started.replace(tzinfo=UTC)
                age_s = (_dt.now(UTC) - started).total_seconds()
                if age_s > _ZOMBIE_RENDERING_TIMEOUT_S:
                    is_zombie = True
                    import structlog
                    structlog.get_logger(
                        "chalybclip.api.dashboard"
                    ).warning(
                        "clip_download.zombie_render_recovered",
                        clip_id=clip_id,
                        age_s=int(age_s),
                    )
            except (ValueError, TypeError):
                pass

        if is_zombie:
            try:
                await ClipsRepo(db).reset_render_state(clip_id)
                clip.render_state = "idle"
                clip.render_progress_pct = 0
                clip.render_error = None
            except Exception:  # noqa: BLE001
                pass
            # Fall through to the schedule-new-render branch below.
        else:
            return JSONResponse(
                status_code=202,
                content={
                    "state": "rendering",
                    "progress_pct": int(clip.render_progress_pct or 0),
                    "error": None,
                    "status_url": f"/dashboard/clips/{clip_id}/render-status?quality={quality}",
                },
            )

    # Idle (default) — schedule the background render. The repo
    # transition is atomic on (state != 'rendering') so a concurrent
    # request for the same clip is a no-op.
    if not original.exists():
        # Phase 2a — the local copy may have been reclaimed (retention,
        # or the pipeline ran on a worker). The render needs real local
        # bytes; pull the bucket copy back into place before giving up.
        from chalybclip.api._artifacts import rehydrate_clip
        if not await rehydrate_clip(
            tenant_id=tenant_id, clip_id=clip_id, clip_path=original
        ):
            raise HTTPException(
                status_code=404,
                detail=f"source clip file missing from disk: {original}",
            )

    await ClipsRepo(db).mark_render_started(clip_id)

    # Build the base URL the background recorder uses to dial back
    # into this server. Same logic the legacy inline path used.
    from chalybclip.settings import get_settings
    settings = get_settings()
    cookie_val = request.cookies.get("chalybclip_token", "")
    explicit_base = (settings.public_url or "").strip()
    if explicit_base and explicit_base != "http://localhost:8000":
        base_url = explicit_base
    else:
        scheme = (
            request.headers.get("x-forwarded-proto")
            or request.url.scheme
            or "https"
        )
        host = request.headers.get("host") or request.url.netloc
        base_url = f"{scheme}://{host}"

    from chalybclip.api._clip_render import render_clip_in_background
    background_tasks.add_task(
        render_clip_in_background,
        clip_id=clip_id,
        tenant_id=tenant_id,
        duration_s=float(clip.duration_s),
        audio_source_path=original,
        output_path=rendered,
        base_url=base_url,
        auth_cookie_value=cookie_val or None,
        width=target_w,
        height=target_h,
        db_path=resolve_db_target(settings),
    )
    return JSONResponse(
        status_code=202,
        content={
            "state": "rendering",
            "progress_pct": 0,
            "error": None,
            "status_url": f"/dashboard/clips/{clip_id}/render-status?quality={quality}",
        },
    )


async def _resolve_overlay_context(db: Database, clip: object) -> dict:
    """Compute the per-clip overlay context that BOTH the editor preview
    (clip_detail.html) and the headless capture (clip_render.html) need.

    Drift fix Step 2 — collapses three previously-divergent
    per-handler computations into a single source of truth:

      1. `face_zone` — coarse hint derived from the framing verdict's
         safe_crop_box.y. CSS rules `[data-face-zone="top|lower"]`
         shift overlays to dodge the face. Previously computed only
         in `clip_detail()`; absent in `clip_render_view()` so the
         export's CSS rules never fired (selectors had no attribute
         to match). Now computed once here.

      2. `_zone_profile` (brand_kit-aware fallback) — chooses between
         the per-clip `target_platform` override, then the
         brand_kit's default, then the catchall `"all"`. Previously
         the export had only the first half of the chain (no brand_kit
         fallback) so Kick-targeted brand kits exported without the
         Kick-platform position tweaks.

      3. `banner_url_display` — normalized "KICK.COM/HANDLE" string.
         Editor previously did this with inline Jinja (lines 3580-3607
         of clip_detail.html); export had a half-implementation in
         the handler. Now both surfaces call _format_kick_url with
         the same input + show the same output.

    Returns a dict the caller spreads into the template context:

      ov, _captions, _banner, _top_hook, banner_url_display,
      _zone_profile, _fit_mode, face_zone, render_watermark,
      show_live_pill, brand_kit
    """
    from chalybclip.branding import resolve_brand_kit_for_candidate
    from chalybclip.db import CandidatesRepo, TenantsRepo

    ov = clip.overlay_config or {}
    _captions = ov.get("captions") if isinstance(ov.get("captions"), dict) else {}
    _banner = ov.get("banner") if isinstance(ov.get("banner"), dict) else {}
    _top_hook = ov.get("top_hook") if isinstance(ov.get("top_hook"), dict) else {}

    # ---- face_zone (was: clip_detail-only) ----
    face_zone: str | None = None
    if clip.candidate_id:
        try:
            for cand in await CandidatesRepo(db).list_for_stream(clip.stream_id):
                if cand.id != clip.candidate_id:
                    continue
                ev = cand.evidence or {}
                framing = ev.get("framing") if isinstance(ev, dict) else None
                if not isinstance(framing, dict):
                    break
                crop = framing.get("safe_crop_box")
                if not isinstance(crop, dict):
                    break
                cy = crop.get("y")
                if isinstance(cy, int | float):
                    cy_f = float(cy)
                    if cy_f < 0.10:
                        face_zone = "top"
                    elif cy_f < 0.35:
                        face_zone = "mid"
                    else:
                        face_zone = "lower"
                break
        except Exception:  # noqa: BLE001 — face_zone is best-effort
            face_zone = None

    # ---- brand_kit (used by zone_profile fallback + watermark) ----
    brand_kit = None
    try:
        brand_kit = await resolve_brand_kit_for_candidate(
            db, stream_id=clip.stream_id, speaker_label=None,
        )
    except Exception:  # noqa: BLE001
        brand_kit = None

    # ---- _zone_profile (was: editor-only fallback chain) ----
    _zone_profile = ov.get("target_platform")
    if not _zone_profile and brand_kit is not None:
        _zone_profile = getattr(brand_kit, "target_platform", None)
    _zone_profile = _zone_profile or "all"

    # ---- _fit_mode (same in both before, kept here for completeness) ----
    _fit_mode = ov.get("fit_mode") or "smart_crop"

    # ---- banner_url_display (was: editor inline-Jinja vs export half-impl) ----
    banner_url_display = ""
    if _banner.get("enabled", False):
        _banner_platform = str(_banner.get("platform") or "kick").lower()
        _raw_url = str(_banner.get("url") or "")
        if _banner_platform == "kick":
            # _format_kick_url accepts any of: "aldo", "@aldo",
            # "kick.com/aldo", "https://kick.com/aldo",
            # "https://www.kick.com/aldo" and emits canonical
            # "KICK.COM/ALDO". Same helper the burn-step uses, so the
            # editor preview, the export, and the burned MP4 ALL agree
            # on what string lands in the banner.
            from chalybclip.clip.overlay_burn import _format_kick_url  # type: ignore[attr-defined]
            try:
                banner_url_display = _format_kick_url(_raw_url)
            except Exception:  # noqa: BLE001
                banner_url_display = _raw_url.upper()
            if not banner_url_display:
                # Empty input — show a placeholder so the operator sees
                # the banner shape before they type a URL.
                banner_url_display = "KICK.COM/YOURHANDLE"
        else:
            # Non-kick source (platform_band / minimal): the URL already
            # carries the platform-correct domain from
            # overlay_defaults.source_banner — never rewrite it to
            # KICK.COM. Just uppercase to match the burn.
            banner_url_display = _raw_url.strip().upper()

    show_live_pill = bool(_banner.get("live_badge", False))

    # ---- render_watermark (tier-gated, same in both before) ----
    render_watermark = True
    try:
        tenant = await TenantsRepo(db).get(clip.tenant_id)
        tier = (tenant.tier if tenant else "free") or "free"
        if tier != "free":
            if brand_kit is not None and not getattr(
                brand_kit, "show_chalybclip_credit", True,
            ):
                render_watermark = False
    except Exception:  # noqa: BLE001
        render_watermark = True

    return {
        "ov": ov,
        "_captions": _captions,
        "_banner": _banner,
        "_top_hook": _top_hook,
        "banner_url_display": banner_url_display,
        "_zone_profile": _zone_profile,
        "_fit_mode": _fit_mode,
        "face_zone": face_zone,
        "render_watermark": render_watermark,
        "show_live_pill": show_live_pill,
        "brand_kit": brand_kit,
    }


@router.get("/clips/{clip_id}/render", response_class=HTMLResponse)
async def clip_render_view(
    request: Request,
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Slice O.19 — minimal render-mode page for headless capture.

    Returns a stripped-down HTML page rendering ONLY the .nc-preview
    frame at 1080×1920 native. No editor chrome (nav, sidebar, tabs,
    controls). Playwright (slice O.20) opens this URL at viewport
    1080×1920 and screen-records the playback. What you see here IS
    what gets exported — by construction, no separate burn pipeline.

    Drift fix Step 2 — uses the shared _resolve_overlay_context helper
    so face_zone / zone_profile / banner_url / brand_kit threading
    matches the editor's clip_detail handler bit-for-bit. The
    `_overlay_styles.html` and `_overlay_markup.html` partials in
    this template's render carry the unified overlay CSS + markup
    that previously had drifted between the two surfaces.

    Tenant-scoped: the underlying ClipsRepo.get filters by current
    tenant context, so cross-tenant clip-id guessing 404s.
    """
    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")

    overlay_ctx = await _resolve_overlay_context(db, clip)
    return templates.TemplateResponse(
        request,
        "clip_render.html",
        {"clip": clip, **overlay_ctx},
    )


@router.get("/clips/{clip_id}/media")
async def clip_media(
    clip_id: str,
    source: str = "original",
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Stream the cut MP4 for inline <video> playback on the clip detail page.

    Slice M.6 — defaults to the ORIGINAL `clip.mp4` (no burns).
    Operator-reported bug: the editor was serving `clip_final.mp4`
    (which has captions BAKED INTO PIXELS) while the live preview
    ALSO renders styled overlay captions on top — operators saw
    every caption line twice: the burned plain-white one underneath
    + the karaoke-styled overlay on top.

    The editor's job is to compose; the burn is what we SHIP. To see
    the burned-final pixels, pass `?source=final` (used only by the
    "Final" preview-mode tab, if/when wired). Everywhere else gets
    the source so the styled overlays don't conflict with anything.

    Returns 404 if the clip row is missing or the on-disk file disappeared
    (e.g., out/ was nuked between runs). Tenant-bound so one tenant can't
    fetch another's clip even by guessing the id.
    """
    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    original = Path(clip.path)
    final = original.parent / "clip_final.mp4"

    # Phase 2a — the Final tab's auto-burn below needs the original bytes
    # on local disk. If retention (or a worker-run pipeline) reclaimed
    # them, pull the bucket copy back before deciding what to serve.
    if source == "final" and not final.exists() and not original.exists():
        from chalybclip.api._artifacts import rehydrate_clip
        await rehydrate_clip(
            tenant_id=tenant_id, clip_id=clip_id, clip_path=original
        )

    # Slice O.13 — when the editor's "Final" tab requests the burned
    # version and it's missing on disk, regenerate on the fly using
    # the saved overlay_config. Guarantees the Final tab is never a
    # broken video element. Same auto-burn pattern as the download
    # endpoint (slice O.12).
    if source == "final" and not final.exists() and original.exists():
        try:
            cfg = clip.overlay_config or {}
            await _burn_overlays_for_clip(
                db=db, clip_id=clip_id, overlay_config=cfg
            )
        except Exception:  # noqa: BLE001 — non-fatal, fall back below
            pass

    if source == "final" and final.exists():
        clip_path = final
    else:
        # Default: ORIGINAL (no captions burned in pixels). Fall back
        # to final only if the original is missing for some reason.
        clip_path = original if original.exists() else final
    if not clip_path.exists():
        # Phase 2a — no local copy at all: serve the bucket copy of the
        # cut original via redirect (no bytes through the web box). The
        # redirect is minted blind — a missing object 404s at the bucket
        # exactly like the local miss 404s here.
        from chalybclip.api._artifacts import artifact_redirect_url
        from chalybclip.integrations.storage import clip_media_key

        url = await artifact_redirect_url(clip_media_key(tenant_id, clip_id))
        if url is not None:
            return RedirectResponse(url=url, status_code=302)
        raise HTTPException(
            status_code=404,
            detail=f"clip file missing from disk: {clip_path}",
        )
    return FileResponse(
        path=clip_path,
        media_type="video/mp4",
        # Don't set Content-Disposition: attachment — we want inline playback.
        filename=clip_path.name,
    )


@router.get("/clips/{clip_id}/thumbnail")
async def clip_thumbnail(
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Serve the per-clip thumbnail JPEG for the inbox clip cards.

    Returns 404 when the clip row has no `thumbnail_frame_path` (the
    pipeline skipped the picker step) or when the file is gone from disk.
    """
    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    if not clip.thumbnail_frame_path:
        raise HTTPException(status_code=404, detail="no thumbnail for this clip")
    thumb_path = Path(clip.thumbnail_frame_path)
    if not thumb_path.exists():
        # Phase 2a — serve the bucket copy when the local one is gone
        # (retention reclaim, or the pipeline ran on a worker).
        from chalybclip.api._artifacts import artifact_redirect_url
        from chalybclip.integrations.storage import clip_thumbnail_key

        url = await artifact_redirect_url(
            clip_thumbnail_key(tenant_id, clip_id)
        )
        if url is not None:
            return RedirectResponse(url=url, status_code=302)
        raise HTTPException(
            status_code=404,
            detail=f"thumbnail file missing from disk: {thumb_path}",
        )
    return FileResponse(
        path=thumb_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=300"},
    )


@router.get("/clips/{clip_id}/intelligence.json")
async def clip_intelligence(
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Per-clip intelligence markers (audio peaks / scene cuts /
    laughter reactions / chat-heat spikes / face-emotion changes).

    Aggregated read-only across the existing transcripts +
    visual_signals + chat_replay surfaces. Returns a flat
    `{markers: [{kind, ts, score, label}, ...]}` JSON shape — no
    caching needed (the underlying tables are already indexed by
    stream_id and the aggregation is cheap).
    """
    import json as _json

    from chalybclip.clip import compute_intelligence

    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    markers = await compute_intelligence(db, clip_id=clip_id)

    # Slice M.1 — surface the voice-trigger phrase that fired this
    # clip as a first-class timeline marker. Operator reported the
    # missing signal: "en el video dije clipea esto y no lo marco
    # abajo como que escucho al usuario y como genero el clip basado
    # en eso". The candidate's evidence carries the phrase + the
    # timestamp; the timeline now shows it as a green dot labeled
    # 🎯 with the matched phrase so the operator FEELS the AI's
    # detection.
    #
    # Slice M.3 — `compute_intelligence` now ALSO scans the clip's
    # transcript for trigger phrases (independent of how the candidate
    # was generated). When that scan already found the phrase, this
    # candidate-evidence path is a duplicate — skip it.
    transcript_already_found_trigger = any(
        m.kind == "voice_trigger" for m in markers
    )
    trigger_marker: dict[str, object] | None = None
    if clip.candidate_id and not transcript_already_found_trigger:
        for cand in await CandidatesRepo(db).list_for_stream(clip.stream_id):
            if cand.id == clip.candidate_id:
                ev = cand.evidence or {}
                if isinstance(ev, dict):
                    phrase = ev.get("phrase")
                    if isinstance(phrase, str) and phrase:
                        # Convert absolute stream timestamp to clip-
                        # relative. cand.ts is the phrase START in the
                        # stream; the clip starts at clip.start_s.
                        # (DB row uses `ts`; the in-memory detector
                        # model `chalybclip.detect.models.Candidate`
                        # uses `timestamp` — easy mix-up that already
                        # bit this endpoint once with a 500.)
                        clip_rel = max(0.0, float(cand.ts) - clip.start_s)
                        kind_label = (
                            "Voice trigger fired"
                            if ev.get("trigger_kind") != "retroactive"
                            else "Retroactive trigger fired"
                        )
                        trigger_marker = {
                            "kind": "voice_trigger",
                            "ts": clip_rel,
                            "score": float(ev.get("confidence") or 0.9),
                            "label": f"🎯 {kind_label}: \"{phrase}\"",
                        }
                break

    out_markers: list[dict[str, object]] = [
        {
            "kind": m.kind,
            "ts": m.ts,
            "score": m.score,
            "label": m.label,
        }
        for m in markers
    ]
    # Sort by ts and put the trigger first so it leads the legend.
    if trigger_marker is not None:
        out_markers.insert(0, trigger_marker)

    return Response(
        content=_json.dumps(
            {
                "markers": out_markers,
                "duration_s": clip.duration_s,
            }
        ),
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/clips/{clip_id}/captions.json")
async def clip_captions(
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Word-level captions sliced to the clip window — slice F.7-F.

    Returns the same caption lines the renderer's ASS pass burns into
    the MP4, so the live editor preview and the published clip
    match exactly. Each line carries:

      - ts / end_ts (clip-relative seconds)
      - text (joined line)
      - words: [{ts, end_ts, text, emphasis}]
      - emphasis: strongest emphasis tag on the line

    Empty {lines: []} when the transcript has no word-level data
    overlapping the clip window — the preview gracefully renders the
    fallback placeholder copy in that case.
    """
    import json as _json

    from chalybclip.clip import captions_for_clip, lines_to_json
    from chalybclip.db import TranscriptsRepo

    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    transcript = await TranscriptsRepo(db).get(clip.stream_id)

    # Slice F.7-G — diagnostics block so the editor's empty-state can
    # explain *why* no captions are showing instead of just hinting
    # "re-run the pipeline" (the user asked: "i keep getting this
    # error even though im talking in spanish"). The four real
    # failure modes we want to distinguish:
    #
    #   1. transcript hasn't been written yet  → run pipeline
    #   2. transcript is present but EMPTY     → VAD ate everything,
    #                                            or audio extract was silent
    #   3. transcript has segments but no word-level data
    #                                          → pre-F.7-F run, re-transcribe
    #   4. words exist but none overlap THIS clip window
    #                                          → clip cut outside the
    #                                            transcribed span
    #
    # The editor's JS turns the diagnostics into a one-line empty-state.
    diag: dict[str, object] = {
        "transcript_present": transcript is not None,
        "language": transcript.language if transcript else None,
        "transcript_segment_count": 0,
        "transcript_word_count": 0,
        "transcript_span_s": None,
        "clip_window_s": [clip.start_s, clip.end_s],
        "reason": "no_transcript",
    }
    if transcript is None:
        body = {"lines": [], "duration_s": clip.duration_s, "diagnostics": diag}
    else:
        # Inspect the raw transcript shape without re-running the chunker
        # so the diagnostics survive even when captions_for_clip returns [].
        try:
            raw_segments = _json.loads(transcript.segments_json or "[]")
        except _json.JSONDecodeError:
            raw_segments = []
        if isinstance(raw_segments, list):
            diag["transcript_segment_count"] = len(raw_segments)
            total_words = 0
            min_ts: float | None = None
            max_ts: float | None = None
            for seg in raw_segments:
                if not isinstance(seg, dict):
                    continue
                seg_words = seg.get("words")
                if isinstance(seg_words, list):
                    total_words += len(seg_words)
                # transcript timestamps are stream-relative ("ts"/"end_ts"
                # from Pydantic, "start"/"end" from legacy fixtures).
                start_v = seg.get("ts", seg.get("start"))
                end_v = seg.get("end_ts", seg.get("end"))
                if isinstance(start_v, int | float):
                    min_ts = float(start_v) if min_ts is None else min(min_ts, float(start_v))
                if isinstance(end_v, int | float):
                    max_ts = float(end_v) if max_ts is None else max(max_ts, float(end_v))
            diag["transcript_word_count"] = total_words
            if min_ts is not None and max_ts is not None:
                diag["transcript_span_s"] = [min_ts, max_ts]

        lines = captions_for_clip(
            transcript.segments_json or "",
            clip_start_s=clip.start_s,
            clip_end_s=clip.end_s,
        )
        # Classify the failure mode for the empty-state hint.
        # Slice N.2 — "window_outside_transcript" was a catch-all
        # that incorrectly fired even when the clip window WAS inside
        # the transcript span but just happened to have no spoken
        # words (silence between sentences, music-only stretches,
        # crossfade gaps). Now we actually CHECK the span boundaries
        # before claiming the window is outside; otherwise the
        # diagnostic is simply "silent stretch" — the clip's audio
        # band had no transcribable speech.
        if lines:
            diag["reason"] = "ok"
        elif diag["transcript_segment_count"] == 0:
            diag["reason"] = "transcript_empty"
        elif diag["transcript_word_count"] == 0:
            diag["reason"] = "no_word_timestamps"
        else:
            span = diag.get("transcript_span_s")
            if (
                isinstance(span, list)
                and len(span) == 2
                and (clip.end_s < span[0] or clip.start_s > span[1])
            ):
                diag["reason"] = "window_outside_transcript"
            else:
                # Clip window IS inside the transcript span but no
                # words fall in it — the audio band itself is silent /
                # non-verbal across this stretch.
                diag["reason"] = "silent_stretch"
        body = {
            "lines": lines_to_json(lines),
            "duration_s": clip.duration_s,
            "diagnostics": diag,
        }
    return Response(
        content=_json.dumps(body),
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/clips/{clip_id}/waveform.json")
async def clip_waveform(
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Serve the per-clip audio waveform as a list of normalized peaks.

    Computed on first request via ffmpeg PCM extraction, then cached
    next to the clip MP4 as `waveform.json` for instant subsequent
    loads. Returns `[]` (200, not 404) when extraction fails so the
    editor's scrubber gracefully degrades to a flat baseline rather
    than spamming the console with 404s on every clip without audio.
    """
    import json as _json

    from chalybclip.clip import load_or_compute_waveform

    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    clip_path = Path(clip.path)
    if not clip_path.exists():
        # Phase 2a — waveform extraction needs local bytes; pull the
        # bucket copy back before degrading to the flat baseline.
        from chalybclip.api._artifacts import rehydrate_clip
        if not await rehydrate_clip(
            tenant_id=tenant_id, clip_id=clip_id, clip_path=clip_path
        ):
            return Response(
                content="[]",
                media_type="application/json",
                headers={"Cache-Control": "no-store"},
            )
    peaks = load_or_compute_waveform(clip_path)
    return Response(
        content=_json.dumps(peaks),
        media_type="application/json",
        # The on-disk cache means we can be aggressive with browser
        # caching too; the file changes only when ops manually delete it.
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/streams/{stream_id}/source")
async def stream_source(
    stream_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> FileResponse:
    """Serve the original uploaded/downloaded source video for preview."""
    stream = await StreamsRepo(db).get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="stream not found")
    src = Path(stream.source_video_path)
    if not src.exists():
        raise HTTPException(
            status_code=404,
            detail=f"source video missing from disk: {src}",
        )
    return FileResponse(
        path=src,
        media_type="video/mp4",
        filename=src.name,
    )


@router.post(
    "/streams/{stream_id}/delete",
    dependencies=[Depends(require_full_scope)],
)
async def stream_delete(
    stream_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Slice O.11 — hard-delete a stream + cascade everything beneath it.

    Two-phase: DB cascade first (FKs handle clips/candidates/variants/
    publish_jobs/etc), then best-effort filesystem cleanup of the
    per-stream output directory derived from `source_video_path`.

    Blocks deletion while the pipeline is running so we don't kneecap
    a worker mid-flight (the cascade would yank rows it's still
    writing to). Ingested / done / failed streams are fair game.
    """
    repo = StreamsRepo(db)
    stream = await repo.get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="stream not found")
    if stream.status == "running":
        raise HTTPException(
            status_code=409,
            detail="stream is currently running — wait for it to finish or fail before deleting",
        )

    # A scheduled Zernio post fetches our signed media URL only when its
    # slot arrives — deleting the stream now cascades the clip row away
    # and that fetch 404s days later (the user gets a "post failed"
    # email with no visible cause). Refuse and point at the pending
    # posts instead of silently arming the time bomb.
    pending = await ZernioPublishesRepo(db).pending_for_stream(stream_id)
    if pending:
        posts = ", ".join(p["post_id"] for p in pending[:5])
        raise HTTPException(
            status_code=409,
            detail=(
                f"{len(pending)} scheduled post(s) still use this stream's "
                f"clips ({posts}). Cancel or let them publish in the "
                f"Publish Center first — deleting now would make them fail "
                f"at publish time."
            ),
        )

    # Resolve the on-disk stream directory BEFORE the DB row goes
    # away. The convention is `<output_dir>/<stream_id>/source.<ext>`,
    # so the parent of source_video_path IS the per-stream dir. Using
    # the row's own path is safer than reconstructing — it survives
    # output_dir config drift between ingest time and delete time.
    stream_dir: Path | None = None
    try:
        src = Path(stream.source_video_path).resolve()
        parent = src.parent
        # Guard: only nuke the directory if its name is the stream id.
        # Belt-and-suspenders against a misconfigured stream pointing
        # at, say, `/Users/picasso/Movies/source.mp4` whose parent is
        # NOT a disposable stream dir.
        if parent.name == stream_id:
            stream_dir = parent
    except Exception:  # noqa: BLE001 — path parsing failures are non-fatal
        stream_dir = None

    # Phase 2a — collect this stream's clip ids BEFORE the cascade so we
    # can drop their bucket copies too. Best-effort, like the fs cleanup.
    clip_ids: list[str] = []
    try:
        clip_ids = [c.id for c in await ClipsRepo(db).list_for_stream(stream_id)]
    except Exception:  # bucket cleanup is best-effort
        clip_ids = []

    deleted = await repo.delete(stream_id)
    if not deleted:
        # Shouldn't happen — we just fetched the row — but if a parallel
        # delete beat us, treat it as a no-op success.
        return RedirectResponse(url="/dashboard/streams", status_code=303)

    if clip_ids:
        try:
            from chalybclip.integrations.storage import (
                build_artifact_store,
                clip_key_family,
            )
            from chalybclip.settings import get_settings

            _store = build_artifact_store(get_settings())
            if _store is not None:
                for cid in clip_ids:
                    for key in clip_key_family(tenant_id, cid):
                        await _store.delete(key=key)
        except Exception:  # rows are gone; bucket cleanup stays best-effort
            import structlog
            structlog.get_logger("chalybclip.api.dashboard").warning(
                "stream_delete.bucket_cleanup_failed", stream_id=stream_id
            )

    # Best-effort filesystem cleanup. Failures here don't fail the
    # delete — the DB row is already gone, the operator's goal is
    # met. Worst case there's a stale folder of mp4s the operator
    # can rm by hand.
    if stream_dir and stream_dir.exists():
        import shutil
        try:
            shutil.rmtree(stream_dir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass

    await EventsRepo(db).emit(
        type="stream.deleted",
        payload={"stream_id": stream_id, "fs_dir": str(stream_dir) if stream_dir else None},
    )
    return RedirectResponse(url="/dashboard/streams?deleted=1", status_code=303)


@router.get("/calibration", response_class=HTMLResponse)
async def calibration_view(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Per-platform rescore-vs-views calibration table.

    Pearson r over the paired (rescore_score, views) values for each
    platform's last 30 days. Surfaces the data the operator needs to
    decide whether the scoring system has earned the right to auto-
    publish (see PHASE_3.md hard rules).
    """
    from chalybclip.metrics import compute_calibration

    reports = []
    for platform in ("youtube", "tiktok", "instagram", "buffer"):
        reports.append(await compute_calibration(db, platform=platform))
    return templates.TemplateResponse(
        request,
        "calibration.html",
        {"reports": reports},
    )


# Slice H.1 — manual status transitions removed.
# Status now derives entirely from the lifecycle handlers:
#   - cut/ready_for_review → approved : via clip_overlay_finalize ("Ship")
#   - any              → rejected : via clip_reject ("Reject & close")
#   - approved         → published: via the publish worker after a
#                                   successful platform post
# The legacy PATCH /clips/{id}/status endpoint + dashboard panel are
# gone. The full lifecycle table still lives in
# chalybclip.api.routers.clips._VALID_STATUS_TRANSITIONS for the
# bearer-token JSON API path (PATCH /api/clips/{id}/status) — that
# stays available for headless automations; only the dashboard's
# manual-clicker is removed.


# ---------- Personas ----------


@router.get("/personas", response_class=HTMLResponse)
async def personas_list(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    personas = await PersonasRepo(db).list_for_tenant()
    return templates.TemplateResponse(request, "personas.html", {"personas": personas})


@router.post("/personas", dependencies=[Depends(require_full_scope)])
async def personas_create(
    request: Request,
    id: str = Form(...),
    name: str = Form(...),
    primary_language: str = Form(...),
    voice_prompt: str = Form(...),
    target_languages: str = Form(""),
    routing_tags: str = Form(""),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    repo = PersonasRepo(db)
    if await repo.get(id) is not None:
        raise HTTPException(status_code=409, detail=f"persona {id!r} already exists")
    await repo.create(
        persona_id=id,
        name=name,
        primary_language=primary_language,
        voice_prompt=voice_prompt,
        target_languages=_split_csv(target_languages),
        routing_tags=_split_csv(routing_tags),
    )
    return RedirectResponse(url="/dashboard/personas", status_code=303)


@router.get("/personas/{persona_id}/edit", response_class=HTMLResponse)
async def persona_edit_form(
    request: Request,
    persona_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    persona = await PersonasRepo(db).get(persona_id)
    if persona is None:
        raise HTTPException(status_code=404, detail="persona not found")
    return templates.TemplateResponse(request, "persona_edit.html", {"persona": persona})


@router.post(
    "/personas/{persona_id}",
    dependencies=[Depends(require_full_scope)],
)
async def persona_edit_submit(
    request: Request,
    persona_id: str,
    name: str = Form(...),
    primary_language: str = Form(...),
    voice_prompt: str = Form(...),
    target_languages: str = Form(""),
    routing_tags: str = Form(""),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    repo = PersonasRepo(db)
    if await repo.get(persona_id) is None:
        raise HTTPException(status_code=404, detail="persona not found")
    await repo.upsert(
        persona_id=persona_id,
        name=name,
        primary_language=primary_language,
        voice_prompt=voice_prompt,
        target_languages=_split_csv(target_languages),
        routing_tags=_split_csv(routing_tags),
    )
    return RedirectResponse(url="/dashboard/personas", status_code=303)


# ---------- LLM ----------


@router.get("/llm-calls", response_class=HTMLResponse)
async def llm_calls_view(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    # LLM-calls ledger shows raw USD per call — admin/economics only, never
    # a creator surface. 404 to non-admins.
    if not getattr(request.state, "is_admin", False):
        raise HTTPException(status_code=404, detail="not found")

    from chalybclip.cost import compute_cost_projection

    calls = await LLMCallsRepo(db).list_for_tenant(limit=200)
    projection = await compute_cost_projection(db)
    total_usd = sum(c.cost_usd_micros for c in calls) / 1_000_000.0
    return templates.TemplateResponse(
        request,
        "llm_calls.html",
        {"calls": calls, "total_usd": total_usd, "projection": projection},
    )


@router.get("/settings/llm", response_class=HTMLResponse)
async def llm_settings_view(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
) -> Response:
    from chalybclip.llm.config import load_llm_config

    cfg = load_llm_config()
    return templates.TemplateResponse(
        request,
        "llm_settings.html",
        {"providers": cfg.providers, "routing": cfg.routing},
    )


# ---------- Site / landing settings (admin) ----------
#
# Slice O.57 — operator-editable landing values that we don't want to
# hard-code in i18n.py + redeploy for. Currently the ChalybClip price shown
# in the landing's "what does an editor cost" comparison; the backing
# table (platform_settings) is a generic key/value store so future
# site-wide knobs land here too.
#
# Admin-gated the same way as the other operator-only pages
# (_health/diarize, _health/queue): non-admins get a 404 so the page
# isn't even discoverable. The public landing reads these keys at render
# time (see api/app.py root()).

_LANDING_PRICE_ES_KEY = "landing_price_es"
_LANDING_PRICE_EN_KEY = "landing_price_en"


@router.get("/settings/site", response_class=HTMLResponse, include_in_schema=False)
async def site_settings_view(
    request: Request,
    saved: int = 0,
    db: Database = Depends(get_db),
) -> Response:
    """Admin-only site settings form. 404 for non-admins."""
    if not getattr(request.state, "is_admin", False):
        raise HTTPException(status_code=404, detail="not found")
    values = await PlatformSettingsRepo(db).get_many(
        [_LANDING_PRICE_ES_KEY, _LANDING_PRICE_EN_KEY]
    )
    return templates.TemplateResponse(
        request,
        "site_settings.html",
        {
            "price_es": values.get(_LANDING_PRICE_ES_KEY, ""),
            "price_en": values.get(_LANDING_PRICE_EN_KEY, ""),
            "saved": bool(saved),
        },
    )


@router.post(
    "/settings/site",
    include_in_schema=False,
    dependencies=[Depends(require_full_scope)],
)
async def site_settings_save(
    request: Request,
    price_es: str = Form(""),
    price_en: str = Form(""),
    db: Database = Depends(get_db),
) -> Response:
    """Persist the landing price strings. Empty values clear back to the
    i18n placeholder on the landing. Admin-only; 404 for non-admins."""
    if not getattr(request.state, "is_admin", False):
        raise HTTPException(status_code=404, detail="not found")
    repo = PlatformSettingsRepo(db)
    await repo.set(_LANDING_PRICE_ES_KEY, price_es.strip())
    await repo.set(_LANDING_PRICE_EN_KEY, price_en.strip())
    return RedirectResponse(url="/dashboard/settings/site?saved=1", status_code=303)


# ---------- Brand kits (voice-markers spec slice C.3) ----------


def _parse_phrase_list(value: str) -> list[str]:
    """Split a textarea value into a phrase list (one per line, trimmed)."""
    return [line.strip() for line in value.splitlines() if line.strip()]


@router.get("/sources", response_class=HTMLResponse)
async def sources_list(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Connected channels — auto-ingest sources (YouTube/Twitch/Kick)."""
    watches = await ChannelWatchesRepo(db).list_for_tenant()
    personas = await PersonasRepo(db).list_for_tenant()
    return templates.TemplateResponse(
        request,
        "sources.html",
        {"watches": watches, "personas": personas},
    )


@router.post("/sources", dependencies=[Depends(require_full_scope)])
async def sources_add(
    request: Request,
    channel_url: str = Form(...),
    persona_id: str = Form(...),
    platform: str = Form(""),
    language: str = Form(""),
    label: str = Form(""),
    max_per_poll: int = Form(3),
    polls_per_day: int = Form(1),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    from chalybclip.db import IntegrityViolation
    from chalybclip.ingest.service import detect_platform

    resolved = platform.strip() or detect_platform(channel_url)
    try:
        await ChannelWatchesRepo(db).create(
            platform=resolved,
            channel_url=channel_url.strip(),
            persona_id=persona_id,
            channel_label=label.strip() or None,
            language=language.strip() or None,
            max_per_poll=int(max_per_poll),
            polls_per_day=int(polls_per_day),
        )
    except IntegrityViolation:
        # UNIQUE(tenant_id, channel_url): this tenant already watches this
        # exact channel URL. A re-add (or a double-submit) must not 500 —
        # bounce back with a friendly banner instead of crashing.
        return RedirectResponse(
            url="/dashboard/sources?error=already_watching", status_code=303,
        )
    await EventsRepo(db).emit(
        type="channel_watch.created",
        payload={"channel_url": channel_url.strip(), "platform": resolved},
    )
    return RedirectResponse(url="/dashboard/sources?added=1", status_code=303)


@router.post("/sources/{watch_id}/toggle", dependencies=[Depends(require_full_scope)])
async def sources_toggle(
    request: Request,
    watch_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    repo = ChannelWatchesRepo(db)
    existing = await repo.get(watch_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="channel watch not found")
    await repo.set_enabled(watch_id, not existing.enabled)
    return RedirectResponse(url="/dashboard/sources", status_code=303)


@router.post("/sources/{watch_id}/delete", dependencies=[Depends(require_full_scope)])
async def sources_delete(
    request: Request,
    watch_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    await ChannelWatchesRepo(db).delete(watch_id)
    return RedirectResponse(url="/dashboard/sources", status_code=303)


@router.post("/sources/{watch_id}/schedule", dependencies=[Depends(require_full_scope)])
async def sources_schedule(
    request: Request,
    watch_id: str,
    polls_per_day: int = Form(...),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Change a channel's poll cadence (times/day). Clamped to 1-96."""
    repo = ChannelWatchesRepo(db)
    if await repo.get(watch_id) is None:
        raise HTTPException(status_code=404, detail="channel watch not found")
    await repo.set_polls_per_day(watch_id, max(1, min(int(polls_per_day), 96)))
    return RedirectResponse(url="/dashboard/sources", status_code=303)


@router.post("/sources/{watch_id}/poll", dependencies=[Depends(require_full_scope)])
async def sources_poll_now(
    request: Request,
    watch_id: str,
    background_tasks: BackgroundTasks,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Scan one watched channel for new VODs right now, ignoring its cadence.

    The listing + per-VOD download can take minutes, so the poll runs as a
    background task and the page redirects immediately with a 'scanning'
    banner; new VODs appear on the streams page as they finish. Uses the
    shared app DB + dispatcher (the request-scoped ones don't outlive the
    response) and `poll_channel_watches`' own per-tenant binding, scoped to
    this single watch via `watch_id` + `respect_schedule=False`.
    """
    from chalybclip.channels import make_channel_ingest_callback, poll_channel_watches
    from chalybclip.settings import get_settings

    if await ChannelWatchesRepo(db).get(watch_id) is None:
        raise HTTPException(status_code=404, detail="channel watch not found")

    app_db = request.app.state.db
    dispatcher = request.app.state.job_dispatcher
    output_dir = Path(get_settings().default_output_dir)
    ingest_callback = make_channel_ingest_callback(
        app_db, dispatcher, output_dir=output_dir
    )

    async def _run() -> None:
        await poll_channel_watches(
            app_db,
            ingest_callback=ingest_callback,
            tenant_id=tenant_id,
            watch_id=watch_id,
            respect_schedule=False,
        )

    background_tasks.add_task(_run)
    return RedirectResponse(url="/dashboard/sources?scanning=1", status_code=303)


@router.get("/brand-kits", response_class=HTMLResponse)
async def brand_kits_list(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    kits = await BrandKitsRepo(db).list_for_tenant()
    return templates.TemplateResponse(
        request,
        "brand_kits.html",
        {"kits": kits},
    )


@router.get("/brand-kits/new", response_class=HTMLResponse)
async def brand_kits_new_form(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
) -> Response:
    from chalybclip.branding import builtin_presets, preset_choices

    return templates.TemplateResponse(
        request,
        "brand_kit_edit.html",
        {
            "kit": None,
            "caption_style": builtin_presets()["karaoke_pop"],
            "caption_preset_choices": preset_choices(),
        },
    )


@router.post("/brand-kits", dependencies=[Depends(require_full_scope)])
async def brand_kits_create(
    request: Request,
    name: str = Form(...),
    primary_color: str = Form("#FF3366"),
    accent_color: str = Form("#FFD700"),
    text_color: str = Form("#FFFFFF"),
    font_family: str = Form("Inter"),
    font_weight: int = Form(800),
    default_layout: str = Form("pip"),
    is_default: str = Form(""),
    handle_tiktok: str = Form(""),
    handle_youtube: str = Form(""),
    handle_instagram: str = Form(""),
    handle_kick: str = Form(""),
    caption_preset: str = Form("karaoke_pop"),
    auto_publish_enabled: str = Form(""),
    auto_publish_platforms: str = Form(""),
    auto_publish_delay_min: int = Form(60),
    forward_phrases: str = Form(""),
    retroactive_phrases: str = Form(""),
    show_chalybclip_outro: str = Form("1"),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Create a brand kit. Phrases come in as newline-separated textareas;
    color pickers and checkboxes ride the standard HTML form contract."""
    from chalybclip.branding.captions import _preset_by_id

    caption_style = _preset_by_id(caption_preset).model_dump()
    kit = await BrandKitsRepo(db).create(
        show_chalybclip_outro=bool(show_chalybclip_outro),
        name=name,
        primary_color=primary_color,
        accent_color=accent_color,
        text_color=text_color or "#FFFFFF",
        font_family=font_family or "Inter",
        font_weight=int(font_weight) if font_weight else 800,
        default_layout=default_layout or "pip",
        is_default=bool(is_default),
        handle_tiktok=handle_tiktok or None,
        handle_youtube=handle_youtube or None,
        handle_instagram=handle_instagram or None,
        handle_kick=handle_kick or None,
        caption_style=caption_style,
        auto_publish_enabled=bool(auto_publish_enabled),
        auto_publish_platforms=_split_csv(auto_publish_platforms),
        auto_publish_delay_min=int(auto_publish_delay_min) if auto_publish_delay_min else 60,
        custom_trigger_phrases=CustomTriggerPhrases(
            forward=_parse_phrase_list(forward_phrases),
            retroactive=_parse_phrase_list(retroactive_phrases),
        ),
    )
    return RedirectResponse(url=f"/dashboard/brand-kits/{kit.id}", status_code=303)


@router.get("/brand-kits/{kit_id}", response_class=HTMLResponse)
async def brand_kits_edit_form(
    request: Request,
    kit_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    from chalybclip.branding import caption_style_or_default, preset_choices
    from chalybclip.safety.policy import preset_for_overrides

    kit = await BrandKitsRepo(db).get(kit_id)
    if kit is None:
        raise HTTPException(status_code=404, detail="brand kit not found")
    caption_style = caption_style_or_default(kit.caption_style)
    return templates.TemplateResponse(
        request,
        "brand_kit_edit.html",
        {
            "kit": kit,
            "caption_style": caption_style,
            "caption_preset_choices": preset_choices(),
            "safety_preset": preset_for_overrides(kit.safety_policy),
        },
    )


@router.post(
    "/brand-kits/{kit_id}",
    dependencies=[Depends(require_full_scope)],
)
async def brand_kits_edit_submit(
    request: Request,
    kit_id: str,
    name: str = Form(...),
    primary_color: str = Form("#FF3366"),
    accent_color: str = Form("#FFD700"),
    text_color: str = Form("#FFFFFF"),
    font_family: str = Form("Inter"),
    font_weight: int = Form(800),
    default_layout: str = Form("pip"),
    caption_preset: str = Form("karaoke_pop"),
    handle_tiktok: str = Form(""),
    handle_youtube: str = Form(""),
    handle_instagram: str = Form(""),
    handle_kick: str = Form(""),
    auto_publish_enabled: str = Form(""),
    auto_publish_platforms: str = Form(""),
    auto_publish_delay_min: int = Form(60),
    forward_phrases: str = Form(""),
    retroactive_phrases: str = Form(""),
    safe_schedule_enabled: str = Form(""),
    content_timezone: str = Form("UTC"),
    safety_preset: str = Form("recommended"),
    show_chalybclip_outro: str = Form(""),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    from chalybclip.branding.captions import _preset_by_id
    from chalybclip.safety.policy import policy_overrides_for_preset

    repo = BrandKitsRepo(db)
    if await repo.get(kit_id) is None:
        raise HTTPException(status_code=404, detail="brand kit not found")
    caption_style = _preset_by_id(caption_preset).model_dump()

    # The raw per-platform JSON was replaced by a simple preset; turn it into
    # the override dict policy_for_kit consumes. "recommended" -> None means
    # "use the built-in defaults".
    safety_policy = policy_overrides_for_preset(safety_preset)

    await repo.update(
        kit_id,
        name=name,
        primary_color=primary_color,
        accent_color=accent_color,
        text_color=text_color,
        font_family=font_family,
        font_weight=int(font_weight),
        default_layout=default_layout,
        caption_style=caption_style,
        handle_tiktok=handle_tiktok,
        handle_youtube=handle_youtube,
        handle_instagram=handle_instagram,
        handle_kick=handle_kick,
        auto_publish_enabled=bool(auto_publish_enabled),
        auto_publish_platforms=_split_csv(auto_publish_platforms),
        auto_publish_delay_min=int(auto_publish_delay_min),
        custom_trigger_phrases=CustomTriggerPhrases(
            forward=_parse_phrase_list(forward_phrases),
            retroactive=_parse_phrase_list(retroactive_phrases),
        ),
        safe_schedule_enabled=bool(safe_schedule_enabled),
        content_timezone=content_timezone or "UTC",
        safety_policy=safety_policy,
        show_chalybclip_outro=bool(show_chalybclip_outro),
    )
    return RedirectResponse(url=f"/dashboard/brand-kits/{kit_id}", status_code=303)


@router.post(
    "/brand-kits/{kit_id}/default",
    dependencies=[Depends(require_full_scope)],
)
async def brand_kits_set_default(
    request: Request,
    kit_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    try:
        await BrandKitsRepo(db).set_default(kit_id)
    except ChalybClipError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return RedirectResponse(url="/dashboard/brand-kits", status_code=303)


@router.post(
    "/brand-kits/{kit_id}/delete",
    dependencies=[Depends(require_full_scope)],
)
async def brand_kits_delete(
    request: Request,
    kit_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    if await BrandKitsRepo(db).get(kit_id) is None:
        raise HTTPException(status_code=404, detail="brand kit not found")
    await BrandKitsRepo(db).delete(kit_id)
    return RedirectResponse(url="/dashboard/brand-kits", status_code=303)


def _brand_kit_asset_dir(output_dir: Path, kit_id: str) -> Path:
    """`<output_dir>/brand_kits/<kit_id>/` — slice D.3 asset root.

    Mirrors the per-stream layout so a future Storage abstraction
    (docs/production_deploy.md §3) can swap to S3/R2 by remapping a
    single prefix.
    """
    return output_dir / "brand_kits" / kit_id


@router.post(
    "/brand-kits/{kit_id}/generate-logo",
    dependencies=[Depends(require_full_scope)],
)
async def brand_kits_generate_logo(
    request: Request,
    kit_id: str,
    style_hint: str = Form("Minimal mark / monogram"),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Call Claude → sanitized SVG → save under brand_kits/<kit_id>/ →
    persist the relative URL + ai_* metadata on the brand kit row.

    Errors bubble up as a 500 with the provider message; the dashboard's
    next render shows the previously-generated logo unchanged (best-effort
    semantics — the kit isn't mutated until both the LLM call AND the disk
    write succeed)."""
    from chalybclip.branding import generate_logo, rasterize_svg_to_png
    from chalybclip.llm import LLMRouter, load_llm_config
    from chalybclip.settings import get_settings

    repo = BrandKitsRepo(db)
    kit = await repo.get(kit_id)
    if kit is None:
        raise HTTPException(status_code=404, detail="brand kit not found")

    output_dir = Path(get_settings().default_output_dir)
    asset_dir = _brand_kit_asset_dir(output_dir, kit_id)
    asset_dir.mkdir(parents=True, exist_ok=True)
    call_log_path = output_dir / "llm_calls_brand_kits.jsonl"
    router_ = LLMRouter(
        config=load_llm_config(),
        call_log_path=call_log_path,
        db=db,
    )
    try:
        logo = await generate_logo(
            tenant_id=tenant_id,
            brand_name=kit.name,
            primary_color=kit.primary_color,
            accent_color=kit.accent_color,
            style=style_hint or "Minimal mark / monogram",
            router=router_,
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"logo generation failed: {e}",
        ) from e

    svg_path = asset_dir / "logo.svg"
    svg_path.write_text(logo.svg, encoding="utf-8")
    # Rasterize best-effort — caller fallback is "SVG only", not failure.
    rasterize_svg_to_png(
        logo.svg, output_path=asset_dir / "logo.png", size_px=1024
    )
    # Use the file-serving endpoint as the canonical URL so the dashboard
    # template doesn't need to know about the on-disk layout.
    logo_url = f"/dashboard/brand-kits/{kit_id}/logo.svg"
    await repo.update(
        kit_id,
        logo_url=logo_url,
        ai_generated=True,
        ai_prompt=style_hint or "Minimal mark / monogram",
        ai_provider="anthropic",
    )
    return RedirectResponse(
        url=f"/dashboard/brand-kits/{kit_id}", status_code=303
    )


@router.get("/brand-kits/{kit_id}/logo.svg")
async def brand_kits_logo_svg(
    request: Request,
    kit_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Serve the saved SVG for the dashboard preview + downstream renderer."""
    from chalybclip.settings import get_settings

    if await BrandKitsRepo(db).get(kit_id) is None:
        raise HTTPException(status_code=404, detail="brand kit not found")
    output_dir = Path(get_settings().default_output_dir)
    svg_path = _brand_kit_asset_dir(output_dir, kit_id) / "logo.svg"
    if not svg_path.exists():
        raise HTTPException(status_code=404, detail="no logo for this kit")
    return FileResponse(
        path=str(svg_path),
        media_type="image/svg+xml",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/brand-kits/{kit_id}/logo.png")
async def brand_kits_logo_png(
    request: Request,
    kit_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Serve the rasterized PNG (when cairosvg was installed at generate-time).

    Falls through to 404 when the SVG was generated but rasterization was
    skipped — the dashboard's `<img>` fallback covers that case."""
    from chalybclip.settings import get_settings

    if await BrandKitsRepo(db).get(kit_id) is None:
        raise HTTPException(status_code=404, detail="brand kit not found")
    output_dir = Path(get_settings().default_output_dir)
    png_path = _brand_kit_asset_dir(output_dir, kit_id) / "logo.png"
    if not png_path.exists():
        raise HTTPException(status_code=404, detail="no logo png for this kit")
    return FileResponse(
        path=str(png_path),
        media_type="image/png",
        headers={"Cache-Control": "no-cache"},
    )
