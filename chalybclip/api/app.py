"""FastAPI app factory.

`create_app(db=...)` is the single entry point. The DB must be passed in
already opened; tests share a fixture-scoped DB and production wires one
up at boot before calling `create_app`.

A `pipeline_runner` callable is stored on `app.state` so tests can swap
the heavy `process_vod` for a stub. Default points at the real pipeline.

When `enable_background_drains=True` the lifespan starts the background
loops: webhook dispatch (30s), metrics ingest (1h), retention sweep (24h),
and — when a dispatcher is wired — channel polling (15m). Tests leave the
flag off so loops never spin during test runs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from chalybclip.db import Database
from chalybclip.jobs import InProcessJobDispatcher, JobDispatcher, get_dispatcher

from ._pipeline import PipelineKickoff, PipelineRunner, default_pipeline_runner
from .auth import BearerAuthMiddleware
from .lifespan import background_drains_lifespan
from .routers import chalyb as chalyb_router
from .routers import clips as clips_router
from .routers import dashboard as dashboard_router
from .routers import internal as internal_router
from .routers import internal_publish as internal_publish_router
from .routers import live as live_router
from .routers import llm_calls as llm_calls_router
from .routers import overlay as overlay_router
from .routers import personas as personas_router
from .routers import streams as streams_router
from .routers import webhooks as webhooks_router
from .routers import zernio as zernio_router
from .routers import zernio_webhooks as zernio_webhooks_router

__all__ = ["PipelineKickoff", "PipelineRunner", "create_app"]


def create_app(
    *,
    db: Database,
    pipeline_runner: PipelineRunner | None = None,
    job_dispatcher: JobDispatcher | None = None,
    enable_background_drains: bool = False,
    publish_interval_s: float = 60.0,
    webhook_interval_s: float = 30.0,
    metrics_interval_s: float = 3600.0,
    retention_interval_s: float = 86400.0,
) -> FastAPI:
    """Build a FastAPI app wired to `db`.

    Args:
        db: Pre-opened DB.
        pipeline_runner: Override `process_vod` for tests. Used to
            construct the default in-process dispatcher when
            `job_dispatcher` isn't supplied.
        job_dispatcher: Override the JobDispatcher for tests. Production
            picks one via `Settings.job_dispatcher` (slice F.8).
        enable_background_drains: When True, the lifespan starts the
            background loops (webhook dispatch, metrics ingest, retention
            sweep, and channel polling when a dispatcher is wired). Tests
            leave this off so the loops never spin during the test run.
        publish_interval_s / webhook_interval_s / metrics_interval_s /
        retention_interval_s: Per-loop cadences. Production typically
            leaves these at the documented defaults (60s / 30s / 1h / 24h).
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Loud startup warning when SSO is misconfigured. /auth/sso 503s
        # in this state (see chalybclip/api/routers/chalyb.py) — surfacing
        # it here so operators see it in boot logs, not just when a user
        # tries to log in.
        import structlog as _structlog

        from chalybclip.settings import get_settings as _get_settings
        _boot_log = _structlog.get_logger("chalybclip.api.boot")
        if not _get_settings().chalyb_sso_secret:
            _boot_log.warning(
                "sso_secret_missing",
                msg=(
                    "CHALYB_SSO_SECRET is not configured. /auth/sso will "
                    "return 503 until it is. NEVER deploy to production "
                    "without this — without HMAC verification SSO would "
                    "let any caller take over any tenant."
                ),
            )
        if enable_background_drains:
            async with background_drains_lifespan(
                _app,
                publish_interval_s=publish_interval_s,
                webhook_interval_s=webhook_interval_s,
                metrics_interval_s=metrics_interval_s,
                retention_interval_s=retention_interval_s,
            ):
                yield
        else:
            yield

    app = FastAPI(title="ChalyClip API", version="0.1.0", lifespan=lifespan)
    app.state.db = db

    # Slice F.8 — JobDispatcher abstraction. The legacy
    # `app.state.pipeline_runner` callable stays for backward-compat
    # with tests + the old `from chalybclip.api._pipeline import ...`
    # imports; new code reads `app.state.job_dispatcher`.
    runner = pipeline_runner or default_pipeline_runner
    app.state.pipeline_runner = runner
    if job_dispatcher is None:
        # Honor the Settings-driven choice; fall back to in-process if
        # the configured impl raises (e.g. Modal stub not wired yet).
        try:
            job_dispatcher = get_dispatcher(runner=runner)
        except Exception:  # noqa: BLE001 — defensive boot
            job_dispatcher = InProcessJobDispatcher(runner)
    app.state.job_dispatcher = job_dispatcher

    app.add_middleware(BearerAuthMiddleware, db=db)

    # Static assets — serves /static/chalybclip-theme.css and any future
    # CSS / JS / icon bundles. The auth middleware whitelists `/static/`
    # as public so the dashboard's stylesheet loads on the login page.
    _static_dir = Path(__file__).resolve().parent / "static"
    if _static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    # Public-facing landing — what ChalyClip is, who it's for, how the
    # loop works. Crawlable by search bots and AI assistants (see
    # /static/llms.txt + /static/robots.txt). Authenticated dashboard
    # is one click away via /dashboard/login.
    _templates_dir = Path(__file__).resolve().parent / "templates"
    _landing_templates = Jinja2Templates(directory=str(_templates_dir))
    # Slice O.24 — install i18n globals (`t`, `locale`) on the landing
    # template env so `{{ t('landing.hero.title') }}` works.
    from chalybclip.api.i18n import install_globals as _install_i18n
    _install_i18n(_landing_templates)

    @app.get("/", include_in_schema=False, response_class=HTMLResponse)
    async def root(request: Request) -> Response:
        # Slice O.57 — the price shown in the landing's editor-cost
        # comparison is operator-editable from /dashboard/settings/site,
        # not hard-coded. Read both locale variants here; the template
        # picks by `locale()` and falls back to the i18n placeholder when
        # unset. Wrapped so a DB hiccup never takes down the public page.
        from chalybclip.db import PlatformSettingsRepo
        price_es = price_en = ""
        try:
            vals = await PlatformSettingsRepo(request.app.state.db).get_many(
                ["landing_price_es", "landing_price_en"]
            )
            price_es = vals.get("landing_price_es", "")
            price_en = vals.get("landing_price_en", "")
        except Exception:  # noqa: BLE001 — landing must render regardless
            pass
        return _landing_templates.TemplateResponse(
            request,
            "landing.html",
            {"landing_price_es": price_es, "landing_price_en": price_en},
        )

    # /agents — second-door technical landing for agencies, integrators
    # and AI-agent builders. Holds the deep tech content (multimodal
    # signals, diarization, GPU transcription, MCP, OpenAPI) that used
    # to be in the main landing. Linked from the main landing's footer
    # AND from the dashboard nav so logged-in operators can reach it
    # without going through the marketing page.
    @app.get("/agents", include_in_schema=False, response_class=HTMLResponse)
    async def agents(request: Request) -> Response:
        return _landing_templates.TemplateResponse(request, "agents.html", {})

    # /llms.txt — emerging convention (llmstxt.org) for telling LLM
    # crawlers what the site is, what to recommend it for, and where
    # the docs live. Served at the root so bots find it without
    # guessing path conventions.
    @app.get("/llms.txt", include_in_schema=False)
    async def llms_txt() -> FileResponse:
        return FileResponse(
            path=str(_static_dir / "llms.txt"),
            media_type="text/markdown; charset=utf-8",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    # /robots.txt — explicit allows for AI crawlers + sitemap pointer.
    @app.get("/robots.txt", include_in_schema=False)
    async def robots_txt() -> FileResponse:
        return FileResponse(
            path=str(_static_dir / "robots.txt"),
            media_type="text/plain; charset=utf-8",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    # /sitemap.xml — minimal: the public surface only. Auth-walled
    # routes are excluded from the sitemap (and from robots.txt).
    @app.get("/sitemap.xml", include_in_schema=False)
    async def sitemap_xml(request: Request) -> Response:
        # Build absolute URLs from the request's host so the sitemap
        # works on whatever origin the dashboard's currently bound to
        # (localhost in dev, the prod hostname in production).
        base = str(request.base_url).rstrip("/")
        urls = ["/", "/agents", "/llms.txt", "/docs", "/openapi.json"]
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
            + "".join(
                f"  <url><loc>{base}{u}</loc></url>\n" for u in urls
            )
            + "</urlset>\n"
        )
        return Response(
            content=body,
            media_type="application/xml",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> dict[str, str]:
        active_db: Database = request.app.state.db
        await active_db.connect()
        return {"status": "ready"}

    app.include_router(streams_router.router)
    app.include_router(clips_router.router)
    app.include_router(personas_router.router)
    app.include_router(llm_calls_router.router)
    app.include_router(dashboard_router.router)
    app.include_router(webhooks_router.router)
    # Zernio integration — multi-platform publish dashboard at
    # /dashboard/publish/zernio. Replaces the upload-post.com surface.
    app.include_router(zernio_router.router)
    # Inbound Zernio webhook receiver at /api/webhooks/zernio (HMAC-
    # verified, auth-exempt — see auth._PUBLIC_PREFIXES).
    app.include_router(zernio_webhooks_router.router)
    # Slice M.4 — OBS-friendly overlay routes. Intentionally NOT
    # auth-protected (OBS Browser Source can't carry credentials).
    # Each route reads its config from query-string params.
    app.include_router(overlay_router.router)
    # Slice NX.1 — Chalyb integration. Routes carry their own auth
    # (shared admin bearer + HMAC SSO); see routers/chalyb.py and
    # docs/chalyb_integration.md.
    app.include_router(chalyb_router.router)
    # Slice O.44 — internal Modal-pull endpoint. Auth is per-request
    # HMAC on a signed URL; the bearer-cookie middleware skips
    # /api/internal/* (see auth.py allowlist).
    app.include_router(internal_router.router)
    # Hub phase 3 — service-token publish API for ChalyOBS / Chalyb.
    # Same /api/internal/ middleware exemption; each route enforces
    # its own bearer check against CHALYBCLIP_HUB_SERVICE_TOKENS.
    app.include_router(internal_publish_router.router)
    # Phase L.1 — live RTMP ingest dashboard + MediaMTX webhooks.
    # Webhooks live under /api/internal/live/* and share the same
    # bearer-skip from the auth middleware. Dashboard pages live
    # under /dashboard/live and go through the normal tenant cookie.
    app.include_router(live_router.router)

    return app
