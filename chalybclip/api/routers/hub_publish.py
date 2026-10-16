"""Hub API — Zernio publishing (/api/hub/v1/publish/…).

The Chalyb hub renders the whole Publish Center (Centro de publicación) at
/app/clips/publicar and calls these routes server-to-server with its
per-request Bearer launch token (see api/hub_api.py). Nothing here forks a
business rule: almost every route mounts the SAME handler function the
dashboard uses (routers/zernio.py), so validation, tier gates, tenant
isolation and Zernio calls are shared. The auth middleware binds the hub
caller exactly like a dashboard session (`request.state.tenant_id`,
`token_scope="full"`, `tenant_tier`), so `tenant_binder`,
`require_full_scope` and `require_paid_tier` behave the same.

What this module adds on top:

  * `_hubify` — runs a dashboard handler and answers in the hub's shape:
    errors become the shared `{"ok": false, "error": {code, message, …}}`
    body (engine copy kept for 4xx; 5xx get the generic message, never a
    raw vendor error), and the form routes' 303 redirects become
    `{"ok": true, …query}` JSON.
  * ownership gates for the routes that act on a Zernio id taken from the
    URL (posts, automations, sequences). The Zernio API key is
    company-wide, so the id must be proven to be the caller's BEFORE the
    vendor call. (ChalyClip PR #88 adds the same checks to the dashboard
    handlers; these stay as defence in depth.)
  * `GET /publish/overview` — the dashboard page's data as JSON.
  * `POST /publish/connect` — the OAuth popup returns to the HUB's landing
    page (`return_url`, same origin as CHALYB_BASE_URL), never to an engine
    page.
"""

from __future__ import annotations

import functools
import inspect
import json
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.routing import APIRoute
from starlette.types import Message

from chalybclip.api.hub_api import HubError, hub_error_response, hub_router
from chalybclip.db import Database, HubPublishJobsRepo, TenantsRepo, ZernioPublishesRepo
from chalybclip.integrations.zernio import ZernioError
from chalybclip.settings import get_settings

from ..deps import get_db, require_full_scope, tenant_binder
from ..status_gate import require_paid_tier
from . import dashboard
from . import zernio as zn

router = hub_router(tags=["hub-publish"])

P = "/publish"

Gate = Callable[[dict[str, Any]], Awaitable[None]]


# ---------- response shaping ----------


def _code_for(status_code: int, reason: str | None = None) -> str:
    if status_code == 402:
        return (
            "needs_plan"
            if reason in ("account_limit",) or (reason or "").startswith("paywall")
            else "no_credits"
        )
    return {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        409: "conflict",
        413: "too_large",
        422: "bad_request",
        423: "forbidden",
        429: "rate_limited",
        503: "unavailable",
    }.get(status_code, "bad_request" if status_code < 500 else "internal")


def _hub_error(status_code: int, message: str | None, reason: str | None) -> HubError:
    # 4xx copy is the engine's own operator-facing text (validation, caps,
    # state conflicts). 5xx messages embed vendor/infra detail (Zernio
    # bodies, env var names) — the hub shows its generic copy instead.
    keep = message if (message and status_code < 500) else None
    code = _code_for(status_code, reason)
    return HubError(
        code,
        keep,
        status_code=status_code,
        reason=reason or ("vendor_error" if status_code == 502 else None),
    )


def _from_http_exception(exc: HTTPException) -> HubError:
    detail: Any = exc.detail
    reason: str | None = None
    message: str | None = None
    if isinstance(detail, dict):
        r = detail.get("reason")
        reason = r if isinstance(r, str) else None
        m = detail.get("message")
        message = m if isinstance(m, str) else None
    elif isinstance(detail, str):
        message = detail
    return _hub_error(exc.status_code, message, reason)


def _to_hub(resp: Any) -> Any:
    """Normalize a dashboard handler's response for the hub."""
    if isinstance(resp, RedirectResponse) and resp.status_code == 303:
        # Form routes answer 303 → /dashboard/publish/zernio?queued=…; the
        # flash params ARE the result (queued post id, reopened count…).
        location = resp.headers.get("location", "")
        params = dict(parse_qsl(urlsplit(location).query))
        return JSONResponse({"ok": True, **params})
    if isinstance(resp, JSONResponse) and resp.status_code >= 400:
        try:
            body = json.loads(bytes(resp.body))
        except ValueError:
            body = {}
        if not isinstance(body, dict):
            body = {}
        err = body.get("error")
        message = err if isinstance(err, str) else None
        reason = body.get("reason") if isinstance(body.get("reason"), str) else None
        e = _hub_error(resp.status_code, message, reason)
        out = hub_error_response(
            e.code,
            e.message,
            status_code=e.status_code,
            reason=e.reason,
            retryable=e.retryable,
        )
        return out
    return resp


def _hubify(endpoint: Callable[..., Awaitable[Any]], *, gate: Gate | None = None) -> Any:
    """Wrap a dashboard handler so FastAPI resolves EXACTLY its parameters
    and dependencies (same signature, annotations evaluated against the
    handler's own module), then shape its response for the hub."""
    sig = inspect.signature(endpoint, eval_str=True)

    @functools.wraps(endpoint)
    async def wrapper(**kwargs: Any) -> Any:
        try:
            if gate is not None:
                await gate(kwargs)
            return _to_hub(await endpoint(**kwargs))
        except HTTPException as exc:
            raise _from_http_exception(exc) from exc

    wrapper.__signature__ = sig  # type: ignore[attr-defined]
    return wrapper


class _JsonAsFormRoute(APIRoute):
    """For the dashboard's FORM routes (publish one clip, publish a draft,
    claim a profile): the hub speaks JSON, so a JSON object body is re-encoded
    as the urlencoded form those handlers already parse — same field names,
    lists joined with commas (`platforms`). Nothing else changes."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            ctype = request.headers.get("content-type", "")
            if ctype.split(";")[0].strip().lower() == "application/json":
                data = await zn._read_json(request)
                if not isinstance(data, dict):
                    raise HubError("bad_request")
                fields: list[tuple[str, str]] = []
                for key, value in data.items():
                    if value is None:
                        continue
                    if isinstance(value, list):
                        value = ",".join(str(v) for v in value)
                    elif isinstance(value, bool):
                        value = "true" if value else "false"
                    fields.append((str(key), str(value)))
                body = urlencode(fields).encode()
                headers = [
                    (k, v)
                    for k, v in request.scope["headers"]
                    if k not in (b"content-type", b"content-length")
                ]
                headers += [
                    (b"content-type", b"application/x-www-form-urlencoded"),
                    (b"content-length", str(len(body)).encode()),
                ]
                sent = False

                async def receive() -> Message:
                    nonlocal sent
                    if sent:
                        return {"type": "http.disconnect"}
                    sent = True
                    return {"type": "http.request", "body": body, "more_body": False}

                request = Request({**request.scope, "headers": headers}, receive)
            return await original(request)

        return handler


# Dashboard routes that take a urlencoded form (see _JsonAsFormRoute).
_FORM_ROUTES = frozenset({"/post/{clip_id}", "/draft/{post_id}/publish", "/accounts/claim"})


def _mount(
    method: str, path: str, endpoint: Callable[..., Awaitable[Any]], *, gate: Gate | None = None
) -> None:
    router.add_api_route(
        P + path,
        _hubify(endpoint, gate=gate),
        methods=[method],
        include_in_schema=False,
        name=f"hub_publish:{endpoint.__name__}",
        route_class_override=_JsonAsFormRoute if path in _FORM_ROUTES else None,
    )


# ---------- ownership gates (company-wide Zernio key) ----------


async def _profile_id(db: Database, tenant_id: str) -> str | None:
    tenant = await TenantsRepo(db).get(tenant_id)
    return tenant.zernio_profile_id if tenant else None


def _ids(rows: Any) -> set[str]:
    out: set[str] = set()
    for r in rows or []:
        if isinstance(r, dict):
            rid = r.get("_id") or r.get("id")
            if rid:
                out.add(str(rid))
    return out


async def _post_owner(db: Database, post_id: str) -> str | None:
    """Tenant our own records attribute the post to (dashboard publishes →
    zernio_publishes; hub/ChalyOBS publishes → hub_publish_jobs)."""
    row = await ZernioPublishesRepo(db).get_by_post_id(post_id)
    if row is not None:
        return row.tenant_id
    return await HubPublishJobsRepo(db).get_tenant_for_post(post_id)


def _not_found() -> HubError:
    return HubError("not_found")


async def _gate_status(kw: dict[str, Any]) -> None:
    """Job page / poll: only posts our records give to this tenant."""
    owner = await _post_owner(kw["db"], kw["post_id"])
    if owner != kw["tenant_id"]:
        raise _not_found()


async def _gate_owned_or_listed(
    kw: dict[str, Any],
    lister: Callable[[str], Awaitable[Any]],
) -> None:
    db, tenant_id, post_id = kw["db"], kw["tenant_id"], kw["post_id"]
    owner = await _post_owner(db, post_id)
    if owner is not None:
        if owner != tenant_id:
            raise _not_found()
        return
    # No local record (posted straight on Zernio): it must be in the list
    # the hub's buttons come from, read with THIS tenant's profile.
    profile_id = await _profile_id(db, tenant_id)
    if not profile_id or not get_settings().zernio_api_key:
        raise _not_found()
    try:
        listed = _ids(await lister(profile_id))
    except ZernioError:
        raise HubError("unavailable", status_code=502, reason="vendor_error") from None
    if post_id not in listed:
        raise _not_found()


async def _gate_retry(kw: dict[str, Any]) -> None:
    client = zn._build_client()
    await _gate_owned_or_listed(kw, lambda pid: client.list_failed(profile_id=pid))


async def _gate_cancel(kw: dict[str, Any]) -> None:
    async def scheduled(pid: str) -> Any:
        feed = await zn._build_client().list_posts(
            status="scheduled",
            profile_id=pid,
            sort_by="scheduled-asc",
            limit=100,
        )
        return feed.get("posts") if isinstance(feed, dict) else []

    await _gate_owned_or_listed(kw, scheduled)


def _gate_listed(id_param: str, lister: str) -> Gate:
    async def gate(kw: dict[str, Any]) -> None:
        profile_id = await _profile_id(kw["db"], kw["tenant_id"])
        if not profile_id or not get_settings().zernio_api_key:
            raise _not_found()
        try:
            rows = await getattr(zn._build_client(), lister)(profile_id=profile_id)
        except ZernioError:
            raise HubError("unavailable", status_code=502, reason="vendor_error") from None
        if str(kw[id_param]) not in _ids(rows):
            raise _not_found()

    return gate


# ---------- reused dashboard handlers ----------
# (method, hub path under /publish, handler, gate)

_REUSED: list[tuple[str, str, Callable[..., Awaitable[Any]], Gate | None]] = [
    # profile + accounts
    ("POST", "/profile", zn.zernio_create_profile, None),
    ("GET", "/accounts.json", zn.zernio_accounts_json, None),
    ("POST", "/accounts/{account_id}/disconnect", zn.zernio_disconnect_account, None),
    ("POST", "/accounts/claim", zn.zernio_claim_existing, None),
    ("POST", "/unlink", zn.zernio_unlink, None),
    ("POST", "/fb-pages", zn.zernio_facebook_pages, None),
    ("POST", "/fb-page/select", zn.zernio_facebook_select_page, None),
    # publish (single / bulk / drafts / manual state)
    ("POST", "/post/{clip_id}", zn.zernio_post_clip, None),
    ("POST", "/bulk-post", zn.zernio_bulk, None),
    ("POST", "/compose/{clip_id}", zn.zernio_compose_clip, None),
    ("POST", "/draft/{post_id}/publish", zn.zernio_draft_publish, None),
    ("POST", "/draft/{post_id}/delete", zn.zernio_draft_delete, None),
    ("POST", "/clip/{clip_id}/mark-published", zn.zernio_mark_published, None),
    ("POST", "/clip/{clip_id}/reopen", zn.zernio_reopen_clip, None),
    ("POST", "/reopen-all", zn.zernio_reopen_all, None),
    ("GET", "/clips/{clip_id}/growth.json", zn.zernio_clip_growth_json, None),
    # Thumbnail JPEG (or a 302 to the bucket copy) for the clip rows.
    ("GET", "/clips/{clip_id}/thumbnail", dashboard.clip_thumbnail, None),
    # history / status
    ("GET", "/feed.json", zn.zernio_feed, None),
    ("GET", "/status/{post_id}.json", zn.zernio_status, _gate_status),
    ("GET", "/failed.json", zn.zernio_failed_json, None),
    ("POST", "/retry/{post_id}", zn.zernio_retry_one, _gate_retry),
    ("POST", "/retry-all", zn.zernio_retry_all, None),
    ("POST", "/reprocess-failed", zn.zernio_reprocess_failed, None),
    ("POST", "/reprocess/{post_id}", zn.zernio_reprocess_one, None),
    # schedule
    ("GET", "/schedule.json", zn.zernio_schedule_json, None),
    ("POST", "/schedule/slots", zn.zernio_save_slots, None),
    ("POST", "/schedule/queue/{queue_id}/delete", zn.zernio_delete_queue, None),
    ("POST", "/schedule/cancel/{post_id}", zn.zernio_cancel_scheduled, _gate_cancel),
    ("POST", "/schedule/auto", zn.zernio_schedule_auto, None),
    ("GET", "/schedule/auto/progress", zn.zernio_schedule_auto_progress, None),
    ("GET", "/schedule/next-best.json", zn.zernio_next_best_json, None),
    ("GET", "/best-time.json", zn.zernio_best_time_json, None),
    ("GET", "/calendar.json", zn.zernio_calendar_json, None),
    ("GET", "/rendimiento.json", zn.zernio_performance_json, None),
    # autopilot
    ("GET", "/autopublish.json", zn.zernio_autopublish_json, None),
    ("POST", "/autopublish/save", zn.zernio_autopublish_save, None),
    ("GET", "/autopublish/rules.json", zn.zernio_pacing_rules_json, None),
    ("POST", "/autopublish/rules/save", zn.zernio_pacing_rules_save, None),
    # community
    ("GET", "/community/settings.json", zn.zernio_community_settings_json, None),
    ("POST", "/community/settings", zn.zernio_community_save_settings, None),
    # inbox
    ("GET", "/inbox/comments.json", zn.zernio_inbox_comments_json, None),
    ("POST", "/inbox/comments/reply", zn.zernio_inbox_comment_reply, None),
    ("POST", "/inbox/comments/{comment_action}", zn.zernio_inbox_comment_action, None),
    ("GET", "/inbox/conversations.json", zn.zernio_inbox_conversations_json, None),
    ("GET", "/inbox/messages.json", zn.zernio_inbox_messages_json, None),
    ("POST", "/inbox/conversations/reply", zn.zernio_inbox_send_message, None),
    ("POST", "/inbox/conversations/archive", zn.zernio_inbox_archive, None),
    # growth
    ("GET", "/growth/contacts.json", zn.zernio_growth_contacts_json, None),
    ("GET", "/growth/automations.json", zn.zernio_growth_automations_json, None),
    ("POST", "/growth/automations", zn.zernio_growth_create_automation, None),
    (
        "POST",
        "/growth/automations/{automation_id}/toggle",
        zn.zernio_growth_toggle_automation,
        _gate_listed("automation_id", "list_comment_automations"),
    ),
    ("GET", "/growth/sequences.json", zn.zernio_growth_sequences_json, None),
    ("POST", "/growth/sequences", zn.zernio_growth_create_sequence, None),
    (
        "POST",
        "/growth/sequences/{sequence_id}/toggle",
        zn.zernio_growth_toggle_sequence,
        _gate_listed("sequence_id", "list_sequences"),
    ),
    (
        "POST",
        "/growth/sequences/{sequence_id}/enroll",
        zn.zernio_growth_enroll_sequence,
        _gate_listed("sequence_id", "list_sequences"),
    ),
    ("POST", "/growth/broadcasts/send", zn.zernio_growth_send_broadcast, None),
    # feature-flagged (404 while the flag is off)
    ("GET", "/ads/campaigns.json", zn.zernio_ads_campaigns_json, None),
    ("POST", "/ads/boost", zn.zernio_ads_boost, None),
    ("GET", "/whatsapp/status.json", zn.zernio_whatsapp_status_json, None),
]


# ---------- hub-only routes ----------
# Declared BEFORE the reused mounts so a literal path never loses to a
# path-parameter route.


def _row_json(row: Any) -> dict[str, Any]:
    return {
        "post_id": row.post_id,
        "clip_id": row.clip_id,
        "platforms": [p for p in (row.platforms or "").split(",") if p],
        "content": row.content,
        "created_at": row.created_at,
        "scheduled_for": row.scheduled_for,
    }


@router.get(P + "/overview", include_in_schema=False)
async def hub_publish_overview(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """The Publish Center's data (the dashboard page's context) as JSON.
    Auto-provisions the tenant's Zernio profile on first open, like the
    page does."""
    try:
        ctx = await zn.dashboard_context(request, tenant_id=tenant_id, db=db)
    except HTTPException as exc:
        raise _from_http_exception(exc) from exc
    labels = {key: label for key, label, _icon in zn._SUPPORTED_PLATFORMS}
    return JSONResponse(
        {
            "ok": True,
            "configured": ctx["configured"],
            "profile": (
                {"id": ctx["profile_id"], "name": ctx["profile_name"]}
                if ctx["profile_id"]
                else None
            ),
            "accounts": [
                {"platform": a.platform.lower(), "account_id": a.account_id}
                for a in ctx["accounts"]
            ],
            "account_limit": ctx["account_limit"],
            "account_count": ctx["account_count"],
            "connect_fetch_failed": ctx["connect_fetch_failed"],
            "platforms": [{"id": k, "label": v} for k, v in labels.items()],
            "community_platforms": [
                {"id": k, "label": label} for k, label, _i in ctx["community_platforms"]
            ],
            "community_connected": ctx["community_connected"],
            "stats": ctx["stats"],
            "history": ctx["history"],
            "published_clip_ids": ctx["published_clip_ids"],
            "publishable": [
                {k: c[k] for k in ("id", "stream_id", "duration_s", "title", "status")}
                for c in ctx["publishable_clips"]
            ],
            "publishable_total": ctx["publishable_total"],
            "drafts": [_row_json(d) for d in ctx["drafts"]],
            "is_pro": ctx["is_pro"],
            "feature_ads": bool(ctx["feature_ads"]),
            "feature_whatsapp": bool(ctx["feature_whatsapp"]),
        }
    )


def _hub_return_url(raw: str, platform: str) -> str | None:
    """`raw` with ?platform=… when it is an https (or local-dev http) URL on
    the hub's own origin (CHALYB_BASE_URL); None otherwise. Keeps the OAuth
    redirect from becoming an open redirect even if a caller is buggy."""
    base = (get_settings().chalyb_base_url or "").strip()
    if not base or not raw:
        return None
    try:
        want, got = urlsplit(base), urlsplit(raw)
    except ValueError:
        return None
    if got.scheme not in ("https", "http") or not got.hostname or got.username:
        return None
    if got.scheme == "http" and got.hostname not in ("localhost", "127.0.0.1"):
        return None

    def bare(host: str | None) -> str:
        h = (host or "").lower()
        return h[4:] if h.startswith("www.") else h

    if bare(got.hostname) != bare(want.hostname) or got.port != want.port:
        return None
    query = [(k, v) for k, v in parse_qsl(got.query) if k != "platform"]
    query.append(("platform", platform))
    return urlunsplit((got.scheme, got.netloc, got.path, urlencode(query), ""))


@router.post(P + "/connect", include_in_schema=False)
async def hub_publish_connect(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Mint a hosted-OAuth URL for one platform. Body: {platform,
    return_url}. Zernio sends the popup back to `return_url` (the hub's
    /app/clips/publicar/conectado landing), which tells the hub tab and
    closes — or, for Facebook, shows the hub's page picker."""
    data = await zn._read_json(request)
    if not isinstance(data, dict):
        raise HubError("bad_request")
    platform = str(data.get("platform") or "").strip().lower()
    if platform not in zn._CONNECTABLE_PLATFORM_IDS:
        raise HubError("bad_request", reason="unsupported_platform")
    redirect_back = _hub_return_url(str(data.get("return_url") or ""), platform)
    if redirect_back is None:
        raise HubError("bad_request", reason="bad_return_url")
    try:
        resp = await zn.connect_auth_url_response(
            request,
            tenant_id=tenant_id,
            db=db,
            platform=platform,
            redirect_back=redirect_back,
        )
    except HTTPException as exc:
        raise _from_http_exception(exc) from exc
    shaped: Response = _to_hub(resp)
    return shaped


for _method, _path, _endpoint, _gate in _REUSED:
    _mount(_method, _path, _endpoint, gate=_gate)


__all__ = ["router"]
