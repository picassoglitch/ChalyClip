"""Auto-publish on approve, as a background job.

Approving a clip used to await the whole auto-publish inline — render
(minutes) + Zernio post — so "Aprobar y seguir" could hang for minutes. Now
`clip_overlay_finalize` queues this job after its 1080 pre-render and
returns at once.

The job's state lives in `zernio_publishes`, the table Publicar already
reads, under a local placeholder post id `autopub-<clip>-<ts>`:

  * `pending` while the job runs (the editor shows "publicando…");
  * on success the real Zernio post row takes over and the placeholder is
    marked `deleted` (hidden everywhere);
  * on a skip (cap reached, no connected targets, degenerate copy) it's
    `deleted` too — same as before, nothing was attempted;
  * on failure it's `failed` and the error is kept as an event; Publicar's
    failed list includes these rows and its Reintentar re-runs the job.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import structlog
from fastapi import BackgroundTasks, Request

from chalybclip.db import Database, EventsRepo, ZernioPublishesRepo
from chalybclip.tenancy import bound_tenant

PLACEHOLDER_PREFIX = "autopub-"

_log = structlog.get_logger(__name__)
_running: set[asyncio.Task[Any]] = set()


def is_placeholder(post_id: str) -> bool:
    return post_id.startswith(PLACEHOLDER_PREFIX)


async def _autopilot_targets(db: Database, tenant_id: str) -> list[str] | None:
    """The configured targets when auto-publish on approve would fire for
    this tenant (setting on, Zernio configured, top tier — Piloto automático
    is a VIP perk, same gate as the publish path); None when it wouldn't."""
    from chalybclip.db import AutopublishSettingsRepo, TenantsRepo
    from chalybclip.settings import get_settings
    from chalybclip.tiers import TOP_TIERS, normalize_tier

    if not get_settings().zernio_api_key:
        return None
    s = await AutopublishSettingsRepo(db).get(tenant_id)
    if not (s and s["enabled"] and s["mode"] == "on_approve"):
        return None
    tenant = await TenantsRepo(db).get(tenant_id)
    if normalize_tier(tenant.tier if tenant else None) not in TOP_TIERS:
        return None
    return [t.strip() for t in str(s.get("targets") or "").split(",") if t.strip()]


async def _queue(db: Database, *, tenant_id: str, clip_id: str, targets: list[str]) -> str:
    placeholder = f"{PLACEHOLDER_PREFIX}{clip_id}-{int(time.time() * 1000)}"
    with bound_tenant(tenant_id):
        await ZernioPublishesRepo(db).record(
            post_id=placeholder,
            tenant_id=tenant_id,
            clip_id=clip_id,
            platforms=targets,
            content=None,
            status="pending",
            options_json=json.dumps({"kind": "autopublish"}),
        )
    return placeholder


def _without_session(request: Request) -> Request:
    """The approving request minus its session cookie: the job runs after
    the response, when that cookie may be stale. With no cookie the render
    signs its own /render link (api/_clip_render) instead."""
    headers = [(k, v) for k, v in request.scope.get("headers", []) if k.lower() != b"cookie"]
    return Request({**request.scope, "headers": headers})


async def _connected_targets(tenant_id: str, db: Database) -> list[str]:
    """Where an empty-targets auto-publish would have gone: every connected
    account. Best-effort — [] when Zernio can't be asked."""
    try:
        from chalybclip.api.routers.zernio import _build_client, _connected_platforms
        from chalybclip.db import TenantsRepo

        tenant = await TenantsRepo(db).get(tenant_id)
        if not tenant or not tenant.zernio_profile_id:
            return []
        accounts = await _build_client().list_accounts(profile_id=tenant.zernio_profile_id)
        return sorted(_connected_platforms(accounts))
    except Exception:
        return []


async def run_autopublish(
    *, request: Request, db: Database, tenant_id: str, clip_id: str, placeholder: str
) -> None:
    """The job body. Never raises."""
    from chalybclip.api.routers.zernio import maybe_autopublish_on_approve

    request = _without_session(request)
    repo = ZernioPublishesRepo(db)
    try:
        with bound_tenant(tenant_id):
            post_id = await maybe_autopublish_on_approve(
                request=request, db=db, tenant_id=tenant_id, clip_id=clip_id, raise_errors=True
            )
            await repo.set_status(placeholder, status="deleted")
            await EventsRepo(db).emit(
                type="clip.autopublish_done",
                payload={"clip_id": clip_id, "post_id": post_id, "placeholder": placeholder},
            )
    except Exception as e:  # recorded where Publicar shows it
        _log.warning("autopublish.job_failed", tenant_id=tenant_id, clip_id=clip_id, error=str(e))
        try:
            with bound_tenant(tenant_id):
                await repo.set_status(placeholder, status="failed")
                row = await repo.get_by_post_id(placeholder)
                if row is not None and not row.platforms:
                    targets = await _connected_targets(tenant_id, db)
                    if targets:
                        await _set_platforms(db, placeholder, targets)
                await EventsRepo(db).emit(
                    type="clip.autopublish_failed",
                    payload={
                        "clip_id": clip_id,
                        "placeholder": placeholder,
                        "error": str(e)[:300],
                    },
                )
        except Exception:
            pass


async def _set_platforms(db: Database, post_id: str, platforms: list[str]) -> None:
    conn = await db.connect()
    await conn.execute(
        "UPDATE zernio_publishes SET platforms = ? WHERE post_id = ?",
        (",".join(platforms), post_id),
    )
    await conn.commit()


async def enqueue_autopublish(
    background_tasks: BackgroundTasks,
    *,
    request: Request,
    db: Database,
    tenant_id: str,
    clip_id: str,
) -> bool:
    """Queue the job (after anything already on `background_tasks`, i.e.
    the approve pre-render). False when auto-publish on approve is off."""
    targets = await _autopilot_targets(db, tenant_id)
    if targets is None:
        return False
    placeholder = await _queue(db, tenant_id=tenant_id, clip_id=clip_id, targets=targets)
    background_tasks.add_task(
        run_autopublish,
        request=request,
        db=db,
        tenant_id=tenant_id,
        clip_id=clip_id,
        placeholder=placeholder,
    )
    return True


async def retry_autopublish(
    *, request: Request, db: Database, tenant_id: str, placeholder: str
) -> dict[str, Any]:
    """Reintentar from Publicar for a failed placeholder row: queue a fresh
    job for the same clip and retire the old row."""
    repo = ZernioPublishesRepo(db)
    row = await repo.get_by_post_id(placeholder)
    if row is None or row.tenant_id != tenant_id or not is_placeholder(placeholder):
        return {"ok": False, "error": "No encontramos esa publicación."}
    if (row.status or "") != "failed":
        return {"ok": False, "error": "Esa publicación ya se está procesando."}
    targets = [p for p in (row.platforms or "").split(",") if p]
    new = await _queue(db, tenant_id=tenant_id, clip_id=row.clip_id, targets=targets)
    await repo.set_status(placeholder, status="deleted")
    task = asyncio.create_task(
        run_autopublish(
            request=request, db=db, tenant_id=tenant_id, clip_id=row.clip_id, placeholder=new
        )
    )
    _running.add(task)
    task.add_done_callback(_running.discard)
    return {"ok": True, "post_id": new}


def _platforms_or_unknown(raw: object) -> list[str | None]:
    names: list[str | None] = [x for x in str(raw or "").split(",") if x]
    return names or [None]


async def failed_placeholders(db: Database, tenant_id: str) -> list[dict[str, Any]]:
    """Failed auto-publish jobs, in Publicar's failed-list shape."""
    conn = await db.connect()
    cur = await conn.execute(
        "SELECT post_id, clip_id, platforms, created_at FROM zernio_publishes "
        "WHERE tenant_id = ? AND status = 'failed' AND post_id LIKE ? "
        "ORDER BY created_at DESC LIMIT 50",
        (tenant_id, f"{PLACEHOLDER_PREFIX}%"),
    )
    out: list[dict[str, Any]] = []
    for r in await cur.fetchall():
        d = dict(r)
        out.append(
            {
                "post_id": d["post_id"],
                "content": "",
                "created_at": d["created_at"],
                "clip_id": d["clip_id"],
                # The real targets; when they couldn't be known (no targets
                # set and Zernio unreachable) one entry without a platform.
                "platforms": [
                    {
                        "platform": p,
                        "error": "No se pudo publicar automáticamente al aprobar.",
                        "hint": "Reintenta; si vuelve a fallar, publícalo desde Publicar.",
                    }
                    for p in _platforms_or_unknown(d.get("platforms"))
                ],
            }
        )
    return out


async def clip_publish_state(db: Database, tenant_id: str, clip_id: str) -> dict[str, Any]:
    """What the editor shows about publishing this clip."""
    conn = await db.connect()
    cur = await conn.execute(
        "SELECT post_id, status, platforms FROM zernio_publishes "
        "WHERE tenant_id = ? AND clip_id = ? "
        "AND COALESCE(status, '') NOT IN ('deleted', 'cancelled') "
        "ORDER BY created_at DESC LIMIT 10",
        (tenant_id, clip_id),
    )
    rows = [dict(r) for r in await cur.fetchall()]
    auto = next((r["status"] for r in rows if is_placeholder(str(r["post_id"]))), None)
    posts = [
        {
            "status": r["status"] or "published",
            "platforms": [p for p in str(r["platforms"] or "").split(",") if p],
        }
        for r in rows
        if not is_placeholder(str(r["post_id"]))
    ]
    return {"auto": auto, "posts": posts}


__all__ = [
    "PLACEHOLDER_PREFIX",
    "clip_publish_state",
    "enqueue_autopublish",
    "failed_placeholders",
    "is_placeholder",
    "retry_autopublish",
    "run_autopublish",
]
