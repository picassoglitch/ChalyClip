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


async def _wants_autopublish(db: Database, tenant_id: str) -> bool:
    from chalybclip.db import AutopublishSettingsRepo
    from chalybclip.settings import get_settings

    if not get_settings().zernio_api_key:
        return False
    s = await AutopublishSettingsRepo(db).get(tenant_id)
    return bool(s and s["enabled"] and s["mode"] == "on_approve")


async def _queue(db: Database, *, tenant_id: str, clip_id: str) -> str:
    placeholder = f"{PLACEHOLDER_PREFIX}{clip_id}-{int(time.time() * 1000)}"
    with bound_tenant(tenant_id):
        await ZernioPublishesRepo(db).record(
            post_id=placeholder,
            tenant_id=tenant_id,
            clip_id=clip_id,
            platforms=[],
            content=None,
            status="pending",
            options_json=json.dumps({"kind": "autopublish"}),
        )
    return placeholder


async def run_autopublish(
    *, request: Request, db: Database, tenant_id: str, clip_id: str, placeholder: str
) -> None:
    """The job body. Never raises."""
    from chalybclip.api.routers.zernio import maybe_autopublish_on_approve

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
    if not await _wants_autopublish(db, tenant_id):
        return False
    placeholder = await _queue(db, tenant_id=tenant_id, clip_id=clip_id)
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
    new = await _queue(db, tenant_id=tenant_id, clip_id=row.clip_id)
    await repo.set_status(placeholder, status="deleted")
    task = asyncio.create_task(
        run_autopublish(
            request=request, db=db, tenant_id=tenant_id, clip_id=row.clip_id, placeholder=new
        )
    )
    _running.add(task)
    task.add_done_callback(_running.discard)
    return {"ok": True, "post_id": new}


async def failed_placeholders(db: Database, tenant_id: str) -> list[dict[str, Any]]:
    """Failed auto-publish jobs, in Publicar's failed-list shape."""
    conn = await db.connect()
    cur = await conn.execute(
        "SELECT post_id, clip_id, created_at FROM zernio_publishes "
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
                "platforms": [
                    {
                        "platform": "auto",
                        "error": "No se pudo publicar automáticamente al aprobar.",
                        "hint": "Reintenta; si vuelve a fallar, publícalo desde Publicar.",
                    }
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
