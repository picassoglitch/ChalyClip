"""Editor export ("Descargar video") rendered on the worker.

The API only flips the clip to `rendering` and hands this request over; the
worker pulls the cut original from the bucket, runs the same recorder the
API used to run (`render_clip_in_background`), uploads the validated MP4 to
`clip_export_key(...)`, and only then marks the clip ready. Status and
download read the DB + bucket, so any API instance can answer them and an
API deploy no longer kills a render.

Every state write is pinned to the render generation the API started
(`render_started_at`), so a render superseded by an edit + new click can't
overwrite the newer one. Files the export creates in the clip dir are
deleted afterwards: the worker's /tmp is memory.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from chalybclip.integrations.storage.keys import EXPORT_RESOLUTIONS

_log = structlog.get_logger(__name__)

# One export at a time per clip dir: they share the pulled source and the
# format base files, and the cleanup deletes what the export created.
_clip_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


@dataclass(frozen=True)
class ExportRequest:
    tenant_id: str
    clip_id: str
    quality: str
    width: int
    height: int
    base_url: str
    auth_query: str
    render_started_at: str
    # Epoch seconds by which the worker gives up (API's zombie limit minus a margin).
    deadline_ts: float

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ExportRequest:
        """Validate the API's payload; ValueError names the bad field."""
        def _s(name: str) -> str:
            v = str(payload.get(name) or "").strip()
            if not v:
                raise ValueError(f"{name} is required")
            return v

        quality = _s("quality")
        if quality not in EXPORT_RESOLUTIONS:
            raise ValueError(f"unknown quality {quality!r}")
        try:
            width = int(payload.get("width") or 0)
            height = int(payload.get("height") or 0)
            deadline_ts = float(payload.get("deadline_ts") or 0)
        except (TypeError, ValueError) as e:
            raise ValueError(f"bad numeric field: {e}") from e
        if width <= 0 or height <= 0 or deadline_ts <= 0:
            raise ValueError("width, height and deadline_ts must be positive")
        return cls(
            tenant_id=_s("tenant_id"),
            clip_id=_s("clip_id"),
            quality=quality,
            width=width,
            height=height,
            base_url=_s("base_url"),
            auth_query=_s("auth_query"),
            render_started_at=_s("render_started_at"),
            deadline_ts=deadline_ts,
        )

    @property
    def job_key(self) -> tuple[str, str, str, str]:
        return (self.tenant_id, self.clip_id, self.quality, self.render_started_at)


def _timeout_error(timeout_s: float) -> str:
    return (
        f"render timed out: no result after {int(timeout_s) // 60} min on the "
        "export worker"
    )


async def run_export(req: ExportRequest, *, render: Any = None) -> dict[str, Any]:
    """Render, upload and mark the clip. Returns `{status, error}`, where
    status is the clip's render_state afterwards (ready / failed / skipped).
    Never raises: every failure lands on the clip row for the UI to show.

    `render` is injectable for tests; the default is the real recorder."""
    async with _clip_locks[f"{req.tenant_id}/{req.clip_id}"]:
        return await _run_export_locked(req, render=render)


async def _run_export_locked(req: ExportRequest, *, render: Any) -> dict[str, Any]:
    from chalybclip.api._artifacts import rehydrate_clip
    from chalybclip.api._clip_render import _safe_unlink, render_clip_in_background
    from chalybclip.clip.export_cache import current_fingerprint, upload_export
    from chalybclip.db import ClipsRepo, Database
    from chalybclip.integrations.storage import build_artifact_store
    from chalybclip.settings import get_settings, resolve_db_target
    from chalybclip.tenancy import bound_tenant

    render = render or render_clip_in_background
    settings = get_settings()
    db_target = resolve_db_target(settings)
    store = build_artifact_store(settings)
    db = Database(db_target)
    await db.connect()
    repo = ClipsRepo(db)
    clip_dir: Path | None = None
    dir_existed = True
    before: set[str] = set()

    async def _current() -> Any:
        with bound_tenant(req.tenant_id):
            return await repo.get(req.clip_id)

    def _is_current(row: Any) -> bool:
        return (
            row is not None
            and row.render_state == "rendering"
            and row.render_started_at == req.render_started_at
        )

    async def _fail(error: str) -> dict[str, Any]:
        with bound_tenant(req.tenant_id):
            await repo.mark_render_failed(
                req.clip_id, error=error[:500], if_started_at=req.render_started_at
            )
        return {"status": "failed", "error": error}

    try:
        clip = await _current()
        if not _is_current(clip):
            # Retry, reset or a newer click happened after the API dispatched.
            return {"status": "skipped", "error": "superseded"}
        if store is None:
            return await _fail("export worker has no object storage configured")
        timeout_s = req.deadline_ts - time.time()
        if timeout_s <= 0:
            return await _fail(_timeout_error(0))

        with bound_tenant(req.tenant_id):
            fingerprint = await current_fingerprint(
                db, tenant_id=req.tenant_id, clip=clip, quality=req.quality
            )
        if fingerprint is None:
            return await _fail("No pudimos leer tu plan. Reintenta la descarga.")

        source = Path(clip.path)
        clip_dir = source.parent
        dir_existed = clip_dir.is_dir()
        before = {p.name for p in clip_dir.iterdir()} if dir_existed else set()
        if not await rehydrate_clip(
            tenant_id=req.tenant_id, clip_id=req.clip_id, clip_path=source
        ):
            return await _fail(f"source clip missing from storage: {source.name}")

        output = clip_dir / f"clip_render_{req.quality}.mp4"
        for p in (output, output.with_suffix(".ass"), output.with_suffix(".partial.mp4")):
            _safe_unlink(p)

        async def _store(path: Path) -> None:
            if not _is_current(await _current()):
                raise RuntimeError("superseded by a newer export")
            await upload_export(
                store,
                tenant_id=req.tenant_id,
                clip_id=req.clip_id,
                quality=req.quality,
                fingerprint=fingerprint,
                local_path=path,
            )

        try:
            await asyncio.wait_for(
                render(
                    clip_id=req.clip_id,
                    tenant_id=req.tenant_id,
                    duration_s=float(clip.duration_s),
                    audio_source_path=source,
                    output_path=output,
                    base_url=req.base_url,
                    auth_cookie_value=None,
                    auth_query=req.auth_query,
                    width=req.width,
                    height=req.height,
                    db_path=db_target,
                    on_rendered=_store,
                    render_started_at=req.render_started_at,
                ),
                timeout=timeout_s,
            )
        except TimeoutError:
            _log.warning("export.timeout", clip_id=req.clip_id, timeout_s=round(timeout_s))
            return await _fail(_timeout_error(timeout_s))

        fresh = await _current()
        if fresh is None or fresh.render_started_at != req.render_started_at:
            return {"status": "skipped", "error": "superseded"}
        _log.info(
            "export.finished", clip_id=req.clip_id, quality=req.quality, state=fresh.render_state
        )
        return {"status": fresh.render_state, "error": fresh.render_error}
    except Exception as e:  # the row must leave 'rendering'
        _log.exception("export.crashed", clip_id=req.clip_id)
        with contextlib.suppress(Exception):
            return await _fail(f"export worker crashed: {type(e).__name__}: {e}")
        return {"status": "failed", "error": str(e)}
    finally:
        if clip_dir is not None:
            _cleanup(clip_dir, keep=before, remove_dir=not dir_existed)
        await db.close()


def _cleanup(clip_dir: Path, *, keep: set[str], remove_dir: bool) -> None:
    """Delete what this export created in the clip dir (source, format bases,
    render, sidecars); the dir too when the export created it."""
    import shutil

    if remove_dir:
        shutil.rmtree(clip_dir, ignore_errors=True)
        return
    with contextlib.suppress(OSError):
        for p in clip_dir.iterdir():
            if p.name not in keep:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)


__all__ = ["ExportRequest", "run_export"]
