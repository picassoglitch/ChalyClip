"""Hands editor exports to the worker (`POST /exports` on `chalybclip worker`).

The API service is CPU-throttled and its /tmp is per instance, so it no
longer renders exports itself when a worker is configured. Dispatch waits
for the worker's `accepted` line (so an unreachable worker fails the click
right away), then a background task drains the stream until `done`. The
worker writes progress, ready/failed and the bucket object; nothing here is
needed for status or download.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import httpx
import structlog

from chalybclip.errors import ChalybClipError
from chalybclip.integrations.gcp_identity import CloudRunIdTokenAuth, is_cloud_run_url

_log = structlog.get_logger(__name__)

# Covers a scale-from-zero worker start (image with Chromium) plus the token fetch.
_ACCEPT_TIMEOUT_S = 90.0

# Strong refs so the drain tasks aren't garbage-collected mid-stream.
_drains: set[asyncio.Task[None]] = set()


class ExportDispatchError(ChalybClipError):
    """The worker never accepted the export (unreachable, refused, misconfigured)."""


class ExportWorkerMissingError(ExportDispatchError):
    """The endpoint has no /exports route (an older worker or the Modal app):
    the API renders in-process instead."""


def export_worker_url() -> str | None:
    """The worker's export endpoint, or None → render in-process (old path).

    Requires the same worker the pipeline already dispatches to, plus object
    storage: the bucket is how the result gets back to the API."""
    from chalybclip.integrations.storage import build_artifact_store
    from chalybclip.settings import get_settings

    settings = get_settings()
    if not getattr(settings, "export_on_worker", False):
        return None
    if (getattr(settings, "job_dispatcher", "") or "").strip().lower() != "modal":
        return None
    endpoint = (getattr(settings, "modal_pipeline_endpoint_url", None) or "").strip()
    token = (getattr(settings, "modal_token", None) or "").strip()
    if not endpoint or not token or build_artifact_store(settings) is None:
        return None
    if ".modal.run" in endpoint:
        # The Modal pipeline app only serves the pipeline kickoff (no Chromium, no /exports).
        return None
    return endpoint.rstrip("/") + "/exports"


async def dispatch_export(
    *,
    url: str,
    tenant_id: str,
    clip_id: str,
    quality: str,
    width: int,
    height: int,
    base_url: str,
    auth_query: str,
    render_started_at: str,
    deadline_ts: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    """Start the export on the worker. Returns once the worker accepted it;
    raises ExportDispatchError otherwise."""
    from chalybclip.settings import get_settings

    payload: dict[str, Any] = {
        "auth_token": (get_settings().modal_token or "").strip(),
        "tenant_id": tenant_id,
        "clip_id": clip_id,
        "quality": quality,
        "width": width,
        "height": height,
        "base_url": base_url,
        "auth_query": auth_query,
        "render_started_at": render_started_at,
        "deadline_ts": deadline_ts,
    }
    auth = CloudRunIdTokenAuth(url) if is_cloud_run_url(url) else None
    # Read timeout spans the heartbeat gap, not the render; the render's own
    # deadline is enforced on the worker.
    client = httpx.AsyncClient(
        auth=auth,
        transport=transport,
        timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0),
    )
    resp: httpx.Response | None = None
    try:
        resp = await asyncio.wait_for(
            client.send(client.build_request("POST", url, json=payload), stream=True),
            timeout=_ACCEPT_TIMEOUT_S,
        )
        if resp.status_code in (404, 405):
            raise ExportWorkerMissingError(f"export worker has no /exports (HTTP {resp.status_code})")
        if resp.status_code != 200:
            body = (await resp.aread()).decode("utf-8", "replace")[:300]
            raise ExportDispatchError(f"export worker answered HTTP {resp.status_code}: {body}")
        lines = resp.aiter_lines()
        first = await asyncio.wait_for(_next_event(lines), timeout=_ACCEPT_TIMEOUT_S)
        if (first or {}).get("event") != "accepted":
            raise ExportDispatchError(f"export worker did not accept the job: {first!r}")
    except ExportDispatchError:
        await _close(resp, client)
        raise
    except (TimeoutError, httpx.HTTPError) as e:
        await _close(resp, client)
        raise ExportDispatchError(
            f"export worker unreachable: {type(e).__name__}: {e}"
        ) from e

    task = asyncio.create_task(_drain(lines, resp, client, clip_id=clip_id, quality=quality))
    _drains.add(task)
    task.add_done_callback(_drains.discard)
    _log.info("export.dispatched", clip_id=clip_id, quality=quality)


async def _next_event(lines: Any) -> dict[str, Any] | None:
    async for line in lines:
        line = line.strip()
        if line:
            try:
                parsed = json.loads(line)
            except ValueError:
                return {"event": "garbled", "line": line[:200]}
            return parsed if isinstance(parsed, dict) else {"event": "garbled"}
    return None


async def _drain(
    lines: Any, resp: httpx.Response, client: httpx.AsyncClient, *, clip_id: str, quality: str
) -> None:
    """Hold the stream open until the worker's `done`. A broken stream is only
    logged: the worker owns the clip row, and the zombie timeout covers a dead one."""
    try:
        while True:
            event = await _next_event(lines)
            if event is None:
                _log.warning("export.stream_ended_early", clip_id=clip_id, quality=quality)
                return
            if event.get("event") == "done":
                _log.info(
                    "export.done", clip_id=clip_id, quality=quality,
                    status=event.get("status"), error=(str(event.get("error") or "")[:200] or None),
                )
                return
    except Exception as e:  # observability only
        _log.warning("export.stream_lost", clip_id=clip_id, quality=quality, error=str(e)[:200])
    finally:
        await _close(resp, client)


async def _close(resp: httpx.Response | None, client: httpx.AsyncClient) -> None:
    with contextlib.suppress(Exception):
        if resp is not None:
            await resp.aclose()
    with contextlib.suppress(Exception):
        await client.aclose()


async def drain_exports() -> None:
    """Await every in-flight stream (tests)."""
    if _drains:
        await asyncio.gather(*list(_drains), return_exceptions=True)


__all__ = [
    "ExportDispatchError",
    "ExportWorkerMissingError",
    "dispatch_export",
    "drain_exports",
    "export_worker_url",
]
