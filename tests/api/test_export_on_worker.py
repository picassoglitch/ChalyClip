"""Editor export on the worker — the API side.

With a worker configured, "Descargar video" only flips the clip to rendering
and hands the job to the worker; status and download read the DB + bucket,
so an instance that never rendered anything still serves the result.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from chalybclip.clip.export_cache import current_fingerprint
from chalybclip.db import ClipsRepo, Database
from chalybclip.integrations.storage import clip_export_key
from chalybclip.tenancy import bound_tenant

from .conftest import auth
from .test_clip_render_state import _seed_clip


class _Store:
    """In-memory ArtifactStore with object metadata."""

    def __init__(self) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self.deleted: list[str] = []

    async def upload(self, *, local_path: Path, key: str, content_type: str | None = None,
                     content_disposition: str | None = None,
                     metadata: dict[str, str] | None = None) -> None:
        self.objects[key] = {
            "bytes": Path(local_path).read_bytes(),
            "content_type": content_type,
            "content_disposition": content_disposition,
            "metadata": dict(metadata or {}),
        }

    async def metadata(self, *, key: str) -> dict[str, str] | None:
        obj = self.objects.get(key)
        return None if obj is None else dict(obj["metadata"])

    async def delete(self, *, key: str) -> None:
        self.deleted.append(key)
        self.objects.pop(key, None)

    async def presigned_url(self, *, key: str, ttl_seconds: int) -> str:
        return f"https://storage.example/{key}?sig=1"

    def public_url(self, key: str) -> str | None:
        return None


_WORKER = "https://chalybclip-worker-abc-uc.a.run.app/exports"


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    store = _Store()
    settings = type("S", (), {
        "default_output_dir": str(tmp_path),
        "db_path": str(tmp_path / "x.db"),
        "public_url": "https://chalybclip.chalyb.com",
        "object_storage_presign_ttl_s": 3600,
    })()
    monkeypatch.setattr("chalybclip.settings.get_settings", lambda: settings)
    monkeypatch.setattr("chalybclip.api.routers.dashboard._export_store", lambda: store)
    monkeypatch.setattr(
        "chalybclip.integrations.storage.build_artifact_store", lambda _s: store
    )
    monkeypatch.setattr("chalybclip.jobs.export.export_worker_url", lambda: _WORKER)
    monkeypatch.setattr(
        "chalybclip.api.routers.internal.sign_render_query",
        lambda **kw: f"tenant={kw['tenant_id']}&exp=9&sig=s&ttl={kw['ttl_seconds']}",
    )
    dispatched: list[dict[str, Any]] = []

    async def _dispatch(**kw: Any) -> None:
        dispatched.append(kw)

    monkeypatch.setattr("chalybclip.jobs.export.dispatch_export", _dispatch)

    async def _never(**_kw: Any) -> None:
        raise AssertionError("the API must not render when a worker is configured")

    monkeypatch.setattr("chalybclip.api._clip_render.render_clip_in_background", _never)
    return {"store": store, "dispatched": dispatched, "tmp": tmp_path}


async def _clip(db: Database, tenant_id: str) -> Any:
    with bound_tenant(tenant_id):
        return await ClipsRepo(db).get("clp_test")


async def _park_export(env: dict[str, Any], db: Database, tenant_id: str, q: str = "1080") -> str:
    clip = await _clip(db, tenant_id)
    key = clip_export_key(tenant_id, "clp_test", resolution=q)
    with bound_tenant(tenant_id):
        fp = await current_fingerprint(db, tenant_id=tenant_id, clip=clip, quality=q)
    env["store"].objects[key] = {
        "bytes": b"mp4", "content_type": "video/mp4", "content_disposition": None,
        "metadata": {"fp": fp},
    }
    return key


async def test_download_dispatches_to_worker_without_local_source(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]],
    env: dict[str, Any],
) -> None:
    tenant_id = tenants["alice"]["id"]
    clip_path = await _seed_clip(db, tenant_id, output_dir=env["tmp"])
    clip_path.unlink()  # the API's /tmp never had (or lost) the source

    r = await client.get(
        "/dashboard/clips/clp_test/download", headers=auth(tenants["alice"]["token"])
    )
    assert r.status_code == 202, r.text
    assert r.json()["state"] == "rendering"
    assert len(env["dispatched"]) == 1
    job = env["dispatched"][0]
    assert job["url"] == _WORKER
    assert job["clip_id"] == "clp_test" and job["tenant_id"] == tenant_id
    assert (job["quality"], job["width"], job["height"]) == ("1080", 1080, 1920)
    assert job["base_url"] == "https://chalybclip.chalyb.com"
    assert job["auth_query"].startswith(f"tenant={tenant_id}&")
    row = await _clip(db, tenant_id)
    assert row.render_state == "rendering"
    assert job["render_started_at"] == row.render_started_at
    # 30 s clip → 15 min zombie limit; the worker gives up a minute earlier.
    assert abs(job["deadline_ts"] - (time.time() + 15 * 60 - 60)) < 30


async def test_worker_refusal_fails_the_click_with_retry(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]],
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chalybclip.jobs.export import ExportDispatchError

    async def _down(**_kw: Any) -> None:
        raise ExportDispatchError("export worker unreachable: ConnectError")

    monkeypatch.setattr("chalybclip.jobs.export.dispatch_export", _down)
    tenant_id = tenants["alice"]["id"]
    await _seed_clip(db, tenant_id, output_dir=env["tmp"])

    r = await client.get(
        "/dashboard/clips/clp_test/download", headers=auth(tenants["alice"]["token"])
    )
    assert r.status_code == 409
    body = r.json()
    assert body["state"] == "failed"
    assert "unreachable" in body["error"]
    assert body["retry_url"].endswith("/render-retry?quality=1080")
    row = await _clip(db, tenant_id)
    assert row.render_state == "failed" and "unreachable" in (row.render_error or "")


async def test_finished_export_in_bucket_is_ready_and_downloads_from_any_instance(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]],
    env: dict[str, Any],
) -> None:
    tenant_id = tenants["alice"]["id"]
    await _seed_clip(db, tenant_id, output_dir=env["tmp"], render_state="ready")
    key = await _park_export(env, db, tenant_id)
    h = auth(tenants["alice"]["token"])

    s = await client.get("/dashboard/clips/clp_test/render-status", headers=h)
    assert s.json()["state"] == "ready"
    assert s.json()["download_url"].endswith("/download?quality=1080")

    d = await client.get("/dashboard/clips/clp_test/download", headers=h)
    assert d.status_code == 302
    assert d.headers["location"] == f"https://storage.example/{key}?sig=1"
    assert env["dispatched"] == []


async def test_edit_after_export_is_a_cache_miss(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]],
    env: dict[str, Any],
) -> None:
    tenant_id = tenants["alice"]["id"]
    await _seed_clip(db, tenant_id, output_dir=env["tmp"], render_state="ready")
    await _park_export(env, db, tenant_id)
    with bound_tenant(tenant_id):
        await ClipsRepo(db).set_overlay_config(
            "clp_test", overlay_config={"top_hook": {"enabled": True, "text": "nuevo"}}
        )
    h = auth(tenants["alice"]["token"])

    s = await client.get("/dashboard/clips/clp_test/render-status", headers=h)
    # 'ready' belongs to the old edit: nothing servable → idle, not a dead link.
    assert s.json()["state"] == "idle"
    d = await client.get("/dashboard/clips/clp_test/download", headers=h)
    assert d.status_code == 202
    assert len(env["dispatched"]) == 1


async def test_other_quality_in_bucket_does_not_count(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]],
    env: dict[str, Any],
) -> None:
    tenant_id = tenants["alice"]["id"]
    await _seed_clip(db, tenant_id, output_dir=env["tmp"], render_state="ready")
    await _park_export(env, db, tenant_id, q="4k")  # free tier asks for 1080
    s = await client.get(
        "/dashboard/clips/clp_test/render-status", headers=auth(tenants["alice"]["token"])
    )
    assert s.json()["state"] == "idle"


async def test_retry_drops_the_bucket_export(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]],
    env: dict[str, Any],
) -> None:
    tenant_id = tenants["alice"]["id"]
    await _seed_clip(db, tenant_id, output_dir=env["tmp"], render_state="ready")
    key = await _park_export(env, db, tenant_id)
    r = await client.post(
        "/dashboard/clips/clp_test/render-retry", headers=auth(tenants["alice"]["token"])
    )
    assert r.status_code == 200
    assert key in env["store"].deleted and key not in env["store"].objects


async def test_hub_render_link_redirects_to_bucket(
    db: Database, tenants: dict[str, dict[str, str]], env: dict[str, Any],
) -> None:
    from starlette.requests import Request

    from chalybclip.api.routers import hub_editor

    tenant_id = tenants["alice"]["id"]
    await _seed_clip(db, tenant_id, output_dir=env["tmp"], render_state="ready")
    key = await _park_export(env, db, tenant_id)
    req = Request({"type": "http", "method": "GET", "headers": [], "query_string": b""})
    with bound_tenant(tenant_id):
        resp = await hub_editor._render_media(req, db, tenant_id, "clp_test", {"q": "1080"})
    assert resp.status_code == 302
    assert resp.headers["location"] == f"https://storage.example/{key}?sig=1"


async def test_in_process_render_also_parks_export_in_bucket(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]],
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollback mode (CHALYBCLIP_EXPORT_ON_WORKER=false): the API renders, and
    the result still lands in the bucket for the other instances."""
    monkeypatch.setattr("chalybclip.jobs.export.export_worker_url", lambda: None)
    calls: list[dict[str, Any]] = []

    async def _runner(**kw: Any) -> None:
        calls.append(kw)

    monkeypatch.setattr("chalybclip.api._clip_render.render_clip_in_background", _runner)
    tenant_id = tenants["alice"]["id"]
    await _seed_clip(db, tenant_id, output_dir=env["tmp"])
    r = await client.get(
        "/dashboard/clips/clp_test/download", headers=auth(tenants["alice"]["token"])
    )
    assert r.status_code == 202
    assert env["dispatched"] == []
    (call,) = calls
    out = env["tmp"] / "out.mp4"
    out.write_bytes(b"rendered")
    await call["on_rendered"](out)
    obj = env["store"].objects[clip_export_key(tenant_id, "clp_test", resolution="1080")]
    assert obj["bytes"] == b"rendered"
    assert obj["content_disposition"] == 'attachment; filename="chalybclip_clp_test.mp4"'
    clip = await _clip(db, tenant_id)
    with bound_tenant(tenant_id):
        fp = await current_fingerprint(db, tenant_id=tenant_id, clip=clip, quality="1080")
    assert obj["metadata"]["fp"] == fp
    assert call["render_started_at"] == clip.render_started_at


async def test_in_process_upload_failure_still_serves_the_local_render(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]],
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("chalybclip.jobs.export.export_worker_url", lambda: None)
    calls: list[dict[str, Any]] = []

    async def _runner(**kw: Any) -> None:
        calls.append(kw)

    async def _boom(**_kw: Any) -> None:
        raise OSError("bucket down")

    monkeypatch.setattr("chalybclip.api._clip_render.render_clip_in_background", _runner)
    env["store"].upload = _boom
    tenant_id = tenants["alice"]["id"]
    await _seed_clip(db, tenant_id, output_dir=env["tmp"])
    await client.get("/dashboard/clips/clp_test/download", headers=auth(tenants["alice"]["token"]))
    out = env["tmp"] / "out.mp4"
    out.write_bytes(b"rendered")
    await calls[0]["on_rendered"](out)  # must not raise: the render stays ready


async def test_endpoint_without_exports_falls_back_to_in_process(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]],
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chalybclip.jobs.export import ExportWorkerMissingError

    async def _missing(**_kw: Any) -> None:
        raise ExportWorkerMissingError("export worker has no /exports (HTTP 404)")

    calls: list[dict[str, Any]] = []

    async def _runner(**kw: Any) -> None:
        calls.append(kw)

    monkeypatch.setattr("chalybclip.jobs.export.dispatch_export", _missing)
    monkeypatch.setattr("chalybclip.api._clip_render.render_clip_in_background", _runner)
    tenant_id = tenants["alice"]["id"]
    await _seed_clip(db, tenant_id, output_dir=env["tmp"])
    r = await client.get(
        "/dashboard/clips/clp_test/download", headers=auth(tenants["alice"]["token"])
    )
    assert r.status_code == 202
    assert len(calls) == 1
    assert (await _clip(db, tenant_id)).render_state == "rendering"


async def test_bucket_export_of_one_quality_downloads_while_another_renders(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]],
    env: dict[str, Any],
) -> None:
    """render_state is per clip: a 4K render in flight must not hide the finished 1080."""
    tenant_id = tenants["alice"]["id"]
    await _seed_clip(db, tenant_id, output_dir=env["tmp"], render_state="rendering")
    key = await _park_export(env, db, tenant_id)
    d = await client.get("/dashboard/clips/clp_test/download", headers=auth(tenants["alice"]["token"]))
    assert d.status_code == 302
    assert d.headers["location"] == f"https://storage.example/{key}?sig=1"
