"""Editor export on the worker — `run_export` against a real DB.

The recorder itself (Chromium) is swapped for real ffmpeg output; the rest is
the production path: rehydrate from the bucket, `render_clip_in_background`
(validation, outro decision, state writes), upload with the fingerprint,
scratch cleanup.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from chalybclip.clip.export_cache import current_fingerprint
from chalybclip.db import ClipsRepo, TenantsRepo
from chalybclip.integrations.storage import clip_export_key, clip_media_key
from chalybclip.tenancy import bound_tenant
from chalybclip.workers.export_job import ExportRequest, run_export

from .._db_backend import migrated_database, pg_enabled
from ..api.test_clip_render_state import _seed_clip

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


class _Store:
    def __init__(self) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self.fail_upload = False

    async def upload(self, *, local_path: Path, key: str, content_type: str | None = None,
                     content_disposition: str | None = None,
                     metadata: dict[str, str] | None = None) -> None:
        if self.fail_upload:
            raise OSError("bucket said no")
        self.objects[key] = {
            "bytes": Path(local_path).read_bytes(), "content_type": content_type,
            "content_disposition": content_disposition, "metadata": dict(metadata or {}),
        }

    async def download(self, *, key: str, dest: Path) -> Path | None:
        obj = self.objects.get(key)
        if obj is None:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(obj["bytes"])
        return dest


def _ffmpeg_clip(path: Path, seconds: float, size: str = "180x320") -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest",
         "-movflags", "+faststart", str(path)],
        check=True,
    )


@pytest_asyncio.fixture
async def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[dict[str, Any]]:
    async for db in migrated_database(tmp_path, name="w.db"):
        store = _Store()
        settings = type("S", (), {
            "database_url": os.environ.get("DATABASE_URL") if pg_enabled() else None,
            "db_path": str(tmp_path / "w.db"),
            "default_output_dir": str(tmp_path / "out"),
        })()
        monkeypatch.setattr("chalybclip.settings.get_settings", lambda s=settings: s)
        monkeypatch.setattr("chalybclip.integrations.storage.build_artifact_store", lambda _s, st=store: st)
        tenant = await TenantsRepo(db).create(name="Worker Co")
        clip_path = await _seed_clip(db, tenant.id, output_dir=tmp_path / "out")
        with bound_tenant(tenant.id):
            await ClipsRepo(db).mark_render_started("clp_test")
            started_at = (await ClipsRepo(db).get("clp_test")).render_started_at
        # The worker's /tmp never had the cut: only the bucket does.
        src = tmp_path / "source.mp4"
        _ffmpeg_clip(src, 30)
        store.objects[clip_media_key(tenant.id, "clp_test")] = {"bytes": src.read_bytes(), "metadata": {}}
        clip_path.unlink()
        yield {"db": db, "store": store, "tenant": tenant.id, "clip_path": clip_path,
               "started_at": started_at}


def _req(tenant_id: str, started_at: str = "2026-11-07T00:00:00+00:00", **over: Any) -> ExportRequest:
    base: dict[str, Any] = {
        "tenant_id": tenant_id, "clip_id": "clp_test", "quality": "1080",
        "width": 1080, "height": 1920, "base_url": "https://api.example",
        "auth_query": "tenant=t&exp=1&sig=s", "render_started_at": started_at,
        "deadline_ts": time.time() + 120,
    }
    base.update(over)
    return ExportRequest.from_payload(base)


def _fake_recorder(monkeypatch: pytest.MonkeyPatch, seen: list[dict[str, Any]]) -> None:
    """Stand in for Chromium: write a real MP4 where the recorder would."""

    async def _record(**kw: Any) -> Path:
        seen.append(kw)
        assert kw["audio_source_path"].is_file()  # rehydrated before the render
        _ffmpeg_clip(kw["output_path"], kw["duration_s"])
        return kw["output_path"]

    monkeypatch.setattr("chalybclip.clip.hybrid_recorder.record_clip_hybrid", _record)


async def _row(world: dict[str, Any]) -> Any:
    with bound_tenant(world["tenant"]):
        return await ClipsRepo(world["db"]).get("clp_test")


async def test_renders_uploads_with_fingerprint_marks_ready_and_cleans_scratch(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []
    _fake_recorder(monkeypatch, seen)

    result = await run_export(_req(world["tenant"], world["started_at"]))

    assert result == {"status": "ready", "error": None}
    (call,) = seen
    assert call["base_url"] == "https://api.example"
    assert call["auth_query"] == "tenant=t&exp=1&sig=s"
    assert call["auth_cookie_value"] is None
    assert (call["width"], call["height"]) == (1080, 1920)
    row = await _row(world)
    assert row.render_state == "ready"
    obj = world["store"].objects[clip_export_key(world["tenant"], "clp_test", resolution="1080")]
    with bound_tenant(world["tenant"]):
        fp = await current_fingerprint(world["db"], tenant_id=world["tenant"], clip=row, quality="1080")
    assert obj["metadata"] == {"fp": fp}
    assert obj["content_type"] == "video/mp4"
    assert obj["content_disposition"] == 'attachment; filename="chalybclip_clp_test.mp4"'
    assert obj["bytes"][4:8] == b"ftyp"
    # Worker /tmp is memory: render, sidecars and the pulled source are gone.
    clip_dir = world["clip_path"].parent
    assert sorted(p.name for p in clip_dir.iterdir()) == []


async def test_missing_source_in_bucket_fails_clearly(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    world["store"].objects.clear()
    seen: list[dict[str, Any]] = []
    _fake_recorder(monkeypatch, seen)
    result = await run_export(_req(world["tenant"], world["started_at"]))
    assert result["status"] == "failed"
    assert seen == []
    row = await _row(world)
    assert row.render_state == "failed"
    assert "source clip missing" in (row.render_error or "")


async def test_reset_clip_is_skipped(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    with bound_tenant(world["tenant"]):
        await ClipsRepo(world["db"]).reset_render_state("clp_test")
    seen: list[dict[str, Any]] = []
    _fake_recorder(monkeypatch, seen)
    result = await run_export(_req(world["tenant"], world["started_at"]))
    assert result["status"] == "skipped"
    assert seen == []
    assert (await _row(world)).render_state == "idle"


async def test_timeout_marks_failed_with_retry_message(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _hang(**_kw: Any) -> Path:
        await asyncio.sleep(30)
        raise AssertionError("unreachable")

    monkeypatch.setattr("chalybclip.clip.hybrid_recorder.record_clip_hybrid", _hang)
    result = await run_export(_req(world["tenant"], world["started_at"], deadline_ts=time.time() + 0.5))
    assert result["status"] == "failed"
    row = await _row(world)
    assert row.render_state == "failed"
    assert "timed out" in (row.render_error or "")


async def test_upload_failure_never_reports_ready(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    world["store"].fail_upload = True
    _fake_recorder(monkeypatch, [])
    result = await run_export(_req(world["tenant"], world["started_at"]))
    assert result["status"] == "failed"
    row = await _row(world)
    assert row.render_state == "failed"
    assert "could not store the export" in (row.render_error or "")


@pytest.mark.parametrize(
    ("field", "value"),
    [("quality", "8k"), ("clip_id", ""), ("width", 0), ("deadline_ts", "x"), ("auth_query", ""),
     ("render_started_at", "")],
)
def test_request_validation(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        _req("ten_x", **{field: value})


async def test_superseded_render_never_overwrites_the_newer_one(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overlay edit + new click mid-render: the old render must not upload,
    mark ready, or mark failed over the new generation."""
    async def _record(**kw: Any) -> Path:
        # The user edits and clicks again while this render runs.
        with bound_tenant(world["tenant"]):
            repo = ClipsRepo(world["db"])
            await repo.reset_render_state("clp_test")
            await asyncio.sleep(0.01)
            await repo.mark_render_started("clp_test")
        _ffmpeg_clip(kw["output_path"], kw["duration_s"])
        return kw["output_path"]

    monkeypatch.setattr("chalybclip.clip.hybrid_recorder.record_clip_hybrid", _record)
    result = await run_export(_req(world["tenant"], world["started_at"]))
    assert result["status"] == "skipped"
    row = await _row(world)
    assert row.render_state == "rendering"  # the newer generation, untouched
    assert row.render_started_at != world["started_at"]
    assert world["store"].objects.keys() == {clip_media_key(world["tenant"], "clp_test")}


async def test_square_export_leaves_no_format_files_behind(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-vertical exports build format bases in the clip dir; all scratch goes."""
    async def _record(**kw: Any) -> Path:
        (kw["output_path"].parent / "clip_landscape.mp4").write_bytes(b"base")
        (kw["output_path"].parent / "clip_square_100_130.mp4").write_bytes(b"fmt")
        _ffmpeg_clip(kw["output_path"], kw["duration_s"])
        return kw["output_path"]

    monkeypatch.setattr("chalybclip.clip.hybrid_recorder.record_clip_hybrid", _record)
    keep = world["clip_path"].parent / "keep_me.txt"
    keep.write_text("pipeline file")
    result = await run_export(_req(world["tenant"], world["started_at"]))
    assert result["status"] == "ready"
    assert [p.name for p in world["clip_path"].parent.iterdir()] == ["keep_me.txt"]


async def test_two_qualities_of_one_clip_run_one_after_the_other(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """They share the pulled source; the first one's cleanup must not pull it
    out from under the second."""
    active = 0
    peak = 0

    async def _render(**kw: Any) -> None:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        assert kw["audio_source_path"].is_file()
        await asyncio.sleep(0.05)
        active -= 1

    reqs = [_req(world["tenant"], world["started_at"], quality=q) for q in ("1080", "4k")]
    await asyncio.gather(*(run_export(r, render=_render) for r in reqs))
    assert peak == 1
