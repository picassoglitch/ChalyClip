"""/api/hub/v1 core — jobs, uploads, streams, live, sources, brand kits,
personas, calibration and admin pages, as the hub calls them."""

from __future__ import annotations

import datetime as _dt
import io
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from chalybclip.api.routers import hub_streams
from chalybclip.db import (
    BrandKitsRepo,
    ClipsRepo,
    Database,
    EventsRepo,
    PersonasRepo,
    StreamsRepo,
    TenantsRepo,
)
from chalybclip.db.models import ClipRow, StreamRow
from chalybclip.integrations.chalyb.admission import AdmissionRefused
from chalybclip.integrations.chalyb.sso import sign_sso_token
from chalybclip.settings import get_settings
from chalybclip.tenancy import bound_tenant

_SECRET = "hub_core_secret"


@pytest.fixture(autouse=True)
def hub_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setenv("CHALYB_SSO_SECRET", _SECRET)
    monkeypatch.setenv("CHALYBCLIP_DEFAULT_OUTPUT_DIR", str(tmp_path / "out"))
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


def hdr(tenant_id: str, tier: str = "pro") -> dict[str, str]:
    tok = sign_sso_token(
        user_id=f"u_{tenant_id}", email="a@example.com", tenant_id=tenant_id,
        secret=_SECRET, tier=tier,
    )
    return {"Authorization": f"Bearer {tok}"}


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


async def _persona(db: Database, tenant_id: str, pid: str = "default") -> None:
    with bound_tenant(tenant_id):
        if await PersonasRepo(db).get(pid) is None:
            await PersonasRepo(db).create(
                persona_id=pid, name="Default", primary_language="es",
                target_languages=["es"], voice_prompt="x", routing_tags=[],
            )


async def _stream(
    db: Database, tenant_id: str, tmp_path: Path, *, sid: str, status: str = "done",
    clips: list[tuple[str, str]] | None = None,
) -> None:
    src = tmp_path / f"{sid}.mp4"
    src.write_bytes(b"\x00" * 64)
    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=sid, tenant_id=tenant_id, vod_url="https://kick.com/x", platform="kick",
                title="Mi stream", channel="aldo", duration_s=600.0,
                source_video_path=str(src), source_audio_path="/nope.wav",
                status=status, created_at=_now(),
            )
        )
        rows = []
        for cid, st in clips or []:
            f = tmp_path / f"{cid}.mp4"
            f.write_bytes(b"clip-" + cid.encode())
            rows.append(
                ClipRow(
                    id=cid, stream_id=sid, tenant_id=tenant_id, candidate_id=None,
                    start_s=10.0, end_s=40.0, duration_s=30.0, width=1080, height=1920,
                    path=str(f), status=st, created_at=_now(), publishability_score=70,
                )
            )
        if rows:
            await ClipsRepo(db).upsert_many(rows)


def _err(r: httpx.Response, status: int, code: str, reason: str | None = None) -> None:
    assert r.status_code == status, r.text
    e = r.json()["error"]
    assert e["code"] == code, e
    if reason is not None:
        assert e["reason"] == reason, e


# ── start + URL jobs ────────────────────────────────────────────────────


async def test_start_reports_personas_and_upload_availability(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    alice = tenants["alice"]["id"]
    r = await client.get("/api/hub/v1/start", headers=hdr(alice))
    assert r.status_code == 200
    body = r.json()
    assert body["has_persona"] is False
    assert body["upload"]["enabled"] is False  # no bucket in tests
    await _persona(db, alice)
    body = (await client.get("/api/hub/v1/start", headers=hdr(alice))).json()
    assert body["default_persona_id"] == "default"


async def test_url_job_validates_and_creates_pending_stream(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    alice = tenants["alice"]["id"]
    r = await client.post(
        "/api/hub/v1/url-jobs", json={"vod_url": "https://kick.com/v/1"}, headers=hdr(alice)
    )
    _err(r, 400, "bad_request", "no_persona")
    await _persona(db, alice)
    r = await client.post(
        "/api/hub/v1/url-jobs", json={"vod_url": "https://example.com/v"}, headers=hdr(alice)
    )
    _err(r, 400, "bad_request", "link_unsupported")
    r = await client.post(
        "/api/hub/v1/url-jobs",
        json={"vod_url": "https://kick.com/video/abc", "persona_id": "nope"},
        headers=hdr(alice),
    )
    _err(r, 400, "bad_request", "unknown_persona")
    r = await client.post(
        "/api/hub/v1/url-jobs", json={"vod_url": " https://kick.com/video/abc "}, headers=hdr(alice)
    )
    assert r.status_code == 200, r.text
    sid = r.json()["stream_id"]
    with bound_tenant(alice):
        row = await StreamsRepo(db).get(sid)
        events = await EventsRepo(db).list_for_tenant()
    assert row is not None and row.status == "pending" and row.vod_url == "https://kick.com/video/abc"
    assert any(e.type == "stream.created" and e.payload["stream_id"] == sid for e in events)


async def test_admission_refusals_keep_their_reason(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alice = tenants["alice"]["id"]
    await _persona(db, alice)
    for reason, status, code in (
        ("minutes_cap", 429, "rate_limited"),
        ("no_tokens", 402, "no_credits"),
        ("video_too_long", 413, "too_large"),
    ):
        async def refuse(*_a: Any, _r: str = reason, **_k: Any) -> Any:
            raise AdmissionRefused(_r)

        monkeypatch.setattr(hub_streams, "admit_job", refuse)
        r = await client.post(
            "/api/hub/v1/url-jobs", json={"vod_url": "https://kick.com/video/a"}, headers=hdr(alice)
        )
        _err(r, status, code, reason)
        assert r.json()["error"]["message"]  # Spanish copy from the admission table


async def test_paused_tenant_cannot_start_work(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    alice = tenants["alice"]["id"]
    await _persona(db, alice)
    await TenantsRepo(db).set_status(alice, "paused")
    r = await client.post(
        "/api/hub/v1/url-jobs", json={"vod_url": "https://kick.com/video/a"}, headers=hdr(alice)
    )
    _err(r, 423, "forbidden", "tenant_paused")


# ── streams ─────────────────────────────────────────────────────────────


async def test_streams_list_detail_progress_and_media(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]], tmp_path: Path
) -> None:
    alice = tenants["alice"]["id"]
    await _persona(db, alice)
    await _stream(
        db, alice, tmp_path, sid="str_a1",
        clips=[("clp_a1", "approved"), ("clp_a2", "cut")],
    )
    r = await client.get("/api/hub/v1/streams", headers=hdr(alice))
    assert r.status_code == 200
    [s] = r.json()["streams"]
    assert (s["id"], s["clip_count"], s["ready_count"], s["status_kind"]) == (
        "str_a1", 2, 1, "complete",
    )
    d = (await client.get("/api/hub/v1/streams/str_a1", headers=hdr(alice))).json()
    assert d["overview"]["total_clips"] == 2
    assert d["pending_approval_count"] == 1
    assert d["cost"]["estimate_tokens"] > 0
    assert "usd_micros" not in d["cost"]  # never USD for creators
    assert d["progress"]["steps"][0]["name"] == "ingest"
    clip = next(c for c in d["clips"] if c["id"] == "clp_a1")
    media = await client.get(clip["media_url"])
    assert media.status_code == 200 and media.content == b"clip-clp_a1"
    assert (await client.get(d["source_url"])).status_code == 200
    p = (await client.get("/api/hub/v1/streams/str_a1/progress", headers=hdr(alice))).json()
    assert p["clip_count"] == 2


async def test_progress_ignores_failures_before_a_rerun(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]], tmp_path: Path
) -> None:
    alice = tenants["alice"]["id"]
    await _stream(db, alice, tmp_path, sid="str_f1", status="failed")
    with bound_tenant(alice):
        await EventsRepo(db).emit(
            type="pipeline.failed",
            payload={"stream_id": "str_f1", "error": "ingest failed: yt-dlp", "error_type": "X"},
        )
    p = (await client.get("/api/hub/v1/streams/str_f1/progress", headers=hdr(alice))).json()
    assert p["has_failed"] and p["failure"]["kind"] == "source"
    assert "error" not in p["failure"]  # operator text only for engine admins
    with bound_tenant(alice):
        await EventsRepo(db).emit(type="stream.rerun_requested", payload={"stream_id": "str_f1"})
    p = (await client.get("/api/hub/v1/streams/str_f1/progress", headers=hdr(alice))).json()
    assert p["failure"] is None and p["is_running"]


async def test_other_tenant_sees_nothing(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]], tmp_path: Path
) -> None:
    alice, bob = tenants["alice"]["id"], tenants["bob"]["id"]
    await _persona(db, bob)
    await _stream(db, alice, tmp_path, sid="str_x1", clips=[("clp_x1", "approved")])
    assert (await client.get("/api/hub/v1/streams", headers=hdr(bob))).json()["streams"] == []
    assert (await client.get("/api/hub/v1/clips", headers=hdr(bob))).json()["clips"] == []
    for method, path, body in (
        ("GET", "/api/hub/v1/streams/str_x1", None),
        ("GET", "/api/hub/v1/streams/str_x1/progress", None),
        ("POST", "/api/hub/v1/streams/str_x1/rerun", {"persona_id": "default"}),
        ("POST", "/api/hub/v1/streams/str_x1/approve-all", None),
        ("POST", "/api/hub/v1/streams/str_x1/download-approved", None),
        ("POST", "/api/hub/v1/streams/str_x1/settle", None),
        ("DELETE", "/api/hub/v1/streams/str_x1", None),
    ):
        r = await client.request(method, path, json=body, headers=hdr(bob))
        _err(r, 404, "not_found")
    with bound_tenant(alice):
        assert await StreamsRepo(db).get("str_x1") is not None


async def test_approve_all_zip_delete(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]], tmp_path: Path
) -> None:
    alice = tenants["alice"]["id"]
    await _stream(
        db, alice, tmp_path, sid="str_z1",
        clips=[("clp_z1", "cut"), ("clp_z2", "ready_for_review"), ("clp_z3", "rejected")],
    )
    r = await client.post("/api/hub/v1/streams/str_z1/download-approved", headers=hdr(alice))
    _err(r, 404, "not_found", "none_approved")
    r = await client.post("/api/hub/v1/streams/str_z1/approve-all", headers=hdr(alice))
    assert r.json() == {"approved": 2}
    url = (
        await client.post("/api/hub/v1/streams/str_z1/download-approved", headers=hdr(alice))
    ).json()["url"]
    z = await client.get(url)
    assert z.status_code == 200 and "attachment" in z.headers["content-disposition"]
    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert len(names) == 2
    with bound_tenant(alice):
        await StreamsRepo(db).set_status("str_z1", status="running")
    _err(await client.delete("/api/hub/v1/streams/str_z1", headers=hdr(alice)), 409, "conflict", "running")
    with bound_tenant(alice):
        await StreamsRepo(db).set_status("str_z1", status="done")
    r = await client.delete("/api/hub/v1/streams/str_z1", headers=hdr(alice))
    assert r.json() == {"deleted": True}
    with bound_tenant(alice):
        assert await StreamsRepo(db).get("str_z1") is None


async def test_settle_once_per_run(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]], tmp_path: Path
) -> None:
    alice = tenants["alice"]["id"]
    await _stream(db, alice, tmp_path, sid="str_s1")
    url = "/api/hub/v1/streams/str_s1/settle"
    [row] = (await client.get("/api/hub/v1/streams", headers=hdr(alice))).json()["streams"]
    assert row["settled"] is False
    assert (await client.post(url, headers=hdr(alice))).json() == {"first": True}
    [row] = (await client.get("/api/hub/v1/streams", headers=hdr(alice))).json()["streams"]
    assert row["settled"] is True
    assert (await client.post(url, headers=hdr(alice))).json() == {"first": False}
    with bound_tenant(alice):
        await EventsRepo(db).emit(type="stream.rerun_requested", payload={"stream_id": "str_s1"})
    assert (await client.post(url, headers=hdr(alice))).json() == {"first": True}


async def test_rerun_refuses_unknown_persona_and_gone_source(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]], tmp_path: Path
) -> None:
    alice = tenants["alice"]["id"]
    await _persona(db, alice)
    await _stream(db, alice, tmp_path, sid="str_r1")
    r = await client.post(
        "/api/hub/v1/streams/str_r1/rerun", json={"persona_id": "ghost"}, headers=hdr(alice)
    )
    _err(r, 400, "bad_request", "unknown_persona")
    # URL source with no on-disk stream.json → re-fetch stub, admitted, queued.
    r = await client.post(
        "/api/hub/v1/streams/str_r1/rerun", json={"persona_id": "default"}, headers=hdr(alice)
    )
    assert r.status_code == 200, r.text
    with bound_tenant(alice):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_r2", tenant_id=alice, vod_url="upload://a.mp4", platform="upload",
                duration_s=1.0, source_video_path="/gone.mp4", source_audio_path="/gone.wav",
                status="failed", created_at=_now(),
            )
        )
    r = await client.post(
        "/api/hub/v1/streams/str_r2/rerun", json={"persona_id": "default"}, headers=hdr(alice)
    )
    _err(r, 409, "conflict", "source_gone")


# ── uploads ─────────────────────────────────────────────────────────────


class FakeStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.deleted: list[str] = []

    async def presigned_put_url(self, *, key: str, ttl_seconds: int, content_type: str) -> str:
        return f"https://bucket.test/{key}?sig=put"

    async def presigned_url(self, *, key: str, ttl_seconds: int) -> str:
        return f"https://bucket.test/{key}?sig=get"

    async def size(self, *, key: str) -> int | None:
        return len(self.objects[key]) if key in self.objects else None

    async def download(self, *, key: str, dest: Path) -> Path | None:
        if key not in self.objects:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.objects[key])
        return dest

    async def delete(self, *, key: str) -> None:
        self.deleted.append(key)
        self.objects.pop(key, None)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    import chalybclip.api._pipeline as pipeline_mod
    import chalybclip.integrations.storage as storage_mod

    s = FakeStore()
    monkeypatch.setattr(storage_mod, "build_artifact_store", lambda _settings: s)
    monkeypatch.setattr(hub_streams, "_probe_minutes_url", lambda _url: 3.0)
    runs: list[dict[str, Any]] = []

    async def fake_runner(**kw: Any) -> None:
        runs.append(kw)

    monkeypatch.setattr(pipeline_mod, "upload_pipeline_runner", fake_runner)
    s.runs = runs  # type: ignore[attr-defined]
    return s


async def test_upload_flow(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]], store: FakeStore
) -> None:
    alice, bob = tenants["alice"]["id"], tenants["bob"]["id"]
    await _persona(db, alice)
    body = {"filename": "mi video.mp4", "size_bytes": 1000, "content_type": "video/mp4"}
    _err(
        await client.post(
            "/api/hub/v1/uploads", json={**body, "content_type": "text/html"}, headers=hdr(alice)
        ),
        400, "bad_request", "bad_type",
    )
    r = await client.post("/api/hub/v1/uploads", json=body, headers=hdr(alice))
    assert r.status_code == 200, r.text
    up = r.json()
    sid, ticket = up["stream_id"], up["ticket"]
    assert up["method"] == "PUT" and up["upload_url"].startswith("https://bucket.test/uploads/")
    assert up["headers"] == {"Content-Type": "video/mp4"}
    done = f"/api/hub/v1/uploads/{sid}/complete"
    # Nothing reached the bucket yet.
    _err(await client.post(done, json={"ticket": ticket}, headers=hdr(alice)), 400, "bad_request", "not_uploaded")
    # Tampered ticket / another tenant's call.
    _err(await client.post(done, json={"ticket": ticket[:-2] + "00"}, headers=hdr(alice)), 400, "bad_request", "bad_ticket")
    _err(await client.post(done, json={"ticket": ticket}, headers=hdr(bob)), 404, "not_found")
    key = next(iter(k for k in [up["upload_url"].split("bucket.test/")[1].split("?")[0]]))
    store.objects[key] = b"\x00" * 1000
    r = await client.post(done, json={"ticket": ticket}, headers=hdr(alice))
    assert r.status_code == 200, r.text
    with bound_tenant(alice):
        row = await StreamsRepo(db).get(sid)
    assert row is not None and row.platform == "upload" and row.title == "mi video.mp4"
    assert store.runs and store.runs[0]["stream_id"] == sid  # type: ignore[attr-defined]
    assert key in store.deleted  # pulled here, bucket copy dropped
    # A repeated complete is a no-op.
    assert (await client.post(done, json={"ticket": ticket}, headers=hdr(alice))).status_code == 200


async def test_upload_abort_and_unavailable(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    store: FakeStore,
) -> None:
    alice = tenants["alice"]["id"]
    await _persona(db, alice)
    up = (
        await client.post(
            "/api/hub/v1/uploads",
            json={"filename": "a.mov", "size_bytes": 10, "content_type": "video/quicktime"},
            headers=hdr(alice),
        )
    ).json()
    r = await client.post(
        f"/api/hub/v1/uploads/{up['stream_id']}/abort", json={"ticket": up["ticket"]}, headers=hdr(alice)
    )
    assert r.json() == {"ok": True}


async def test_upload_needs_object_storage(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    alice = tenants["alice"]["id"]
    await _persona(db, alice)
    r = await client.post(
        "/api/hub/v1/uploads",
        json={"filename": "a.mp4", "size_bytes": 10, "content_type": "video/mp4"},
        headers=hdr(alice),
    )
    _err(r, 503, "unavailable", "no_storage")


# ── settings surfaces ───────────────────────────────────────────────────


async def test_live_key_rotate(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    alice = tenants["alice"]["id"]
    r1 = (await client.post("/api/hub/v1/live/rotate-key", headers=hdr(alice))).json()
    r2 = (await client.post("/api/hub/v1/live/rotate-key", headers=hdr(alice))).json()
    assert r1["key"]["value"] != r2["key"]["value"]
    assert (await client.get("/api/hub/v1/live", headers=hdr(alice))).json()["key"]["value"] == r2["key"]["value"]
    _err(
        await client.post("/api/hub/v1/live/connection", json={"enabled": True}, headers=hdr(alice)),
        403, "needs_plan",
    )


async def test_sources_crud(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    alice, bob = tenants["alice"]["id"], tenants["bob"]["id"]
    await _persona(db, alice)
    body = {"channel_url": "https://www.youtube.com/@creator/videos", "persona_id": "default"}
    r = await client.post("/api/hub/v1/sources", json=body, headers=hdr(alice))
    assert r.status_code == 200, r.text
    wid = r.json()["watch"]["id"]
    assert r.json()["watch"]["platform"] == "youtube"
    _err(await client.post("/api/hub/v1/sources", json=body, headers=hdr(alice)), 409, "conflict", "already_watching")
    r = await client.patch(
        f"/api/hub/v1/sources/{wid}", json={"enabled": False, "polls_per_day": 6}, headers=hdr(alice)
    )
    assert r.json()["watch"]["enabled"] is False and r.json()["watch"]["polls_per_day"] == 6
    _err(await client.patch(f"/api/hub/v1/sources/{wid}", json={"enabled": True}, headers=hdr(bob)), 404, "not_found")
    _err(await client.delete(f"/api/hub/v1/sources/{wid}", headers=hdr(bob)), 404, "not_found")
    assert (await client.delete(f"/api/hub/v1/sources/{wid}", headers=hdr(alice))).json() == {"deleted": True}


async def test_personas_crud(client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]) -> None:
    alice = tenants["alice"]["id"]
    p = {
        "id": "gamer", "name": "Gamer", "primary_language": "es", "voice_prompt": "Hola",
        "target_languages": ["es", "en"], "routing_tags": ["gaming"],
    }
    assert (await client.post("/api/hub/v1/personas", json=p, headers=hdr(alice))).status_code == 200
    _err(await client.post("/api/hub/v1/personas", json=p, headers=hdr(alice)), 409, "conflict")
    _err(
        await client.post("/api/hub/v1/personas", json={**p, "id": "bad id!"}, headers=hdr(alice)),
        422, "bad_request",
    )
    r = await client.put(
        "/api/hub/v1/personas/gamer", json={**p, "name": "Gamer 2"}, headers=hdr(alice)
    )
    assert r.json()["persona"]["name"] == "Gamer 2"
    names = [x["id"] for x in (await client.get("/api/hub/v1/personas", headers=hdr(alice))).json()["personas"]]
    assert "gamer" in names


async def test_brand_kits_crud_and_logo(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alice, bob = tenants["alice"]["id"], tenants["bob"]["id"]
    kit = {
        "name": "Aldo", "caption_preset": "bold_block", "handle_kick": "aldo",
        "forward_phrases": ["córtalo", " "], "retroactive_phrases": ["eso"],
        "safety_preset": "cautious", "content_timezone": "America/Mexico_City",
        "show_chalybclip_outro": False,
    }
    _err(
        await client.post("/api/hub/v1/brand-kits", json={**kit, "caption_preset": "x"}, headers=hdr(alice)),
        400, "bad_request",
    )
    _err(
        await client.post("/api/hub/v1/brand-kits", json={**kit, "content_timezone": "Mars/Base"}, headers=hdr(alice)),
        400, "bad_request",
    )
    r = await client.post("/api/hub/v1/brand-kits", json=kit, headers=hdr(alice))
    assert r.status_code == 200, r.text
    k = r.json()["kit"]
    assert k["forward_phrases"] == ["córtalo"] and k["safety_preset"] == "cautious"
    assert k["show_chalybclip_outro"] is False and k["caption_preset"] == "bold_block"
    # Free tier keeps the end card whatever is sent.
    r = await client.post("/api/hub/v1/brand-kits", json=kit, headers=hdr(alice, tier="free"))
    assert r.json()["kit"]["show_chalybclip_outro"] is True
    kid = k["id"]
    r = await client.put(f"/api/hub/v1/brand-kits/{kid}", json={**kit, "name": "Aldo 2"}, headers=hdr(alice))
    assert r.json()["kit"]["name"] == "Aldo 2"
    _err(await client.get(f"/api/hub/v1/brand-kits/{kid}", headers=hdr(bob)), 404, "not_found")
    assert (await client.post(f"/api/hub/v1/brand-kits/{kid}/default", headers=hdr(alice))).json() == {"ok": True}

    import chalybclip.branding as branding

    class Logo:
        svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect width="10" height="10"/></svg>'

    async def fake_logo(**_kw: Any) -> Logo:
        return Logo()

    monkeypatch.setattr(branding, "generate_logo", fake_logo)
    monkeypatch.setattr(branding, "rasterize_svg_to_png", lambda *_a, **_k: None)
    r = await client.post(f"/api/hub/v1/brand-kits/{kid}/logo", json={"style_hint": "Monograma"}, headers=hdr(alice))
    assert r.status_code == 200, r.text
    assert r.json()["kit"]["logo_svg"].startswith("<svg") and r.json()["kit"]["ai_prompt"] == "Monograma"
    lst = (await client.get("/api/hub/v1/brand-kits", headers=hdr(alice))).json()
    assert "bold_block" in lst["caption_presets"] and len(lst["kits"]) == 2
    assert (await client.delete(f"/api/hub/v1/brand-kits/{kid}", headers=hdr(alice))).json() == {"deleted": True}
    with bound_tenant(alice):
        assert await BrandKitsRepo(db).get(kid) is None


async def test_calibration_and_admin_pages(
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alice = tenants["alice"]["id"]
    r = await client.get("/api/hub/v1/calibration", headers=hdr(alice))
    assert [x["platform"] for x in r.json()["reports"]] == ["youtube", "tiktok", "instagram", "buffer"]
    for path in ("/api/hub/v1/admin/llm-calls", "/api/hub/v1/admin/llm-settings", "/api/hub/v1/admin/site-settings"):
        _err(await client.get(path, headers=hdr(alice)), 404, "not_found")
    monkeypatch.setenv("CHALYBCLIP_ADMIN_TENANT_IDS", alice)
    get_settings.cache_clear()
    assert (await client.get("/api/hub/v1/me", headers=hdr(alice))).json()["is_admin"] is True
    assert (await client.get("/api/hub/v1/admin/llm-calls", headers=hdr(alice))).status_code == 200
    assert (await client.get("/api/hub/v1/admin/llm-settings", headers=hdr(alice))).status_code == 200
    r = await client.put(
        "/api/hub/v1/admin/site-settings", json={"price_es": " $99 ", "price_en": ""}, headers=hdr(alice)
    )
    assert r.json() == {"price_es": "$99", "price_en": ""}
    assert (await client.get("/api/hub/v1/admin/site-settings", headers=hdr(alice))).json()["price_es"] == "$99"


async def test_old_runs_are_never_announced(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]], tmp_path: Path
) -> None:
    """History the hub finds after cutover counts as settled (no notice)."""
    alice = tenants["alice"]["id"]
    old = (_dt.datetime.now(_dt.UTC) - _dt.timedelta(days=3)).isoformat()
    with bound_tenant(alice):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_old", tenant_id=alice, vod_url="https://kick.com/x", platform="kick",
                duration_s=1.0, source_video_path="/x.mp4", source_audio_path="/x.wav",
                status="done", created_at=old,
            )
        )
    [row] = (await client.get("/api/hub/v1/streams", headers=hdr(alice))).json()["streams"]
    assert row["settled"] is True
    r = await client.post("/api/hub/v1/streams/str_old/settle", headers=hdr(alice))
    assert r.json() == {"first": False}


async def test_persona_id_taken_by_another_account_is_409(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    """personas.id is a global primary key: an id another tenant already
    uses must answer 409, not 500."""
    alice, bob = tenants["alice"]["id"], tenants["bob"]["id"]
    p = {"id": "shared", "name": "X", "primary_language": "es", "voice_prompt": "v"}
    assert (await client.post("/api/hub/v1/personas", json=p, headers=hdr(alice))).status_code == 200
    r = await client.post("/api/hub/v1/personas", json=p, headers=hdr(bob))
    _err(r, 409, "conflict", "exists")


async def test_progress_marks_one_failed_step(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]], tmp_path: Path
) -> None:
    alice = tenants["alice"]["id"]
    await _stream(db, alice, tmp_path, sid="str_f2", status="failed")
    with bound_tenant(alice):
        await EventsRepo(db).emit(type="pipeline.step.start", payload={"stream_id": "str_f2", "step": "ingest"})
        await EventsRepo(db).emit(
            type="pipeline.step.failed",
            payload={"stream_id": "str_f2", "step": "ingest", "error": "yt-dlp: 404"},
        )
        await EventsRepo(db).emit(
            type="pipeline.failed", payload={"stream_id": "str_f2", "error": "IngestError: yt-dlp"}
        )
    p = (await client.get("/api/hub/v1/streams/str_f2/progress", headers=hdr(alice))).json()
    assert [s["status"] for s in p["steps"]][:2] == ["failed", "pending"]
    assert p["steps"][0]["error_kind"] == "source"
    assert "error" not in p["steps"][0]
