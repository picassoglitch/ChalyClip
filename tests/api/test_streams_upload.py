"""POST /streams/upload — multipart upload bypasses yt-dlp."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from chalybclip.api import create_app
from chalybclip.db import Database, StreamsRepo
from chalybclip.tenancy import bound_tenant

from .conftest import auth


def _stub_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real ffmpeg/ffprobe in CI — short-circuit both shell-outs and the
    PATH-availability check that gates the upload endpoints."""

    def fake_extract_audio(_video: Path, audio: Path) -> None:
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"\x00fakeaudio")

    def fake_ffprobe(_video: Path) -> float:
        return 17.5

    monkeypatch.setattr("chalybclip.ingest.service._extract_audio", fake_extract_audio)
    monkeypatch.setattr("chalybclip.ingest.service._ffprobe_duration", fake_ffprobe)
    # The handlers call `is_ffmpeg_available()` to gate the upload — stub
    # it in both modules where it's looked up so the fast-path returns True.
    monkeypatch.setattr("chalybclip.ingest.service.is_ffmpeg_available", lambda: True)
    monkeypatch.setattr("chalybclip.ingest.is_ffmpeg_available", lambda: True)


async def test_upload_creates_stream_and_kicks_off_pipeline(
    db: Database,
    tenants: dict[str, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Multipart POST writes the file, runs ingest_uploaded, persists, schedules pipeline."""
    from chalybclip.api import PipelineKickoff

    _stub_ffmpeg(monkeypatch)
    monkeypatch.setattr(
        "chalybclip.settings.get_settings",
        lambda: type("S", (), {"default_output_dir": str(tmp_path / "out")})(),
    )

    captured: list[PipelineKickoff] = []

    async def fake_runner(kickoff: PipelineKickoff) -> None:
        captured.append(kickoff)

    app = create_app(db=db, pipeline_runner=fake_runner)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as c:
        files = {"file": ("recording.mp4", b"\x00fakevideo", "video/mp4")}
        r = await c.post(
            "/streams/upload",
            files=files,
            data={"persona_id": "aldo"},
            headers=auth(tenants["alice"]["token"]),
        )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["platform"] == "upload"
    assert body["title"] == "recording.mp4"
    assert body["duration_s"] == 17.5

    # Stream row landed under Alice's tenant.
    with bound_tenant(tenants["alice"]["id"]):
        row = await StreamsRepo(db).get(body["id"])
    assert row is not None
    assert row.platform == "upload"

    # Background pipeline task fired with the right kickoff.
    assert len(captured) == 1
    assert captured[0].tenant_id == tenants["alice"]["id"]
    assert captured[0].persona_id == "aldo"
    assert captured[0].stream.platform == "upload"


async def test_upload_rejects_empty_file(
    db: Database,
    tenants: dict[str, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A zero-byte upload is a configuration mistake, not a stream."""
    _stub_ffmpeg(monkeypatch)
    monkeypatch.setattr(
        "chalybclip.settings.get_settings",
        lambda: type("S", (), {"default_output_dir": str(tmp_path / "out")})(),
    )

    app = create_app(db=db)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as c:
        files = {"file": ("empty.mp4", b"", "video/mp4")}
        r = await c.post(
            "/streams/upload",
            files=files,
            data={"persona_id": "aldo"},
            headers=auth(tenants["alice"]["token"]),
        )
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


async def test_dashboard_upload_redirects_to_stream_detail(
    db: Database,
    tenants: dict[str, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The /dashboard/streams/upload form redirects to /dashboard/streams/<id>.

    UX refactor — the redirect now happens IMMEDIATELY after bytes
    arrive (placeholder row inserted, background runner scheduled),
    not after the full ingest_uploaded call. Verify the placeholder
    has the expected shape: status='pending', duration_s=0, pseudo
    upload:// URL.

    Bearer auth instead of cookie login because the dashboard cookie
    flow is pre-broken in test fixtures (slice O.22 redirected login
    to chalyb)."""
    _stub_ffmpeg(monkeypatch)
    monkeypatch.setattr(
        "chalybclip.settings.get_settings",
        lambda: type("S", (), {
            "default_output_dir": str(tmp_path / "out"),
            "db_path": str(tmp_path / "test.db"),
        })(),
    )

    # Block the background runner so we can inspect ONLY the
    # endpoint's synchronous work (placeholder insert + redirect).
    # Without this stub the runner would try to call real
    # process_vod against a fake DB path.
    scheduled: list[dict] = []

    async def fake_runner(**kwargs: object) -> None:
        scheduled.append(kwargs)

    monkeypatch.setattr(
        "chalybclip.api._pipeline.upload_pipeline_runner", fake_runner,
    )

    app = create_app(db=db)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://api.test"
    ) as c:
        files = {"file": ("clip.mp4", b"\x00fakevideo", "video/mp4")}
        r = await c.post(
            "/dashboard/streams/upload",
            files=files,
            data={"persona_id": "aldo"},
            headers=auth(tenants["alice"]["token"]),
            follow_redirects=False,
        )
    assert r.status_code == 303, r.text
    location = r.headers["location"]
    assert location.startswith("/dashboard/streams/str_")
    stream_id = location.removeprefix("/dashboard/streams/")

    # Placeholder row landed with the right shape.
    with bound_tenant(tenants["alice"]["id"]):
        row = await StreamsRepo(db).get(stream_id)
    assert row is not None
    assert row.platform == "upload"
    assert row.vod_url == "upload://clip.mp4"
    assert row.status == "pending"
    # duration_s is 0 in the placeholder — the runner will UPSERT
    # the real value after ffprobe in the background.
    assert row.duration_s == 0.0
    assert row.title == "clip.mp4"


async def test_upload_503s_when_ffmpeg_missing(
    db: Database,
    tenants: dict[str, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When ffmpeg isn't on PATH, the upload endpoint refuses up front with a
    503 + install hint, so a multi-GB upload doesn't fail mid-pipeline."""
    monkeypatch.setattr(
        "chalybclip.settings.get_settings",
        lambda: type("S", (), {"default_output_dir": str(tmp_path / "out")})(),
    )
    monkeypatch.setattr("chalybclip.ingest.service.is_ffmpeg_available", lambda: False)
    monkeypatch.setattr("chalybclip.ingest.is_ffmpeg_available", lambda: False)

    app = create_app(db=db)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://api.test"
    ) as c:
        files = {"file": ("v.mp4", b"\x00fakevideo", "video/mp4")}
        r = await c.post(
            "/streams/upload",
            files=files,
            data={"persona_id": "aldo"},
            headers=auth(tenants["alice"]["token"]),
        )
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert "ffmpeg" in detail
    assert "winget" in detail or "brew" in detail


async def test_upload_requires_full_scope(
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """No bearer = 401 (auth middleware rejects before form parsing)."""
    app = create_app(db=db)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as c:
        files = {"file": ("x.mp4", b"\x00x", "video/mp4")}
        r = await c.post(
            "/streams/upload",
            files=files,
            data={"persona_id": "aldo"},
        )
    assert r.status_code == 401


# ---- Dashboard upload — background-runner contract ----


async def test_dashboard_upload_does_not_block_on_ingest(
    db: Database,
    tenants: dict[str, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Operator-visible win — the endpoint redirects BEFORE the audio
    extract runs. We assert no _extract_audio call happens during the
    synchronous request lifecycle (background tasks fire AFTER the
    response is dispatched)."""
    _stub_ffmpeg(monkeypatch)
    monkeypatch.setattr(
        "chalybclip.settings.get_settings",
        lambda: type("S", (), {
            "default_output_dir": str(tmp_path / "out"),
            "db_path": str(tmp_path / "test.db"),
        })(),
    )

    # Block the background runner so no real ingest fires after the
    # response (tests would otherwise hit a real ffmpeg path).
    async def fake_runner(**kwargs: object) -> None:
        return None
    monkeypatch.setattr(
        "chalybclip.api._pipeline.upload_pipeline_runner", fake_runner,
    )

    # Tripwire on _extract_audio — if the endpoint synchronously calls
    # it, this test fails immediately with a clear message.
    extract_calls: list[tuple] = []

    def trip(video: Path, audio: Path) -> None:
        extract_calls.append((video, audio))
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"\x00")
    monkeypatch.setattr("chalybclip.ingest.service._extract_audio", trip)

    app = create_app(db=db)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://api.test"
    ) as c:
        files = {"file": ("v.mp4", b"\x00fakevideo", "video/mp4")}
        r = await c.post(
            "/dashboard/streams/upload",
            files=files,
            data={"persona_id": "aldo"},
            headers=auth(tenants["alice"]["token"]),
            follow_redirects=False,
        )
    assert r.status_code == 303
    # The redirect went out before audio extract ran.
    assert extract_calls == [], (
        f"audio extract ran synchronously: {extract_calls}"
    )


async def test_dashboard_upload_schedules_background_runner(
    db: Database,
    tenants: dict[str, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Endpoint schedules `upload_pipeline_runner` with the placeholder
    stream_id + tenant + persona + tempfile path. The runner is what
    actually moves the file + extracts audio + runs the pipeline."""
    _stub_ffmpeg(monkeypatch)
    monkeypatch.setattr(
        "chalybclip.settings.get_settings",
        lambda: type("S", (), {
            "default_output_dir": str(tmp_path / "out"),
            "db_path": str(tmp_path / "test.db"),
        })(),
    )

    captured: list[dict] = []

    async def fake_runner(**kwargs: object) -> None:
        captured.append(kwargs)
    monkeypatch.setattr(
        "chalybclip.api._pipeline.upload_pipeline_runner", fake_runner,
    )

    app = create_app(db=db)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://api.test"
    ) as c:
        files = {"file": ("clip.mp4", b"\x00fakevideo", "video/mp4")}
        r = await c.post(
            "/dashboard/streams/upload",
            files=files,
            data={"persona_id": "aldo"},
            headers=auth(tenants["alice"]["token"]),
            follow_redirects=False,
        )
    assert r.status_code == 303

    # Wait briefly for FastAPI's BackgroundTasks to fire. They run
    # AFTER the response is sent but during the same task group.
    import asyncio
    for _ in range(50):
        if captured:
            break
        await asyncio.sleep(0.01)
    assert len(captured) == 1, "runner not scheduled"
    kw = captured[0]
    assert kw["tenant_id"] == tenants["alice"]["id"]
    assert kw["persona_id"] == "aldo"
    assert kw["title"] == "clip.mp4"
    assert kw["stream_id"].startswith("str_")
    assert kw["tmp_path"].exists()
