"""Upload admission: the hub is asked BEFORE the body is read, the transfer
stops once it passes the tier's max_upload_mb, and every refusal maps to
the right status + Spanish message with the reservation released."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from chalybclip.api import create_app
from chalybclip.api._admission import read_capped_form
from chalybclip.db import Database, StreamsRepo
from chalybclip.db.usage_repos import UsageOutboxRepo
from chalybclip.integrations.chalyb.admission import AdmissionRefused, AdmitResult
from chalybclip.tenancy import bound_tenant

from .conftest import auth

_BOUNDARY = "XyZb0undary"


def _multipart_chunks(file_bytes: int, chunk: int) -> list[bytes]:
    head = (
        f"--{_BOUNDARY}\r\nContent-Disposition: form-data; name=\"persona_id\"\r\n\r\n"
        f"aldo\r\n--{_BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"big.mp4\"\r\nContent-Type: video/mp4\r\n\r\n"
    ).encode()
    tail = f"\r\n--{_BOUNDARY}--\r\n".encode()
    body = head + b"\x00" * file_bytes + tail
    return [body[i : i + chunk] for i in range(0, len(body), chunk)]


async def test_capped_form_stops_reading_once_over_the_cap() -> None:
    """A 3 MiB body against a ~100 KiB cap: the parser must give up after a
    few chunks, not drain all 48 of them."""
    chunks = _multipart_chunks(3 * 1024 * 1024, 64 * 1024)
    pulled = 0

    async def receive() -> dict:
        nonlocal pulled
        if pulled < len(chunks):
            pulled += 1
            return {
                "type": "http.request", "body": chunks[pulled - 1],
                "more_body": pulled < len(chunks),
            }
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http", "method": "POST", "path": "/", "query_string": b"",
        "headers": [(b"content-type", f"multipart/form-data; boundary={_BOUNDARY}".encode())],
    }
    with pytest.raises(HTTPException) as ei:
        await read_capped_form(Request(scope, receive), max_bytes=100 * 1024)
    assert ei.value.status_code == 413
    assert "plan" in ei.value.detail  # Spanish, user-facing
    assert pulled <= 4, f"kept reading after the cap: {pulled} chunks"


async def test_capped_form_parses_a_small_upload() -> None:
    chunks = _multipart_chunks(1024, 512)
    it = iter(chunks)

    async def receive() -> dict:
        nxt = next(it, None)
        return {"type": "http.request", "body": nxt or b"", "more_body": nxt is not None}

    scope = {
        "type": "http", "method": "POST", "path": "/", "query_string": b"",
        "headers": [(b"content-type", f"multipart/form-data; boundary={_BOUNDARY}".encode())],
    }
    file, fields = await read_capped_form(Request(scope, receive), max_bytes=1024 * 1024)
    assert fields == {"persona_id": "aldo"}
    assert file.filename == "big.mp4"
    assert len(await file.read()) == 1024


# ---- routes ---------------------------------------------------------------


def _stub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict]:
    monkeypatch.setattr("chalybclip.ingest.service._ffprobe_duration", lambda _p: 600.0)
    monkeypatch.setattr("chalybclip.ingest.service.is_ffmpeg_available", lambda: True)
    monkeypatch.setattr("chalybclip.ingest.is_ffmpeg_available", lambda: True)
    monkeypatch.setattr(
        "chalybclip.settings.get_settings",
        lambda: type("S", (), {
            "default_output_dir": str(tmp_path / "out"),
            "db_path": str(tmp_path / "test.db"),
        })(),
    )
    scheduled: list[dict] = []

    async def fake_runner(**kwargs: object) -> None:
        scheduled.append(kwargs)

    monkeypatch.setattr("chalybclip.api._pipeline.upload_pipeline_runner", fake_runner)
    return scheduled


def _hub(monkeypatch: pytest.MonkeyPatch, answers: list[AdmitResult | Exception]) -> list[dict]:
    """Script the hub's admit answers; record what was asked."""
    asked: list[dict] = []

    async def fake_admit(db, *, tenant_id, request, client=None):
        asked.append({"upload_mb": request.upload_mb, "minutes": request.source_minutes,
                      "boost": request.boost, "job": request.external_job_id})
        ans = answers.pop(0)
        if isinstance(ans, Exception):
            raise ans
        return ans

    monkeypatch.setattr("chalybclip.jobs.usage.admit", fake_admit)
    return asked


def _admitted(**limits: object) -> AdmitResult:
    return AdmitResult(allowed=True, reservation_id="res-x", lane="standard", limits=dict(limits))


async def _post(app, tenants, *, size: int, url: str = "/dashboard/streams/upload",
                data: dict | None = None) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as c:
        return await c.post(
            url,
            files={"file": ("clip.mp4", b"\x00" * size, "video/mp4")},
            data=data or {"persona_id": "aldo"},
            headers=auth(tenants["alice"]["token"]),
            follow_redirects=False,
        )


async def _settle_outcome(db: Database, reservation: str) -> str | None:
    row = await UsageOutboxRepo(db).get(f"settle_{reservation}")
    return row.payload["outcome"] if row else None


async def test_upload_over_tier_cap_is_413_and_releases_reservation(
    db: Database, tenants, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scheduled = _stub(monkeypatch, tmp_path)
    _hub(monkeypatch, [_admitted(max_upload_mb=0.05)])  # ~52 KB
    r = await _post(create_app(db=db), tenants, size=300 * 1024)
    assert r.status_code == 413, r.text
    assert "plan" in r.json()["detail"]
    assert scheduled == []
    assert await _settle_outcome(db, "res-x") == "cancelled"
    # Nothing left on disk.
    uploads = tmp_path / "out" / ".uploads"
    assert not uploads.exists() or not any(uploads.iterdir())


async def test_refused_before_reading_returns_402_spanish(
    db: Database, tenants, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scheduled = _stub(monkeypatch, tmp_path)
    asked = _hub(monkeypatch, [AdmissionRefused("no_tokens")])
    r = await _post(create_app(db=db), tenants, size=1024)
    assert r.status_code == 402
    assert "tokens" in r.json()["detail"].lower()
    assert scheduled == []
    # Asked once, with the declared size, before any probe.
    assert len(asked) == 1 and asked[0]["minutes"] == 0


async def test_concurrency_refusal_is_429(
    db: Database, tenants, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub(monkeypatch, tmp_path)
    _hub(monkeypatch, [AdmissionRefused("concurrency")])
    r = await _post(create_app(db=db), tenants, size=1024, url="/streams/upload")
    assert r.status_code == 429


async def test_probe_readmit_refusal_cancels_and_returns_413(
    db: Database, tenants, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Check twice: the first admit passes on size, the post-probe one
    refuses on length → 413, reservation cancelled, nothing scheduled."""
    scheduled = _stub(monkeypatch, tmp_path)
    asked = _hub(monkeypatch, [_admitted(), AdmissionRefused("video_too_long")])
    r = await _post(create_app(db=db), tenants, size=2048)
    assert r.status_code == 413
    assert "largo" in r.json()["detail"]
    assert scheduled == []
    assert len(asked) == 2
    assert asked[0]["job"] == asked[1]["job"]  # same external_job_id
    assert asked[1]["minutes"] == pytest.approx(10.0)  # ffprobe 600 s
    assert await _settle_outcome(db, "res-x") == "cancelled"


async def test_admitted_upload_carries_reservation_and_boost_to_runner(
    db: Database, tenants, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scheduled = _stub(monkeypatch, tmp_path)
    asked = _hub(monkeypatch, [
        _admitted(max_upload_mb=100),
        AdmitResult(allowed=True, reservation_id="res-x", lane="boost"),
    ])
    r = await _post(
        create_app(db=db), tenants, size=4096, data={"persona_id": "aldo", "boost": "1"},
    )
    assert r.status_code == 303, r.text
    assert asked[0]["boost"] is None and asked[1]["boost"] is True
    assert asked[1]["upload_mb"] == pytest.approx(4096 / 1024 / 1024)
    for _ in range(50):
        if scheduled:
            break
        await asyncio.sleep(0.01)
    kw = scheduled[0]
    assert kw["reservation_id"] == "res-x" and kw["lane"] == "boost"
    assert kw["usage_job_id"] == asked[0]["job"]
    stream_id = r.headers["location"].removeprefix("/dashboard/streams/")
    with bound_tenant(tenants["alice"]["id"]):
        assert await StreamsRepo(db).get(stream_id) is not None
