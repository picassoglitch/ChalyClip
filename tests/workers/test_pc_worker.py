"""PC worker HTTP contract — must match what ModalJobDispatcher +
poll_until_terminal already speak: POST → 303 Location, GET → 303 while
running, 200 terminal JSON when done. Runner is injected; no real
pipeline work happens here."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from chalybclip.integrations.modal_http import poll_until_terminal
from chalybclip.jobs.base import PipelineKickoff
from chalybclip.workers import create_worker_app

_TOKEN = "wtok_test"


def _stream_payload(stream_id: str = "str_w1") -> dict[str, Any]:
    return {
        "id": stream_id,
        "tenant_id": "ten_w",
        "vod_url": "https://kick.com/x/videos/1",
        "platform": "kick",
        "title": "t",
        "channel": "c",
        "duration_s": 60.0,
        "source_video_path": "/tmp/v.mp4",
        "source_audio_path": "/tmp/a.wav",
    }


def _kickoff_body(stream_id: str = "str_w1") -> dict[str, Any]:
    return {
        "auth_token": _TOKEN,
        "tenant_id": "ten_w",
        "persona_id": "psn_w",
        "language": "es",
        "stream": _stream_payload(stream_id),
    }


@pytest.fixture
def worker_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("CHALYBCLIP_WORKER_TOKEN", _TOKEN)
    monkeypatch.setenv("DATABASE_URL", "postgresql://x/test")
    monkeypatch.setenv("CHALYBCLIP_OBJECT_STORAGE_BUCKET", "bucket")
    monkeypatch.setenv("CHALYBCLIP_DEFAULT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("CHALYBCLIP_EXPORT_RENDER_BASE_URL", "https://api.example")
    from chalybclip.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://worker.test"
    )


@pytest.mark.asyncio
async def test_full_kickoff_poll_terminal_flow(worker_env: None) -> None:
    release = asyncio.Event()
    ran: list[PipelineKickoff] = []

    async def runner(kickoff: PipelineKickoff) -> None:
        ran.append(kickoff)
        await release.wait()

    app = create_worker_app(runner=runner)
    async with _client(app) as client:
        resp = await client.post("/", json=_kickoff_body())
        assert resp.status_code == 303
        poll_url = resp.headers["location"]
        assert poll_url.startswith("/jobs/")
        assert f"t={_TOKEN}" in poll_url

        # Still running → 303 back to itself.
        still = await client.get(poll_url)
        assert still.status_code == 303

        release.set()
        await asyncio.sleep(0)  # let the task finish
        done = await client.get(poll_url)
        assert done.status_code == 200
        body = done.json()
        assert body["status"] == "done"
        assert body["stream_id"] == "str_w1"

    assert ran[0].tenant_id == "ten_w"
    assert ran[0].persona_id == "psn_w"
    assert ran[0].stream.id == "str_w1"


@pytest.mark.asyncio
async def test_dispatcher_poll_helper_speaks_the_worker_protocol(
    worker_env: None,
) -> None:
    """End-to-end with the REAL client-side poll loop the dispatcher uses."""

    async def runner(kickoff: PipelineKickoff) -> None:
        await asyncio.sleep(0.05)

    app = create_worker_app(runner=runner)
    async with _client(app) as client:
        initial = await client.post("/", json=_kickoff_body("str_poll"))
        final = await poll_until_terminal(
            client=client, initial=initial,
            deadline_s=5.0, poll_interval_s=0.02, label="pc-worker",
        )
        assert final.status_code == 200
        assert final.json()["status"] == "done"


@pytest.mark.asyncio
async def test_failed_run_returns_terminal_failed_body(worker_env: None) -> None:
    async def runner(kickoff: PipelineKickoff) -> None:
        raise RuntimeError("ffmpeg exploded")

    app = create_worker_app(runner=runner)
    async with _client(app) as client:
        resp = await client.post("/", json=_kickoff_body("str_boom"))
        final = await poll_until_terminal(
            client=client, initial=resp,
            deadline_s=5.0, poll_interval_s=0.02,
        )
        assert final.status_code == 200  # failed body, not HTTP error
        body = final.json()
        assert body["status"] == "failed"
        assert "ffmpeg exploded" in body["error"]


@pytest.mark.asyncio
async def test_rejects_bad_token_and_missing_fields(worker_env: None) -> None:
    async def runner(kickoff: PipelineKickoff) -> None:  # pragma: no cover
        raise AssertionError("must not run")

    app = create_worker_app(runner=runner)
    async with _client(app) as client:
        bad = dict(_kickoff_body(), auth_token="wrong")
        assert (await client.post("/", json=bad)).status_code == 401

        incomplete = dict(_kickoff_body())
        incomplete.pop("stream")
        assert (await client.post("/", json=incomplete)).status_code == 400

        # Poll without the token query param is refused too.
        assert (await client.get("/jobs/whatever")).status_code == 401


@pytest.mark.asyncio
async def test_duplicate_kickoff_reuses_the_running_job(worker_env: None) -> None:
    release = asyncio.Event()
    runs = 0

    async def runner(kickoff: PipelineKickoff) -> None:
        nonlocal runs
        runs += 1
        await release.wait()

    app = create_worker_app(runner=runner)
    async with _client(app) as client:
        first = await client.post("/", json=_kickoff_body("str_dup"))
        second = await client.post("/", json=_kickoff_body("str_dup"))
        assert first.headers["location"] == second.headers["location"]
        release.set()
        await asyncio.sleep(0)
    assert runs == 1


@pytest.mark.asyncio
async def test_concurrency_cap_queues_second_job(
    worker_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default cap is 1: a second stream's job waits for the slot instead of
    running concurrently (one GPU — parallel pipelines starve each other)."""
    monkeypatch.setenv("CHALYBCLIP_WORKER_MAX_JOBS", "1")
    release = asyncio.Event()
    running: list[str] = []

    async def runner(kickoff: PipelineKickoff) -> None:
        running.append(kickoff.stream.id)
        await release.wait()

    app = create_worker_app(runner=runner)
    async with _client(app) as client:
        r1 = await client.post("/", json=_kickoff_body("str_cap_a"))
        r2 = await client.post("/", json=_kickoff_body("str_cap_b"))
        assert r1.status_code == 303 and r2.status_code == 303
        await asyncio.sleep(0.05)
        assert running == ["str_cap_a"]  # b queued, not started
        # Both jobs still poll as in-progress.
        assert (await client.get(r2.headers["location"])).status_code == 303
        release.set()
        await asyncio.sleep(0.05)
        assert running == ["str_cap_a", "str_cap_b"]  # b ran after a freed the slot


@pytest.mark.asyncio
async def test_llm_proxy_requires_bearer_and_forwards(worker_env: None) -> None:
    import respx

    async def runner(kickoff: PipelineKickoff) -> None:  # pragma: no cover
        raise AssertionError("must not run")

    app = create_worker_app(runner=runner)
    async with _client(app) as client:
        # No/wrong bearer → refused; the GPU is not public.
        r = await client.post("/v1/chat/completions", json={"model": "m"})
        assert r.status_code == 401

        with respx.mock() as mock:
            upstream = mock.post(
                "http://127.0.0.1:11434/v1/chat/completions"
            ).mock(
                return_value=httpx.Response(
                    200, json={"choices": [{"message": {"content": "{}"}}]}
                )
            )
            ok = await client.post(
                "/v1/chat/completions",
                json={"model": "qwen2.5:7b-instruct-q4_K_M"},
                headers={"Authorization": f"Bearer {_TOKEN}"},
            )
        assert ok.status_code == 200
        assert upstream.called
        sent = upstream.calls.last.request.content.decode()
        assert "qwen2.5:7b-instruct-q4_K_M" in sent


@pytest.mark.asyncio
async def test_preflight_refuses_without_shared_db(
    worker_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("CHALYBCLIP_DATABASE_URL", raising=False)

    async def runner(kickoff: PipelineKickoff) -> None:  # pragma: no cover
        raise AssertionError("must not run")

    app = create_worker_app(runner=runner)
    async with _client(app) as client:
        resp = await client.post("/", json=_kickoff_body())
        assert resp.status_code == 500
        assert "DATABASE_URL" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_prefixed_database_url_reaches_settings(
    worker_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker configured only with CHALYBCLIP_DATABASE_URL must end up with
    Settings.database_url set — Settings binds the un-prefixed DATABASE_URL
    alias, and without normalization the run passes preflight yet writes to
    a worker-local SQLite the dashboard never sees."""
    import os

    from chalybclip.settings import get_settings

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("CHALYBCLIP_DATABASE_URL", "postgresql://u:p@shared/db")

    async def runner(kickoff: PipelineKickoff) -> None:  # pragma: no cover
        raise AssertionError("must not run")

    try:
        create_worker_app(runner=runner)
        assert os.environ["DATABASE_URL"] == "postgresql://u:p@shared/db"
        assert get_settings().database_url == "postgresql://u:p@shared/db"
    finally:
        # DATABASE_URL was set via os.environ inside create_worker_app —
        # scrub it and the settings cache so later tests see a clean env.
        os.environ.pop("DATABASE_URL", None)
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_parked_upload_fields_reach_the_kickoff(worker_env: None) -> None:
    ran: list[PipelineKickoff] = []

    async def runner(kickoff: PipelineKickoff) -> None:
        ran.append(kickoff)

    app = create_worker_app(runner=runner)
    body = _kickoff_body()
    body["stream"]["vod_url"] = "upload://clip.mp4"
    body["stream"]["platform"] = "upload"
    body["source_object_key"] = "uploads/ten_w/str_w1/source.mp4"
    body["title"] = "clip.mp4"
    async with _client(app) as client:
        resp = await client.post("/", json=body)
        assert resp.status_code == 303
        for _ in range(50):
            if ran:
                break
            await asyncio.sleep(0.01)
    assert ran[0].source_object_key == "uploads/ten_w/str_w1/source.mp4"
    assert ran[0].title == "clip.mp4"


# ---- editor exports (POST /exports) ---------------------------------------


def _export_body(**over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "auth_token": _TOKEN, "tenant_id": "ten_w", "clip_id": "clp_w", "quality": "4k",
        "width": 2160, "height": 3840, "base_url": "https://api.example",
        "auth_query": "tenant=ten_w&exp=1&sig=s", "render_started_at": "s1", "deadline_ts": 9e9,
    }
    body.update(over)
    return body


def _events(text: str) -> list[dict[str, Any]]:
    import json

    return [json.loads(line) for line in text.splitlines() if line.strip()]


@pytest.mark.asyncio
async def test_export_streams_accepted_heartbeat_done(
    worker_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chalybclip.workers import pc_worker

    seen: list[Any] = []

    async def _run(req: Any) -> dict[str, Any]:
        seen.append(req)
        await asyncio.sleep(0.25)
        return {"status": "ready", "error": None}

    monkeypatch.setattr("chalybclip.workers.export_job.run_export", _run)
    monkeypatch.setattr(pc_worker, "_EXPORT_HEARTBEAT_S", 0.05)
    async with _client(create_worker_app()) as client:
        resp = await client.post("/exports", json=_export_body())
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    events = _events(resp.text)
    assert events[0] == {"event": "accepted"}
    assert {"event": "heartbeat"} in events[1:-1]
    assert events[-1] == {"event": "done", "status": "ready", "error": None}
    (req,) = seen
    assert (req.clip_id, req.quality, req.width, req.height) == ("clp_w", "4k", 2160, 3840)


@pytest.mark.asyncio
async def test_export_rejects_bad_token_payload_and_missing_bucket(
    worker_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _client(create_worker_app()) as client:
        assert (await client.post("/exports", json=_export_body(auth_token="nope"))).status_code == 401
        assert (await client.post("/exports", json=_export_body(quality="8k"))).status_code == 400
        monkeypatch.delenv("CHALYBCLIP_OBJECT_STORAGE_BUCKET")
        assert (await client.post("/exports", json=_export_body())).status_code == 500


@pytest.mark.asyncio
async def test_export_resend_joins_the_running_render(
    worker_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    runs = 0

    async def _run(_req: Any) -> dict[str, Any]:
        nonlocal runs
        runs += 1
        await release.wait()
        return {"status": "ready", "error": None}

    monkeypatch.setattr("chalybclip.workers.export_job.run_export", _run)
    async with _client(create_worker_app()) as client:
        first = asyncio.create_task(client.post("/exports", json=_export_body()))
        second = asyncio.create_task(client.post("/exports", json=_export_body()))
        await asyncio.sleep(0.05)
        release.set()
        r1, r2 = await first, await second
    assert runs == 1
    assert _events(r1.text)[-1]["status"] == _events(r2.text)[-1]["status"] == "ready"


@pytest.mark.asyncio
async def test_api_dispatcher_speaks_the_export_protocol(
    worker_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The API's dispatch_export against the real worker app (in-memory)."""
    from chalybclip.jobs.export import dispatch_export, drain_exports

    monkeypatch.setenv("CHALYBCLIP_MODAL_TOKEN", _TOKEN)
    from chalybclip.settings import get_settings

    get_settings.cache_clear()
    seen: list[Any] = []

    async def _run(req: Any) -> dict[str, Any]:
        seen.append(req)
        return {"status": "failed", "error": "boom"}

    monkeypatch.setattr("chalybclip.workers.export_job.run_export", _run)
    transport = httpx.ASGITransport(app=create_worker_app())
    await dispatch_export(
        url="http://worker.test/exports", tenant_id="ten_w", clip_id="clp_w", quality="2k",
        width=1440, height=2560, base_url="https://api.example", auth_query="q=1",
        render_started_at="2026-11-07T00:00:00+00:00", deadline_ts=9e9, transport=transport,
    )
    await drain_exports()
    (req,) = seen
    assert (req.quality, req.width, req.height, req.deadline_ts) == ("2k", 1440, 2560, 9e9)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url",
    ["https://evil.example", "http://169.254.169.254", "https://api.example.evil.example", "file:///etc"],
)
async def test_export_refuses_a_base_url_that_is_not_our_origin(
    worker_env: None, monkeypatch: pytest.MonkeyPatch, base_url: str
) -> None:
    """The recorder carries a signed render query to base_url: never to another host."""
    ran: list[Any] = []

    async def _run(req: Any) -> dict[str, Any]:
        ran.append(req)
        return {"status": "ready", "error": None}

    monkeypatch.setattr("chalybclip.workers.export_job.run_export", _run)
    async with _client(create_worker_app()) as client:
        r = await client.post("/exports", json=_export_body(base_url=base_url))
    assert r.status_code == 400
    assert ran == []


@pytest.mark.asyncio
async def test_export_fails_closed_without_a_configured_origin(
    worker_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chalybclip.settings import get_settings

    monkeypatch.delenv("CHALYBCLIP_EXPORT_RENDER_BASE_URL")
    monkeypatch.delenv("PUBLIC_URL", raising=False)
    get_settings.cache_clear()
    async with _client(create_worker_app()) as client:
        r = await client.post("/exports", json=_export_body())
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_export_renders_against_the_configured_origin(
    worker_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []

    async def _run(req: Any) -> dict[str, Any]:
        seen.append(req)
        return {"status": "ready", "error": None}

    monkeypatch.setattr("chalybclip.workers.export_job.run_export", _run)
    async with _client(create_worker_app()) as client:
        r = await client.post("/exports", json=_export_body(base_url="https://api.example/x/y"))
    assert r.status_code == 200
    assert seen[0].base_url == "https://api.example"
