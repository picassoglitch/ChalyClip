"""ModalJobDispatcher behavior — Phase 2b.

Pins the dispatch contract against a mocked Modal endpoint:
  * kickoff serialization + bearer token in the POST body,
  * `jobs.active` registration spanning dispatch → remote-terminal,
  * same-stream dedupe,
  * in-process fallback for non-http sources (`upload://`, `live://`),
  * Modal's 303-poll protocol,
  * failure surfacing split: HTTP errors emit `pipeline.failed`; a poll
    deadline does NOT (the remote run may still be alive).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from chalybclip.jobs import ModalJobDispatcher, PipelineKickoff
from chalybclip.jobs.active import active_stream_ids

_REAL_ASYNC_CLIENT = httpx.AsyncClient  # capture BEFORE any monkeypatch


@pytest.fixture(autouse=True)
def _clean_active_registry() -> Iterator[None]:
    """Clear the process-global registry before AND after each test —
    the dedup test registers a stream it never releases, which would
    otherwise leak into later-collected test files."""
    from chalybclip.jobs import active

    active._active.clear()
    yield
    active._active.clear()


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Poll loop ticks instantly (patch is on the global asyncio module,
    which covers the shared loop in chalybclip.integrations.modal_http)."""
    async def _noop(_s: float) -> None:
        return None
    monkeypatch.setattr(asyncio, "sleep", _noop)


@dataclass
class _FakeStream:
    id: str = "str_TEST"
    vod_url: str = "https://kick.com/x/videos/1"

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return {"id": self.id, "vod_url": self.vod_url}


@dataclass
class _FallbackProbe:
    calls: list[PipelineKickoff] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "probe"

    async def dispatch_pipeline(
        self, kickoff: PipelineKickoff, *, background_tasks: object = None
    ) -> None:
        self.calls.append(kickoff)


def _make_kickoff(
    stream_id: str = "str_TEST",
    vod_url: str = "https://kick.com/x/videos/1",
) -> PipelineKickoff:
    return PipelineKickoff(
        tenant_id="ten_TEST",
        stream=cast("Any", _FakeStream(id=stream_id, vod_url=vod_url)),
        persona_id="per_TEST",
        output_dir=Path("./out"),
        language="es",
    )


def _dispatcher(**kw: Any) -> ModalJobDispatcher:
    kw.setdefault("endpoint_url", "https://modal.test/pipeline")
    kw.setdefault("bearer_token", "bear")
    kw.setdefault("timeout_s", 5.0)
    kw.setdefault("poll_interval_s", 0.0)
    return ModalJobDispatcher(**kw)


def _patch_client(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    import chalybclip.jobs.modal as jobs_modal

    monkeypatch.setattr(
        jobs_modal.httpx, "AsyncClient",
        lambda *a, **kw: _REAL_ASYNC_CLIENT(
            transport=httpx.MockTransport(handler)
        ),
    )


def _no_failure_emit(dispatcher: ModalJobDispatcher) -> list[str]:
    """Replace the DB-writing failure emitter with a recorder."""
    recorded: list[str] = []

    async def _record(kickoff: PipelineKickoff, *, error: str) -> None:
        recorded.append(error)

    dispatcher._emit_failure = _record  # type: ignore[method-assign]
    return recorded


async def test_dispatch_posts_kickoff_and_releases_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        seen.append(json.loads(request.content))
        return httpx.Response(
            200, json={"stream_id": "str_TEST", "status": "done", "n_clips": 3}
        )

    d = _dispatcher()
    _patch_client(monkeypatch, handler)
    errors = _no_failure_emit(d)

    await d.dispatch_pipeline(_make_kickoff())
    # Registered at dispatch time, before the remote run completes.
    assert "str_TEST" in active_stream_ids()
    await d.drain()

    assert len(seen) == 1
    body = seen[0]
    assert body["auth_token"] == "bear"
    assert body["tenant_id"] == "ten_TEST"
    assert body["persona_id"] == "per_TEST"
    assert body["language"] == "es"
    assert body["stream"]["id"] == "str_TEST"
    assert body["stream"]["vod_url"].startswith("https://")
    # Terminal 'done' → registration released, nothing marked failed.
    assert "str_TEST" not in active_stream_ids()
    assert errors == []


async def test_dispatch_dedupes_stream_already_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        posts["n"] += 1
        return httpx.Response(200, json={"status": "done"})

    d = _dispatcher()
    _patch_client(monkeypatch, handler)

    from chalybclip.jobs.active import register

    register("str_TEST")  # e.g. a run already executing remotely
    await d.dispatch_pipeline(_make_kickoff())
    await d.drain()
    assert posts["n"] == 0
    # The pre-existing registration is untouched (not our run to release).
    assert "str_TEST" in active_stream_ids()


async def test_non_http_source_routes_to_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("must not reach Modal for upload:// sources")

    fallback = _FallbackProbe()
    d = _dispatcher(fallback=fallback)
    _patch_client(monkeypatch, handler)

    kickoff = _make_kickoff(vod_url="upload://local.mp4")
    await d.dispatch_pipeline(kickoff)
    await d.drain()

    assert fallback.calls == [kickoff]
    # The fallback dispatcher owns registration for its own runs.
    assert "str_TEST" not in active_stream_ids()


async def test_303_polling_flow_reaches_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    poll_url = "https://modal.test/?__modal_function_call_id=fc-1"
    gets = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(303, headers={"location": poll_url})
        gets["n"] += 1
        if gets["n"] < 3:
            return httpx.Response(303, headers={"location": poll_url})
        return httpx.Response(200, json={"status": "done", "n_clips": 1})

    d = _dispatcher()
    _patch_client(monkeypatch, handler)
    errors = _no_failure_emit(d)

    await d.dispatch_pipeline(_make_kickoff())
    await d.drain()

    assert gets["n"] == 3
    assert errors == []
    assert "str_TEST" not in active_stream_ids()


async def test_polling_follows_relative_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Modal's Location header is documented as relative
    (`?__modal_function_call_id=fc-...`) — the loop must resolve it
    against the request URL, not choke on it."""
    gets = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                303, headers={"location": "?__modal_function_call_id=fc-1"}
            )
        gets["n"] += 1
        assert request.url.params.get("__modal_function_call_id") == "fc-1"
        return httpx.Response(200, json={"status": "done"})

    d = _dispatcher()
    _patch_client(monkeypatch, handler)
    errors = _no_failure_emit(d)

    await d.dispatch_pipeline(_make_kickoff())
    await d.drain()

    assert gets["n"] == 1
    assert errors == []


async def test_http_error_emits_pipeline_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="function crashed")

    d = _dispatcher()
    _patch_client(monkeypatch, handler)
    errors = _no_failure_emit(d)

    await d.dispatch_pipeline(_make_kickoff())
    await d.drain()

    assert len(errors) == 1
    assert "500" in errors[0]
    assert "str_TEST" not in active_stream_ids()


async def test_transport_error_emits_pipeline_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    d = _dispatcher()
    _patch_client(monkeypatch, handler)
    errors = _no_failure_emit(d)

    await d.dispatch_pipeline(_make_kickoff())
    await d.drain()

    assert len(errors) == 1
    assert "dispatch error" in errors[0]
    assert "str_TEST" not in active_stream_ids()


async def test_poll_deadline_does_not_mark_run_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deadline = lost TRACKING, not a lost run — the worker keeps writing
    step events, so emitting pipeline.failed would lie to the operator."""
    poll_url = "https://modal.test/?__modal_function_call_id=fc-1"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(303, headers={"location": poll_url})

    d = _dispatcher(timeout_s=0.0)  # deadline expires on the first redirect
    _patch_client(monkeypatch, handler)
    errors = _no_failure_emit(d)

    await d.dispatch_pipeline(_make_kickoff())
    await d.drain()

    assert errors == []
    assert "str_TEST" not in active_stream_ids()
