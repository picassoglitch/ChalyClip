"""API → worker export dispatch: when it routes to the worker, and how a
worker that can't take the job surfaces (raise, so the click fails with retry)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from chalybclip.jobs import export as export_mod
from chalybclip.jobs.export import ExportDispatchError, dispatch_export, export_worker_url


def _settings(**over: Any) -> Any:
    base = {
        "export_on_worker": True,
        "job_dispatcher": "modal",
        "modal_pipeline_endpoint_url": "https://chalybclip-worker-x-uc.a.run.app/",
        "modal_token": "tok",
        "object_storage_bucket": "media",
    }
    base.update(over)
    return type("S", (), base)()


@pytest.mark.parametrize(
    ("over", "expected"),
    [
        ({}, "https://chalybclip-worker-x-uc.a.run.app/exports"),
        ({"export_on_worker": False}, None),
        ({"job_dispatcher": "in_process"}, None),
        ({"modal_pipeline_endpoint_url": ""}, None),
        ({"modal_token": None}, None),
        ({"object_storage_bucket": None}, None),
    ],
)
def test_export_worker_url(monkeypatch: pytest.MonkeyPatch, over: dict[str, Any], expected: str | None) -> None:
    s = _settings(**over)
    monkeypatch.setattr("chalybclip.settings.get_settings", lambda: s)
    monkeypatch.setattr(
        "chalybclip.integrations.storage.build_artifact_store",
        lambda st: object() if getattr(st, "object_storage_bucket", None) else None,
    )
    assert export_worker_url() == expected


def test_setting_defaults_on_and_env_turns_it_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from chalybclip.settings import Settings

    assert Settings().export_on_worker is True
    monkeypatch.setenv("CHALYBCLIP_EXPORT_ON_WORKER", "false")
    assert Settings().export_on_worker is False


async def _dispatch(handler: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("chalybclip.settings.get_settings", lambda: _settings())
    await dispatch_export(
        url="http://worker.test/exports", tenant_id="t", clip_id="c", quality="1080",
        width=1080, height=1920, base_url="https://api", auth_query="q",
        render_started_at="s1", deadline_ts=9e9,
        transport=httpx.MockTransport(handler),
    )


async def test_worker_error_status_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="worker misconfigured: DATABASE_URL is not set")

    with pytest.raises(ExportDispatchError, match=r"HTTP 500.*DATABASE_URL"):
        await _dispatch(handler, monkeypatch)


async def test_unreachable_worker_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=req)

    with pytest.raises(ExportDispatchError, match="unreachable"):
        await _dispatch(handler, monkeypatch)


async def test_not_accepted_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not the worker</html>\n")

    with pytest.raises(ExportDispatchError, match="did not accept"):
        await _dispatch(handler, monkeypatch)


async def test_accepted_returns_and_drains(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        import json

        sent.append(json.loads(req.content))
        return httpx.Response(
            200, text='{"event":"accepted"}\n{"event":"heartbeat"}\n{"event":"done","status":"ready"}\n'
        )

    await _dispatch(handler, monkeypatch)
    await export_mod.drain_exports()
    assert sent[0]["auth_token"] == "tok"
    assert sent[0]["clip_id"] == "c" and sent[0]["render_started_at"] == "s1"


def test_modal_app_endpoint_keeps_exports_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(modal_pipeline_endpoint_url="https://me--chalybclip-pipeline-run.modal.run")
    monkeypatch.setattr("chalybclip.settings.get_settings", lambda: s)
    monkeypatch.setattr("chalybclip.integrations.storage.build_artifact_store", lambda _s: object())
    assert export_worker_url() is None


async def test_endpoint_without_exports_route_is_reported_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chalybclip.jobs.export import ExportWorkerMissingError

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Not Found"})

    with pytest.raises(ExportWorkerMissingError):
        await _dispatch(handler, monkeypatch)


@pytest.mark.parametrize(
    ("url", "origin"),
    [
        ("https://chalybclip.chalyb.com", "https://chalybclip.chalyb.com"),
        ("https://chalybclip.chalyb.com/", "https://chalybclip.chalyb.com"),
        ("http://127.0.0.1:58100/x", "http://127.0.0.1:58100"),
        ("", None),
        (None, None),
        ("ftp://host", None),
        ("not a url", None),
    ],
)
def test_render_origin(url: str | None, origin: str | None) -> None:
    from chalybclip.jobs.export import render_origin

    assert render_origin(url) == origin


def test_render_origin_reads_public_url_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    from chalybclip.settings import Settings

    monkeypatch.delenv("CHALYBCLIP_EXPORT_RENDER_BASE_URL", raising=False)
    monkeypatch.setenv("PUBLIC_URL", "https://chalybclip.chalyb.com")
    assert Settings().export_render_base_url == "https://chalybclip.chalyb.com"
    monkeypatch.setenv("CHALYBCLIP_EXPORT_RENDER_BASE_URL", "https://render.internal")
    assert Settings().export_render_base_url == "https://render.internal"
