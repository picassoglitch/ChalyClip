"""CloudRunIdTokenAuth — metadata-server ID tokens for IAM-locked Cloud Run."""

from __future__ import annotations

import httpx
import pytest

from chalybclip.integrations import gcp_identity
from chalybclip.integrations.gcp_identity import (
    CloudRunIdTokenAuth,
    is_cloud_run_url,
)


def _echo_auth(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text=request.headers.get("authorization", ""))


async def _call(auth: CloudRunIdTokenAuth, path: str = "/") -> str:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(_echo_auth), auth=auth
    ) as client:
        resp = await client.get(f"https://w-abc-uc.a.run.app{path}")
        return resp.text


async def test_audience_is_service_origin_and_token_is_cached() -> None:
    calls: list[str] = []

    def metadata(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params["audience"])
        return httpx.Response(200, text="tok-1")

    auth = CloudRunIdTokenAuth(
        "https://w-abc-uc.a.run.app/some/path",
        metadata_transport=httpx.MockTransport(metadata),
    )
    assert await _call(auth) == "Bearer tok-1"
    assert await _call(auth, "/jobs/j1") == "Bearer tok-1"
    assert calls == ["https://w-abc-uc.a.run.app"]


async def test_token_is_reminted_before_it_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens = iter(["tok-1", "tok-2"])

    def metadata(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=next(tokens))

    now = [1000.0]
    monkeypatch.setattr(gcp_identity.time, "monotonic", lambda: now[0])
    auth = CloudRunIdTokenAuth(
        "https://w-abc-uc.a.run.app",
        metadata_transport=httpx.MockTransport(metadata),
    )
    assert await _call(auth) == "Bearer tok-1"
    now[0] += 50 * 60  # past the re-mint window, inside the 1 h validity
    assert await _call(auth) == "Bearer tok-2"


async def test_no_metadata_server_sends_request_without_header() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no metadata server", request=request)

    auth = CloudRunIdTokenAuth(
        "https://w-abc-uc.a.run.app",
        metadata_transport=httpx.MockTransport(down),
    )
    assert await _call(auth) == ""


def test_is_cloud_run_url() -> None:
    assert is_cloud_run_url("https://chalybclip-worker-znilbomw3q-uc.a.run.app")
    assert not is_cloud_run_url("https://x--pipeline.modal.run")
    assert not is_cloud_run_url("https://evil.run.app.example.com")
