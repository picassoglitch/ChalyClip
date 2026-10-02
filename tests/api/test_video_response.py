"""video_file_response keeps every response under Cloud Run's 32 MiB cap."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from chalybclip.api import _video_response
from chalybclip.api._video_response import video_file_response


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, bytes]:
    monkeypatch.setattr(_video_response, "MAX_RANGE_BYTES", 1000)
    data = bytes(range(256)) * 20  # 5120 bytes
    path = tmp_path / "clip.mp4"
    path.write_bytes(data)

    app = FastAPI()

    @app.api_route("/v", methods=["GET", "HEAD"])
    async def v(request: Request):  # type: ignore[no-untyped-def]
        return video_file_response(request, path, filename="clip.mp4")

    return TestClient(app), data


def test_open_ended_range_is_capped(client: tuple[TestClient, bytes]) -> None:
    c, data = client
    r = c.get("/v", headers={"Range": "bytes=0-"})
    assert r.status_code == 206
    assert r.headers["content-range"] == f"bytes 0-999/{len(data)}"
    assert r.content == data[:1000]


def test_small_and_tail_ranges_untouched(client: tuple[TestClient, bytes]) -> None:
    c, data = client
    r = c.get("/v", headers={"Range": "bytes=10-19"})
    assert r.content == data[10:20]
    r = c.get("/v", headers={"Range": "bytes=4500-"})
    assert r.content == data[4500:]


def test_no_range_streams_without_content_length(
    client: tuple[TestClient, bytes],
) -> None:
    c, data = client
    r = c.get("/v")
    assert r.status_code == 200
    assert "content-length" not in r.headers
    assert r.headers["accept-ranges"] == "bytes"
    assert r.content == data


def test_head_reports_real_size(client: tuple[TestClient, bytes]) -> None:
    c, data = client
    r = c.head("/v")
    assert r.status_code == 200
    assert r.headers["content-length"] == str(len(data))
