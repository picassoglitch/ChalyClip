"""CLI smoke test for the `ingest` command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from chalybclip.cli import app
from chalybclip.ingest import service as ingest_service


def test_ingest_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "VOD URL" in result.stdout


def test_ingest_command_json_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = {"duration": 120.0, "title": "t", "uploader": "u"}

    def fake_download(
        *,
        vod_url: str,
        target_path: Path,
        cookies_from_browser: str | None = None,
        cookies_file: str | None = None,
        platform: str = "unknown",
    ) -> dict[str, Any]:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"v")
        return info

    def fake_extract(video_path: Path, audio_path: Path) -> None:
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"a")

    monkeypatch.setattr(ingest_service, "_download_vod", fake_download)
    monkeypatch.setattr(ingest_service, "_extract_audio", fake_extract)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "ingest",
            "https://kick.com/c/videos/1",
            "--output-dir",
            str(tmp_path),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["platform"] == "kick"
    assert payload["tenant_id"] == "default"
    assert payload["duration_s"] == 120.0
    assert payload["id"].startswith("str_")
