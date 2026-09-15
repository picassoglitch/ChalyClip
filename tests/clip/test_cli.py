"""CLI smoke tests for `chalybclip cut`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from chalybclip.cli import app
from chalybclip.clip import service as clip_service
from chalybclip.detect import CandidateBatch

from ._fixtures import make_candidate, seed_stream


def _stub_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], *, what: str) -> None:
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake")

    monkeypatch.setattr(clip_service, "_run_ffmpeg", fake_run)


def _seed_artifacts(tmp_path: Path) -> Path:
    stream = seed_stream(tmp_path)
    stream_dir = tmp_path / stream.id
    (stream_dir / "stream.json").write_text(
        stream.model_dump_json(indent=2), encoding="utf-8"
    )
    batch = CandidateBatch(
        stream_id=stream.id,
        tenant_id="default",
        candidates=[make_candidate(timestamp=120.0)],
    )
    (stream_dir / "candidates.json").write_text(
        batch.model_dump_json(indent=2), encoding="utf-8"
    )
    return stream_dir


def test_cut_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["cut", "--help"])
    assert result.exit_code == 0
    assert "Stream ID" in result.stdout


def test_cut_command_writes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_ffmpeg(monkeypatch)
    stream_dir = _seed_artifacts(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "cut",
            stream_dir.name,
            "--output-dir",
            str(tmp_path),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["stream_id"] == stream_dir.name
    assert len(payload["clips"]) == 1
    clip = payload["clips"][0]
    assert clip["id"].startswith("clp_")
    assert Path(clip["path"]).exists()


def test_cut_command_unknown_stream_exits_1(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["cut", "str_NOTHING", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "cut failed" in result.stderr or "cut failed" in result.output


def test_cut_command_no_candidates_exits_1(tmp_path: Path) -> None:
    """Missing candidates.json (no detect run) should fail cleanly."""
    stream = seed_stream(tmp_path)
    (tmp_path / stream.id / "stream.json").write_text(
        stream.model_dump_json(indent=2), encoding="utf-8"
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["cut", stream.id, "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "candidates not found" in (result.stderr or result.output)
