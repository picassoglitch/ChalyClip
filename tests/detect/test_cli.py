"""CLI smoke tests for `chalybclip detect`."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from chalybclip.cli import app
from chalybclip.transcribe import Segment, Transcript, Word

from ._fixtures import make_stream


def _seed_artifacts(tmp_path: Path, *, stream_id: str = "str_01CLI") -> Path:
    """Write stream.json + transcript.json so the detect CLI has something to load."""
    stream_dir = tmp_path / stream_id
    source_dir = stream_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "video.mp4").write_bytes(b"v")
    (source_dir / "audio.wav").write_bytes(b"a")

    stream = make_stream(stream_id=stream_id)
    stream = stream.model_copy(
        update={
            "source_video_path": source_dir / "video.mp4",
            "source_audio_path": source_dir / "audio.wav",
        }
    )
    (stream_dir / "stream.json").write_text(
        stream.model_dump_json(indent=2), encoding="utf-8"
    )

    word = Word(ts=10.0, end_ts=11.0, text=" clipéalo", prob=0.9)
    transcript = Transcript(
        stream_id=stream_id,
        tenant_id="default",
        language="es",
        duration_s=12.0,
        model="tiny",
        segments=[Segment(ts=10.0, end_ts=11.0, text=" clipéalo", words=[word])],
    )
    (source_dir / "transcript.json").write_text(
        transcript.model_dump_json(indent=2), encoding="utf-8"
    )
    return stream_dir


def _seed_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "chalybclip.yaml"
    config_path.write_text(
        """
detection:
  voice:
    enabled: true
    weight: 1.0
    fuzzy_distance: 1
    phrases:
      es: ["clipéalo"]
  merge_window_s: 30
""",
        encoding="utf-8",
    )
    return config_path


def test_detect_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["detect", "--help"])
    assert result.exit_code == 0
    assert "Stream ID" in result.stdout


def test_detect_writes_candidates_json(tmp_path: Path) -> None:
    stream_dir = _seed_artifacts(tmp_path)
    config_path = _seed_config(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "detect",
            stream_dir.name,
            "--output-dir",
            str(tmp_path),
            "--config",
            str(config_path),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["stream_id"] == stream_dir.name
    assert payload["tenant_id"] == "default"
    assert len(payload["candidates"]) == 1
    assert payload["candidates"][0]["evidence"]["phrase"] == "clipéalo"

    candidates_path = stream_dir / "candidates.json"
    assert candidates_path.exists()


def test_detect_unknown_stream_exits_1(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["detect", "str_NOTHING", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "detect failed" in result.stderr or "detect failed" in result.output
