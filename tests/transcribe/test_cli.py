"""CLI smoke tests for `chalybclip transcribe`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from chalybclip.cli import app
from chalybclip.ingest import Stream

from ._fakes import FakeInfo, FakeSegment, FakeWhisperModel, FakeWord


@pytest.fixture(autouse=True)
def _patch_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch `faster_whisper.WhisperModel` directly — the local provider
    # lazy-imports it inside `_run_inprocess` — and flip the provider
    # into in-process mode via the factory's env override.
    import sys
    import types

    fw = sys.modules.get("faster_whisper")
    if fw is None:
        fw = types.ModuleType("faster_whisper")
        sys.modules["faster_whisper"] = fw
    FakeWhisperModel.reset()
    monkeypatch.setattr(fw, "WhisperModel", FakeWhisperModel, raising=False)
    monkeypatch.setenv("CHALYBCLIP_TRANSCRIBE_INPROCESS", "1")
    from chalybclip.settings import get_settings

    get_settings.cache_clear()
    FakeWhisperModel.canned_info = FakeInfo(language="es", duration=5.0)
    FakeWhisperModel.canned_segments = [
        FakeSegment(
            start=0.0,
            end=1.0,
            text="hola",
            words=[FakeWord(start=0.0, end=1.0, word="hola", probability=0.9)],
        )
    ]


def _seed_stream(tmp_path: Path, stream_id: str = "str_01CLI") -> Stream:
    stream_dir = tmp_path / stream_id
    source_dir = stream_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    video = source_dir / "video.mp4"
    audio = source_dir / "audio.wav"
    video.write_bytes(b"v")
    audio.write_bytes(b"a")
    stream = Stream(
        id=stream_id,
        tenant_id="default",
        vod_url="https://kick.com/c/videos/1",
        platform="kick",
        title="t",
        channel="c",
        duration_s=5.0,
        source_video_path=video,
        source_audio_path=audio,
    )
    (stream_dir / "stream.json").write_text(
        stream.model_dump_json(indent=2), encoding="utf-8"
    )
    return stream


def test_transcribe_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "Stream ID" in result.stdout


def test_transcribe_command_json_output(tmp_path: Path) -> None:
    stream = _seed_stream(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "transcribe",
            stream.id,
            "--output-dir",
            str(tmp_path),
            "--device",
            "cpu",
            "--compute-type",
            "int8",
            "--model",
            "tiny",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["stream_id"] == stream.id
    assert payload["model"] == "tiny"
    assert payload["language"] == "es"
    assert len(payload["segments"]) == 1

    ctor_args, ctor_kwargs = FakeWhisperModel.constructor_calls[0]
    assert ctor_args == ("tiny",)
    assert ctor_kwargs == {"device": "cpu", "compute_type": "int8"}


def test_transcribe_command_unknown_stream_exits_1(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["transcribe", "str_NOTHING", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "transcribe failed" in result.stderr or "transcribe failed" in result.output
