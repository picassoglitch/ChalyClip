"""CLI smoke tests for `chalybclip process`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from chalybclip.cli import app
from chalybclip.clip import service as clip_service
from chalybclip.config import ChalybClipConfig, VoiceDetectorConfig
from chalybclip.ingest import service as ingest_service
from chalybclip.llm import config as llm_config_module
from chalybclip.llm import router as router_module
from chalybclip.variants import personas as personas_module
from tests.llm._fakes import FakeProvider  # type: ignore[import]
from tests.llm._fixtures import make_llm_config  # type: ignore[import]
from tests.pipeline.test_process_vod import (  # type: ignore[import]
    _force_inprocess_whisper,
)
from tests.transcribe._fakes import (  # type: ignore[import]
    FakeInfo,
    FakeSegment,
    FakeWhisperModel,
    FakeWord,
)


def _stub_everything(
    monkeypatch: pytest.MonkeyPatch, *, fake_provider: FakeProvider
) -> None:
    # Ingest
    def fake_download(
        *,
        vod_url,
        target_path,
        cookies_from_browser=None,
        cookies_file=None,
        platform="unknown",
    ):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"v")
        return {"duration": 600.0, "title": "t", "uploader": "u"}

    def fake_extract(video_path, audio_path):
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"a")

    monkeypatch.setattr(ingest_service, "_download_vod", fake_download)
    monkeypatch.setattr(ingest_service, "_extract_audio", fake_extract)

    # Whisper
    FakeWhisperModel.reset()
    FakeWhisperModel.canned_info = FakeInfo(language="es", duration=600.0)
    FakeWhisperModel.canned_segments = [
        FakeSegment(
            start=119.5,
            end=121.0,
            text="clipéalo",
            words=[FakeWord(start=120.0, end=121.0, word="clipéalo", probability=0.93)],
        )
    ]
    _force_inprocess_whisper(monkeypatch)

    # ffmpeg
    def fake_ffmpeg(cmd, *, what):
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake")

    monkeypatch.setattr(clip_service, "_run_ffmpeg", fake_ffmpeg)

    # YAML configs
    config = ChalybClipConfig()
    config.detection.voice = VoiceDetectorConfig(
        enabled=True, weight=1.0, fuzzy_distance=2, phrases={"es": ["clipéalo"]}
    )
    monkeypatch.setattr("chalybclip.pipeline.load_config", lambda: config)
    monkeypatch.setattr(
        llm_config_module,
        "load_llm_config",
        lambda *_a, **_k: make_llm_config(retry_attempts=1, purpose="hook_generation"),
    )
    monkeypatch.setattr(
        "chalybclip.pipeline.load_llm_config",
        lambda: make_llm_config(retry_attempts=1, purpose="hook_generation"),
    )

    # Personas
    from chalybclip.variants import Persona

    personas = {
        "aldo_villanueva": Persona(
            id="aldo_villanueva",
            name="Aldo Villanueva",
            target_languages=["es"],
            primary_language="es",
            voice_prompt="Direct.",
        )
    }
    monkeypatch.setattr(personas_module, "load_personas", lambda *_a, **_k: personas)
    monkeypatch.setattr("chalybclip.pipeline.load_personas", lambda: personas)

    # LLM provider — wire the fake into the default factory so the router
    # the orchestrator builds picks it up.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def factory(name, _config, _api_key):
        return fake_provider if name == "anthropic" else None

    monkeypatch.setattr(router_module, "_default_provider_factory", factory)


def test_process_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["process", "--help"])
    assert result.exit_code == 0
    assert "VOD URL" in result.stdout


def test_process_command_writes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeProvider("anthropic")
    fake.queue_success({"hooks": [{"text": "HOOK"}]})  # the clip's auto-hook
    _stub_everything(monkeypatch, fake_provider=fake)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "process",
            "https://kick.com/aldovillanueva/videos/abc",
            "--persona",
            "aldo_villanueva",
            "--output-dir",
            str(tmp_path),
            "--n",
            "2",
            "--no-db",  # this CLI test predates Task 1 dual-write
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    manifest = json.loads(result.stdout)
    assert manifest["persona_id"] == "aldo_villanueva"
    assert manifest["language"] == "es"
    assert len(manifest["candidates"]) == 1
    assert len(manifest["clip_entries"]) == 1
    # One deterministic stub variant per clip (the LLM generator was removed).
    assert len(manifest["clip_entries"][0]["variants"]) == 1

    stream_id = manifest["stream"]["id"]
    assert (tmp_path / stream_id / "manifest.json").exists()


def test_process_command_unknown_persona_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeProvider("anthropic")
    _stub_everything(monkeypatch, fake_provider=fake)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "process",
            "https://kick.com/aldovillanueva/videos/abc",
            "--persona",
            "ghost",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "unknown persona" in (result.stderr or result.output)


def test_process_command_invalid_quality_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "process",
            "https://kick.com/aldovillanueva/videos/abc",
            "--persona",
            "aldo_villanueva",
            "--quality",
            "ultra",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert "must be 'standard' or 'premium'" in (result.stderr or result.output)
