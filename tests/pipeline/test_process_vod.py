"""End-to-end tests for `process_vod` with every external layer stubbed."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from chalybclip.clip import service as clip_service
from chalybclip.config import ChalybClipConfig, VoiceDetectorConfig
from chalybclip.errors import VariantError
from chalybclip.ingest import service as ingest_service
from chalybclip.llm import LLMRouter
from chalybclip.llm.config import ProviderConfig
from chalybclip.pipeline import PipelineDeps, StreamManifest, load_manifest, process_vod
from chalybclip.variants import Persona
from tests.llm._fakes import FakeProvider  # type: ignore[import]
from tests.llm._fixtures import make_llm_config  # type: ignore[import]
from tests.transcribe._fakes import (  # type: ignore[import]
    FakeInfo,
    FakeSegment,
    FakeWhisperModel,
    FakeWord,
)


def _stub_ingest(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Replace yt-dlp + ffmpeg-audio-extract with file-creating no-ops."""
    download_calls: list[Path] = []

    def fake_download(
        *,
        vod_url: str,
        target_path: Path,
        cookies_from_browser: str | None = None,
        cookies_file: str | None = None,
        platform: str = "unknown",
    ) -> dict[str, Any]:
        download_calls.append(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"\x00fakevideo")
        return {"duration": 600.0, "title": "Aldo VOD", "uploader": "aldovillanueva"}

    def fake_extract(video_path: Path, audio_path: Path) -> None:
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"RIFFfake")

    monkeypatch.setattr(ingest_service, "_download_vod", fake_download)
    monkeypatch.setattr(ingest_service, "_extract_audio", fake_extract)
    return download_calls


def _force_inprocess_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the in-process Whisper path so the WhisperModel monkeypatch
    # below actually intercepts the call. The default production path
    # spawns a subprocess that wouldn't see our patched fake. The
    # provider factory reads CHALYBCLIP_TRANSCRIBE_INPROCESS from the
    # environment, and the local provider lazy-imports
    # `faster_whisper.WhisperModel` inside `_run_inprocess`, so we patch
    # the module entry in sys.modules (stubbing it out when the real
    # faster-whisper isn't importable in the test env).
    import sys
    import types

    monkeypatch.setenv("CHALYBCLIP_TRANSCRIBE_INPROCESS", "1")
    # Defensive: invalidate the Settings singleton so the factory sees a
    # fresh `transcribe_provider` (tests share a cached Settings instance).
    from chalybclip.settings import get_settings

    get_settings.cache_clear()
    fw = sys.modules.get("faster_whisper")
    if fw is None:
        fw = types.ModuleType("faster_whisper")
        sys.modules["faster_whisper"] = fw
    monkeypatch.setattr(fw, "WhisperModel", FakeWhisperModel, raising=False)


def _stub_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_inprocess_whisper(monkeypatch)
    FakeWhisperModel.reset()
    FakeWhisperModel.canned_info = FakeInfo(language="es", duration=600.0)
    # One trigger phrase ("clipéalo") at t=120, plus filler.
    FakeWhisperModel.canned_segments = [
        FakeSegment(
            start=0.0,
            end=2.5,
            text="hola mundo",
            words=[
                FakeWord(start=0.0, end=0.5, word="hola", probability=0.95),
                FakeWord(start=0.6, end=1.2, word="mundo", probability=0.92),
            ],
        ),
        FakeSegment(
            start=119.5,
            end=121.0,
            text="no manches clipéalo",
            words=[
                FakeWord(start=119.5, end=120.0, word="no", probability=0.9),
                FakeWord(start=120.0, end=120.4, word="manches", probability=0.9),
                FakeWord(start=120.5, end=121.0, word="clipéalo", probability=0.93),
            ],
        ),
    ]


def _stub_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], *, what: str) -> None:
        calls.append(cmd)
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake mp4 bytes")

    monkeypatch.setattr(clip_service, "_run_ffmpeg", fake_run)
    return calls


def _make_config() -> ChalybClipConfig:
    cfg = ChalybClipConfig()
    cfg.detection.voice = VoiceDetectorConfig(
        enabled=True,
        weight=1.0,
        fuzzy_distance=2,
        phrases={"es": ["clipéalo"]},
    )
    cfg.detection.merge_window_s = 30.0
    return cfg


def _make_personas() -> dict[str, Persona]:
    return {
        "aldo_villanueva": Persona(
            id="aldo_villanueva",
            name="Aldo Villanueva",
            target_languages=["es", "en"],
            primary_language="es",
            voice_prompt="Direct entrepreneur voice.",
            routing_tags=["mindset"],
        )
    }


def _hook_payload() -> dict:
    """Canned auto-hook response — the only LLM call left in the pipeline
    (variants are a deterministic stub now)."""
    return {"hooks": [{"text": "HOOK VIRAL"}]}


def _make_router_factory(provider: FakeProvider):
    def factory(stream_dir: Path) -> LLMRouter:
        config = make_llm_config(retry_attempts=1, purpose="hook_generation")

        def provider_factory(
            name: str, _config: ProviderConfig, _api_key: str
        ):
            return provider if name == "anthropic" else None

        return LLMRouter(
            config=config,
            api_keys={"anthropic": "k"},
            provider_factory=provider_factory,
            call_log_path=stream_dir / "llm_calls.jsonl",
        )

    return factory


def _run_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    n_variants: int = 3,
    queue_n_payloads: int = 2,
    force: bool = False,
) -> tuple[StreamManifest, FakeProvider, list[list[str]], list[Path]]:
    download_calls = _stub_ingest(monkeypatch)
    _stub_whisper(monkeypatch)
    ffmpeg_calls = _stub_ffmpeg(monkeypatch)

    fake = FakeProvider("anthropic")
    for _ in range(queue_n_payloads):
        fake.queue_success(_hook_payload())

    deps = PipelineDeps(
        config=_make_config(),
        llm_config=make_llm_config(retry_attempts=1, purpose="hook_generation"),
        personas=_make_personas(),
        router_factory=_make_router_factory(fake),
    )
    manifest = asyncio.run(
        process_vod(
            tenant_id="default",
            vod_url="https://kick.com/aldovillanueva/videos/abc",
            output_dir=tmp_path,
            persona_id="aldo_villanueva",
            language="es",
            n_variants=n_variants,
            deps=deps,
            force=force,
        )
    )
    return manifest, fake, ffmpeg_calls, download_calls


def test_pipeline_runs_full_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, fake, ffmpeg_calls, _ = _run_pipeline(
        tmp_path, monkeypatch, n_variants=3, queue_n_payloads=1
    )

    assert manifest.stream.platform == "kick"
    assert manifest.transcript.segment_count == 2
    assert manifest.transcript.word_count == 5
    assert len(manifest.candidates) == 1
    assert manifest.candidates[0].evidence["phrase"] == "clipéalo"
    assert len(manifest.clip_entries) == 1
    # Variants are a deterministic stub now: exactly one per clip, carrying
    # the auto-hook as its title card.
    assert len(manifest.clip_entries[0].variants) == 1
    assert manifest.clip_entries[0].variants[0].id == "v_stub"
    assert manifest.clip_entries[0].variants[0].title_card_text == "HOOK VIRAL"
    assert manifest.persona_id == "aldo_villanueva"
    assert manifest.language == "es"

    stream_dir = tmp_path / manifest.stream.id
    assert (stream_dir / "manifest.json").exists()
    assert (stream_dir / "source" / "audio.wav").exists()
    assert (stream_dir / "source" / "transcript.json").exists()
    assert (stream_dir / "candidates.json").exists()
    assert (stream_dir / "clips_manifest.json").exists()
    assert (stream_dir / "llm_calls.jsonl").exists()

    clip_dir = manifest.clip_entries[0].clip.path.parent
    assert (clip_dir / "clip.mp4").exists()
    assert (clip_dir / "metadata.json").exists()
    assert (clip_dir / "variants.json").exists()

    # 1 candidate × (fast cut + reformat) = 2 ffmpeg calls
    assert len(ffmpeg_calls) == 2
    # 1 LLM call for the single clip's auto-hook
    assert len(fake.calls) == 1


def test_manifest_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, *_ = _run_pipeline(tmp_path, monkeypatch)
    reloaded = load_manifest(tmp_path / manifest.stream.id)
    assert reloaded == manifest


def test_manifest_records_llm_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, *_ = _run_pipeline(tmp_path, monkeypatch)
    assert manifest.llm_spend.total_calls == 1
    # Pricing fixture: 100 tokens in × 0.80 + 50 tokens out × 4.00 = 280 micros
    assert manifest.llm_spend.total_cost_usd_micros == 280


def test_pipeline_idempotent_with_explicit_stream_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running with `stream_id=first.stream.id` reuses every cached artifact."""
    download_calls = _stub_ingest(monkeypatch)
    _stub_whisper(monkeypatch)
    ffmpeg_calls = _stub_ffmpeg(monkeypatch)

    fake = FakeProvider("anthropic")
    # Only one payload queued — the second run must hit caches at every step
    # (including the auto-hook, cached in the clip's variants.json).
    fake.queue_success(_hook_payload())

    deps = PipelineDeps(
        config=_make_config(),
        llm_config=make_llm_config(retry_attempts=1, purpose="hook_generation"),
        personas=_make_personas(),
        router_factory=_make_router_factory(fake),
    )
    first = asyncio.run(
        process_vod(
            tenant_id="default",
            vod_url="https://kick.com/aldovillanueva/videos/abc",
            output_dir=tmp_path,
            persona_id="aldo_villanueva",
            language="es",
            n_variants=3,
            deps=deps,
        )
    )
    download_count_before = len(download_calls)
    ffmpeg_count_before = len(ffmpeg_calls)

    second = asyncio.run(
        process_vod(
            tenant_id="default",
            vod_url="https://kick.com/aldovillanueva/videos/abc",
            output_dir=tmp_path,
            persona_id="aldo_villanueva",
            stream_id=first.stream.id,
            language="es",
            n_variants=3,
            deps=deps,
        )
    )
    assert second.stream.id == first.stream.id
    # All side-effects skipped on second run.
    assert len(download_calls) == download_count_before
    assert len(ffmpeg_calls) == ffmpeg_count_before
    assert len(fake.calls) == 1


def test_pipeline_force_reruns_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`force=True` ignores caches at every step."""
    download_calls = _stub_ingest(monkeypatch)
    _stub_whisper(monkeypatch)
    ffmpeg_calls = _stub_ffmpeg(monkeypatch)

    fake = FakeProvider("anthropic")
    fake.queue_success(_hook_payload())
    fake.queue_success(_hook_payload())  # one for each run

    deps = PipelineDeps(
        config=_make_config(),
        llm_config=make_llm_config(retry_attempts=1, purpose="hook_generation"),
        personas=_make_personas(),
        router_factory=_make_router_factory(fake),
    )
    first = asyncio.run(
        process_vod(
            tenant_id="default",
            vod_url="https://kick.com/aldovillanueva/videos/abc",
            output_dir=tmp_path,
            persona_id="aldo_villanueva",
            language="es",
            n_variants=3,
            deps=deps,
        )
    )
    asyncio.run(
        process_vod(
            tenant_id="default",
            vod_url="https://kick.com/aldovillanueva/videos/abc",
            output_dir=tmp_path,
            persona_id="aldo_villanueva",
            stream_id=first.stream.id,
            language="es",
            n_variants=3,
            deps=deps,
            force=True,
        )
    )
    # Both runs did real work.
    assert len(download_calls) == 2
    # 1 candidate × (cut + reformat) × 2 runs
    assert len(ffmpeg_calls) == 4
    assert len(fake.calls) == 2


def test_pipeline_unknown_persona_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deps = PipelineDeps(
        config=_make_config(),
        llm_config=make_llm_config(),
        personas=_make_personas(),
    )
    with pytest.raises(VariantError, match="unknown persona"):
        asyncio.run(
            process_vod(
                tenant_id="default",
                vod_url="https://kick.com/aldovillanueva/videos/abc",
                output_dir=tmp_path,
                persona_id="ghost",
                deps=deps,
            )
        )


def test_pipeline_no_triggers_falls_back_to_interval_clips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stream with no trigger phrase still produces clips: when no detector
    fires, detect falls back to evenly spaced interval candidates (silent /
    no-speech VODs must not yield an empty run)."""
    _stub_ingest(monkeypatch)
    _stub_ffmpeg(monkeypatch)

    # Reset whisper to a transcript with no trigger words.
    _force_inprocess_whisper(monkeypatch)
    FakeWhisperModel.reset()
    FakeWhisperModel.canned_info = FakeInfo(language="es", duration=60.0)
    FakeWhisperModel.canned_segments = [
        FakeSegment(
            start=0.0,
            end=2.0,
            text="just talking",
            words=[FakeWord(start=0.0, end=2.0, word="hola", probability=0.9)],
        )
    ]

    fake = FakeProvider("anthropic")  # no payloads queued — hooks fall back
    deps = PipelineDeps(
        config=_make_config(),
        llm_config=make_llm_config(retry_attempts=1, purpose="hook_generation"),
        personas=_make_personas(),
        router_factory=_make_router_factory(fake),
    )
    manifest = asyncio.run(
        process_vod(
            tenant_id="default",
            vod_url="https://kick.com/aldovillanueva/videos/abc",
            output_dir=tmp_path,
            persona_id="aldo_villanueva",
            n_variants=3,
            deps=deps,
        )
    )
    # Ingest stub reports a 600s VOD → 5 interval candidates (capped).
    assert len(manifest.candidates) == 5
    assert all(c.reason == "interval" for c in manifest.candidates)
    assert len(manifest.clip_entries) == 5
    assert manifest.llm_spend.total_calls == 0
    assert (tmp_path / manifest.stream.id / "manifest.json").exists()


def test_pipeline_writes_manifest_with_clip_paths_relative_to_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, *_ = _run_pipeline(tmp_path, monkeypatch)
    raw = json.loads(
        (tmp_path / manifest.stream.id / "manifest.json").read_text("utf-8")
    )
    assert raw["schema_version"] == 1
    assert raw["stream"]["id"] == manifest.stream.id
    assert raw["clip_entries"][0]["clip"]["id"].startswith("clp_")
