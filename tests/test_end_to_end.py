"""Phase 0 end-to-end smoke test (the test PHASE_0.md task #8 calls for).

Runs the full `process_vod` pipeline with every external layer stubbed (no
network, no Whisper download, no ffmpeg, no Anthropic) and verifies the
exit-criterion artifact tree is produced. Fast enough for CI.

To run with a real VOD, drop the stubs and set `ANTHROPIC_API_KEY`. That's
out of scope for unit CI but the same `process_vod` entry point is used.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.pipeline.test_process_vod import (  # type: ignore[import]
    _hook_payload,
    _make_config,
    _make_personas,
    _make_router_factory,
    _stub_ffmpeg,
    _stub_ingest,
    _stub_whisper,
)
from tests.llm._fakes import FakeProvider  # type: ignore[import]
from tests.llm._fixtures import make_llm_config  # type: ignore[import]

from chalybclip.pipeline import PipelineDeps, process_vod


def test_phase_0_exit_criterion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reproduces the PHASE_0.md exit-criterion artifact tree end-to-end."""
    _stub_ingest(monkeypatch)
    _stub_whisper(monkeypatch)
    _stub_ffmpeg(monkeypatch)

    fake = FakeProvider("anthropic")
    fake.queue_success(_hook_payload())

    deps = PipelineDeps(
        config=_make_config(),
        llm_config=make_llm_config(retry_attempts=1),
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
            n_variants=5,
            deps=deps,
        )
    )

    stream_dir = tmp_path / manifest.stream.id

    # The exact tree PHASE_0.md promises in its exit criterion.
    assert (stream_dir / "manifest.json").exists()
    assert (stream_dir / "source" / "audio.wav").exists()
    assert (stream_dir / "source" / "transcript.json").exists()
    assert (stream_dir / "candidates.json").exists()
    assert (stream_dir / "clips_manifest.json").exists()
    assert (stream_dir / "llm_calls.jsonl").exists()
    assert manifest.clip_entries, "expected at least one clip"
    for entry in manifest.clip_entries:
        clip_dir = entry.clip.path.parent
        assert (clip_dir / "clip.mp4").exists()
        assert (clip_dir / "metadata.json").exists()
        assert (clip_dir / "variants.json").exists()
        # Variants are a deterministic stub now: exactly one per clip,
        # carrying the auto-hook (the only LLM call left in the pipeline).
        assert len(entry.variants) == 1

    # Cost tracking is non-optional (CLAUDE.md hard rule #6).
    assert manifest.llm_spend.total_calls == len(manifest.clip_entries)
    assert manifest.llm_spend.total_cost_usd_micros > 0
