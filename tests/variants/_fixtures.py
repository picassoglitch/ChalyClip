"""Shared fixtures for variant tests."""

from __future__ import annotations

from pathlib import Path

from chalybclip.clip import Clip
from chalybclip.detect import Candidate
from chalybclip.llm import LLMRouter
from chalybclip.llm.config import ProviderConfig
from chalybclip.variants import Persona

# pylint: disable=relative-beyond-top-level
from tests.llm._fakes import FakeProvider  # type: ignore[import]
from tests.llm._fixtures import make_llm_config  # type: ignore[import]


def make_persona(
    *,
    persona_id: str = "aldo_villanueva",
    primary_language: str = "es",
) -> Persona:
    return Persona(
        id=persona_id,
        name="Aldo Villanueva",
        target_languages=[primary_language, "en"],
        primary_language=primary_language,
        voice_prompt="Direct, confrontational, entrepreneur voice. No emojis.",
        routing_tags=["mindset", "irl"],
    )


def make_clip(tmp_path: Path, *, clip_id: str = "clp_01TEST") -> Clip:
    """Create a Clip with on-disk metadata.json under a fake stream layout."""
    stream_id = "str_01TEST"
    clip_dir = tmp_path / stream_id / "clips" / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / "clip.mp4"
    clip_path.write_bytes(b"fake")
    candidate = Candidate(
        timestamp=120.0,
        score=0.9,
        reason="voice",
        evidence={
            "phrase": "clipéalo",
            "language": "es",
            "transcript_snippet": "no manches eso clipéalo",
            "distance": 0,
            "confidence": 0.9,
        },
    )
    clip = Clip(
        id=clip_id,
        tenant_id="default",
        stream_id=stream_id,
        candidate=candidate,
        start_s=90.0,
        end_s=135.0,
        duration_s=45.0,
        width=1080,
        height=1920,
        path=clip_path,
    )
    (clip_dir / "metadata.json").write_text(
        clip.model_dump_json(indent=2), encoding="utf-8"
    )
    return clip


def make_router_with_fake(provider: FakeProvider) -> LLMRouter:
    """Build a real LLMRouter wired to a fake provider (no network)."""
    config = make_llm_config(retry_attempts=1)

    def factory(name: str, _config: ProviderConfig, _api_key: str):
        return provider if name == "anthropic" else None

    return LLMRouter(
        config=config,
        api_keys={"anthropic": "k"},
        provider_factory=factory,
    )
