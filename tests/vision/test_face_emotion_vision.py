"""Vision face-emotion detector — verdict translation + heuristic fallback."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from chalybclip.llm import LLMRouter, MemoryFrameStore
from chalybclip.llm.config import ProviderConfig
from chalybclip.vision.face_emotion_vision import detect_face_emotions_vision
from tests.llm._fakes import FakeProvider  # type: ignore[import]
from tests.llm._fixtures import make_llm_config  # type: ignore[import]


def _factory(providers: dict[str, FakeProvider]):
    def _build(name: str, _config: ProviderConfig, _api_key: str) -> FakeProvider | None:
        return providers.get(name)

    return _build


def _write_short_video(path: Path) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (320, 240))
    for _ in range(60):  # 2.0s at 30fps
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()


async def test_vision_emotion_returns_per_sample_labels(tmp_path: Path) -> None:
    video = tmp_path / "vid.mp4"
    _write_short_video(video)

    fake = FakeProvider("anthropic")
    # 0.5 Hz over 2s -> 1 sample.
    fake.queue_success(
        {"has_face": True, "emotion": "shock", "confidence": 0.85}
    )
    config = make_llm_config(purpose="vision_rescore", retry_attempts=1)
    router = LLMRouter(
        config=config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
    )
    frames = await detect_face_emotions_vision(
        tenant_id="default",
        stream_id="str_x",
        video_path=video,
        router=router,
        sample_rate_hz=0.5,
        frame_store=MemoryFrameStore(),
    )
    assert len(frames) == 1
    assert frames[0].has_face is True
    assert frames[0].emotion == "shock"
    assert frames[0].confidence == 0.85


async def test_vision_emotion_falls_back_on_llm_error(tmp_path: Path) -> None:
    video = tmp_path / "vid.mp4"
    _write_short_video(video)

    fake = FakeProvider("anthropic")
    fake.queue_fatal("provider down")
    config = make_llm_config(purpose="vision_rescore", retry_attempts=1)
    router = LLMRouter(
        config=config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
    )
    frames = await detect_face_emotions_vision(
        tenant_id="default",
        stream_id="str_x",
        video_path=video,
        router=router,
        sample_rate_hz=0.5,
        frame_store=MemoryFrameStore(),
    )
    # Heuristic fallback returns at least one frame for a 2s video at 0.5 Hz.
    # No face in the synthetic blank video, so has_face=False.
    assert len(frames) >= 1
    assert all(f.has_face is False for f in frames)


async def test_vision_emotion_no_face_clears_emotion(tmp_path: Path) -> None:
    video = tmp_path / "vid.mp4"
    _write_short_video(video)

    fake = FakeProvider("anthropic")
    fake.queue_success({"has_face": False, "emotion": None, "confidence": 0.0})
    config = make_llm_config(purpose="vision_rescore", retry_attempts=1)
    router = LLMRouter(
        config=config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
    )
    frames = await detect_face_emotions_vision(
        tenant_id="default",
        stream_id="str_x",
        video_path=video,
        router=router,
        sample_rate_hz=0.5,
        frame_store=MemoryFrameStore(),
    )
    assert len(frames) == 1
    assert frames[0].has_face is False
    assert frames[0].emotion is None


async def test_vision_emotion_unknown_label_falls_back_to_none(tmp_path: Path) -> None:
    """If the LLM returns a label outside the allowlist, we drop it to None."""
    video = tmp_path / "vid.mp4"
    _write_short_video(video)

    fake = FakeProvider("anthropic")
    fake.queue_success(
        {"has_face": True, "emotion": "ecstatic", "confidence": 0.9}
    )
    config = make_llm_config(purpose="vision_rescore", retry_attempts=1)
    router = LLMRouter(
        config=config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
    )
    frames = await detect_face_emotions_vision(
        tenant_id="default",
        stream_id="str_x",
        video_path=video,
        router=router,
        sample_rate_hz=0.5,
        frame_store=MemoryFrameStore(),
    )
    assert len(frames) == 1
    assert frames[0].has_face is True
    assert frames[0].emotion is None  # invalid label dropped
