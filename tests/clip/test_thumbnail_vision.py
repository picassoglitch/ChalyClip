"""Vision thumbnail picker — index translation + heuristic fallback."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from chalybclip.clip.thumbnail_vision import pick_thumbnail_vision
from chalybclip.llm import LLMRouter, MemoryFrameStore
from chalybclip.llm.config import ProviderConfig
from tests.llm._fakes import FakeProvider  # type: ignore[import]
from tests.llm._fixtures import make_llm_config  # type: ignore[import]


def _factory(providers: dict[str, FakeProvider]):
    def _build(name: str, _config: ProviderConfig, _api_key: str) -> FakeProvider | None:
        return providers.get(name)

    return _build


def _write_random_video(path: Path, *, n_frames: int = 60) -> None:
    """Random-pattern video frames — Phase 1's heuristic finds them sharp."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (320, 240))
    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        writer.write(rng.integers(0, 256, (240, 320, 3), dtype=np.uint8))
    writer.release()


async def test_vision_picker_picks_indexed_frame(tmp_path: Path) -> None:
    video = tmp_path / "vid.mp4"
    _write_random_video(video, n_frames=60)

    fake = FakeProvider("anthropic")
    fake.queue_success({"index": 3, "reason": "best face expression"})
    config = make_llm_config(purpose="vision_rescore", retry_attempts=1)
    router = LLMRouter(
        config=config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
    )
    blob, ts, breakdown = await pick_thumbnail_vision(
        tenant_id="default",
        stream_id="str_x",
        video_path=video,
        start_s=0.0,
        end_s=2.0,
        router=router,
        sample_n=5,
        frame_store=MemoryFrameStore(),
    )
    assert blob.startswith(b"\xff\xd8\xff")  # JPEG magic
    assert breakdown["rescore_index"] == 3.0
    assert breakdown["rescore_reason"] == "best face expression"
    # ts = 0 + 3 * (2/4) = 1.5
    assert abs(float(ts) - 1.5) < 0.01


async def test_vision_picker_falls_back_on_llm_error(tmp_path: Path) -> None:
    video = tmp_path / "vid.mp4"
    _write_random_video(video, n_frames=60)

    fake = FakeProvider("anthropic")
    fake.queue_fatal("provider 5xx")
    config = make_llm_config(purpose="vision_rescore", retry_attempts=1)
    router = LLMRouter(
        config=config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
    )
    # Patch out the haar cascade detection in the heuristic fallback so it
    # doesn't actually run face detection on the test stub.
    blob, _ts, breakdown = await pick_thumbnail_vision(
        tenant_id="default",
        stream_id="str_x",
        video_path=video,
        start_s=0.0,
        end_s=2.0,
        router=router,
        sample_n=5,
        frame_store=MemoryFrameStore(),
    )
    assert blob.startswith(b"\xff\xd8\xff")
    assert breakdown.get("fallback_reason") == "llm_error"
    assert "rescore_index" not in breakdown


async def test_vision_picker_clamps_out_of_range_index(tmp_path: Path) -> None:
    """If the model returns an index past the end, we clamp instead of crashing."""
    video = tmp_path / "vid.mp4"
    _write_random_video(video, n_frames=60)

    fake = FakeProvider("anthropic")
    fake.queue_success({"index": 99, "reason": "weird answer"})
    config = make_llm_config(purpose="vision_rescore", retry_attempts=1)
    router = LLMRouter(
        config=config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
    )
    blob, _ts, breakdown = await pick_thumbnail_vision(
        tenant_id="default",
        stream_id="str_x",
        video_path=video,
        start_s=0.0,
        end_s=2.0,
        router=router,
        sample_n=5,
        frame_store=MemoryFrameStore(),
    )
    assert blob
    # Clamped to last index = 4.
    assert breakdown["rescore_index"] == 4.0
