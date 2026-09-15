"""Vision smart-crop picker — verdict translation + heuristic fallback."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from chalybclip.clip.smart_crop_vision import compute_smart_crop_box_vision
from chalybclip.errors import LLMError
from chalybclip.llm import LLMRouter, MemoryFrameStore
from chalybclip.llm.config import ProviderConfig
from tests.llm._fakes import FakeProvider  # type: ignore[import]
from tests.llm._fixtures import make_llm_config  # type: ignore[import]


def _factory(providers: dict[str, FakeProvider]):
    def _build(name: str, _config: ProviderConfig, _api_key: str) -> FakeProvider | None:
        return providers.get(name)

    return _build


def _write_solid_video(
    path: Path, *, w: int, h: int, fps: float, duration_s: float
) -> None:
    n = round(fps * duration_s)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for _ in range(n):
        writer.write(np.zeros((h, w, 3), dtype=np.uint8))
    writer.release()


async def test_vision_picker_returns_box_translated_from_verdict(tmp_path: Path) -> None:
    video = tmp_path / "vid.mp4"
    _write_solid_video(video, w=1920, h=1080, fps=30.0, duration_s=2.0)

    fake = FakeProvider("anthropic")
    # Verdict says "crop centered on x=0.55, width 0.32 of source"
    # -> source_w=1920 -> w_request=614 -> snapped to canonical 9:16 (607).
    # x_frac=0.55 -> x_request=1056.
    fake.queue_success({"x_frac": 0.55, "width_frac": 0.32, "reason": "face on right"})
    config = make_llm_config(purpose="vision_rescore", retry_attempts=1)
    router = LLMRouter(
        config=config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
    )
    box = await compute_smart_crop_box_vision(
        tenant_id="default",
        stream_id="str_x",
        video_path=video,
        start_s=0.0,
        end_s=2.0,
        router=router,
        frame_store=MemoryFrameStore(),
    )
    assert box.h == 1080
    canonical = (1080 * 9) // 16
    assert box.w == canonical
    # Picker clamped + centered the box near the 0.55-of-source position.
    assert 800 <= box.x <= 1313  # within sane left-edge range


async def test_vision_picker_falls_back_to_heuristic_on_llm_error(
    tmp_path: Path,
) -> None:
    video = tmp_path / "vid.mp4"
    _write_solid_video(video, w=1920, h=1080, fps=30.0, duration_s=1.0)

    fake = FakeProvider("anthropic")
    fake.queue_fatal("provider 401")
    config = make_llm_config(purpose="vision_rescore", retry_attempts=1)
    router = LLMRouter(
        config=config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
    )
    box = await compute_smart_crop_box_vision(
        tenant_id="default",
        stream_id="str_x",
        video_path=video,
        start_s=0.0,
        end_s=1.0,
        router=router,
        frame_store=MemoryFrameStore(),
    )
    # Heuristic picks centered crop on a blank video.
    assert box.h == 1080
    assert box.x == (1920 - box.w) // 2


async def test_vision_picker_short_circuits_for_already_portrait_source(
    tmp_path: Path,
) -> None:
    """A 9:16 source needs no LLM call - return the full frame."""
    video = tmp_path / "portrait.mp4"
    _write_solid_video(video, w=480, h=1080, fps=30.0, duration_s=1.0)

    fake = FakeProvider("anthropic")
    # No queued response - if the picker calls the provider this fails.
    config = make_llm_config(purpose="vision_rescore", retry_attempts=1)
    router = LLMRouter(
        config=config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
    )
    box = await compute_smart_crop_box_vision(
        tenant_id="default",
        stream_id="str_x",
        video_path=video,
        start_s=0.0,
        end_s=1.0,
        router=router,
    )
    assert box.x == 0 and box.w == 480 and box.h == 1080
    assert fake.calls == []


# Silence unused-import warning.
_ = (pytest, LLMError)
