"""Tests for `LLMRouter.complete_multimodal` - shares retry/fallback/log
with the text path, so we mostly verify that images make it through and
the shared error machinery still triggers.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from chalybclip.errors import LLMError
from chalybclip.llm import LLMRouter, MultimodalImage
from chalybclip.llm.config import ProviderConfig

from ._fakes import FakeProvider
from ._fixtures import make_llm_config


class TinySchema(BaseModel):
    answer: str


def _factory(providers: dict[str, FakeProvider]):
    def _build(name: str, _config: ProviderConfig, _api_key: str) -> FakeProvider | None:
        return providers.get(name)

    return _build


def test_multimodal_passes_images_to_provider(tmp_path: Path) -> None:
    fake = FakeProvider("anthropic")
    fake.queue_success({"answer": "ok"})
    config = make_llm_config()
    router = LLMRouter(
        config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
        call_log_path=tmp_path / "llm.jsonl",
    )
    images = [
        MultimodalImage(media_type="image/jpeg", data=b"\xff\xd8\xff\x00frame-1"),
        MultimodalImage(media_type="image/jpeg", data=b"\xff\xd8\xff\x00frame-2"),
    ]
    result = asyncio.run(
        router.complete_multimodal(
            tenant_id="alice",
            purpose="variant_generation",
            system="caption it",
            user="3 variants",
            images=images,
            schema=TinySchema,
        )
    )
    assert result.answer == "ok"
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["kind"] == "multimodal"
    assert call["n_images"] == 2
    assert call["images"][0].data == b"\xff\xd8\xff\x00frame-1"


def test_multimodal_rejects_empty_images_list() -> None:
    config = make_llm_config()
    router = LLMRouter(config, api_keys={"anthropic": "k"})
    with pytest.raises(LLMError, match="at least one image"):
        asyncio.run(
            router.complete_multimodal(
                tenant_id="alice",
                purpose="variant_generation",
                system="s",
                user="u",
                images=[],
                schema=TinySchema,
            )
        )


def test_multimodal_logs_call_to_jsonl(tmp_path: Path) -> None:
    fake = FakeProvider("anthropic")
    fake.queue_success({"answer": "x"}, input_tokens=200, output_tokens=80)
    log_path = tmp_path / "llm_calls.jsonl"
    config = make_llm_config(pricing_input=1.0, pricing_output=5.0)
    router = LLMRouter(
        config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
        call_log_path=log_path,
    )
    asyncio.run(
        router.complete_multimodal(
            tenant_id="alice",
            purpose="variant_generation",
            system="s",
            user="u",
            images=[MultimodalImage(data=b"jpg")],
            schema=TinySchema,
        )
    )
    rows = [json.loads(line) for line in log_path.read_text("utf-8").splitlines() if line]
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "ok"
    assert row["purpose"] == "variant_generation"
    assert row["input_tokens"] == 200
    assert row["output_tokens"] == 80
    # cost = 200 * 1.0 + 80 * 5.0 = 600 (in micros, since pricing is per Mtok)
    assert row["cost_usd_micros"] == 600


def test_multimodal_retries_on_retryable(tmp_path: Path) -> None:
    fake = FakeProvider("anthropic")
    fake.queue_retryable("503")
    fake.queue_success({"answer": "second-try"})
    config = make_llm_config(retry_attempts=3)
    router = LLMRouter(
        config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
        call_log_path=tmp_path / "llm.jsonl",
    )
    result = asyncio.run(
        router.complete_multimodal(
            tenant_id="alice",
            purpose="variant_generation",
            system="s",
            user="u",
            images=[MultimodalImage(data=b"f")],
            schema=TinySchema,
        )
    )
    assert result.answer == "second-try"
    assert len(fake.calls) == 2


def test_multimodal_falls_through_to_secondary_provider(tmp_path: Path) -> None:
    primary = FakeProvider("anthropic")
    primary.queue_fatal("401")
    secondary = FakeProvider("openai")
    secondary.queue_success({"answer": "from-fallback"})

    config = make_llm_config(primary="anthropic", fallbacks=["openai"])
    router = LLMRouter(
        config,
        api_keys={"anthropic": "k", "openai": "k"},
        provider_factory=_factory({"anthropic": primary, "openai": secondary}),
        call_log_path=tmp_path / "llm.jsonl",
    )
    result = asyncio.run(
        router.complete_multimodal(
            tenant_id="alice",
            purpose="variant_generation",
            system="s",
            user="u",
            images=[MultimodalImage(data=b"f")],
            schema=TinySchema,
        )
    )
    assert result.answer == "from-fallback"
    assert len(primary.calls) == 1
    assert len(secondary.calls) == 1
