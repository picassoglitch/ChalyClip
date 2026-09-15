"""OpenAICompatibleProvider — the self-hosted / open-model path.

Covers: keyless construction through the real router, schema-in-prompt +
JSON-mode request shape, defensive parsing (<think> blocks, fences,
surrounding prose), retryable vs fatal status mapping, and the
data-URL image shape for multimodal calls.
"""

from __future__ import annotations

import asyncio
import base64
import json

import httpx
import pytest
import respx
from pydantic import BaseModel

from chalybclip.errors import LLMError
from chalybclip.llm import LLMConfig, LLMRouter
from chalybclip.llm.config import (
    ProviderConfig,
    ProviderModelsConfig,
    RetryConfig,
    RoutingRule,
)
from chalybclip.llm.openai_compatible_provider import (
    OpenAICompatibleProvider,
    _extract_json_object,
)
from chalybclip.llm.provider import MultimodalImage, RetryableLLMError

_BASE = "http://llm.test/v1"


class TinySchema(BaseModel):
    answer: str


def _config(*, purpose: str = "hook_generation") -> LLMConfig:
    return LLMConfig(
        providers={
            "openllm": ProviderConfig(
                kind="openai_compatible",
                api_key_env="OPENLLM_API_KEY",
                api_key_required=False,
                base_url=_BASE,
                models=ProviderModelsConfig(standard="qwen2.5:7b-instruct"),
            ),
        },
        routing={purpose: RoutingRule(default_quality="standard", primary="openllm")},
        retry=RetryConfig(max_attempts=2, initial_backoff_s=0.0, backoff_multiplier=1.0),
    )


def _completion(content: str, *, prompt_tokens: int = 11, completion_tokens: int = 7) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def test_keyless_provider_completes_through_router(tmp_path) -> None:
    """No API key anywhere — the router still constructs the provider
    (api_key_required: false) and returns a validated model. Zero paid APIs."""
    router = LLMRouter(
        _config(),
        api_keys={},  # explicitly empty: nothing set in the environment
        call_log_path=tmp_path / "llm_calls.jsonl",
    )
    with respx.mock() as mock:
        route = mock.post(f"{_BASE}/chat/completions").mock(
            return_value=httpx.Response(200, json=_completion('{"answer": "hola"}'))
        )
        result = asyncio.run(
            router.complete(
                tenant_id="t", purpose="hook_generation",
                system="sys", user="usr", schema=TinySchema,
            )
        )
    assert result.answer == "hola"

    body = json.loads(route.calls.last.request.content.decode())
    assert body["model"] == "qwen2.5:7b-instruct"
    assert body["response_format"] == {"type": "json_object"}
    # No Authorization header when keyless.
    assert "authorization" not in {k.lower() for k in route.calls.last.request.headers}
    # The schema rides in the system prompt so any runtime can comply.
    assert "json" in body["messages"][0]["content"].lower()
    assert "answer" in body["messages"][0]["content"]

    rows = (tmp_path / "llm_calls.jsonl").read_text("utf-8").splitlines()
    row = json.loads(rows[-1])
    assert row["status"] == "ok"
    assert row["cost_usd_micros"] == 0  # open model: no pricing entry → $0


def test_non_json_reply_is_retried_then_succeeds(tmp_path) -> None:
    router = LLMRouter(
        _config(), api_keys={}, call_log_path=tmp_path / "llm_calls.jsonl",
    )
    with respx.mock() as mock:
        route = mock.post(f"{_BASE}/chat/completions")
        route.side_effect = [
            httpx.Response(200, json=_completion("sure! here you go:")),  # ramble
            httpx.Response(
                200,
                json=_completion('```json\n{"answer": "42"}\n```'),  # fenced
            ),
        ]
        result = asyncio.run(
            router.complete(
                tenant_id="t", purpose="hook_generation",
                system="s", user="u", schema=TinySchema,
            )
        )
    assert result.answer == "42"
    assert route.call_count == 2


def test_5xx_is_retryable_and_4xx_is_fatal() -> None:
    provider = OpenAICompatibleProvider(
        api_key="", config=_config().providers["openllm"]
    )

    async def _call() -> None:
        await provider.complete(
            tenant_id="t", model="m", system="s", user="u", schema=TinySchema
        )

    with respx.mock() as mock:
        mock.post(f"{_BASE}/chat/completions").mock(
            return_value=httpx.Response(503, text="overloaded")
        )
        with pytest.raises(RetryableLLMError):
            asyncio.run(_call())

    with respx.mock() as mock:
        mock.post(f"{_BASE}/chat/completions").mock(
            return_value=httpx.Response(404, text="model not found")
        )
        with pytest.raises(LLMError) as exc_info:
            asyncio.run(_call())
        assert not isinstance(exc_info.value, RetryableLLMError)


def test_multimodal_sends_data_url_image_parts() -> None:
    provider = OpenAICompatibleProvider(
        api_key="sk-hosted", config=_config().providers["openllm"]
    )
    img = MultimodalImage(media_type="image/jpeg", data=b"\xff\xd8fakejpeg")
    with respx.mock() as mock:
        route = mock.post(f"{_BASE}/chat/completions").mock(
            return_value=httpx.Response(200, json=_completion('{"answer": "seen"}'))
        )
        result = asyncio.run(
            provider.complete_multimodal(
                tenant_id="t", model="qwen2.5vl:7b", system="s", user="describe",
                images=[img], schema=TinySchema,
            )
        )
    assert result.output == {"answer": "seen"}
    body = json.loads(route.calls.last.request.content.decode())
    parts = body["messages"][1]["content"]
    expected_b64 = base64.b64encode(b"\xff\xd8fakejpeg").decode("ascii")
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"] == f"data:image/jpeg;base64,{expected_b64}"
    assert parts[-1] == {"type": "text", "text": "describe"}
    # Hosted gateways get the bearer header when a key IS provided.
    assert route.calls.last.request.headers["authorization"] == "Bearer sk-hosted"


def test_extract_json_tolerates_think_blocks_fences_and_prose() -> None:
    assert _extract_json_object('{"a": 1}') == {"a": 1}
    assert _extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json_object(
        "<think>the user wants json...</think>\nHere it is: {\"a\": 1} hope that helps!"
    ) == {"a": 1}
    assert _extract_json_object("no json here") is None
    assert _extract_json_object('[1, 2, 3]') is None  # object required
