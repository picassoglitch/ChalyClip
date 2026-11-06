"""Gaps found in the full-pass of fix/llm-structured-output, now fixed: mixed routing
reserves for the paid purposes, and a context-window stop is billed once, not retried."""

from __future__ import annotations

from pathlib import Path

import pytest

from chalybclip.llm import (
    ContextWindowExceededError,
    RetryableLLMError,
    TruncatedOutputError,
    estimator,
)
from chalybclip.llm.anthropic_provider import _parse_message
from chalybclip.llm.config import load_llm_config

from .test_anthropic_wire import message

EXAMPLE = Path(__file__).resolve().parents[2] / "config" / "llm.example.yaml"


def test_mixed_routing_reserves_for_the_paid_standard_purposes(tmp_path: Path) -> None:
    # Vision on the free local VL model, text purposes still on Haiku 5.5.
    text = EXAMPLE.read_text("utf-8").replace(
        "  vision_rescore:                               # frame-based rescore of detect candidates\n"
        "    default_quality: standard\n"
        "    primary: anthropic",
        "  vision_rescore:                               # frame-based rescore of detect candidates\n"
        "    default_quality: standard\n"
        "    primary: openllm_vision",
    )
    cfg_path = tmp_path / "llm.yaml"
    cfg_path.write_text(text, "utf-8")
    cfg = load_llm_config(cfg_path)
    assert cfg.routing["vision_rescore"].primary == "openllm_vision"  # the edit applied
    assert cfg.routing["hook_generation"].primary == "anthropic"
    assert cfg.routing["viral_detection"].primary == "anthropic"
    est = estimator.estimate_pipeline_run(duration_seconds=3600, llm_config=cfg)
    # Priced at Haiku 5.5 (the paid standard model), not at the free VL model.
    assert est.llm_cost_usd_micros == round(est.total_tokens * 0.40)


def test_context_window_stop_is_not_retried() -> None:
    from anthropic.types import Message

    reply = message(
        "claude-haiku-5-5",
        '{"moments": [',
        raw=True,
        stop="model_context_window_exceeded",
        input_tokens=990_000,
    )
    msg = Message.model_validate(reply.body)
    with pytest.raises((TruncatedOutputError, RetryableLLMError)) as info:
        _parse_message(msg, model="claude-haiku-5-5")
    assert not isinstance(info.value, RetryableLLMError)
    assert isinstance(info.value, ContextWindowExceededError)
    assert info.value.usage is not None and info.value.usage.input_tokens == 990_000


async def test_router_bills_a_context_window_stop_once(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from chalybclip.errors import LLMError
    from chalybclip.llm import LLMRouter
    from chalybclip.llm.anthropic_provider import AnthropicProvider
    from chalybclip.llm.config import ProviderConfig, ProviderModelsConfig
    from chalybclip.variants.hooks import HookBatch

    from ._fixtures import make_llm_config

    calls: list[dict[str, object]] = []

    async def create(**kw: object) -> object:
        calls.append(kw)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"hooks": [')],
            stop_reason="model_context_window_exceeded",
            stop_details=None,
            usage=SimpleNamespace(input_tokens=990_000, output_tokens=40),
        )

    provider = AnthropicProvider(
        api_key="k", config=ProviderConfig(models=ProviderModelsConfig(standard="m"))
    )
    provider._client = SimpleNamespace(messages=SimpleNamespace(create=create))  # type: ignore[assignment]
    log = tmp_path / "calls.jsonl"
    router = LLMRouter(
        make_llm_config(standard_model="claude-haiku-5-5", retry_attempts=3).model_copy(
            update={"pricing": {}}
        ),
        api_keys={"anthropic": "k"},
        provider_factory=lambda name, _c, _k: provider if name == "anthropic" else None,
        call_log_path=log,
    )
    with pytest.raises(LLMError, match="ContextWindowExceededError"):
        await router.complete(
            tenant_id="t", purpose="variant_generation", system="s", user="u", schema=HookBatch
        )
    assert len(calls) == 1
    import json

    row = json.loads(log.read_text("utf-8").splitlines()[-1])
    assert (row["attempts"], row["input_tokens"], row["output_tokens"]) == (1, 990_000, 40)
    # One request over 100K: the long-prompt tier, billed once.
    assert row["cost_usd_micros"] == round(990_000 * 0.50 + 40 * 2.50)
