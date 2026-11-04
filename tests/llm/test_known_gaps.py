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
