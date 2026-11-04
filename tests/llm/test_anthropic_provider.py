"""AnthropicProvider request/response shape, against a fake SDK client."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, Field

from chalybclip.errors import LLMError
from chalybclip.llm import (
    LLMRouter,
    MultimodalImage,
    RetryableLLMError,
    TruncatedOutputError,
    VariantBatch,
)
from chalybclip.llm.anthropic_provider import AnthropicProvider, output_format_for
from chalybclip.llm.config import ProviderConfig, ProviderModelsConfig

# The 4.5/5 pair stays covered for a config rollback; the 5.5 family rejects
# forced tool_choice with a 400.
MODELS = [
    "claude-haiku-4-5",
    "claude-opus-5",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
]


class Verdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=50)


class _FakeMessages:
    def __init__(self, replies: list[SimpleNamespace]) -> None:
        self.replies = replies
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.replies.pop(0)


def _reply(
    content: list[SimpleNamespace], *, stop: str = "end_turn", **usage: int
) -> SimpleNamespace:
    base = {"input_tokens": 10, "output_tokens": 5}
    base.update(usage)
    return SimpleNamespace(
        content=content, stop_reason=stop, stop_details=None, usage=SimpleNamespace(**base)
    )


def _text(obj: Any) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=obj if isinstance(obj, str) else json.dumps(obj))


def _provider(*replies: SimpleNamespace) -> tuple[AnthropicProvider, _FakeMessages]:
    cfg = ProviderConfig(
        api_key_env="ANTHROPIC_API_KEY",
        models=ProviderModelsConfig(standard="m"),
        max_output_tokens=1234,
    )
    provider = AnthropicProvider(api_key="test", config=cfg)
    fake = _FakeMessages(list(replies))
    provider._client = SimpleNamespace(messages=fake)  # type: ignore[assignment]
    return provider, fake


@pytest.mark.parametrize("model", MODELS)
async def test_text_call_uses_output_config_not_forced_tools(model: str) -> None:
    provider, fake = _provider(_reply([_text({"score": 0.7, "reason": "ok"})]))
    res = await provider.complete(
        tenant_id="t", model=model, system="sys", user="hi", schema=Verdict
    )
    call = fake.calls[0]
    assert "tool_choice" not in call and "tools" not in call
    assert call["model"] == model
    assert call["max_tokens"] == 1234
    assert call["output_config"] == {"format": output_format_for(Verdict)}
    assert call["messages"] == [{"role": "user", "content": "hi"}]
    assert res.output == {"score": 0.7, "reason": "ok"}
    assert (res.input_tokens, res.output_tokens, res.model) == (10, 5, model)


async def test_multimodal_call_sends_images_then_text_with_output_config() -> None:
    provider, fake = _provider(_reply([_text({"score": 0.1, "reason": "flat"})]))
    res = await provider.complete_multimodal(
        tenant_id="t",
        model="claude-opus-5-5",
        system="s",
        user="rate",
        images=[MultimodalImage(data=b"\xff\xd8jpeg")],
        schema=Verdict,
    )
    call = fake.calls[0]
    assert "tool_choice" not in call
    content = call["messages"][0]["content"]
    assert [b["type"] for b in content] == ["image", "text"]
    assert content[0]["source"]["data"] == "/9hqcGVn"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert res.output["reason"] == "flat"


def test_output_format_is_api_safe() -> None:
    schema = output_format_for(Verdict)["schema"]
    assert schema["additionalProperties"] is False
    score = schema["properties"]["score"]
    assert "minimum" not in score and "maximum" not in score  # moved into the description
    assert "minimum: 0.0" in score["description"]
    reason = schema["properties"]["reason"]
    assert "minLength" not in reason and "maxLength" not in reason
    nested = output_format_for(VariantBatch)["schema"]["$defs"]["Variant"]
    assert nested["additionalProperties"] is False
    assert output_format_for(Verdict) is output_format_for(Verdict)  # cached


async def test_thinking_blocks_before_the_json_are_ignored() -> None:
    provider, _ = _provider(
        _reply(
            [
                SimpleNamespace(type="thinking", thinking=""),
                _text({"score": 1.0, "reason": "x"}),
            ]
        )
    )
    res = await provider.complete(
        tenant_id="t", model="claude-opus-5-5", system="s", user="u", schema=Verdict
    )
    assert res.output == {"score": 1.0, "reason": "x"}


@pytest.mark.parametrize(
    ("reply", "match"),
    [
        (_reply([_text("[1, 2]")], output_tokens=99), "not a JSON object"),
        (_reply([], output_tokens=99), "not a JSON object"),
        (_reply([_text("")], stop="refusal", output_tokens=99), "refused"),
    ],
)
async def test_unusable_reply_is_retryable_and_billed(reply: SimpleNamespace, match: str) -> None:
    provider, _ = _provider(reply)
    with pytest.raises(RetryableLLMError, match=match) as ei:
        await provider.complete(
            tenant_id="t", model="claude-opus-5-5", system="s", user="u", schema=Verdict
        )
    assert ei.value.usage is not None and ei.value.usage.output_tokens == 99


async def test_max_tokens_is_not_retried_but_is_billed() -> None:
    reply = _reply([_text('{"score": 0.5, "rea')], stop="max_tokens", output_tokens=1234)
    provider, _ = _provider(reply)
    with pytest.raises(TruncatedOutputError) as ei:
        await provider.complete(
            tenant_id="t", model="claude-opus-5-5", system="s", user="u", schema=Verdict
        )
    assert not isinstance(ei.value, RetryableLLMError)
    assert ei.value.usage is not None and ei.value.usage.output_tokens == 1234


async def test_router_bills_a_truncated_attempt_once(tmp_path: Path) -> None:
    from tests.llm._fixtures import make_llm_config

    reply = _reply([_text("{")], stop="max_tokens", input_tokens=1000, output_tokens=2000)
    provider, fake = _provider(reply)
    router = LLMRouter(
        make_llm_config(purpose="p"),
        api_keys={"anthropic": "x"},
        call_log_path=tmp_path / "c.jsonl",
        provider_factory=lambda name, _c, _k: provider if name == "anthropic" else None,
    )
    with pytest.raises(LLMError):
        await router.complete(tenant_id="t", purpose="p", system="s", user="u", schema=Verdict)
    assert len(fake.calls) == 1  # not retried
    row = json.loads((tmp_path / "c.jsonl").read_text("utf-8"))
    assert (row["status"], row["input_tokens"], row["output_tokens"]) == ("error", 1000, 2000)
    assert row["cost_usd_micros"] > 0


async def test_effort_is_sent_only_to_models_that_accept_it() -> None:
    provider, fake = _provider(
        _reply([_text({"score": 0.5, "reason": "a"})]),
        _reply([_text({"score": 0.5, "reason": "b"})]),
        _reply([_text({"score": 0.5, "reason": "c"})]),
    )
    for model, effort in (
        ("claude-haiku-5-5", "low"),
        ("claude-haiku-4-5", "low"),
        ("claude-opus-5-5", None),
    ):
        await provider.complete(
            tenant_id="t", model=model, system="s", user="u", schema=Verdict, effort=effort
        )
    haiku55, haiku45, opus55 = fake.calls
    assert haiku55["output_config"]["effort"] == "low"
    assert "effort" not in haiku45["output_config"]  # Haiku 4.5 rejects effort
    assert "effort" not in opus55["output_config"]  # unset keeps the model default
    for call in fake.calls:
        assert not {"thinking", "temperature", "top_p", "top_k"} & call.keys()


def test_sdk_client_does_not_retry_on_its_own() -> None:
    provider, _ = _provider()
    assert AnthropicProvider(api_key="k", config=provider._config)._client.max_retries == 0


EXTRACTION = ("viral_detection", "vision_rescore", "clip_scoring")
CREATIVE = ("hook_generation", "logo_generation", "variant_generation_visual", "agent_decision")


def test_committed_example_config_targets_the_5_5_family() -> None:
    from chalybclip.llm.config import load_llm_config

    cfg = load_llm_config(Path(__file__).parents[2] / "config" / "llm.example.yaml")
    anthropic_cfg = cfg.providers["anthropic"]
    assert (anthropic_cfg.models.standard, anthropic_cfg.models.premium) == (
        "claude-haiku-5-5",
        "claude-opus-5-5",
    )
    assert anthropic_cfg.timeout_s >= 300
    assert {p: cfg.routing[p].effort for p in EXTRACTION} == dict.fromkeys(EXTRACTION, "low")
    assert {p: cfg.routing[p].effort for p in CREATIVE} == dict.fromkeys(CREATIVE)
    assert set(EXTRACTION + CREATIVE) == set(cfg.routing)  # a new purpose must pick a side
