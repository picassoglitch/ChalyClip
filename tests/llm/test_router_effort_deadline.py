"""Per-purpose effort reaches the provider; a router call has a wall-time cap."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pytest
from pydantic import BaseModel

from chalybclip.errors import LLMError
from chalybclip.llm import LLMRouter, MultimodalImage, ProviderResult
from chalybclip.llm.config import LLMConfig, ProviderConfig, ProviderModelsConfig, RoutingRule

from ._fakes import FakeProvider
from ._fixtures import make_llm_config


class Answer(BaseModel):
    answer: str


def _router(config: LLMConfig, fake: FakeProvider, log: Path | None = None) -> LLMRouter:
    return LLMRouter(
        config,
        api_keys={"anthropic": "x"},
        provider_factory=lambda name, _c, _k: fake if name == "anthropic" else None,
        call_log_path=log,
    )


def _with_rules(config: LLMConfig, **rules: RoutingRule) -> LLMConfig:
    return config.model_copy(update={"routing": {**config.routing, **rules}})


async def test_effort_follows_the_purpose_not_the_model() -> None:
    config = _with_rules(
        make_llm_config(),
        extract=RoutingRule(primary="anthropic", effort="low"),
        create=RoutingRule(primary="anthropic"),
    )
    fake = FakeProvider("anthropic")
    for _ in range(3):
        fake.queue_success({"answer": "ok"})
    router = _router(config, fake)
    await router.complete(tenant_id="t", purpose="extract", system="s", user="u", schema=Answer)
    await router.complete(tenant_id="t", purpose="create", system="s", user="u", schema=Answer)
    await router.complete_multimodal(
        tenant_id="t",
        purpose="extract",
        system="s",
        user="u",
        images=[MultimodalImage(data=b"x")],
        schema=Answer,
    )
    assert [(c["kind"], c["effort"]) for c in fake.calls] == [
        ("text", "low"),
        ("text", None),
        ("multimodal", "low"),
    ]


class _SlowProvider(FakeProvider):
    """Every call sleeps longer than the deadline allows."""

    def __init__(self, delay_s: float) -> None:
        super().__init__("anthropic")
        self.delay_s = delay_s

    async def complete(self, **kw: object) -> ProviderResult:  # type: ignore[override]
        self.calls.append({"kind": "text"})
        await asyncio.sleep(self.delay_s)
        return ProviderResult(output={"answer": "late"}, model="m")


def _short_timeout(config: LLMConfig, timeout_s: float) -> LLMConfig:
    providers = dict(config.providers)
    providers["anthropic"] = providers["anthropic"].model_copy(update={"timeout_s": timeout_s})
    return config.model_copy(update={"providers": providers})


@pytest.mark.parametrize(("quality", "factor"), [("standard", 2.0), ("premium", 1.0)])
async def test_call_is_cut_at_its_quality_deadline(
    tmp_path: Path, quality: str, factor: float
) -> None:
    config = _short_timeout(make_llm_config(retry_attempts=5), timeout_s=0.05)
    assert config.retry.deadline_timeouts[quality] == factor  # type: ignore[index]
    slow = _SlowProvider(delay_s=10.0)
    log = tmp_path / "calls.jsonl"
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(LLMError, match="wall-time deadline"):
        await _router(config, slow, log).complete(
            tenant_id="t",
            purpose="variant_generation",
            system="s",
            user="u",
            schema=Answer,
            quality=quality,  # type: ignore[arg-type]
        )
    elapsed = loop.time() - started
    assert elapsed < 0.05 * factor + 0.5  # not 5 attempts x 10 s
    assert len(slow.calls) == 1  # the cancelled attempt is not retried
    row = json.loads(log.read_text("utf-8").splitlines()[-1])
    assert row["status"] == "error"


async def test_retries_stop_once_the_deadline_has_passed() -> None:
    # Each attempt fails fast but the backoff (1 s) outlasts the 2 x 0.05 s budget.
    config = _short_timeout(make_llm_config(retry_attempts=5, initial_backoff_s=1.0), 0.05)
    fake = FakeProvider("anthropic")
    for _ in range(5):
        fake.queue_retryable()
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(LLMError, match="wall-time deadline"):
        await _router(config, fake).complete(
            tenant_id="t", purpose="variant_generation", system="s", user="u", schema=Answer
        )
    assert loop.time() - started < 0.6
    assert len(fake.calls) == 1


def test_default_deadlines_are_two_and_one_timeouts() -> None:
    cfg = LLMConfig(
        providers={"anthropic": ProviderConfig(models={"standard": "m"})}  # type: ignore[arg-type]
    )
    assert cfg.retry.deadline_timeouts == {"standard": 2.0, "premium": 1.0}


async def test_deadline_cut_is_unbilled_but_reported_for_reconciliation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = _short_timeout(make_llm_config(retry_attempts=3), timeout_s=0.05)
    config = config.model_copy(update={"pricing": {}})
    providers = dict(config.providers)
    providers["anthropic"] = providers["anthropic"].model_copy(
        update={
            "models": ProviderModelsConfig(standard="claude-haiku-5-5"),
            "max_output_tokens": 1_000,
        }
    )
    config = config.model_copy(update={"providers": providers})
    log = tmp_path / "calls.jsonl"
    with (
        caplog.at_level(logging.WARNING, logger="chalybclip.llm.router"),
        pytest.raises(LLMError) as ei,
    ):
        await _router(config, _SlowProvider(10.0), log).complete(
            tenant_id="t",
            purpose="variant_generation",
            system="s" * 40,
            user="u" * 360,
            schema=Answer,
        )
    assert "in_flight_cancelled=true" in str(ei.value)
    row = json.loads(log.read_text("utf-8").splitlines()[-1])
    assert row["in_flight_cancelled"] is True
    assert (row["cost_usd_micros"], row["input_tokens"], row["output_tokens"]) == (0, 0, 0)
    record = next(r for r in caplog.records if "llm.unbilled_timeout" in r.getMessage())
    payload = json.loads(record.getMessage().split(" ", 1)[1])
    # 400 chars ~ 100 tokens at $0.10 + 1,000 max output at $0.50 = 10 + 500 µ$.
    assert payload == {
        "purpose": "variant_generation",
        "provider": "anthropic",
        "model": "claude-haiku-5-5",
        "in_flight_cancelled": True,
        "est_input_tokens": 100,
        "max_output_tokens": 1_000,
        "est_cost_usd_micros": 510,
    }


async def test_deadline_error_chains_the_last_attempt_error() -> None:
    config = _short_timeout(make_llm_config(retry_attempts=5, initial_backoff_s=1.0), 0.05)
    fake = FakeProvider("anthropic")
    for _ in range(5):
        fake.queue_retryable("503 overloaded")
    with pytest.raises(LLMError) as ei:
        await _router(config, fake).complete(
            tenant_id="t", purpose="variant_generation", system="s", user="u", schema=Answer
        )
    assert "503 overloaded" in str(ei.value.__cause__)


async def test_each_fallback_gets_its_own_deadline(tmp_path: Path) -> None:
    config = make_llm_config(retry_attempts=1, fallbacks=["openai"])
    providers = {
        name: cfg.model_copy(update={"timeout_s": 0.2}) for name, cfg in config.providers.items()
    }
    config = config.model_copy(update={"providers": providers})
    slow = _SlowProvider(1.0)  # primary: cut at 2 x 0.2 s
    fallback = FakeProvider("openai")
    fallback.queue_success({"answer": "ok"})

    async def late_fallback(**kw: object) -> ProviderResult:
        await asyncio.sleep(0.25)  # past the primary's window, inside its own
        return await FakeProvider.complete(fallback, **kw)  # type: ignore[arg-type]

    fallback.complete = late_fallback  # type: ignore[method-assign]
    router = LLMRouter(
        config,
        api_keys={"anthropic": "x", "openai": "y"},
        provider_factory=lambda name, _c, _k: {"anthropic": slow, "openai": fallback}.get(name),
        call_log_path=tmp_path / "c.jsonl",
    )
    out = await router.complete(
        tenant_id="t", purpose="variant_generation", system="s", user="u", schema=Answer
    )
    assert out.answer == "ok"


def test_yaml_price_masking_a_long_prompt_tier_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="chalybclip.llm.config"):
        LLMConfig.model_validate(
            {"pricing": {"anthropic": {"claude-haiku-5-5": {"input_per_mtok_usd": 0.1}}}}
        )
        LLMConfig.model_validate(
            {"pricing": {"anthropic": {"claude-opus-5-5": {"input_per_mtok_usd": 4}}}}
        )
    masked = [r for r in caplog.records if "llm.pricing_masks_long_tier" in r.getMessage()]
    assert len(masked) == 1 and "claude-haiku-5-5" in masked[0].getMessage()
