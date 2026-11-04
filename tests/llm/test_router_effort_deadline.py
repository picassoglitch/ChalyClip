"""Per-purpose effort reaches the provider; a router call has a wall-time cap."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from chalybclip.errors import LLMError
from chalybclip.llm import LLMRouter, MultimodalImage, ProviderResult
from chalybclip.llm.config import LLMConfig, ProviderConfig, RoutingRule

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
