"""Correct LLM metering: committed price table, prompt-cache pricing,
conservative rate for unpriced paid models, and billing EVERY attempt
(schema-invalid retries, failed calls) — with the token split on the
usage event."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from chalybclip.db import Database, LLMCallsRepo, TenantsRepo, apply_migrations
from chalybclip.db.usage_repos import UsageOutboxRepo
from chalybclip.errors import LLMError
from chalybclip.llm import LLMRouter, ProviderResult, RetryableLLMError
from chalybclip.llm.anthropic_provider import _parse_message
from chalybclip.llm.config import LLMConfig, ProviderConfig, ProviderModelsConfig
from chalybclip.llm.prices import CONSERVATIVE_RATES, cost_micros, lookup_rates
from chalybclip.tenancy import bound_tenant

from ._fakes import FakeProvider
from ._fixtures import make_llm_config


class Answer(BaseModel):
    answer: str


def _factory(provider: FakeProvider):
    def _build(name, _config, _api_key):
        return provider if name == "anthropic" else None

    return _build


# ---- price table -------------------------------------------------------


def test_cache_tokens_are_priced_at_anthropic_multipliers() -> None:
    rates = lookup_rates("anthropic", "claude-haiku-4-5")
    assert rates is not None
    # $1 in / $5 out; cache read 0.1x input, 5-minute cache write 1.25x.
    got = cost_micros(
        rates, input_tokens=1_000, output_tokens=500,
        cache_read_tokens=10_000, cache_write_tokens=2_000,
    )
    assert got == 1_000 * 1 + 500 * 5 + round(10_000 * 0.1) + round(2_000 * 1.25)
    assert got == 7_000


@pytest.mark.parametrize(
    ("model", "rin", "rout"),
    [
        ("claude-haiku-4-5", 1.0, 5.0),
        ("claude-haiku-4-5-20251001", 1.0, 5.0),  # dated snapshot → family
        ("claude-sonnet-4-5", 3.0, 15.0),
        ("claude-opus-5", 5.0, 25.0),
        ("claude-opus-4-7", 5.0, 25.0),
    ],
)
def test_committed_prices_cover_referenced_models(model: str, rin: float, rout: float) -> None:
    rates = lookup_rates("anthropic", model)
    assert rates is not None
    assert (rates.input, rates.output) == (rin, rout)


def test_unknown_paid_model_uses_conservative_rate() -> None:
    cfg = LLMConfig(providers={
        "anthropic": ProviderConfig(models=ProviderModelsConfig(standard="claude-x-9")),
    })
    rates, known = cfg.rates_for("anthropic", "claude-x-9")
    assert not known
    assert rates == CONSERVATIVE_RATES
    assert rates.output >= 25.0


def test_self_hosted_model_is_free() -> None:
    cfg = LLMConfig(providers={
        "openllm": ProviderConfig(
            kind="openai_compatible", api_key_required=False,
            models=ProviderModelsConfig(standard="qwen"),
        ),
    })
    rates, known = cfg.rates_for("openllm", "qwen")
    assert known and rates.input == 0 and rates.output == 0


def test_yaml_pricing_overrides_committed_table() -> None:
    cfg = make_llm_config(standard_model="claude-haiku-4-5", pricing_input=0.5, pricing_output=2)
    rates, known = cfg.rates_for("anthropic", "claude-haiku-4-5")
    assert known and (rates.input, rates.output) == (0.5, 2)


def test_committed_example_config_prices_its_models() -> None:
    """Prod loads config/llm.example.yaml (llm.yaml is gitignored): every
    anthropic model it routes must resolve to a real price."""
    from chalybclip.llm.config import load_llm_config

    cfg = load_llm_config(Path(__file__).parents[2] / "config" / "llm.example.yaml")
    models = cfg.providers["anthropic"].models
    for model in (models.standard, models.premium):
        assert model
        _rates, known = cfg.rates_for("anthropic", model)
        assert known, model


# ---- anthropic usage parsing -------------------------------------------


def _message(*, tool: bool, **usage: int) -> SimpleNamespace:
    content = [SimpleNamespace(type="tool_use", input={"answer": "x"})] if tool else [
        SimpleNamespace(type="text", text="oops")
    ]
    return SimpleNamespace(content=content, usage=SimpleNamespace(**usage))


def test_anthropic_parse_reads_cache_tokens() -> None:
    res = _parse_message(_message(
        tool=True, input_tokens=100, output_tokens=20,
        cache_read_input_tokens=5_000, cache_creation_input_tokens=1_000,
    ), model="claude-haiku-4-5")
    assert (res.input_tokens, res.output_tokens) == (100, 20)
    assert (res.cache_read_tokens, res.cache_write_tokens) == (5_000, 1_000)
    assert res.output == {"answer": "x"}


def test_anthropic_missing_tool_use_still_reports_usage() -> None:
    with pytest.raises(RetryableLLMError) as ei:
        _parse_message(_message(
            tool=False, input_tokens=300, output_tokens=40,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        ), model="claude-haiku-4-5")
    assert ei.value.usage is not None
    assert ei.value.usage.input_tokens == 300


# ---- router: every attempt billed --------------------------------------


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "t.db")
    await apply_migrations(d)
    await TenantsRepo(d).create(tenant_id="ten_a", name="A")
    try:
        yield d
    finally:
        await d.close()


class _CacheFake(FakeProvider):
    def queue_raw(self, result: ProviderResult | Exception) -> None:
        self._responses.append(result)


async def test_schema_retry_bills_both_attempts_with_cache_split(db: Database) -> None:
    fake = _CacheFake("anthropic")
    # Attempt 1: schema-invalid output (still billed). Attempt 2: valid.
    fake.queue_raw(ProviderResult(
        output={"wrong": 1}, input_tokens=1_000, output_tokens=100,
        cache_read_tokens=4_000, cache_write_tokens=0, model="claude-haiku-4-5",
    ))
    fake.queue_raw(ProviderResult(
        output={"answer": "ok"}, input_tokens=1_000, output_tokens=200,
        cache_read_tokens=4_000, cache_write_tokens=800, model="claude-haiku-4-5",
    ))
    cfg = make_llm_config(standard_model="claude-haiku-4-5", retry_attempts=3)
    cfg.pricing = {}  # use the committed table
    router = LLMRouter(cfg, api_keys={"anthropic": "k"}, provider_factory=_factory(fake), db=db)
    with bound_tenant("ten_a"):
        out = await router.complete(
            tenant_id="ten_a", purpose="variant_generation", system="s", user="u",
            schema=Answer,
        )
        rows = await LLMCallsRepo(db).list_for_tenant()
    assert out.answer == "ok"
    expected = 2_000 * 1 + 300 * 5 + round(8_000 * 0.1) + round(800 * 1.25)
    assert len(rows) == 1
    assert rows[0].cost_usd_micros == expected
    assert rows[0].input_tokens == 2_000 and rows[0].output_tokens == 300

    ev = (await UsageOutboxRepo(db).get(rows[0].id)).payload
    assert ev["kind"] == "llm.tokens"
    assert ev["cost_usd_micros"] == expected
    assert ev["metadata"]["tokens"] == {
        "input": 2_000, "output": 300, "cache_read": 8_000, "cache_write": 800,
    }
    assert ev["amount"] == 2_000 + 300 + 8_000 + 800


async def test_failed_call_tokens_are_billed(db: Database) -> None:
    """Every attempt schema-invalid → the call fails, but the provider
    billed all three responses — so do we."""
    fake = _CacheFake("anthropic")
    for _ in range(3):
        fake.queue_raw(ProviderResult(
            output={"wrong": 1}, input_tokens=500, output_tokens=50,
            model="claude-haiku-4-5",
        ))
    cfg = make_llm_config(standard_model="claude-haiku-4-5", retry_attempts=3)
    cfg.pricing = {}
    router = LLMRouter(cfg, api_keys={"anthropic": "k"}, provider_factory=_factory(fake), db=db)
    with bound_tenant("ten_a"), pytest.raises(LLMError):
        await router.complete(
            tenant_id="ten_a", purpose="variant_generation", system="s", user="u",
            schema=Answer,
        )
    with bound_tenant("ten_a"):
        rows = await LLMCallsRepo(db).list_for_tenant()
    assert rows[0].status == "error"
    assert rows[0].cost_usd_micros == 1_500 * 1 + 150 * 5
    ev = (await UsageOutboxRepo(db).get(rows[0].id)).payload
    assert ev["cost_usd_micros"] == 2_250
    assert ev["metadata"]["status"] == "error"


async def test_unknown_paid_model_logs_error_and_bills_high(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    fake = _CacheFake("anthropic")
    fake.queue_raw(ProviderResult(
        output={"answer": "ok"}, input_tokens=1_000, output_tokens=1_000, model="claude-new-7",
    ))
    cfg = make_llm_config(standard_model="claude-new-7", retry_attempts=1)
    cfg.pricing = {}
    router = LLMRouter(cfg, api_keys={"anthropic": "k"}, provider_factory=_factory(fake), db=db)
    with caplog.at_level(logging.ERROR, logger="chalybclip.llm.router"), bound_tenant("ten_a"):
        await router.complete(
            tenant_id="ten_a", purpose="variant_generation", system="s", user="u",
            schema=Answer,
        )
        rows = await LLMCallsRepo(db).list_for_tenant()
    assert "llm.price_unknown" in caplog.text
    assert rows[0].cost_usd_micros == round(
        1_000 * CONSERVATIVE_RATES.input + 1_000 * CONSERVATIVE_RATES.output
    )
    assert rows[0].cost_usd_micros > 0
