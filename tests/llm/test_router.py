"""LLMRouter tests — provider chain, retries, validation, cost, JSONL log."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pytest
from pydantic import BaseModel

from chalybclip.errors import LLMError
from chalybclip.llm import LLMRouter, VariantBatch
from chalybclip.llm.config import ProviderConfig

from ._fakes import FakeProvider
from ._fixtures import make_llm_config


def _read_call_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def _factory(providers: dict[str, FakeProvider]):
    def _build(name: str, _config: ProviderConfig, _api_key: str) -> FakeProvider | None:
        return providers.get(name)

    return _build


class TinySchema(BaseModel):
    answer: str


def test_unknown_purpose_raises() -> None:
    config = make_llm_config()
    router = LLMRouter(config, api_keys={"anthropic": "x"})
    with pytest.raises(LLMError, match="unknown routing purpose"):
        asyncio.run(
            router.complete(
                tenant_id="default",
                purpose="not_a_real_purpose",
                system="s",
                user="u",
                schema=TinySchema,
            )
        )


def test_returns_validated_pydantic_model(tmp_path: Path) -> None:
    fake = FakeProvider("anthropic")
    fake.queue_success({"answer": "42"}, input_tokens=10, output_tokens=5)

    log_path = tmp_path / "llm_calls.jsonl"
    config = make_llm_config()
    router = LLMRouter(
        config,
        api_keys={"anthropic": "key"},
        provider_factory=_factory({"anthropic": fake}),
        call_log_path=log_path,
    )

    result = asyncio.run(
        router.complete(
            tenant_id="alice",
            purpose="variant_generation",
            system="be helpful",
            user="what is the answer",
            schema=TinySchema,
        )
    )
    assert isinstance(result, TinySchema)
    assert result.answer == "42"
    assert len(fake.calls) == 1
    assert fake.calls[0]["model"] == "claude-haiku-4-5-20251001"
    assert fake.calls[0]["tenant_id"] == "alice"


def test_writes_one_jsonl_row_per_call(tmp_path: Path) -> None:
    fake = FakeProvider("anthropic")
    fake.queue_success({"answer": "x"}, input_tokens=1000, output_tokens=500)

    log_path = tmp_path / "llm_calls.jsonl"
    config = make_llm_config(pricing_input=0.80, pricing_output=4.00)
    router = LLMRouter(
        config,
        api_keys={"anthropic": "key"},
        provider_factory=_factory({"anthropic": fake}),
        call_log_path=log_path,
    )
    asyncio.run(
        router.complete(
            tenant_id="alice",
            purpose="variant_generation",
            system="s",
            user="u",
            schema=TinySchema,
        )
    )

    rows = _read_call_log(log_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["tenant_id"] == "alice"
    assert row["purpose"] == "variant_generation"
    assert row["provider"] == "anthropic"
    assert row["status"] == "ok"
    assert row["input_tokens"] == 1000
    assert row["output_tokens"] == 500
    # cost = 1000 * 0.80 + 500 * 4.00 = 2800 micros
    assert row["cost_usd_micros"] == 2800
    assert row["attempts"] == 1


def test_retries_on_retryable_error_then_succeeds(tmp_path: Path) -> None:
    fake = FakeProvider("anthropic")
    fake.queue_retryable("503")
    fake.queue_retryable("503")
    fake.queue_success({"answer": "ok"})

    config = make_llm_config(retry_attempts=3, initial_backoff_s=0.0)
    router = LLMRouter(
        config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
        call_log_path=tmp_path / "llm_calls.jsonl",
    )
    result = asyncio.run(
        router.complete(
            tenant_id="t",
            purpose="variant_generation",
            system="s",
            user="u",
            schema=TinySchema,
        )
    )
    assert result.answer == "ok"
    assert len(fake.calls) == 3

    rows = _read_call_log(tmp_path / "llm_calls.jsonl")
    assert rows[-1]["attempts"] == 3
    assert rows[-1]["status"] == "ok"


def test_retries_exhaust_then_fall_through_to_fallback(tmp_path: Path) -> None:
    primary = FakeProvider("anthropic")
    for _ in range(3):
        primary.queue_retryable("503")
    fallback = FakeProvider("openai")
    fallback.queue_success({"answer": "from-fallback"})

    config = make_llm_config(
        primary="anthropic", fallbacks=["openai"], retry_attempts=3, initial_backoff_s=0.0
    )
    router = LLMRouter(
        config,
        api_keys={"anthropic": "a", "openai": "o"},
        provider_factory=_factory({"anthropic": primary, "openai": fallback}),
        call_log_path=tmp_path / "llm_calls.jsonl",
    )
    result = asyncio.run(
        router.complete(
            tenant_id="t",
            purpose="variant_generation",
            system="s",
            user="u",
            schema=TinySchema,
        )
    )
    assert result.answer == "from-fallback"
    assert len(primary.calls) == 3
    assert len(fallback.calls) == 1

    rows = _read_call_log(tmp_path / "llm_calls.jsonl")
    # one error row for anthropic + one ok row for openai
    assert len(rows) == 2
    assert rows[0]["provider"] == "anthropic" and rows[0]["status"] == "error"
    assert rows[1]["provider"] == "openai" and rows[1]["status"] == "ok"


def test_all_providers_fail_raises(tmp_path: Path) -> None:
    primary = FakeProvider("anthropic")
    for _ in range(3):
        primary.queue_retryable("503")
    fallback = FakeProvider("openai")
    for _ in range(3):
        fallback.queue_retryable("503")

    config = make_llm_config(
        primary="anthropic", fallbacks=["openai"], retry_attempts=3, initial_backoff_s=0.0
    )
    router = LLMRouter(
        config,
        api_keys={"anthropic": "a", "openai": "o"},
        provider_factory=_factory({"anthropic": primary, "openai": fallback}),
        call_log_path=tmp_path / "llm_calls.jsonl",
    )
    with pytest.raises(LLMError, match="all providers failed"):
        asyncio.run(
            router.complete(
                tenant_id="t",
                purpose="variant_generation",
                system="s",
                user="u",
                schema=TinySchema,
            )
        )


def test_fatal_error_is_not_retried(tmp_path: Path) -> None:
    fake = FakeProvider("anthropic")
    fake.queue_fatal("401 unauthorized")

    config = make_llm_config(retry_attempts=3, initial_backoff_s=0.0)
    router = LLMRouter(
        config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
        call_log_path=tmp_path / "llm_calls.jsonl",
    )
    with pytest.raises(LLMError):
        asyncio.run(
            router.complete(
                tenant_id="t",
                purpose="variant_generation",
                system="s",
                user="u",
                schema=TinySchema,
            )
        )
    assert len(fake.calls) == 1  # no retries


def test_billing_error_trips_lockout_and_skips_provider(tmp_path: Path) -> None:
    """A billing 400 (credits drained / usage cap) is durable, not transient:
    the first failure must trip a per-provider lockout so subsequent calls
    fail fast WITHOUT hitting the API — one drained account produced 1,868
    identical failed calls in two weeks before this breaker existed."""
    from chalybclip.llm.router import reset_billing_lockouts

    reset_billing_lockouts()
    try:
        fake = FakeProvider("anthropic")
        fake.queue_fatal(
            "anthropic 400: Your credit balance is too low to access the "
            "Anthropic API. Please go to Plans & Billing."
        )

        config = make_llm_config(retry_attempts=3, initial_backoff_s=0.0)
        router = LLMRouter(
            config,
            api_keys={"anthropic": "k"},
            provider_factory=_factory({"anthropic": fake}),
            call_log_path=tmp_path / "llm_calls.jsonl",
        )

        def _call() -> None:
            asyncio.run(
                router.complete(
                    tenant_id="t", purpose="variant_generation",
                    system="s", user="u", schema=TinySchema,
                )
            )

        with pytest.raises(LLMError):
            _call()
        assert len(fake.calls) == 1

        # Second call: provider is locked out — no new API attempt at all.
        with pytest.raises(LLMError, match="billing lockout"):
            _call()
        assert len(fake.calls) == 1
    finally:
        reset_billing_lockouts()


def test_billing_lockout_expires_with_the_clock(tmp_path: Path) -> None:
    import datetime as _dt

    from chalybclip.llm.router import reset_billing_lockouts

    reset_billing_lockouts()
    try:
        now = _dt.datetime(2026, 7, 12, 12, 0, tzinfo=_dt.UTC)

        def clock() -> _dt.datetime:
            return now

        fake = FakeProvider("anthropic")
        fake.queue_fatal("anthropic 400: You have reached your specified API usage limits.")
        fake.queue_success({"answer": "back"})

        config = make_llm_config(retry_attempts=1, initial_backoff_s=0.0)
        router = LLMRouter(
            config,
            api_keys={"anthropic": "k"},
            provider_factory=_factory({"anthropic": fake}),
            call_log_path=tmp_path / "llm_calls.jsonl",
            clock=clock,
        )

        def _call() -> TinySchema:
            return asyncio.run(
                router.complete(
                    tenant_id="t", purpose="variant_generation",
                    system="s", user="u", schema=TinySchema,
                )
            )

        with pytest.raises(LLMError):
            _call()
        # Past the lockout window the breaker clears and calls flow again.
        now = now + _dt.timedelta(hours=2)
        assert _call().answer == "back"
        assert len(fake.calls) == 2
    finally:
        reset_billing_lockouts()


def test_validation_error_retries_then_fails(tmp_path: Path) -> None:
    fake = FakeProvider("anthropic")
    # All three attempts return malformed output (missing required field).
    for _ in range(3):
        fake.queue_success({"wrong_field": "nope"})

    config = make_llm_config(retry_attempts=3, initial_backoff_s=0.0)
    router = LLMRouter(
        config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
        call_log_path=tmp_path / "llm_calls.jsonl",
    )
    with pytest.raises(LLMError, match="all providers failed"):
        asyncio.run(
            router.complete(
                tenant_id="t",
                purpose="variant_generation",
                system="s",
                user="u",
                schema=TinySchema,
            )
        )
    assert len(fake.calls) == 3


def test_quality_override_picks_premium_model() -> None:
    fake = FakeProvider("anthropic")
    fake.queue_success({"answer": "x"})

    config = make_llm_config()
    router = LLMRouter(
        config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
    )
    asyncio.run(
        router.complete(
            tenant_id="t",
            purpose="variant_generation",
            system="s",
            user="u",
            schema=TinySchema,
            quality="premium",
        )
    )
    assert fake.calls[0]["model"] == "claude-opus-4-7"


def test_missing_api_key_skips_provider(tmp_path: Path) -> None:
    primary = FakeProvider("anthropic")
    fallback = FakeProvider("openai")
    fallback.queue_success({"answer": "from-fallback"})

    config = make_llm_config(primary="anthropic", fallbacks=["openai"])
    router = LLMRouter(
        config,
        api_keys={"anthropic": "", "openai": "o"},
        provider_factory=_factory({"anthropic": primary, "openai": fallback}),
        call_log_path=tmp_path / "llm_calls.jsonl",
    )
    result = asyncio.run(
        router.complete(
            tenant_id="t",
            purpose="variant_generation",
            system="s",
            user="u",
            schema=TinySchema,
        )
    )
    assert result.answer == "from-fallback"
    assert primary.calls == []  # never invoked


def test_variant_batch_schema_round_trips(tmp_path: Path) -> None:
    """Real Phase 0 schema should accept a batch the LLM might produce."""
    fake = FakeProvider("anthropic")
    fake.queue_success(
        {
            "variants": [
                {
                    "id": "v_1",
                    "language": "es",
                    "caption": "Captura este momento.",
                    "title_card_text": "MOMENTO",
                    "hashtags": ["clip", "streamer"],
                },
                {
                    "id": "v_2",
                    "language": "es",
                    "caption": "No te lo pierdas.",
                    "title_card_text": "",
                    "hashtags": [],
                },
            ]
        }
    )
    config = make_llm_config()
    router = LLMRouter(
        config,
        api_keys={"anthropic": "k"},
        provider_factory=_factory({"anthropic": fake}),
        call_log_path=tmp_path / "llm_calls.jsonl",
    )
    batch = asyncio.run(
        router.complete(
            tenant_id="t",
            purpose="variant_generation",
            system="s",
            user="u",
            schema=VariantBatch,
        )
    )
    assert len(batch.variants) == 2
    assert batch.variants[0].id == "v_1"
    assert batch.variants[0].hashtags == ["clip", "streamer"]
