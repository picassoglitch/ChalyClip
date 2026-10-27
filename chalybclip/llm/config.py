"""Loader + Pydantic models for `config/llm.yaml`.

Kept separate from `chalybclip.config` because LLM config is its own YAML file
(API keys come via environment, model/route choices via this YAML).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from chalybclip.errors import ChalybClipError

from .prices import CONSERVATIVE_RATES, TokenRates, lookup_rates

Quality = Literal["standard", "premium"]


class ProviderModelsConfig(BaseModel):
    """Provider's per-quality model choices."""

    model_config = ConfigDict(extra="allow")

    standard: str
    premium: str | None = None


class RateLimitsConfig(BaseModel):
    """Best-effort rate limits — Phase 0 doesn't enforce these yet."""

    model_config = ConfigDict(extra="allow")

    requests_per_minute: int = 0
    input_tokens_per_minute: int = 0
    output_tokens_per_minute: int = 0


ProviderKind = Literal["anthropic", "openai_compatible"]


class ProviderConfig(BaseModel):
    """One LLM provider entry.

    `kind` picks the wire protocol: "anthropic" (the vendor SDK) or
    "openai_compatible" (any `/chat/completions` runtime — Ollama, vLLM,
    LM Studio, llama.cpp, OpenRouter). Self-hosted runtimes need no API
    key: set `api_key_required: false` and leave the env var unset."""

    kind: ProviderKind = "anthropic"
    api_key_env: str = ""
    api_key_required: bool = True
    base_url: str = ""
    # When set and the env var is non-empty, it overrides `base_url` — so
    # prod can point at a hosted/tunneled runtime without editing YAML.
    base_url_env: str = ""
    models: ProviderModelsConfig
    rate_limits: RateLimitsConfig = Field(default_factory=RateLimitsConfig)
    timeout_s: float = Field(default=30.0, gt=0.0)
    # openai_compatible only: send response_format={"type":"json_object"}.
    # Disable for runtimes that reject the parameter.
    json_mode: bool = True


class RoutingRule(BaseModel):
    """Per-purpose routing decision (which provider, fallbacks, default quality)."""

    default_quality: Quality = "standard"
    primary: str
    fallbacks: list[str] = Field(default_factory=list)


class RetryConfig(BaseModel):
    max_attempts: int = Field(default=3, ge=1)
    initial_backoff_s: float = Field(default=1.0, ge=0.0)
    backoff_multiplier: float = Field(default=4.0, ge=1.0)
    retryable_status_codes: list[int] = Field(default_factory=lambda: [429, 500, 502, 503, 504])


class CircuitBreakerConfig(BaseModel):
    """Reserved for Phase 1+. Not enforced in Phase 0."""

    error_rate_threshold: float = 0.20
    window_s: float = 300.0
    recovery_s: float = 300.0


class ModelPricing(BaseModel):
    """USD per 1 M tokens. Used to compute `cost_usd_micros`.

    Cache rates are optional: unset, they derive from the input rate with
    Anthropic's multipliers (read 0.1x, 5-minute write 1.25x)."""

    input_per_mtok_usd: float = 0.0
    output_per_mtok_usd: float = 0.0
    cache_read_per_mtok_usd: float | None = None
    cache_write_per_mtok_usd: float | None = None


class LLMConfig(BaseModel):
    """Root model for `config/llm.yaml`."""

    model_config = ConfigDict(extra="allow")

    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    routing: dict[str, RoutingRule] = Field(default_factory=dict)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    pricing: dict[str, dict[str, ModelPricing]] = Field(default_factory=dict)

    def model_for(self, provider: str, quality: Quality) -> str:
        """Resolve `(provider, quality)` to a concrete model name."""
        cfg = self.providers.get(provider)
        if cfg is None:
            raise ChalybClipError(f"unknown provider: {provider}")
        if quality == "premium" and cfg.models.premium:
            return cfg.models.premium
        return cfg.models.standard

    def pricing_for(self, provider: str, model: str) -> ModelPricing:
        """Lookup pricing or fall back to zero (so cost stays computable)."""
        return self.pricing.get(provider, {}).get(model, ModelPricing())

    def rates_for(self, provider: str, model: str) -> tuple[TokenRates, bool]:
        """Resolve the rates a call is billed at → `(rates, known)`.

        Order: this config's `pricing:` entry (explicit override), then the
        committed table in `prices.py`. A model neither knows is priced at
        the conservative (highest) rate when its provider is a paid API
        (`kind: anthropic`) and `known=False` so the caller can log it; a
        self-hosted openai_compatible runtime genuinely costs nothing.
        """
        override = self.pricing.get(provider, {}).get(model)
        if override is not None:
            return (
                TokenRates(
                    input=override.input_per_mtok_usd,
                    output=override.output_per_mtok_usd,
                    cache_read=override.cache_read_per_mtok_usd,
                    cache_write=override.cache_write_per_mtok_usd,
                ),
                True,
            )
        committed = lookup_rates(provider, model)
        if committed is not None:
            return committed, True
        cfg = self.providers.get(provider)
        if (cfg is not None and cfg.kind == "anthropic") or provider == "anthropic":
            return CONSERVATIVE_RATES, False
        return TokenRates(input=0.0, output=0.0), True


_DEFAULT_PATH = Path("config/llm.yaml")
_EXAMPLE_PATH = Path("config/llm.example.yaml")


def load_llm_config(path: Path | None = None) -> LLMConfig:
    """Load `config/llm.yaml`, falling back to the example file or defaults."""
    candidates: list[Path] = [Path(path)] if path is not None else [_DEFAULT_PATH, _EXAMPLE_PATH]
    for candidate in candidates:
        if candidate.exists():
            try:
                with candidate.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ChalybClipError(f"failed to parse {candidate}: {e}") from e
            return LLMConfig.model_validate(data)
    if path is not None:
        raise ChalybClipError(f"llm config not found: {path}")
    return LLMConfig()


@lru_cache(maxsize=1)
def get_llm_config() -> LLMConfig:
    """Cached default loader."""
    return load_llm_config()
