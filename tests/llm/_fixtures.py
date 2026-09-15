"""Helpers for building LLMConfig in tests."""

from __future__ import annotations

from chalybclip.llm import LLMConfig
from chalybclip.llm.config import (
    ModelPricing,
    ProviderConfig,
    ProviderModelsConfig,
    RetryConfig,
    RoutingRule,
)


def make_llm_config(
    *,
    primary: str = "anthropic",
    fallbacks: list[str] | None = None,
    standard_model: str = "claude-haiku-4-5-20251001",
    premium_model: str = "claude-opus-4-7",
    purpose: str = "variant_generation",
    retry_attempts: int = 3,
    initial_backoff_s: float = 0.0,  # tests should not actually sleep
    pricing_input: float = 0.80,
    pricing_output: float = 4.00,
) -> LLMConfig:
    """Build a minimal LLMConfig with one routing rule and a couple of providers."""
    providers = {
        "anthropic": ProviderConfig(
            api_key_env="ANTHROPIC_API_KEY",
            models=ProviderModelsConfig(standard=standard_model, premium=premium_model),
        ),
        "openai": ProviderConfig(
            api_key_env="OPENAI_API_KEY",
            models=ProviderModelsConfig(standard="gpt-4o-mini", premium="gpt-4o"),
        ),
    }
    return LLMConfig(
        providers=providers,
        routing={
            purpose: RoutingRule(
                default_quality="standard",
                primary=primary,
                fallbacks=fallbacks or [],
            ),
        },
        retry=RetryConfig(
            max_attempts=retry_attempts,
            initial_backoff_s=initial_backoff_s,
            backoff_multiplier=2.0,
        ),
        pricing={
            "anthropic": {
                standard_model: ModelPricing(
                    input_per_mtok_usd=pricing_input,
                    output_per_mtok_usd=pricing_output,
                ),
            },
        },
    )
