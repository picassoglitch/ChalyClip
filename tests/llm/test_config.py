"""Tests for `chalybclip/llm/config.py` (LLM YAML loader)."""

from __future__ import annotations

from pathlib import Path

import pytest

from chalybclip.errors import ChalybClipError
from chalybclip.llm import LLMConfig, load_llm_config


def test_load_llm_config_returns_defaults_when_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # no config/llm.yaml or example here
    config = load_llm_config(None)
    assert isinstance(config, LLMConfig)
    assert config.providers == {}


def test_load_llm_config_reads_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "llm.yaml"
    yaml_path.write_text(
        """
providers:
  anthropic:
    api_key_env: ANTHROPIC_API_KEY
    base_url: https://api.anthropic.com
    models:
      standard: claude-haiku-4-5-20251001
      premium: claude-opus-4-7
    timeout_s: 30

routing:
  variant_generation:
    default_quality: standard
    primary: anthropic
    fallbacks: [openai]

retry:
  max_attempts: 3
  initial_backoff_s: 1.0
  backoff_multiplier: 4.0
  retryable_status_codes: [429, 500, 502]

pricing:
  anthropic:
    claude-haiku-4-5-20251001:
      input_per_mtok_usd: 0.80
      output_per_mtok_usd: 4.00
""",
        encoding="utf-8",
    )
    config = load_llm_config(yaml_path)
    assert "anthropic" in config.providers
    assert config.providers["anthropic"].models.standard == "claude-haiku-4-5-20251001"
    assert config.routing["variant_generation"].fallbacks == ["openai"]
    assert config.retry.max_attempts == 3
    pricing = config.pricing_for("anthropic", "claude-haiku-4-5-20251001")
    assert pricing.input_per_mtok_usd == 0.80


def test_model_for_resolves_quality(tmp_path: Path) -> None:
    yaml_path = tmp_path / "llm.yaml"
    yaml_path.write_text(
        """
providers:
  anthropic:
    api_key_env: ANTHROPIC_API_KEY
    models:
      standard: haiku
      premium: opus
""",
        encoding="utf-8",
    )
    config = load_llm_config(yaml_path)
    assert config.model_for("anthropic", "standard") == "haiku"
    assert config.model_for("anthropic", "premium") == "opus"


def test_pricing_for_unknown_returns_zero() -> None:
    config = LLMConfig()
    pricing = config.pricing_for("ghost-provider", "ghost-model")
    assert pricing.input_per_mtok_usd == 0.0
    assert pricing.output_per_mtok_usd == 0.0


def test_load_llm_config_missing_explicit_path_raises(tmp_path: Path) -> None:
    with pytest.raises(ChalybClipError, match="llm config not found"):
        load_llm_config(tmp_path / "missing.yaml")


def test_load_llm_config_invalid_yaml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("providers: : :\n  bad", encoding="utf-8")
    with pytest.raises(ChalybClipError, match="failed to parse"):
        load_llm_config(bad)
