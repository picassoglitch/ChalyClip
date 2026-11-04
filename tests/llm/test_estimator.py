"""The pipeline estimate prices tokens at the routed standard model's rates."""

from __future__ import annotations

from pathlib import Path

import pytest

from chalybclip.llm import estimator
from chalybclip.llm.config import LLMConfig, load_llm_config

EXAMPLE = Path(__file__).parents[2] / "config" / "llm.example.yaml"


def _example() -> LLMConfig:
    return load_llm_config(EXAMPLE)


def test_committed_config_estimates_at_haiku_5_5() -> None:
    cfg = _example()
    assert estimator.standard_model(cfg) == ("anthropic", "claude-haiku-5-5")
    # 0.75 x $0.50 + 0.25 x $0.10
    assert estimator.blended_usd_per_mtok(cfg) == pytest.approx(0.40)


def test_yaml_rollback_to_haiku_4_5_raises_the_estimate_with_it(tmp_path: Path) -> None:
    rollback = tmp_path / "llm.yaml"
    rollback.write_text(
        EXAMPLE.read_text("utf-8").replace(
            "standard: claude-haiku-5-5", "standard: claude-haiku-4-5"
        ),
        "utf-8",
    )
    cfg = load_llm_config(rollback)
    assert estimator.standard_model(cfg) == ("anthropic", "claude-haiku-4-5")
    # 0.75 x $5 + 0.25 x $1, the pre-swap $4 blend.
    assert estimator.blended_usd_per_mtok(cfg) == pytest.approx(4.0)
    est = estimator.estimate_pipeline_run(duration_seconds=3600, llm_config=cfg)
    assert est.llm_cost_usd_micros == round(est.total_tokens * 4.0)


def test_estimate_llm_cost_uses_the_loaded_config() -> None:
    est = estimator.estimate_pipeline_run(duration_seconds=3600, llm_config=_example())
    assert est.llm_cost_usd_micros == round(est.total_tokens * 0.40)


def test_nothing_routed_reserves_no_llm_spend() -> None:
    assert estimator.blended_usd_per_mtok(LLMConfig()) == 0.0
