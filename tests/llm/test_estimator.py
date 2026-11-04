"""The pipeline estimate prices tokens at the routed standard model's rates."""

from __future__ import annotations

from pathlib import Path

from chalybclip.llm import estimator
from chalybclip.llm.config import load_llm_config
from chalybclip.llm.prices import lookup_rates


def test_estimate_model_is_the_committed_standard_model() -> None:
    cfg = load_llm_config(Path(__file__).parents[2] / "config" / "llm.example.yaml")
    assert cfg.providers["anthropic"].models.standard == estimator.ESTIMATE_MODEL


def test_blended_rate_comes_from_prices_py() -> None:
    rates = lookup_rates("anthropic", estimator.ESTIMATE_MODEL)
    assert rates is not None
    expected = 0.75 * rates.output + 0.25 * rates.input
    assert expected == estimator._LLM_BLENDED_USD_PER_MTOK
    assert round(estimator._LLM_BLENDED_USD_PER_MTOK, 4) == 0.40  # Haiku 5.5: $0.10 / $0.50


def test_estimate_llm_cost_uses_the_blended_rate() -> None:
    est = estimator.estimate_pipeline_run(duration_seconds=3600)
    assert est.llm_cost_usd_micros == round(est.total_tokens * 0.40)
