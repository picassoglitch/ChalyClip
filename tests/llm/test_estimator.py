"""The pipeline estimate prices tokens at the routed standard model's rates."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml

from chalybclip.errors import ChalybClipError
from chalybclip.llm import estimator
from chalybclip.llm.config import LLMConfig, load_llm_config

EXAMPLE = Path(__file__).parents[2] / "config" / "llm.example.yaml"


def _example() -> LLMConfig:
    return load_llm_config(EXAMPLE)


def test_committed_config_estimates_at_haiku_5_5() -> None:
    cfg = _example()
    assert estimator.standard_models(cfg) == [("anthropic", "claude-haiku-5-5")]
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
    assert estimator.standard_models(cfg) == [("anthropic", "claude-haiku-4-5")]
    # 0.75 x $5 + 0.25 x $1, the pre-swap $4 blend.
    assert estimator.blended_usd_per_mtok(cfg) == pytest.approx(4.0)
    est = estimator.estimate_pipeline_run(duration_seconds=3600, llm_config=cfg)
    assert est.llm_cost_usd_micros == round(est.total_tokens * 4.0)


def test_estimate_llm_cost_uses_the_loaded_config() -> None:
    est = estimator.estimate_pipeline_run(duration_seconds=3600, llm_config=_example())
    assert est.llm_cost_usd_micros == round(est.total_tokens * 0.40)


OPUS_5_5_BLEND = 0.75 * 20.0 + 0.25 * 4.0  # $16/MTok, the premium fallback


def _bad_yaml() -> LLMConfig:
    raise ChalybClipError("failed to parse config/llm.yaml")


def _invalid_yaml() -> LLMConfig:
    return LLMConfig.model_validate({"retry": {"max_attempts": 0}})


def _unreadable() -> LLMConfig:
    raise PermissionError("config/llm.yaml")


@pytest.mark.parametrize("loader", [_bad_yaml, _invalid_yaml, _unreadable])
def test_unreadable_config_reserves_at_the_premium_rate_and_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, loader: Any
) -> None:
    monkeypatch.setattr(estimator, "get_llm_config", loader)
    with caplog.at_level(logging.ERROR, logger="chalybclip.llm.estimator"):
        assert estimator.blended_usd_per_mtok() == pytest.approx(OPUS_5_5_BLEND)
    record = next(r for r in caplog.records if "llm.estimate_config_unreadable" in r.message)
    assert record.exc_info is not None


def test_nothing_routed_reserves_at_the_premium_rate_not_zero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="chalybclip.llm.estimator"):
        assert estimator.blended_usd_per_mtok(LLMConfig()) == pytest.approx(OPUS_5_5_BLEND)
    assert any("llm.estimate_nothing_routed" in r.message for r in caplog.records)


def test_unpriced_standard_model_logs_price_unknown(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    yaml_path = tmp_path / "llm.yaml"
    yaml_path.write_text(
        EXAMPLE.read_text("utf-8").replace("standard: claude-haiku-5-5", "standard: claude-x-9"),
        "utf-8",
    )
    with caplog.at_level(logging.ERROR, logger="chalybclip.llm.estimator"):
        estimator.blended_usd_per_mtok(load_llm_config(yaml_path))
    assert any("llm.price_unknown" in r.message for r in caplog.records)


def test_all_free_routing_reserves_nothing_for_llm() -> None:
    data = yaml.safe_load(EXAMPLE.read_text("utf-8"))
    for name, rule in data["routing"].items():
        if rule["default_quality"] == "standard":
            rule["primary"] = "openllm_vision" if "vision" in name else "openllm"
    cfg = LLMConfig.model_validate(data)
    assert {p for p, _ in estimator.standard_models(cfg)} == {"openllm", "openllm_vision"}
    assert estimator.blended_usd_per_mtok(cfg) == 0.0


def test_one_paid_standard_purpose_sets_the_rate() -> None:
    data = yaml.safe_load(EXAMPLE.read_text("utf-8"))
    for name, rule in data["routing"].items():
        if rule["default_quality"] == "standard" and name != "hook_generation":
            rule["primary"] = "openllm"
    cfg = LLMConfig.model_validate(data)
    assert estimator.blended_usd_per_mtok(cfg) == pytest.approx(0.40)  # hooks on Haiku 5.5
