"""Tests for the YAML config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from chalybclip.config import ChalybClipConfig, load_config
from chalybclip.errors import ChalybClipError


def test_load_config_returns_defaults_when_no_file(tmp_path: Path) -> None:
    config = load_config(None)
    assert isinstance(config, ChalybClipConfig)


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "chalybclip.yaml"
    yaml_path.write_text(
        """
detection:
  voice:
    enabled: true
    weight: 0.7
    fuzzy_distance: 1
    phrases:
      es: ["clipéalo"]
      en: ["clip this"]
  merge_window_s: 45
""",
        encoding="utf-8",
    )
    config = load_config(yaml_path)
    assert config.detection.voice.weight == 0.7
    assert config.detection.voice.fuzzy_distance == 1
    assert config.detection.voice.phrases == {"es": ["clipéalo"], "en": ["clip this"]}
    assert config.detection.merge_window_s == 45.0


def test_load_config_explicit_path_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ChalybClipError, match="config file not found"):
        load_config(tmp_path / "does_not_exist.yaml")


def test_load_config_tolerates_extra_sections(tmp_path: Path) -> None:
    yaml_path = tmp_path / "chalybclip.yaml"
    yaml_path.write_text(
        """
detection:
  voice:
    enabled: true
clip:
  pre_roll_s: 30
  post_roll_s: 15
variants:
  per_persona: 5
""",
        encoding="utf-8",
    )
    config = load_config(yaml_path)
    assert config.detection.voice.enabled is True


def test_load_config_invalid_yaml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("detection: : :\n  invalid", encoding="utf-8")
    with pytest.raises(ChalybClipError, match="failed to parse"):
        load_config(bad)
