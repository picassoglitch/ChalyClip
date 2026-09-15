"""Tests for the persona loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from chalybclip.errors import ChalybClipError, VariantError
from chalybclip.variants import Persona, get_persona, load_personas


def _write(yaml_path: Path, body: str) -> Path:
    yaml_path.write_text(body, encoding="utf-8")
    return yaml_path


def test_load_personas_round_trips(tmp_path: Path) -> None:
    yaml_path = _write(
        tmp_path / "personas.yaml",
        """
personas:
  aldo_villanueva:
    name: "Aldo Villanueva"
    target_languages: [es, en]
    primary_language: es
    voice_prompt: |
      Direct entrepreneur voice.
    routing_tags: [mindset, irl]
  chalyb_academy:
    name: "Chalyb Academy"
    target_languages: [es]
    primary_language: es
    voice_prompt: "Teacher voice"
""",
    )
    personas = load_personas(yaml_path)
    assert set(personas) == {"aldo_villanueva", "chalyb_academy"}
    assert personas["aldo_villanueva"].id == "aldo_villanueva"
    assert personas["aldo_villanueva"].primary_language == "es"
    assert "Direct entrepreneur" in personas["aldo_villanueva"].voice_prompt
    assert personas["chalyb_academy"].routing_tags == []


def test_load_personas_missing_explicit_path_raises(tmp_path: Path) -> None:
    with pytest.raises(ChalybClipError, match="personas config not found"):
        load_personas(tmp_path / "missing.yaml")


def test_load_personas_returns_empty_when_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_personas(None) == {}


def test_load_personas_invalid_yaml_raises(tmp_path: Path) -> None:
    bad = _write(tmp_path / "bad.yaml", "personas: : :\n  bad")
    with pytest.raises(ChalybClipError, match="failed to parse"):
        load_personas(bad)


def test_get_persona_unknown_raises(tmp_path: Path) -> None:
    yaml_path = _write(
        tmp_path / "personas.yaml",
        """
personas:
  aldo_villanueva:
    name: "Aldo"
    target_languages: [es]
    primary_language: es
    voice_prompt: "x"
""",
    )
    with pytest.raises(VariantError, match="unknown persona"):
        get_persona("does_not_exist", path=yaml_path)


def test_get_persona_returns_persona(tmp_path: Path) -> None:
    yaml_path = _write(
        tmp_path / "personas.yaml",
        """
personas:
  aldo_villanueva:
    name: "Aldo"
    target_languages: [es]
    primary_language: es
    voice_prompt: "x"
""",
    )
    p = get_persona("aldo_villanueva", path=yaml_path)
    assert isinstance(p, Persona)
    assert p.id == "aldo_villanueva"
