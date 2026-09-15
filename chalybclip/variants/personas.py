"""Persona loader — `config/personas.yaml` parsed into `Persona` models.

Personas have an id (the YAML key) plus name, voice_prompt, target/primary
languages, and routing tags. The `voice_prompt` is the system-prompt
fragment fed to the LLM by the variant generator.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from chalybclip.errors import ChalybClipError, VariantError


class Persona(BaseModel):
    """One brand voice — Aldo Villanueva, Chalyb Academy, AARA, etc."""

    id: str
    name: str
    target_languages: list[str] = Field(default_factory=list)
    primary_language: str
    voice_prompt: str
    routing_tags: list[str] = Field(default_factory=list)


_DEFAULT_PATH = Path("config/personas.yaml")
_EXAMPLE_PATH = Path("config/personas.example.yaml")


def load_personas(path: Path | None = None) -> dict[str, Persona]:
    """Load personas from `config/personas.yaml` (falls back to the example)."""
    candidates: list[Path] = [Path(path)] if path is not None else [_DEFAULT_PATH, _EXAMPLE_PATH]
    for candidate in candidates:
        if candidate.exists():
            try:
                with candidate.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ChalybClipError(f"failed to parse {candidate}: {e}") from e
            raw = data.get("personas", {})
            return {pid: Persona.model_validate({**p, "id": pid}) for pid, p in raw.items()}
    if path is not None:
        raise ChalybClipError(f"personas config not found: {path}")
    return {}


def get_persona(persona_id: str, *, path: Path | None = None) -> Persona:
    """Look up one persona by id; raises `VariantError` if unknown."""
    personas = load_personas(path)
    if persona_id not in personas:
        known = ", ".join(sorted(personas)) or "(none)"
        raise VariantError(f"unknown persona {persona_id!r}; known: {known}")
    return personas[persona_id]


@lru_cache(maxsize=1)
def get_personas_cached() -> dict[str, Persona]:
    """Cached default loader."""
    return load_personas()
