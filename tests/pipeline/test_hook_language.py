"""Hooks are written in the video's language, not the literal 'auto'."""

from __future__ import annotations

from chalybclip.pipeline import _resolve_hook_language


def test_auto_falls_back_to_detected_language() -> None:
    assert _resolve_hook_language(requested="auto", detected="es", persona_language="en") == "es"


def test_explicit_language_wins() -> None:
    assert _resolve_hook_language(requested="en", detected="es", persona_language="es") == "en"


def test_persona_language_when_nothing_detected() -> None:
    assert _resolve_hook_language(requested=None, detected=None, persona_language="pt") == "pt"
    assert _resolve_hook_language(requested="auto", detected="", persona_language=None) == "es"
