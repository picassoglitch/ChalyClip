"""Caption preset library — schema + lookup helpers (slice D.2)."""

from __future__ import annotations

import pytest

from chalybclip.branding import (
    CaptionShadow,
    CaptionStyle,
    builtin_presets,
    caption_style_or_default,
    preset_choices,
)


def test_all_four_presets_present() -> None:
    """Spec §3.4 ships four presets — every one must be in builtin_presets()."""
    presets = builtin_presets()
    assert set(presets.keys()) == {
        "karaoke_pop",
        "typewriter",
        "bold_block",
        "subtle",
    }


def test_each_preset_id_matches_its_field() -> None:
    """Sanity check that a preset's stored id matches the dict key — caught
    once already during authoring."""
    for key, style in builtin_presets().items():
        assert style.preset == key, f"preset {key} has style.preset={style.preset}"


def test_preset_choices_covers_every_preset() -> None:
    ids = {pid for pid, _ in preset_choices()}
    assert ids == set(builtin_presets().keys())


def test_caption_style_or_default_none_returns_karaoke_pop() -> None:
    out = caption_style_or_default(None)
    assert out.preset == "karaoke_pop"


def test_caption_style_or_default_empty_returns_karaoke_pop() -> None:
    out = caption_style_or_default({})
    assert out.preset == "karaoke_pop"


def test_caption_style_or_default_roundtrips_a_full_dict() -> None:
    original = builtin_presets()["bold_block"]
    out = caption_style_or_default(original.model_dump())
    assert out == original


def test_caption_style_or_default_falls_back_on_malformed() -> None:
    """Partial / malformed stored dict with a known preset id falls back
    to that preset's full defaults — not a hard failure."""
    malformed = {"preset": "typewriter", "bogus_field": 123}
    out = caption_style_or_default(malformed)
    assert out.preset == "typewriter"
    assert out.font_family == "JetBrains Mono"  # typewriter preset's font


def test_caption_style_or_default_unknown_preset_returns_karaoke_pop() -> None:
    out = caption_style_or_default({"preset": "not_a_real_preset"})
    assert out.preset == "karaoke_pop"


def test_caption_style_validates_font_weight_range() -> None:
    """Pydantic constraint: 100 <= font_weight <= 900."""
    with pytest.raises(Exception):  # noqa: B017 - any validation error
        CaptionStyle(font_weight=50)
    with pytest.raises(Exception):  # noqa: B017
        CaptionStyle(font_weight=1500)


def test_shadow_subschema_serializes() -> None:
    """The nested shadow block round-trips through model_dump."""
    style = CaptionStyle(shadow=CaptionShadow(enabled=False, offset_y=10))
    d = style.model_dump()
    assert d["shadow"]["enabled"] is False
    assert d["shadow"]["offset_y"] == 10
    restored = CaptionStyle.model_validate(d)
    assert restored.shadow.enabled is False
