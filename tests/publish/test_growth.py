"""Growth Score tests — deterministic scorer (no LLM)."""

from __future__ import annotations

from chalybclip.score.growth import GrowthInput, fallback_card, score_clip


def test_fallback_uses_publishability() -> None:
    inp = GrowthInput(
        clip_id="c1", duration_s=12, caption="hi", platforms=["tiktok", "x"],
        publishability_score=82,
    )
    card = fallback_card(inp)
    assert card.overall_score == 82
    assert {p.platform for p in card.platforms} == {"tiktok", "twitter"}
    assert all(p.verdict == "publish" for p in card.platforms)
    assert all(p.label == "Very Good" for p in card.platforms)


def test_fallback_low_score_archives() -> None:
    inp = GrowthInput(clip_id="c1", duration_s=12, caption="hi", platforms=["tiktok"],
                      publishability_score=20)
    card = fallback_card(inp)
    assert card.decision == "archive"
    assert card.platforms[0].verdict == "skip"


def test_score_clip_publishes_when_fits_and_above_floor() -> None:
    inp = GrowthInput(
        clip_id="c1", duration_s=15, caption="hi", platforms=["tiktok", "x"],
        publishability_score=82,
    )
    card = score_clip(inp)
    assert card.overall_score == 82
    assert card.decision == "publish_all"
    assert all(p.verdict == "publish" for p in card.platforms)


def test_score_clip_skips_platform_over_duration_ceiling() -> None:
    # Bluesky's short-form ceiling is 60s; a 120s clip must be skipped there.
    inp = GrowthInput(
        clip_id="c1", duration_s=120, caption="hi", platforms=["bluesky"],
        publishability_score=80,
    )
    card = score_clip(inp)
    assert card.platforms[0].verdict == "skip"
    assert "ceiling" in card.platforms[0].reason
    # Publishable but fits no connected platform → archive.
    assert card.decision == "archive"


def test_score_clip_normalizes_platform_aliases() -> None:
    inp = GrowthInput(clip_id="c1", duration_s=12, caption="hi", platforms=["x"],
                      publishability_score=70)
    card = score_clip(inp)
    assert card.platforms[0].platform == "twitter"  # alias normalized


def test_score_clip_derives_content_tags_for_fatigue() -> None:
    inp = GrowthInput(
        clip_id="c1", duration_s=12, caption="increíble jugada", platforms=["tiktok"],
        publishability_score=70, stream_title="Valorant ranked clutch",
    )
    card = score_clip(inp)
    # Deterministic, lowercase, drops short/stop words; carries the theme.
    assert "valorant" in card.content_tags
    assert all(t == t.lower() and "#" not in t for t in card.content_tags)


def test_score_clip_empty_platforms_falls_back() -> None:
    inp = GrowthInput(clip_id="c1", duration_s=12, caption="hi", platforms=[],
                      publishability_score=55)
    card = score_clip(inp)
    assert card.platforms == []
    assert card.decision in {"archive", "publish_select", "skip"}
