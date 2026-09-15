"""ComposedPost assembly — the caption / caption_sans_tags split.

`caption` embeds the hashtag line (posted as-is by on_approve and the
dashboard). `caption_sans_tags` is for the growth-engine asset matrix,
which carries hashtags separately and appends them once at post time —
feeding it the full caption shipped every tag block twice.
"""

from __future__ import annotations

from chalybclip.publish.compose import assemble_caption


def test_assemble_caption_orders_blocks() -> None:
    out = assemble_caption(
        hook="EL HOOK", body="cuerpo del post",
        hashtags=["#clips", "#gaming"], handle_suffix="@aldo",
    )
    assert out == "EL HOOK\n\ncuerpo del post\n\n#clips #gaming\n\n@aldo"


def test_sans_tags_variant_drops_only_the_tag_line() -> None:
    with_tags = assemble_caption(
        hook="EL HOOK", body="cuerpo", hashtags=["#clips"], handle_suffix="@aldo",
    )
    sans_tags = assemble_caption(
        hook="EL HOOK", body="cuerpo", hashtags=[], handle_suffix="@aldo",
    )
    assert "#clips" in with_tags
    assert "#clips" not in sans_tags
    assert "EL HOOK" in sans_tags and "@aldo" in sans_tags


def test_growth_content_plus_asset_tags_never_duplicates() -> None:
    """End-to-end shape of the fix: sans-tags caption through the asset
    matrix + caption_with_tags() yields exactly ONE tag block."""
    from chalybclip.publish.assets import build_platform_assets

    sans_tags = assemble_caption(
        hook="HOOK", body="cuerpo", hashtags=[], handle_suffix="",
    )
    asset = build_platform_assets(
        clip_id="c", base_caption=sans_tags, base_hashtags=["#clips"],
        base_title=None, hook="HOOK", platforms=["instagram"],
    )["instagram"]
    posted = asset.caption_with_tags()
    assert posted.count("#clips") == 1
