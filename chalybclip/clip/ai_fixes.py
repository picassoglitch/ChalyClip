"""Apply AI Fixes — slice I.3.

Non-destructive auto-improvements applied to a clip's `overlay_config`
when the operator clicks "Apply AI fixes" in the editor. Each fix is
SAFE — it only adds / corrects fields that are missing or actively
breaking the publish. We never:

  - rewrite hook text the operator typed
  - change the chosen Clip Style preset
  - turn off something the operator explicitly turned on
  - touch per-clip media (no auto-trim — that's G.4's job)

Fixes the engine applies:

  - reposition captions out of the platform's caption danger zone
  - enable banner when the URL is set but the toggle is off
  - enable safe-zones preview when collisions exist
  - set caption lead_ms to the 120ms recommended default when 0/unset
  - enable top hook when the clip has a hook title in the form
  - pick the recommended banner_variant for the current clip_style
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chalybclip.branding.platform_zones import (
    OverlayRect,
    detect_collisions,
)


@dataclass(frozen=True)
class FixOutcome:
    """One change the engine made, surfaced to the dashboard so the
    operator sees what got fixed."""

    field: str           # dotted path: "banner.enabled", "captions.position"
    before: object
    after: object
    why: str             # short human reason


@dataclass(frozen=True)
class AIFixesResult:
    """Result of running the fixes engine — the new config + the diff."""

    new_overlay_config: dict[str, object]
    fixes: list[FixOutcome]


# Sources where a Kick/stream-style bottom banner (repost-page URL, LIVE
# badge, etc.) actually makes sense. Uploaded files and unknown sources get
# no banner fixes — the banner would just paste an irrelevant handle.
_STREAM_SOURCES: frozenset[str] = frozenset({"kick", "twitch", "youtube"})


def apply_ai_fixes(
    *,
    overlay_config: dict[str, object] | None,
    safe_zone_platform: str = "tiktok",
    brand_kit_url: str | None = None,
    source_platform: str | None = None,
) -> AIFixesResult:
    """Apply non-destructive fixes to the overlay config. Returns the
    updated dict + a list of what was changed (for the dashboard's
    "Here's what AI fixed" feedback).

    Slice N.1 — added more impactful fixes so the post-fix retention
    score moves up noticeably:
      - enable captions if disabled (+ ~12 score from caption-pacing
        bonus when speaking_intensity falls in the sweet spot)
      - fill banner.url from the operator's brand_kit_url (passed by
        caller) when the banner has no URL yet
    """
    out: dict[str, Any] = dict(overlay_config or {})
    banner: dict[str, Any] = dict(out.get("banner") or {})
    captions: dict[str, Any] = dict(out.get("captions") or {})
    top_hook: dict[str, Any] = dict(out.get("top_hook") or {})
    out["banner"] = banner
    out["captions"] = captions
    out["top_hook"] = top_hook

    fixes: list[FixOutcome] = []

    # Banner fixes (Kick repost-page URL / LIVE badge / variant) only make
    # sense when the clip came from a stream. For uploads / unknown sources
    # we skip them — pasting a brand-kit handle onto a random upload is noise.
    is_stream_source = (source_platform or "").strip().lower() in _STREAM_SOURCES

    # ---- Fix 1a: enable captions if disabled -------------------
    # Captions disabled = the publishability scorer dings the clip
    # with "no on-screen text, lowers retention". Flipping on is
    # always safe — the renderer already knows how to NOT burn
    # captions when there's no transcript.
    if captions.get("enabled") is False:
        captions["enabled"] = True
        fixes.append(FixOutcome(
            field="captions.enabled", before=False, after=True,
            why="Activé los subtítulos — sin texto en pantalla la gente se va antes",
        ))

    # ---- Fix 1b: lead_ms — DELETED in slice N.2 -----------------
    # Was: nudge lead_ms 0/None → 120ms ("recommended read-ahead").
    # Operator pushback: 120ms is arbitrary and the auto-fix surfaced
    # it as a checkmark even when nothing visible changed for them.
    # Caption lead-time stays at whatever the operator picked (or the
    # brand-kit default, which we no longer override). Removed in
    # both places: this engine + the renderer's word_captions module.

    # ---- Fix 1c: fill banner URL from brand kit -----------------
    # Operator's brand kit usually has a kick URL/handle saved. If
    # banner.url is empty and we have one to use, fill it. This
    # unblocks Fix 3 below (auto-enable banner when URL exists).
    if is_stream_source and brand_kit_url and not (banner.get("url") or "").strip():
        before = banner.get("url")
        banner["url"] = brand_kit_url
        fixes.append(FixOutcome(
            field="banner.url", before=before, after=brand_kit_url,
            why="Puse la URL del banner, tomada de tu estilo",
        ))

    # ---- Fix 2: silently set safe_zone_platform if missing -------
    # Slice N.2 — set the value but DON'T surface as a fix to the
    # operator. The picked target is invisible scoring config and
    # listing it as a fix made the diff feel like padding.
    if not out.get("safe_zone_platform"):
        out["safe_zone_platform"] = safe_zone_platform

    # ---- Fix 3: enable banner when URL exists but toggle is off ---
    url = (banner.get("url") or "").strip() if isinstance(banner.get("url"), str) else ""
    if is_stream_source and url and not banner.get("enabled"):
        banner["enabled"] = True
        fixes.append(FixOutcome(
            field="banner.enabled", before=False, after=True,
            why="Activé el banner de abajo — ya hay URL",
        ))

    # ---- Fix 4: pick the right banner variant for the clip style ---
    # The Kick-branded variants (repost_page wordmark, green_block, the
    # black-bar classic) only fit a KICK-sourced clip. For the other
    # stream platforms (twitch / youtube) the banner-ful styles map to
    # `platform_band` — the brand-colored band + white glyph that credits
    # the source correctly (see overlay_defaults.source_banner). The
    # logo-free `kick_minimal_url` stays platform-agnostic.
    style_id = (out.get("clip_style") or "").strip()
    recommended_variant: str | None = None
    source_p = (source_platform or "").strip().lower()
    if not is_stream_source:
        style_id = ""  # skip banner-variant suggestion for non-stream clips
    if style_id == "repost_page_viral":
        recommended_variant = (
            "kick_repost_page" if source_p == "kick" else "platform_band"
        )
    elif style_id == "clean_creator":
        recommended_variant = "kick_minimal_url"
    elif style_id == "gaming_chaos":
        recommended_variant = (
            "kick_green_block" if source_p == "kick" else "platform_band"
        )
    elif style_id == "documentary":
        recommended_variant = (
            "kick_black_bar_classic" if source_p == "kick" else "platform_band"
        )
    elif style_id == "minimal_native":
        recommended_variant = "kick_minimal_url"
    if recommended_variant and banner.get("variant") not in (recommended_variant, ""):
        before = banner.get("variant")
        # Only flip if the variant doesn't match the chosen style. Don't
        # touch a deliberate operator pick (variant already set to
        # something else) — opt for "we suggest" rather than "we force".
        # Heuristic: leave the variant alone when the operator has
        # explicitly set it. We detect "explicit" by it being non-None.
        if before is None:
            banner["variant"] = recommended_variant
            fixes.append(FixOutcome(
                field="banner.variant", before=before, after=recommended_variant,
                why=f"Elegí el banner que combina con tu estilo {style_id}",
            ))

    # ---- Fix 5: reposition captions out of platform danger zone ---
    if captions.get("enabled", True):
        cap_rects = [_caption_rect(captions.get("position"))]
        warnings = detect_collisions(cap_rects, out.get("safe_zone_platform") or safe_zone_platform)
        if any(w.severity in ("block", "warn") and w.overlay == "captions" for w in warnings):
            before_pos = captions.get("position")
            # Try positions in order of "least likely to collide on a
            # vertical short" until one comes back clean.
            for trial in ("upper_third", "centered", "lower_third", "bottom"):
                if trial == before_pos:
                    continue
                trial_rect = _caption_rect(trial)
                trial_warnings = detect_collisions(
                    [trial_rect],
                    out.get("safe_zone_platform") or safe_zone_platform,
                )
                trial_caption_warnings = [
                    w for w in trial_warnings if w.overlay == "captions"
                ]
                if not trial_caption_warnings:
                    captions["position"] = trial
                    fixes.append(FixOutcome(
                        field="captions.position", before=before_pos, after=trial,
                        why=f"Moví los subtítulos a {trial} — los tapaba la app",
                    ))
                    break

    return AIFixesResult(new_overlay_config=out, fixes=fixes)


def _caption_rect(position: object) -> OverlayRect:
    """Mirror the JS helper that turns a caption position string into
    a fractional rectangle on the 9:16 frame."""
    pos = str(position or "lower_third")
    y = {
        "upper_third": 0.27,
        "centered": 0.46,
        "lower_third": 0.58,
        "bottom": 0.88,
    }.get(pos, 0.58)
    return OverlayRect(name="captions", x=0.04, y=y, w=0.92, h=0.10)


__all__ = ["AIFixesResult", "FixOutcome", "apply_ai_fixes"]
