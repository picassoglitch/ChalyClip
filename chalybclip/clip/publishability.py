"""Publishability verdict — slice I.3.

Separate from `viral_score`: viral asks "is this clip cookable?";
publishability asks "is THIS render safe to ship?" The two are
correlated but not the same — a 95-viral clip with captions covered by
the platform's caption dock is NOT publish-ready.

The verdict combines:

  - hook clarity        (title/top hook text present + readable)
  - caption readability (words-per-second band)
  - face / subject presence
  - motion + audio energy (rough proxy for "something is happening")
  - safe-zone compliance (no critical zone collisions)
  - dead-air risk
  - clip style completeness (banner URL set when banner is on)
  - integrity flags     (G.4 — reserved hook, currently always pass)

Scored 0-100 and bucketed:

  publish_ready : 75-100  - ship without manual review
  needs_edit    : 55-74   - one or two fixable issues
  reject        : 0-54    - structural problems

Each item that contributes contributes a `reason` (positive) or a
`warning` (negative). The dashboard renders them as the AI Director
checklist alongside the score ring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from chalybclip.branding.platform_zones import (
    OverlayRect,
    detect_collisions,
)

from .breakdown import ClipBreakdown

PublishStatus = Literal["publish_ready", "needs_edit", "reject"]


@dataclass(frozen=True)
class PublishabilityVerdict:
    """The single source of truth for "is this clip ready to ship?"."""

    score: int                          # 0-100
    status: PublishStatus               # bucketed for fast UI rendering
    reasons: list[str] = field(default_factory=list)   # what's GOOD
    warnings: list[str] = field(default_factory=list)  # what's missing / broken
    recommended_action: str = ""        # one-liner CTA for the dashboard
    blocking: list[str] = field(default_factory=list)  # critical issues → reject


def compute_publishability(
    *,
    breakdown: ClipBreakdown,
    overlay_config: dict[str, object] | None,
    safe_zone_platform: str = "tiktok",
) -> PublishabilityVerdict:
    """Score + classify a clip given its breakdown + the operator's
    current overlay settings.

    `overlay_config` is the per-clip overlay dict (the same shape
    `_persist_branding_to_brand_kit` consumes). When unset we use safe
    defaults so the verdict still produces a sensible read.
    """
    overlay = overlay_config if isinstance(overlay_config, dict) else {}
    banner = overlay.get("banner") if isinstance(overlay.get("banner"), dict) else {}
    captions = (
        overlay.get("captions") if isinstance(overlay.get("captions"), dict) else {}
    )
    top_hook = (
        overlay.get("top_hook") if isinstance(overlay.get("top_hook"), dict) else {}
    )
    assert isinstance(banner, dict)
    assert isinstance(captions, dict)
    assert isinstance(top_hook, dict)

    # Slice N.7 — baseline lifted to 50 (was 35). Bonuses still
    # stack for clips that earn them. With the new publish_ready
    # threshold at 55, a clip with one decent signal usually clears.
    # 100 is reserved for clips that hit every bonus (rare).
    score = 50
    reasons: list[str] = []
    warnings: list[str] = []
    blocking: list[str] = []

    # ---- hook ---------------------------------------------------
    # Slice N.6 — "no hook text" warning REMOVED. Operator pushback:
    # don't flag/judge the user's content choices. Some clips are
    # designed to play without a hook (raw reactions, gameplay
    # punchlines). Hook presence earns a bonus; absence is silent.
    raw_title = overlay.get("title_text")
    hook_title = raw_title.strip() if isinstance(raw_title, str) else ""
    top_hook_text = (
        top_hook.get("text", "").strip()
        if isinstance(top_hook.get("text"), str)
        else ""
    )
    has_hook = bool(hook_title or top_hook_text)
    if has_hook:
        score += 15
        if len(hook_title or top_hook_text) >= 25:
            reasons.append("Strong hook — viewers see the payoff in the first second")
        else:
            reasons.append("Hook text present")
    # No hook → no bonus, no warning. Operator's call.

    # ---- caption readability -----------------------------------
    # Slice N.6 — all caption warnings removed (pacing, transcript-not-
    # ready, captions-disabled). They were content-level: words-per-
    # second is dictated by the streamer's speech rate, and disabling
    # captions is a deliberate operator choice. Reward when captions
    # are in the sweet spot; silent otherwise.
    captions_enabled = bool(captions.get("enabled", True))
    wps = breakdown.speaking_intensity
    if captions_enabled and wps is not None and 0.8 <= wps <= 2.2:
        score += 15
        reasons.append("Captions are at a comfortable read pace")
    elif captions_enabled and wps is not None and (0.5 <= wps < 0.8 or 2.2 < wps <= 3.0):
        score += 5
        # Partial bonus, no warning.

    # ---- face presence -----------------------------------------
    # Slice N.4 — face presence is content-level (we can't change
    # the underlying video). Reward when high but DON'T warn when
    # low — operator pushback: "this isn't something the AI can fix
    # so it shouldn't impact the score negatively". Low face just
    # means the bonus isn't earned, not that the clip is broken.
    fp = breakdown.face_presence
    if fp is not None:
        if fp >= 0.55:
            score += 12
            reasons.append("Strong face presence — viewers see the subject clearly")
        elif fp >= 0.30:
            score += 6
            reasons.append("Subject visible most of the clip")
        # Below 0.30 = no bonus, no warning. Quietly noted in the
        # L.6 AI decisions instead.

    # ---- motion + audio energy ----------------------------------
    motion = breakdown.motion_score
    if motion is not None:
        if motion >= 0.5:
            score += 8
            reasons.append("Motion / visual interest is high")
        elif motion >= 0.2:
            score += 4
    if breakdown.heuristic_score >= 0.6:
        score += 5
    elif breakdown.heuristic_score >= 0.3:
        score += 2

    # ---- safe-zone compliance ----------------------------------
    # Slice N.6 — collision WARNINGS removed. The L.3 always-on red
    # zone overlay on the preview already teaches the operator
    # visually where content gets covered by platform UI. Surfacing
    # the same info again as a stacked text warning list is the
    # exact noise the operator told us to kill ("lets not flag or
    # judge on the users content").
    # We still REWARD a clean composition (no collisions = +12) but
    # don't penalize one — the operator sees the red wash on the
    # preview and can decide.
    overlays = _overlay_rects_from_config(overlay)
    coll_warnings = detect_collisions(overlays, safe_zone_platform)
    if not coll_warnings:
        score += 12
        reasons.append(
            f"Captions / banner / hook are safe for {safe_zone_platform.title()} UI"
        )
    # else: no penalty, no warning — the L.3 zones do the talking.

    # ---- dead-air risk -----------------------------------------
    # Slice N.4 — softened. Reward when active speech is there;
    # don't penalize silent clips (operator can't add audio that
    # isn't there). Silent clips just don't earn the bonus.
    if wps is not None:
        if wps >= 0.8:
            score += 8
        elif wps >= 0.4:
            score += 4
        # Below 0.4 = no bonus, no penalty. Surfaced via the L.6
        # AI decisions ("Mostly silent clip") as informational
        # rather than as a publishability warning.

    # ---- clip style completeness --------------------------------
    # Slice N.6 — "banner on but no URL set" warning removed.
    # The M.9 URL normalizer auto-fills "KICK.COM/YOURHANDLE" when
    # the field is empty, so there's no broken state to flag. URL
    # presence earns a small bonus; absence is silent.
    if banner.get("enabled"):
        url = (banner.get("url") or "").strip() if isinstance(banner.get("url"), str) else ""
        if url:
            score += 5

    # ---- integrity hooks (slice G.4) ---------------------------
    # breakdown.integrity_issues is the freeze/silence detector
    # output from clip_breakdown(). Each is a dict with kind /
    # start_s / end_s / label. Penalty per issue is small (-6) so
    # a single 3-second freeze in the middle of an otherwise-great
    # clip doesn't tank it to "reject" — but two or three issues
    # add up and push the verdict down a tier. Each issue surfaces
    # as a warning (using .label) so the operator can decide whether
    # to trim the clip's window with the G.4b auto-trim button.
    issues = getattr(breakdown, "integrity_issues", None) or ()
    for issue in issues:
        score -= 6
        if isinstance(issue, dict):
            label = issue.get("label") or "stream integrity issue"
            warnings.append(str(label))
        else:
            # Legacy str shape — accept it so old callers don't break.
            warnings.append(str(issue))
    # Three or more integrity issues = the underlying VOD is too
    # broken in this window to safely ship; mark as blocking so
    # the verdict bucket flips to reject even if the score is
    # somehow still above 30.
    if len(issues) >= 3:
        blocking.append(
            f"stream integrity: {len(issues)} issues detected — re-trim or skip"
        )

    # ---- clamp + bucket + recommend -----------------------------
    # Slice N.7 — thresholds lowered again: publish_ready 60 → 50,
    # needs_edit 35 → 30. With the N.7 baseline lift (35→55), this
    # means a typical clip lands publish_ready before any bonuses
    # stack. Confidence-by-default: the AI's "this is worth
    # clipping" gets reflected in the verdict instead of fighting it.
    score = max(0, min(100, score))
    if blocking:
        status: PublishStatus = "reject"
    elif score >= 55:
        status = "publish_ready"
    elif score >= 30:
        status = "needs_edit"
    else:
        status = "reject"

    recommended_action = _pick_recommendation(
        status=status,
        score=score,
        warnings=warnings,
        blocking=blocking,
        has_hook=has_hook,
        captions_enabled=captions_enabled,
    )

    # Slice K.2 — cap the lists shown by default. The full set still
    # lands in the Advanced Analysis accordion the dashboard renders
    # below; the hero verdict card only shows the top 3 positives +
    # top 2 fixes. Long lists drown the emotional decision moment.
    reasons_short = [_shorten(r) for r in reasons[:3]]
    warnings_short = [_shorten(w) for w in warnings[:2]]
    blocking_short = [_shorten(b) for b in blocking[:1]]

    return PublishabilityVerdict(
        score=score,
        status=status,
        reasons=reasons_short,
        warnings=warnings_short,
        recommended_action=recommended_action,
        blocking=blocking_short,
    )


def _shorten(msg: str, *, max_chars: int = 60) -> str:
    """Trim a verdict bullet to chip-length for the hero card. Full
    text remains available via the bullet's title= attribute (the
    dashboard renders the long form as a hover tooltip)."""
    msg = msg.strip()
    if len(msg) <= max_chars:
        return msg
    # Cut on a word boundary inside max_chars when possible.
    cut = msg.rfind(" ", 0, max_chars - 1)
    if cut < max_chars // 2:
        cut = max_chars - 1
    return msg[:cut].rstrip(",.;:") + "…"


def _overlay_rects_from_config(overlay: dict[str, object]) -> list[OverlayRect]:
    """Approximate the on-frame rectangles of each visible overlay so
    `detect_collisions` can test them against the platform's zones.
    Coordinates match the JS counterpart in clip_detail.html so server
    + client agree on what's where."""
    banner = overlay.get("banner") if isinstance(overlay.get("banner"), dict) else {}
    captions = (
        overlay.get("captions") if isinstance(overlay.get("captions"), dict) else {}
    )
    top_hook = (
        overlay.get("top_hook") if isinstance(overlay.get("top_hook"), dict) else {}
    )
    assert isinstance(banner, dict)
    assert isinstance(captions, dict)
    assert isinstance(top_hook, dict)
    rects: list[OverlayRect] = []
    if banner.get("enabled"):
        variant = str(banner.get("variant") or "kick_repost_page")
        band_frac = {
            "kick_minimal_url": 0.04,
            "kick_green_block": 0.08,
            "kick_black_bar_classic": 0.05,
            "kick_repost_page": 0.07,
            "platform_band": 0.05,
        }.get(variant, 0.07)
        rects.append(
            OverlayRect(name="banner", x=0.0, y=1.0 - band_frac, w=1.0, h=band_frac)
        )
    th_text = top_hook.get("text") if isinstance(top_hook.get("text"), str) else ""
    th_text = (th_text or "").strip()
    if top_hook.get("enabled") and th_text:
        rects.append(OverlayRect(name="top_hook", x=0.08, y=0.08, w=0.84, h=0.14))
    if captions.get("enabled", True):
        pos = str(captions.get("position") or "lower_third")
        cap_y = {
            "upper_third": 0.27,
            "centered": 0.46,
            "lower_third": 0.58,
            "bottom": 0.88,
        }.get(pos, 0.58)
        rects.append(OverlayRect(name="captions", x=0.04, y=cap_y, w=0.92, h=0.10))
    return rects


def _pick_recommendation(
    *,
    status: PublishStatus,
    score: int,
    warnings: list[str],
    blocking: list[str],
    has_hook: bool,
    captions_enabled: bool,
) -> str:
    """One-liner CTA the dashboard surfaces under the score."""
    if status == "reject" and blocking:
        return "Fix the highlighted issues — they will be visible on the published post"
    if not has_hook and score < 75:
        return "Add a hook headline up top — every viral clip starts with one"
    if not captions_enabled:
        return "Turn captions on — they're the #1 retention driver on muted scrolls"
    if status == "needs_edit":
        return "Click Apply AI fixes — most warnings auto-resolve"
    if status == "publish_ready":
        return "Ready to ship — Publish Confidence is high"
    return "Review the listed warnings, then re-check confidence"


# ---- AI Director — human-readable recommendation strings -----


@dataclass(frozen=True)
class DirectorLine:
    """One line in the AI Director panel."""

    kind: Literal["good", "warn", "block", "info"]
    text: str


def director_lines(
    *,
    breakdown: ClipBreakdown,
    verdict: PublishabilityVerdict,
) -> list[DirectorLine]:
    """Build the AI Director panel's bullet list.

    More conversational than the verdict.reasons/warnings — these are
    what the operator reads at a glance to decide "do I trust this?".
    Returns a small (≤6) ordered list so the panel stays scannable.
    """
    lines: list[DirectorLine] = []
    # 1. Headline reason — the strongest positive signal.
    if breakdown.face_presence is not None and breakdown.face_presence >= 0.55:
        lines.append(DirectorLine(
            kind="good", text="Strong face presence detected",
        ))
    elif breakdown.reaction_confidence is not None and breakdown.reaction_confidence >= 0.6:
        lines.append(DirectorLine(
            kind="good", text="Vision LLM rated this a strong reaction moment",
        ))
    elif breakdown.motion_score is not None and breakdown.motion_score >= 0.5:
        lines.append(DirectorLine(
            kind="good", text="Motion energy is high — visually punchy",
        ))

    # 2. Caption read pace.
    wps = breakdown.speaking_intensity
    if wps is not None:
        if 0.8 <= wps <= 2.2:
            lines.append(DirectorLine(kind="good", text="Captions are at a comfortable read pace"))
        elif wps < 0.4:
            lines.append(DirectorLine(kind="warn", text="Long silent stretches — dead-air risk is high"))
        elif wps > 3.0:
            lines.append(DirectorLine(kind="warn", text="Speech is too fast for captions to keep up"))

    # 3. Safe-zone hits.
    if verdict.blocking:
        lines.append(DirectorLine(kind="block", text=verdict.blocking[0]))
    elif any("UI" in w or "covers" in w for w in verdict.warnings):
        first_zone = next(
            (w for w in verdict.warnings if "covers" in w or "UI" in w), None,
        )
        if first_zone:
            lines.append(DirectorLine(kind="warn", text=first_zone))

    # 4. Hook quality.
    if not any(w.startswith("No hook") for w in verdict.warnings):
        # Hook exists; mention it's strong when it's also long enough.
        if any(r.startswith("Strong hook") for r in verdict.reasons):
            lines.append(DirectorLine(kind="good", text="Hook is sharp and clear"))
    else:
        lines.append(DirectorLine(kind="warn", text="Hook is missing — first second is silent"))

    # 5. Closer — the recommended action.
    if verdict.recommended_action:
        lines.append(DirectorLine(kind="info", text=verdict.recommended_action))

    return lines[:6]


__all__ = [
    "DirectorLine",
    "PublishStatus",
    "PublishabilityVerdict",
    "compute_publishability",
    "director_lines",
]
