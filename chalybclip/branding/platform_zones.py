"""Per-platform safe-zone definitions — slice I.2.

Each short-form platform paints UI chrome over the video: status bar,
profile pill, right-side action column, caption + CTA dock at the
bottom. If our captions / banner / hook live inside those rectangles,
they get COVERED on the actual platform — the operator's clip ships
looking great in our preview and gets eaten by Reels chrome in the
wild.

This module is the single source of truth for which rectangles each
platform covers. Values are FRACTIONS of the 9:16 frame (0.0 - 1.0)
so the same definition drives:

  - the preview-only dashed overlay the editor shows
  - the collision detector that warns "caption overlaps Reels CTA"
  - the future auto-reposition that nudges captions out of danger

Measurements are from the platform's current public posting UI as of
2026 — accurate-enough to ship; not pixel-perfect (each app A/B tests
the chrome constantly). Tune via `_PLATFORM_ZONES` if any platform
shifts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PlatformId = Literal["tiktok", "reels", "shorts", "kick", "clean"]
ZoneKind = Literal[
    "top_unsafe",
    "bottom_unsafe",
    "right_action",
    "caption_danger",
    "profile_username",
]


@dataclass(frozen=True)
class SafeZone:
    """One rectangular danger zone on a 9:16 frame.

    Coordinates are FRACTIONAL — multiply by the frame's width/height
    to get pixel positions. `(x, y)` is the top-left corner; `(w, h)`
    is the rect size. `kind` lets the UI color/label it; `reason` is
    the human-readable warning shown when an overlay collides.
    """

    kind: ZoneKind
    x: float
    y: float
    w: float
    h: float
    reason: str


# Per-platform zone catalogs. Each list is the FULL chrome that platform
# paints on top of the playback frame; the collision checker iterates
# every entry.
_PLATFORM_ZONES: dict[PlatformId, list[SafeZone]] = {
    "tiktok": [
        SafeZone(
            kind="top_unsafe",
            x=0.0, y=0.0, w=1.0, h=0.08,
            reason="TikTok top status / nav bar covers this strip",
        ),
        SafeZone(
            kind="profile_username",
            x=0.0, y=0.74, w=0.70, h=0.10,
            reason="TikTok profile pill + username lives in the bottom-left",
        ),
        SafeZone(
            kind="caption_danger",
            x=0.0, y=0.84, w=0.78, h=0.10,
            reason="TikTok caption + hashtags overlay covers this band",
        ),
        SafeZone(
            kind="bottom_unsafe",
            x=0.0, y=0.94, w=1.0, h=0.06,
            reason="TikTok audio chip + bottom nav covers this strip",
        ),
        SafeZone(
            kind="right_action",
            x=0.86, y=0.30, w=0.14, h=0.55,
            reason="TikTok right-side action column (like / comment / share / audio)",
        ),
    ],
    "reels": [
        SafeZone(
            kind="top_unsafe",
            x=0.0, y=0.0, w=1.0, h=0.09,
            reason="Instagram Reels top status + camera notch zone",
        ),
        SafeZone(
            kind="profile_username",
            x=0.0, y=0.76, w=0.65, h=0.10,
            reason="Reels profile + follow button live bottom-left",
        ),
        SafeZone(
            kind="caption_danger",
            x=0.0, y=0.85, w=0.78, h=0.11,
            reason="Reels caption + hashtags overlay covers this band",
        ),
        SafeZone(
            kind="bottom_unsafe",
            x=0.0, y=0.94, w=1.0, h=0.06,
            reason="Reels audio + bottom nav covers this strip",
        ),
        SafeZone(
            kind="right_action",
            x=0.87, y=0.32, w=0.13, h=0.55,
            reason="Reels right-side action column (heart / message / share / audio)",
        ),
    ],
    "shorts": [
        SafeZone(
            kind="top_unsafe",
            x=0.0, y=0.0, w=1.0, h=0.07,
            reason="YouTube Shorts top search + Shorts label",
        ),
        SafeZone(
            kind="profile_username",
            x=0.0, y=0.72, w=0.70, h=0.08,
            reason="Shorts channel + Subscribe button lives bottom-left",
        ),
        SafeZone(
            kind="caption_danger",
            x=0.0, y=0.80, w=0.78, h=0.12,
            reason="Shorts title + description covers this band",
        ),
        SafeZone(
            kind="bottom_unsafe",
            x=0.0, y=0.93, w=1.0, h=0.07,
            reason="Shorts audio chip + nav covers this strip",
        ),
        SafeZone(
            kind="right_action",
            x=0.88, y=0.30, w=0.12, h=0.58,
            reason="Shorts right-side action column (like / dislike / comment / share)",
        ),
    ],
    "kick": [
        # Kick clips are watched mostly on desktop, so chrome is lighter.
        # Mobile Kick still paints a top header + bottom nav.
        SafeZone(
            kind="top_unsafe",
            x=0.0, y=0.0, w=1.0, h=0.06,
            reason="Kick clip header + back button",
        ),
        SafeZone(
            kind="bottom_unsafe",
            x=0.0, y=0.92, w=1.0, h=0.08,
            reason="Kick player controls + timeline",
        ),
    ],
    "clean": [
        # No-op platform — used when the operator wants the raw preview
        # without any simulated chrome.
    ],
}


def zones_for_platform(platform: str | None) -> list[SafeZone]:
    """Return the zone list for `platform`, defaulting to `tiktok`
    (the strictest mainstream short-form chrome — if a clip is safe
    for TikTok, it's almost always safe for Reels + Shorts too)."""
    if isinstance(platform, str) and platform in _PLATFORM_ZONES:
        return _PLATFORM_ZONES[platform]  # type: ignore[literal-required,index,unused-ignore]
    return _PLATFORM_ZONES["tiktok"]


def platform_choices() -> list[tuple[PlatformId, str]]:
    """Dropdown labels for the `platform_overlay_preview` selector."""
    return [
        ("clean", "Clean — no simulated UI"),
        ("tiktok", "TikTok"),
        ("reels", "Instagram Reels"),
        ("shorts", "YouTube Shorts"),
        ("kick", "Kick clip page"),
    ]


# ---- Collision detection ----


@dataclass(frozen=True)
class OverlayRect:
    """One overlay element on the frame (caption / banner / hook)."""

    name: str  # "captions", "banner", "top_hook"
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class CollisionWarning:
    """One zone-collision finding for the dashboard banner."""

    overlay: str  # "captions" / "banner" / "top_hook"
    zone: ZoneKind
    severity: Literal["warn", "block"]
    message: str


def detect_collisions(
    overlays: list[OverlayRect],
    platform: str | None,
) -> list[CollisionWarning]:
    """Cross every overlay rect against every zone and emit warnings
    for rectangles that overlap. Severity is `block` only for the
    bottom_unsafe + top_unsafe bands (platform UI literally CANNOT
    be moved, so an overlay there ships broken). Everything else is
    `warn` because the operator may genuinely want their caption on
    top of the profile dock (rare but valid)."""
    if not overlays:
        return []
    zones = zones_for_platform(platform)
    warnings: list[CollisionWarning] = []
    for overlay in overlays:
        for zone in zones:
            if _rects_overlap(overlay, zone):
                severity: Literal["warn", "block"] = (
                    "block" if zone.kind in ("top_unsafe", "bottom_unsafe") else "warn"
                )
                warnings.append(
                    CollisionWarning(
                        overlay=overlay.name,
                        zone=zone.kind,
                        severity=severity,
                        message=f"{overlay.name.title()}: {zone.reason}",
                    )
                )
    return warnings


def _rects_overlap(a: OverlayRect, b: SafeZone) -> bool:
    """Axis-aligned-rectangle overlap test on fractional coords."""
    if a.x + a.w <= b.x or b.x + b.w <= a.x:
        return False
    if a.y + a.h <= b.y or b.y + b.h <= a.y:
        return False
    return True


__all__ = [
    "CollisionWarning",
    "OverlayRect",
    "PlatformId",
    "SafeZone",
    "ZoneKind",
    "detect_collisions",
    "platform_choices",
    "zones_for_platform",
]
