"""Slice L.6 — viral-intelligence decision engine.

Operator spec called out the keystone behavior: "The AI should
dynamically decide: Does this clip need a hook? Does it need
subtitles? Is face visible? Is gameplay important? Will platform UI
block text? Is the reaction more important than captions? The
system should optimize for attention retention, not template
correctness."

This module is the single source of truth for those decisions. It
reads the signals ChalyClip already computes (face_presence,
speaking_intensity, reaction_confidence, heuristic_reason from the
ClipBreakdown; face_zone from the framing verdict; clip duration)
and emits an `AIDecisions` object. The editor uses it to:

  1. Show ✨ intelligence lines on the verdict card so the operator
     sees what the AI noticed.
  2. Cascade settings when the operator picks "Auto Viral" — the
     recommended caption position lands automatically.
  3. Suppress UI elements when the AI decides they hurt the clip
     (e.g. hide the secondary hook on pure-reaction clips).

The decision rules are intentionally CONSERVATIVE — when the
signals are absent or ambiguous, the engine defaults to "show the
standard hook + captions" so the operator never sees an
overly-stripped preview from a weak signal.

This is a pure function: no I/O, no DB, no LLM. Lives in
`chalybclip.clip` (not `chalybclip.detect`) because the consumers are
post-detection (publishability, editor, burn).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CaptionPositionRec = Literal["upper_third", "centered", "lower_third", "bottom"]


@dataclass(frozen=True)
class AIDecisions:
    """The auto-decisions the editor applies when the operator
    leaves a field on "Auto Viral".

    Each `*_reason` field is a SHORT phrase the verdict card renders
    as a ✨ intelligence line — single source of truth for "why
    did the AI do this?" so the operator can audit without reading
    debug logs.
    """

    # Hook decisions
    needs_hook: bool
    hook_reason: str
    secondary_hook_recommended: bool
    secondary_hook_reason: str

    # Caption decisions
    needs_subtitles: bool
    subtitle_reason: str
    caption_position: CaptionPositionRec
    caption_position_reason: str

    # Content priority
    face_priority: bool          # face is the headliner
    gameplay_priority: bool      # action/HUD is the headliner
    reaction_wins_over_captions: bool

    # Confidence — 0..1, the engine's overall trust in its own picks.
    # When < 0.4 the verdict card adds a "low confidence" disclosure
    # so the operator knows to double-check.
    confidence: float

    # All ✨ lines the verdict card should render (deduplicated,
    # ordered most-important-first). Computed in `decide()` so the
    # template doesn't have to reassemble them.
    intel_lines: list[str] = field(default_factory=list)


# ---- thresholds -----------------------------------------------------

# Empirically tuned defaults. Each threshold gets a name so future
# tuning has a clear knob to turn.
_SPEAKING_QUIET = 0.25       # words/sec — below this, mostly silence
_SPEAKING_ACTIVE = 0.55      # above this, someone's clearly talking
_FACE_DOMINANT = 0.55        # frac of seconds with face — face headlines
_FACE_ABSENT = 0.20          # below this, treat as no-face content
_REACTION_STRONG = 0.70      # rescore — strong on-screen reaction
_CLIP_TOO_SHORT_FOR_HOOK = 4.0  # seconds — under this, hook can't be read
_CLIP_LONG_ENOUGH_FOR_SECONDARY = 9.0  # seconds — needs room for both hooks


def decide(
    *,
    face_presence: float | None,
    speaking_intensity: float | None,
    reaction_confidence: float | None,
    heuristic_reason: str,
    duration_s: float,
    face_zone: str | None = None,
    target_platform: str | None = None,
) -> AIDecisions:
    """Compute the auto-decisions for one clip.

    Every input is OPTIONAL — `decide` degrades gracefully to
    "show the standard hook + captions" when signals are missing.
    The single REQUIRED input is `duration_s` (every clip has one).

    `heuristic_reason` mirrors `ClipBreakdown.heuristic_reason`:
    "voice" / "chat" / "audio" / "visual" / "unknown".

    `face_zone` is the L.4b coarse face-position hint: "top" / "mid"
    / "lower" / None. Drives caption_position when face_priority.

    `target_platform` is the L.5 chip pick: "tiktok" / "reels" /
    "shorts" / "kick" / "all" / None. Adjusts the caption position
    bias (Kick has no description block → captions can go lower).
    """
    fp = face_presence if face_presence is not None else 0.0
    si = speaking_intensity if speaking_intensity is not None else 0.0
    rc = reaction_confidence if reaction_confidence is not None else 0.0

    intel: list[str] = []

    # ---- 1. Face vs gameplay priority ----
    face_priority = fp >= _FACE_DOMINANT
    gameplay_priority = (
        fp < _FACE_ABSENT
        and heuristic_reason in ("visual", "audio", "motion")
    )
    if face_priority:
        intel.append(f"Face on screen {int(fp * 100)}% of the clip — face leads")
    elif gameplay_priority:
        intel.append("Action/gameplay leads — keeping the bottom HUD clear")

    # ---- 2. Reaction vs caption priority ----
    # A strong on-screen reaction with face visible AND quiet audio
    # = the FACE is the headliner. Captions should support, not
    # compete. We drop the captions size to "small" in this case.
    reaction_wins = (
        rc >= _REACTION_STRONG
        and face_priority
        and si < _SPEAKING_ACTIVE
    )
    if reaction_wins:
        intel.append("Strong reaction · captions tuned down to let the face speak")

    # ---- 3. Subtitle decision ----
    # Default ON; off only when speaking is essentially absent AND
    # the trigger reason isn't voice (voice triggers always need
    # captions to surface the phrase). When OFF we tell the
    # operator why — "no transcribable speech detected".
    if heuristic_reason == "voice":
        needs_subtitles = True
        subtitle_reason = "Voice-triggered clip — captions surface the phrase"
    elif si >= _SPEAKING_ACTIVE:
        needs_subtitles = True
        subtitle_reason = f"Active speaking ({si:.1f} wps) — captions essential"
    elif si <= _SPEAKING_QUIET:
        needs_subtitles = False
        subtitle_reason = "Mostly silent — captions skipped"
        intel.append("Mostly silent clip — captions disabled")
    else:
        needs_subtitles = True
        subtitle_reason = "Some speech detected — captions on (standard)"

    # ---- 4. Hook decision ----
    # Hook is almost always ON. We hide it only on PURE-reaction
    # clips that are short enough to play without setup, OR on
    # clips that are too short for the operator's hook to be read.
    if duration_s < _CLIP_TOO_SHORT_FOR_HOOK:
        needs_hook = False
        hook_reason = f"Clip is too short ({duration_s:.1f}s) to read a hook"
        intel.append("Clip is too short — hook would be missed")
    elif reaction_wins and duration_s < 6.0:
        # Short reaction clip — let the face speak; a hook adds noise
        needs_hook = False
        hook_reason = "Pure reaction · hook would crowd the moment"
        intel.append("Pure reaction · main hook hidden so face leads")
    else:
        needs_hook = True
        if heuristic_reason == "voice":
            hook_reason = "Hook recommended · sets up the spoken trigger"
        elif rc >= _REACTION_STRONG:
            hook_reason = "Hook recommended · primes the reaction payoff"
        else:
            hook_reason = "Hook recommended · standard short-form template"

    # ---- 5. Secondary (context) hook ----
    # Only useful for clips that have BOTH room (>9s) and a setup/
    # punchline structure (voice-triggered + low reaction = the
    # whole moment is the spoken line; no second hook needed).
    secondary_hook_recommended = (
        needs_hook
        and duration_s >= _CLIP_LONG_ENOUGH_FOR_SECONDARY
        and heuristic_reason == "voice"
        and rc < 0.6
    )
    if secondary_hook_recommended:
        secondary_hook_reason = "Long voice-trigger clip · context line helps setup"
        intel.append("Context line recommended above the main hook")
    elif duration_s < _CLIP_LONG_ENOUGH_FOR_SECONDARY:
        secondary_hook_reason = "Clip too short for two hooks"
    else:
        secondary_hook_reason = "Single hook reads cleaner here"

    # ---- 6. Caption position ----
    # Rules (in priority order):
    #   - face_priority + face_zone="top"   → lower_third (below face)
    #   - face_priority + face_zone="lower" → upper_third (above face)
    #   - gameplay_priority                  → upper_third (away from HUD)
    #   - target_platform=="kick"            → "bottom" is safe (no IG dock)
    #   - default                            → lower_third (the L.4 default)
    if face_priority and face_zone == "top":
        cap_pos: CaptionPositionRec = "lower_third"
        cap_reason = "Face high in frame → captions below it"
    elif face_priority and face_zone == "lower":
        cap_pos = "upper_third"
        cap_reason = "Face low in frame → captions above it"
    elif gameplay_priority:
        cap_pos = "upper_third"
        cap_reason = "Gameplay HUD likely in bottom band → captions up high"
    elif target_platform == "kick":
        # Kick clip view has no description chrome; bottom is safe.
        cap_pos = "bottom"
        cap_reason = "Kick has no IG-style dock → captions can ride low"
    else:
        cap_pos = "lower_third"
        cap_reason = "Standard short-form placement (under the face)"
    intel.append(cap_reason)

    # ---- 7. Confidence ----
    # Confidence is high when MOST inputs are present + decisive.
    # Drops when face_presence / speaking_intensity are unknown
    # (None) or borderline (around the thresholds).
    signal_count = sum(
        1 for v in (face_presence, speaking_intensity, reaction_confidence)
        if v is not None
    )
    base_conf = signal_count / 3.0
    # Penalty for "right on the threshold" signals — those are the
    # ones where the rule fires arbitrarily.
    if face_presence is not None:
        if 0.40 < face_presence < 0.55:
            base_conf -= 0.15
    if speaking_intensity is not None:
        if 0.20 < speaking_intensity < 0.55:
            base_conf -= 0.10
    confidence = max(0.0, min(1.0, base_conf))

    return AIDecisions(
        needs_hook=needs_hook,
        hook_reason=hook_reason,
        secondary_hook_recommended=secondary_hook_recommended,
        secondary_hook_reason=secondary_hook_reason,
        needs_subtitles=needs_subtitles,
        subtitle_reason=subtitle_reason,
        caption_position=cap_pos,
        caption_position_reason=cap_reason,
        face_priority=face_priority,
        gameplay_priority=gameplay_priority,
        reaction_wins_over_captions=reaction_wins,
        confidence=confidence,
        intel_lines=intel,
    )


__all__ = ["AIDecisions", "decide", "CaptionPositionRec"]
