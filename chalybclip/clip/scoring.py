"""AI-scoring helper — composite scores derived from `ClipBreakdown`.

Surfaces a small set of human-readable scores for the clip-editor's
"AI insights" strip. The breakdown already aggregates every per-clip
signal we have (heuristic score, motion, face presence, speaking
intensity, vision-LLM rescore); this module turns those raw numbers
into the four creator-facing scores the UI shows:

  * Viral score (0-100)        — composite "is this clip cookable?"
  * Hook strength              — HIGH / MEDIUM / DEVELOPING
  * Caption readability        — GOOD / OK / CHECK
  * Dead-air risk              — LOW / MED / HIGH

These are HEURISTICS, not magic. Their job is to make the operator's
"should I publish this?" call faster — they should never replace
manual review entirely. Each score is also annotated with a one-line
"why" the UI can show on hover so the operator understands what
pushed it up or down.

The scoring formulas are deliberately simple + tunable. When
`reaction_confidence` (vision-LLM rescore) is available the viral
score weights it heavily — the rescore is the strongest single
signal we have. When it's not present we fall back to the heuristic
score plus face-presence boost.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

from .breakdown import ClipBreakdown

HookStrength = Literal["HIGH", "MEDIUM", "DEVELOPING"]
ReadabilityLabel = Literal["GOOD", "OK", "CHECK"]
DeadAirLabel = Literal["LOW", "MED", "HIGH"]


class AIScoreCard(NamedTuple):
    """One-shot bundle of all four scores + their explanation strings."""

    viral_score: int  # 0-100
    viral_why: str

    hook_strength: HookStrength
    hook_why: str

    caption_readability: ReadabilityLabel
    readability_why: str

    dead_air_risk: DeadAirLabel
    dead_air_why: str


# ---- viral score ----


def _viral_score(b: ClipBreakdown) -> tuple[int, str]:
    """Composite 0-100 viral score.

    Slice N.5 — softened. Operator pushback: previous formula was
    `heuristic_score × 90` which produced a "2/100" score for clips
    where the heuristic was weak (e.g. clips picked up via visual or
    chat signals, not voice triggers). That number killed trust.

    Now the score draws on MULTIPLE signals so a clip with low heuristic
    but strong face / speaking / reaction still scores reasonably.

    Slice N.7 — baseline lifted 30 → 55 to match the publishability
    rewrite. The AI ALREADY decided this moment was worth clipping
    when it became a candidate; the viral floor should reflect that
    instead of starting at "barely worth ranking". Bonuses still
    stack for clips that earn them.

       - baseline                   55   (the AI's vote of confidence)
       - reaction_confidence × 30   primary signal when present
       - heuristic_score × 20       fallback when no rescore
       - face_presence × 10         talking-head bump
       - speaking_intensity bonus   capped at +6 for active speech
    """
    base = 55.0
    bits: list[str] = ["baseline 55"]

    if b.reaction_confidence is not None:
        rc = b.reaction_confidence * 30
        base += rc
        bits.append(f"reaction {b.reaction_confidence:.2f} → +{rc:.0f}")
    else:
        # No rescore → lean on the heuristic at a softer weight.
        h = b.heuristic_score * 20
        base += h
        bits.append(f"heuristic ({b.heuristic_reason}) {b.heuristic_score:.2f} → +{h:.0f}")

    if b.face_presence is not None and b.face_presence > 0.3:
        fb = min(10, int((b.face_presence - 0.3) * 18))
        base += fb
        bits.append(f"face {b.face_presence:.2f} → +{fb}")

    if b.speaking_intensity is not None:
        if b.speaking_intensity >= 0.8:
            base += 6
            bits.append("speaking active → +6")
        elif b.speaking_intensity >= 0.4:
            base += 3
            bits.append("some speech → +3")

    return int(max(0, min(100, round(base)))), " · ".join(bits)


# ---- hook strength ----


def _hook_strength(viral: int) -> tuple[HookStrength, str]:
    """Banded label for the viral score.
    Slice N.7 — bands lowered again 65/40 → 60/40. Combined with
    the N.7 baseline lift (30 → 55), this means clips that the AI
    flagged as candidates START at MEDIUM ("solid candidate") instead
    of DEVELOPING. DEVELOPING is now reserved for the very rare clip
    with no real signals."""
    if viral >= 60:
        return "HIGH", f"viral {viral} ≥ 60 — push this clip first"
    if viral >= 40:
        return "MEDIUM", f"viral {viral} in 40-59 — solid candidate"
    return "DEVELOPING", f"viral {viral} < 40 — a strong hook can lift it"


# ---- caption readability ----


def _caption_readability(b: ClipBreakdown) -> tuple[ReadabilityLabel, str]:
    """Words-per-second sweet spot for burned-in captions.

    Industry-standard subtitle pacing maxes out around 2.5 wps before
    viewers stop reading. Below 0.5 wps the captions feel sparse and
    the operator should consider trimming dead air or shortening the clip.
    """
    wps = b.speaking_intensity
    if wps is None:
        return "CHECK", "no transcript yet — can't measure caption pacing"
    if 0.8 <= wps <= 2.2:
        return "GOOD", f"{wps:.1f} words/sec — comfortable read pace"
    if 0.5 <= wps < 0.8 or 2.2 < wps <= 3.0:
        return "OK", f"{wps:.1f} words/sec — viewers may struggle"
    return "CHECK", f"{wps:.1f} words/sec — outside the readable band"


# ---- dead-air risk ----


def _dead_air_risk(b: ClipBreakdown) -> tuple[DeadAirLabel, str]:
    """Inverse of speaking_intensity — short-form viewers bounce on quiet."""
    wps = b.speaking_intensity
    if wps is None:
        return "MED", "no transcript — can't measure dead-air"
    if wps >= 0.8:
        return "LOW", f"{wps:.1f} words/sec — clip is talking through"
    if wps >= 0.4:
        return "MED", f"{wps:.1f} words/sec — some quiet stretches"
    return "HIGH", f"{wps:.1f} words/sec — long quiet stretches; trim or skip"


# ---- public entry ----


def compute_ai_scores(b: ClipBreakdown) -> AIScoreCard:
    """Compute all four AI scores from a clip's breakdown."""
    viral, viral_why = _viral_score(b)
    hook, hook_why = _hook_strength(viral)
    readability, readability_why = _caption_readability(b)
    dead_air, dead_air_why = _dead_air_risk(b)
    return AIScoreCard(
        viral_score=viral,
        viral_why=viral_why,
        hook_strength=hook,
        hook_why=hook_why,
        caption_readability=readability,
        readability_why=readability_why,
        dead_air_risk=dead_air,
        dead_air_why=dead_air_why,
    )
