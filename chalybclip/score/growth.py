"""Pre-publish Growth Score — deterministic, no LLM.

Given a clip's signals (duration, publishability), its composed
caption/hashtags/hook, and the target platforms, `score_clip` returns a
`GrowthScoreCard`: an overall growth score, a per-platform fit + publish/skip
verdict, and the clip's own content tags. The planner
(`publish.growth_engine.plan_growth_publish`) reads `overall_score`,
`platforms[].platform/verdict/score`, and `content_tags` off this card.

This used to be an LLM call (Anthropic Opus). It is now a pure decision tree:

    * per-platform verdict = the clip fits the platform's short-form duration
      ceiling (`platform_specs.fits_duration`) AND clears the publishability
      floor. An over-long clip is 'skip' on a surface that won't take it.
    * per-platform score  = publishability + a small duration-fit bonus, so a
      tight 20s clip outranks a 130s one on short-form and the allocator picks
      best-fit first.
    * content_tags        = deterministic keywords from the stream title,
      detect reason, and caption — comparable across clips so fatigue spacing
      still holds near-duplicate themes.

Pure and synchronous: no router, no DB, no network, no cost. `fallback_card`
remains for the degenerate no-platforms case.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from chalybclip.llm import GrowthScoreCard, PlatformGrowthScore
from chalybclip.publish.pacing import canonical_platform
from chalybclip.publish.platform_specs import fits_duration, spec_for

# Below this publishability score a clip is held back everywhere (the Skip
# band). The per-platform floor the operator tunes lives in the planner
# (`min_score`); this is only the coarse "is it worth posting at all" gate.
_PUBLISH_FLOOR = 40

# Tokens that carry no theme signal — dropped from content tags so fatigue
# spacing compares on the words that actually describe the clip. Bilingual
# (content is Spanish-first, code/titles often English).
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "this", "that", "from", "your", "you",
        "when", "what", "just", "como", "para", "pero", "esta", "este", "esto",
        "todo", "todos", "muy", "los", "las", "una", "uno", "del", "que",
        "por", "con", "sin", "más", "mas", "vod", "clip", "stream", "video",
        "live", "directo", "twitch", "kick", "youtube",
    }
)

_TOKEN_RE = re.compile(r"[a-záéíóúñ0-9]+")


def _band_label(score: int) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 80:
        return "Very Good"
    if score >= 65:
        return "Good"
    if score >= 40:
        return "Weak"
    return "Skip"


@dataclass(frozen=True)
class GrowthInput:
    """Everything the deterministic scorer sees about one clip."""

    clip_id: str
    duration_s: float
    caption: str
    platforms: list[str]
    hashtags: list[str] = field(default_factory=list)
    hook: str = ""
    stream_title: str | None = None
    heuristic_reason: str | None = None
    # Signal numbers from ClipBreakdown (all optional; absent = unknown).
    motion_score: float | None = None
    face_presence: float | None = None
    speaking_intensity: float | None = None
    reaction_confidence: float | None = None
    publishability_score: int | None = None
    # Content tags of recently published clips for this tenant — the fatigue
    # context. Unused by the deterministic scorer (it derives a clip's own
    # tags), kept so callers building GrowthInput don't have to branch.
    recent_content_tags: list[list[str]] = field(default_factory=list)


def _derive_content_tags(inp: GrowthInput) -> list[str]:
    """3-6 lowercase theme descriptors for fatigue spacing, from the clip's
    stream title, detect reason, caption, and hook. Deterministic: same input
    → same tags, so `assess_batch_fatigue` can compare themes across clips."""
    text = " ".join(
        s
        for s in (
            inp.stream_title or "",
            inp.heuristic_reason or "",
            inp.caption or "",
            inp.hook or "",
        )
        if s
    ).lower()
    tags: list[str] = []
    for tok in _TOKEN_RE.findall(text):
        if len(tok) < 4 or tok in _STOPWORDS or tok in tags:
            continue
        tags.append(tok)
        if len(tags) >= 6:
            break
    return tags


def _duration_fit_bonus(platform: str, duration_s: float, base: int) -> int:
    """Reward a clip that sits comfortably under a platform's ceiling: a 20s
    clip is a stronger short than a 130s one. 0..+10 on top of `base`, capped
    at 100. No length known → no bonus."""
    if not duration_s or duration_s <= 0:
        return base
    ceiling = spec_for(platform).max_duration_s
    if ceiling <= 0:
        return base
    headroom = max(0.0, 1.0 - min(1.0, duration_s / ceiling))
    return min(100, base + round(headroom * 10))


def score_clip(inp: GrowthInput) -> GrowthScoreCard:
    """Deterministic Growth Score card for one clip. No LLM, no DB, no cost.

    Every connected platform gets a verdict: 'publish' when the clip fits that
    platform's duration ceiling and clears the publishability floor, else
    'skip'. Score is publishability plus a duration-fit bonus so the allocator
    ranks best-fit first. `overall_score` is the publishability headline."""
    if not inp.platforms:
        return fallback_card(inp)

    base = inp.publishability_score if inp.publishability_score is not None else 50
    base = max(0, min(100, int(base)))
    tags = _derive_content_tags(inp)

    platforms: list[PlatformGrowthScore] = []
    n_publish = 0
    for p in inp.platforms:
        canon = canonical_platform(p)
        fits = fits_duration(canon, inp.duration_s)
        score = _duration_fit_bonus(canon, inp.duration_s, base) if fits else base
        publish = fits and base >= _PUBLISH_FLOOR
        if publish:
            n_publish += 1
            reason = f"Fits {canon}; publishability {base}."
        elif not fits:
            reason = (
                f"Clip {inp.duration_s:.0f}s exceeds {canon}'s "
                f"{spec_for(canon).max_duration_s:.0f}s short-form ceiling."
            )
        else:
            reason = f"Publishability {base} below the {_PUBLISH_FLOOR} floor."
        platforms.append(
            PlatformGrowthScore(
                platform=canon,
                score=score,
                verdict="publish" if publish else "skip",
                label=_band_label(score),
                reason=reason,
            )
        )

    if n_publish == len(platforms) and n_publish > 0:
        decision = "publish_all"
    elif n_publish > 0:
        decision = "publish_select"
    elif base >= _PUBLISH_FLOOR:
        decision = "archive"  # publishable, but fits no connected platform
    else:
        decision = "skip"

    return GrowthScoreCard(
        overall_score=base,
        decision=decision,
        platforms=platforms,
        recommendations=[],
        content_tags=tags,
        summary=f"Deterministic score from publishability ({base}).",
    )


def fallback_card(inp: GrowthInput) -> GrowthScoreCard:
    """Degenerate card when there are no target platforms (or as a safe
    default). Uses the publishability score as the overall growth score (50
    when unknown) and gives every listed platform that same score with a
    'publish' verdict unless it lands in the Skip band. Carries NO content
    tags — the "no information" path leaves fatigue spacing a no-op (only
    `score_clip` derives tags)."""
    base = inp.publishability_score if inp.publishability_score is not None else 50
    base = max(0, min(100, int(base)))
    verdict = "publish" if base >= _PUBLISH_FLOOR else "skip"
    platforms = [
        PlatformGrowthScore(
            platform=canonical_platform(p),
            score=base,
            verdict=verdict,
            label=_band_label(base),
            reason="Publishability fallback (no platform-fit data).",
        )
        for p in inp.platforms
    ]
    decision = "publish_select" if base >= _PUBLISH_FLOOR else "archive"
    return GrowthScoreCard(
        overall_score=base,
        decision=decision,
        platforms=platforms,
        recommendations=[],
        content_tags=[],
        summary="Fallback score from publishability.",
    )
