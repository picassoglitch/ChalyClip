"""Content-fatigue detection — space similar clips instead of saturating.

The scenario the spec calls out: a user uploads 50 Warzone clips; the last six
are the same map, same gun, same bit. Blasting them back-to-back burns the
audience. So before scheduling, we compare each clip's content tags (from the
Growth Score card) against what was recently published AND against the clips
already accepted earlier in this same batch, and HOLD the ones that would
over-saturate a theme — they roll to the next run.

Pure and deterministic: tag sets in, hold verdicts out. The tags themselves
come from the LLM (`GrowthScoreCard.content_tags`); the spacing policy lives
here so it's testable without a model.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


def _norm(tags: Iterable[str]) -> set[str]:
    return {t.strip().lower() for t in tags if t and t.strip()}


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard similarity of two tag sets (0 = disjoint, 1 = identical).
    Empty-vs-anything is 0 — we never call an untagged clip a duplicate."""
    sa, sb = _norm(a), _norm(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


@dataclass(frozen=True)
class FatigueVerdict:
    clip_id: str
    hold: bool
    similarity: float  # peak similarity that drove the decision
    reason: str


def assess_batch_fatigue(
    clips: Sequence[tuple[str, Sequence[str]]],
    *,
    recent_tags: Sequence[Sequence[str]] | None = None,
    threshold: float = 0.6,
    max_similar_per_window: int = 2,
) -> list[FatigueVerdict]:
    """Decide which clips to publish now vs hold for spacing.

    Args:
        clips: `(clip_id, content_tags)` in intended publish order. Order
            matters — earlier clips claim the theme; later near-duplicates are
            the ones held.
        recent_tags: content tags of clips already published recently (the
            running history). A clip too similar to this history counts toward
            the window from the start.
        threshold: Jaccard at/above which two clips count as "the same theme".
        max_similar_per_window: how many clips of one theme may go out in this
            run before the rest are held. 2 lets a pair through, holds the 3rd+.

    Returns a verdict per clip in input order. Held clips are deferred, not
    dropped — the caller reschedules them on the next sweep.
    """
    history = [list(t) for t in (recent_tags or [])]
    # How many already-published clips match each emerging theme. We seed the
    # window counter from history so a theme that's already saturated holds
    # immediately.
    accepted: list[set[str]] = []
    out: list[FatigueVerdict] = []

    for clip_id, tags in clips:
        tagset = _norm(tags)
        # Peak similarity vs history + already-accepted-this-batch.
        peers = history + [list(s) for s in accepted]
        sim = max((jaccard(tagset, p) for p in peers), default=0.0)
        # Count how many same-theme clips already cleared (history + accepted).
        similar_count = sum(
            1 for p in peers if jaccard(tagset, p) >= threshold
        )
        if tagset and sim >= threshold and similar_count >= max_similar_per_window:
            out.append(
                FatigueVerdict(
                    clip_id=clip_id,
                    hold=True,
                    similarity=round(sim, 3),
                    reason=(
                        f"{similar_count} similar clips already queued/published "
                        f"(theme similarity {sim:.0%} ≥ {threshold:.0%}); holding to "
                        "space the content."
                    ),
                )
            )
            continue
        accepted.append(tagset)
        out.append(
            FatigueVerdict(
                clip_id=clip_id,
                hold=False,
                similarity=round(sim, 3),
                reason="ok",
            )
        )
    return out
