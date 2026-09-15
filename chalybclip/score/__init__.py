"""Pre-publish Growth Score — deterministic, no LLM.

`score.growth.score_clip` turns a clip's signals + caption + target platforms
into a `GrowthScoreCard`: an overall growth score, a per-platform fit +
publish/skip verdict, and the content tags that drive fatigue spacing.
Computed BEFORE publishing so the engine optimizes instead of blasting.
"""

from .growth import GrowthInput, fallback_card, score_clip

__all__ = ["GrowthInput", "fallback_card", "score_clip"]
