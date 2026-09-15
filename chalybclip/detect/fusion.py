"""Weighted multi-signal candidate fusion — slice G.1.

Replaces the old max-score-wins `_merge_candidates` with a proper
weighted formula. Goal: a moment that ONE detector flags at 0.9 should
not outscore a moment that THREE detectors all flag at 0.6 — the
multi-detector consensus is the stronger signal.

The fusion runs per cluster (same temporal grouping as before, default
30-second window). For each cluster:

  1. Group the cluster's evidence by detector source. Each detector
     can contribute at most ONCE (the max score it produced in the
     cluster), so a noisy detector firing 5× in a row doesn't dominate.

  2. Compute the weighted sum:
        weighted_score =
            0.35 × voice
          + 0.20 × visual
          + 0.15 × audio
          + 0.15 × chat
          + 0.10 × viral_llm
          + 0.05 × transcript_hook

     Missing detectors contribute 0.

  3. Add overlap_bonus:
        +0.05 if 2 distinct detectors fire within OVERLAP_WINDOW_S of
            the anchor timestamp
        +0.10 if 3+ distinct detectors fire within OVERLAP_WINDOW_S

  4. Add evidence bonus:
        +0.05 if any cluster member's evidence records face presence
        +0.05 if any member reports strong emotion or large motion

  5. Pick the anchor timestamp: the member of the cluster with the
     highest *combined local evidence* — i.e. the timestamp where the
     most distinct detectors fire within ±5s of it. Ties broken by raw
     score so the existing behavior is a degenerate case (no overlap →
     pick the highest-score member, same as max-score-wins).

  6. Clamp final to [0, 1].

The fused Candidate carries the legacy fields plus an `evidence.fusion`
block describing every contribution so the dashboard's "why is this a
77/100" explainer has real data behind it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from chalybclip.detect.models import Candidate


@dataclass(frozen=True)
class FusionWeights:
    """Per-detector multipliers — see module docstring for rationale."""

    voice: float = 0.35
    visual: float = 0.20
    audio: float = 0.15
    chat: float = 0.15
    viral: float = 0.10
    transcript_hook: float = 0.05


@dataclass(frozen=True)
class FusionBonuses:
    """Additive bumps when independent signals corroborate each other."""

    # +0.05 if 2 distinct detectors fire within `overlap_window_s` of the
    # anchor; +0.10 if 3+ do. Encourages high-consensus moments.
    two_detectors: float = 0.05
    three_plus_detectors: float = 0.10
    # +0.05 if any cluster member has face presence in its evidence
    # (the visual detector tags this when face_emotion ≠ None).
    face_visible: float = 0.05
    # +0.05 if any member reports a strong emotion (shock/laugh/smile)
    # or a motion spike — these correlate with high-engagement clips.
    strong_signal: float = 0.05


@dataclass(frozen=True)
class FusionConfig:
    """Top-level fusion configuration; threads into `chalybclip.config`."""

    cluster_window_s: float = 30.0
    """Candidates within this many seconds collapse into one cluster."""

    overlap_window_s: float = 10.0
    """For the overlap_bonus: how close detectors must fire to corroborate."""

    weights: FusionWeights = field(default_factory=FusionWeights)
    bonuses: FusionBonuses = field(default_factory=FusionBonuses)


# Reason → weight key. `CandidateReason` is a Literal so we keep this
# mapping explicit; if a new reason ever lands and is left out here,
# it contributes 0 (safe default; surfaces as a missing-detector bug).
_REASON_TO_WEIGHT_KEY: dict[str, str] = {
    "voice": "voice",
    "visual": "visual",
    "audio": "audio",
    "chat": "chat",
    "viral": "viral",
}


def fuse_candidates(
    candidates: Iterable[Candidate],
    *,
    config: FusionConfig | None = None,
) -> list[Candidate]:
    """Cluster, weight-fuse, and rank candidates.

    The output preserves the legacy `Candidate` shape so downstream
    code (cut, dashboard) keeps working. The fusion bookkeeping is
    surfaced under `evidence.fusion` for transparency.
    """
    cfg = config or FusionConfig()
    items = list(candidates)
    if not items:
        return []
    if cfg.cluster_window_s <= 0:
        return sorted(items, key=lambda c: c.timestamp)

    # ---- 1. Cluster by timestamp ----
    # Slice O.48 — voice triggers don't merge with other voice
    # triggers. Operator request: "for all the times the user says
    # the keyword to clip regardless of how many times he says it"
    # — every "clipea esto" should produce its own clip, not get
    # collapsed into the previous one because they happened within
    # cluster_window_s of each other.
    # Cross-signal clustering still applies (a voice trigger + a
    # visual spike + an audio peak within 30s = one richer clip with
    # all three as evidence), only voice-to-voice is blocked.
    sorted_items = sorted(items, key=lambda c: c.timestamp)
    clusters: list[list[Candidate]] = [[sorted_items[0]]]
    for c in sorted_items[1:]:
        prev_cluster = clusters[-1]
        within_window = (
            c.timestamp - prev_cluster[-1].timestamp <= cfg.cluster_window_s
        )
        # Block voice + voice in the same cluster.
        voice_voice_collision = (
            c.reason == "voice"
            and any(member.reason == "voice" for member in prev_cluster)
        )
        if within_window and not voice_voice_collision:
            prev_cluster.append(c)
        else:
            clusters.append([c])

    # ---- 2. Fuse each cluster ----
    fused: list[Candidate] = []
    for cluster in clusters:
        fused.append(_fuse_cluster(cluster, cfg=cfg))
    return fused


def _fuse_cluster(cluster: list[Candidate], *, cfg: FusionConfig) -> Candidate:
    """Compute the weighted score for one cluster + pick the best anchor."""
    # Per-detector best score: a chatty detector firing 5× shouldn't
    # dominate the sum — only its top score counts toward fusion.
    per_detector_best: dict[str, float] = {}
    for c in cluster:
        key = _REASON_TO_WEIGHT_KEY.get(c.reason, c.reason)
        if c.score > per_detector_best.get(key, 0.0):
            per_detector_best[key] = c.score

    # Transcript-hook is not its own CandidateReason yet (slice G.1 keeps
    # the surface minimal). When a future detector emits transcript-hook
    # candidates with `reason="voice"` and `evidence.kind="transcript_hook"`,
    # this is where we'd pick that out. For now: 0.
    transcript_hook_score = _peek_transcript_hook_score(cluster)

    weights = cfg.weights
    weighted = (
        weights.voice * per_detector_best.get("voice", 0.0)
        + weights.visual * per_detector_best.get("visual", 0.0)
        + weights.audio * per_detector_best.get("audio", 0.0)
        + weights.chat * per_detector_best.get("chat", 0.0)
        + weights.viral * per_detector_best.get("viral", 0.0)
        + weights.transcript_hook * transcript_hook_score
    )

    # ---- Pick the anchor timestamp ----
    # Score each cluster member by the number of OTHER distinct detector
    # types firing within ±overlap_window_s of it. Tiebreak by raw score
    # so a single dominant detector still picks its own timestamp.
    anchor = _pick_anchor(cluster, overlap_s=cfg.overlap_window_s)

    # ---- Overlap bonus around the anchor ----
    distinct_around_anchor = _distinct_detectors_within(
        cluster, t0=anchor.timestamp, window_s=cfg.overlap_window_s
    )
    if distinct_around_anchor >= 3:
        overlap_bonus = cfg.bonuses.three_plus_detectors
    elif distinct_around_anchor == 2:
        overlap_bonus = cfg.bonuses.two_detectors
    else:
        overlap_bonus = 0.0

    # ---- Evidence-driven bonuses ----
    face_bonus = (
        cfg.bonuses.face_visible if _any_face_present(cluster) else 0.0
    )
    strong_bonus = (
        cfg.bonuses.strong_signal if _any_strong_signal(cluster) else 0.0
    )

    final_score = min(1.0, max(0.0, weighted + overlap_bonus + face_bonus + strong_bonus))

    fusion_evidence: dict[str, Any] = {
        "per_detector_best": per_detector_best,
        "transcript_hook_score": transcript_hook_score,
        "weighted_sum": round(weighted, 4),
        "overlap_bonus": round(overlap_bonus, 4),
        "face_bonus": round(face_bonus, 4),
        "strong_signal_bonus": round(strong_bonus, 4),
        "distinct_detectors_at_anchor": distinct_around_anchor,
        "cluster_size": len(cluster),
    }

    return Candidate(
        timestamp=anchor.timestamp,
        score=round(final_score, 4),
        reason=anchor.reason,  # winning detector kind for routing/UI
        evidence={
            **anchor.evidence,
            "fusion": fusion_evidence,
            "matches": [c.evidence for c in cluster],
            "merged_count": len(cluster),
        },
    )


# ---- Anchor selection ----


def _pick_anchor(cluster: list[Candidate], *, overlap_s: float) -> Candidate:
    """Pick the cluster member whose timestamp has the most CORROBORATING
    distinct detectors within ±overlap_s. Ties → highest raw score."""
    if len(cluster) == 1:
        return cluster[0]
    best: Candidate | None = None
    best_distinct = -1
    best_score = -1.0
    for c in cluster:
        distinct = _distinct_detectors_within(
            cluster, t0=c.timestamp, window_s=overlap_s, exclude_idx=None
        )
        if distinct > best_distinct or (
            distinct == best_distinct and c.score > best_score
        ):
            best = c
            best_distinct = distinct
            best_score = c.score
    assert best is not None
    return best


def _distinct_detectors_within(
    cluster: list[Candidate],
    *,
    t0: float,
    window_s: float,
    exclude_idx: int | None = None,
) -> int:
    """Count distinct detector reasons firing within ±window_s of `t0`."""
    seen: set[str] = set()
    for i, c in enumerate(cluster):
        if exclude_idx is not None and i == exclude_idx:
            continue
        if abs(c.timestamp - t0) <= window_s:
            seen.add(c.reason)
    return len(seen)


# ---- Evidence-driven bonuses ----


def _any_face_present(cluster: list[Candidate]) -> bool:
    """True if any cluster member's evidence reports a face was visible.

    Two signals we trust:
      - `face_emotion`: visual detector sets this to a non-None label
        (smile/laugh/shock/anger/sad) when a face is detected
      - `face_presence`: clip-breakdown style — fraction of seconds with
        a face. Treat > 0 as "at least once present"
    """
    for c in cluster:
        ev = c.evidence or {}
        if ev.get("face_emotion"):
            return True
        fp = ev.get("face_presence")
        if isinstance(fp, int | float) and fp > 0:
            return True
    return False


_STRONG_EMOTION_LABELS = frozenset({"shock", "laugh", "smile"})


def _any_strong_signal(cluster: list[Candidate]) -> bool:
    """True if any cluster member has either a strong emotion label
    or a notable motion spike in its evidence."""
    for c in cluster:
        ev = c.evidence or {}
        if ev.get("face_emotion") in _STRONG_EMOTION_LABELS:
            return True
        kind = ev.get("kind")
        if isinstance(kind, str) and kind in {"motion_spike", "scene_cut"}:
            return True
        # `motion_score` from the breakdown — a normalized 0..1 reading.
        # Anything in the top quartile is "notable".
        ms = ev.get("motion_score")
        if isinstance(ms, int | float) and ms >= 0.75:
            return True
    return False


def _peek_transcript_hook_score(cluster: list[Candidate]) -> float:
    """Reserved hook for a future transcript-hook detector. Currently a
    no-op; checks if any candidate evidence already carries a key the
    pipeline might supply downstream so a future detector can plug in
    without changing fusion's signature.
    """
    for c in cluster:
        ev = c.evidence or {}
        score = ev.get("transcript_hook_score")
        if isinstance(score, int | float):
            return max(0.0, min(1.0, float(score)))
    return 0.0


__all__ = [
    "FusionBonuses",
    "FusionConfig",
    "FusionWeights",
    "fuse_candidates",
]
