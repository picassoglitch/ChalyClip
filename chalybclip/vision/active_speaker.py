"""Heuristic active-speaker selection from mouth-region lip motion.

Given, per candidate face, a short window of mouth-region MOTION values
(frame-to-frame absolute pixel difference over the lower third of the face
box, sampled across consecutive frames) — and, optionally, a time-aligned
audio-energy envelope — decide which face is the one talking.

This is a HEURISTIC audio-visual speaker detector, NOT a learned model like
TalkNet or Light-ASD. It has no viseme/phoneme model and no face-voice
embedding; it bets on two cheap, robust signals:

  * lip motion        — a speaking mouth moves; a listening mouth is still;
  * audio-visual sync — the speaker's lip motion rises and falls together
    with the audio energy, an idle mouth does not.

`pick_active_speaker` is a PURE function — no OpenCV, no ffmpeg, no I/O, not
even numpy — so the decision maths is unit-tested in isolation. The
video/audio plumbing that produces its inputs lives in
`nexoclip.vision.face_track`.

Upgrade path
------------
Swap the body of `pick_active_speaker` (or feed it a per-candidate speaking
probability from Light-ASD / TalkNet) and the caller is unchanged: the
contract is "per-candidate mouth-motion windows in, chosen index out". The
frame-diff heuristic is kept dependency-free on purpose so it runs on the
operator's box with zero installs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _mean(xs: Sequence[float]) -> float:
    """Arithmetic mean; 0.0 for an empty sequence (never divides by zero)."""
    return sum(xs) / len(xs) if xs else 0.0


def motion_magnitude(window: Sequence[float]) -> float:
    """Mean mouth motion over a candidate's window (0.0 when empty).

    A plain mean, on purpose: a short window is already "recent", and
    keeping two equal-energy candidates genuinely tied lets the audio-sync
    term be the decider instead of an arbitrary recency bias.
    """
    return _mean(window)


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson correlation of two equal-length series, in [-1, 1].

    Returns 0.0 — "no sync evidence", never a ZeroDivisionError — when the
    series differ in length or either one is flat (zero variance). A length
    mismatch is expected, not an error: a freshly-appeared face has a
    shorter motion window than the audio envelope until it fills.
    """
    n = len(a)
    if n == 0 or n != len(b):
        return 0.0
    ma, mb = _mean(a), _mean(b)
    cov = sum(((x - ma) * (y - mb) for x, y in zip(a, b, strict=True)), 0.0)
    var_a = sum(((x - ma) ** 2 for x in a), 0.0)
    var_b = sum(((y - mb) ** 2 for y in b), 0.0)
    if var_a <= 0.0 or var_b <= 0.0:
        return 0.0
    return cov / (math.sqrt(var_a) * math.sqrt(var_b))


def score_candidate(
    motion: Sequence[float],
    audio: Sequence[float] | None,
    *,
    audio_weight: float,
    motion_scale: float,
) -> float:
    """Speaking score for one candidate.

    Two additive terms: the candidate's lip-motion magnitude (normalised by
    `motion_scale` — the frame's strongest mouth motion — so it lands in
    [0, 1]), plus, when audio is supplied, how positively that lip motion
    correlates with the audio envelope. The sync term is non-negative and
    additive, so audio can only ever RAISE a score: it breaks ties and
    rewards lip-sync, it never pulls a silent-but-moving mouth below its
    motion score.
    """
    magnitude = motion_magnitude(motion)
    magnitude_norm = magnitude / motion_scale if motion_scale > 0.0 else 0.0
    sync = max(0.0, _pearson(motion, audio)) if audio is not None else 0.0
    return magnitude_norm + audio_weight * sync


def pick_active_speaker(
    motion_per_candidate: Sequence[Sequence[float]],
    *,
    audio_energy: Sequence[float] | None = None,
    audio_weight: float = 0.5,
    min_motion: float = 0.0,
) -> int | None:
    """Index of the most-likely speaking candidate, or None.

    `motion_per_candidate[i]` is candidate *i*'s mouth-motion window
    (parallel across candidates, oldest -> newest). `audio_energy` is the
    time-aligned audio envelope over those same frames, or None to run
    lip-motion-only. `audio_weight` <= 1 keeps lip motion the primary
    signal and audio the tie-breaker.

    Returns:
      * ``None`` — no candidates, or every candidate's motion is below
        `min_motion` (nobody is visibly speaking, so the caller should fall
        back to its size/persistence heuristic);
      * ``0`` — exactly one candidate (nothing to disambiguate);
      * ``i`` — otherwise the argmax speaking score (ties resolve to the
        lowest index).
    """
    count = len(motion_per_candidate)
    if count == 0:
        return None
    if count == 1:
        return 0

    magnitudes = [motion_magnitude(w) for w in motion_per_candidate]
    strongest = max(magnitudes)
    if strongest < min_motion:
        return None  # no lip-motion evidence on any face — caller decides

    scale = strongest if strongest > 0.0 else 1.0
    best_index = 0
    best_score = float("-inf")
    for i, motion in enumerate(motion_per_candidate):
        score = score_candidate(
            motion, audio_energy, audio_weight=audio_weight, motion_scale=scale
        )
        if score > best_score:
            best_score = score
            best_index = i
    return best_index


def mouth_roi_box(
    x: int, y: int, w: int, h: int, *, frame_w: int, frame_h: int
) -> tuple[int, int, int, int]:
    """Lower-third-of-face mouth ROI as ``(x0, y0, x1, y1)``, frame-clamped.

    The mouth sits in the lower third of a frontal-face box; the ROI is also
    inset horizontally so cheek and background pixels don't swamp the lip
    signal. Returns a ZERO-AREA box (``x0 == x1`` or ``y0 == y1``) for a
    degenerate / out-of-frame face rect — the caller reads that as "no
    motion sample this frame".
    """
    inset = max(0, w // 6)
    x0 = max(0, min(frame_w, x + inset))
    x1 = max(0, min(frame_w, x + w - inset))
    y0 = max(0, min(frame_h, y + (2 * h) // 3))
    y1 = max(0, min(frame_h, y + h))
    if x1 < x0:
        x1 = x0
    if y1 < y0:
        y1 = y0
    return x0, y0, x1, y1


__all__ = [
    "motion_magnitude",
    "mouth_roi_box",
    "pick_active_speaker",
    "score_candidate",
]
