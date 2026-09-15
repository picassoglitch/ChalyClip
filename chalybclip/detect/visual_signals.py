"""Visual signal fusion — turns the per-second VisualSignalTrack from
`analyze-video` into Candidates.

A second is "clip-worthy" visually when at least one of:
    1. `scene_cut` — something in the frame just changed
    2. an emotion *transition* — neutral/None this second's predecessor,
       a strong emotion (smile / laugh / shock) this second
    3. a motion *spike* — current motion energy > spike_ratio * the
       rolling baseline over the prior `motion_baseline_window_s`

We use edge-triggered emotion (transitions, not steady states) because a
multi-second smile would otherwise emit one Candidate per second; the
clip-worthy moment is the start of the smile, not the middle of it.

Each firing second emits one Candidate with reason="visual"; the score
is `weight * (cut + emotion + motion) / max_total`, capped at `weight`.
The fusion step in `detect_candidates` then clusters adjacent visual
seconds with each other (and with voice/chat/audio) via merge_window_s.
"""

from __future__ import annotations

from collections import deque

from chalybclip.config import VisualConfig
from chalybclip.errors import DetectionError
from chalybclip.ingest import Stream
from chalybclip.vision import VisualSignalTrack

from .models import Candidate


def detect_visual_candidates(
    tenant_id: str,
    stream: Stream,
    track: VisualSignalTrack,
    config: VisualConfig,
) -> list[Candidate]:
    """Return visual-fusion candidates for `stream`.

    Returns `[]` when the detector is disabled or there are no signals.
    Raises on tenant or stream mismatches (CLAUDE.md hard rule #1).
    """
    if not config.enabled or not track.signals:
        return []
    if tenant_id != stream.tenant_id:
        raise DetectionError(f"tenant mismatch: caller={tenant_id!r}, stream={stream.tenant_id!r}")
    if tenant_id != track.tenant_id:
        raise DetectionError(f"tenant mismatch: caller={tenant_id!r}, track={track.tenant_id!r}")
    if stream.id != track.stream_id:
        raise DetectionError(f"stream/track mismatch: stream={stream.id} track={track.stream_id}")

    strong_emotions = set(config.emotion_labels)
    max_total = config.cut_weight + config.emotion_weight + config.motion_weight
    if max_total <= 0:
        return []

    # Rolling motion baseline via deque of the prior window's energies.
    motion_window: deque[tuple[float, float]] = deque()
    motion_sum = 0.0

    # Edge-triggered emotion: only fire when we transition INTO a strong
    # label from a non-strong one (or from None).
    prev_emotion_strong = False

    candidates: list[Candidate] = []
    for sig in track.signals:
        # Update rolling motion baseline (uses values BEFORE this second).
        cutoff = sig.ts_offset_s - config.motion_baseline_window_s
        while motion_window and motion_window[0][0] < cutoff:
            _, dropped = motion_window.popleft()
            motion_sum -= dropped

        sub_scores: dict[str, float] = {}

        if sig.scene_cut:
            sub_scores["scene_cut"] = config.cut_weight

        is_strong = sig.face_emotion in strong_emotions
        if is_strong and not prev_emotion_strong:
            sub_scores["emotion"] = config.emotion_weight
        prev_emotion_strong = is_strong

        if sig.motion_energy is not None and len(motion_window) > 0:
            # Mean over the actual prior samples (not the window's nominal
            # length) — a partially-filled window otherwise produces an
            # artificially low baseline that flags every early sample.
            baseline = motion_sum / len(motion_window)
            if baseline > 0.0 and sig.motion_energy > config.motion_spike_ratio * baseline:
                ratio = sig.motion_energy / baseline
                sub_scores["motion"] = config.motion_weight * min(
                    1.0, ratio / config.motion_spike_ratio
                )

        # Add this second to the window AFTER computing baseline so the
        # current frame doesn't bias its own threshold.
        if sig.motion_energy is not None:
            motion_window.append((sig.ts_offset_s, sig.motion_energy))
            motion_sum += sig.motion_energy

        if sub_scores:
            total = sum(sub_scores.values())
            score = config.weight * min(1.0, total / max_total)
            candidates.append(
                Candidate(
                    timestamp=sig.ts_offset_s,
                    score=score,
                    reason="visual",
                    evidence={
                        "scene_cut": bool(sig.scene_cut),
                        "face_emotion": sig.face_emotion,
                        "motion_energy": sig.motion_energy,
                        "sub_scores": sub_scores,
                    },
                )
            )

    return candidates
