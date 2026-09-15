"""Dynamic clip windowing — slice G.1.

Replaces the static `pre_roll_s = 30 / post_roll_s = 15` defaults from
`ClipConfig` with a per-candidate window that:

  1. Picks a target *duration band* based on the candidate's kind:
       - reaction      → 10-22s (humor / laugh / shock)
       - quote         → 12-25s (quotable / hot_take / controversial)
       - story         → 35-60s (drama / emotional / vulnerable)
       - retroactive   → ≤60s, anchored at the trigger (backward only)
       - default       → 30s pre-roll / 15s post-roll (legacy)

  2. Snaps start/end to clean boundaries using the transcript's segment
     punctuation. We never start in the middle of a sentence — start
     gets pulled BACK to the previous segment boundary inside the band.
     Similarly the end is extended FORWARD to the next segment boundary
     within tolerance.

  3. Falls back to the static `ClipConfig` numbers when transcript
     boundaries aren't available (e.g. no transcript, no overlap).

The output is a `WindowPlan` with the chosen `start_s` / `end_s` plus
a short `reason` string so the dashboard can show "started at sentence
boundary, ended after laugh peak".

Audio-energy-drop + scene-cut snapping are future work — the signature
of `plan_clip_window` is set up for them via optional inputs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from chalybclip.detect.models import Candidate
from chalybclip.transcribe import Transcript
from chalybclip.transcribe.models import Segment

WindowKind = Literal["reaction", "quote", "story", "retroactive", "default"]


@dataclass(frozen=True)
class WindowBand:
    """Target duration range for one window kind."""

    min_s: float
    max_s: float
    target_s: float  # initial target before boundary snapping


# Hard-coded but config-friendly — operators who want to tune the band
# can monkey-patch this dict or override via the future `ClipConfig.windows`.
WINDOW_BANDS: dict[WindowKind, WindowBand] = {
    "reaction":    WindowBand(min_s=10.0, max_s=22.0, target_s=15.0),
    "quote":       WindowBand(min_s=12.0, max_s=25.0, target_s=18.0),
    "story":       WindowBand(min_s=35.0, max_s=60.0, target_s=45.0),
    "retroactive": WindowBand(min_s=10.0, max_s=60.0, target_s=30.0),
    "default":     WindowBand(min_s=20.0, max_s=60.0, target_s=45.0),
}


# Viral-LLM `type` values → window kind. Maps the spec's "Add dynamic
# clip windowing" rules onto the categories the viral detector emits.
_VIRAL_TYPE_TO_KIND: dict[str, WindowKind] = {
    "humor":         "reaction",
    "reaction":      "reaction",
    "shock":         "reaction",
    "quotable":      "quote",
    "hot_take":      "quote",
    "controversial": "quote",
    "drama":         "story",
    "emotional":     "story",
    "vulnerable":    "story",
}


@dataclass(frozen=True)
class WindowPlan:
    """The chosen window with bookkeeping."""

    start_s: float
    end_s: float
    kind: WindowKind
    reason: str

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def classify_window_kind(candidate: Candidate) -> WindowKind:
    """Pick the window kind for a candidate based on its reason + evidence."""
    ev = candidate.evidence or {}

    # Retroactive voice triggers carry kind=retroactive in evidence.
    if ev.get("trigger_kind") == "retroactive":
        return "retroactive"

    # Viral-LLM candidates classify by the `type` field the LLM returned.
    viral_type = ev.get("type") or ev.get("viral_type")
    if isinstance(viral_type, str):
        kind = _VIRAL_TYPE_TO_KIND.get(viral_type.lower())
        if kind is not None:
            return kind

    # Visual + audio detectors without further context behave like reactions —
    # they fire on punchy single moments (cut + emotion or RMS spike).
    if candidate.reason in {"visual", "audio"}:
        return "reaction"

    # Chat-heat: spikes around big moments → quote-sized window so we
    # cover both the moment AND the in-stream context that caused it.
    if candidate.reason == "chat":
        return "quote"

    # Forward voice trigger or viral default: legacy "default" window.
    return "default"


def _lift_band(band: WindowBand, kind: WindowKind, ceiling: float) -> WindowBand:
    """Apply the monetization length lift to a SUBSTANTIVE band.

    When `ceiling` > 60s, raise the band's max_s to the ceiling and pull its
    target up toward it (so the clip actually clears 60s, not just *may*).
    Punchy bands (reaction/quote) and a default/no-lift ceiling are returned
    unchanged — keeping behavior identical when the operator hasn't opted in.
    """
    if ceiling <= 60.0 or kind not in ("story", "retroactive", "default"):
        return band
    return WindowBand(
        min_s=band.min_s,
        max_s=ceiling,
        target_s=min(ceiling, max(band.target_s, ceiling - 5.0)),
    )


def plan_clip_window(
    *,
    candidate: Candidate,
    transcript: Transcript | None,
    stream_duration_s: float,
    fallback_pre_roll_s: float = 30.0,
    fallback_post_roll_s: float = 15.0,
    max_clip_duration_s: float = 60.0,
) -> WindowPlan:
    """Pick the window kind, build the target window, snap to clean
    transcript boundaries, and clamp to the VOD bounds.

    `max_clip_duration_s` lifts the substantive bands so clips can run past
    60s (TikTok Creator Rewards eligibility). Default 60.0 = no change.
    """
    if stream_duration_s <= 0:
        return WindowPlan(
            start_s=0.0, end_s=0.0, kind="default", reason="non-positive stream duration"
        )

    kind = classify_window_kind(candidate)
    ceiling = max(60.0, float(max_clip_duration_s))
    band = _lift_band(WINDOW_BANDS[kind], kind, ceiling)
    ts = candidate.timestamp
    # When the operator lifts the ceiling, a default clip should reach the
    # monetization length too — extend its post-roll so its window targets the
    # lifted band instead of the legacy 30/15 = 45s. No lift → exact legacy.
    lifted = ceiling > 60.0 and kind in ("story", "retroactive", "default")

    # ---- 1. Initial target window per kind ----
    if kind == "retroactive":
        # Backward only: anchored at the trigger, look BACK up to
        # max_s. The trigger itself is the right-hand boundary.
        start_s = max(0.0, ts - band.target_s)
        end_s = min(stream_duration_s, ts)
        reason_parts: list[str] = [f"retroactive (≤{band.max_s:.0f}s)"]
    elif kind == "default" and not lifted:
        # Legacy 30/15 split — passes through the caller's overrides
        # so the existing ClipConfig fields still mean something.
        start_s = max(0.0, ts - fallback_pre_roll_s)
        end_s = min(stream_duration_s, ts + fallback_post_roll_s)
        reason_parts = [f"default {fallback_pre_roll_s:.0f}/{fallback_post_roll_s:.0f}s"]
    else:
        # Symmetric-ish around the trigger, biased toward more PRE-roll
        # so the viewer gets context before the payoff. The bias is the
        # band's target × 0.55 / 0.45 — punchy enough to feel snappy.
        pre = band.target_s * 0.55
        post = band.target_s * 0.45
        start_s = max(0.0, ts - pre)
        end_s = min(stream_duration_s, ts + post)
        reason_parts = [f"{kind} band {band.min_s:.0f}-{band.max_s:.0f}s"]

    # ---- 2. Snap to transcript segment boundaries ----
    if transcript is not None and transcript.segments:
        snapped_start, start_note = _snap_start_back(
            target_start=start_s, transcript=transcript, max_pull_s=band.max_s - (end_s - start_s)
        )
        snapped_end, end_note = _snap_end_forward(
            target_end=end_s, transcript=transcript, max_extend_s=band.max_s - (end_s - start_s)
        )
        if snapped_start != start_s:
            reason_parts.append(start_note)
            start_s = snapped_start
        if snapped_end != end_s:
            reason_parts.append(end_note)
            end_s = snapped_end

    # ---- 3. Enforce band min/max ----
    duration = end_s - start_s
    if kind != "default" or lifted:  # legacy default has no strict cap
        if duration < band.min_s:
            # Too short after snapping — pad to min by extending POST-side
            # (keep the start where the boundary was found).
            end_s = min(stream_duration_s, start_s + band.min_s)
            reason_parts.append(f"padded to min {band.min_s:.0f}s")
        elif duration > band.max_s:
            # Too long — trim the END (preserve the leading context we
            # carefully snapped to).
            end_s = start_s + band.max_s
            reason_parts.append(f"capped at {band.max_s:.0f}s")

    # ---- 4. Final clamps ----
    start_s = max(0.0, start_s)
    end_s = max(start_s, min(stream_duration_s, end_s))

    return WindowPlan(
        start_s=start_s,
        end_s=end_s,
        kind=kind,
        reason="; ".join(reason_parts),
    )


# ---- Boundary snapping helpers ----


def _snap_start_back(
    *,
    target_start: float,
    transcript: Transcript,
    max_pull_s: float,
) -> tuple[float, str]:
    """Pull `target_start` BACK to the start of the segment it lands in,
    so we never begin mid-sentence. Capped at `max_pull_s` extra
    seconds (so the band's max duration is respected).
    """
    if max_pull_s <= 0:
        return target_start, ""
    candidate_start = target_start
    candidate_note = ""
    for seg in _segments_overlapping(transcript, target_start):
        # If the segment's start is BEFORE target_start, we can pull back.
        if seg.ts < target_start and (target_start - seg.ts) <= max_pull_s:
            candidate_start = seg.ts
            candidate_note = "snapped start to sentence boundary"
            break
    return candidate_start, candidate_note


def _snap_end_forward(
    *,
    target_end: float,
    transcript: Transcript,
    max_extend_s: float,
) -> tuple[float, str]:
    """Extend `target_end` FORWARD to the next segment boundary so we
    don't end mid-sentence. Capped at `max_extend_s`.
    """
    if max_extend_s <= 0:
        return target_end, ""
    candidate_end = target_end
    candidate_note = ""
    for seg in transcript.segments:
        if seg.ts >= target_end:
            # The next segment starts AFTER target_end — that's the
            # cleanest cut: end the clip when the next speaker
            # utterance begins.
            extend_by = seg.ts - target_end
            if extend_by <= max_extend_s:
                candidate_end = seg.ts
                candidate_note = "snapped end to sentence boundary"
            break
        if seg.end_ts > target_end:
            # The segment we land in extends past target_end — finish
            # that segment cleanly if within budget.
            extend_by = seg.end_ts - target_end
            if extend_by <= max_extend_s:
                candidate_end = seg.end_ts
                candidate_note = "extended end to finish utterance"
            break
    return candidate_end, candidate_note


def _segments_overlapping(transcript: Transcript, t: float) -> Iterable[Segment]:
    """Yield segments whose [ts, end_ts] contains `t`, then segments
    earlier than `t` in reverse (used for pull-back lookup)."""
    forward = [s for s in transcript.segments if s.ts <= t <= s.end_ts]
    yield from forward
    earlier = sorted(
        (s for s in transcript.segments if s.end_ts <= t),
        key=lambda s: s.ts,
        reverse=True,
    )
    yield from earlier


__all__ = [
    "WINDOW_BANDS",
    "WindowBand",
    "WindowKind",
    "WindowPlan",
    "classify_window_kind",
    "plan_clip_window",
]
