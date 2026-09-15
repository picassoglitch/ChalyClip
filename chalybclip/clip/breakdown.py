"""Confidence breakdown for a single clip — the "why this clip?" panel.

Joins three already-persisted tables for the bound tenant:
  * `visual_signals` -> motion_score (avg motion_energy across the clip's
    seconds) + face_presence (frac of seconds with a detected face).
  * `transcripts.segments_json` -> speaking_intensity (words per second
    during the clip's window).
  * `candidates` -> reaction_confidence (rescore_score) +
    rescore_delta (rescore_score - heuristic_score).

This is the observability hook your scope-adjusted PHASE_2 mandated:
without it, "why did the system pick THIS clip?" stays a guessing game.
The dashboard renders the dataclass; tests pin the math.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from chalybclip.db import (
    CandidatesRepo,
    ClipsRepo,
    Database,
    TranscriptsRepo,
    VisualSignalsRepo,
)
from chalybclip.errors import ClipError
from chalybclip.tenancy import current_tenant_id


@dataclass(frozen=True)
class ClipBreakdown:
    """All the "why" numbers the dashboard's panel renders."""

    clip_id: str
    duration_s: float
    motion_score: float | None  # avg motion_energy across the clip seconds
    face_presence: float | None  # fraction of seconds with a face detected
    speaking_intensity: float | None  # words per second over the clip window
    reaction_confidence: float | None  # candidates.rescore_score
    rescore_delta: float | None  # rescore_score - heuristic_score
    rescore_reason: str | None
    heuristic_reason: str  # candidates.reason (voice / chat / audio / visual)
    heuristic_score: float
    # Slice G.4 — stream integrity findings. Each issue is a structured
    # dict with timestamps (in source-stream coordinates) so the G.4b
    # auto-trim service can read them. Empty tuple = clip is clean.
    # Shape: {"kind": "freeze" | "silence", "start_s": float,
    #         "end_s": float, "label": str}.
    # The publishability scorer reads `.label`; the trim service reads
    # `.start_s` / `.end_s`. Detection is per-clip (not per-stream) so
    # a clean clip from a glitchy VOD still passes.
    #
    # MVP detects:
    #   - freeze frames  (consecutive seconds with motion_energy ~ 0)
    #   - silent gaps    (consecutive seconds with no spoken words)
    # Deferred (need extra pipeline passes):
    #   - bitrate drops (ffprobe at ingest)
    #   - PTS skip / desync (ffprobe -show_packets)
    integrity_issues: tuple[dict[str, object], ...] = ()


async def clip_breakdown(db: Database, clip_id: str) -> ClipBreakdown:
    """Compute the breakdown panel data for `clip_id` under the bound tenant.

    Raises `ClipError` if the clip doesn't exist (or belongs to another
    tenant - tenancy enforcement happens at the repo layer).
    """
    tenant_id = current_tenant_id()  # noqa: F841 - asserted via repo calls

    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise ClipError(f"clip not found: {clip_id}")

    # ---- visual signals: per-second rows in [floor(start_s), ceil(end_s)) ----
    visuals = await VisualSignalsRepo(db).list_for_stream(clip.stream_id)
    in_window = [
        v
        for v in visuals
        if clip.start_s <= float(v["ts_offset_s"]) < clip.end_s
    ]
    motion_score: float | None
    face_presence: float | None
    if in_window:
        motion_values = [
            float(v["motion_energy"])
            for v in in_window
            if v["motion_energy"] is not None
        ]
        motion_score = (
            sum(motion_values) / len(motion_values) if motion_values else None
        )
        face_count = sum(
            1 for v in in_window if v["face_emotion"] is not None
        )
        face_presence = face_count / len(in_window)
    else:
        motion_score = None
        face_presence = None

    # ---- transcript: count words whose ts falls in the clip window ----
    transcript = await TranscriptsRepo(db).get(clip.stream_id)
    speaking_intensity: float | None = None
    if transcript is not None:
        try:
            segments = json.loads(transcript.segments_json or "[]")
        except json.JSONDecodeError:
            segments = []
        word_count = 0
        for seg in segments:
            for word in seg.get("words") or []:
                ts = word.get("ts")
                if isinstance(ts, int | float) and clip.start_s <= ts < clip.end_s:
                    word_count += 1
        if clip.duration_s > 0:
            speaking_intensity = word_count / clip.duration_s

    # ---- candidate: rescore + heuristic_reason from the source candidate ----
    reaction_confidence: float | None = None
    rescore_delta: float | None = None
    rescore_reason: str | None = None
    heuristic_reason = "unknown"
    heuristic_score = 0.0
    if clip.candidate_id is not None:
        # `list_for_stream` filters by tenant + stream; we then pick the row
        # we want by id.
        candidates = await CandidatesRepo(db).list_for_stream(clip.stream_id)
        match = next((c for c in candidates if c.id == clip.candidate_id), None)
        if match is not None:
            heuristic_reason = match.reason
            heuristic_score = float(match.score)
            if match.rescore_score is not None:
                reaction_confidence = float(match.rescore_score)
                rescore_delta = float(match.rescore_score) - float(match.score)
            rescore_reason = match.rescore_reason

    # ---- integrity issues (slice G.4) ----
    # Per-clip detection: freeze (consecutive seconds with near-zero
    # motion) + silence (consecutive seconds with no spoken words).
    # Both run on data we already have on hand; no extra ffmpeg pass.
    integrity_issues = _detect_integrity_issues(
        clip_start_s=clip.start_s,
        clip_end_s=clip.end_s,
        visuals=in_window,
        transcript=transcript,
    )

    return ClipBreakdown(
        clip_id=clip_id,
        duration_s=clip.duration_s,
        motion_score=motion_score,
        face_presence=face_presence,
        speaking_intensity=speaking_intensity,
        reaction_confidence=reaction_confidence,
        rescore_delta=rescore_delta,
        rescore_reason=rescore_reason,
        heuristic_reason=heuristic_reason,
        heuristic_score=heuristic_score,
        integrity_issues=integrity_issues,
    )


# ---- integrity detectors (slice G.4) ----
#
# Both detectors look for *runs* of bad seconds rather than single
# blips — one-second motion dips are noise, but a 3-second freeze
# is real. Tuning constants are at the top of each function so we
# can re-tune from prod telemetry without code changes elsewhere.


def _detect_integrity_issues(
    *,
    clip_start_s: float,
    clip_end_s: float,
    visuals: list,  # type: ignore[type-arg]  # aiosqlite.Row is opaque
    transcript: object | None,
) -> tuple[dict[str, object], ...]:
    """Return structured issues, source-stream coordinate timestamps."""
    issues: list[dict[str, object]] = []
    issues.extend(_detect_freeze_frames(visuals=visuals))
    issues.extend(
        _detect_silent_gaps(
            clip_start_s=clip_start_s,
            clip_end_s=clip_end_s,
            transcript=transcript,
        )
    )
    return tuple(issues)


def _detect_freeze_frames(*, visuals: list) -> list[dict[str, object]]:  # type: ignore[type-arg]
    """Find runs of ≥3 consecutive seconds with motion_energy below 0.05.

    visual_signals already runs per-second motion analysis at ingest,
    so we just walk the rows in timestamp order. A "frozen" stream
    rarely has motion_energy of exactly 0 (camera noise, codec
    artifacts) so we use a small threshold instead of `== 0`.
    """
    MIN_RUN_S = 3
    THRESHOLD = 0.05
    if not visuals:
        return []
    rows = sorted(visuals, key=lambda v: float(v["ts_offset_s"]))
    out: list[dict[str, object]] = []
    run_start: float | None = None
    last_ts: float | None = None
    for v in rows:
        ts = float(v["ts_offset_s"])
        me = v["motion_energy"]
        is_frozen = me is not None and float(me) < THRESHOLD
        if is_frozen:
            if run_start is None:
                run_start = ts
            last_ts = ts
        else:
            if run_start is not None and last_ts is not None:
                end_s = last_ts + 1
                length = end_s - run_start
                if length >= MIN_RUN_S:
                    out.append({
                        "kind": "freeze",
                        "start_s": float(run_start),
                        "end_s": float(end_s),
                        "label": (
                            f"freeze: {int(length)}s zero-motion at "
                            f"{int(run_start)}-{int(end_s)}s"
                        ),
                    })
            run_start = None
            last_ts = None
    # Tail run
    if run_start is not None and last_ts is not None:
        end_s = last_ts + 1
        length = end_s - run_start
        if length >= MIN_RUN_S:
            out.append({
                "kind": "freeze",
                "start_s": float(run_start),
                "end_s": float(end_s),
                "label": (
                    f"freeze: {int(length)}s zero-motion at "
                    f"{int(run_start)}-{int(end_s)}s"
                ),
            })
    return out


def _detect_silent_gaps(
    *,
    clip_start_s: float,
    clip_end_s: float,
    transcript: object | None,
) -> list[dict[str, object]]:
    """Find runs of ≥5 seconds with no spoken words.

    Walks the transcript's word-level timestamps. A natural pause
    between sentences is shorter than 5s, so 5s+ usually means
    the audio actually dropped, the streamer muted, or whisper
    missed a stretch (which is also worth flagging — operator
    should listen back).
    """
    MIN_RUN_S = 5
    if transcript is None:
        return []
    try:
        segments = json.loads(getattr(transcript, "segments_json", None) or "[]")
    except json.JSONDecodeError:
        return []
    word_timestamps: list[float] = []
    for seg in segments:
        for word in seg.get("words") or []:
            ts = word.get("ts")
            if isinstance(ts, int | float) and clip_start_s <= float(ts) < clip_end_s:
                word_timestamps.append(float(ts))
    if not word_timestamps:
        # Whole clip silent — that's the dead-air signal already
        # surfaced by speaking_intensity. Don't double-flag.
        return []
    word_timestamps.sort()
    out: list[dict[str, object]] = []
    # Check leading silence (from clip_start_s to first word).
    if word_timestamps[0] - clip_start_s >= MIN_RUN_S:
        out.append({
            "kind": "silence",
            "start_s": float(clip_start_s),
            "end_s": float(word_timestamps[0]),
            "label": (
                f"silence: {int(word_timestamps[0] - clip_start_s)}s without speech "
                f"at {int(clip_start_s)}-{int(word_timestamps[0])}s"
            ),
        })
    # Gaps between adjacent words.
    for prev, cur in zip(word_timestamps, word_timestamps[1:], strict=False):
        gap = cur - prev
        if gap >= MIN_RUN_S:
            out.append({
                "kind": "silence",
                "start_s": float(prev),
                "end_s": float(cur),
                "label": (
                    f"silence: {int(gap)}s without speech at "
                    f"{int(prev)}-{int(cur)}s"
                ),
            })
    # Trailing silence (last word to clip_end_s).
    if clip_end_s - word_timestamps[-1] >= MIN_RUN_S:
        out.append({
            "kind": "silence",
            "start_s": float(word_timestamps[-1]),
            "end_s": float(clip_end_s),
            "label": (
                f"silence: {int(clip_end_s - word_timestamps[-1])}s without "
                f"speech at {int(word_timestamps[-1])}-{int(clip_end_s)}s"
            ),
        })
    return out
