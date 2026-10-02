"""LLM-based viral-moment detector.

Reads the full transcript, asks Claude (or whichever provider the router
picks for `viral_detection`) to identify the moments most likely to go
viral on TikTok / Reels / YouTube Shorts. Returns one Candidate per moment.

This is the universal detector — it works on any content, not just streams
where the host explicitly says 'clipéalo'. Particularly useful for:

  * Controversy / hot takes (the user noticed these go viral)
  * Emotional peaks (laughter, anger, shock)
  * Quotable one-liners that work standalone
  * Drama / interpersonal conflict

Costs one LLM call per stream (not per moment), so it's cheap even on long
VODs. Falls back gracefully — if the call fails or the budget governor
trips, returns an empty list and the pipeline continues with whatever the
heuristic detectors found.
"""

from __future__ import annotations

import math
from typing import Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field

from chalybclip.config import ViralConfig
from chalybclip.errors import BudgetExceeded, DetectionError, LLMError
from chalybclip.ingest import Stream
from chalybclip.llm import LLMRouter
from chalybclip.llm.config import Quality
from chalybclip.transcribe import Transcript
from chalybclip.transcribe.models import Word

from .models import Candidate

_log = structlog.get_logger(__name__)

_PURPOSE = "viral_detection"

ViralType = Literal[
    "controversial",
    "emotional",
    "quotable",
    "hot_take",
    "drama",
    "humor",
    "shock",
    "vulnerable",
    "other",
]


class ViralMoment(BaseModel):
    """One moment the LLM thinks could go viral."""

    model_config = ConfigDict(extra="forbid")

    timestamp_s: float = Field(
        ge=0.0,
        description=(
            "Exact start of the clip in VOD seconds: the start time of the "
            "first word of the hook line."
        ),
    )
    duration_s: float = Field(
        ge=1.0,
        le=120.0,
        description=(
            "Exact clip length in seconds, so timestamp_s + duration_s is the "
            "end time of the last word of the payoff. Typically 8-60s."
        ),
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Viral potential 0-1. Reserve 0.8+ for bangers — strong opinion, "
            "high emotion, or genuinely shocking content."
        ),
    )
    reason: str = Field(
        min_length=4,
        max_length=240,
        description="One-line plain-English explanation for a human reviewer.",
    )
    type: ViralType = Field(description="Tag describing why this moment works.")
    transcript_snippet: str = Field(
        max_length=600,
        description="The actual quote(s) that make this moment, copied verbatim.",
    )


class ViralMomentList(BaseModel):
    """Top moments, ranked. Empty list when nothing is genuinely viral."""

    model_config = ConfigDict(extra="forbid")

    moments: list[ViralMoment] = Field(default_factory=list)


_SYSTEM_PROMPT = """\
You are a short-form video editor. The transcript below comes from a \
livestream VOD or from a creator's own recorded video. Find the clips most \
likely to go viral on TikTok, Instagram Reels and YouTube Shorts, and that \
also work when cross-posted to all of them: the kind of moment that makes a \
scroller stop and a friend share it.

Things that work:
  * Hot takes / strong opinions — especially counter-narrative or controversial
  * Emotional peaks — anger, shock, vulnerability, infectious laughter
  * Quotable one-liners that punch standalone, no setup needed
  * Practical value — a tip, a reveal, a "nobody tells you this" insight
  * Drama / conflict — between speakers, with audience, with a topic
  * Shocking revelations or genuinely unexpected statements
  * Moments where the speaker is clearly fired up or breaking script

Do NOT pick:
  * Dead air, transitions, sign-on / sign-off, sponsor reads
  * Generic banter, meta-commentary about the stream or the recording itself
  * Low-effort content the speaker is clearly phoning in
  * Moments without a clear hook in the first 3 seconds

Each transcript line carries its time in seconds — "[start-end] text" per \
sentence, or "[HH:MM:SS] (start s) text" per 30s block on very long VODs. \
Use those times to place every clip as exactly as you can:
  * timestamp_s = the start time of the first word of the hook. Start ON the \
hook — no warm-up, no "okay so", no breath before it.
  * duration_s = from timestamp_s to the end time of the last word of the \
payoff. Stop right after the payoff lands; never run into the next topic, a \
pause, or the speaker reaching to stop the recording.
  * The clip must make sense on its own to someone who never saw the rest.
  * Recorded videos often contain retakes: the speaker says a line, stops, \
and says it again. Never include both attempts — pick the cleanest, most \
energetic take and place the clip around that one only.

Give each strong idea its own clip. You MAY return two clips that overlap \
when they are genuinely different cuts — e.g. a punchy 8-20s hook-only cut \
AND a fuller 25-60s cut with the context — since both get posted. Never \
return two near-identical windows.

For each clip include:
  * timestamp_s and duration_s as above
  * score: 0.0-1.0 viral potential. Be selective. Reserve 0.8+ for bangers.
  * reason: one line explaining why this would go viral
  * type: classify the trigger from {controversial, emotional, quotable, \
hot_take, drama, humor, shock, vulnerable, other}
  * transcript_snippet: the actual words, copied verbatim from the transcript

Return the clips ranked by score. If the transcript doesn't have genuine \
viral content, return fewer clips or an empty list. Do NOT pad.
"""


def _format_transcript(transcript: Transcript, *, window_s: float = 30.0) -> str:
    """Group transcript segments into ~30s windows with [HH:MM:SS] timestamps.

    Whisper-medium emits a segment every couple seconds; sending each one
    individually wastes tokens and makes the LLM choose pointless granular
    starts. 30-second windows match the typical short-form clip length and
    keep the prompt under reasonable budget for hour-long VODs.
    """
    if not transcript.segments:
        return "(empty transcript)"
    lines: list[str] = []
    window_start: float | None = None
    window_text: list[str] = []
    for seg in transcript.segments:
        if window_start is None:
            window_start = seg.ts
        if seg.ts - window_start >= window_s and window_text:
            lines.append(_fmt_window(window_start, window_text))
            window_start = seg.ts
            window_text = []
        # The Segment model exposes the text via `.text` (preferred) or
        # joined word texts as fallback.
        text = getattr(seg, "text", None) or " ".join(w.text for w in seg.words)
        window_text.append(text.strip())
    if window_start is not None and window_text:
        lines.append(_fmt_window(window_start, window_text))
    return "\n".join(lines)


# Above this length the per-sentence listing gets token-heavy; long VODs
# fall back to the 30s windows (moment placement is then refined by the
# word-level snapping in clip windowing).
_SENTENCE_FORMAT_MAX_S = 90 * 60

# A line break happens at sentence punctuation, at a pause this long, or
# once a line spans this many seconds — whichever comes first.
_SENTENCE_GAP_S = 0.7
_SENTENCE_MAX_S = 15.0


def _format_for_llm(transcript: Transcript, *, duration_s: float) -> str:
    """Per-sentence "[start-end] text" lines so the model can place clip
    edges on exact words; 30s windows for very long VODs."""
    words = [w for seg in transcript.segments for w in seg.words]
    if not words or duration_s > _SENTENCE_FORMAT_MAX_S:
        return _format_transcript(transcript)
    lines: list[str] = []
    line: list[Word] = []
    for w in words:
        if line and (
            w.ts - line[-1].end_ts >= _SENTENCE_GAP_S
            or w.end_ts - line[0].ts > _SENTENCE_MAX_S
        ):
            lines.append(_fmt_sentence(line))
            line = []
        line.append(w)
        if w.text.rstrip().endswith((".", "?", "!", "…")):
            lines.append(_fmt_sentence(line))
            line = []
    if line:
        lines.append(_fmt_sentence(line))
    return "\n".join(lines)


def _fmt_sentence(words: list[Word]) -> str:
    text = " ".join(w.text.strip() for w in words).strip()
    return f"[{words[0].ts:.2f}-{words[-1].end_ts:.2f}] {text}"


def _dedupe_near_identical(moments: list[ViralMoment]) -> list[ViralMoment]:
    """Drop a moment whose window is nearly the same as a higher-ranked one
    (IoU above 0.7). A short hook cut inside a longer story cut has low IoU,
    so both survive — that pairing is deliberate."""
    kept: list[ViralMoment] = []
    for m in moments:
        a0, a1 = m.timestamp_s, m.timestamp_s + m.duration_s
        duplicate = False
        for k in kept:
            b0, b1 = k.timestamp_s, k.timestamp_s + k.duration_s
            inter = max(0.0, min(a1, b1) - max(a0, b0))
            union = max(a1, b1) - min(a0, b0)
            if union > 0 and inter / union > 0.7:
                duplicate = True
                break
        if not duplicate:
            kept.append(m)
    return kept


def _fmt_window(ts: float, parts: list[str]) -> str:
    h = int(ts // 3600)
    m = int((ts % 3600) // 60)
    s = int(ts % 60)
    stamp = f"[{h:02d}:{m:02d}:{s:02d}]"
    return f"{stamp} ({ts:.1f}s) {' '.join(parts).strip()}"


async def detect_viral_moments(
    *,
    tenant_id: str,
    stream: Stream,
    transcript: Transcript,
    router: LLMRouter,
    config: ViralConfig,
) -> list[Candidate]:
    """Return LLM-scored viral candidates for `stream`. Empty list when disabled
    or the call fails — never raises into the pipeline."""
    if not config.enabled:
        return []
    if tenant_id != stream.tenant_id:
        raise DetectionError(
            f"tenant mismatch: caller={tenant_id!r}, stream={stream.tenant_id!r}"
        )
    if tenant_id != transcript.tenant_id:
        raise DetectionError(
            f"tenant mismatch: caller={tenant_id!r}, "
            f"transcript={transcript.tenant_id!r}"
        )

    formatted = _format_for_llm(transcript, duration_s=stream.duration_s)
    quality: Quality = "premium" if config.quality == "premium" else "standard"

    try:
        response = await router.complete(
            tenant_id=tenant_id,
            purpose=_PURPOSE,
            system=_SYSTEM_PROMPT,
            user=formatted,
            schema=ViralMomentList,
            quality=quality,
        )
    except BudgetExceeded as e:
        _log.warning("viral.skipped_budget", reason=str(e), stream_id=stream.id)
        return []
    except LLMError as e:
        _log.warning("viral.skipped_llm_error", reason=str(e), stream_id=stream.id)
        return []

    # Scale the cap with video length so a longer VOD isn't truncated to a
    # fixed handful: short-form cadence is ~1 clip/minute. The LLM is told
    # not to pad and the min_score gate trims below, so this rarely binds —
    # it just stops a 2-hour stream from being capped at the short-video
    # default. No artificial ceiling; the detector keeps whatever it finds.
    duration_cap = math.ceil(max(1.0, stream.duration_s / 60.0))
    effective_max = max(config.max_moments, duration_cap)
    moments = response.moments[:effective_max]
    moments = [m for m in moments if m.score >= config.min_score]
    moments = _dedupe_near_identical(
        sorted(moments, key=lambda m: m.score, reverse=True)
    )
    moments.sort(key=lambda m: m.timestamp_s)

    candidates: list[Candidate] = []
    for m in moments:
        candidates.append(
            Candidate(
                timestamp=m.timestamp_s,
                score=config.weight * m.score,
                reason="viral",
                evidence={
                    "viral_score": m.score,
                    "viral_type": m.type,
                    "reason": m.reason,
                    "transcript_snippet": m.transcript_snippet,
                    "estimated_duration_s": m.duration_s,
                    # Exact clip edges picked by the model — clip windowing
                    # cuts on these (snapped to word boundaries) instead of
                    # padding a band around `timestamp`.
                    "start_s": m.timestamp_s,
                    "end_s": m.timestamp_s + m.duration_s,
                },
            )
        )
    _log.info(
        "viral.detected",
        stream_id=stream.id,
        count=len(candidates),
        scores=[round(c.score, 3) for c in candidates],
    )
    return candidates
