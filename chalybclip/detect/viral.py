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

    timestamp_s: float = Field(ge=0.0, description="Start of the moment in VOD seconds.")
    duration_s: float = Field(
        ge=1.0,
        le=120.0,
        description="Estimated length of the interesting bit, typically 5-60s.",
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
You are analyzing a livestream transcript to find the moments most likely \
to go viral on TikTok, Instagram Reels, and YouTube Shorts. You are looking \
for the kind of moments that make a scroller stop and a friend share with \
another friend.

Score on 0.0-1.0 for VIRAL POTENTIAL. Things that work:
  * Hot takes / strong opinions — especially counter-narrative or controversial
  * Emotional peaks — anger, shock, vulnerability, infectious laughter
  * Quotable one-liners that punch standalone, no setup needed
  * Drama / conflict — between speakers, with audience, with a topic
  * Shocking revelations or genuinely unexpected statements
  * Moments where the speaker is clearly fired up or breaking script

Do NOT pick:
  * Dead air, transitions, sign-on / sign-off, sponsor reads
  * Generic banter, meta-commentary about the stream itself
  * Low-effort content the speaker is clearly phoning in
  * Moments without a clear hook in the first 3 seconds

For each moment include:
  * timestamp_s: where it starts in the VOD (use the [HH:MM:SS] markers below)
  * duration_s: how long the interesting bit lasts, 5-60s typically
  * score: 0.0-1.0. Be selective. Reserve 0.8+ for genuine bangers.
  * reason: one line explaining why this would go viral
  * type: classify the trigger from {controversial, emotional, quotable, \
hot_take, drama, humor, shock, vulnerable, other}
  * transcript_snippet: the actual words, copied verbatim from the transcript

Return the top moments ranked by score. If the transcript doesn't have \
genuine viral content, return fewer moments or an empty list. Do NOT pad.
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

    formatted = _format_transcript(transcript)
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
