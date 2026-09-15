"""Conversions between service models and DB row models.

Service models (`chalybclip.ingest.Stream`, `chalybclip.detect.Candidate`,
`chalybclip.clip.Clip`, etc.) are the in-memory shape that the pipeline
flows through. DB row models (`StreamRow`, `CandidateRow`, ...) are the
persistence shape with serialized JSON columns and audit fields. These
helpers keep the conversion logic in one place so adding a column to a
DB model doesn't require touching every service.

Candidate IDs are deterministic on `(stream_id, ts, reason)` so re-running
detect produces the same `cnd_<hash>` ids and INSERT OR IGNORE is a no-op.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json

from chalybclip.clip import Clip
from chalybclip.detect import Candidate
from chalybclip.ids import new_id
from chalybclip.ingest import Stream
from chalybclip.llm import Variant
from chalybclip.transcribe import Transcript

from .models import (
    CandidateRow,
    ClipRow,
    StreamRow,
    TranscriptRow,
    VariantRow,
)


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


def candidate_pk(stream_id: str, candidate: Candidate) -> str:
    """Deterministic ID for a candidate on `(stream_id, ts, reason)`.

    Re-running detect with the same inputs produces the same ID, so the
    UPSERT is idempotent and clip.candidate_id stays valid across resumes.
    """
    payload = f"{stream_id}:{candidate.timestamp:.6f}:{candidate.reason}"
    h = hashlib.blake2b(payload.encode("utf-8"), digest_size=10).hexdigest()
    return f"cnd_{h}"


def stream_to_row(stream: Stream, *, status: str = "ingested") -> StreamRow:
    return StreamRow(
        id=stream.id,
        tenant_id=stream.tenant_id,
        vod_url=stream.vod_url,
        platform=stream.platform,
        title=stream.title,
        channel=stream.channel,
        duration_s=stream.duration_s,
        source_video_path=str(stream.source_video_path),
        source_audio_path=str(stream.source_audio_path),
        status=status,
        created_at=_now(),
    )


def transcript_to_row(transcript: Transcript) -> TranscriptRow:
    """Serialize Whisper output. The full segments+words tree goes into
    `segments_json` so the dashboard can render word-level timestamps
    without re-running Whisper.
    """
    segments_payload = [seg.model_dump() for seg in transcript.segments]
    return TranscriptRow(
        stream_id=transcript.stream_id,
        tenant_id=transcript.tenant_id,
        language=transcript.language,
        duration_s=transcript.duration_s,
        model=transcript.model,
        segments_json=json.dumps(segments_payload, ensure_ascii=False),
        created_at=_now(),
    )


def candidate_to_row(candidate: Candidate, *, stream_id: str, tenant_id: str) -> CandidateRow:
    return CandidateRow(
        id=candidate_pk(stream_id, candidate),
        stream_id=stream_id,
        tenant_id=tenant_id,
        ts=candidate.timestamp,
        score=candidate.score,
        reason=candidate.reason,
        evidence=dict(candidate.evidence),
        created_at=_now(),
    )


def clip_to_row(clip: Clip) -> ClipRow:
    smart_box: dict[str, float] | None = None
    if clip.smart_crop_box is not None:
        smart_box = {
            "x": float(clip.smart_crop_box.x),
            "y": float(clip.smart_crop_box.y),
            "w": float(clip.smart_crop_box.w),
            "h": float(clip.smart_crop_box.h),
        }
    return ClipRow(
        id=clip.id,
        stream_id=clip.stream_id,
        tenant_id=clip.tenant_id,
        candidate_id=candidate_pk(clip.stream_id, clip.candidate),
        start_s=clip.start_s,
        end_s=clip.end_s,
        duration_s=clip.duration_s,
        width=clip.width,
        height=clip.height,
        path=str(clip.path),
        smart_crop_box=smart_box,
        thumbnail_frame_path=str(clip.thumbnail_path) if clip.thumbnail_path else None,
        status="cut",
        created_at=_now(),
    )


def variant_to_row(
    variant: Variant,
    *,
    clip_id: str,
    tenant_id: str,
    persona_id: str,
    model: str | None = None,
) -> VariantRow:
    """The LLM-provided `variant.id` (e.g. `v_1`) is a per-batch alias only;
    the DB row gets a fresh ULID so the variants PK is globally unique
    across clips and personas.
    """
    return VariantRow(
        id=new_id("var"),
        clip_id=clip_id,
        tenant_id=tenant_id,
        persona_id=persona_id,
        language=variant.language,
        caption=variant.caption,
        title_card_text=variant.title_card_text,
        hashtags=list(variant.hashtags),
        model=model,
        created_at=_now(),
    )
