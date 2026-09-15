"""Speaker identity resolution — match a VOD's diarization output against
the tenant's persistent `speakers` table.

Cosine-similarity match per (within-VOD) speaker. When the match clears
the configured threshold, the existing speaker is reused; otherwise a
new pending speaker row is auto-created so the operator can label it
later in the dashboard.

Threshold tuning lives in `DiarizationConfig.match_threshold` (default
0.75) and `DiarizationConfig.min_speech_for_id_s` (default 30) — speakers
with too little speech in this VOD are recorded as VOD-scoped rows but
NOT auto-matched (too little signal to risk a wrong merge).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import structlog

from chalybclip.config import DiarizationConfig
from chalybclip.db import Database, SpeakersRepo, VodSpeakersRepo

from .models import Diarization

_log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ResolutionOutcome:
    """Aggregate report from one resolution pass — surfaces in pipeline logs."""

    matched: int  # within-VOD speakers matched to an existing identity
    created: int  # new persistent speakers auto-created from this VOD
    unresolved: int  # speakers below min_speech_for_id_s, left pending


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Plain Python cosine similarity. Vectors are short (~192 floats) so
    even a naive loop is fast enough — no numpy dependency just for this."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _weighted_merge(
    old: list[float], old_weight: float, new: list[float], new_weight: float
) -> list[float]:
    """Duration-weighted average of two embeddings. Used when an existing
    speaker shows up in a new VOD — we fold the new evidence into their
    persistent fingerprint."""
    total = old_weight + new_weight
    if total <= 0.0:
        return list(old)
    return [
        (o * old_weight + n * new_weight) / total
        for o, n in zip(old, new, strict=True)
    ]


async def resolve_speakers(
    *,
    db: Database,
    stream_id: str,
    diarization: Diarization,
    config: DiarizationConfig,
) -> ResolutionOutcome:
    """For each speaker in `diarization`, find or create a persistent identity
    and persist the (stream, label, speaker_id) link in `vod_speakers`.

    Skippable: when diarization is itself skipped (no segments), this is a
    no-op that returns zeros. Safe to call unconditionally from the pipeline.
    """
    if diarization.skipped or not diarization.embeddings:
        return ResolutionOutcome(matched=0, created=0, unresolved=0)

    speakers_repo = SpeakersRepo(db)
    vod_speakers_repo = VodSpeakersRepo(db)
    known = await speakers_repo.list_for_tenant()

    # Idempotency (hard rule 4): a (stream, label) pair contributes to a
    # persistent fingerprint AT MOST once. On a re-run (cached diarization,
    # force, recovery re-dispatch) the labels resolved last time must not
    # fold their embedding in again — that double-counts total_speech_s and
    # drifts the fingerprint toward this one VOD.
    already_resolved = {
        r.speaker_label: r.resolved_speaker_id
        for r in await vod_speakers_repo.list_for_stream(stream_id)
        if r.resolved_speaker_id
    }

    matched = 0
    created = 0
    unresolved = 0

    for emb in diarization.embeddings:
        prior_id = already_resolved.get(emb.speaker_label)
        if prior_id is not None:
            matched += 1
            _log.info(
                "speaker.already_resolved",
                stream_id=stream_id,
                speaker_label=emb.speaker_label,
                speaker_id=prior_id,
            )
            continue
        # Score against every known speaker.
        best_id: str | None = None
        best_sim = 0.0
        best_old_total: float = 0.0
        best_old_vec: list[float] | None = None
        for s in known:
            if not s.embedding:
                continue
            sim = _cosine_sim(emb.embedding, s.embedding)
            if sim > best_sim:
                best_sim = sim
                best_id = s.id
                best_old_total = s.total_speech_s
                best_old_vec = s.embedding

        resolved_id: str | None = None
        confidence: float | None = None

        # Below min-speech: store the VOD-scoped row but don't auto-match.
        if emb.total_speech_s < config.min_speech_for_id_s:
            unresolved += 1
            await vod_speakers_repo.upsert(
                stream_id=stream_id,
                speaker_label=emb.speaker_label,
                resolved_speaker_id=None,
                confidence=best_sim if best_id else None,
                total_speech_s=emb.total_speech_s,
                embedding=emb.embedding,
            )
            _log.info(
                "speaker.unresolved_min_speech",
                stream_id=stream_id,
                speaker_label=emb.speaker_label,
                total_speech_s=emb.total_speech_s,
                min_required_s=config.min_speech_for_id_s,
            )
            continue

        if best_id is not None and best_sim >= config.match_threshold:
            # Existing identity wins. Fold the new embedding into their
            # persistent fingerprint (duration-weighted average).
            resolved_id = best_id
            confidence = best_sim
            assert best_old_vec is not None
            merged = _weighted_merge(
                best_old_vec, best_old_total, emb.embedding, emb.total_speech_s
            )
            await speakers_repo.update_embedding(
                speaker_id=best_id,
                embedding=merged,
                total_speech_s=best_old_total + emb.total_speech_s,
            )
            matched += 1
            _log.info(
                "speaker.matched",
                stream_id=stream_id,
                speaker_label=emb.speaker_label,
                speaker_id=best_id,
                cosine_sim=round(best_sim, 4),
            )
        else:
            # No match — auto-create a pending identity. Operator labels
            # it later via /speakers in the dashboard (slice E).
            new_speaker = await speakers_repo.create(
                display_name=f"Unknown ({emb.speaker_label})",
                embedding=emb.embedding,
                total_speech_s=emb.total_speech_s,
            )
            resolved_id = new_speaker.id
            confidence = best_sim if best_id else None
            created += 1
            _log.info(
                "speaker.created",
                stream_id=stream_id,
                speaker_label=emb.speaker_label,
                speaker_id=new_speaker.id,
                best_existing_sim=round(best_sim, 4),
            )

        await vod_speakers_repo.upsert(
            stream_id=stream_id,
            speaker_label=emb.speaker_label,
            resolved_speaker_id=resolved_id,
            confidence=confidence,
            total_speech_s=emb.total_speech_s,
            embedding=emb.embedding,
        )

    return ResolutionOutcome(
        matched=matched, created=created, unresolved=unresolved
    )
