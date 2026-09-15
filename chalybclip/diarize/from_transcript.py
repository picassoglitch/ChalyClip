"""Task A2 — derive a Diarization from a transcript's per-utterance
speaker labels (AssemblyAI's `speaker_labels=true` output).

When the operator opts into `detection.diarization.source="transcribe"`,
the pipeline skips the GPU-bound pyannote pass entirely and calls
`diarization_from_transcript()` after the transcribe step to materialize
a Diarization with the same shape every downstream consumer
(detect.service, persona/brand-per-speaker code, dashboard surfaces)
already reads.

What we lose by going through transcribe-speakers instead of pyannote:
  * Embeddings — AssemblyAI doesn't emit a voice fingerprint. The
    cross-video `resolve_speakers` step needs embeddings to match
    a new VOD's "speaker A" against the tenant's persistent identity,
    so we skip resolve_speakers when this adapter is in play. Speaker
    labels are PER-VIDEO only.
  * Sub-utterance precision — pyannote can split a single sentence
    into two turns when two speakers overlap; AssemblyAI assigns the
    whole utterance to one speaker. Acceptable for v1; the overlap
    case is rare in single-host content.

What stays the same:
  * `diarization.overlap_speaker(ts, end_ts)` works identically.
  * `diarization.speaker_label_at(timestamp)` works identically.
  * Downstream candidate.evidence['speaker_label'] keeps reading the
    same field, so persona/brand selection on a per-speaker basis is
    unchanged.

TODO (cross-video identity): re-add embeddings via a Modal job that
runs pyannote-embedding on the audio file, AFTER we know whether
the cross-video speaker UX actually drives conversion. Until then
this adapter ships the per-video labels alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .models import Diarization, DiarizationSegment

if TYPE_CHECKING:
    from chalybclip.transcribe.models import Transcript


def diarization_from_transcript(
    transcript: "Transcript",
    *,
    tenant_id: str,
    stream_id: str,
) -> Diarization:
    """Build a Diarization out of a Transcript's per-utterance
    speakers.

    Behavior:
      * Empty `transcript.speakers` → returns Diarization(skipped=True,
        reason="transcript has no speaker labels") so downstream code
        treats it the same way as a disabled pyannote step.
      * Otherwise consecutive segments with the same speaker get merged
        into one DiarizationSegment turn — matches pyannote's
        contiguous-turn output instead of one micro-turn per sentence.
      * No embeddings (AssemblyAI doesn't expose them); `embeddings`
        stays empty. resolve_speakers must be skipped at the call site.
    """
    if not transcript.speakers:
        return Diarization(
            stream_id=stream_id,
            tenant_id=tenant_id,
            segments=[],
            embeddings=[],
            skipped=True,
            skip_reason=(
                "transcript carries no speaker labels — diarization "
                "source=transcribe needs the transcribe provider to "
                "emit utterance-level speaker info "
                "(AssemblyAI speaker_labels=true)."
            ),
        )

    # Merge consecutive same-speaker segments. We walk the segments
    # in order; whenever the speaker changes, close the open turn
    # and open a new one. Segments without a speaker label break the
    # merge — they're emitted as None turns so the gap stays visible
    # in the diarization timeline.
    segments: list[DiarizationSegment] = []
    open_turn: DiarizationSegment | None = None

    for seg in transcript.segments:
        if not seg.speaker:
            # Close any open turn before the unlabeled gap.
            if open_turn is not None:
                segments.append(open_turn)
                open_turn = None
            continue

        if open_turn is None or open_turn.speaker_label != seg.speaker:
            # Close prior turn.
            if open_turn is not None:
                segments.append(open_turn)
            open_turn = DiarizationSegment(
                ts=seg.ts,
                end_ts=seg.end_ts,
                speaker_label=seg.speaker,
            )
        else:
            # Extend the open turn's end.
            open_turn = open_turn.model_copy(update={"end_ts": seg.end_ts})

    if open_turn is not None:
        segments.append(open_turn)

    return Diarization(
        stream_id=stream_id,
        tenant_id=tenant_id,
        segments=segments,
        embeddings=[],
        skipped=False,
        skip_reason=None,
    )


__all__ = ["diarization_from_transcript"]
