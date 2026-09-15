"""Task A2 — derive Diarization from a transcript's speaker labels.

The adapter is the seam that lets the pipeline drop pyannote and
still feed every downstream consumer (detect.service,
persona/brand-per-speaker code) a `Diarization` object with the
same `.overlap_speaker(...)` / `.speaker_label_at(...)` API.
"""

from __future__ import annotations

from chalybclip.diarize import diarization_from_transcript
from chalybclip.transcribe.models import Segment, Transcript, Word


def _transcript(
    *,
    segments: list[Segment],
    speakers: list[str],
) -> Transcript:
    return Transcript(
        stream_id="str_t",
        tenant_id="ten_t",
        language="es",
        duration_s=segments[-1].end_ts if segments else 0.0,
        model="assemblyai-best",
        segments=segments,
        speakers=speakers,
    )


def _seg(ts: float, end_ts: float, text: str, speaker: str | None) -> Segment:
    return Segment(
        ts=ts, end_ts=end_ts, text=text, speaker=speaker,
        words=[Word(ts=ts, end_ts=end_ts, text=text, prob=0.9)],
    )


def test_returns_skipped_when_transcript_has_no_speakers() -> None:
    """Pre-AssemblyAI transcripts (or AAI with speaker_labels=false)
    carry no speakers — the adapter returns a skipped Diarization
    with an actionable reason, matching the pyannote-disabled path."""
    transcript = _transcript(
        segments=[_seg(0.0, 1.0, "hola", None)],
        speakers=[],
    )
    d = diarization_from_transcript(
        transcript, tenant_id="ten_t", stream_id="str_t",
    )
    assert d.skipped is True
    assert d.skip_reason is not None
    assert "speaker_labels" in d.skip_reason.lower()
    assert d.segments == []
    assert d.embeddings == []


def test_consecutive_same_speaker_segments_merge_into_one_turn() -> None:
    """pyannote emits contiguous turns; the adapter mirrors that by
    merging adjacent same-speaker AAI segments instead of producing
    one micro-turn per sentence."""
    transcript = _transcript(
        segments=[
            _seg(0.0, 1.0, "hola", "A"),
            _seg(1.0, 2.5, "mundo", "A"),
            _seg(2.5, 3.0, "qué tal", "B"),
            _seg(3.0, 5.0, "bien gracias", "B"),
        ],
        speakers=["A", "B"],
    )
    d = diarization_from_transcript(
        transcript, tenant_id="ten_t", stream_id="str_t",
    )
    assert d.skipped is False
    assert len(d.segments) == 2
    assert d.segments[0].speaker_label == "A"
    assert d.segments[0].ts == 0.0
    assert d.segments[0].end_ts == 2.5
    assert d.segments[1].speaker_label == "B"
    assert d.segments[1].ts == 2.5
    assert d.segments[1].end_ts == 5.0


def test_speaker_switch_starts_new_turn() -> None:
    """Alternating speakers stay as separate turns — no merge."""
    transcript = _transcript(
        segments=[
            _seg(0.0, 1.0, "a1", "A"),
            _seg(1.0, 2.0, "b1", "B"),
            _seg(2.0, 3.0, "a2", "A"),
        ],
        speakers=["A", "B"],
    )
    d = diarization_from_transcript(
        transcript, tenant_id="ten_t", stream_id="str_t",
    )
    assert [s.speaker_label for s in d.segments] == ["A", "B", "A"]


def test_unlabeled_segment_breaks_merge() -> None:
    """A segment with speaker=None (e.g. AssemblyAI skipped a noisy
    span) ends the open turn so the timeline gap stays visible."""
    transcript = _transcript(
        segments=[
            _seg(0.0, 1.0, "a1", "A"),
            _seg(1.0, 1.5, "<noise>", None),
            _seg(1.5, 2.5, "a2", "A"),
        ],
        speakers=["A"],
    )
    d = diarization_from_transcript(
        transcript, tenant_id="ten_t", stream_id="str_t",
    )
    # Two A turns, gap covers the unlabeled stretch.
    assert len(d.segments) == 2
    assert d.segments[0].end_ts == 1.0
    assert d.segments[1].ts == 1.5


def test_overlap_speaker_works_on_derived_diarization() -> None:
    """The detect service calls `diarization.overlap_speaker(ts, end_ts)`
    via duck-typing. Verify the derived Diarization implements that
    contract identically to pyannote's."""
    transcript = _transcript(
        segments=[
            _seg(10.0, 20.0, "a", "A"),
            _seg(20.0, 30.0, "b", "B"),
        ],
        speakers=["A", "B"],
    )
    d = diarization_from_transcript(
        transcript, tenant_id="ten_t", stream_id="str_t",
    )
    # Wholly inside A.
    assert d.overlap_speaker(12.0, 18.0) == "A"
    # Straddles A→B but more time on B.
    assert d.overlap_speaker(19.0, 26.0) == "B"


def test_no_embeddings_emitted() -> None:
    """Cross-video identity (resolve_speakers) needs embeddings;
    the transcribe-mode adapter doesn't have them. Caller must skip
    resolve_speakers in this mode."""
    transcript = _transcript(
        segments=[_seg(0.0, 1.0, "a", "A")],
        speakers=["A"],
    )
    d = diarization_from_transcript(
        transcript, tenant_id="ten_t", stream_id="str_t",
    )
    assert d.embeddings == []
