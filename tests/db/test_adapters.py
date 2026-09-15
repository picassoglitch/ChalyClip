"""Tests for service-model -> DB-row conversion helpers."""

from __future__ import annotations

import json
from pathlib import Path

from chalybclip.clip import Clip
from chalybclip.db.adapters import (
    candidate_pk,
    candidate_to_row,
    clip_to_row,
    stream_to_row,
    transcript_to_row,
    variant_to_row,
)
from chalybclip.detect import Candidate
from chalybclip.ingest import Stream
from chalybclip.llm import Variant
from chalybclip.transcribe import Segment, Transcript, Word


def _stream(tenant: str = "ten_a") -> Stream:
    return Stream(
        id="str_01ABC",
        tenant_id=tenant,
        vod_url="https://kick.com/c/videos/1",
        platform="kick",
        title="t",
        channel="c",
        duration_s=600.0,
        source_video_path=Path("/out/str_01ABC/source/video.mp4"),
        source_audio_path=Path("/out/str_01ABC/source/audio.wav"),
    )


def test_stream_to_row_round_trips_fields() -> None:
    s = _stream()
    row = stream_to_row(s)
    assert row.id == s.id
    assert row.tenant_id == s.tenant_id
    assert row.platform == s.platform
    assert row.duration_s == s.duration_s
    # Path -> str
    assert row.source_video_path == str(s.source_video_path)
    assert row.status == "ingested"
    assert row.created_at  # not empty


def test_candidate_pk_is_deterministic() -> None:
    """Same (stream_id, ts, reason) -> same id; rerunning detect is idempotent."""
    c1 = Candidate(timestamp=120.5, score=0.9, reason="voice", evidence={"phrase": "x"})
    c2 = Candidate(timestamp=120.5, score=0.7, reason="voice", evidence={"phrase": "y"})
    # Different score/evidence but same (ts, reason) -> same pk.
    assert candidate_pk("str_A", c1) == candidate_pk("str_A", c2)
    # Different reason -> different pk.
    c3 = Candidate(timestamp=120.5, score=0.9, reason="chat", evidence={})
    assert candidate_pk("str_A", c1) != candidate_pk("str_A", c3)
    # Different stream -> different pk.
    assert candidate_pk("str_A", c1) != candidate_pk("str_B", c1)


def test_candidate_pk_has_cnd_prefix() -> None:
    c = Candidate(timestamp=10.0, score=1.0, reason="voice", evidence={})
    assert candidate_pk("str_X", c).startswith("cnd_")


def test_candidate_to_row_carries_evidence() -> None:
    c = Candidate(
        timestamp=120.5,
        score=0.9,
        reason="voice",
        evidence={"phrase": "clipéalo", "language": "es"},
    )
    row = candidate_to_row(c, stream_id="str_A", tenant_id="ten_a")
    assert row.id.startswith("cnd_")
    assert row.stream_id == "str_A"
    assert row.tenant_id == "ten_a"
    assert row.ts == 120.5
    assert row.evidence == {"phrase": "clipéalo", "language": "es"}


def test_clip_to_row_links_candidate_pk() -> None:
    candidate = Candidate(timestamp=120.0, score=0.9, reason="voice", evidence={})
    clip = Clip(
        id="clp_01ABC",
        tenant_id="ten_a",
        stream_id="str_A",
        candidate=candidate,
        start_s=90.0,
        end_s=135.0,
        duration_s=45.0,
        width=1080,
        height=1920,
        path=Path("/out/str_A/clips/clp_01ABC/clip.mp4"),
    )
    row = clip_to_row(clip)
    assert row.id == "clp_01ABC"
    assert row.candidate_id == candidate_pk("str_A", candidate)
    assert row.path == str(clip.path)


def test_variant_to_row_uses_fresh_ulid() -> None:
    """LLM's variant.id is a per-batch alias; DB row gets a real ULID."""
    v = Variant(
        id="v_1",
        language="es",
        caption="captura este momento",
        title_card_text="MOMENTO",
        hashtags=["clip"],
    )
    row = variant_to_row(
        v, clip_id="clp_A", tenant_id="ten_a", persona_id="aldo_villanueva"
    )
    assert row.id != "v_1"
    assert row.id.startswith("var_")
    assert row.clip_id == "clp_A"
    assert row.persona_id == "aldo_villanueva"
    assert row.caption == v.caption
    assert row.hashtags == ["clip"]


def test_variant_to_row_two_calls_distinct_ids() -> None:
    """Same variant input twice => two distinct DB rows (replace-on-write)."""
    v = Variant(id="v_1", language="es", caption="x", title_card_text="", hashtags=[])
    a = variant_to_row(v, clip_id="c", tenant_id="t", persona_id="p")
    b = variant_to_row(v, clip_id="c", tenant_id="t", persona_id="p")
    assert a.id != b.id


def test_transcript_to_row_serializes_segments() -> None:
    t = Transcript(
        stream_id="str_A",
        tenant_id="ten_a",
        language="es",
        duration_s=10.5,
        model="medium",
        segments=[
            Segment(
                ts=0.0,
                end_ts=2.0,
                text="hola",
                words=[Word(ts=0.0, end_ts=1.0, text="hola", prob=0.95)],
            )
        ],
    )
    row = transcript_to_row(t)
    assert row.stream_id == "str_A"
    assert row.language == "es"
    assert row.model == "medium"
    payload = json.loads(row.segments_json)
    assert payload[0]["text"] == "hola"
    assert payload[0]["words"][0]["text"] == "hola"
