"""Per-clip intelligence aggregator (slice F.7-C)."""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest_asyncio

from chalybclip.clip.intelligence import (
    _audio_peaks_from_segments,
    _chat_heat_spikes,
    _face_emotion_changes,
    _reactions_from_segments,
    _scene_cuts_and_motion_peaks,
    compute_intelligence,
)
from chalybclip.db import (
    CandidatesRepo,
    ClipsRepo,
    Database,
    StreamsRepo,
    TenantsRepo,
    TranscriptsRepo,
    apply_migrations,
)
from chalybclip.db.models import (
    CandidateRow,
    ClipRow,
    StreamRow,
    TranscriptRow,
)
from chalybclip.tenancy import bound_tenant


@pytest_asyncio.fixture
async def intel_db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "intel.db")
    try:
        await apply_migrations(d)
        yield d
    finally:
        await d.close()


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


# ---- transcript helpers (pure functions) ---------------------


def test_audio_peaks_emits_buckets_above_threshold() -> None:
    """Many quiet seconds + one bursty second → only the burst fires.
    The threshold is 1.8× the median bucket, so the test needs the
    median to be small (mostly silence) for a 6-word burst to clear."""
    segs = (
        # Background: many 1-word seconds (so median = 1, threshold = 2).
        [{"start": float(i), "end": float(i + 1), "text": "uh"} for i in range(10)]
        # Plus one bursty second well above the threshold.
        + [{"start": 5.0, "end": 6.0, "text": "OH MY GOD WHAT WAS THAT BRO"}]
    )
    peaks = _audio_peaks_from_segments(segs, clip_start=0, clip_end=20)
    assert any(p.kind == "audio_peak" for p in peaks)
    # The bursty bucket near ts=5 fires.
    assert any(4 <= p.ts <= 6 for p in peaks)
    assert all(0 <= p.score <= 1 for p in peaks)


def test_audio_peaks_filters_to_clip_window() -> None:
    """Segments before clip_start / after clip_end are ignored."""
    segs = [
        {"start": 0, "end": 1, "text": "EARLY EARLY EARLY EARLY EARLY"},  # outside
        {"start": 11, "end": 12, "text": "LATE LATE LATE LATE LATE"},     # outside
        {"start": 5, "end": 6, "text": "INSIDE BURST WORDS WORDS WORDS"}, # inside
    ]
    peaks = _audio_peaks_from_segments(segs, clip_start=2, clip_end=10)
    # Only the inside burst fires.
    for p in peaks:
        assert 0 <= p.ts <= 8  # relative to clip_start=2


def test_reactions_picks_up_laughter_tokens() -> None:
    segs = [
        {"start": 5, "end": 6, "text": "haha lmao what was that jajaja"},
        {"start": 7, "end": 8, "text": "no laughs here just talking"},
    ]
    reactions = _reactions_from_segments(segs, clip_start=0, clip_end=10)
    assert len(reactions) == 1
    assert reactions[0].kind == "reaction"
    assert "laugh" in reactions[0].label
    assert reactions[0].score > 0


def test_reactions_handles_punctuation_around_laughter() -> None:
    """`haha,` and `lmao!` and `JAJA?` should still register."""
    segs = [
        {"start": 1, "end": 2, "text": "haha! lmao, JAJA? rofl"},
    ]
    reactions = _reactions_from_segments(segs, clip_start=0, clip_end=5)
    assert reactions
    assert reactions[0].score > 0.5


def test_reactions_empty_segments_returns_empty_list() -> None:
    assert _reactions_from_segments([], clip_start=0, clip_end=10) == []


# ---- visual_signals helpers ----------------------------------


def test_scene_cuts_each_get_a_marker() -> None:
    rows = [
        {"ts_offset_s": 1.0, "scene_cut": True,  "motion_energy": 0.0, "face_emotion": None},
        {"ts_offset_s": 2.0, "scene_cut": False, "motion_energy": 0.5, "face_emotion": None},
        {"ts_offset_s": 3.0, "scene_cut": True,  "motion_energy": 0.0, "face_emotion": None},
    ]
    out = _scene_cuts_and_motion_peaks(rows, clip_start=0, clip_end=10)
    cuts = [m for m in out if m.kind == "scene_cut"]
    assert len(cuts) == 2
    # Marker ts is relative to clip_start.
    assert {round(m.ts) for m in cuts} == {1, 3}


def test_motion_peaks_fire_at_top_percentile() -> None:
    rows = [
        {"ts_offset_s": float(i), "scene_cut": False,
         "motion_energy": 0.1, "face_emotion": None}
        for i in range(10)
    ]
    rows.append(
        {"ts_offset_s": 10.0, "scene_cut": False,
         "motion_energy": 5.0, "face_emotion": None}  # huge spike
    )
    out = _scene_cuts_and_motion_peaks(rows, clip_start=0, clip_end=20)
    # Spike fires.
    spikes = [m for m in out if m.kind == "audio_peak" and "Motion" in m.label]
    assert spikes
    assert spikes[0].ts == 10.0


def test_face_emotion_changes_only_emit_on_transition() -> None:
    """3 seconds of 'neutral' + 'happy' + 'happy' → one marker."""
    rows = [
        {"ts_offset_s": 1.0, "scene_cut": False, "motion_energy": 0.0, "face_emotion": "neutral"},
        {"ts_offset_s": 2.0, "scene_cut": False, "motion_energy": 0.0, "face_emotion": "happy"},
        {"ts_offset_s": 3.0, "scene_cut": False, "motion_energy": 0.0, "face_emotion": "happy"},
        {"ts_offset_s": 4.0, "scene_cut": False, "motion_energy": 0.0, "face_emotion": "shock"},
    ]
    out = _face_emotion_changes(rows, clip_start=0, clip_end=10)
    # Three transitions (None→neutral → happy → shock); same-emotion repeats fold.
    assert len(out) == 3
    for m in out:
        assert m.kind == "face_emotion"
        assert "Face emotion" in m.label


# ---- chat-heat -----------------------------------------------


class _FakeMsg:
    """Duck-typed ChatMessage for the heat-spike test."""
    def __init__(self, ts: float):
        self.ts = ts


def test_chat_heat_spikes_fire_when_messages_burst() -> None:
    """Median bucket size = 1 msg/s, threshold is 2× the median (max 3).
    A bucket with 5 messages must fire."""
    msgs = (
        [_FakeMsg(1.0)]
        + [_FakeMsg(2.0)]
        + [_FakeMsg(3.0)] * 5  # spike
        + [_FakeMsg(4.0)]
        + [_FakeMsg(5.0)]
    )
    out = _chat_heat_spikes(msgs, clip_start=0, clip_end=10)
    assert out
    spike = out[0]
    assert spike.kind == "chat_heat"
    assert spike.ts == 3.0
    assert "5 messages" in spike.label


def test_chat_heat_no_messages_returns_empty() -> None:
    assert _chat_heat_spikes([], clip_start=0, clip_end=10) == []


# ---- compute_intelligence end-to-end ------------------------


async def _seed_for_intel(
    db: Database,
    *,
    tenant_id: str,
    transcript_segments: list[dict] | None = None,
) -> None:
    await TenantsRepo(db).create(tenant_id=tenant_id, name="A")
    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_i", tenant_id=tenant_id, vod_url="x",
                platform="kick", title="t", channel="c",
                duration_s=120.0,
                source_video_path="/tmp/v", source_audio_path="/tmp/a",
                status="ingested", created_at=_now(),
            )
        )
        await CandidatesRepo(db).upsert_many([
            CandidateRow(
                id="cnd_i", stream_id="str_i", tenant_id=tenant_id,
                ts=10.0, score=0.9, reason="voice", evidence={},
                created_at=_now(),
            ),
        ])
        await ClipsRepo(db).upsert_many([
            ClipRow(
                id="clp_i", stream_id="str_i", tenant_id=tenant_id,
                candidate_id="cnd_i",
                start_s=5.0, end_s=15.0, duration_s=10.0,
                width=1080, height=1920,
                path="/tmp/c.mp4",
                status="cut", created_at=_now(),
            ),
        ])
        if transcript_segments is not None:
            await TranscriptsRepo(db).upsert(
                TranscriptRow(
                    stream_id="str_i", tenant_id=tenant_id,
                    language="en", duration_s=120.0,
                    model="whisper-medium",
                    segments_json=json.dumps(transcript_segments),
                    created_at=_now(),
                )
            )


async def test_compute_intelligence_returns_empty_for_unknown_clip(
    intel_db: Database,
) -> None:
    await TenantsRepo(intel_db).create(tenant_id="aldo", name="A")
    with bound_tenant("aldo"):
        out = await compute_intelligence(intel_db, clip_id="clp_nope")
    assert out == []


async def test_compute_intelligence_returns_empty_when_no_signals(
    intel_db: Database,
) -> None:
    """Clip exists but no transcript / no visual_signals / no chat → []."""
    await _seed_for_intel(intel_db, tenant_id="aldo")
    with bound_tenant("aldo"):
        out = await compute_intelligence(intel_db, clip_id="clp_i")
    assert out == []


async def test_compute_intelligence_aggregates_transcript_signals(
    intel_db: Database,
) -> None:
    """When the transcript shows a laughter burst inside the clip's
    window, the markers list contains a reaction marker with ts
    relative to clip start."""
    segs = [
        {"start": 6.0, "end": 7.0, "text": "haha lmao what is that"},
        {"start": 8.0, "end": 9.0, "text": "OH MY GOD HE DID IT WHAT THE FUCK"},
    ]
    await _seed_for_intel(intel_db, tenant_id="aldo", transcript_segments=segs)
    with bound_tenant("aldo"):
        out = await compute_intelligence(intel_db, clip_id="clp_i")
    kinds = {m.kind for m in out}
    assert "reaction" in kinds
    # All marker ts are 0..clip.duration_s (= 10).
    for m in out:
        assert 0.0 <= m.ts <= 10.0
    # Sorted by ts.
    assert list(out) == sorted(out, key=lambda m: m.ts)
