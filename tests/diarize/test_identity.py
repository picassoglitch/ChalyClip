"""Embedding-match speaker identity resolution.

Cosine similarity is computed without numpy — these tests verify the
match/create/unresolved branches end-to-end against a real SQLite DB.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pytest

from chalybclip.config import DiarizationConfig
from chalybclip.db import (
    Database,
    SpeakersRepo,
    StreamsRepo,
    TenantsRepo,
    VodSpeakersRepo,
    apply_migrations,
)
from chalybclip.db.models import StreamRow
from chalybclip.diarize import resolve_speakers
from chalybclip.diarize.identity import _cosine_sim, _weighted_merge
from chalybclip.diarize.models import (
    Diarization,
    DiarizationSegment,
    SpeakerEmbedding,
)
from chalybclip.tenancy import bound_tenant


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


async def _seed(db: Database) -> str:
    """Create a tenant + stream, return the tenant_id and stream_id."""
    tenant = await TenantsRepo(db).create(name="Aldo")
    with bound_tenant(tenant.id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_idtest",
                tenant_id=tenant.id,
                vod_url="upload://x.mp4",
                platform="upload",
                title="t",
                channel=None,
                duration_s=120.0,
                source_video_path="/tmp/v.mp4",
                source_audio_path="/tmp/a.wav",
                status="ingested",
                created_at=_now(),
            )
        )
    return tenant.id


@pytest.fixture
async def migrated_db(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    await apply_migrations(db)
    yield db
    await db.close()


# ---- pure math ----


def test_cosine_sim_orthogonal_is_zero() -> None:
    assert _cosine_sim([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == 0.0


def test_cosine_sim_identical_is_one() -> None:
    assert _cosine_sim([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_sim_mismatched_lengths_returns_zero() -> None:
    assert _cosine_sim([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


def test_weighted_merge_averages_by_duration() -> None:
    # 60s of [1,1,1] folded with 30s of [4,4,4] → (1*60 + 4*30) / 90 = 2.0
    merged = _weighted_merge([1.0, 1.0, 1.0], 60.0, [4.0, 4.0, 4.0], 30.0)
    for v in merged:
        assert v == pytest.approx(2.0)


# ---- end-to-end resolution ----


async def test_unknown_speaker_creates_new_persistent_identity(
    migrated_db: Database,
) -> None:
    tenant_id = await _seed(migrated_db)
    with bound_tenant(tenant_id):
        diar = Diarization(
            stream_id="str_idtest",
            tenant_id=tenant_id,
            segments=[
                DiarizationSegment(ts=0.0, end_ts=60.0, speaker_label="SPEAKER_00")
            ],
            embeddings=[
                SpeakerEmbedding(
                    speaker_label="SPEAKER_00",
                    embedding=[0.1, 0.2, 0.3, 0.4],
                    total_speech_s=60.0,
                )
            ],
        )
        outcome = await resolve_speakers(
            db=migrated_db,
            stream_id="str_idtest",
            diarization=diar,
            config=DiarizationConfig(enabled=True, min_speech_for_id_s=30.0),
        )

        assert outcome.created == 1
        assert outcome.matched == 0
        assert outcome.unresolved == 0
        speakers = await SpeakersRepo(migrated_db).list_for_tenant()
        assert len(speakers) == 1
        assert speakers[0].display_name.startswith("Unknown")

        vsps = await VodSpeakersRepo(migrated_db).list_for_stream("str_idtest")
        assert len(vsps) == 1
        assert vsps[0].resolved_speaker_id == speakers[0].id


async def test_known_speaker_gets_matched_above_threshold(
    migrated_db: Database,
) -> None:
    """Same voice in a second VOD: similarity > threshold → matched."""
    tenant_id = await _seed(migrated_db)
    with bound_tenant(tenant_id):
        # Seed an existing speaker with embedding [1, 0, 0].
        existing = await SpeakersRepo(migrated_db).create(
            display_name="Aldo",
            is_self=True,
            embedding=[1.0, 0.0, 0.0],
            total_speech_s=600.0,
        )

        # New diarization: almost the same vector (sim ~ 0.998).
        diar = Diarization(
            stream_id="str_idtest",
            tenant_id=tenant_id,
            segments=[
                DiarizationSegment(ts=0.0, end_ts=120.0, speaker_label="SPEAKER_00")
            ],
            embeddings=[
                SpeakerEmbedding(
                    speaker_label="SPEAKER_00",
                    embedding=[0.99, 0.05, 0.05],
                    total_speech_s=120.0,
                )
            ],
        )
        outcome = await resolve_speakers(
            db=migrated_db,
            stream_id="str_idtest",
            diarization=diar,
            config=DiarizationConfig(
                enabled=True, match_threshold=0.75, min_speech_for_id_s=30.0
            ),
        )

        assert outcome.matched == 1
        assert outcome.created == 0
        # The persistent speaker's total_speech grew by the new VOD's contribution.
        updated = await SpeakersRepo(migrated_db).get(existing.id)
        assert updated is not None
        assert updated.total_speech_s == pytest.approx(720.0)
        # And the vod_speakers row points at the same persistent id.
        vsps = await VodSpeakersRepo(migrated_db).list_for_stream("str_idtest")
        assert vsps[0].resolved_speaker_id == existing.id
        assert (vsps[0].confidence or 0.0) > 0.9


async def test_below_min_speech_left_unresolved(
    migrated_db: Database,
) -> None:
    """Speakers with too little total speech don't auto-merge —
    too little signal to risk wrong-merging two voices."""
    tenant_id = await _seed(migrated_db)
    with bound_tenant(tenant_id):
        # Existing identity exists, but the new VOD speaker has only 5s of speech.
        await SpeakersRepo(migrated_db).create(
            display_name="Cano",
            embedding=[1.0, 0.0, 0.0],
            total_speech_s=300.0,
        )
        diar = Diarization(
            stream_id="str_idtest",
            tenant_id=tenant_id,
            segments=[
                DiarizationSegment(ts=10.0, end_ts=15.0, speaker_label="SPEAKER_01")
            ],
            embeddings=[
                SpeakerEmbedding(
                    speaker_label="SPEAKER_01",
                    embedding=[0.99, 0.05, 0.05],
                    total_speech_s=5.0,  # below threshold
                )
            ],
        )
        outcome = await resolve_speakers(
            db=migrated_db,
            stream_id="str_idtest",
            diarization=diar,
            config=DiarizationConfig(
                enabled=True, match_threshold=0.75, min_speech_for_id_s=30.0
            ),
        )
        assert outcome.unresolved == 1
        assert outcome.matched == 0
        assert outcome.created == 0
        vsps = await VodSpeakersRepo(migrated_db).list_for_stream("str_idtest")
        assert vsps[0].resolved_speaker_id is None


async def test_resolve_skipped_diarization_is_noop(migrated_db: Database) -> None:
    """When diarization was skipped (no token / pyannote missing), resolution
    must short-circuit cleanly — zero counters, no rows created."""
    tenant_id = await _seed(migrated_db)
    with bound_tenant(tenant_id):
        diar = Diarization(
            stream_id="str_idtest",
            tenant_id=tenant_id,
            skipped=True,
            skip_reason="HF_TOKEN not set",
        )
        outcome = await resolve_speakers(
            db=migrated_db,
            stream_id="str_idtest",
            diarization=diar,
            config=DiarizationConfig(enabled=True),
        )
        assert (outcome.matched, outcome.created, outcome.unresolved) == (0, 0, 0)
        assert len(await SpeakersRepo(migrated_db).list_for_tenant()) == 0


async def test_resolve_speakers_is_idempotent_on_rerun(
    migrated_db: Database,
) -> None:
    """Regression (hard rule 4): re-running the pipeline on the same stream
    must NOT fold the same VOD's embedding into the persistent fingerprint
    again — that double-counts total_speech_s and drifts the embedding."""
    tenant_id = await _seed(migrated_db)
    diar = Diarization(
        stream_id="str_idtest",
        tenant_id=tenant_id,
        segments=[
            DiarizationSegment(ts=0.0, end_ts=60.0, speaker_label="SPEAKER_00")
        ],
        embeddings=[
            SpeakerEmbedding(
                speaker_label="SPEAKER_00",
                embedding=[1.0, 0.0, 0.0],
                total_speech_s=60.0,
            )
        ],
        skipped=False,
    )
    with bound_tenant(tenant_id):
        first = await resolve_speakers(
            db=migrated_db,
            stream_id="str_idtest",
            diarization=diar,
            config=DiarizationConfig(),
        )
        assert first.created == 1

        speakers = await SpeakersRepo(migrated_db).list_for_tenant()
        assert len(speakers) == 1
        assert speakers[0].total_speech_s == pytest.approx(60.0)

        # Same stream, same diarization — e.g. cached diarization on a
        # pipeline re-run. Must be a no-op on the persistent fingerprint.
        second = await resolve_speakers(
            db=migrated_db,
            stream_id="str_idtest",
            diarization=diar,
            config=DiarizationConfig(),
        )
        assert second.created == 0
        assert second.matched == 1

        speakers = await SpeakersRepo(migrated_db).list_for_tenant()
        assert len(speakers) == 1
        assert speakers[0].total_speech_s == pytest.approx(60.0)  # not 120

        rows = await VodSpeakersRepo(migrated_db).list_for_stream("str_idtest")
        assert len(rows) == 1
