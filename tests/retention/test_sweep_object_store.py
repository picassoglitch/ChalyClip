"""Phase 2a — the retention sweep drops a clip's bucket objects with its row.

Without this, retention deletes the row + local files but the R2 copies
(clip.mp4 / thumbnail.jpg / publish render) become unreachable orphans that
accumulate forever.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import chalybclip.integrations.storage as storage_mod
from chalybclip.db import CandidatesRepo, ClipsRepo, Database, StreamsRepo, TenantsRepo
from chalybclip.db.models import CandidateRow, ClipRow, StreamRow
from chalybclip.integrations.storage import clip_key_family
from chalybclip.retention import sweep_retention
from chalybclip.tenancy import bound_tenant

from .conftest import days_ago_iso


class _RecordingStore:
    """Fake ArtifactStore that records deletes."""

    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def delete(self, *, key: str) -> None:
        self.deleted.append(key)


async def _seed_stream_with_clip(
    db: Database, *, tenant_id: str, output_dir: Path, clip_age_days: int
) -> str:
    stream_dir = output_dir / "str_1"
    (stream_dir / "source").mkdir(parents=True, exist_ok=True)
    video = stream_dir / "source" / "video.mp4"
    video.write_bytes(b"\x00" * 64)
    clip_dir = stream_dir / "clips" / "clp_1"
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / "clip.mp4"
    clip_path.write_bytes(b"\xff" * 64)

    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_1",
                tenant_id=tenant_id,
                vod_url="https://kick.com/x/videos/1",
                platform="kick",
                title="t",
                channel="c",
                duration_s=60.0,
                source_video_path=str(video),
                source_audio_path=str(video),
                status="done",
                created_at=days_ago_iso(1),  # stream inside VOD window
            )
        )
        await CandidatesRepo(db).upsert_many(
            [
                CandidateRow(
                    id="cnd_1",
                    stream_id="str_1",
                    tenant_id=tenant_id,
                    ts=10.0,
                    score=0.9,
                    reason="voice",
                    evidence={},
                    created_at=days_ago_iso(clip_age_days),
                )
            ]
        )
        await ClipsRepo(db).upsert_many(
            [
                ClipRow(
                    id="clp_1",
                    stream_id="str_1",
                    tenant_id=tenant_id,
                    candidate_id="cnd_1",
                    start_s=0.0,
                    end_s=10.0,
                    duration_s=10.0,
                    width=1080,
                    height=1920,
                    path=str(clip_path),
                    thumbnail_frame_path=str(clip_dir / "thumbnail.jpg"),
                    status="cut",
                    created_at=days_ago_iso(clip_age_days),
                )
            ]
        )
    return tenant_id


async def test_sweep_deletes_bucket_objects_for_expired_clips(
    retention_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _RecordingStore()
    monkeypatch.setattr(storage_mod, "build_artifact_store", lambda _s: store)

    t = await TenantsRepo(retention_db).create(name="Aldo")
    await _seed_stream_with_clip(
        retention_db, tenant_id=t.id, output_dir=tmp_path, clip_age_days=120
    )

    reports = await sweep_retention(retention_db, output_dir=tmp_path)

    assert reports[0].clips_deleted == 1
    assert sorted(store.deleted) == sorted(clip_key_family(t.id, "clp_1"))


async def test_sweep_leaves_bucket_alone_inside_window(
    retention_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _RecordingStore()
    monkeypatch.setattr(storage_mod, "build_artifact_store", lambda _s: store)

    t = await TenantsRepo(retention_db).create(name="Aldo")
    await _seed_stream_with_clip(
        retention_db, tenant_id=t.id, output_dir=tmp_path, clip_age_days=5
    )

    reports = await sweep_retention(retention_db, output_dir=tmp_path)

    assert reports[0].clips_deleted == 0
    assert store.deleted == []


async def test_dry_run_never_touches_the_bucket(
    retention_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _RecordingStore()
    built: list[bool] = []

    def _build(_s: object) -> _RecordingStore:
        built.append(True)
        return store

    monkeypatch.setattr(storage_mod, "build_artifact_store", _build)

    t = await TenantsRepo(retention_db).create(name="Aldo")
    await _seed_stream_with_clip(
        retention_db, tenant_id=t.id, output_dir=tmp_path, clip_age_days=120
    )

    reports = await sweep_retention(retention_db, output_dir=tmp_path, dry_run=True)

    assert reports[0].clips_deleted == 1  # reported, not acted on
    assert store.deleted == []
    assert built == []  # store never even constructed on dry runs
