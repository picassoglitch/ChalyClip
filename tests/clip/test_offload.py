"""Phase 2a — R2 offload of clip artifacts + rehydration.

`offload_clip_artifacts` mirrors each cut clip (mp4 + thumbnail) into the
object store after cut/auto-correct; `ensure_local_clip` pulls the bucket
copy back when a byte-needing path (render, waveform) finds the local file
reclaimed. Exercised against an in-memory fake store — no boto3.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from chalybclip.clip import ensure_local_clip, offload_clip_artifacts
from chalybclip.integrations.storage import (
    clip_key_family,
    clip_media_key,
    clip_render_key,
    clip_thumbnail_key,
)


class FakeArtifactStore:
    """In-memory ArtifactStore: key -> bytes."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.uploads: list[str] = []

    async def upload(
        self, *, local_path: Path, key: str, content_type: str | None = None
    ) -> None:
        self.objects[key] = Path(local_path).read_bytes()
        self.uploads.append(key)

    async def download(self, *, key: str, dest: Path) -> Path | None:
        if key not in self.objects:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.objects[key])
        return dest

    async def presigned_url(self, *, key: str, ttl_seconds: int) -> str:
        return f"https://bucket.example/{key}?exp={ttl_seconds}"

    def public_url(self, key: str) -> str | None:
        return None

    async def exists(self, *, key: str) -> bool:
        return key in self.objects

    async def delete(self, *, key: str) -> None:
        self.objects.pop(key, None)


def _clip(tmp_path: Path, clip_id: str, *, with_thumb: bool = True) -> object:
    clip_dir = tmp_path / "clips" / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    mp4 = clip_dir / "clip.mp4"
    mp4.write_bytes(b"mp4-" + clip_id.encode())
    thumb = clip_dir / "thumbnail.jpg"
    if with_thumb:
        thumb.write_bytes(b"jpg-" + clip_id.encode())
    # `thumbnail_path` is the pipeline Clip model's field name — the shape
    # cut_clips actually returns (the DB row calls it thumbnail_frame_path;
    # a dedicated test below covers that alias).
    return SimpleNamespace(
        id=clip_id,
        path=str(mp4),
        thumbnail_path=str(thumb) if with_thumb else None,
    )


async def test_offload_uploads_mp4_and_thumbnail(tmp_path: Path) -> None:
    store = FakeArtifactStore()
    clips = [_clip(tmp_path, "clp_1"), _clip(tmp_path, "clp_2")]

    n = await offload_clip_artifacts(store, tenant_id="ten_a", clips=clips)

    assert n == 4
    assert store.objects[clip_media_key("ten_a", "clp_1")] == b"mp4-clp_1"
    assert store.objects[clip_thumbnail_key("ten_a", "clp_2")] == b"jpg-clp_2"


async def test_offload_is_idempotent_unless_forced(tmp_path: Path) -> None:
    store = FakeArtifactStore()
    clips = [_clip(tmp_path, "clp_1")]
    assert await offload_clip_artifacts(store, tenant_id="t", clips=clips) == 2

    # Second run: both objects already in the bucket — nothing re-uploads.
    assert await offload_clip_artifacts(store, tenant_id="t", clips=clips) == 0
    assert len(store.uploads) == 2

    # force=True (a force re-cut produced new bytes) re-uploads.
    assert (
        await offload_clip_artifacts(store, tenant_id="t", clips=clips, force=True)
        == 2
    )


async def test_offload_accepts_db_row_thumbnail_alias(tmp_path: Path) -> None:
    """ClipRow-shaped objects name the same file `thumbnail_frame_path` —
    both spellings must offload (the prod bug this pins: only the mp4 was
    uploaded and every bucket thumbnail redirect 404'd)."""
    store = FakeArtifactStore()
    clip_dir = tmp_path / "clips" / "clp_9"
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip.mp4").write_bytes(b"mp4")
    (clip_dir / "thumbnail.jpg").write_bytes(b"jpg")
    row_shaped = SimpleNamespace(
        id="clp_9",
        path=str(clip_dir / "clip.mp4"),
        thumbnail_frame_path=str(clip_dir / "thumbnail.jpg"),
    )

    n = await offload_clip_artifacts(store, tenant_id="t", clips=[row_shaped])

    assert n == 2
    assert clip_thumbnail_key("t", "clp_9") in store.objects


def test_pipeline_clip_model_field_name_is_pinned() -> None:
    """The offload reads `thumbnail_path` off the objects cut_clips
    returns — fail loudly here if the model field is ever renamed."""
    from chalybclip.clip.models import Clip

    assert "thumbnail_path" in Clip.model_fields


async def test_offload_skips_missing_files_and_bad_clips(tmp_path: Path) -> None:
    store = FakeArtifactStore()
    no_thumb = _clip(tmp_path, "clp_1", with_thumb=False)
    ghost = SimpleNamespace(id="clp_2", path=str(tmp_path / "gone.mp4"),
                            thumbnail_frame_path=None)
    anonymous = SimpleNamespace(id="", path=None, thumbnail_frame_path=None)

    n = await offload_clip_artifacts(
        store, tenant_id="t", clips=[no_thumb, ghost, anonymous]
    )

    assert n == 1  # just clp_1's mp4
    assert list(store.objects) == [clip_media_key("t", "clp_1")]


async def test_offload_continues_past_a_failing_upload(tmp_path: Path) -> None:
    class ExplodingStore(FakeArtifactStore):
        async def upload(self, *, local_path: Path, key: str,
                         content_type: str | None = None) -> None:
            if "clp_1" in key:
                raise RuntimeError("R2 hiccup")
            await super().upload(
                local_path=local_path, key=key, content_type=content_type
            )

    store = ExplodingStore()
    clips = [_clip(tmp_path, "clp_1"), _clip(tmp_path, "clp_2")]

    n = await offload_clip_artifacts(store, tenant_id="t", clips=clips)

    # clp_1's two objects failed; clp_2's two landed. No raise.
    assert n == 2
    assert clip_media_key("t", "clp_2") in store.objects


async def test_ensure_local_clip_noop_when_file_present(tmp_path: Path) -> None:
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"here")
    # Store is None — must not matter when the file is already local.
    assert await ensure_local_clip(
        None, tenant_id="t", clip_id="clp_1", clip_path=p
    )


async def test_ensure_local_clip_rehydrates_from_bucket(tmp_path: Path) -> None:
    store = FakeArtifactStore()
    store.objects[clip_media_key("t", "clp_1")] = b"bucket-bytes"
    dest = tmp_path / "clips" / "clp_1" / "clip.mp4"

    assert await ensure_local_clip(
        store, tenant_id="t", clip_id="clp_1", clip_path=dest
    )
    assert dest.read_bytes() == b"bucket-bytes"


async def test_ensure_local_clip_false_when_unavailable(tmp_path: Path) -> None:
    dest = tmp_path / "clip.mp4"
    assert not await ensure_local_clip(
        None, tenant_id="t", clip_id="clp_1", clip_path=dest
    )
    assert not await ensure_local_clip(
        FakeArtifactStore(), tenant_id="t", clip_id="clp_1", clip_path=dest
    )


def test_key_family_covers_media_thumbnail_and_render() -> None:
    fam = clip_key_family("ten_a", "clp_1")
    assert clip_media_key("ten_a", "clp_1") in fam
    assert clip_thumbnail_key("ten_a", "clp_1") in fam
    assert clip_render_key("ten_a", "clp_1") in fam


def test_render_key_matches_publish_router_contract() -> None:
    # The publish path (api/routers/internal.py) has uploaded to this exact
    # key since Phase 1 — the shared builder must never drift from it.
    from chalybclip.api.routers.internal import artifact_key_for_clip

    assert (
        artifact_key_for_clip("ten_a", "clp_1")
        == "clips/ten_a/clp_1/clip_render_1080.mp4"
    )
