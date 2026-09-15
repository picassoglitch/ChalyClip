"""Tests for `cut_clips` with the ffmpeg subprocess stubbed out."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from chalybclip.clip import Clip, cut_clips, load_clips
from chalybclip.clip import service as clip_service
from chalybclip.config import ClipConfig
from chalybclip.errors import ClipError

from ._fixtures import make_candidate, seed_stream


def _stub_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Replace `_run_ffmpeg` with a recorder that creates the expected output file."""
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], *, what: str) -> None:
        calls.append(cmd)
        # The output path is always the last argument in our two builders.
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake mp4 bytes")

    monkeypatch.setattr(clip_service, "_run_ffmpeg", fake_run)
    return calls


def _stub_smart_crop_and_thumbnail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make smart_crop + thumbnail no-ops for tests with placeholder videos.

    The real CV libs can't decode the seed_stream's `b"\x00\x00fakevideo"`
    bytes; the production path is `_safe_smart_crop` / `_safe_thumbnail`
    which catch ClipError and skip — these stubs short-circuit that path
    so tests can assert on ffmpeg args without coupling to vision deps.
    """
    monkeypatch.setattr(clip_service, "_safe_smart_crop", lambda **_kw: None)
    monkeypatch.setattr(clip_service, "_safe_thumbnail", lambda **_kw: (None, None))


def test_cuts_one_clip_per_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_ffmpeg(monkeypatch)
    stream = seed_stream(tmp_path)
    candidates = [
        make_candidate(timestamp=120.0),
        make_candidate(timestamp=300.0, phrase="saca un clip"),
    ]

    clips = asyncio.run(
        cut_clips(
            tenant_id="default",
            stream=stream,
            candidates=candidates,
            output_dir=tmp_path,
        )
    )

    assert len(clips) == 2
    assert all(isinstance(c, Clip) for c in clips)
    # Two candidates × two ffmpeg passes (fast cut + reformat)
    assert len(calls) == 4

    # Manifest written at the stream root
    manifest_path = tmp_path / stream.id / "clips_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text("utf-8"))
    assert payload["stream_id"] == stream.id
    assert len(payload["clips"]) == 2

    # Per-clip layout
    for clip in clips:
        assert clip.id.startswith("clp_")
        assert clip.path.exists()
        assert clip.path.name == "clip.mp4"
        assert (clip.path.parent / "metadata.json").exists()
        assert clip.width == 1080 and clip.height == 1920


def test_ffmpeg_invocations_carry_correct_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_ffmpeg(monkeypatch)
    stream = seed_stream(tmp_path)
    candidate = make_candidate(timestamp=120.0)

    asyncio.run(
        cut_clips(
            tenant_id="default",
            stream=stream,
            candidates=[candidate],
            output_dir=tmp_path,
        )
    )

    fast_cut, reformat = calls
    # Fast cut: -ss BEFORE -i (fast seek) + -accurate_seek re-encode for a
    # frame-accurate trim (`-c copy` was dropped when accurate cutting
    # landed; audio re-encodes to AAC at the cut point).
    assert fast_cut[0] == "ffmpeg"
    assert "-ss" in fast_cut and "-accurate_seek" in fast_cut
    ss_idx = fast_cut.index("-ss")
    i_idx = fast_cut.index("-i")
    assert ss_idx < i_idx, "fast cut must put -ss before -i (keyframe-snap fast seek)"
    assert fast_cut[ss_idx + 1] == "90.000"
    assert fast_cut[fast_cut.index("-t") + 1] == "45.000"
    assert fast_cut[fast_cut.index("-c:a") + 1] == "aac"

    # Reformat: 9:16 crop + scale + libx264 + aac
    assert reformat[0] == "ffmpeg"
    vf_idx = reformat.index("-vf")
    assert "crop=ih*9/16:ih,scale=1080:1920" == reformat[vf_idx + 1]
    assert reformat[reformat.index("-c:v") + 1] == "libx264"
    assert reformat[reformat.index("-c:a") + 1] == "aac"
    assert reformat[reformat.index("-preset") + 1] == "fast"
    assert reformat[reformat.index("-crf") + 1] == "19"  # config default


def test_cut_clips_idempotent_on_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_ffmpeg(monkeypatch)
    stream = seed_stream(tmp_path)
    candidate = make_candidate(timestamp=60.0)

    first = asyncio.run(
        cut_clips(
            tenant_id="default",
            stream=stream,
            candidates=[candidate],
            output_dir=tmp_path,
        )
    )
    second = asyncio.run(
        cut_clips(
            tenant_id="default",
            stream=stream,
            candidates=[candidate],
            output_dir=tmp_path,
        )
    )
    assert second == first
    # No additional ffmpeg invocations on the second call.
    assert len(calls) == 2


def test_cut_clips_force_recuts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_ffmpeg(monkeypatch)
    stream = seed_stream(tmp_path)
    candidate = make_candidate(timestamp=60.0)

    asyncio.run(
        cut_clips(
            tenant_id="default",
            stream=stream,
            candidates=[candidate],
            output_dir=tmp_path,
        )
    )
    asyncio.run(
        cut_clips(
            tenant_id="default",
            stream=stream,
            candidates=[candidate],
            output_dir=tmp_path,
            force=True,
        )
    )
    # Second run does both ffmpeg passes again.
    assert len(calls) == 4


def test_cut_clips_skips_zero_duration_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anchor past the end of the stream produces no clip but no error."""
    _stub_ffmpeg(monkeypatch)
    stream = seed_stream(tmp_path)
    candidate = make_candidate(timestamp=stream.duration_s + 50.0)

    clips = asyncio.run(
        cut_clips(
            tenant_id="default",
            stream=stream,
            candidates=[candidate],
            output_dir=tmp_path,
        )
    )
    assert clips == []


def test_cut_clips_tenant_mismatch_raises(tmp_path: Path) -> None:
    stream = seed_stream(tmp_path, tenant_id="alice")
    with pytest.raises(ClipError, match="tenant mismatch"):
        asyncio.run(
            cut_clips(
                tenant_id="bob",
                stream=stream,
                candidates=[],
                output_dir=tmp_path,
            )
        )


def test_cut_clips_missing_video_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_ffmpeg(monkeypatch)
    stream = seed_stream(tmp_path)
    stream.source_video_path.unlink()
    with pytest.raises(ClipError, match="source video missing"):
        asyncio.run(
            cut_clips(
                tenant_id="default",
                stream=stream,
                candidates=[make_candidate(timestamp=10.0)],
                output_dir=tmp_path,
            )
        )


def test_cut_clips_uses_custom_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_ffmpeg(monkeypatch)
    stream = seed_stream(tmp_path)
    candidate = make_candidate(timestamp=120.0)
    cfg = ClipConfig(
        pre_roll_s=10.0,
        post_roll_s=5.0,
        output_width=720,
        output_height=1280,
        encoder="libx265",
        preset="medium",
        crf=20,
    )

    clips = asyncio.run(
        cut_clips(
            tenant_id="default",
            stream=stream,
            candidates=[candidate],
            output_dir=tmp_path,
            config=cfg,
        )
    )
    assert clips[0].duration_s == pytest.approx(15.0)
    assert clips[0].width == 720 and clips[0].height == 1280

    fast_cut, reformat = calls
    assert fast_cut[fast_cut.index("-ss") + 1] == "110.000"
    assert fast_cut[fast_cut.index("-t") + 1] == "15.000"
    vf_idx = reformat.index("-vf")
    assert reformat[vf_idx + 1] == "crop=ih*9/16:ih,scale=720:1280"
    assert reformat[reformat.index("-c:v") + 1] == "libx265"


def test_load_clips_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_ffmpeg(monkeypatch)
    stream = seed_stream(tmp_path)
    asyncio.run(
        cut_clips(
            tenant_id="default",
            stream=stream,
            candidates=[make_candidate(timestamp=60.0)],
            output_dir=tmp_path,
        )
    )
    manifest = load_clips(tmp_path / stream.id)
    assert manifest.stream_id == stream.id
    assert len(manifest.clips) == 1


def test_branded_thumbnails_are_persisted_when_pickable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slice D.4: when `_safe_thumbnail` returns real JPEG bytes, the
    branded compositor runs and the variant paths are persisted on the
    Clip + in manifest.json."""
    import io
    from types import SimpleNamespace

    from PIL import Image

    _stub_ffmpeg(monkeypatch)
    monkeypatch.setattr(clip_service, "_safe_smart_crop", lambda **_kw: None)
    # Hand back a real JPEG so the branded compositor has something
    # decodable — but skip the raw save (Path=None).
    src = Image.new("RGB", (1920, 1080), (40, 40, 40))
    buf = io.BytesIO()
    src.save(buf, format="JPEG", quality=85)
    jpeg = buf.getvalue()
    monkeypatch.setattr(
        clip_service, "_safe_thumbnail", lambda **_kw: (None, jpeg)
    )

    # Minimal duck-typed brand kit — the compositor reads attributes
    # via getattr, no DB row needed.
    fake_kit = SimpleNamespace(
        primary_color="#FF3366",
        accent_color="#FFD700",
        text_color="#FFFFFF",
        handle_tiktok="@aara_art",
        handle_youtube=None,
        handle_instagram=None,
        handle_kick=None,
    )
    stream = seed_stream(tmp_path)
    candidate = make_candidate(timestamp=60.0)
    clips = asyncio.run(
        cut_clips(
            tenant_id="default",
            stream=stream,
            candidates=[candidate],
            output_dir=tmp_path,
            brand_kits=[fake_kit],
        )
    )
    assert len(clips) == 1
    c = clips[0]
    assert c.thumbnail_16x9_path is not None
    assert c.thumbnail_9x16_path is not None
    assert c.thumbnail_1x1_path is not None
    assert c.thumbnail_16x9_path.exists()
    assert c.thumbnail_16x9_path.name == "thumb_16x9.jpg"

    # Manifest serialization includes the new paths.
    manifest_path = tmp_path / stream.id / "clips_manifest.json"
    payload = json.loads(manifest_path.read_text("utf-8"))
    persisted = payload["clips"][0]
    assert persisted["thumbnail_16x9_path"]
    assert persisted["thumbnail_9x16_path"]
    assert persisted["thumbnail_1x1_path"]


def test_no_brand_kit_still_renders_neutral_branded_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When `brand_kits=None`, the compositor uses neutral grey defaults
    so the publisher still has aspect-correct thumbnails."""
    import io

    from PIL import Image

    _stub_ffmpeg(monkeypatch)
    monkeypatch.setattr(clip_service, "_safe_smart_crop", lambda **_kw: None)
    src = Image.new("RGB", (1920, 1080), (200, 200, 200))
    buf = io.BytesIO()
    src.save(buf, format="JPEG", quality=85)
    jpeg = buf.getvalue()
    monkeypatch.setattr(
        clip_service, "_safe_thumbnail", lambda **_kw: (None, jpeg)
    )

    stream = seed_stream(tmp_path)
    clips = asyncio.run(
        cut_clips(
            tenant_id="default",
            stream=stream,
            candidates=[make_candidate(timestamp=60.0)],
            output_dir=tmp_path,
        )
    )
    assert clips[0].thumbnail_16x9_path is not None


def test_branded_thumbnails_absent_when_raw_pick_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the raw frame pick fails (returns None bytes), the branded
    step is skipped and the Clip fields stay None."""
    _stub_ffmpeg(monkeypatch)
    _stub_smart_crop_and_thumbnail(monkeypatch)
    stream = seed_stream(tmp_path)
    clips = asyncio.run(
        cut_clips(
            tenant_id="default",
            stream=stream,
            candidates=[make_candidate(timestamp=60.0)],
            output_dir=tmp_path,
        )
    )
    c = clips[0]
    assert c.thumbnail_16x9_path is None
    assert c.thumbnail_9x16_path is None
    assert c.thumbnail_1x1_path is None
