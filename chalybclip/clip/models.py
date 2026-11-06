"""Pydantic schemas for the clip module."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from chalybclip.detect import Candidate


class SmartCropBox(BaseModel):
    """Pixel-space crop window over the source frame.

    `(x, y)` is the top-left corner; `(w, h)` is the crop size in source
    pixels. Phase 1's smart_crop chooses these so faces stay in frame
    after the 9:16 reformat.
    """

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)


# Below this total horizontal travel (source pixels) a "tracking" path is
# indistinguishable from a static crop, so it collapses to the cheaper
# static box and produces byte-identical output. ~24px ≈ a barely-visible
# pan on a 1080p-wide source.
_MIN_DYNAMIC_TRAVEL_PX = 24


class ReframeKeyframe(BaseModel):
    """One (time, crop-left-x) anchor on a dynamic reframe path.

    `t_s` is CLIP-relative seconds (0.0 = clip start), matching the cut
    clip's own timeline that the ffmpeg crop expression reads via `t`.
    `x` is the crop window's LEFT edge in SOURCE pixels.
    """

    t_s: float = Field(ge=0.0)
    x: int = Field(ge=0)


class ReframeTrack(BaseModel):
    """A moving 9:16 crop window that follows the active subject.

    Where `SmartCropBox` holds one crop for the whole clip, a
    ReframeTrack carries a time-ordered list of crop-left-x anchors the
    renderer interpolates between, so the 9:16 column pans to keep the
    speaker framed (OpusClip's "ReframeAnything" behaviour).

    `w`/`h` are the fixed crop size in source pixels — a height-full 9:16
    column; only the horizontal position moves. `keyframes` are sorted by
    `t_s`, strictly increasing in time, and pre-clamped so every
    `x ∈ [0, source_w - w]` (the renderer never has to clamp at runtime).
    """

    source_w: int = Field(gt=0)
    source_h: int = Field(gt=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)
    keyframes: list[ReframeKeyframe] = Field(default_factory=list)

    @property
    def is_dynamic(self) -> bool:
        """True when the path actually moves — ≥2 anchors spanning more
        than a trivial pixel delta. A near-constant track is reported as
        NOT dynamic so the caller falls back to the cheaper static crop
        (identical result, one fewer moving part)."""
        if len(self.keyframes) < 2:
            return False
        xs = [k.x for k in self.keyframes]
        return (max(xs) - min(xs)) >= _MIN_DYNAMIC_TRAVEL_PX


class Clip(BaseModel):
    """One vertical clip cut + reformatted from a source VOD."""

    id: str = Field(description="ULID with `clp_` prefix.")
    tenant_id: str
    stream_id: str
    candidate: Candidate
    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)
    duration_s: float = Field(ge=0.0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    path: Path
    smart_crop_box: SmartCropBox | None = None
    thumbnail_path: Path | None = None
    # Branded thumbnail variants — voice-markers spec slice D.4.
    # Each is the absolute path to a JPEG sized for that aspect ratio.
    # Missing entries (e.g. when Pillow couldn't decode the source frame)
    # leave the field at None; publishers fall back to `thumbnail_path`.
    thumbnail_16x9_path: Path | None = None
    thumbnail_9x16_path: Path | None = None
    thumbnail_1x1_path: Path | None = None


class ClipManifest(BaseModel):
    """Saved at `<stream_dir>/clips_manifest.json`."""

    stream_id: str
    tenant_id: str
    clips: list[Clip] = Field(default_factory=list)
