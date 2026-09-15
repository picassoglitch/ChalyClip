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
