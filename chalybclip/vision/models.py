"""Pydantic schemas for the local vision pipeline.

The four detector outputs (scene cuts, motion energy, face/emotion, OCR)
fold into one row per second of stream timeline -- the `VisualSignal`
model -- which mirrors the `visual_signals` DB table 1:1.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FaceEmotion = Literal["neutral", "smile", "laugh", "shock", "anger", "sad"]


class SceneCut(BaseModel):
    """A scene boundary at `ts` seconds (the start of the new scene)."""

    model_config = ConfigDict(extra="forbid")
    ts: float = Field(ge=0.0)


class MotionFrame(BaseModel):
    """Aggregated frame-diff magnitude at `ts` seconds."""

    model_config = ConfigDict(extra="forbid")
    ts: float = Field(ge=0.0)
    energy: float = Field(ge=0.0, description="Mean abs frame-diff in [0, 1].")


class FaceFrame(BaseModel):
    """One face/emotion sample. Phase 1 only labels `neutral` and `smile`
    via a mouth-aspect-ratio heuristic; Phase 2 swaps in a real classifier
    (vision LLM or a dedicated emotion model)."""

    model_config = ConfigDict(extra="forbid")
    ts: float = Field(ge=0.0)
    has_face: bool
    emotion: FaceEmotion | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class VisualSignal(BaseModel):
    """One row matching the `visual_signals` DB table.

    Phase 1 fills `scene_cut`, `face_emotion`, and `motion_energy`.
    `text_changed` (OCR) is reserved for Phase 2 — see PHASE_1.md §13.6.5.
    """

    model_config = ConfigDict(extra="forbid")
    ts_offset_s: float = Field(ge=0.0)
    scene_cut: bool = False
    face_emotion: FaceEmotion | None = None
    motion_energy: float | None = Field(default=None, ge=0.0)
    text_changed: bool = False


class VisualSignalTrack(BaseModel):
    """Per-stream collection of one-row-per-second visual signals."""

    stream_id: str
    tenant_id: str
    signals: list[VisualSignal] = Field(default_factory=list)
