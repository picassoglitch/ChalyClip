"""Pydantic schemas for the detect module."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CandidateReason = Literal["voice", "chat", "audio", "visual", "viral", "interval"]


class Candidate(BaseModel):
    """A point in the stream that one or more detectors flagged as clip-worthy."""

    model_config = ConfigDict(extra="forbid")

    timestamp: float = Field(ge=0.0, description="Anchor time in seconds.")
    score: float = Field(ge=0.0, description="phrase_weight * confidence (post-merge).")
    reason: CandidateReason = "voice"
    evidence: dict[str, Any] = Field(default_factory=dict)


class CandidateBatch(BaseModel):
    """Saved alongside the stream so `cut` can re-use detector output."""

    stream_id: str
    tenant_id: str
    candidates: list[Candidate] = Field(default_factory=list)
