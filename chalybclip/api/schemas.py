"""Pydantic request/response models for the REST API.

These are deliberately separate from the DB row models in `chalybclip.db.models`
so the wire format can evolve without churning the DB schema (and vice
versa). Most response models are 1:1 trims of the row models with `path`-
typed fields rendered as strings.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ClipStatus = Literal["cut", "ready_for_review", "approved", "rejected", "published"]


class StreamCreateRequest(BaseModel):
    """Body for `POST /streams`."""

    model_config = ConfigDict(extra="forbid")
    vod_url: str = Field(min_length=1)
    persona_id: str = Field(min_length=1)
    language: str | None = None


class StreamResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    vod_url: str
    platform: str
    title: str | None = None
    channel: str | None = None
    duration_s: float
    status: str
    created_at: str


class CandidateResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    stream_id: str
    ts: float
    score: float
    reason: str
    evidence: dict[str, object] = Field(default_factory=dict)
    created_at: str


class ClipResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    stream_id: str
    candidate_id: str | None = None
    start_s: float
    end_s: float
    duration_s: float
    width: int
    height: int
    path: str
    smart_crop_box: dict[str, float] | None = None
    thumbnail_frame_path: str | None = None
    status: str
    created_at: str


class ClipUpdateRequest(BaseModel):
    """Body for `PATCH /clips/{id}`. Only `status` is editable in Phase 1."""

    model_config = ConfigDict(extra="forbid")
    status: ClipStatus | None = None


class PublishRequest(BaseModel):
    """Body for `POST /clips/{id}/publish`.

    `account_ids` defaults to "every connected account on this tenant" when
    omitted - the common case in Phase 1's single-tenant dashboard. The
    publisher's idempotency lives in Task 11.
    """

    model_config = ConfigDict(extra="forbid")
    variant_id: str
    account_ids: list[str] | None = None


class PublishJobResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    clip_id: str
    variant_id: str
    account_id: str
    platform: str
    status: str
    attempts: int
    scheduled_for: str | None = None
    created_at: str


class PersonaCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    primary_language: str = Field(min_length=2)
    target_languages: list[str] = Field(default_factory=list)
    voice_prompt: str = Field(min_length=1)
    routing_tags: list[str] = Field(default_factory=list)


class PersonaUpdateRequest(BaseModel):
    """Body for `PATCH /personas/{id}`. Only set fields are updated."""

    model_config = ConfigDict(extra="forbid")
    name: str | None = None
    primary_language: str | None = None
    target_languages: list[str] | None = None
    voice_prompt: str | None = None
    routing_tags: list[str] | None = None


class PersonaResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    name: str
    primary_language: str
    target_languages: list[str]
    voice_prompt: str
    routing_tags: list[str]
    created_at: str


class LLMCallResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    purpose: str
    provider: str
    model: str
    quality: str
    input_tokens: int
    output_tokens: int
    cost_usd_micros: int
    status: str
    error: str | None = None
    attempts: int
    ts: str


class ErrorResponse(BaseModel):
    detail: str
