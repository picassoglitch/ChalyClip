"""Persona CRUD."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from chalybclip.db import Database, PersonasRepo

from ..deps import get_db, require_full_scope, tenant_binder
from ..schemas import PersonaCreateRequest, PersonaResponse, PersonaUpdateRequest

router = APIRouter(prefix="/personas", tags=["personas"])


@router.get("", response_model=list[PersonaResponse])
async def list_personas(
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> list[PersonaResponse]:
    rows = await PersonasRepo(db).list_for_tenant()
    return [PersonaResponse.model_validate(r.model_dump()) for r in rows]


@router.post(
    "",
    response_model=PersonaResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_full_scope)],
)
async def create_persona(
    payload: PersonaCreateRequest,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> PersonaResponse:
    repo = PersonasRepo(db)
    if await repo.get(payload.id) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"persona id {payload.id!r} already exists",
        )
    row = await repo.create(
        persona_id=payload.id,
        name=payload.name,
        primary_language=payload.primary_language,
        target_languages=payload.target_languages,
        voice_prompt=payload.voice_prompt,
        routing_tags=payload.routing_tags,
    )
    return PersonaResponse.model_validate(row.model_dump())


@router.patch(
    "/{persona_id}",
    response_model=PersonaResponse,
    dependencies=[Depends(require_full_scope)],
)
async def update_persona(
    persona_id: str,
    payload: PersonaUpdateRequest,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> PersonaResponse:
    repo = PersonasRepo(db)
    existing = await repo.get(persona_id)
    if existing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="persona not found")
    merged = await repo.upsert(
        persona_id=persona_id,
        name=payload.name if payload.name is not None else existing.name,
        primary_language=payload.primary_language
        if payload.primary_language is not None
        else existing.primary_language,
        target_languages=payload.target_languages
        if payload.target_languages is not None
        else existing.target_languages,
        voice_prompt=payload.voice_prompt
        if payload.voice_prompt is not None
        else existing.voice_prompt,
        routing_tags=payload.routing_tags
        if payload.routing_tags is not None
        else existing.routing_tags,
    )
    return PersonaResponse.model_validate(merged.model_dump())
