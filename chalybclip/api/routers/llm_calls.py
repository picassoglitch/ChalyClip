"""Cost dashboard endpoint - tail of llm_calls for the active tenant."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from chalybclip.db import Database, LLMCallsRepo

from ..deps import get_db, tenant_binder
from ..schemas import LLMCallResponse

router = APIRouter(prefix="/llm-calls", tags=["llm-calls"])


@router.get("", response_model=list[LLMCallResponse])
async def list_llm_calls(
    limit: int = Query(default=100, ge=1, le=1000),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> list[LLMCallResponse]:
    rows = await LLMCallsRepo(db).list_for_tenant(limit=limit)
    return [LLMCallResponse.model_validate(r.model_dump()) for r in rows]
