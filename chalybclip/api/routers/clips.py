"""Clip detail + status update."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from chalybclip.db import (
    ClipsRepo,
    Database,
    EventsRepo,
)

from ..deps import get_db, require_full_scope, tenant_binder
from ..schemas import (
    ClipResponse,
    ClipUpdateRequest,
)

router = APIRouter(prefix="/clips", tags=["clips"])

# clips.status moves through: cut -> ready_for_review -> approved | rejected
# -> (publish) -> published. The dashboard drives transitions; the worker
# (Task 11) writes `published` after a successful Buffer post.
#
# `published` is terminal for the *generic* transition graph (the editor's
# Complete/finalize and PATCH /clips both refuse it). Reopening a published
# clip back to `approved` for a re-publish is a deliberate recovery action
# behind its own Publish-Center endpoint ("Republicar"), which writes the
# status directly — see api/routers/zernio.py._reopen_published_clip.
_VALID_STATUS_TRANSITIONS: dict[str, set[str]] = {
    "cut": {"ready_for_review", "rejected"},
    "ready_for_review": {"approved", "rejected"},
    "approved": {"rejected", "published"},
    "rejected": {"ready_for_review"},
    "published": set(),
}


@router.get("/{clip_id}", response_model=ClipResponse)
async def get_clip(
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> ClipResponse:
    row = await ClipsRepo(db).get(clip_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="clip not found")
    return ClipResponse.model_validate(row.model_dump())


@router.patch(
    "/{clip_id}",
    response_model=ClipResponse,
    dependencies=[Depends(require_full_scope)],
)
async def update_clip(
    clip_id: str,
    payload: ClipUpdateRequest,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> ClipResponse:
    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="clip not found")

    if payload.status is not None and payload.status != clip.status:
        allowed = _VALID_STATUS_TRANSITIONS.get(clip.status, set())
        if payload.status not in allowed:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"cannot transition clip from {clip.status!r} to {payload.status!r}",
            )
        # Direct UPDATE - the repo's bulk upserts are insert-only by design,
        # so a one-off status change goes via SQL here.
        conn = await db.connect()
        await conn.execute(
            "UPDATE clips SET status = ? WHERE id = ? AND tenant_id = ?",
            (payload.status, clip_id, tenant_id),
        )
        await conn.commit()
        await EventsRepo(db).emit(
            type=f"clip.{payload.status}",
            payload={"clip_id": clip_id, "from": clip.status, "to": payload.status},
        )

    refreshed = await ClipsRepo(db).get(clip_id)
    assert refreshed is not None
    return ClipResponse.model_validate(refreshed.model_dump())
