"""Webhook subscription CRUD via the REST API.

All routes scoped to the bound tenant. Secret is generated server-side
on create and returned in the response *once* (mirrors api_tokens).
Subsequent reads never expose the secret.
"""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from chalybclip.db import (
    Database,
    WebhookSecretsRepo,
    WebhookSubscriptionsRepo,
)
from chalybclip.webhooks import UnsafeWebhookUrlError, assert_webhook_url_safe

from ..deps import get_db, require_full_scope, tenant_binder

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# Default grace window for a rotated-out secret. Subscribers should re-fetch
# the active secret list before this elapses.
_DEFAULT_ROTATION_GRACE_S = 86400.0  # 24h


class WebhookCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(min_length=1, max_length=2048)
    types: list[str] = Field(default_factory=list)

    @field_validator("url")
    @classmethod
    def _ssrf_safe(cls, v: str) -> str:
        # First line of SSRF defense: reject at registration time. The
        # dispatcher re-checks at send time too (DNS rebinding defense).
        try:
            assert_webhook_url_safe(v)
        except UnsafeWebhookUrlError as e:
            raise ValueError(str(e)) from e
        return v


class WebhookCreateResponse(BaseModel):
    """Response on create — includes the secret. Other reads omit it."""

    model_config = ConfigDict(extra="ignore")
    id: str
    url: str
    types: list[str]
    secret: str  # only returned at create time
    status: str
    created_at: str


class WebhookResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    url: str
    types: list[str]
    status: str
    created_at: str
    last_dispatch_ts: str | None = None
    failure_count: int = 0


@router.post(
    "",
    response_model=WebhookCreateResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_full_scope)],
)
async def create_webhook(
    payload: WebhookCreateRequest,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> WebhookCreateResponse:
    """Register a subscriber URL. Secret is minted now and shown once."""
    secret = secrets.token_hex(32)
    sub = await WebhookSubscriptionsRepo(db).create(
        url=payload.url, types=payload.types, secret=secret
    )
    return WebhookCreateResponse(
        id=sub.id,
        url=sub.url,
        types=sub.types,
        secret=secret,
        status=sub.status,
        created_at=sub.created_at,
    )


@router.get("", response_model=list[WebhookResponse])
async def list_webhooks(
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> list[WebhookResponse]:
    rows = await WebhookSubscriptionsRepo(db).list_for_tenant()
    return [
        WebhookResponse(
            id=r.id,
            url=r.url,
            types=r.types,
            status=r.status,
            created_at=r.created_at,
            last_dispatch_ts=r.last_dispatch_ts,
            failure_count=r.failure_count,
        )
        for r in rows
    ]


@router.delete(
    "/{sub_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_full_scope)],
)
async def delete_webhook(
    sub_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> None:
    """Idempotent delete: 204 if the row was there, 404 if it wasn't."""
    if not await WebhookSubscriptionsRepo(db).delete(sub_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="webhook not found"
        )


# ---------------------------------------------------------------------------
# Phase 3 #2: secret rotation
# ---------------------------------------------------------------------------


class RotateSecretRequest(BaseModel):
    """Optional override of the rotation grace window."""

    model_config = ConfigDict(extra="forbid")
    grace_s: float | None = Field(
        default=None,
        gt=0,
        description="Seconds the prior secret stays valid. Default: 24h.",
    )


class RotateSecretResponse(BaseModel):
    """Response from `POST /webhooks/{id}/rotate-secret`.

    `secret` is the new value (returned ONCE; mirrors api_tokens / create).
    `prior_secret_expires_at` is the deadline subscribers have to update
    their config before the previous secret stops being honored.
    """

    model_config = ConfigDict(extra="ignore")
    id: str
    secret: str
    prior_secret_expires_at: str


class ActiveSecretResponse(BaseModel):
    """One past secret still inside its grace window."""

    model_config = ConfigDict(extra="ignore")
    id: str
    expires_at: str
    created_at: str
    # The secret value is included because the dashboard's primary use case
    # is "show me what to update my subscriber to" - and these are by
    # design rotated-out secrets, not keys.
    secret: str


@router.post(
    "/{sub_id}/rotate-secret",
    response_model=RotateSecretResponse,
    dependencies=[Depends(require_full_scope)],
)
async def rotate_secret(
    sub_id: str,
    payload: RotateSecretRequest,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> RotateSecretResponse:
    """Rotate the subscription's HMAC secret.

    The prior secret stays honored for `grace_s` (default 24h) so
    subscribers can update without missing deliveries. The new secret is
    returned ONCE; subsequent reads only show its existence, not the value.
    """
    grace_s = payload.grace_s if payload.grace_s is not None else _DEFAULT_ROTATION_GRACE_S
    new_secret = secrets.token_hex(32)
    try:
        await WebhookSecretsRepo(db).rotate(
            sub_id, new_secret=new_secret, ttl_s=grace_s
        )
    except Exception:
        # `rotate()` raises ChalybClipError on missing subscription; tenancy
        # enforcement happens inside the SQL.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="webhook not found"
        ) from None

    # Compute the prior_secret_expires_at by reading the freshest version row.
    versions = await WebhookSecretsRepo(db).list_active_for_subscription(sub_id)
    prior_expires_at = versions[0].expires_at if versions else ""
    return RotateSecretResponse(
        id=sub_id,
        secret=new_secret,
        prior_secret_expires_at=prior_expires_at,
    )


@router.get(
    "/{sub_id}/secrets",
    response_model=list[ActiveSecretResponse],
)
async def list_active_secrets(
    sub_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> list[ActiveSecretResponse]:
    """Past secrets still inside the rotation grace window.

    The *current* secret is on the subscription itself and is shown only
    at create / rotate time; this endpoint specifically lists rotated-out
    secrets that subscribers may still see in the wild during the window.
    """
    if await WebhookSubscriptionsRepo(db).get(sub_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="webhook not found"
        )
    versions = await WebhookSecretsRepo(db).list_active_for_subscription(sub_id)
    return [
        ActiveSecretResponse(
            id=v.id,
            expires_at=v.expires_at,
            created_at=v.created_at,
            secret=v.secret,
        )
        for v in versions
    ]
