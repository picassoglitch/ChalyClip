"""FastAPI dependencies.

Two main jobs:
    * Hand routes the shared `Database` instance pinned to app.state.
    * Bind `current_tenant_id()` for the duration of each handler so the
      repos pick up the right tenant via contextvars.

Scope checks (`require_full_scope`) live here so handlers don't repeat
the same `if scope != "full": raise HTTPException(403)` boilerplate.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends, HTTPException, Request, status

from chalybclip.db import Database
from chalybclip.tenancy import bound_tenant


def get_db(request: Request) -> Database:
    """Return the app-wide DB handle stashed on `app.state.db`."""
    db: Database | None = getattr(request.app.state, "db", None)
    if db is None:
        raise RuntimeError("app.state.db is not initialized")
    return db


def get_token_scope(request: Request) -> str:
    """Return the scope from the auth middleware (`full` or `read`)."""
    scope: str | None = getattr(request.state, "token_scope", None)
    if scope is None:
        # The auth middleware sets this on every authenticated request;
        # this branch fires only for misconfigured public routes.
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthenticated")
    return scope


def require_full_scope(scope: str = Depends(get_token_scope)) -> str:
    """Reject `read` tokens for write endpoints."""
    if scope != "full":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"this endpoint requires scope=full, token has scope={scope!r}",
        )
    return scope


async def tenant_binder(request: Request) -> AsyncIterator[str]:
    """Bind `current_tenant_id()` for the handler.

    Use as: `tenant_id: str = Depends(tenant_binder)`. Repos called from
    inside the handler will see the correct tenant via contextvars; the
    binding is unwound when the handler returns. Async generator (not
    sync) so the contextvar set/reset happens in the same task as the
    handler - FastAPI runs sync generator deps in a threadpool, which
    breaks contextvars.
    """
    tenant_id: str | None = getattr(request.state, "tenant_id", None)
    if tenant_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthenticated")
    with bound_tenant(tenant_id):
        yield tenant_id
