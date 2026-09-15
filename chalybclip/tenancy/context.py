"""ContextVar-based tenancy binding.

Every code path that touches tenant-scoped data calls `current_tenant_id()`
to discover its tenant. The CLI binds via the `Settings.default_tenant_id`
in process_vod (Phase 0 carry-over); FastAPI handlers bind via the auth
middleware. Repos call this independently as defense in depth.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from chalybclip.errors import TenancyError

_tenant_var: ContextVar[str | None] = ContextVar("chalybclip_tenant_id", default=None)


def current_tenant_id() -> str:
    """Return the currently-bound tenant_id. Raises if unbound.

    Use this at the top of any service function or repo method that
    operates on tenant-scoped data.
    """
    value = _tenant_var.get()
    if value is None:
        raise TenancyError("no tenant bound; wrap the call in `bound_tenant(<tenant_id>)`")
    return value


def current_tenant_id_optional() -> str | None:
    """Same as `current_tenant_id` but returns None instead of raising."""
    return _tenant_var.get()


@contextmanager
def bound_tenant(tenant_id: str) -> Iterator[None]:
    """Bind `tenant_id` to the current task for the duration of the block.

    Nestable: nested `bound_tenant` calls properly restore the outer
    binding on exit, including on exception.
    """
    if not tenant_id:
        raise TenancyError("tenant_id must be a non-empty string")
    token = _tenant_var.set(tenant_id)
    try:
        yield
    finally:
        _tenant_var.reset(token)


def assert_tenant(expected: str) -> None:
    """Raise unless `expected` matches the bound tenant."""
    actual = current_tenant_id()
    if actual != expected:
        raise TenancyError(f"tenant mismatch: bound={actual!r}, expected={expected!r}")
