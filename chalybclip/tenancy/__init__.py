"""Tenancy contract — contextvars + token hashing.

The contract surface is intentionally tiny: callers either bind a tenant
via `bound_tenant(...)` or read it via `current_tenant_id()`. Repos call
`current_tenant_id()` at every read/write so cross-tenant access is
impossible even if a handler is buggy.
"""

from .context import (
    assert_tenant,
    bound_tenant,
    current_tenant_id,
    current_tenant_id_optional,
)
from .tokens import hash_token, mint_token, verify_scope

__all__ = [
    "assert_tenant",
    "bound_tenant",
    "current_tenant_id",
    "current_tenant_id_optional",
    "hash_token",
    "mint_token",
    "verify_scope",
]
