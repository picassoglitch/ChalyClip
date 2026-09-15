"""API token minting + hashing.

Raw tokens are shown to the user once at issuance and never again. We
store only `sha256(raw)` in `api_tokens.hash`. Auth flow:

    presented_token  ─sha256─►  hash  ─lookup─►  tenant_id  ─bind─►  contextvar

The hash is the equality key, never the raw token.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from chalybclip.errors import TenancyError
from chalybclip.ids import new_id

Scope = Literal["full", "read"]
_VALID_SCOPES: frozenset[str] = frozenset({"full", "read"})


def mint_token() -> tuple[str, str]:
    """Return `(raw, hash)`. Show `raw` to the user once; persist `hash`."""
    raw = new_id("tok")
    return raw, hash_token(raw)


def hash_token(raw: str) -> str:
    """sha256 of the raw bearer token (lowercase hex)."""
    if not raw:
        raise TenancyError("cannot hash empty token")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_scope(token_scope: str, required: Scope) -> bool:
    """Return True if `token_scope` is allowed to perform a `required` action.

    `full` includes `read`; `read` does not include `full`.
    """
    if token_scope not in _VALID_SCOPES:
        raise TenancyError(f"unknown token scope: {token_scope!r}")
    if required not in _VALID_SCOPES:
        raise TenancyError(f"unknown required scope: {required!r}")
    if required == "read":
        return token_scope in ("full", "read")
    return token_scope == "full"
