"""Chalyb ↔ ChalyClip bridge.

Two surfaces:
  * `service.provision_tenant_for_chalyb` — idempotent: creates a tenant + an
    api_token bound to the Chalyb user_id, OR returns the existing pair if
    that user_id already maps to one. Called from POST /api/admin/tenants.
  * `sso.verify_token` + `sso.sign_token` — HMAC-SHA256 over the JSON payload
    Chalyb signs when the user clicks "Abrir ChalyClip ↗". Called from
    GET /auth/sso.

The wire contract is documented in docs/chalyb_integration.md.
"""

from .service import (
    ChalybProvisionResult,
    ChalybServiceError,
    provision_tenant_for_chalyb,
    sync_tenant_tier,
)
from .sso import SsoTokenError, SsoTokenPayload, verify_sso_token

__all__ = [
    "ChalybProvisionResult",
    "ChalybServiceError",
    "SsoTokenError",
    "SsoTokenPayload",
    "provision_tenant_for_chalyb",
    "sync_tenant_tier",
    "verify_sso_token",
]
