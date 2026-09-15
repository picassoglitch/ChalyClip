"""MCP server — translation layer over the REST surface.

Phase 3 #3 lock-down rule: this module adds *no new business logic*. Every
tool is a thin adapter over an existing repo call or service-layer
function so external agents (Claude Code, Cursor, etc.) drive the same
state transitions a human would via the dashboard or REST API.

Auth model: the server takes one API token at startup (env var or CLI
flag). The token resolves to a tenant_id via `api_tokens.lookup_by_hash`,
and every tool call binds that tenant via `bound_tenant(...)`. No
cross-tenant access; rejected tokens fail fast at server boot.
"""

from .server import build_server, resolve_tenant_from_token, run_stdio_server

__all__ = [
    "build_server",
    "resolve_tenant_from_token",
    "run_stdio_server",
]
