"""Google ID-token auth for calls to IAM-protected Cloud Run services.

The worker service only admits callers holding `roles/run.invoker`, so the
web box must send `Authorization: Bearer <id-token>` minted for the
worker's URL. On Cloud Run the token comes from the metadata server; off
GCP (local dev, tests, Modal) there is none and requests go out without
the header, exactly as before.

Tokens live 1 h and a pipeline poll loop can run for hours, so the token
is cached, re-minted shortly before it expires, and re-minted once more
if the worker still answers 401/403.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator

import httpx
import structlog

_log = structlog.get_logger(__name__)

_METADATA_IDENTITY_URL = (
    "http://metadata.google.internal/computeMetadata/v1/"
    "instance/service-accounts/default/identity"
)
# Google ID tokens are valid for 3600 s; re-mint well before that.
_TOKEN_TTL_S = 45 * 60.0
_METADATA_TIMEOUT_S = 5.0


def is_cloud_run_url(url: str) -> bool:
    host = httpx.URL(url).host
    return host.endswith(".run.app")


class CloudRunIdTokenAuth(httpx.Auth):
    """Attaches a metadata-server ID token for `audience` to requests sent
    to that origin (never to other hosts, e.g. after a redirect).

    `audience` is the target service's base URL (scheme + host). Fetch
    failures are logged and the request is sent unauthenticated; Cloud Run
    then answers 403 and the caller handles it.
    """

    def __init__(
        self,
        audience: str,
        *,
        metadata_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        url = httpx.URL(audience)
        self._audience = f"{url.scheme}://{url.host}"
        self._metadata_transport = metadata_transport
        self._token: str | None = None
        self._fetched_at = 0.0

    async def _get_token(self, *, force: bool = False) -> str | None:
        fresh = time.monotonic() - self._fetched_at < _TOKEN_TTL_S
        if self._token and fresh and not force:
            return self._token
        try:
            async with httpx.AsyncClient(
                timeout=_METADATA_TIMEOUT_S,
                transport=self._metadata_transport,
            ) as client:
                resp = await client.get(
                    _METADATA_IDENTITY_URL,
                    params={"audience": self._audience},
                    headers={"Metadata-Flavor": "Google"},
                )
                resp.raise_for_status()
        except httpx.HTTPError as e:
            _log.warning(
                "gcp_identity.token_unavailable",
                audience=self._audience,
                error=type(e).__name__,
                status=(
                    e.response.status_code
                    if isinstance(e, httpx.HTTPStatusError) else None
                ),
            )
            return None
        token = resp.text.strip()
        if not token:
            _log.warning(
                "gcp_identity.token_unavailable",
                audience=self._audience,
                error="empty_body",
            )
            return None
        self._token = token
        self._fetched_at = time.monotonic()
        return token

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        if f"{request.url.scheme}://{request.url.host}" != self._audience:
            yield request
            return
        token = await self._get_token()
        if token:
            request.headers["Authorization"] = f"Bearer {token}"
        response = yield request
        if response.status_code not in (401, 403):
            return
        # A rejected token (or a failed mint) shouldn't fail an hours-long
        # poll loop: mint a fresh one and resend once.
        await response.aread()
        token = await self._get_token(force=True)
        if token:
            request.headers["Authorization"] = f"Bearer {token}"
            yield request


__all__ = ["CloudRunIdTokenAuth", "is_cloud_run_url"]
