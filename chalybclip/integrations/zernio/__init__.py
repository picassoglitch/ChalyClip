"""Zernio integration — multi-platform publishing layer.

ChalybClip-the-company holds ONE Zernio API key (a `sk_...` bearer
secret). Each ChalybClip tenant maps to ONE Zernio `profileId` (their
multi-tenant primitive). Tenants connect their TikTok / IG / YT / X /
LinkedIn etc. accounts via Zernio's hosted OAuth (we mint a per-
platform `authUrl`); Zernio stores tokens; we never see them, never
refresh them. To publish, we POST a video URL + target accounts to
Zernio's /posts endpoint, scoped by the tenant's `profileId`.

  client.py    — async httpx wrapper around the Zernio REST API
  profiles.py  — create_profile_for_tenant() — POST /profiles + persist
                 the returned profileId + name on the tenant row
  webhooks.py  — inbound webhook signature verification + parsing

Replaces the upload-post.com integration (commit history under
`integrations/upload_post/`).
"""

from .analytics import (
    normalize_list,
    normalize_metrics,
    normalize_post,
    sum_headline,
)
from .capabilities import can_hide_comment, can_send_attachment
from .client import (
    ZernioAccount,
    ZernioClient,
    ZernioConnectLink,
    ZernioError,
    ZernioFacebookPage,
    ZernioPostResult,
    ZernioPostStatus,
    ZernioProfile,
)
from .errors import (
    is_transient,
    post_is_auto_retryable,
    spanish_hint,
    summarize_failed_platforms,
)
from .events import process_pending, process_zernio_event
from .profiles import create_profile_for_tenant, ensure_zernio_profile_for_tenant
from .webhooks import (
    parse_post_event,
    register_zernio_webhook,
    verify_zernio_signature,
)

__all__ = [
    "ZernioAccount",
    "ZernioClient",
    "ZernioConnectLink",
    "ZernioError",
    "ZernioFacebookPage",
    "ZernioPostResult",
    "ZernioPostStatus",
    "ZernioProfile",
    "can_hide_comment",
    "can_send_attachment",
    "create_profile_for_tenant",
    "ensure_zernio_profile_for_tenant",
    "is_transient",
    "normalize_list",
    "normalize_metrics",
    "normalize_post",
    "parse_post_event",
    "post_is_auto_retryable",
    "process_pending",
    "process_zernio_event",
    "register_zernio_webhook",
    "spanish_hint",
    "sum_headline",
    "summarize_failed_platforms",
    "verify_zernio_signature",
]
