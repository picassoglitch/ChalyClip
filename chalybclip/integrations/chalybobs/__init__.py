"""ChalybOBS integration — the bidirectional live connection switch.

ChalybOBS is the streaming front-door; ChalybClip is the clipper. The
"connection" between them is a single per-tenant flag whose source of
truth lives in ChalybOBS (chalybobs_sessions.clips_enabled). This module
lets ChalybClip read + flip that flag so the switch works from the
ChalybClip Live page too.

Both sides present the same shared internal bearer
(CHALYBCLIP_INTERNAL_SIGNING_SECRET == ChalybOBS's CHALYBOBS_RELAY_SECRET).
Keyed by the tenant's Chalyb user id (external_user_id), which ChalybOBS
uses as its tenant_id.
"""

from .connection import get_connection, set_connection

__all__ = ["get_connection", "set_connection"]
