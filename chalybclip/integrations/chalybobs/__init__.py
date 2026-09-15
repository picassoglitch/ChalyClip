"""ChalyOBS integration — the bidirectional live connection switch.

ChalyOBS is the streaming front-door; ChalyClip is the clipper. The
"connection" between them is a single per-tenant flag whose source of
truth lives in ChalyOBS (chalybobs_sessions.clips_enabled). This module
lets ChalyClip read + flip that flag so the switch works from the
ChalyClip Live page too.

Both sides present the same shared internal bearer
(CHALYBCLIP_INTERNAL_SIGNING_SECRET == ChalyOBS's CHALYBOBS_RELAY_SECRET).
Keyed by the tenant's Chalyb user id (external_user_id), which ChalyOBS
uses as its tenant_id.
"""

from .connection import get_connection, set_connection

__all__ = ["get_connection", "set_connection"]
