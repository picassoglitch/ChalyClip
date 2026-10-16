"""Hub API — Zernio publishing routes (/api/hub/v1/publish/zernio …).

Owned by the publishing workstream. Placeholder from the foundation PR so
app.py can include it; add routes to `router` (built with `hub_router()`,
which applies the hub auth + tenant binding).
"""

from __future__ import annotations

from chalybclip.api.hub_api import hub_router

router = hub_router(tags=["hub-publish"])
