"""OBS-friendly HTML overlay routes (slice M.4).

Each route renders a standalone HTML page that:
  - has a fully TRANSPARENT body (so OBS Browser Source can composite
    the overlay on top of the actual stream pixels)
  - reads all config from query-string parameters (OBS Browser Source
    doesn't pass headers or cookies, so per-tenant auth isn't on the
    table here — operators publish a "URL with my config baked in")
  - is self-contained: no external runtime deps beyond fonts the page
    @font-faces itself.

The first overlay is the Kick repost-page banner — the same brand
treatment ChalyClip burns into MP4 exports, but as a live HTML layer
operators can use directly on their OBS scene or in a 3rd-party clip
viewer that supports browser sources.

USAGE (OBS Browser Source):
  URL:      http://<chalybclip-host>/overlay/kick?channelId=clavicular
  Width:    1080
  Height:   1920
  Custom CSS (in OBS): body { background: transparent !important; }
  ☑ Refresh browser when scene becomes active

Optional query params:
  channelId    user's Kick channel id (drives "KICK.COM/<channelId>")
  handle       follow-card @handle override (defaults to ChalyClip's
               own Kick handle — "reelonkick")
  followLabel  text on the green pill (defaults to "Follow")
  scale        overall scale factor, 0.5–2.0 (defaults to 1.0)
  starfield    "1" to render the preview starfield background; OMIT
               for production / OBS mode (default = transparent)

The component lives in `templates/overlay_kick.html`. It's deliberately
NOT under the /dashboard prefix so it never picks up the dashboard's
tenant_binder dependency — OBS can hit it cold.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Standalone Jinja2 instance so this router doesn't pull on the
# dashboard's templating singleton (which carries extra globals we
# don't need on a transparent overlay).
_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
_templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# Our own brand handle — used as the DEFAULT follow-card @handle.
# Spec: "The HANDLE is OURS and is the default. Hardcode our handle
# as the default value." Operators can override per-overlay via the
# `handle` query string.
_CHALYBCLIP_DEFAULT_HANDLE = "reelonkick"

router = APIRouter(prefix="/overlay", tags=["overlay"])


@router.get("/kick", response_class=HTMLResponse)
async def overlay_kick(
    request: Request,
    channelId: str = Query(
        default="yourhandle",
        description=(
            "User-configured Kick channel id. Drives the "
            "KICK.COM/<channelId> text in the URL bar."
        ),
        max_length=64,
    ),
    handle: str = Query(
        default=_CHALYBCLIP_DEFAULT_HANDLE,
        description=(
            "Follow-card @handle. Defaults to ChalyClip's own Kick "
            "handle so the card always reads correctly even when the "
            "operator hasn't set their own."
        ),
        max_length=64,
    ),
    followLabel: str = Query(
        default="Follow",
        description="Text on the green Follow pill.",
        max_length=24,
    ),
    scale: float = Query(
        default=1.0,
        ge=0.5,
        le=2.0,
        description=(
            "Overall scale factor — multiplies every dimension via "
            "the --kb-scale CSS variable. Useful for OBS scenes that "
            "want the banner at non-1080p resolutions."
        ),
    ),
    starfield: int = Query(
        default=0,
        description=(
            "1 to render the preview starfield background; OMIT for "
            "production/OBS use where the body must stay transparent."
        ),
        ge=0,
        le=1,
    ),
) -> HTMLResponse:
    """Render the Kick repost-page overlay as a standalone HTML page."""
    return _templates.TemplateResponse(
        request,
        "overlay_kick.html",
        {
            "channel_id": _sanitize(channelId),
            "handle": _sanitize(handle).lstrip("@"),
            "follow_label": _sanitize(followLabel),
            "scale": scale,
            "starfield": bool(starfield),
        },
    )


def _sanitize(s: str) -> str:
    """Strip whitespace + drop control chars. The overlay is rendered
    into HTML attributes / text content so Jinja's autoescape covers
    the XSS surface; this is purely for visual cleanliness (newlines
    inside a URL bar look ridiculous)."""
    if not s:
        return ""
    out = []
    for ch in s:
        if ch == "\n" or ch == "\r" or ch == "\t":
            continue
        if ord(ch) < 32:
            continue
        out.append(ch)
    return "".join(out).strip()
