"""Per-platform inbox capability gating (Hub phase 9).

The spec documents which platforms support which inbox actions. We
gate at the route layer so the UI never offers an action a platform
will reject (and the operator gets a clear "no soportado" instead of a
raw Zernio error).
"""
from __future__ import annotations

from typing import Final

# Hide-comment: Facebook, Instagram, Threads, X/Twitter (per the
# hideInboxComment description).
HIDE_COMMENT_PLATFORMS: Final[frozenset[str]] = frozenset(
    {"facebook", "instagram", "threads", "twitter"}
)

# DM platforms that accept attachments. Bluesky DMs are text-only
# (plan note); the message webhook enum is instagram/facebook/telegram/
# whatsapp, and WhatsApp/Telegram/Messenger/IG accept media. Keep this
# conservative — text always works, attachments only where listed.
DM_ATTACHMENT_PLATFORMS: Final[frozenset[str]] = frozenset(
    {"instagram", "facebook", "telegram", "whatsapp"}
)


def can_hide_comment(platform: str | None) -> bool:
    return (platform or "").strip().lower() in HIDE_COMMENT_PLATFORMS


def can_send_attachment(platform: str | None) -> bool:
    return (platform or "").strip().lower() in DM_ATTACHMENT_PLATFORMS


__all__ = [
    "DM_ATTACHMENT_PLATFORMS",
    "HIDE_COMMENT_PLATFORMS",
    "can_hide_comment",
    "can_send_attachment",
]
