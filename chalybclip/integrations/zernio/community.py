"""Community notifications — Discord/Telegram announce-on-publish + the
weekly digest (Hub phase 11).

Discord and Telegram are NOT clip targets; they're the streamer's
community channels. On post.published the hub posts a rich embed
(Discord) / text (Telegram) announcing the fresh clip, with the
webhook identity customized to the tenant's brand. Pure builders here
keep the payload shaping testable; the wiring lives in events.py.
"""
from __future__ import annotations

from typing import Any, Final

# Chalyb cyberpunk lime (#c5f82a) as the Discord embed accent, decimal.
_EMBED_COLOR: Final[int] = 0xC5F82A


def _first_published_url(post: dict[str, Any]) -> str | None:
    """The first platform publishedUrl from a post.published payload —
    the link the community embed points at."""
    platforms = post.get("platforms")
    if not isinstance(platforms, list):
        return None
    for p in platforms:
        if isinstance(p, dict):
            url = p.get("publishedUrl") or p.get("platformPostUrl")
            if isinstance(url, str) and url:
                return url
    return None


def _platform_names(post: dict[str, Any]) -> list[str]:
    platforms = post.get("platforms")
    out: list[str] = []
    if isinstance(platforms, list):
        for p in platforms:
            if isinstance(p, dict) and isinstance(p.get("platform"), str):
                out.append(p["platform"])
    return out


def build_discord_embed(
    post: dict[str, Any],
    *,
    brand_name: str | None = None,
    brand_avatar_url: str | None = None,
    thumbnail_url: str | None = None,
) -> dict[str, Any]:
    """Build the Discord platformSpecificData for a clip announcement:
    one rich embed (title, link, platform list, thumbnail) + the
    tenant's brand identity (webhookUsername/avatar)."""
    title = (post.get("content") or "Nuevo clip").strip()[:256] or "Nuevo clip"
    url = _first_published_url(post)
    platforms = _platform_names(post)
    embed: dict[str, Any] = {
        "title": title,
        "color": _EMBED_COLOR,
        "description": (
            "📢 ¡Nuevo clip publicado en "
            + (", ".join(platforms) if platforms else "tus redes")
            + "!"
        ),
    }
    if url:
        embed["url"] = url
    if thumbnail_url:
        embed["thumbnail"] = {"url": thumbnail_url}
    if platforms:
        embed["fields"] = [
            {"name": "Plataformas", "value": ", ".join(platforms), "inline": True}
        ]
    data: dict[str, Any] = {"embeds": [embed]}
    if brand_name:
        data["webhookUsername"] = brand_name[:80]
    if brand_avatar_url:
        data["webhookAvatarUrl"] = brand_avatar_url
    return data


def build_telegram_text(post: dict[str, Any]) -> str:
    """Plain-text announcement for Telegram (no embed model)."""
    title = (post.get("content") or "Nuevo clip").strip()
    url = _first_published_url(post)
    platforms = _platform_names(post)
    line = f"📢 ¡Nuevo clip! {title}"
    if platforms:
        line += f"\nEn: {', '.join(platforms)}"
    if url:
        line += f"\n{url}"
    return line


def build_notification_payload(
    post: dict[str, Any],
    *,
    discord_account_id: str | None,
    telegram_account_id: str | None,
    brand_name: str | None = None,
    brand_avatar_url: str | None = None,
    thumbnail_url: str | None = None,
) -> tuple[list[tuple[str, str]], dict[str, dict[str, Any]], str]:
    """Build (platforms, platformSpecificData, fallback_text) for the
    community notification createPost. Only includes the channels that
    are configured."""
    platforms: list[tuple[str, str]] = []
    psd: dict[str, dict[str, Any]] = {}
    if discord_account_id:
        platforms.append(("discord", discord_account_id))
        psd["discord"] = build_discord_embed(
            post, brand_name=brand_name, brand_avatar_url=brand_avatar_url,
            thumbnail_url=thumbnail_url,
        )
    if telegram_account_id:
        platforms.append(("telegram", telegram_account_id))
    return platforms, psd, build_telegram_text(post)


def build_weekly_digest_text(totals: dict[str, int | None], *, days: int = 7) -> str:
    """Plain-text weekly digest from the phase-7 headline totals. A
    missing metric shows '—' (no fake zeros)."""
    def fmt(v: int | None) -> str:
        return "—" if v is None else f"{v:,}"

    return (
        f"📊 Resumen de la semana ({days} días)\n"
        f"👁 {fmt(totals.get('views'))} views · "
        f"❤️ {fmt(totals.get('likes'))} likes · "
        f"💬 {fmt(totals.get('comments'))} comentarios · "
        f"🔁 {fmt(totals.get('shares'))} compartidos"
    )


__all__ = [
    "build_discord_embed",
    "build_notification_payload",
    "build_telegram_text",
    "build_weekly_digest_text",
]
