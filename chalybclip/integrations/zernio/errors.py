"""Per-platform publish-error classification + Spanish hints.

Zernio tags each failed platform target with an `errorCategory` (a
stable enum) and `errorSource` (user/platform/system). This module
turns that into:
  - a Spanish operator hint (what to DO about it), and
  - a `transient` flag: is this worth ONE automatic retry?

Single source of truth for both the dashboard FAILED list and the
auto-retry policy (phase 6), and the structured error CODES the
internal API surfaces to ChalyOBS / Chalyb.
"""
from __future__ import annotations

import datetime as _dt
import re
from typing import Any, Final

# Default backoff for a user_abuse failure with no parseable wait time.
_DEFAULT_ABUSE_COOLDOWN = _dt.timedelta(hours=6)

# errorCategory enum from the OpenAPI PlatformTarget schema. New values
# may appear; anything unmapped falls back to `unknown` handling.
#   auth_expired      — token expired/revoked → reconnect
#   user_content      — wrong format / too long → fix the content
#   user_abuse        — rate limits / spam flags → wait + retry
#   account_issue     — account config problems
#   platform_rejected — policy violation
#   platform_error    — platform 5xx / maintenance → transient
#   system_error      — Zernio infra → transient
#   unknown           — unclassified

# Categories worth exactly ONE automatic retry after a delay: the
# failure is plausibly temporary (platform hiccup, infra blip, a
# velocity cap that clears). Everything else needs a human (reconnect,
# fix caption, policy) — auto-retrying those just burns quota.
TRANSIENT_CATEGORIES: Final[frozenset[str]] = frozenset(
    {"platform_error", "system_error", "user_abuse"}
)

# Spanish operator hints, keyed by errorCategory. Action-oriented:
# every line tells the operator what to do, not just what broke.
_HINTS_ES: Final[dict[str, str]] = {
    "auth_expired": "La cuenta perdió la sesión — reconéctala desde los chips de arriba y reintenta.",
    "user_content": "El contenido no pasó: caption muy larga, formato o duración no soportada. Ajústalo y reintenta.",
    "user_abuse": "Límite de frecuencia de la plataforma — espera unos minutos y reintenta (no publiques en ráfaga).",
    "account_issue": "Problema de configuración de la cuenta en la plataforma — revísala y reintenta.",
    "platform_rejected": "La plataforma rechazó el post por sus políticas — cambia el contenido antes de reintentar.",
    "platform_error": "Error temporal de la plataforma (caída o mantenimiento) — reintenta en un momento.",
    "system_error": "Error interno al publicar — reintenta; si persiste, avísanos.",
    "unknown": "Falló la publicación. Revisa el detalle del error y reintenta.",
}

# A few keyword nudges for when Zernio gives a message but no (or
# `unknown`) category — pattern-match the human message to a better
# hint. Lowercased substring → category.
_MESSAGE_HINTS: Final[tuple[tuple[str, str], ...]] = (
    ("token", "auth_expired"),
    ("reconnect", "auth_expired"),
    ("expired", "auth_expired"),
    ("too long", "user_content"),
    ("caption", "user_content"),
    ("duration", "user_content"),
    ("media", "user_content"),
    ("rate limit", "user_abuse"),
    ("velocity", "user_abuse"),
    ("cooldown", "user_abuse"),
)


def classify_category(category: str | None, message: str | None) -> str:
    """Resolve a usable category: the explicit one when meaningful,
    otherwise a guess from the human message, else `unknown`."""
    cat = (category or "").strip().lower()
    if cat and cat != "unknown" and cat in _HINTS_ES:
        return cat
    msg = (message or "").lower()
    for needle, mapped in _MESSAGE_HINTS:
        if needle in msg:
            return mapped
    return cat if cat in _HINTS_ES else "unknown"


def spanish_hint(category: str | None, message: str | None = None) -> str:
    """Operator-facing Spanish hint for one failed platform target."""
    return _HINTS_ES[classify_category(category, message)]


def is_transient(category: str | None, message: str | None = None) -> bool:
    """True when this failure is worth ONE automatic retry."""
    return classify_category(category, message) in TRANSIENT_CATEGORIES


def summarize_failed_platforms(post: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a failed post's platform targets into rows the FAILED
    list renders: {platform, error, category, hint, transient}.

    Only the platforms that actually failed are returned (a `partial`
    post has some published, some failed)."""
    out: list[dict[str, Any]] = []
    platforms = post.get("platforms")
    if not isinstance(platforms, list):
        return out
    for p in platforms:
        if not isinstance(p, dict):
            continue
        status = str(p.get("status") or "").lower()
        if status not in ("failed", "error"):
            continue
        category = p.get("errorCategory")
        message = p.get("errorMessage") or p.get("error")
        out.append(
            {
                "platform": p.get("platform"),
                "error": message,
                "category": classify_category(
                    category if isinstance(category, str) else None,
                    message if isinstance(message, str) else None,
                ),
                "hint": spanish_hint(
                    category if isinstance(category, str) else None,
                    message if isinstance(message, str) else None,
                ),
                "transient": is_transient(
                    category if isinstance(category, str) else None,
                    message if isinstance(message, str) else None,
                ),
            }
        )
    return out


def parse_cooldown(message: str | None) -> _dt.timedelta:
    """How long to park a platform after a user_abuse failure, read from the
    error text. "wait 1438 minutes" → 1438m; "2 hours" → 2h; "tomorrow" /
    "daily ... limit" → 24h. Falls back to a sane default when unparseable."""
    msg = (message or "").lower()
    m = re.search(r"(\d+)\s*(minute|min|hour|hr|day)", msg)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("min"):
            return _dt.timedelta(minutes=n)
        if unit in ("hour", "hr"):
            return _dt.timedelta(hours=n)
        return _dt.timedelta(days=n)
    if "tomorrow" in msg or "daily" in msg or "per day" in msg:
        return _dt.timedelta(hours=24)
    return _DEFAULT_ABUSE_COOLDOWN


def cooldowns_from_failed_post(post: dict[str, Any]) -> dict[str, _dt.timedelta]:
    """{platform: cooldown} for every platform on `post` that failed with a
    user_abuse signal. Empty when nothing abuse-related failed."""
    out: dict[str, _dt.timedelta] = {}
    for row in summarize_failed_platforms(post):
        if row.get("category") != "user_abuse":
            continue
        platform = row.get("platform")
        if isinstance(platform, str) and platform:
            out[platform.lower()] = parse_cooldown(
                row.get("error") if isinstance(row.get("error"), str) else None
            )
    return out


def failure_anchor(post: dict[str, Any]) -> _dt.datetime | None:
    """WHEN this post failed, best-effort, as an aware UTC datetime.

    The cooldown a failed post implies runs from the FAILURE moment, not
    from whenever a sweep happens to re-read Zernio's failed list — the
    list never shrinks (the once-only auto-retry of a rate-limited post
    just fails again), so anchoring at read time re-armed every cooldown
    on every sweep and parked the platform forever. None when the post
    carries no parseable timestamp; callers must then SKIP the post
    rather than guess "now" (guessing recreates the eternal re-arm)."""
    for key in ("failedAt", "updatedAt", "createdAt", "scheduledFor"):
        raw = post.get(key)
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            ts = _dt.datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=_dt.UTC)
        return ts.astimezone(_dt.UTC)
    return None


def post_is_auto_retryable(post: dict[str, Any]) -> bool:
    """A post qualifies for the once-only auto-retry when AT LEAST one
    failed platform is transient and NONE need human action — retrying
    a token-expired or policy-rejected target would just fail again."""
    rows = summarize_failed_platforms(post)
    if not rows:
        return False
    return all(r["transient"] for r in rows)


__all__ = [
    "TRANSIENT_CATEGORIES",
    "classify_category",
    "cooldowns_from_failed_post",
    "failure_anchor",
    "is_transient",
    "parse_cooldown",
    "post_is_auto_retryable",
    "spanish_hint",
    "summarize_failed_platforms",
]
