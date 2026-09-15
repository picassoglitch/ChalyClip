"""Canonical subscription tiers — single source of truth.

ChalybClip recognizes exactly three internal tiers, lowest → highest:

    free        — ingest + clip + DOWNLOAD (watermarked). No publishing.
    pro         — "mid" tier. Adds the paid perks below the top tier
                  (e.g. Drive export — see task #31). NOT publishing.
    all_access  — "top" tier. Adds external publishing (Zernio /
                  TikTok / YouTube / IG auto-posting).

Chalyb (the IdP) sometimes labels the top tier differently on its
side — e.g. it sends `partner` in the SSO token for partner accounts.
Those are ALIASES of our canonical tiers. Before this module existed
the alias was silently dropped (the provisioning validator only
accepted the three canonical names), so a `partner` user landed as
`free` in ChalybClip — losing every paid perk including the upload-post
profile limit. Centralizing the alias map here means a new label from
Chalyb is a one-line addition, applied everywhere tier is ingested
or read.

Rule: normalize at every boundary where a tier ENTERS ChalybClip
(provisioning, SSO sync) and at the request-scoped READ choke point
(auth middleware). Downstream gates then compare against the
canonical sets without re-normalizing.
"""

from __future__ import annotations

from typing import Final

# The only tier strings the rest of the codebase should ever compare
# against. Anything stored / read goes through normalize_tier first.
FREE: Final = "free"
PRO: Final = "pro"
ALL_ACCESS: Final = "all_access"

CANONICAL_TIERS: Final[frozenset[str]] = frozenset({FREE, PRO, ALL_ACCESS})

# Tiers that may publish to external platforms (TikTok / YouTube / IG).
# Top tier only — `pro` is the mid tier (Drive export), `free` downloads
# locally with a watermark.
TOP_TIERS: Final[frozenset[str]] = frozenset({ALL_ACCESS})

# Tiers above free — any paid perk that isn't specifically top-tier.
PAID_TIERS: Final[frozenset[str]] = frozenset({PRO, ALL_ACCESS})

# Connected-social-account cap per tier for the Zernio publish surface.
# `pro` CAN publish, but with exactly ONE connected account; `all_access`
# (VIP) is unlimited (None). `free` connects nothing — publishing stays
# behind the paid gate.
ZERNIO_ACCOUNT_LIMITS: Final[dict[str, int | None]] = {
    FREE: 0,
    PRO: 1,
    ALL_ACCESS: None,
}


def zernio_account_limit(raw: str | None) -> int | None:
    """Connected-account cap for a (possibly raw/aliased) tier label.

    None = unlimited. Unknown labels normalize to `free` → 0, the
    least-privilege default.
    """
    return ZERNIO_ACCOUNT_LIMITS.get(normalize_tier(raw), 0)

# Aliases Chalyb (or future billing sources) may use for a canonical
# tier. Keyed lowercase. Extend this — NOT the comparison sites — when
# a new label shows up. `partner` == our top tier `all_access`, and so is
# `vip` — Chalyb renamed its top tier ALL_ACCESS → VIP, so every SSO
# token and provisioning call for top-tier users now carries 'vip'.
# Without this entry those users land as `free` (the normalize fallback)
# and lose every paid perk on login.
_ALIASES: Final[dict[str, str]] = {
    "partner": ALL_ACCESS,
    "partners": ALL_ACCESS,
    "enterprise": ALL_ACCESS,
    "allaccess": ALL_ACCESS,
    "all-access": ALL_ACCESS,
    "vip": ALL_ACCESS,
}


def resolve_tier_alias(raw: str | None) -> str | None:
    """Map a raw tier label to its canonical name, or None if it isn't a
    tier we recognize.

    Used where "unrecognized → don't touch the existing value" is the
    right call (the provisioning validator, the SSO sync write): a typo
    or a brand-new label we haven't mapped must NOT silently downgrade
    an existing paid tenant to free. Returns None in that case so the
    caller can skip the override.
    """
    if not raw:
        return None
    v = raw.lower().strip()
    v = _ALIASES.get(v, v)
    return v if v in CANONICAL_TIERS else None


def normalize_tier(raw: str | None) -> str:
    """Map a raw tier label to its canonical name, defaulting to `free`.

    Used at READ time where we always need a concrete tier to gate on
    (the auth middleware that sets request.state.tenant_tier). An
    unrecognized / missing value resolves to `free` — the safe default
    (least privilege), since a read-time fallback only ever GRANTS
    fewer perks, never more.
    """
    return resolve_tier_alias(raw) or FREE


__all__ = [
    "FREE",
    "PRO",
    "ALL_ACCESS",
    "CANONICAL_TIERS",
    "TOP_TIERS",
    "PAID_TIERS",
    "ZERNIO_ACCOUNT_LIMITS",
    "resolve_tier_alias",
    "normalize_tier",
    "zernio_account_limit",
]
