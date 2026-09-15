"""Canonical tier normalization — partner → all_access, and the
"don't override on garbage" semantics that protect existing paid
tenants from a typo'd tier label.

Background: Chalyb sends `partner` for the top tier. Before
chalybclip.tiers existed, the provisioning validator + SSO sync only
accepted {free, pro, all_access} and silently dropped `partner`, so
a partner tenant landed as `free` in ChalybClip — losing every paid
perk (including the upload-post profile limit that 403'd the
operator). These tests pin the alias map + the two normalization
modes.
"""

from __future__ import annotations

import pytest

from chalybclip.tiers import (
    ALL_ACCESS,
    FREE,
    PAID_TIERS,
    PRO,
    TOP_TIERS,
    normalize_tier,
    resolve_tier_alias,
    zernio_account_limit,
)


# ---- alias map: partner == all_access ----


@pytest.mark.parametrize(
    "raw",
    ["partner", "Partner", "  PARTNER ", "partners", "enterprise",
     "allaccess", "all-access",
     # Chalyb renamed its top tier ALL_ACCESS → VIP; SSO tokens and
     # provisioning calls carry 'vip' now.
     "vip", "VIP"],
)
def test_partner_and_friends_map_to_all_access(raw: str) -> None:
    assert resolve_tier_alias(raw) == ALL_ACCESS
    assert normalize_tier(raw) == ALL_ACCESS


@pytest.mark.parametrize("raw", ["free", "FREE", " pro ", "all_access"])
def test_canonical_tiers_pass_through(raw: str) -> None:
    expected = raw.strip().lower()
    assert resolve_tier_alias(raw) == expected
    assert normalize_tier(raw) == expected


# ---- the two modes differ on unrecognized input ----


def test_resolve_returns_none_for_unrecognized() -> None:
    """resolve_tier_alias → None for anything we don't recognize, so the
    caller (provisioning validator, SSO sync) can KEEP the tenant's
    current tier rather than overwrite it with garbage."""
    assert resolve_tier_alias("wizard") is None
    assert resolve_tier_alias("") is None
    assert resolve_tier_alias(None) is None


def test_normalize_defaults_unrecognized_to_free() -> None:
    """normalize_tier → free for unrecognized, used at READ time where a
    concrete tier is always needed and least-privilege is the safe
    fallback."""
    assert normalize_tier("wizard") == FREE
    assert normalize_tier("") == FREE
    assert normalize_tier(None) == FREE


# ---- the sets gates compare against ----


def test_top_tier_is_all_access_only() -> None:
    """Publishing is top-tier only. pro (mid) + free must NOT be in it."""
    assert TOP_TIERS == {ALL_ACCESS}
    assert PRO not in TOP_TIERS
    assert FREE not in TOP_TIERS


def test_paid_tiers_excludes_free_includes_pro_and_all_access() -> None:
    assert FREE not in PAID_TIERS
    assert PRO in PAID_TIERS
    assert ALL_ACCESS in PAID_TIERS


# ---- per-tier connected-account cap (Zernio publish surface) ----


def test_account_limit_pro_is_one() -> None:
    """pro CAN publish, capped at one connected social account."""
    assert zernio_account_limit(PRO) == 1


def test_account_limit_all_access_unlimited_incl_partner_alias() -> None:
    """all_access (VIP) is unlimited (None) — and the partner alias
    resolves to the same."""
    assert zernio_account_limit(ALL_ACCESS) is None
    assert zernio_account_limit("partner") is None


def test_account_limit_free_and_unknown_are_zero() -> None:
    """free connects nothing; unrecognized labels normalize to free."""
    assert zernio_account_limit(FREE) == 0
    assert zernio_account_limit("wizard") == 0
    assert zernio_account_limit(None) == 0
