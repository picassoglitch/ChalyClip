"""Publish tier gates.

require_top_tier gates the LEGACY native publish routes (clips.py
/publish + the dashboard Buffer matrix): all_access only. The Zernio
publish surface is gated by require_paid_tier — pro CAN publish there,
capped at one connected account (tiers.zernio_account_limit, tested in
tests/test_tiers.py); all_access (and its `partner` alias, normalized
upstream) gets unlimited accounts. free gets 402 everywhere.

These tests exercise the gate function directly with a faked
request.state.tenant_tier — the gate's only input — so they pin the
tier policy without standing up the whole publish stack.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from chalybclip.api.status_gate import require_paid_tier, require_top_tier


def _req(tier: str | None) -> SimpleNamespace:
    """A stand-in Request whose .state.tenant_tier is what the gate
    reads. The real value is set + normalized by the auth middleware."""
    return SimpleNamespace(state=SimpleNamespace(tenant_tier=tier))


# ---- require_top_tier: publish = all_access only ----


def test_top_tier_allows_all_access() -> None:
    # Should not raise.
    require_top_tier(_req("all_access"))


@pytest.mark.parametrize("tier", ["free", "pro", None, "wizard"])
def test_top_tier_blocks_everything_below(tier: str | None) -> None:
    with pytest.raises(HTTPException) as ei:
        require_top_tier(_req(tier))
    assert ei.value.status_code == 402
    assert ei.value.detail["reason"] == "paywall_publish"


def test_top_tier_blocks_pro_specifically() -> None:
    """The behavior change this commit introduces: pro (mid tier) can
    no longer publish — that's reserved for all_access. pro gets Drive
    export instead (task #31)."""
    with pytest.raises(HTTPException) as ei:
        require_top_tier(_req("pro"))
    assert ei.value.status_code == 402
    assert ei.value.detail["current_tier"] == "pro"


# ---- require_paid_tier still distinguishes free from paid ----


@pytest.mark.parametrize("tier", ["pro", "all_access"])
def test_paid_tier_allows_pro_and_all_access(tier: str) -> None:
    require_paid_tier(_req(tier))


def test_paid_tier_blocks_free() -> None:
    with pytest.raises(HTTPException) as ei:
        require_paid_tier(_req("free"))
    assert ei.value.status_code == 402
