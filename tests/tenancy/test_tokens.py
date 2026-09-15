"""Tests for token minting + hashing + scope verification."""

from __future__ import annotations

import pytest

from chalybclip.errors import TenancyError
from chalybclip.tenancy import hash_token, mint_token, verify_scope


def test_mint_token_returns_raw_and_hash() -> None:
    raw, hashed = mint_token()
    assert raw.startswith("tok_")
    assert hashed != raw
    assert len(hashed) == 64  # sha256 hex


def test_mint_token_is_unique_per_call() -> None:
    seen = set()
    for _ in range(50):
        raw, _ = mint_token()
        assert raw not in seen
        seen.add(raw)


def test_hash_token_is_deterministic() -> None:
    raw = "tok_DETERMINISTIC"
    assert hash_token(raw) == hash_token(raw)


def test_hash_token_distinct_inputs_distinct_outputs() -> None:
    assert hash_token("tok_A") != hash_token("tok_B")


def test_hash_token_rejects_empty() -> None:
    with pytest.raises(TenancyError, match="empty"):
        hash_token("")


def test_verify_scope_full_includes_read() -> None:
    assert verify_scope("full", "read") is True
    assert verify_scope("full", "full") is True


def test_verify_scope_read_excludes_full() -> None:
    assert verify_scope("read", "read") is True
    assert verify_scope("read", "full") is False


def test_verify_scope_unknown_token_scope_raises() -> None:
    with pytest.raises(TenancyError, match="unknown token scope"):
        verify_scope("admin", "read")


def test_verify_scope_unknown_required_scope_raises() -> None:
    with pytest.raises(TenancyError, match="unknown required scope"):
        verify_scope("full", "admin")
