"""Tests for the tenancy contextvar contract."""

from __future__ import annotations

import asyncio

import pytest

from chalybclip.errors import TenancyError
from chalybclip.tenancy import (
    assert_tenant,
    bound_tenant,
    current_tenant_id,
    current_tenant_id_optional,
)


def test_current_tenant_id_unbound_raises() -> None:
    with pytest.raises(TenancyError, match="no tenant bound"):
        current_tenant_id()


def test_current_tenant_id_optional_returns_none_when_unbound() -> None:
    assert current_tenant_id_optional() is None


def test_bound_tenant_sets_and_clears() -> None:
    with bound_tenant("ten_a"):
        assert current_tenant_id() == "ten_a"
    with pytest.raises(TenancyError):
        current_tenant_id()


def test_bound_tenant_nests_correctly() -> None:
    with bound_tenant("ten_a"):
        assert current_tenant_id() == "ten_a"
        with bound_tenant("ten_b"):
            assert current_tenant_id() == "ten_b"
        assert current_tenant_id() == "ten_a"


def test_bound_tenant_restores_on_exception() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with bound_tenant("ten_a"):
            raise RuntimeError("boom")
    assert current_tenant_id_optional() is None


def test_bound_tenant_restores_outer_on_inner_exception() -> None:
    with bound_tenant("ten_a"):
        with pytest.raises(RuntimeError, match="boom"):
            with bound_tenant("ten_b"):
                raise RuntimeError("boom")
        assert current_tenant_id() == "ten_a"


def test_bound_tenant_rejects_empty_string() -> None:
    with pytest.raises(TenancyError, match="non-empty"):
        with bound_tenant(""):
            pass


def test_assert_tenant_passes_on_match() -> None:
    with bound_tenant("ten_a"):
        assert_tenant("ten_a")  # no raise


def test_assert_tenant_raises_on_mismatch() -> None:
    with bound_tenant("ten_a"):
        with pytest.raises(TenancyError, match="tenant mismatch"):
            assert_tenant("ten_b")


def test_assert_tenant_raises_when_unbound() -> None:
    with pytest.raises(TenancyError):
        assert_tenant("ten_a")


def test_contextvars_isolated_across_async_tasks() -> None:
    """ContextVar binding doesn't leak across `asyncio.gather` siblings."""

    async def task(tenant: str) -> str:
        with bound_tenant(tenant):
            await asyncio.sleep(0)
            return current_tenant_id()

    async def main() -> tuple[str, str]:
        return await asyncio.gather(task("ten_a"), task("ten_b"))

    a, b = asyncio.run(main())
    assert a == "ten_a"
    assert b == "ten_b"
