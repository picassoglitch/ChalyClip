"""Render T3 — balance chip endpoint coerces all balance values to
scalar shapes before the template renders. Eliminates the
`balance_chip.render_failed: unhashable type: 'dict'` spam in prod.

The exact root cause was unknown (could be a schema drift, a JSON
nested-dict landing in monthly_used, or a Jinja extension hashing
under the hood). Easier to scrub upfront than chase it; these
tests pin the contract so a future refactor doesn't drop the
coercion.
"""

from __future__ import annotations

from chalybclip.api.routers.dashboard import _coerce_balance_to_scalars


def test_none_balance_passes_through_as_none() -> None:
    assert _coerce_balance_to_scalars(None) is None


def test_non_dict_balance_collapses_to_none() -> None:
    """A list / model / unexpected object should NOT be passed to the
    template — the template expects dict-like with .get(). Returning
    None makes the template fall through to the 'no balance' branch."""
    assert _coerce_balance_to_scalars("not a dict") is None
    assert _coerce_balance_to_scalars([1, 2, 3]) is None
    assert _coerce_balance_to_scalars(42) is None


def test_well_formed_balance_passes_values() -> None:
    bal = {
        "remaining": 12_000,
        "unlimited": False,
        "monthly_used": 5_000,
        "at": "2026-06-03T12:00:00Z",
    }
    out = _coerce_balance_to_scalars(bal)
    # Token T1 — the coercer now also carries the usage-report status
    # (report_ok / report_error). Absent in the input → None (tri-state
    # "never reported"), not a false failure.
    assert out["remaining"] == 12_000
    assert out["unlimited"] is False
    assert out["monthly_used"] == 5_000
    assert out["at"] == "2026-06-03T12:00:00Z"
    assert out["report_ok"] is None
    assert out["report_error"] is None


def test_nested_dict_in_monthly_used_collapses_to_zero() -> None:
    """The exact-shape repro that suggested 'unhashable type: dict' —
    if a nested dict ever lands in monthly_used (or remaining), the
    template's '{:,}'.format() would TypeError. Coerce to 0 instead."""
    bal = {
        "remaining": 12_000,
        "unlimited": False,
        "monthly_used": {"foo": "bar"},
        "at": None,
    }
    out = _coerce_balance_to_scalars(bal)
    assert out["monthly_used"] == 0
    assert out["remaining"] == 12_000


def test_dict_in_remaining_collapses_to_zero() -> None:
    bal = {
        "remaining": {"nested": True},
        "unlimited": False,
        "monthly_used": 0,
        "at": None,
    }
    out = _coerce_balance_to_scalars(bal)
    assert out["remaining"] == 0


def test_dict_in_unlimited_does_not_become_true() -> None:
    """A non-empty dict is truthy in Python so naive bool(dict) returns
    True — that'd flip the chip to the 'infinity' display incorrectly.
    Explicit isinstance check defends against that."""
    bal = {
        "remaining": 1,
        "unlimited": {"flag": True},
        "monthly_used": 1,
        "at": None,
    }
    out = _coerce_balance_to_scalars(bal)
    assert out["unlimited"] is False


def test_dict_in_at_collapses_to_none() -> None:
    bal = {
        "remaining": 1,
        "unlimited": False,
        "monthly_used": 1,
        "at": {"iso": "2026-06-03"},
    }
    out = _coerce_balance_to_scalars(bal)
    assert out["at"] is None


def test_missing_keys_default_to_safe_values() -> None:
    """Partial balance dicts (future schema additions / older callers)
    should still render cleanly."""
    out = _coerce_balance_to_scalars({})
    assert out == {
        "remaining": 0,
        "unlimited": False,
        "monthly_used": 0,
        "at": None,
        # Token T1 — report status absent → tri-state None (never
        # reported), never a false "failing" signal.
        "report_ok": None,
        "report_error": None,
    }


def test_float_remaining_becomes_int() -> None:
    bal = {
        "remaining": 12_345.6,
        "unlimited": False,
        "monthly_used": 5_000.0,
        "at": None,
    }
    out = _coerce_balance_to_scalars(bal)
    assert out["remaining"] == 12_345
    assert out["monthly_used"] == 5_000
    assert isinstance(out["remaining"], int)


def test_bool_remaining_stays_an_int() -> None:
    """bool is a subclass of int — bool(True) → 1 is fine for the
    chip's display math; just don't crash."""
    bal = {
        "remaining": True,
        "unlimited": False,
        "monthly_used": 0,
        "at": None,
    }
    out = _coerce_balance_to_scalars(bal)
    assert out["remaining"] == 1
