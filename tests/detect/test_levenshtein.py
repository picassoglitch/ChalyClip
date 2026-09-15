"""Tests for the hand-rolled Levenshtein implementation."""

from __future__ import annotations

import pytest

from chalybclip.detect.levenshtein import levenshtein


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("", "", 0),
        ("a", "", 1),
        ("", "abc", 3),
        ("kitten", "sitting", 3),
        ("clip this", "clip this", 0),
        ("clip this", "klip thiz", 2),
        ("clipéalo", "clipealo", 1),
        ("saca un clip", "saca un clipo", 1),
        ("flaw", "lawn", 2),
    ],
)
def test_levenshtein_exact(a: str, b: str, expected: int) -> None:
    assert levenshtein(a, b) == expected


def test_levenshtein_symmetric() -> None:
    assert levenshtein("abc", "xyz") == levenshtein("xyz", "abc")


def test_levenshtein_max_dist_short_circuits() -> None:
    """When max_dist is set, the function may return any value > max_dist."""
    result = levenshtein("a" * 100, "b" * 100, max_dist=2)
    assert result > 2


def test_levenshtein_within_max_dist_is_exact() -> None:
    assert levenshtein("kitten", "sitten", max_dist=2) == 1
    assert levenshtein("kitten", "sittin", max_dist=2) == 2


def test_levenshtein_length_diff_short_circuit() -> None:
    assert levenshtein("a", "abcde", max_dist=2) > 2
