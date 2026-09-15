"""Levenshtein edit distance with early termination.

A single internal use case (fuzzy phrase matching at <=2 edits) so we keep
this hand-rolled rather than pulling in `python-Levenshtein` or `rapidfuzz`.
For longer strings the early-exit on `max_dist` keeps the typical case cheap.
"""

from __future__ import annotations


def levenshtein(a: str, b: str, *, max_dist: int | None = None) -> int:
    """Edit distance between `a` and `b`.

    If `max_dist` is set and the true distance exceeds it, the function
    may return any value `> max_dist` without computing the exact answer.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    if max_dist is not None and abs(len(a) - len(b)) > max_dist:
        return max_dist + 1

    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        row_min = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            if curr[j] < row_min:
                row_min = curr[j]
        if max_dist is not None and row_min > max_dist:
            return max_dist + 1
        prev = curr
    return prev[-1]
