"""Series ("Parte 1/N") grouping + hook tagging — pure-function tests.

Contiguous clips are one long moment the cutter split; they publish as a
numbered series so a viewer who lands on any part goes hunting for the
rest. Interval-fallback clips are evenly spaced by construction and must
never chain into a fake series.
"""

from __future__ import annotations

from types import SimpleNamespace

from chalybclip.pipeline import _series_hook, _series_parts


def _clip(clip_id: str, start: float, end: float, reason: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=clip_id,
        start_s=start,
        end_s=end,
        candidate=SimpleNamespace(reason=reason),
    )


def test_series_parts_groups_contiguous_clips() -> None:
    clips = [
        # Deliberately out of order — grouping sorts by start_s.
        _clip("c2", 40.0, 70.0, "voice trigger"),
        _clip("c1", 0.0, 30.0, "voice trigger"),
        _clip("c3", 200.0, 230.0, "chat heat"),
    ]
    parts = _series_parts(clips)
    # c1→c2 gap is 10s (≤ 20s window) → a 2-part series; c3 stands alone.
    assert parts["c1"][:2] == (1, 2)
    assert parts["c2"][:2] == (2, 2)
    assert parts["c1"][2] == parts["c2"][2]  # same run key
    assert "c3" not in parts


def test_series_parts_never_chains_interval_clips() -> None:
    clips = [
        _clip("a", 0.0, 30.0, "interval"),
        _clip("b", 35.0, 65.0, "interval"),
    ]
    assert _series_parts(clips) == {}


def test_series_parts_gap_breaks_the_run() -> None:
    clips = [
        _clip("a", 0.0, 30.0, "voice trigger"),
        _clip("b", 100.0, 130.0, "voice trigger"),  # 70s gap > 20s window
    ]
    assert _series_parts(clips) == {}


def test_series_hook_tag_and_char_budget() -> None:
    assert _series_hook("El chat explotó", 1, 3, language="es") == (
        "El chat explotó — Parte 1/3"
    )
    assert _series_hook("boom goes the run", 1, 2, language="en").endswith(
        "— Part 1/2"
    )
    long_base = "x" * 120
    tagged = _series_hook(long_base, 2, 3, language="es")
    assert len(tagged) <= 90
    assert tagged.endswith("— Parte 2/3")


def test_series_hook_never_stacks_tags() -> None:
    # Recovering a base from a cached, already-tagged hook must not
    # produce "… — Parte 1/3 — Parte 2/3".
    assert _series_hook("Base — Parte 1/3", 2, 3, language="es") == (
        "Base — Parte 2/3"
    )
