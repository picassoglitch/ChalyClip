"""Content-fatigue detection unit tests (Phase 4)."""

from __future__ import annotations

from chalybclip.publish.fatigue import assess_batch_fatigue, jaccard


def test_jaccard_basics() -> None:
    assert jaccard(["a", "b"], ["a", "b"]) == 1.0
    assert jaccard(["a", "b"], ["c", "d"]) == 0.0
    assert jaccard([], ["a"]) == 0.0
    assert jaccard(["a", "b", "c", "d"], ["a", "b"]) == 0.5


def test_holds_third_similar_clip() -> None:
    # Three near-identical Warzone clips; window=2 lets two through, holds 3rd.
    clips = [
        ("c1", ["warzone", "verdansk", "ak47", "funny"]),
        ("c2", ["warzone", "verdansk", "ak47", "clutch"]),
        ("c3", ["warzone", "verdansk", "ak47", "rage"]),
    ]
    verdicts = assess_batch_fatigue(clips, max_similar_per_window=2, threshold=0.5)
    held = [v.clip_id for v in verdicts if v.hold]
    assert held == ["c3"]


def test_distinct_clips_all_pass() -> None:
    clips = [
        ("c1", ["warzone", "verdansk"]),
        ("c2", ["minecraft", "build"]),
        ("c3", ["irl", "cooking"]),
    ]
    verdicts = assess_batch_fatigue(clips, threshold=0.5)
    assert all(not v.hold for v in verdicts)


def test_recent_history_saturates_theme_immediately() -> None:
    # Two of the theme already went out recently → window full → hold the first
    # matching clip in this batch.
    clips = [("c1", ["valorant", "ascent", "ace"])]
    recent = [["valorant", "ascent", "ace"], ["valorant", "ascent", "clutch"]]
    verdicts = assess_batch_fatigue(
        clips, recent_tags=recent, threshold=0.5, max_similar_per_window=2
    )
    assert verdicts[0].hold is True


def test_untagged_clip_never_held() -> None:
    clips = [("c1", []), ("c2", []), ("c3", [])]
    verdicts = assess_batch_fatigue(clips, threshold=0.1, max_similar_per_window=1)
    assert all(not v.hold for v in verdicts)
