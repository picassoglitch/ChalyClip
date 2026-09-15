"""Hands-free auto-publish score extraction.

Regression for `pipeline.autopublish_handsfree_failed error="'Candidate'
object has no attribute 'id'"` — the pipeline built the (clip_id, score)
list for the hands-free sweep by joining a `candidates` list on `c.id`,
but the detect Candidate has no `.id` and the domain Clip has no
`.candidate_id`. The crash silently blocked ALL hands-free auto-publish.
"""

from __future__ import annotations

from types import SimpleNamespace

from chalybclip.detect.models import Candidate
from chalybclip.pipeline import _handsfree_clip_scores


def test_candidate_has_no_id_attr() -> None:
    # Pins the root cause: accessing `.id` on a detect Candidate raises.
    cand = Candidate(timestamp=12.0, score=0.91, reason="viral")
    assert not hasattr(cand, "id")


def test_handsfree_clip_scores_reads_score_off_each_clip() -> None:
    entries = [
        SimpleNamespace(
            clip=SimpleNamespace(
                id="clp_a", candidate=Candidate(timestamp=1.0, score=0.95),
            )
        ),
        SimpleNamespace(
            clip=SimpleNamespace(
                id="clp_b", candidate=Candidate(timestamp=2.0, score=0.40),
            )
        ),
    ]
    assert _handsfree_clip_scores(entries) == [("clp_a", 0.95), ("clp_b", 0.40)]


def test_handsfree_clip_scores_empty() -> None:
    assert _handsfree_clip_scores([]) == []
