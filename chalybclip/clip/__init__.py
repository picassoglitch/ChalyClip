"""Clip cutting + 9:16 reformat module."""

from .breakdown import ClipBreakdown, clip_breakdown
from .intelligence import Marker, MarkerKind, compute_intelligence
from .models import Clip, ClipManifest
from .overlay_burn import (
    build_ass,
    build_filter_graph,
    build_srt,
    burn_overlays,
    captions_artifact_for_clip,
)
from .ai_fixes import AIFixesResult, FixOutcome, apply_ai_fixes
from .auto_correct import (
    AUTO_CORRECTIONS_FILENAME,
    auto_correct_clip,
    load_auto_corrections,
)
from .offload import ensure_local_clip, offload_clip_artifacts
from .publishability import (
    DirectorLine,
    PublishabilityVerdict,
    PublishStatus,
    compute_publishability,
    director_lines,
)
from .scoring import AIScoreCard, compute_ai_scores
from .service import cut_clips, cut_window, load_clips
from .waveform import compute_waveform, load_or_compute as load_or_compute_waveform
from .windowing import (
    WINDOW_BANDS,
    WindowBand,
    WindowKind,
    WindowPlan,
    classify_window_kind,
    plan_clip_window,
)
from .word_captions import (
    CaptionLine,
    CaptionWord,
    Emphasis,
    captions_for_clip,
    chunk_words_to_lines,
    classify_emphasis,
    lines_to_json,
)

__all__ = [
    "AUTO_CORRECTIONS_FILENAME",
    "AIFixesResult",
    "AIScoreCard",
    "CaptionLine",
    "CaptionWord",
    "Clip",
    "ClipBreakdown",
    "ClipManifest",
    "DirectorLine",
    "Emphasis",
    "FixOutcome",
    "Marker",
    "MarkerKind",
    "PublishStatus",
    "PublishabilityVerdict",
    "WINDOW_BANDS",
    "WindowBand",
    "WindowKind",
    "WindowPlan",
    "apply_ai_fixes",
    "auto_correct_clip",
    "build_ass",
    "build_filter_graph",
    "build_srt",
    "burn_overlays",
    "captions_artifact_for_clip",
    "captions_for_clip",
    "chunk_words_to_lines",
    "classify_emphasis",
    "classify_window_kind",
    "clip_breakdown",
    "compute_ai_scores",
    "compute_intelligence",
    "compute_publishability",
    "compute_waveform",
    "cut_clips",
    "cut_window",
    "director_lines",
    "ensure_local_clip",
    "lines_to_json",
    "load_auto_corrections",
    "load_clips",
    "load_or_compute_waveform",
    "offload_clip_artifacts",
    "plan_clip_window",
]
