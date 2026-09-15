"""Detection module — voice + chat heat + audio energy + visual fusion (Phase 1).

Phase 2 adds `vision_rescore` for promoting on-screen reactions over
loud-but-empty heuristic hits.
"""

from .audio_energy import detect_audio_energy
from .chat_heat import detect_chat_heat
from .framing import (
    FramingVerdict,
    RecommendedOutput,
    SourceOrientation,
    SubjectBox,
    analyze_framing,
    to_smart_crop_box,
)
from .fusion import FusionBonuses, FusionConfig, FusionWeights, fuse_candidates
from .models import Candidate, CandidateBatch, CandidateReason
from .service import (
    detect_candidates,
    detect_voice_triggers,
    fallback_interval_candidates,
    load_candidates,
    save_candidates,
)
from .viral import ViralMoment, ViralMomentList, detect_viral_moments
from .vision_rescore import RescoreOutcome, rescore_candidates
from .visual_signals import detect_visual_candidates

__all__ = [
    "Candidate",
    "CandidateBatch",
    "CandidateReason",
    "FramingVerdict",
    "FusionBonuses",
    "FusionConfig",
    "FusionWeights",
    "RecommendedOutput",
    "RescoreOutcome",
    "SourceOrientation",
    "SubjectBox",
    "ViralMoment",
    "ViralMomentList",
    "analyze_framing",
    "detect_audio_energy",
    "detect_candidates",
    "detect_chat_heat",
    "detect_viral_moments",
    "detect_visual_candidates",
    "fallback_interval_candidates",
    "detect_voice_triggers",
    "fuse_candidates",
    "load_candidates",
    "rescore_candidates",
    "save_candidates",
    "to_smart_crop_box",
]
