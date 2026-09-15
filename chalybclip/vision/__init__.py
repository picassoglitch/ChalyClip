"""Local vision pipeline — scene cuts, motion energy, face/emotion.

Phase 1 ships heuristic detectors only (PySceneDetect ContentDetector,
OpenCV frame-diff, MediaPipe Face Mesh + smile heuristic). Multimodal
scoring with a vision LLM is Phase 2.
"""

from .face_emotion import detect_face_emotions
from .frame_sampler import sample_frames, save_frames
from .models import (
    FaceEmotion,
    FaceFrame,
    MotionFrame,
    SceneCut,
    VisualSignal,
    VisualSignalTrack,
)
from .motion import compute_motion_energy
from .scene_detect import detect_scene_cuts
from .service import analyze_video, load_visual_signals, visual_signals_path

__all__ = [
    "FaceEmotion",
    "FaceFrame",
    "MotionFrame",
    "SceneCut",
    "VisualSignal",
    "VisualSignalTrack",
    "analyze_video",
    "compute_motion_energy",
    "detect_face_emotions",
    "detect_scene_cuts",
    "load_visual_signals",
    "sample_frames",
    "save_frames",
    "visual_signals_path",
]
