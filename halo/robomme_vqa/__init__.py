"""Offline-only VQA corpus construction for the HALO RoboMME adaptation.

This package is deliberately separate from the deployed policy. The VLM is
used to construct and judge training records; policy inference never imports
or calls it.
"""

from .contracts import Candidate, JudgeResult, ModelConfig, load_model_config
from .pipeline import OfflineVQAPipeline, VQARequest
from .timeline import TimelineFrame, select_causal_evidence

__all__ = [
    "Candidate",
    "JudgeResult",
    "ModelConfig",
    "OfflineVQAPipeline",
    "TimelineFrame",
    "VQARequest",
    "load_model_config",
    "select_causal_evidence",
]
