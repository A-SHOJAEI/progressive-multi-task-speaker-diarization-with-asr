"""Progressive Multi-Task Speaker Diarization with ASR.

A progressive multi-task learning system that jointly optimizes speaker diarization
and automatic speech recognition through curriculum-based task scheduling.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from progressive_multi_task_speaker_diarization_with_asr.models.model import (
    ProgressiveMultiTaskModel,
)
from progressive_multi_task_speaker_diarization_with_asr.training.trainer import (
    ProgressiveMultiTaskTrainer,
)

__all__ = [
    "ProgressiveMultiTaskModel",
    "ProgressiveMultiTaskTrainer",
]
