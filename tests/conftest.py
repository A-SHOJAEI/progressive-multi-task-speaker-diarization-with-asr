"""Pytest fixtures for testing."""

import numpy as np
import pytest
import torch

from progressive_multi_task_speaker_diarization_with_asr.models.model import (
    ProgressiveMultiTaskModel,
)
from progressive_multi_task_speaker_diarization_with_asr.data.preprocessing import (
    AudioPreprocessor,
)


@pytest.fixture
def device() -> torch.device:
    """Get computation device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def model_config() -> dict:
    """Get default model configuration."""
    return {
        "input_dim": 80,
        "hidden_dim": 128,
        "num_speakers": 3,
        "vocab_size": 29,
        "num_encoder_layers": 2,
        "num_attention_heads": 4,
        "dropout": 0.1,
        "use_cross_attention": True,
    }


@pytest.fixture
def model(model_config: dict, device: torch.device) -> ProgressiveMultiTaskModel:
    """Create test model instance."""
    model = ProgressiveMultiTaskModel(**model_config)
    return model.to(device)


@pytest.fixture
def preprocessor() -> AudioPreprocessor:
    """Create audio preprocessor instance."""
    return AudioPreprocessor(sample_rate=16000, n_mels=80)


@pytest.fixture
def sample_features(device: torch.device) -> torch.Tensor:
    """Generate sample mel spectrogram features."""
    batch_size, seq_len, n_mels = 4, 100, 80
    return torch.randn(batch_size, seq_len, n_mels, device=device)


@pytest.fixture
def sample_diarization_labels(device: torch.device) -> torch.Tensor:
    """Generate sample diarization labels."""
    batch_size, seq_len = 4, 100
    # Random speaker labels (0 for silence, 1-3 for speakers)
    return torch.randint(0, 4, (batch_size, seq_len), device=device)


@pytest.fixture
def sample_lengths(device: torch.device) -> torch.Tensor:
    """Generate sample sequence lengths."""
    return torch.tensor([100, 95, 85, 90], device=device)


@pytest.fixture
def sample_audio_paths() -> list:
    """Generate sample audio file paths."""
    return [
        "audio_sample_1.wav",
        "audio_sample_2.wav",
        "audio_sample_3.wav",
    ]


@pytest.fixture
def sample_diar_labels_np() -> list:
    """Generate sample diarization labels as numpy arrays."""
    return [
        np.random.randint(0, 4, size=100),
        np.random.randint(0, 4, size=95),
        np.random.randint(0, 4, size=85),
    ]
