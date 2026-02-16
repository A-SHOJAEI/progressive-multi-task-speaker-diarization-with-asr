"""Tests for data loading and preprocessing."""

import numpy as np
import pytest
import torch

from progressive_multi_task_speaker_diarization_with_asr.data.preprocessing import (
    AudioPreprocessor,
    extract_mel_spectrogram,
)
from progressive_multi_task_speaker_diarization_with_asr.data.loader import (
    SpeechDiarizationDataset,
    collate_fn,
)


class TestAudioPreprocessor:
    """Tests for AudioPreprocessor class."""

    def test_initialization(self, preprocessor: AudioPreprocessor) -> None:
        """Test preprocessor initialization."""
        assert preprocessor.sample_rate == 16000
        assert preprocessor.n_mels == 80
        assert preprocessor.normalize is True

    def test_extract_mel_spectrogram(self) -> None:
        """Test mel spectrogram extraction."""
        # Create dummy audio
        sample_rate = 16000
        duration = 1.0  # 1 second
        waveform = torch.randn(1, int(sample_rate * duration))

        # Extract features
        mel_spec = extract_mel_spectrogram(waveform, sample_rate, n_mels=80)

        assert mel_spec.shape[0] == 1  # Channels
        assert mel_spec.shape[1] == 80  # Mel bins
        assert mel_spec.shape[2] > 0  # Time frames

    def test_augmentation(self, preprocessor: AudioPreprocessor) -> None:
        """Test audio augmentation."""
        waveform = torch.randn(1, 16000)

        # Test noise augmentation
        augmented = preprocessor.augment_audio(waveform, add_noise=True)
        assert augmented.shape == waveform.shape
        assert not torch.allclose(augmented, waveform)

        # Test time stretching
        augmented = preprocessor.augment_audio(
            waveform, time_stretch=True, stretch_factor=1.1
        )
        assert augmented.shape[1] != waveform.shape[1]


class TestSpeechDiarizationDataset:
    """Tests for SpeechDiarizationDataset class."""

    def test_initialization(
        self, sample_audio_paths: list, sample_diar_labels_np: list
    ) -> None:
        """Test dataset initialization."""
        dataset = SpeechDiarizationDataset(
            audio_paths=sample_audio_paths,
            diarization_labels=sample_diar_labels_np,
            curriculum_stage="full",
        )

        assert len(dataset) == len(sample_audio_paths)

    def test_curriculum_filtering_single(
        self, sample_audio_paths: list, sample_diar_labels_np: list
    ) -> None:
        """Test curriculum filtering for single-speaker stage."""
        # Create labels with single speaker
        single_speaker_labels = [np.ones(100, dtype=int)]

        dataset = SpeechDiarizationDataset(
            audio_paths=sample_audio_paths[:1],
            diarization_labels=single_speaker_labels,
            curriculum_stage="single",
        )

        # Should keep single-speaker samples
        assert len(dataset) <= 1

    def test_resample_labels(self) -> None:
        """Test label resampling."""
        labels = np.array([0, 1, 1, 2, 2, 3])
        target_length = 10

        resampled = SpeechDiarizationDataset._resample_labels(labels, target_length)
        assert len(resampled) == target_length


class TestCollateFunction:
    """Tests for collate function."""

    def test_collate_variable_lengths(self) -> None:
        """Test collating samples with variable lengths."""
        # Features should be (T, n_mels) format
        batch = [
            {
                "features": torch.randn(100, 80),
                "diarization_labels": torch.randint(0, 4, (100,)),
            },
            {
                "features": torch.randn(80, 80),
                "diarization_labels": torch.randint(0, 4, (80,)),
            },
            {
                "features": torch.randn(90, 80),
                "diarization_labels": torch.randint(0, 4, (90,)),
            },
        ]

        collated = collate_fn(batch)

        # Check all sequences padded to max length (B, T, n_mels)
        assert collated["features"].shape == (3, 100, 80)
        assert collated["diarization_labels"].shape == (3, 100)
        assert len(collated["lengths"]) == 3

    def test_collate_with_transcriptions(self) -> None:
        """Test collating with transcriptions."""
        # Features should be (T, n_mels) format
        batch = [
            {
                "features": torch.randn(50, 80),
                "diarization_labels": torch.randint(0, 4, (50,)),
                "transcription": "hello world",
            },
            {
                "features": torch.randn(60, 80),
                "diarization_labels": torch.randint(0, 4, (60,)),
                "transcription": "test speech",
            },
        ]

        collated = collate_fn(batch)

        assert "transcriptions" in collated
        assert len(collated["transcriptions"]) == 2
