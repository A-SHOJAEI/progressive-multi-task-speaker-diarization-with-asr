"""Tests for model components."""

import pytest
import torch

from progressive_multi_task_speaker_diarization_with_asr.models.model import (
    ProgressiveMultiTaskModel,
    SpeakerEncoder,
    DiarizationHead,
    ASRHead,
)
from progressive_multi_task_speaker_diarization_with_asr.models.components import (
    CrossTaskAttention,
    CurriculumScheduler,
    MultiTaskLoss,
)


class TestProgressiveMultiTaskModel:
    """Tests for ProgressiveMultiTaskModel."""

    def test_model_initialization(self, model: ProgressiveMultiTaskModel) -> None:
        """Test model initialization."""
        assert model.input_dim == 80
        assert model.hidden_dim == 128
        assert model.num_speakers == 3
        assert model.vocab_size == 29

    def test_forward_pass(
        self,
        model: ProgressiveMultiTaskModel,
        sample_features: torch.Tensor,
        sample_lengths: torch.Tensor,
    ) -> None:
        """Test forward pass through model."""
        outputs = model(sample_features, sample_lengths)

        assert "diarization_logits" in outputs
        assert "asr_logits" in outputs
        assert "diar_features" in outputs
        assert "asr_features" in outputs

        batch_size, seq_len, _ = sample_features.shape
        assert outputs["diarization_logits"].shape == (batch_size, seq_len, 4)  # 3 speakers + 1 silence
        assert outputs["asr_logits"].shape == (batch_size, seq_len, 29)

    def test_predict_diarization(
        self, model: ProgressiveMultiTaskModel, sample_features: torch.Tensor
    ) -> None:
        """Test diarization prediction."""
        predictions = model.predict_diarization(sample_features)

        batch_size, seq_len, _ = sample_features.shape
        assert predictions.shape == (batch_size, seq_len)
        assert predictions.min() >= 0
        assert predictions.max() <= 3

    def test_predict_asr(
        self, model: ProgressiveMultiTaskModel, sample_features: torch.Tensor
    ) -> None:
        """Test ASR prediction."""
        predictions = model.predict_asr(sample_features)

        batch_size, seq_len, _ = sample_features.shape
        assert predictions.shape == (batch_size, seq_len)


class TestSpeakerEncoder:
    """Tests for SpeakerEncoder."""

    def test_speaker_encoder(self, device: torch.device) -> None:
        """Test speaker encoder forward pass."""
        encoder = SpeakerEncoder(input_dim=128, hidden_dim=64, num_layers=2).to(device)

        features = torch.randn(4, 100, 128, device=device)
        lengths = torch.tensor([100, 95, 90, 85], device=device)

        output = encoder(features, lengths)

        assert output.shape == (4, 100, 128)  # bidirectional: 64 * 2 = 128


class TestDiarizationHead:
    """Tests for DiarizationHead."""

    def test_diarization_head(self, device: torch.device) -> None:
        """Test diarization head."""
        head = DiarizationHead(input_dim=128, num_speakers=3).to(device)

        features = torch.randn(4, 100, 128, device=device)
        logits = head(features)

        assert logits.shape == (4, 100, 4)  # 3 speakers + silence


class TestASRHead:
    """Tests for ASRHead."""

    def test_asr_head(self, device: torch.device) -> None:
        """Test ASR head."""
        head = ASRHead(input_dim=128, vocab_size=29).to(device)

        features = torch.randn(4, 100, 128, device=device)
        logits = head(features)

        assert logits.shape == (4, 100, 29)


class TestCrossTaskAttention:
    """Tests for CrossTaskAttention."""

    def test_cross_attention(self, device: torch.device) -> None:
        """Test cross-task attention mechanism."""
        attn = CrossTaskAttention(hidden_dim=128, num_heads=4).to(device)

        query = torch.randn(4, 100, 128, device=device)
        key_value = torch.randn(4, 100, 128, device=device)

        output = attn(query, key_value)

        assert output.shape == query.shape

    def test_cross_attention_with_mask(self, device: torch.device) -> None:
        """Test cross-attention with mask."""
        attn = CrossTaskAttention(hidden_dim=128, num_heads=4).to(device)

        query = torch.randn(4, 100, 128, device=device)
        key_value = torch.randn(4, 100, 128, device=device)
        mask = torch.ones(4, 100, device=device)
        mask[:, 90:] = 0  # Mask last 10 frames

        output = attn(query, key_value, mask)

        assert output.shape == query.shape


class TestCurriculumScheduler:
    """Tests for CurriculumScheduler."""

    def test_scheduler_initialization(self) -> None:
        """Test curriculum scheduler initialization."""
        scheduler = CurriculumScheduler(total_epochs=90)

        assert len(scheduler.stages) == 3
        assert scheduler.total_epochs == 90

    def test_get_stage(self) -> None:
        """Test getting curriculum stage."""
        scheduler = CurriculumScheduler(total_epochs=90)

        assert scheduler.get_stage(0) == "single"
        assert scheduler.get_stage(25) == "single"
        assert scheduler.get_stage(35) == "overlap"
        assert scheduler.get_stage(70) == "full"

    def test_get_stage_progress(self) -> None:
        """Test getting stage progress."""
        scheduler = CurriculumScheduler(total_epochs=90)

        progress = scheduler.get_stage_progress(0)
        assert 0.0 <= progress <= 1.0

        progress = scheduler.get_stage_progress(29)
        assert 0.9 < progress <= 1.0


class TestMultiTaskLoss:
    """Tests for MultiTaskLoss."""

    def test_loss_initialization(self) -> None:
        """Test loss initialization."""
        criterion = MultiTaskLoss(num_speakers=3, use_dynamic_weighting=True)

        assert criterion.num_speakers == 3
        assert criterion.use_dynamic_weighting is True

    def test_diarization_loss(self, device: torch.device) -> None:
        """Test diarization loss computation."""
        criterion = MultiTaskLoss(num_speakers=3).to(device)

        diar_logits = torch.randn(4, 100, 4, device=device)
        diar_labels = torch.randint(0, 4, (4, 100), device=device)

        loss, losses = criterion(diar_logits, diar_labels)

        assert isinstance(loss, torch.Tensor)
        assert "diarization_loss" in losses
        assert "total_loss" in losses

    def test_get_task_weights(self, device: torch.device) -> None:
        """Test getting task weights."""
        criterion = MultiTaskLoss(num_speakers=3).to(device)

        weights = criterion.get_task_weights()

        assert "diarization" in weights
        assert "asr" in weights
        assert all(w > 0 for w in weights.values())
