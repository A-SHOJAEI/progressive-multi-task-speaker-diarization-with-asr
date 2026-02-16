"""Tests for training components."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from progressive_multi_task_speaker_diarization_with_asr.training.trainer import (
    ProgressiveMultiTaskTrainer,
)
from progressive_multi_task_speaker_diarization_with_asr.models.model import (
    ProgressiveMultiTaskModel,
)
from progressive_multi_task_speaker_diarization_with_asr.evaluation.metrics import (
    DiarizationMetrics,
    ASRMetrics,
)


class TestProgressiveMultiTaskTrainer:
    """Tests for ProgressiveMultiTaskTrainer."""

    @pytest.fixture
    def dummy_dataloader(self, device: torch.device) -> DataLoader:
        """Create a dummy dataloader for testing."""
        features = torch.randn(16, 50, 80)
        labels = torch.randint(0, 4, (16, 50))
        lengths = torch.full((16,), 50)

        # Create a simple dataset
        dataset = TensorDataset(features, labels, lengths)

        def collate(batch):
            feats, labs, lens = zip(*batch)
            return {
                "features": torch.stack(feats),
                "diarization_labels": torch.stack(labs),
                "lengths": torch.stack(lens),
            }

        return DataLoader(dataset, batch_size=4, collate_fn=collate)

    @pytest.fixture
    def trainer(
        self,
        model: ProgressiveMultiTaskModel,
        dummy_dataloader: DataLoader,
        device: torch.device,
    ) -> ProgressiveMultiTaskTrainer:
        """Create trainer instance."""
        return ProgressiveMultiTaskTrainer(
            model=model,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            num_speakers=3,
            device=device,
            max_epochs=5,
            patience=2,
            use_amp=False,
        )

    def test_trainer_initialization(
        self, trainer: ProgressiveMultiTaskTrainer
    ) -> None:
        """Test trainer initialization."""
        assert trainer.max_epochs == 5
        assert trainer.patience == 2
        assert trainer.current_epoch == 0
        assert trainer.best_val_loss == float("inf")

    def test_train_epoch(self, trainer: ProgressiveMultiTaskTrainer) -> None:
        """Test training for one epoch."""
        metrics = trainer.train_epoch()

        assert "train_loss" in metrics
        assert "train_diar_loss" in metrics
        assert metrics["train_loss"] >= 0

    def test_validate(self, trainer: ProgressiveMultiTaskTrainer) -> None:
        """Test validation."""
        metrics = trainer.validate()

        assert "val_loss" in metrics
        assert "val_diar_loss" in metrics
        assert metrics["val_loss"] >= 0

    def test_save_and_load_checkpoint(
        self, trainer: ProgressiveMultiTaskTrainer, tmp_path
    ) -> None:
        """Test checkpoint saving and loading."""
        trainer.checkpoint_dir = tmp_path
        trainer.save_checkpoint("test_checkpoint.pt")

        checkpoint_file = tmp_path / "test_checkpoint.pt"
        assert checkpoint_file.exists()

        # Load checkpoint
        trainer.load_checkpoint("test_checkpoint.pt")
        assert trainer.current_epoch == 0


class TestDiarizationMetrics:
    """Tests for DiarizationMetrics."""

    def test_metrics_initialization(self) -> None:
        """Test metrics initialization."""
        metrics = DiarizationMetrics(num_speakers=3)
        assert metrics.num_speakers == 3
        assert len(metrics.predictions) == 0

    def test_metrics_update_and_compute(self, device: torch.device) -> None:
        """Test updating and computing metrics."""
        metrics = DiarizationMetrics(num_speakers=3)

        predictions = torch.randint(0, 4, (4, 100), device=device)
        targets = torch.randint(0, 4, (4, 100), device=device)

        metrics.update(predictions, targets)
        results = metrics.compute()

        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1_score" in results
        assert "diarization_error_rate" in results
        assert 0 <= results["accuracy"] <= 1

    def test_confusion_matrix(self, device: torch.device) -> None:
        """Test confusion matrix generation."""
        metrics = DiarizationMetrics(num_speakers=3)

        predictions = torch.randint(0, 4, (4, 100), device=device)
        targets = torch.randint(0, 4, (4, 100), device=device)

        metrics.update(predictions, targets)
        cm = metrics.get_confusion_matrix()

        assert cm.shape == (4, 4)  # 3 speakers + silence


class TestASRMetrics:
    """Tests for ASRMetrics."""

    def test_asr_metrics_initialization(self) -> None:
        """Test ASR metrics initialization."""
        metrics = ASRMetrics(vocab_size=29)
        assert metrics.vocab_size == 29

    def test_asr_metrics_compute(self, device: torch.device) -> None:
        """Test ASR metrics computation."""
        metrics = ASRMetrics(vocab_size=29)

        predictions = torch.randint(0, 29, (4, 100), device=device)
        targets = torch.randint(0, 29, (4, 100), device=device)

        metrics.update(predictions, targets)
        results = metrics.compute()

        assert "accuracy" in results
        assert "word_error_rate" in results
        assert 0 <= results["accuracy"] <= 1
        assert 0 <= results["word_error_rate"] <= 1
