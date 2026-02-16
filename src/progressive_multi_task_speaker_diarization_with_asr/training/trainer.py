"""Training utilities for progressive multi-task learning."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from progressive_multi_task_speaker_diarization_with_asr.models.model import (
    ProgressiveMultiTaskModel,
)
from progressive_multi_task_speaker_diarization_with_asr.models.components import (
    MultiTaskLoss,
    CurriculumScheduler,
)

logger = logging.getLogger(__name__)


class ProgressiveMultiTaskTrainer:
    """Trainer for progressive multi-task speaker diarization and ASR."""

    def __init__(
        self,
        model: ProgressiveMultiTaskModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_speakers: int,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        max_epochs: int = 100,
        patience: int = 10,
        gradient_clip: float = 5.0,
        use_amp: bool = True,
        checkpoint_dir: str = "checkpoints",
        use_curriculum: bool = True,
    ):
        """Initialize trainer.

        Args:
            model: Progressive multi-task model.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_speakers: Number of speakers for diarization.
            device: Device to train on.
            learning_rate: Learning rate.
            weight_decay: Weight decay for optimizer.
            max_epochs: Maximum number of epochs.
            patience: Early stopping patience.
            gradient_clip: Gradient clipping threshold.
            use_amp: Whether to use automatic mixed precision.
            checkpoint_dir: Directory to save checkpoints.
            use_curriculum: Whether to use curriculum learning.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.gradient_clip = gradient_clip
        self.use_amp = use_amp

        # Optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_epochs)

        # Loss function with dynamic weighting
        self.criterion = MultiTaskLoss(
            num_speakers=num_speakers, use_dynamic_weighting=True
        ).to(device)

        # Curriculum scheduler
        self.curriculum_scheduler = None
        if use_curriculum:
            self.curriculum_scheduler = CurriculumScheduler(max_epochs)

        # Mixed precision training
        self.scaler = GradScaler('cuda') if use_amp else None

        # Checkpoint management
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.history = {"train_loss": [], "val_loss": []}

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        total_loss = 0.0
        total_diar_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            features = batch["features"].to(self.device)
            diar_labels = batch["diarization_labels"].to(self.device)
            lengths = batch["lengths"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(features, lengths)
                    loss, losses = self.criterion(
                        outputs["diarization_logits"], diar_labels
                    )
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(features, lengths)
                loss, losses = self.criterion(outputs["diarization_logits"], diar_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

            total_loss += losses["total_loss"]
            total_diar_loss += losses["diarization_loss"]
            num_batches += 1

        metrics = {
            "train_loss": total_loss / num_batches,
            "train_diar_loss": total_diar_loss / num_batches,
        }

        return metrics

    def validate(self) -> Dict[str, float]:
        """Validate the model.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_diar_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch["features"].to(self.device)
                diar_labels = batch["diarization_labels"].to(self.device)
                lengths = batch["lengths"].to(self.device)

                if self.use_amp:
                    with autocast('cuda'):
                        outputs = self.model(features, lengths)
                        loss, losses = self.criterion(
                            outputs["diarization_logits"], diar_labels
                        )
                else:
                    outputs = self.model(features, lengths)
                    loss, losses = self.criterion(outputs["diarization_logits"], diar_labels)

                total_loss += losses["total_loss"]
                total_diar_loss += losses["diarization_loss"]
                num_batches += 1

        metrics = {
            "val_loss": total_loss / num_batches,
            "val_diar_loss": total_diar_loss / num_batches,
        }

        return metrics

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.

        Args:
            filename: Name of checkpoint file.
        """
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_loss": self.best_val_loss,
                "history": self.history,
            },
            checkpoint_path,
        )
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint.

        Args:
            filename: Name of checkpoint file.
        """
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def train(self) -> Dict[str, list]:
        """Run full training loop.

        Returns:
            Training history.
        """
        logger.info(f"Starting training for {self.max_epochs} epochs")

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch

            # Get curriculum stage if enabled
            if self.curriculum_scheduler:
                stage = self.curriculum_scheduler.get_stage(epoch)
                logger.info(f"Epoch {epoch+1}/{self.max_epochs} - Curriculum stage: {stage}")

            # Train for one epoch
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update learning rate
            self.scheduler.step()

            # Log metrics
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch+1}/{self.max_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"LR: {current_lr:.6f}"
            )

            # Update history
            self.history["train_loss"].append(train_metrics["train_loss"])
            self.history["val_loss"].append(val_metrics["val_loss"])

            # Save best model
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.epochs_without_improvement = 0
                self.save_checkpoint("best_model.pt")
                logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(
                    f"Early stopping after {epoch+1} epochs "
                    f"({self.patience} epochs without improvement)"
                )
                break

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")

        logger.info("Training completed")
        return self.history
