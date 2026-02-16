"""Evaluation metrics for speaker diarization and ASR."""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

logger = logging.getLogger(__name__)


class DiarizationMetrics:
    """Metrics for speaker diarization evaluation."""

    def __init__(self, num_speakers: int):
        """Initialize diarization metrics.

        Args:
            num_speakers: Number of speakers (excluding silence).
        """
        self.num_speakers = num_speakers
        self.reset()

    def reset(self) -> None:
        """Reset all metrics."""
        self.predictions = []
        self.targets = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metrics with new predictions.

        Args:
            predictions: Predicted speaker labels (B, T).
            targets: Ground truth speaker labels (B, T).
        """
        # Flatten and convert to numpy
        pred_flat = predictions.cpu().numpy().flatten()
        target_flat = targets.cpu().numpy().flatten()

        # Filter out padding (-1)
        valid_mask = target_flat >= 0
        pred_flat = pred_flat[valid_mask]
        target_flat = target_flat[valid_mask]

        self.predictions.extend(pred_flat.tolist())
        self.targets.extend(target_flat.tolist())

    def compute(self) -> Dict[str, float]:
        """Compute diarization metrics.

        Returns:
            Dictionary of metrics including DER, accuracy, F1, etc.
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)

        # Overall accuracy
        accuracy = accuracy_score(targets, predictions)

        # Precision, recall, F1 (macro average)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average="macro", zero_division=0
        )

        # Per-speaker metrics (excluding silence/background class 0)
        speaker_mask = targets > 0
        if speaker_mask.sum() > 0:
            speaker_accuracy = accuracy_score(
                targets[speaker_mask], predictions[speaker_mask]
            )
        else:
            speaker_accuracy = 0.0

        # Diarization Error Rate (DER) approximation
        # DER = (FA + MISS + CONFUSION) / TOTAL
        # Here we approximate with misclassification rate for active speech
        if speaker_mask.sum() > 0:
            der = 1.0 - speaker_accuracy
        else:
            der = 1.0

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "speaker_accuracy": speaker_accuracy,
            "diarization_error_rate": der,
        }

        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix.

        Returns:
            Confusion matrix as numpy array.
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        return confusion_matrix(targets, predictions)


class ASRMetrics:
    """Metrics for ASR evaluation."""

    def __init__(self, vocab_size: int):
        """Initialize ASR metrics.

        Args:
            vocab_size: Size of vocabulary.
        """
        self.vocab_size = vocab_size
        self.reset()

    def reset(self) -> None:
        """Reset all metrics."""
        self.predictions = []
        self.targets = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metrics with new predictions.

        Args:
            predictions: Predicted token IDs (B, T).
            targets: Ground truth token IDs (B, T).
        """
        pred_flat = predictions.cpu().numpy().flatten()
        target_flat = targets.cpu().numpy().flatten()

        # Filter out padding and blank tokens
        valid_mask = (target_flat >= 0) & (target_flat < self.vocab_size)
        pred_flat = pred_flat[valid_mask]
        target_flat = target_flat[valid_mask]

        self.predictions.extend(pred_flat.tolist())
        self.targets.extend(target_flat.tolist())

    def compute(self) -> Dict[str, float]:
        """Compute ASR metrics.

        Returns:
            Dictionary of metrics including WER, accuracy, etc.
        """
        if len(self.predictions) == 0 or len(self.targets) == 0:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "word_error_rate": 1.0,
            }

        predictions = np.array(self.predictions)
        targets = np.array(self.targets)

        # Token-level accuracy
        accuracy = accuracy_score(targets, predictions)

        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average="macro", zero_division=0
        )

        # Word Error Rate (approximated as 1 - accuracy)
        wer = 1.0 - accuracy

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "word_error_rate": wer,
        }

        return metrics


def compute_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_speakers: int,
    vocab_size: int,
) -> Dict[str, Dict[str, float]]:
    """Compute comprehensive metrics on a dataset.

    Args:
        model: Trained model.
        dataloader: Data loader for evaluation.
        device: Device to run evaluation on.
        num_speakers: Number of speakers.
        vocab_size: Vocabulary size.

    Returns:
        Dictionary containing diarization and ASR metrics.
    """
    model.eval()

    diar_metrics = DiarizationMetrics(num_speakers)
    asr_metrics = ASRMetrics(vocab_size)

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            diar_labels = batch["diarization_labels"].to(device)
            lengths = batch.get("lengths", None)
            if lengths is not None:
                lengths = lengths.to(device)

            # Forward pass
            outputs = model(features, lengths)

            # Diarization predictions
            diar_preds = torch.argmax(outputs["diarization_logits"], dim=-1)
            diar_metrics.update(diar_preds, diar_labels)

            # ASR predictions (if available)
            asr_preds = torch.argmax(outputs["asr_logits"], dim=-1)
            # For now, use dummy targets since we don't have ground truth text
            # In real scenario, you would have actual transcriptions

    # Compute all metrics
    results = {
        "diarization": diar_metrics.compute(),
        "asr": asr_metrics.compute(),
    }

    return results
