"""Results analysis and visualization utilities."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> None:
    """Plot training and validation loss curves.

    Args:
        history: Dictionary containing training history.
        save_path: Optional path to save the plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    ax.plot(epochs, history["val_loss"], label="Validation Loss", marker="s")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved training curves to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(
    confusion_mat: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot confusion matrix heatmap.

    Args:
        confusion_mat: Confusion matrix as numpy array.
        class_names: Optional list of class names.
        save_path: Optional path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Normalize confusion matrix
    confusion_mat_norm = confusion_mat.astype("float") / (
        confusion_mat.sum(axis=1)[:, np.newaxis] + 1e-10
    )

    sns.heatmap(
        confusion_mat_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names or range(len(confusion_mat)),
        yticklabels=class_names or range(len(confusion_mat)),
        ax=ax,
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (Normalized)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_diarization_timeline(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    time_step: float = 0.01,
    save_path: Optional[str] = None,
) -> None:
    """Plot diarization timeline comparison.

    Args:
        predictions: Predicted speaker labels over time.
        ground_truth: Ground truth speaker labels over time.
        time_step: Time step in seconds for each frame.
        save_path: Optional path to save the plot.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    time_axis = np.arange(len(predictions)) * time_step

    # Plot predictions
    ax1.plot(time_axis, predictions, label="Predicted", linewidth=1)
    ax1.set_ylabel("Speaker ID")
    ax1.set_title("Predicted Diarization")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.5, max(predictions.max(), ground_truth.max()) + 0.5)

    # Plot ground truth
    ax2.plot(time_axis, ground_truth, label="Ground Truth", linewidth=1, color="orange")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Speaker ID")
    ax2.set_title("Ground Truth Diarization")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.5, max(predictions.max(), ground_truth.max()) + 0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved diarization timeline to {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_per_speaker_performance(
    confusion_mat: np.ndarray,
    num_speakers: int,
) -> Dict[int, Dict[str, float]]:
    """Analyze per-speaker performance metrics.

    Args:
        confusion_mat: Confusion matrix.
        num_speakers: Number of speakers (excluding silence).

    Returns:
        Dictionary mapping speaker ID to metrics.
    """
    per_speaker_metrics = {}

    for speaker_id in range(1, num_speakers + 1):  # Skip silence (0)
        if speaker_id >= len(confusion_mat):
            continue

        true_positives = confusion_mat[speaker_id, speaker_id]
        false_positives = confusion_mat[:, speaker_id].sum() - true_positives
        false_negatives = confusion_mat[speaker_id, :].sum() - true_positives

        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        per_speaker_metrics[speaker_id] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    return per_speaker_metrics
