#!/usr/bin/env python
"""Evaluation script for progressive multi-task speaker diarization with ASR."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from progressive_multi_task_speaker_diarization_with_asr.models.model import (
    ProgressiveMultiTaskModel,
)
from progressive_multi_task_speaker_diarization_with_asr.data.loader import (
    create_dataloaders,
)
from progressive_multi_task_speaker_diarization_with_asr.evaluation.metrics import (
    DiarizationMetrics,
    compute_metrics,
)
from progressive_multi_task_speaker_diarization_with_asr.evaluation.analysis import (
    plot_confusion_matrix,
    analyze_per_speaker_performance,
)
from progressive_multi_task_speaker_diarization_with_asr.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_test_data(num_samples: int = 50, num_speakers: int = 4) -> tuple:
    """Generate synthetic test data.

    Args:
        num_samples: Number of test samples.
        num_speakers: Number of speakers.

    Returns:
        Tuple of (audio_paths, diarization_labels, transcriptions).
    """
    audio_paths = [f"test_audio_{i}.wav" for i in range(num_samples)]
    diarization_labels = []

    for _ in range(num_samples):
        seq_len = np.random.randint(50, 150)
        labels = []
        current_speaker = np.random.randint(0, num_speakers + 1)
        for i in range(seq_len):
            if i % 20 == 0 and np.random.random() < 0.3:
                current_speaker = np.random.randint(0, num_speakers + 1)
            labels.append(current_speaker)
        diarization_labels.append(np.array(labels))

    transcriptions = ["test transcription"] * num_samples
    return audio_paths, diarization_labels, transcriptions


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate progressive multi-task speaker diarization model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation_results.json",
        help="Path to save evaluation results",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create results directory
    results_dir = Path(args.output).parent
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate test data
    logger.info("Generating test data...")
    test_audio, test_labels, test_trans = generate_test_data(
        num_samples=50, num_speakers=config["model"]["num_speakers"]
    )

    # Create dummy train data for dataloader creation
    dummy_audio, dummy_labels, dummy_trans = generate_test_data(
        num_samples=10, num_speakers=config["model"]["num_speakers"]
    )

    # Create dataloaders
    _, test_loader = create_dataloaders(
        train_audio_paths=dummy_audio,
        train_diar_labels=dummy_labels,
        val_audio_paths=test_audio,
        val_diar_labels=test_labels,
        train_transcriptions=dummy_trans,
        val_transcriptions=test_trans,
        batch_size=config["training"].get("batch_size", 16),
        curriculum_stage="full",
        num_workers=0,
        augment_train=False,
    )

    # Load model
    logger.info("Loading model...")
    model = ProgressiveMultiTaskModel(**config["model"])

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded model from {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}, using untrained model")

    model = model.to(device)
    model.eval()

    # Evaluate model
    logger.info("Evaluating model...")
    metrics = compute_metrics(
        model=model,
        dataloader=test_loader,
        device=device,
        num_speakers=config["model"]["num_speakers"],
        vocab_size=config["model"]["vocab_size"],
    )

    # Compute confusion matrix
    diar_metrics = DiarizationMetrics(config["model"]["num_speakers"])

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            diar_labels = batch["diarization_labels"].to(device)
            lengths = batch.get("lengths", None)
            if lengths is not None:
                lengths = lengths.to(device)

            outputs = model(features, lengths)
            diar_preds = torch.argmax(outputs["diarization_logits"], dim=-1)
            diar_metrics.update(diar_preds, diar_labels)

    confusion_mat = diar_metrics.get_confusion_matrix()

    # Per-speaker analysis
    per_speaker = analyze_per_speaker_performance(
        confusion_mat, config["model"]["num_speakers"]
    )

    # Prepare results
    results = {
        "overall_metrics": {
            "diarization": metrics["diarization"],
            "asr": metrics["asr"],
        },
        "per_speaker_metrics": {
            f"speaker_{k}": v for k, v in per_speaker.items()
        },
        "confusion_matrix": confusion_mat.tolist(),
    }

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Plot confusion matrix
    cm_path = results_dir / "confusion_matrix.png"
    speaker_names = ["Silence"] + [
        f"Speaker {i}" for i in range(1, config["model"]["num_speakers"] + 1)
    ]
    plot_confusion_matrix(confusion_mat, class_names=speaker_names, save_path=str(cm_path))

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\nDiarization Metrics:")
    for metric, value in metrics["diarization"].items():
        print(f"  {metric:30s}: {value:.4f}")

    print("\nASR Metrics:")
    for metric, value in metrics["asr"].items():
        print(f"  {metric:30s}: {value:.4f}")

    print("\nPer-Speaker Performance:")
    for speaker, speaker_metrics in per_speaker.items():
        print(f"\n  Speaker {speaker}:")
        for metric, value in speaker_metrics.items():
            print(f"    {metric:28s}: {value:.4f}")

    print("\n" + "=" * 60)
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
