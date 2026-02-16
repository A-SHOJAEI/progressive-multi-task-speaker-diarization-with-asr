#!/usr/bin/env python
"""Training script for progressive multi-task speaker diarization with ASR."""

import argparse
import json
import logging
import random
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
from progressive_multi_task_speaker_diarization_with_asr.training.trainer import (
    ProgressiveMultiTaskTrainer,
)
from progressive_multi_task_speaker_diarization_with_asr.data.loader import (
    create_dataloaders,
)
from progressive_multi_task_speaker_diarization_with_asr.utils.config import load_config
from progressive_multi_task_speaker_diarization_with_asr.evaluation.analysis import (
    plot_training_curves,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def generate_synthetic_data(
    num_samples: int = 200,
    seq_len_range: tuple = (50, 150),
    num_speakers: int = 4,
) -> tuple:
    """Generate synthetic speech diarization data for testing.

    Args:
        num_samples: Number of samples to generate.
        seq_len_range: Range of sequence lengths.
        num_speakers: Number of speakers.

    Returns:
        Tuple of (audio_paths, diarization_labels, transcriptions).
    """
    import soundfile as sf
    import tempfile
    import os

    logger.info(f"Generating {num_samples} synthetic samples...")

    # Create temporary directory for synthetic audio
    temp_dir = Path(tempfile.mkdtemp(prefix="synthetic_audio_"))
    logger.info(f"Creating synthetic audio files in {temp_dir}")

    audio_paths = []
    diarization_labels = []
    transcriptions = []

    for i in range(num_samples):
        seq_len = np.random.randint(*seq_len_range)

        # Generate synthetic audio waveform (random noise)
        # Each frame is 160 samples (hop_length), sample rate is 16000
        audio_len = seq_len * 160  # Match the label sequence length
        waveform = np.random.randn(audio_len).astype(np.float32) * 0.1

        # Save audio file
        audio_path = temp_dir / f"synthetic_audio_{i}.wav"
        sf.write(str(audio_path), waveform, 16000)
        audio_paths.append(str(audio_path))

        # Generate speaker labels with some structure
        # Simulate speaker turns
        labels = []
        current_speaker = np.random.randint(0, num_speakers + 1)
        for j in range(seq_len):
            if j % 20 == 0 and np.random.random() < 0.3:
                # Change speaker occasionally
                current_speaker = np.random.randint(0, num_speakers + 1)
            labels.append(current_speaker)

        diarization_labels.append(np.array(labels))
        transcriptions.append("synthetic transcription")

    return audio_paths, diarization_labels, transcriptions


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train progressive multi-task speaker diarization with ASR"
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
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    # Set random seed
    set_seed(config.get("seed", 42))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create directories
    checkpoint_dir = Path(config["paths"].get("checkpoint_dir", "checkpoints"))
    results_dir = Path(config["paths"].get("results_dir", "results"))
    models_dir = Path(config["paths"].get("models_dir", "models"))

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data (in production, replace with actual data loading)
    logger.info("Generating synthetic training data...")
    all_audio_paths, all_diar_labels, all_transcriptions = generate_synthetic_data(
        num_samples=200,
        num_speakers=config["model"]["num_speakers"],
    )

    # Split data into train/val/test
    train_split = config["data"].get("train_split", 0.8)
    val_split = config["data"].get("val_split", 0.1)

    num_train = int(len(all_audio_paths) * train_split)
    num_val = int(len(all_audio_paths) * val_split)

    train_audio = all_audio_paths[:num_train]
    train_labels = all_diar_labels[:num_train]
    train_trans = all_transcriptions[:num_train]

    val_audio = all_audio_paths[num_train : num_train + num_val]
    val_labels = all_diar_labels[num_train : num_train + num_val]
    val_trans = all_transcriptions[num_train : num_train + num_val]

    logger.info(f"Train samples: {len(train_audio)}, Val samples: {len(val_audio)}")

    try:
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_audio_paths=train_audio,
            train_diar_labels=train_labels,
            val_audio_paths=val_audio,
            val_diar_labels=val_labels,
            train_transcriptions=train_trans,
            val_transcriptions=val_trans,
            batch_size=config["training"].get("batch_size", 16),
            curriculum_stage=config["curriculum"]["stages"][-1],
            num_workers=config["training"].get("num_workers", 4),
            augment_train=config["data"].get("augment_train", True),
        )

        # Create model
        logger.info("Initializing model...")
        model = ProgressiveMultiTaskModel(**config["model"])
        logger.info(
            f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters"
        )

        # Create trainer
        logger.info("Initializing trainer...")
        trainer = ProgressiveMultiTaskTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_speakers=config["model"]["num_speakers"],
            device=device,
            learning_rate=config["training"].get("learning_rate", 0.001),
            weight_decay=config["training"].get("weight_decay", 0.0001),
            max_epochs=config["training"].get("max_epochs", 100),
            patience=config["training"].get("patience", 15),
            gradient_clip=config["training"].get("gradient_clip", 5.0),
            use_amp=config["training"].get("use_amp", True),
            checkpoint_dir=str(checkpoint_dir),
            use_curriculum=config["training"].get("use_curriculum", True),
        )

        # Load checkpoint if provided
        if args.checkpoint:
            logger.info(f"Loading checkpoint from {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)

        # Initialize MLflow (wrapped in try/except)
        try:
            import mlflow

            mlflow.set_experiment("progressive-multitask-diarization")
            mlflow.start_run()
            mlflow.log_params(
                {
                    "learning_rate": config["training"]["learning_rate"],
                    "batch_size": config["training"]["batch_size"],
                    "max_epochs": config["training"]["max_epochs"],
                    "hidden_dim": config["model"]["hidden_dim"],
                    "num_speakers": config["model"]["num_speakers"],
                }
            )
            logger.info("MLflow tracking initialized")
            use_mlflow = True
        except Exception as e:
            logger.warning(f"MLflow not available: {e}. Continuing without MLflow tracking.")
            use_mlflow = False

        # Train model
        logger.info("Starting training...")
        history = trainer.train()

        # Log metrics to MLflow
        if use_mlflow:
            try:
                for epoch, (train_loss, val_loss) in enumerate(
                    zip(history["train_loss"], history["val_loss"])
                ):
                    mlflow.log_metrics(
                        {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
                    )
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"Error logging to MLflow: {e}")

        # Save final model to models directory
        logger.info("Saving final model...")
        final_model_path = models_dir / "best_model.pt"
        best_checkpoint = checkpoint_dir / "best_model.pt"
        if best_checkpoint.exists():
            import shutil

            shutil.copy(best_checkpoint, final_model_path)
            logger.info(f"Model saved to {final_model_path}")

        # Save training history
        history_path = results_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")

        # Plot training curves
        plot_path = results_dir / "training_curves.png"
        plot_training_curves(history, save_path=str(plot_path))

        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
