#!/usr/bin/env python
"""Prediction script for progressive multi-task speaker diarization with ASR."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from progressive_multi_task_speaker_diarization_with_asr.models.model import (
    ProgressiveMultiTaskModel,
)
from progressive_multi_task_speaker_diarization_with_asr.data.preprocessing import (
    AudioPreprocessor,
)
from progressive_multi_task_speaker_diarization_with_asr.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def predict_on_audio(
    audio_path: str,
    model: ProgressiveMultiTaskModel,
    preprocessor: AudioPreprocessor,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, any]:
    """Run prediction on a single audio file.

    Args:
        audio_path: Path to audio file.
        model: Trained model.
        preprocessor: Audio preprocessor.
        device: Device to run inference on.
        threshold: Confidence threshold for speaker activity.

    Returns:
        Dictionary containing predictions and metadata.
    """
    try:
        # Load and preprocess audio
        logger.info(f"Processing audio: {audio_path}")
        waveform, sr = preprocessor.load_audio(audio_path)
        features = preprocessor.extract_features(waveform)

        # Add batch dimension
        features = features.unsqueeze(0).to(device)

        # Run inference
        model.eval()
        with torch.no_grad():
            outputs = model(features)

            # Diarization prediction
            diar_logits = outputs["diarization_logits"]
            diar_probs = torch.softmax(diar_logits, dim=-1)
            speaker_ids = torch.argmax(diar_probs, dim=-1)

            # Apply threshold
            max_probs = torch.max(diar_probs[:, :, 1:], dim=-1)[0]
            speaker_ids[max_probs < threshold] = 0

            # ASR prediction (greedy decoding)
            asr_logits = outputs["asr_logits"]
            asr_preds = torch.argmax(asr_logits, dim=-1)

        # Convert to numpy
        speaker_ids = speaker_ids.cpu().numpy()[0]
        asr_preds = asr_preds.cpu().numpy()[0]
        diar_probs = diar_probs.cpu().numpy()[0]

        # Get speaker segments
        segments = extract_speaker_segments(speaker_ids, frame_duration=0.01)

        results = {
            "audio_path": audio_path,
            "speaker_labels": speaker_ids.tolist(),
            "speaker_segments": segments,
            "confidence_scores": diar_probs.tolist(),
            "asr_tokens": asr_preds.tolist(),
        }

        return results

    except FileNotFoundError:
        logger.error(f"Audio file not found: {audio_path}")
        raise
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise


def extract_speaker_segments(
    speaker_ids: np.ndarray, frame_duration: float = 0.01
) -> List[Dict[str, any]]:
    """Extract speaker turn segments from frame-level predictions.

    Args:
        speaker_ids: Frame-level speaker IDs.
        frame_duration: Duration of each frame in seconds.

    Returns:
        List of speaker segment dictionaries.
    """
    segments = []
    current_speaker = speaker_ids[0]
    start_frame = 0

    for i, speaker_id in enumerate(speaker_ids):
        if speaker_id != current_speaker:
            # End of current segment
            if current_speaker > 0:  # Ignore silence
                segments.append(
                    {
                        "speaker": int(current_speaker),
                        "start_time": start_frame * frame_duration,
                        "end_time": i * frame_duration,
                        "duration": (i - start_frame) * frame_duration,
                    }
                )
            # Start new segment
            current_speaker = speaker_id
            start_frame = i

    # Add final segment
    if current_speaker > 0:
        segments.append(
            {
                "speaker": int(current_speaker),
                "start_time": start_frame * frame_duration,
                "end_time": len(speaker_ids) * frame_duration,
                "duration": (len(speaker_ids) - start_frame) * frame_duration,
            }
        )

    return segments


def main() -> None:
    """Main prediction function."""
    parser = argparse.ArgumentParser(
        description="Run prediction on audio files for speaker diarization and ASR"
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to audio file for prediction",
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
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for speaker activity detection",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save predictions as JSON",
    )
    args = parser.parse_args()

    # Check if audio file exists
    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        logger.info(
            "Note: For demonstration, use synthetic data by running train.py first, "
            "or provide a valid audio file path."
        )
        # Generate synthetic prediction for demonstration
        logger.info("Generating synthetic prediction output for demonstration...")
        results = {
            "audio_path": str(audio_path),
            "speaker_labels": [0, 1, 1, 1, 2, 2, 0, 3, 3, 3],
            "speaker_segments": [
                {
                    "speaker": 1,
                    "start_time": 0.01,
                    "end_time": 0.03,
                    "duration": 0.02,
                },
                {
                    "speaker": 2,
                    "start_time": 0.04,
                    "end_time": 0.05,
                    "duration": 0.01,
                },
                {
                    "speaker": 3,
                    "start_time": 0.07,
                    "end_time": 0.09,
                    "duration": 0.02,
                },
            ],
            "note": "Synthetic output - audio file not found",
        }
    else:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

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
            logger.warning(
                f"Checkpoint not found at {checkpoint_path}, using untrained model"
            )

        model = model.to(device)

        # Create preprocessor
        preprocessor = AudioPreprocessor(**config["data"])

        # Run prediction
        logger.info("Running prediction...")
        results = predict_on_audio(
            str(audio_path), model, preprocessor, device, threshold=args.threshold
        )

    # Print results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"\nAudio: {results['audio_path']}")
    print(f"\nSpeaker Segments:")
    print("-" * 60)

    if "speaker_segments" in results:
        for segment in results["speaker_segments"]:
            print(
                f"  Speaker {segment['speaker']}: "
                f"{segment['start_time']:.2f}s - {segment['end_time']:.2f}s "
                f"(duration: {segment['duration']:.2f}s)"
            )

    print("\n" + "=" * 60)

    # Save results if output path provided
    if args.output:
        import json

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
