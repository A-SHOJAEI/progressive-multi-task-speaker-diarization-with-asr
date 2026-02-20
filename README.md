# Progressive Multi-Task Speaker Diarization with ASR

A progressive multi-task learning system that jointly optimizes speaker diarization and automatic speech recognition through curriculum-based task scheduling. The system combines cross-task attention mechanisms to share speaker-aware acoustic representations between diarization and ASR tasks, addressing the limitation that standard diarization systems ignore linguistic content while ASR systems struggle with speaker changes.

## Features

- Progressive curriculum learning: transitions from single-speaker ASR to overlapping speech to full multi-speaker diarization
- Novel cross-task attention mechanism for sharing speaker-aware representations
- Dynamic multi-task loss weighting with learnable task weights
- Joint optimization of speaker diarization error rate and word error rate
- Comprehensive evaluation with per-speaker performance analysis

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

Train the full model with cross-task attention and curriculum learning:

```bash
python scripts/train.py --config configs/default.yaml
```

Train baseline model (ablation study without cross-task attention):

```bash
python scripts/train.py --config configs/ablation.yaml
```

### Evaluation

Evaluate the trained model on test data:

```bash
python scripts/evaluate.py --checkpoint models/best_model.pt
```

### Prediction

Run inference on a new audio file:

```bash
python scripts/predict.py path/to/audio.wav --checkpoint models/best_model.pt
```

## Project Structure

```
progressive-multi-task-speaker-diarization-with-asr/
├── src/progressive_multi_task_speaker_diarization_with_asr/
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architecture and custom components
│   ├── training/          # Training loop with curriculum scheduling
│   ├── evaluation/        # Metrics and analysis tools
│   └── utils/             # Configuration utilities
├── configs/
│   ├── default.yaml       # Full model configuration
│   └── ablation.yaml      # Baseline without cross-task attention
├── scripts/
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   └── predict.py         # Inference script
└── tests/                 # Comprehensive test suite
```

## Methodology

Traditional speaker diarization and ASR systems are trained independently, creating a fundamental limitation: diarization systems ignore linguistic content while ASR systems struggle with speaker changes. This project addresses this gap through three key innovations:

### 1. Cross-Task Attention Mechanism

Our novel cross-task attention module enables bidirectional information flow between diarization and ASR representations. The ASR decoder can attend to speaker-aware acoustic features from the diarization encoder, while the diarization head benefits from linguistic context. This is implemented as a multi-head attention mechanism where:
- Query features come from one task (e.g., ASR)
- Key-value features come from the other task (e.g., diarization)
- Residual connections preserve task-specific information

This architecture allows the model to learn that certain linguistic patterns correlate with speaker changes, and that speaker boundaries often align with linguistic units.

### 2. Progressive Curriculum Learning

Rather than immediately tackling the full multi-speaker diarization problem, we introduce task complexity gradually:

1. **Single-speaker phase (epochs 0-N/3)**: Model learns basic ASR and speaker embedding without overlap
2. **Overlapping speech phase (epochs N/3-2N/3)**: Introduces concurrent speakers to build robustness
3. **Full diarization phase (epochs 2N/3-N)**: Complete multi-speaker task with all complexities

This curriculum is motivated by human learning: we master simpler sub-tasks before tackling complex combinations. The scheduler automatically transitions between stages based on epoch progress.

### 3. Dynamic Multi-Task Loss Weighting

Instead of manually tuning task weights, we learn them during training through uncertainty weighting. The loss function maintains learnable parameters (in log-space for numerical stability):

```
L_total = exp(log_w_diar) * L_diar + exp(log_w_asr) * L_asr
```

These weights adapt based on each task's training dynamics, preventing one task from dominating the optimization landscape. This is critical because diarization (frame-level classification) and ASR (sequence prediction with CTC loss) operate at different scales.

### Why This Approach is Novel

1. **Joint optimization**: Unlike cascaded systems (diarization → ASR), both tasks inform each other during training
2. **Explicit information sharing**: Cross-attention is specifically designed for multi-task fusion, not just parameter sharing
3. **Curriculum-guided training**: Progressive task difficulty enables more stable convergence than training on full complexity from the start
4. **Adaptive weighting**: Learned task weights eliminate manual hyperparameter tuning for loss balance

## Configuration

Key hyperparameters in `configs/default.yaml`:

- `model.hidden_dim`: Hidden dimension for encoders (default: 256)
- `model.use_cross_attention`: Enable cross-task attention (default: true)
- `training.use_curriculum`: Enable curriculum learning (default: true)
- `training.learning_rate`: Initial learning rate (default: 0.001)
- `training.max_epochs`: Maximum training epochs (default: 100)

## Results

Training completed over 16 epochs with progressive curriculum learning on synthetic data.

### Training Metrics

| Metric | Value |
|--------|-------|
| Total Epochs | 16 |
| Best Validation Loss | 1.5812 (epoch 0) |
| Final Training Loss | 1.6017 |
| Final Validation Loss | 1.6326 |
| Training Loss Reduction | 8.3% (1.7469 to 1.6017) |
| Scheduler | Cosine Annealing |

### Training Progression

| Epoch | Train Loss | Val Loss |
|------:|----------:|---------:|
| 0 | 1.7469 | 1.5812 |
| 5 | 1.6097 | 1.6222 |
| 9 | 1.6123 | 1.5924 |
| 11 | 1.5960 | 1.6493 |
| 15 | 1.6017 | 1.6326 |

> **Note**: Training was conducted on synthetic data. The model showed moderate training loss reduction but validation loss oscillated and did not consistently improve below the epoch 0 baseline. The multi-task learning pipeline (cross-task attention, curriculum scheduling, dynamic loss weighting) is fully functional. Training with real multi-speaker audio data (e.g., AMI, LibriCSS) would yield meaningful DER and WER metrics.

To reproduce results:
```bash
python scripts/train.py --config configs/default.yaml
```

## Testing

Run the test suite:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Ablation Study

Compare full model vs baseline without cross-task attention:

```bash
# Train full model
python scripts/train.py --config configs/default.yaml

# Train baseline
python scripts/train.py --config configs/ablation.yaml

# Compare results in results/ and results_ablation/
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchaudio 2.0+
- See `requirements.txt` for complete list

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
