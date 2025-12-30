# SAM-RFI: Radio Frequency Interference Detection with SAM2

![](https://github.com/preshanth/SAM-RFI/blob/main/samrfi.png)

[![CI](https://github.com/preshanth/SAM-RFI/workflows/CI/badge.svg)](https://github.com/preshanth/SAM-RFI/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

-------------------------------------------------------------------------------------

**Authors:** Derod Deal (dealderod@gmail.com), Preshanth Jagannathan (pjaganna@nrao.edu)

SAM-RFI is a Python package that applies Meta's Segment Anything Model 2 (SAM2) for Radio Frequency Interference (RFI) detection and flagging in radio astronomy data. The system processes CASA measurement sets and generates precise segmentation masks for contaminated visibilities.

## Overview

SAM-RFI leverages the state-of-the-art SAM2 vision transformer for RFI segmentation in radio astronomy visibility data. The package provides a complete pipeline from data generation to trained models capable of detecting and flagging RFI with superior accuracy compared to traditional statistical methods.

**Key Features:**
- SAM2-based segmentation using Hiera transformer architecture
- Physically realistic synthetic data generation with exact ground truth
- Complete training pipeline with validation tracking
- Iterative flagging for progressive RFI cleaning
- GPU-accelerated training and inference
- Command-line interface for all operations
- Modular Python API for custom workflows

---

## Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- CUDA-capable GPU (recommended for training)
- CASA tools (optional, for measurement set operations)

### Basic Installation

```bash
# Clone repository
git clone https://github.com/preshanth/SAM-RFI.git
cd SAM-RFI

# Create conda environment
conda create -n samrfi python=3.12 -y
conda activate samrfi

# Install core dependencies (CPU-only, no GPU/CASA)
pip install pandas>=2.2.0 numpy>=1.26.0 --only-binary :all:
pip install -e .
```

### Installation Options

SAM-RFI supports modular installation based on your needs:

```bash
# GPU support (training and inference)
pip install -e .[gpu]

# CASA tools (measurement set operations)
pip install -e .[casa]

# GPU + CASA (complete functionality)
pip install -e .[gpu,casa]

# Development (all dependencies + testing tools)
pip install -e .[dev]

# Install pre-commit hooks (for development)
pre-commit install
```

**Installation extras:**
- **Core (default)**: Data preprocessing, synthetic data generation, evaluation metrics
- **`[gpu]`**: PyTorch, transformers, SAM2 models (required for training/inference)
- **`[casa]`**: CASA tools for measurement set I/O
- **`[viz]`**: Interactive visualization tools (HoloViews, Bokeh, Datashader)
- **`[dev]`**: All dependencies plus testing and linting tools
- **`[ci]`**: Minimal dependencies for continuous integration

### Verify Installation

```bash
# Check CLI availability
samrfi --help

# Test core imports (no GPU/CASA required)
python -c "from samrfi.data import Preprocessor; from samrfi.data_generation import SyntheticDataGenerator; print('Core installation successful')"

# Test GPU functionality (requires [gpu])
python -c "from samrfi.training import SAM2Trainer; from samrfi.inference import RFIPredictor; print('GPU installation successful')"

# Test CASA functionality (requires [casa])
python -c "from samrfi.data.ms_loader import MSLoader; print('CASA installation successful')"
```

---

## Quick Start

### 1. Generate Synthetic Training Data

Generate physically realistic training data with exact ground truth masks:

```bash
samrfi generate-data \
  --source synthetic \
  --config configs/synthetic_train_4k.yaml \
  --output ./datasets/train_4k
```

**Configuration** (`configs/synthetic_train_4k.yaml`):
```yaml
synthetic:
  num_samples: 4000
  num_channels: 1024
  num_times: 1024
  num_baselines: 2
  num_pols: 4

  # Physical scales (milli-Jansky and Jansky)
  noise_mjy: 1.0                 # 1 mJy Gaussian noise
  rfi_power_min: 1000.0          # 1000 Jy RFI minimum
  rfi_power_max: 10000.0         # 10000 Jy RFI maximum

  # RFI types per sample
  rfi_type_counts:
    narrowband_persistent: 2
    broadband_persistent: 1
    frequency_sweep: 1
    narrowband_bursty: 2
    broadband_bursty: 1

  # Bandpass effects
  enable_bandpass_rolloff: true
  bandpass_polynomial_order: 8
  polarization_correlation: 0.8

processing:
  patch_size: 1024
  stretch: null                  # No stretch for synthetic (preserves physical scales)
  enable_augmentation: true      # 4-way rotation augmentation
  normalize_before_stretch: false
  normalize_after_stretch: false
```

This generates batched datasets saved to `./datasets/train_4k/exact_masks/` with perfect ground truth masks.

### 2. Train SAM2 Model

SAM2 models automatically download from HuggingFace on first use. Models are cached at `~/.cache/huggingface/hub/`.

```bash
samrfi train \
  --config configs/gpu_v100_training.yaml \
  --dataset ./datasets/train_4k/exact_masks \
  --validation-dataset ./datasets/val_1k/exact_masks
```

**Training configuration** (`configs/gpu_v100_training.yaml`):
```yaml
model:
  sam_checkpoint: large          # Options: tiny, small, base_plus, large
  device: cuda

training:
  num_epochs: 10
  batch_size: 12
  learning_rate: 1.0e-5
  weight_decay: 0.0
  save_best_only: true

output:
  output_dir: ./samrfi_data
  save_plots: true
```

**Available SAM2 models:**
- `tiny` (40 MB) - Fastest, lower accuracy
- `small` (180 MB) - Balanced performance
- `base_plus` (330 MB) - Good accuracy
- `large` (850 MB) - Best accuracy, recommended for production

**GPU memory requirements:**
- 11 GB VRAM: `tiny`, batch_size=2
- 32 GB VRAM: `base_plus`, batch_size=12
- 40+ GB VRAM: `large`, batch_size=8-12

### 3. Apply Trained Model

**Single-pass prediction:**
```bash
samrfi predict \
  --model ./samrfi_data/sam2_rfi_best.pth \
  --input observation.ms \
  --patch-size 1024
```

**Iterative prediction** (recommended for deep cleaning):
```bash
samrfi predict \
  --model ./samrfi_data/sam2_rfi_best.pth \
  --input observation.ms \
  --iterations 3 \
  --patch-size 1024
```

Iterative flagging progressively finds fainter RFI by masking already-flagged regions in each iteration. Typically converges in 2-3 passes.

**Prediction options:**
- `--iterations N` - Number of flagging passes (default: 1)
- `--num-antennas N` - Limit number of antennas loaded
- `--patch-size SIZE` - Must match training patch size
- `--stretch {SQRT,LOG10,null}` - Must match training configuration
- `--threshold FLOAT` - Probability threshold (default: adaptive/mean)
- `--no-save` - Preview only, do not write flags to MS

---

## CLI Reference

### Data Generation

```bash
# Generate synthetic training data
samrfi generate-data \
  --source synthetic \
  --config configs/synthetic_train_4k.yaml \
  --output ./datasets/train_4k

# Generate data from measurement set
samrfi generate-data \
  --source ms \
  --config configs/ms_data.yaml \
  --output ./datasets/vla_pband
```

### Training

```bash
# Train with validation dataset
samrfi train \
  --config configs/gpu_v100_training.yaml \
  --dataset ./datasets/train_4k/exact_masks \
  --validation-dataset ./datasets/val_1k/exact_masks

# Resume training from checkpoint
samrfi train \
  --config configs/gpu_v100_training.yaml \
  --dataset ./datasets/train_4k/exact_masks \
  --resume ./samrfi_data/sam2_rfi_best.pth
```

### Inference

```bash
# Single-pass prediction
samrfi predict \
  --model ./samrfi_data/sam2_rfi_best.pth \
  --input observation.ms

# Iterative prediction (3 passes)
samrfi predict \
  --model ./samrfi_data/sam2_rfi_best.pth \
  --input observation.ms \
  --iterations 3
```

### Configuration

```bash
# Create default configuration
samrfi create-config \
  --type {training|data|validation} \
  --output config.yaml

# Validate configuration
samrfi validate-config --config config.yaml
```

---

## Python API

### Core Data Operations (No GPU/CASA Required)

```python
from samrfi.data import Preprocessor, TorchDataset
from samrfi.data_generation import SyntheticDataGenerator
from samrfi.evaluation import compute_iou, compute_ffi

# Generate synthetic data
generator = SyntheticDataGenerator(config_path='configs/synthetic_train_4k.yaml')
dataset = generator.generate(num_samples=1000, output_dir='./datasets/synthetic')

# Preprocess data
import numpy as np
data = np.random.randn(2, 4, 1024, 1024) + 1j * np.random.randn(2, 4, 1024, 1024)
preprocessor = Preprocessor(data)
dataset = preprocessor.create_dataset(patch_size=1024, stretch=None)

# Evaluate predictions
iou = compute_iou(predicted_mask, ground_truth_mask)
ffi = compute_ffi(data, flags=predicted_mask)
```

### Measurement Set Operations (Requires [casa])

```python
from samrfi.data.ms_loader import MSLoader

# Load measurement set
loader = MSLoader('observation.ms')
loader.load(num_antennas=5, mode='DATA')

# Access data
data = loader.data              # Complex visibilities: (baselines, pols, channels, times)
magnitude = loader.magnitude    # Magnitude
flags = loader.load_flags()     # Existing flags

# Save new flags
loader.save_flags(predicted_flags)
```

### Training (Requires [gpu])

```python
from samrfi.training import SAM2Trainer
from samrfi.data import TorchDataset

# Load batched dataset
dataset = TorchDataset.from_directory('./datasets/train_4k/exact_masks')

# Create trainer
trainer = SAM2Trainer(dataset, device='cuda')

# Train model
trainer.train(
    num_epochs=10,
    batch_size=12,
    sam_checkpoint='large',
    learning_rate=1e-5,
    output_dir='./samrfi_data',
    save_best_only=True
)
```

### Inference (Requires [gpu])

```python
from samrfi.inference import RFIPredictor

# Load predictor with trained model
predictor = RFIPredictor(
    model_path='./samrfi_data/sam2_rfi_best.pth',
    device='cuda'
)

# Single-pass prediction
flags = predictor.predict_ms(
    ms_path='observation.ms',
    patch_size=1024,
    save_flags=True
)

# Iterative prediction (3 passes)
flags = predictor.predict_iterative(
    ms_path='observation.ms',
    num_iterations=3,
    patch_size=1024,
    save_flags=True
)

print(f"Flagged {flags.sum() / flags.size * 100:.2f}% of data")
```

### Single Array Validation

```python
from samrfi.inference import RFIPredictor
import numpy as np

# Load model
predictor = RFIPredictor(model_path='model.pth', device='cuda')

# Predict on arbitrary-sized array
data = np.load('baseline_data.npy')  # Any shape, e.g., (2048, 511)
flags, probabilities = predictor.predict_array(
    data,
    threshold=None,  # Adaptive (uses mean of probabilities)
    return_probabilities=True
)

# Save probabilities for custom thresholding
np.save('probabilities.npy', probabilities)

# Apply custom threshold
custom_flags = probabilities > 0.1  # More aggressive flagging
```

---

## Architecture

### Data Pipeline

```
[Measurement Set or Synthetic Generator]
        ↓
    MSLoader.load()
        ├─ Complex visibilities (baselines, pols, channels, times)
        └─ Combine spectral windows
        ↓
    Preprocessor.create_dataset()
        ├─ 4-way rotation augmentation (optional)
        ├─ Patchify into patch_size × patch_size
        ├─ Extract 3-channel features:
        │   • Channel 1: Spatial gradient (edge detection)
        │   • Channel 2: Log amplitude (intensity, [-3, 4])
        │   • Channel 3: Phase ([-π, π] → [0, 1])
        ├─ Apply optional stretch (SQRT/LOG10 for real, None for synthetic)
        └─ ImageNet normalization
        ↓
    BatchedDataset (streaming)
        ├─ Batch files: batch_*.pt + metadata.json
        ├─ On-demand loading in DataLoader workers
        └─ OS filesystem cache for efficiency
```

### Training Pipeline

```
    BatchedDataset
        ↓
    SAMDataset wrapper
        ├─ Extract bounding boxes from ground truth
        ├─ Add random perturbation (±20 pixels)
        └─ Format: {pixel_values, input_boxes, ground_truth_mask}
        ↓
    SAM2Trainer.train()
        ├─ Load SAM2Model from HuggingFace
        ├─ Freeze vision + prompt encoders
        ├─ Train mask decoder only (~10% of parameters)
        ├─ Loss: DiceCELoss (Dice + Cross-Entropy)
        └─ Save: sam2_rfi_best.pth
```

### Inference Pipeline

```
    [Trained Model] + [Measurement Set]
        ↓
    RFIPredictor.predict_ms() or predict_iterative()
        ↓
    MSLoader.load() → Preprocessor → Patches
        ↓
    SAM2Model.forward()
        ├─ Vision encoder: Extract features
        ├─ Prompt encoder: Encode bounding boxes
        └─ Mask decoder: Predict segmentation
        ↓
    Reconstruction
        ├─ Sigmoid(logits) > threshold
        ├─ Reverse rotations
        ├─ Combine patches → full waterfall
        └─ Boolean flags: (baselines, pols, channels, times)
        ↓
    MSLoader.save_flags() → Write to MS FLAG column
```

---

## Iterative Flagging

Iterative flagging progressively discovers deeper RFI by masking known contamination in each pass, revealing fainter interference hidden beneath brighter sources.

### Algorithm

```
Iteration 1: Raw data → Model → Flags_1 (finds bright RFI)
Iteration 2: Data with Flags_1 masked → Model → Flags_2 (finds hidden RFI)
Iteration 3: Data with Flags_1|2 masked → Model → Flags_3 (final cleanup)

Final: Flags_cumulative = Flags_1 | Flags_2 | Flags_3
```

### Guidelines

- **Single pass (N=1)**: Fast, suitable for mild contamination (5-10% flagging)
- **2-3 iterations**: Recommended for deep cleaning (15-30% flagging)
- **>3 iterations**: Diminishing returns, increased risk of over-flagging

### Example

```bash
# Compare single vs iterative
samrfi predict --model model.pth --input obs.ms
# Output: Flagged 12.5% of data

samrfi predict --model model.pth --input obs.ms --iterations 3
# Output: Iteration 1: 12.5%, Iteration 2: 4.2%, Iteration 3: 1.1%
# Total: Flagged 17.8% of data
```

---

## Synthetic Data Generation

### RFI Types

The synthetic data generator produces physically realistic RFI signatures:

1. **Narrowband Persistent** - Continuous narrowband signals (GPS, satellites)
2. **Broadband Persistent** - Continuous wideband interference (power lines, harmonics)
3. **Narrowband Bursty** - Intermittent narrowband pulses (radar, transmitters)
4. **Broadband Bursty** - Transient wideband events (lightning, arcing)
5. **Frequency Sweeps** - Linear and quadratic chirps (scanning radar)

### Physical Parameters

- **Noise**: 1 mJy (milli-Jansky) Gaussian, matches typical system noise
- **RFI Power**: 1000-10000 Jy (Jansky), 10^6-10^7 dynamic range
- **Bandpass**: 8th-order polynomial edge rolloff
- **Polarization**: Correlated RFI across XX/YY feeds (0.8 correlation)

### Ground Truth Advantage

Synthetic data provides exact ground truth masks, enabling supervised training with 100% accurate labels. This is not possible with real observations, where RFI locations are only estimates from statistical flaggers.

**Preservation of Physical Scales:**
```yaml
processing:
  normalize_before_stretch: false  # Critical for synthetic data
  normalize_after_stretch: false
  stretch: null  # Preserves 10^6-10^7 dynamic range
```

---

## Model Management

### Automatic Downloads

SAM2 models are automatically downloaded from HuggingFace Hub on first use and cached locally:

```python
from samrfi.training import SAM2Trainer

# Model downloads automatically if not cached
trainer = SAM2Trainer(dataset, device='cuda')
trainer.train(num_epochs=10, sam_checkpoint='large')
```

**Cache location:** `~/.cache/huggingface/hub/`

### Custom Cache Directory

```bash
export HF_HOME=/path/to/custom/cache
samrfi train --config config.yaml --dataset ./datasets/train
```

### Pre-download Models

```python
from samrfi.utils.model_cache import ModelCache

cache = ModelCache()
cache.download_model('large', show_progress=True)
```

Or via command line:
```bash
python -c "from samrfi.utils.model_cache import ModelCache; ModelCache().download_model('large')"
```

---

## Training

### Resume Training

Training can be resumed from any checkpoint to continue where you left off:

```bash
# Initial training
samrfi train --config config.yaml --dataset ./datasets/train --epochs 10

# Resume and extend to 20 epochs
samrfi train --config config.yaml --dataset ./datasets/train --epochs 20 \
  --resume ./samrfi_data/sam2_rfi_best.pth
```

**Restored state:**
- Model weights
- Optimizer state (momentum, learning rates)
- Training/validation loss history
- Epoch counter

**Checkpoints:**
- `sam2_rfi_best.pth` - Best validation loss (updated during training)
- `model_sam2-large_YYYYMMDD_HHMMSS.pth` - Final checkpoint with full state

### Hyperparameter Tuning

**Fast iteration (debugging):**
```yaml
model:
  sam_checkpoint: tiny
training:
  num_epochs: 3
  batch_size: 8
  learning_rate: 1.0e-4
```

**Production quality:**
```yaml
model:
  sam_checkpoint: large
training:
  num_epochs: 20
  batch_size: 4
  learning_rate: 1.0e-5
  weight_decay: 0.0
```

### Loss Convergence

Expected behavior: Loss decreases from approximately 1.0 to below 0.3 within 10 epochs.

**Troubleshooting stalled training (loss >0.8 after 5 epochs):**
- Adjust learning rate (try 5e-6 or 2e-5)
- Verify data quality (visualize sample patches)
- Modify batch size (try 2 or 8)
- Check for NaN values in input data

---

## Evaluation

### Metrics Module

```python
from samrfi.evaluation import (
    compute_iou,           # Intersection over Union
    compute_precision,     # True Positive Rate
    compute_recall,        # Sensitivity
    compute_f1,            # Harmonic mean of precision/recall
    compute_dice,          # Dice coefficient
    evaluate_segmentation  # All metrics
)

# Evaluate predictions
metrics = evaluate_segmentation(predicted_mask, ground_truth_mask)
# Returns: {'iou': 0.85, 'precision': 0.90, 'recall': 0.82, 'f1': 0.86}
```

### Statistical Validation

```python
from samrfi.evaluation import (
    compute_statistics,              # Before/after statistics
    compute_ffi,                     # Flagging Fidelity Index
    print_statistics_comparison      # Formatted output
)

# Compute Flagging Fidelity Index
ffi_metrics = compute_ffi(data, flags=predicted_mask)
# Returns: {'ffi': 0.65, 'mad_reduction': 0.45, 'std_reduction': 0.52}

# Print comparison
print_statistics_comparison(data, predicted_mask)
```

**Flagging Fidelity Index (FFI):** Measures flagging quality by balancing noise reduction against over-flagging penalty. Higher values indicate better flagging performance.

---

## Development

### Testing

```bash
# Run all tests
pytest tests/ -v

# Unit tests only
pytest tests/unit -v

# Integration tests
pytest tests/integration -v

# With coverage
pytest tests/ --cov=samrfi --cov-report=html

# Skip slow tests
pytest -m "not slow"
```

### Code Quality

Pre-commit hooks are configured to run automatically on `git commit`:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

**Checks performed:**
- Black (code formatting, line length 100)
- Ruff (linting and auto-fixes)
- isort (import sorting)
- Trailing whitespace, EOF, YAML/JSON/TOML validation
- Large file detection (>5MB)

**Manual formatting:**
```bash
# Format code
black src/ tests/ --line-length 100

# Lint code
ruff check src/ tests/ --fix

# Sort imports
isort src/ tests/ --profile black --line-length 100
```

### Type Checking

```bash
# Type check (optional)
mypy src/ --ignore-missing-imports
```

---

## Project Structure

```
SAM-RFI/
├── src/samrfi/
│   ├── cli.py                      # Command-line interface
│   ├── config/                     # Configuration management
│   │   ├── config_loader.py        # YAML configuration loading
│   │   └── validators.py           # Configuration validation
│   ├── data/                       # Data loading and preprocessing
│   │   ├── ms_loader.py            # CASA measurement set I/O
│   │   ├── preprocessor.py         # Waterfall to patches pipeline
│   │   ├── sam_dataset.py          # PyTorch dataset wrapper
│   │   ├── torch_dataset.py        # Batched streaming datasets
│   │   └── gpu_transforms.py       # Kornia-based GPU transforms
│   ├── data_generation/            # Dataset generators
│   │   ├── synthetic_generator.py  # Physics-based RFI simulation
│   │   └── ms_generator.py         # MS to dataset converter
│   ├── training/
│   │   └── sam2_trainer.py         # SAM2 training loop
│   ├── inference/
│   │   └── predictor.py            # RFI prediction (single/iterative)
│   ├── evaluation/                 # Metrics and validation
│   │   ├── metrics.py              # Segmentation metrics
│   │   └── statistics.py           # Flagging quality statistics
│   └── utils/                      # Utilities
│       ├── logger.py               # Logging configuration
│       ├── model_cache.py          # HuggingFace model downloads
│       └── errors.py               # Custom exceptions
│
├── tests/                          # Test suite
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   └── conftest.py                 # Shared fixtures
│
├── configs/                        # Example configurations
│   ├── gpu_*.yaml                  # GPU-specific training configs
│   ├── synthetic_*.yaml            # Synthetic data configs
│   └── validation.yaml             # Validation config
│
├── .github/workflows/              # CI/CD
│   └── ci.yml                      # GitHub Actions workflow
│
├── pyproject.toml                  # Package definition
├── .pre-commit-config.yaml         # Pre-commit hooks
└── README.md                       # This file
```

---

## Citation

A paper describing SAM-RFI is in preparation. In the meantime, if you use this software in your research, please cite the repository:

```bibtex
@software{samrfi2025,
  title = {SAM-RFI: Radio Frequency Interference Detection with SAM2},
  author = {Deal, Derod and Jagannathan, Preshanth},
  year = {2025},
  url = {https://github.com/preshanth/SAM-RFI}
}
```

Please check back for the updated citation once the paper is published.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **Meta AI** - SAM2 architecture and pre-trained models
- **HuggingFace** - Transformers library and model hosting
- **NRAO** - Radio astronomy expertise and computational resources
- **NAC** - National Astronomy Consortium support and funding

---

## Support

- **Issues**: https://github.com/preshanth/SAM-RFI/issues
- **Documentation**: https://sam-rfi.readthedocs.io (coming soon)
- **Contact**: pjaganna@nrao.edu
