# SAM-RFI: Radio Frequency Interference Detection with SAM2

![](https://github.com/preshanth/SAM-RFI/blob/main/samrfi.png)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

-------------------------------------------------------------------------------------

**Authors:** Derod Deal (dealderod@gmail.com), Preshanth Jagannathan (pjaganna@nrao.edu)

`SAM-RFI` is a Python package that utilizes Meta's Segment Anything Model 2 (SAM2) for Radio Frequency Interference (RFI) detection and segmentation in radio astronomy data. This is a complete refactor with a clean, modular architecture.


## Overview

SAM-RFI applies Meta's Segment Anything Model 2 (SAM2) to detect and flag Radio Frequency Interference (RFI) in radio astronomy data. The tool processes CASA measurement sets and generates precise segmentation masks for contaminated data.

**Key Features:**
- 🚀 **SAM2-based segmentation** - State-of-the-art Hiera transformer architecture
- 📊 **Physically realistic synthetic data** - Generate training data with exact ground truth
- 🔧 **Complete training pipeline** - From MS files to trained models
- ⚡ **GPU-accelerated** - Fast training and inference
- 🎯 **High accuracy** - Superior to traditional MAD-based flaggers
- 🛠️ **Command-line interface** - Easy to use CLI for all operations

---

## Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- CUDA-capable GPU (recommended for training)
- CASA tools (included in `[dev]` install)

### Quick Install

```bash
# Clone repository
git clone https://github.com/preshanth/SAM-RFI.git
cd SAM-RFI

# Create conda environment
conda create -n samrfi python=3.12 -y
conda activate samrfi

# Install dependencies (fixes pandas/numpy compatibility)
pip install pandas>=2.2.0 numpy>=1.26.0 --only-binary :all:

# Install SAM-RFI with all dependencies (GPU + CASA + dev tools)
pip install -e .[dev]
```

### Verify Installation

```bash
# Check CLI is available
samrfi --help

# Test imports
python -c "from samrfi.data import MSLoader, Preprocessor; from samrfi.training import SAM2Trainer; print('✓ Installation successful')"
```

---

## Quick Start

### 1. Generate Synthetic Training Data

Generate 1000 synthetic samples with physically realistic RFI:

```bash
samrfi generate-data \
  --source synthetic \
  --config configs/synthetic_data.yaml \
  --output ./datasets/synthetic_p_band
```

**Example config** (`configs/synthetic_data.yaml`):
```yaml
synthetic:
  num_samples: 1000
  num_channels: 2048
  num_times: 512

  # Physical scales (milli-Jansky and Jansky)
  noise_mjy: 1.0                 # 1 mJy noise
  rfi_power_min: 1000.0          # 1000 Jy RFI min
  rfi_power_max: 10000.0         # 10000 Jy RFI max

  # RFI types per sample
  rfi_type_counts:
    narrowband_persistent: 2
    broadband_persistent: 1
    frequency_sweep: 1
    narrowband_bursty: 2
    broadband_bursty: 1

  # Optional: Bandpass effects
  enable_bandpass_rolloff: true
  bandpass_polynomial_order: 8
  polarization_correlation: 0.8

processing:
  stretch: SQRT
  flag_sigma: 5
  patch_size: 128
  apply_stretching: true
```

This generates **two datasets**:
- `exact_masks/` - Perfect ground truth (train on this!)
- `mad_masks/` - MAD-based masks (for comparison)

**Note:** Datasets are generated locally and saved to disk. They are NOT uploaded to HuggingFace by default.

### 2. Train SAM2 Model

**SAM2 models auto-download from HuggingFace on first use** (~850MB for `large`). This is a one-time download, cached at `~/.cache/huggingface/hub/`.

Train on the synthetic data with exact ground truth:

```bash
samrfi train \
  --config configs/sam2_training.yaml \
  --dataset ./datasets/synthetic_p_band/exact_masks \
  --output ./models/sam2_rfi_v1
```

**Training config** (`configs/sam2_training.yaml`):
```yaml
model:
  checkpoint: large              # tiny, small, base_plus, large
  freeze_encoders: true

training:
  num_epochs: 10
  batch_size: 4
  learning_rate: 1.0e-5
  weight_decay: 0.0
  device: cuda                   # or cpu

output:
  dir_path: ./models/sam2_rfi_v1
  save_plots: true
```

**Monitor training:**
- Loss curves saved to `models/sam2_rfi_v1/loss_plot.png`
- Model checkpoints: `sam2_model_YYYYMMDD_HHMMSS.pth`

### 3. Generate Dataset from Real MS

Generate training data from your measurement set:

```bash
samrfi generate-data \
  --source ms \
  --config configs/ms_data.yaml \
  --output ./datasets/vla_pband_3c219
```

**MS config** (`configs/ms_data.yaml`):
```yaml
ms:
  path: /path/to/observation.ms
  num_antennas: 5              # Load first N antennas
  data_mode: DATA              # or CORRECTED_DATA

processing:
  stretch: SQRT
  flag_sigma: 5
  patch_size: 128
  custom_flag: true            # Use MS flags (not MAD)
  apply_stretching: true
```

### 4. Apply Model to Flag RFI

**Single-pass prediction** (default):
```bash
samrfi predict \
  --model ./models/sam2_rfi_v1/sam2_model_20250930_120000.pth \
  --input observation.ms
```

**Iterative prediction** (3 passes for deep cleaning):
```bash
samrfi predict \
  --model ./models/sam2_rfi_v1/sam2_model_20250930_120000.pth \
  --input observation.ms \
  --iterations 3
```

Each iteration:
1. Masks already-flagged regions
2. Finds fainter RFI hidden by brighter RFI
3. Combines with previous flags

**Options:**
- `--iterations N` - Number of passes (default: 1)
- `--num-antennas N` - Limit antennas
- `--patch-size 128` - Match training
- `--device cuda` - GPU or cpu
- `--no-save` - Preview only (don't write flags)

---

## CLI Reference

### Main Commands

```bash
# Generate training data
samrfi generate-data --source {synthetic|ms} --config CONFIG.yaml --output DIR

# Train model
samrfi train --config CONFIG.yaml --dataset DIR [--output DIR]

# Predict RFI flags (single pass)
samrfi predict --model MODEL.pth --input OBSERVATION.ms

# Predict with iterative flagging (N passes)
samrfi predict --model MODEL.pth --input OBSERVATION.ms --iterations N

# Create default config
samrfi create-config --type {training|data} --output CONFIG.yaml

# Validate config
samrfi validate-config --config CONFIG.yaml
```

---

## Iterative Flagging Strategy

Iterative flagging progressively cleans deeper RFI by masking known flags in each pass:

**Why iterative flagging?**
- Bright RFI can hide fainter RFI
- After masking bright sources, fainter ones become visible
- Typically converges in 2-3 iterations

**How it works:**

```
Pass 1: Raw data → Model → Flags_1 (finds bright RFI)
Pass 2: Masked data (with Flags_1) → Model → Flags_2 (finds hidden RFI)
Pass 3: Masked data (with Flags_1|2) → Model → Flags_3 (final cleanup)

Final: Flags_cumulative = Flags_1 | Flags_2 | Flags_3
```

**When to use:**
- **Single pass (N=1)**: Fast, good for mild contamination
- **2-3 iterations**: Recommended for deep cleaning
- **>3 iterations**: Diminishing returns, risk over-flagging

**Example:**
```bash
# Compare single vs iterative
samrfi predict --model sam2.pth --input obs.ms  # 5% flagged
samrfi predict --model sam2.pth --input obs.ms --iterations 3  # 8% flagged
```

---

## Python API

### Load Measurement Set

```python
from samrfi.data import MSLoader

# Load MS
loader = MSLoader('observation.ms')
loader.load(num_antennas=5, mode='DATA')

# Access data
data = loader.data           # Complex visibilities
magnitude = loader.magnitude # Magnitude
flags = loader.load_flags()  # Existing flags

# Save new flags
loader.save_flags(new_flags)
```

### Preprocess Data

```python
from samrfi.data import Preprocessor

# Create preprocessor
preprocessor = Preprocessor(data, flags=flags)

# Generate dataset
dataset = preprocessor.create_dataset(
    patch_size=128,
    stretch='SQRT',
    flag_sigma=5,
    use_custom_flags=True
)

# Save dataset
dataset.save_to_disk('./my_dataset')
```

### Train Model

```python
from samrfi.training import SAM2Trainer
from datasets import load_from_disk

# Load dataset
dataset = load_from_disk('./my_dataset')

# Create trainer
trainer = SAM2Trainer(dataset, device='cuda')

# Train
trainer.train(
    num_epochs=10,
    batch_size=4,
    sam_checkpoint='large',
    learning_rate=1e-5,
    plot=True
)
```

### Apply Trained Model

```python
from samrfi.inference import RFIPredictor

# Load predictor
predictor = RFIPredictor(
    model_path='./models/sam2_rfi.pth',
    sam_checkpoint='large',
    device='cuda'
)

# Single-pass prediction
flags = predictor.predict_ms(
    ms_path='observation.ms',
    save_flags=True
)

# Iterative prediction (3 passes)
flags = predictor.predict_iterative(
    ms_path='observation.ms',
    num_iterations=3,
    save_flags=True
)

print(f"Flagged {flags.sum()/flags.size*100:.2f}% of data")
```

---

## Architecture

```
SAM-RFI Pipeline
================

[1] DATA GENERATION
    ├── MS File OR Synthetic Generator
    ├── MSLoader (load complex visibilities)
    ├── Preprocessor (patchify, normalize, stretch, flag)
    └── HuggingFace Dataset (saved to disk)

[2] TRAINING
    ├── Load Dataset
    ├── SAMDataset (PyTorch wrapper)
    ├── SAM2Trainer (transformers API)
    │   ├── Sam2Processor (image + prompt)
    │   ├── Sam2Model (Hiera backbone)
    │   └── DiceCELoss (segmentation loss)
    └── Trained Model (.pth)

[3] INFERENCE
    ├── Load MS
    ├── RFIPredictor (single or iterative)
    │   ├── Preprocess patches
    │   ├── Apply trained SAM2
    │   └── Reconstruct full flags
    └── Write Flags to MS
```

---

## Synthetic Data Features

### RFI Types
1. **Narrowband Persistent** - GPS, satellites (constant in time)
2. **Broadband Persistent** - Power lines, harmonics
3. **Narrowband Intermittent** - Periodic radar (duty cycle)
4. **Narrowband Bursty** - Random pulsed transmitters
5. **Broadband Bursty** - Lightning, transients
6. **Frequency Sweeps** - Linear & quadratic chirps (radar)

### Physical Realism
- **Noise**: 1 mJy (milli-Jansky) Gaussian
- **RFI Power**: 1000-10000 Jy (Jansky)
- **Dynamic Range**: 10^6 to 10^7 (matches real observations)
- **Bandpass Rolloff**: 8th-order polynomial edge effects
- **Polarization**: Correlated RFI in XX/YY

### Exact Ground Truth
Unlike real data, synthetic data provides **perfect masks**:
- We know exactly where RFI is (we generated it!)
- Enables training with 100% accurate labels
- Compare against MAD-based masks to quantify improvement

---

## Model Management

### Auto-Download Behavior

SAM2 models are **automatically downloaded from HuggingFace** when first needed:

```python
from samrfi.training import SAM2Trainer

# Model auto-downloads on first train() call
trainer = SAM2Trainer(dataset, device='cuda')
trainer.train(num_epochs=10, sam_checkpoint='large')  # Downloads ~850MB if not cached
```

**Available models:**
- `tiny` - 40 MB (fastest, lower accuracy)
- `small` - 180 MB (balanced)
- `base_plus` - 330 MB (good accuracy)
- `large` - 850 MB (best accuracy, **recommended**)

**Models are cached at:** `~/.cache/huggingface/hub/`

### Pre-Download Models (Optional)

To download models before training:

```python
from samrfi.utils import ModelCache

cache = ModelCache()
cache.download_model('large', show_progress=True)  # One-time download with progress bar
```

Or via command line:

```bash
python -c "from samrfi.utils import ModelCache; ModelCache().download_model('large')"
```

### Custom Cache Location

```bash
export HF_HOME=/path/to/custom/cache
samrfi train --config config.yaml --dataset dataset.npz
```

## Training Tips

### GPU Requirements
- **Minimum**: 8GB VRAM (batch_size=1, checkpoint=tiny)
- **Recommended**: 16GB VRAM (batch_size=4, checkpoint=large)
- **Optimal**: 24GB+ VRAM (batch_size=8+, checkpoint=large)

### Hyperparameter Tuning
```yaml
# Fast iteration (debugging)
model:
  checkpoint: tiny
training:
  num_epochs: 3
  batch_size: 8

# Production quality
model:
  checkpoint: large
training:
  num_epochs: 20
  batch_size: 4
  learning_rate: 1.0e-5
```

### Loss Convergence
- **Expected**: Loss should decrease from ~1.0 to <0.3 in 10 epochs
- **Warning**: If loss stuck >0.8 after 5 epochs, check:
  - Learning rate (try 5e-6 or 2e-5)
  - Data quality (visualize patches)
  - Batch size (try 2 or 8)

### Resume Training

Training can be resumed from any checkpoint to continue from where you left off:

```bash
# Initial training (10 epochs)
samrfi train --config config.yaml --dataset train.pt --epochs 10

# Resume and continue to epoch 20
samrfi train --config config.yaml --dataset train.pt --epochs 20 --resume ./samrfi_data/sam2_rfi_best.pth
```

**What gets restored:**
- Model weights
- Optimizer state (momentum, learning rates)
- Training/validation loss history
- Epoch counter (continues from N+1)

**Checkpoints saved:**
- `sam2_rfi_best.pth` - Best validation loss (updated during training)
- `model_sam2-large_*.pth` - Final checkpoint with full training state

**Python API:**
```python
trainer = SAM2Trainer(dataset, device='cuda')

# Resume from checkpoint
losses = trainer.train(
    num_epochs=20,
    batch_size=4,
    sam_checkpoint='large',
    model_path='./samrfi_data/sam2_rfi_best.pth'  # Resume from here
)
```

**Benefits:**
- Train in stages (evaluate after N epochs, continue if needed)
- Recover from crashes/interruptions
- Experiment with different learning rates from same checkpoint
- Loss curves show complete history across resume sessions

---

## Development

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_preprocessor.py -v

# With coverage
pytest tests/ --cov=samrfi --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

---

## Project Structure

```
SAM-RFI/
├── src/samrfi/
│   ├── data/                  # Data loading & preprocessing
│   │   ├── ms_loader.py       # CASA MS loader
│   │   ├── preprocessor.py    # Patch generation pipeline
│   │   └── sam_dataset.py     # PyTorch wrapper
│   ├── data_generation/       # Dataset generators
│   │   ├── ms_generator.py    # MS → dataset
│   │   └── synthetic_generator.py  # Synthetic → dataset
│   ├── training/              # Training
│   │   └── sam2_trainer.py    # SAM2 trainer
│   ├── inference/             # Prediction
│   │   └── predictor.py       # RFI prediction (single/iterative)
│   ├── config/                # Configuration
│   │   └── config_loader.py   # YAML config handling
│   └── cli.py                 # Command-line interface
│
├── tests/                     # Unit tests
├── configs/                   # Example configs
├── legacy/                    # Old implementation (archived)
├── pyproject.toml            # Package definition
└── README.md                 # This file
```

---

## Citation

If you use SAM-RFI in your research, please cite:

```bibtex
@software{samrfi2024,
  title = {SAM-RFI: Radio Frequency Interference Detection with SAM2},
  author = {Deal, Derod and Jagannathan, Preshanth},
  year = {2024},
  url = {https://github.com/preshanth/SAM-RFI}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **Meta AI** - SAM2 architecture and pre-trained models
- **HuggingFace** - Transformers library
- **NRAO** - Radio astronomy expertise and data
- **NAC** - National Astronomy Consortium funding

---

## Support

- **Issues**: https://github.com/preshanth/SAM-RFI/issues
- **Documentation**: https://sam-rfi.readthedocs.io
- **Contact**: pjaganna@nrao.edu
