# Working Guidelines for Claude

## CRITICAL: No Code Without Design Approval

1. **STOP and DIAGNOSE first:**
   - What is the ACTUAL error/problem? (exact error message, line numbers)
   - What are the ACTUAL data types, shapes, memory usage? (profile/measure, don't guess)
   - What is the root cause? (not symptoms)

2. **PROPOSE solution, GET APPROVAL:**
   - Present 2-3 design options with pros/cons
   - Show memory/performance calculations WITH CORRECT DTYPES
   - Get explicit approval before writing ANY code
   - If assumptions needed, STATE THEM CLEARLY and validate first

3. **IMPLEMENT only after approval:**
   - Make approved changes only
   - Test incrementally
   - No scope creep

## Engineering Principles

- **Simple > Clever:** Working code beats elegant code
- **Measure > Guess:** Profile actual usage, don't estimate
- **Validate assumptions:** Check dtypes, shapes, memory before calculating
- **Think like a senior scientist:** Understand the physics/math, then code
- **Under-promise, over-deliver:** Don't add features that weren't requested

## Red Flags (STOP if doing these)

- ❌ Making calculations without checking actual dtypes
- ❌ Writing code before design is approved
- ❌ Adding "nice to have" features without asking
- ❌ Assuming memory/performance without measuring
- ❌ Over-engineering simple problems
- ❌ Writing redundant summaries when work is already documented

## Workflow That Works

**I write code, you execute and report results.**

- I propose code/tests
- You run them and tell me what happened
- I analyze YOUR actual results (not my assumptions)
- We iterate based on REAL data

**Even for small tests:** You run them. I don't guess outcomes.

## Communication

- Provide confidence estimates (High/Medium/Low) with evidence
- Back up findings with actual code references (file:line)
- Think critically - challenge assumptions, including mine
- Be direct and concise - respect the user's time
- **AVOID REDUNDANT SUMMARIES** - don't repeat what's already documented

---

# SAM-RFI v2.0 Project Structure

## Overview

SAM-RFI applies Meta's Segment Anything Model 2 (SAM2) to detect and flag Radio Frequency Interference (RFI) in radio astronomy data. Built on HuggingFace transformers with a clean, modular v2.0 architecture.

**Version:** 2.0.0
**Branch:** `sam2`
**Main Branch:** `main`

---

## Directory Structure

```
SAM-RFI/
├── src/samrfi/              # Production code (v2.0)
│   ├── data/                # Data loading & preprocessing
│   │   ├── ms_loader.py           # CASA measurement set loader
│   │   ├── preprocessor.py        # Patchify, normalize, channel extraction
│   │   ├── sam_dataset.py         # PyTorch Dataset wrapper
│   │   ├── numpy_dataset.py       # .npz format (efficient, no 2GB limit)
│   │   └── hf_dataset_wrapper.py  # HuggingFace conversion
│   │
│   ├── data_generation/     # Dataset generators (one-time)
│   │   ├── synthetic_generator.py # Realistic RFI simulation
│   │   └── ms_generator.py        # MS → dataset converter
│   │
│   ├── training/            # Model training
│   │   └── sam2_trainer.py        # SAM2 training (HF transformers)
│   │
│   ├── inference/           # Apply trained models
│   │   └── predictor.py           # Single + iterative flagging
│   │
│   ├── config/              # Configuration management
│   │   └── config_loader.py       # YAML configs (DataConfig, TrainingConfig)
│   │
│   ├── utils/               # Utilities
│   │   └── model_cache.py         # SAM2 model auto-download from HuggingFace
│   │
│   └── cli.py               # Command-line interface (samrfi)
│
├── configs/                 # YAML configuration files
│   ├── experiments/         # 4 research scenarios
│   │   ├── exp1_synthetic.yaml        # Pure synthetic baseline
│   │   ├── exp2_synthetic_real.yaml   # Mixed training
│   │   ├── exp3_real_threshold.yaml   # Automated flags
│   │   └── exp4_real_human.yaml       # Human-annotated gold standard
│   ├── training_config.yaml          # Full training params
│   ├── synthetic_train_4k.yaml       # 4K synthetic samples
│   ├── synthetic_val_1k.yaml         # 1K validation samples
│   └── v100_validation.yaml          # GPU validation config
│
├── scripts/                 # Training utilities
│   ├── train_sam2.py               # Experiment tracking script
│   ├── run_training.py             # Automated training runner
│   ├── plot_training_results.py    # Loss curve plotting
│   ├── README.md                   # Training workflow docs
│   └── QUICKSTART.md               # One-page training guide
│
├── tests/                   # Unit tests (96% coverage)
│   ├── test_data_generators.py
│   ├── test_sam2_trainer.py
│   ├── test_config_loader.py
│   └── test_cli.py
│
├── docs/                    # ReadTheDocs documentation
│   ├── index.rst                   # Landing page (v2.0 description)
│   ├── installation.rst            # Complete install guide
│   ├── quickstart.rst              # Quick start tutorial
│   ├── api.rst                     # Full API reference
│   ├── conf.py                     # Sphinx config
│   ├── SAM2_native_resolution_findings.md
│   ├── batched_dataset_training.md
│   └── future_directions.md
│
├── legacy/                  # Old v1.0 code (archived, DO NOT USE)
│   ├── samrfi/                     # Old implementation
│   ├── old_training_scripts/       # Old training/ directory
│   └── old_testing_scripts/        # Old testing/ directory
│
├── archive/                 # Archived materials
│   ├── notebooks/                  # Old Jupyter notebooks (won't work with v2.0)
│   └── docs/                       # Historical documentation
│
├── validate_gpu.py          # GPU profiling script (standalone)
├── run_validation.sh        # Validation pipeline automation
│
├── README.md                # Main user documentation
├── refactor_plan.md         # Complete v2.0 architecture docs
├── CLAUDE.md                # This file
├── pyproject.toml           # Package configuration
├── pytest.ini               # Test configuration
└── .gitignore               # Ignores models/, datasets/, *.npz, *.pth, etc.
```

---

## Key Concepts

### 1. Data Flow

```
[MS File or Synthetic]
    ↓
[Data Generation] → datasets/*.npz (NumpyDataset format)
    ↓
[Preprocessing] → Complex → 3 channels (gradient, log_amp, phase)
    ↓
[Training] → SAM2Trainer (HF transformers)
    ↓
[Model] → *.pth checkpoint
    ↓
[Inference] → RFIPredictor (single or iterative)
    ↓
[Output] → FLAGS written to MS
```

### 2. Dataset Formats

**NumpyDataset (.npz)** - CURRENT FORMAT ✅
- Efficient numpy-backed format
- No 2GB Arrow limit (was issue with HuggingFace Dataset)
- 10× faster loading, 50-70% smaller files
- File: `src/samrfi/data/numpy_dataset.py`

**HuggingFace Dataset** - LEGACY (still supported for publishing)
- Optional conversion via `hf_dataset_wrapper.py`
- Used for publishing to HuggingFace Hub
- File: `src/samrfi/data/hf_dataset_wrapper.py`

### 3. Model Auto-Download

**SAM2 models auto-download from HuggingFace on first use:**
- `tiny` - 40 MB
- `small` - 180 MB
- `base_plus` - 330 MB
- `large` - 850 MB (recommended)

**Cache location:** `~/.cache/huggingface/hub/`

**Utility:** `src/samrfi/utils/model_cache.py`
- Check cache: `ModelCache().is_cached('large')`
- Pre-download: `ModelCache().download_model('large')`
- Load: `model, processor = ModelCache().load_model('large')`

### 4. Preprocessing Pipeline

**Physical Scale Preservation** (src/samrfi/data/preprocessor.py:377-379)
- Fixed normalization: `LOG_MIN=-3.0` (1 mJy), `LOG_MAX=4.0` (10,000 Jy)
- Preserves absolute physical meaning across patches
- Per-patch min-max destroyed this (was bug)

**3-Channel Extraction** (preprocessor.py:_extract_channels_from_complex)
- R = Gradient (spatial edges - makes RFI pop!)
- G = Log Amplitude (intensity)
- B = Phase (polarimetric signature)
- Preserves 10^6 dynamic range (PIL conversion destroyed this)

**SAM2 Preprocessing Location**
- Applied ONCE during dataset generation (preprocessor.py:550-568)
- NOT during training (was bottleneck: 0.1 batch/s → 5-10× speedup)
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### 5. Training Configuration

**All hyperparameters in YAML** (configs/training_config.yaml)
- Optimizer: adam/adamw/sgd, weight_decay, betas, eps
- Loss: dicece/dice/ce/focal, sigmoid, squared_pred
- Model: freeze_vision/prompt_encoder, multimask_output
- DataLoader: num_workers, cache_size, prefetch_factor
- Training: log_interval, cuda_cache_clear_interval

**Principle:** Touch code once, configure forever

### 6. Experiment Tracking

**4 Research Scenarios** (configs/experiments/exp{1-4}_*.yaml)
1. **exp1_synthetic** - Pure synthetic (baseline ceiling)
2. **exp2_synthetic_real** - Mixed training (generalization)
3. **exp3_real_threshold** - Automated flags (noisy labels)
4. **exp4_real_human** - Human annotations (gold standard)

**Training Script** (scripts/train_sam2.py)
- Saves losses.npz after each epoch
- Best model checkpointing (lowest val loss)
- Config + git hash archiving (reproducibility)
- Resume from checkpoint support

---

## Important Files to Know

### Configuration
- `configs/training_config.yaml` - Full training pipeline config
- `configs/synthetic_train_4k.yaml` - Synthetic data generation (4K samples)
- `configs/experiments/` - 4 research scenarios

### Core Implementation
- `src/samrfi/training/sam2_trainer.py` - SAM2 trainer (507 lines)
- `src/samrfi/data/preprocessor.py` - Preprocessing pipeline (568 lines)
- `src/samrfi/data_generation/synthetic_generator.py` - RFI simulation (619 lines)
- `src/samrfi/inference/predictor.py` - Iterative flagging (356 lines)
- `src/samrfi/utils/model_cache.py` - Model auto-download (330 lines)

### Documentation
- `README.md` - User guide (quick start, API examples, CLI)
- `refactor_plan.md` - Complete v2.0 architecture (46KB, comprehensive)
- `docs/` - ReadTheDocs (installation, quickstart, API reference)
- `scripts/README.md` - Training workflow guide
- `scripts/QUICKSTART.md` - One-page experiment guide

### Testing
- `tests/` - 52 unit tests (96% coverage)
- `pytest.ini` - Test configuration
- Run: `pytest tests/ -v`

---

## Common Tasks

### Generate Synthetic Data
```bash
samrfi generate-data \
  --source synthetic \
  --config configs/synthetic_train_4k.yaml \
  --output ./datasets/train_4k
```

**Output:** `exact_masks.npz` + `mad_masks.npz`

### Train Model
```bash
samrfi train \
  --config configs/training_config.yaml \
  --dataset ./datasets/train_4k/exact_masks.npz \
  --validation-dataset ./datasets/val_1k/exact_masks.npz
```

**Or with experiment tracking:**
```bash
python scripts/train_sam2.py --config configs/experiments/exp1_synthetic.yaml
```

### Run Inference
```bash
# Single-pass
samrfi predict --model model.pth --input observation.ms

# Iterative (3 passes)
samrfi predict --model model.pth --input observation.ms --iterations 3
```

### Check Model Cache
```python
from samrfi.utils import ModelCache

cache = ModelCache()
cache.clear_cache()  # Shows status of all models
```

### Run Tests
```bash
pytest tests/ -v
pytest tests/test_sam2_trainer.py -v  # Specific test
```

---

## What NOT to Do

❌ **Don't use legacy code** - Everything in `legacy/` is archived v1.0
❌ **Don't manually download models** - Auto-downloads from HuggingFace
❌ **Don't commit large files** - Use .gitignore (models/, datasets/, *.npz, *.pth)
❌ **Don't use HuggingFace Dataset** - Use NumpyDataset (.npz format)
❌ **Don't modify preprocessor without understanding physical scales** - See preprocessor.py:377-379
❌ **Don't run preprocessing during training** - Should be in dataset generation

---

## Design Decisions to Remember

1. **NumpyDataset over HuggingFace Dataset**
   - Reason: 2GB Arrow overflow, 10× faster, 50-70% smaller
   - Files: numpy_dataset.py, hf_dataset_wrapper.py

2. **Preprocessing in Dataset Generation, Not Training**
   - Reason: SAM2 processor ran 160,000× (0.1 batch/s), GPU 15% utilized
   - Fix: Apply once in preprocessor.py:550-568
   - Speedup: 5-10×

3. **Fixed Physical Scale Normalization**
   - Reason: Per-patch min-max destroyed absolute intensity meaning
   - Fix: LOG_MIN=-3.0, LOG_MAX=4.0 (preprocessor.py:377-379)

4. **3-Channel Extraction from Complex Data**
   - Reason: PIL conversion destroyed 10^6 dynamic range
   - Fix: Gradient (edges), log_amp, phase channels
   - File: preprocessor.py:_extract_channels_from_complex

5. **All Hyperparameters in YAML**
   - Reason: Touch code once, configure forever
   - File: configs/training_config.yaml (58 lines, ~20 params)

6. **Training Memory Leak Fixes**
   - Issue: CPU RAM filling to 128GB, killing nodes at 40%
   - Fixes: Removed TQDM, disabled profiling, explicit tensor cleanup
   - Files: sam2_trainer.py, configs/*_validation.yaml

7. **SAM2 Native Resolution (1024×1024)**
   - Reason: Matches SAM2 training resolution (optimal performance)
   - No patching when patch_size >= image dimensions
   - File: preprocessor.py, configs/synthetic_train_4k.yaml

---

## Git Workflow

**Current branch:** `sam2`
**Main branch:** `main`

**What's ignored (.gitignore):**
- `models/` - Auto-downloaded SAM2 weights
- `datasets/` - Generated training data
- `tmp/`, `validation_results/` - Temporary outputs
- `*.pth`, `*.npz`, `*.safetensors` - Model/dataset files
- `archive/` - Historical materials

**Clean repo size:** <10MB (was 28GB before cleanup)

---

## Performance Notes

**GPU Utilization:**
- Before preprocessing fix: 15-20% (CPU bottleneck)
- After: 80%+ (GPU-bound, as expected)

**Training Speed:**
- Before: 0.1 batch/s (2.7 hours/epoch)
- After: Expected 5-10× faster with preprocessing in dataset generation

**Memory:**
- V100 32GB: batch_size=4 recommended
- A100 40GB: batch_size=16+ supported
- Training params in configs/training_config.yaml

**Dataset Sizes:**
- Synthetic 4K: ~11GB (exact_masks.npz)
- Synthetic 1K: ~2.8GB (exact_masks.npz)
- MAD masks typically 1.3× larger than exact

---

## When Things Break

1. **ImportError for samrfi modules**
   - Check: `pip install -e .[dev]` from repo root
   - Check: Python path includes `src/`

2. **CUDA out of memory**
   - Reduce batch_size in training config
   - Clear cache: `torch.cuda.empty_cache()`

3. **Training loss not decreasing**
   - Check data: Are images/masks loading correctly?
   - Check normalization: Should be ImageNet stats
   - Check learning rate: Default 1e-5

4. **Model auto-download failing**
   - Check internet connection
   - Check: `~/.cache/huggingface/` permissions
   - Try: `export HF_HOME=/path/with/space`

5. **Dataset generation OOM**
   - Reduce num_samples in config
   - Check batch processing in synthetic_generator.py

---

## Documentation Locations

**User docs:** README.md, docs/
**Technical docs:** refactor_plan.md (comprehensive)
**Training guide:** scripts/README.md, scripts/QUICKSTART.md
**API reference:** docs/api.rst (ReadTheDocs)
**This file:** CLAUDE.md (project orientation)

---

## Version History

**v2.0.0** (Current)
- HuggingFace transformers SAM2 API
- NumpyDataset format
- Model auto-download
- Complete training pipeline
- 96% test coverage

**v1.0** (Legacy in `legacy/`)
- Manual SAM2 implementation
- HuggingFace Dataset (2GB limit issues)
- No validation tracking
- Archived, do not use

---

## Contact

**Authors:** Derod Deal, Preshanth Jagannathan
**GitHub:** https://github.com/preshanth/SAM-RFI
**Issues:** https://github.com/preshanth/SAM-RFI/issues
