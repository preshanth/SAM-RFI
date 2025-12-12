# SAM-RFI Canonical Training Workflow

This document describes the **single, canonical workflow** for training SAM2 models on RFI data using the `sam2_refactor` branch.

## Quick Start

```bash
# 1. Generate training data (4000 samples)
samrfi generate-data \
  --source synthetic \
  --config configs/synthetic_train_4k.yaml \
  --output ./datasets/train_4k

# 2. Generate validation data (1000 samples)
samrfi generate-data \
  --source synthetic \
  --config configs/synthetic_val_1k.yaml \
  --output ./datasets/val_1k

# 3. Train with validation
samrfi train \
  --config configs/gpu_v100_training.yaml \
  --dataset ./datasets/train_4k/exact_masks \
  --validation-dataset ./datasets/val_1k/exact_masks
```

## Data Generation

### Output Format

Data generation creates **batched directories** (not single files):

```
./datasets/train_4k/exact_masks/
├── batch_000.pt    # First 100 patches
├── batch_001.pt    # Next 100 patches
├── ...
├── batch_039.pt    # Last batch
└── metadata.json   # Dataset metadata
```

Each `batch_*.pt` file contains:
- `images`: Tensor of shape (N, 1024, 1024, 3) - RGB patches normalized with ImageNet stats
- `labels`: Tensor of shape (N, 1024, 1024) - Binary masks (0=clean, 1=RFI)

### Synthetic Data

**Configuration files:**
- `configs/synthetic_train_4k.yaml` - 4000 training samples
- `configs/synthetic_val_1k.yaml` - 1000 validation samples
- `configs/synthetic_test_100.yaml` - 100 test samples

**Key settings:**
```yaml
synthetic:
  num_samples: 4000
  num_channels: 1024  # Square for SAM2
  num_times: 1024

  # Physical scales (DO NOT normalize!)
  noise_mjy: 1.0           # 1 mJy noise
  rfi_power_min: 1000.0    # 1000 Jy
  rfi_power_max: 10000.0   # 10000 Jy

processing:
  normalize_before_stretch: false  # CRITICAL: preserve physical scales
  normalize_after_stretch: false
  stretch: null  # Or SQRT/LOG10 for real data
  patch_size: 1024
  enable_augmentation: true
  augmentation_rotations: 4  # Physics-preserving 4-way rotation
```

### Real Data (from MS)

```bash
samrfi generate-data \
  --source ms \
  --config configs/ms_data.yaml \
  --output ./datasets/vla_pband
```

**Configuration:**
```yaml
ms:
  path: /path/to/observation.ms
  num_antennas: 5
  data_mode: DATA  # or CORRECTED_DATA

processing:
  normalize_before_stretch: true   # Recommended for real data
  normalize_after_stretch: false
  stretch: SQRT
  patch_size: 128
  flag_sigma: 5
```

## Training

### CLI Command

```bash
samrfi train \
  --config configs/gpu_v100_training.yaml \
  --dataset ./datasets/train_4k/exact_masks \
  --validation-dataset ./datasets/val_1k/exact_masks \
  --device cuda \
  --output-dir ./models/experiment_01
```

### Training Configs

Different configs for different GPU memory:

| Config | GPU | VRAM | Batch Size |
|--------|-----|------|------------|
| `gpu_1080ti_training.yaml` | GTX 1080 Ti | 11GB | 2 |
| `gpu_v100_training.yaml` | V100 | 16GB | 4 |
| `gpu_a100_training.yaml` | A100 | 40GB | 8-12 |
| `gpu_h200_training.yaml` | H200 | 80GB | 16-24 |

### Key Training Settings

```yaml
model:
  checkpoint: large  # tiny, small, base_plus, large
  freeze_encoders: true  # Only train mask decoder

training:
  num_epochs: 10
  batch_size: 4
  learning_rate: 1.0e-5
  device: cuda
  use_gpu_transforms: false  # Set true for 10-100x speedup

output:
  dir_path: ./models/experiment_01
  save_plots: true
```

### Output

Training produces:
```
./models/experiment_01/samrfi_data/models/
├── model_sam2-large_stretch-null_sigma-5_patch-torch_size-1024_epochs10_20251212_143022.pth
└── loss_plot.png
```

Loss return format:
- **With validation**: `{"train": [0.8, 0.6, ...], "val": [0.9, 0.7, ...]}`
- **Without validation**: `[0.8, 0.6, 0.5, ...]`

## Dataset Formats (Technical)

### Current Format: Batched Directory

**Created by**: `BatchWriter` in `src/samrfi/data/torch_dataset.py`
**Loaded by**: `BatchedDataset` in `src/samrfi/data/sam_dataset.py`

**Advantages:**
- Streaming: Only loads batches as needed (low memory)
- Parallel I/O: Workers load batches independently
- OS cache friendly: Repeated access is fast
- No RAM budget needed

**Structure:**
```python
{
    'images': torch.Tensor,  # (batch_size, 1024, 1024, 3) float32
    'labels': torch.Tensor   # (batch_size, 1024, 1024) float32
}
```

### Legacy Formats (Backward Compatible)

1. **Single .pt file**: Old TorchDataset format (deprecated)
2. **HuggingFace directory**: Arrow tables format (slow, deprecated)

The CLI automatically detects format based on:
- Directory + `metadata.json` → BatchedDataset
- Directory + `dataset_info.json` → HuggingFace
- `.pt` file → TorchDataset

## Validation During Training

Validation is **optional but recommended**:

```bash
# Training only (faster)
samrfi train --config config.yaml --dataset ./train/

# Training + Validation (recommended)
samrfi train --config config.yaml \
  --dataset ./train/exact_masks \
  --validation-dataset ./val/exact_masks
```

**When validation is used:**
- Validation runs after each epoch
- No gradient updates (eval mode)
- Returns both train and val losses
- Can track overfitting

**Output:**
```
Epoch 1/10 [Train]: ...
Epoch 1/10 [Val]: ...
EPOCH: 1/10 | Train loss: 0.8234 | Val loss: 0.8567

...

Final train loss: 0.3421
Best train loss: 0.3421
Final val loss: 0.3789
Best val loss: 0.3789
```

## Common Pitfalls

1. **Wrong path format**: Use directory path, not `.pt` file
   ```bash
   # ✗ WRONG
   --dataset ./datasets/train/exact_masks.pt

   # ✓ CORRECT
   --dataset ./datasets/train/exact_masks
   ```

2. **Normalizing synthetic data**: Destroys physical scales
   ```yaml
   # ✗ WRONG for synthetic
   normalize_before_stretch: true

   # ✓ CORRECT for synthetic
   normalize_before_stretch: false
   normalize_after_stretch: false
   ```

3. **Patch size mismatch**: Training and inference must match
   ```bash
   # Train with 128
   samrfi train ... (patch_size: 128 in config)

   # ✓ Predict with 128
   samrfi predict --patch-size 128 ...
   ```

4. **Using deprecated scripts**: Use CLI only
   ```bash
   # ✗ DEPRECATED (moved to deprecated/)
   python scripts/train_sam2.py
   python scripts/run_training.py

   # ✓ USE CLI
   samrfi train ...
   ```

## Full Example Workflow

```bash
# Setup
conda activate samrfi
cd /path/to/SAM-RFI

# 1. Generate datasets
samrfi generate-data \
  --source synthetic \
  --config configs/synthetic_train_4k.yaml \
  --output ./datasets/train_4k

samrfi generate-data \
  --source synthetic \
  --config configs/synthetic_val_1k.yaml \
  --output ./datasets/val_1k

# 2. Train model
samrfi train \
  --config configs/gpu_v100_training.yaml \
  --dataset ./datasets/train_4k/exact_masks \
  --validation-dataset ./datasets/val_1k/exact_masks \
  --output-dir ./models/exp01

# 3. Apply to real data
samrfi predict \
  --model ./models/exp01/samrfi_data/models/model_*.pth \
  --input /path/to/observation.ms \
  --iterations 3

# 4. Check results
# Flags are written to MS FLAG column
# Use CASA plotms or similar to visualize
```

## Advanced: GPU Transforms

For 10-100x speedup during training:

```yaml
training:
  use_gpu_transforms: true
```

This moves data preprocessing to GPU (requires Kornia dependency). The transforms are physics-preserving and match the CPU implementation exactly.

## Next Steps: SAM3 Integration

Meta released SAM3 (Nov 2025) with:
- Text-prompted segmentation
- 848M parameters
- Better concept discrimination

Future workflow may support text prompts:
```bash
samrfi train --prompt "narrowband persistent RFI" ...
```

See CLAUDE.md for SAM3 migration path.
