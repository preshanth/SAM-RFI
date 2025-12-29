# Batched Dataset Training Pipeline

## Overview

The training pipeline uses a **batched dataset format** to avoid memory issues during generation and training. Instead of loading entire datasets into RAM, data is split into multiple batch files that are loaded on-demand.

## File Structure

Generated datasets have this structure:
```
datasets/train_4000/
├── exact_masks/
│   ├── batch_000.pt       # 100 samples (~1.3 GB)
│   ├── batch_001.pt
│   ├── ...
│   ├── batch_039.pt
│   └── metadata.json      # Dataset info
├── mad_masks/
│   ├── batch_000.pt
│   ├── ...
│   └── metadata.json
└── generation_metadata.json
```

## Dataset Generation

### Config Parameters

**Memory Management:**
```yaml
synthetic:
  generation_batch_size: 10  # Samples per generation batch (lower = less RAM)
  num_samples: 4000
```

**Processing:**
```yaml
processing:
  patch_size: 1024           # Patch size (1024 = full waterfall)
  num_workers: 4             # Parallel workers
  augmentation:
    rotations: true          # 4-way rotation augmentation
```

### Generate Dataset

```bash
samrfi generate-data \
  --source synthetic \
  --config configs/synthetic_train_4k.yaml \
  --output ./datasets/train_4000
```

**Output:**
- `exact_masks/` - Ground truth masks
- `mad_masks/` - MAD-detected masks
- Batch files written incrementally (no RAM accumulation)

## Training Pipeline

### Quick Start

```bash
# Generate datasets + train
python scripts/run_training.py --config configs/training_config.yaml

# Skip generation (use existing datasets)
python scripts/run_training.py --config configs/training_config.yaml --skip-generation
```

### Training Config

```yaml
# configs/training_config.yaml

data:
  # Dataset generation
  train_generation_config: configs/synthetic_train_4k.yaml
  train_dataset: ./datasets/train_4000
  val_generation_config: configs/synthetic_val_1k.yaml
  val_dataset: ./datasets/val_1000

  mask_type: exact_masks  # or mad_masks

training:
  device: cuda
  num_epochs: 10
  batch_size: 16
  learning_rate: 1.0e-5
  model_checkpoint: large  # tiny, small, base_plus, or large
  output_dir: ./training_output
```

### What Happens

1. **Generate Train Dataset** (if not skipped)
   - Uses `train_generation_config`
   - Writes to `train_dataset/exact_masks/` (batched)

2. **Generate Val Dataset** (if not skipped)
   - Uses `val_generation_config`
   - Writes to `val_dataset/exact_masks/` (batched)

3. **Train SAM2**
   - Loads batched datasets with LRU caching (3 batches in RAM)
   - Trains mask decoder only (encoders frozen)
   - Saves model checkpoints to `output_dir`

### Output

```
training_output/
├── samrfi_data/
│   └── models/
│       ├── model_sam2-large_..._epochs10_*.pth
│       └── loss_plot_sam2-large_..._epochs10_*.png
```

## GPU Validation

Test batched format before full training:

```bash
./run_validation.sh
```

**What it does:**
1. Generates 4k train + 1k val datasets (batched)
2. Profiles batch sizes on your GPU
3. Finds optimal batch size without OOM
4. Generates validation report

## Memory Profile

### Generation (per batch cycle):
- Raw data: ~400 MB (10 samples)
- After augmentation: ~1.6 GB (40 samples after 4-way rotation)
- Preprocessing: ~2 GB peak
- Written to disk immediately, then freed

### Training:
- BatchedDataset cache: ~3.9 GB (3 batch files × 1.3 GB)
- DataLoader workers: ~2 GB
- GPU batch: ~200 MB (batch_size=16)
- Peak RAM: ~6 GB
- Peak VRAM: Depends on model (large ≈ 8-10 GB)

## Troubleshooting

### OOM During Generation

**Symptom:** Process killed during preprocessing

**Fix:** Lower `generation_batch_size` in config:
```yaml
synthetic:
  generation_batch_size: 5  # Reduce from 10
```

### OOM During Training

**Fix 1:** Lower batch size in training config:
```yaml
training:
  batch_size: 8  # Reduce from 16
```

**Fix 2:** Use smaller model:
```yaml
training:
  model_checkpoint: small  # Instead of large
```

### Slow Loading

**Symptom:** Training waits for disk I/O

**Cause:** Batch files not cached, frequent disk reads

**Fix:** Increase cache size in `BatchedDataset`:
```python
dataset = BatchedDataset(path, cache_size=5)  # Default: 3
```

## Technical Details

### BatchedDataset

Loads data on-demand with LRU caching:
```python
from samrfi.data import BatchedDataset

dataset = BatchedDataset('./datasets/train_4000/exact_masks')
# Automatically caches 3 batch files (~3.9 GB)
# Loads new batches as needed
```

### BatchWriter

Writes datasets incrementally during generation:
```python
from samrfi.data import BatchWriter

writer = BatchWriter(output_dir, samples_per_batch=100)
for batch in batches:
    writer.add_batch(batch)  # Accumulates to 100, then writes
writer.finalize()  # Flush remaining + metadata
```

### Integration

Both `SAMDataset` and training scripts work with any dataset that has `__getitem__`:
- BatchedDataset (batched .npz files)
- NumpyDataset (single .npz file)
- HuggingFace Dataset

No code changes needed - just swap the dataset.

## Config Files Reference

### Generation Configs
- `configs/synthetic_train_4k.yaml` - 4000 training samples
- `configs/synthetic_val_1k.yaml` - 1000 validation samples
- `configs/synthetic_test_100.yaml` - 100 test samples

### Training Configs
- `configs/training_config.yaml` - Full pipeline config
- `configs/a100_validation.yaml` - GPU validation config

### Example Commands

```bash
# Test generation (100 samples)
samrfi generate-data \
  --source synthetic \
  --config configs/synthetic_test_100.yaml \
  --output ./datasets/test_100

# Full training pipeline
python scripts/run_training.py --config configs/training_config.yaml

# GPU validation
./run_validation.sh

# Training only (datasets exist)
python scripts/run_training.py \
  --config configs/training_config.yaml \
  --skip-generation
```
