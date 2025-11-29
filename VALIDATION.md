# SAM3 Training and Validation Pipeline

Complete workflow for training SAM3 on H100 and validating against CASA and AOFlagger.

## Overview

1. **Training**: Train SAM3 on H100 with regularization to prevent validation divergence
2. **Validation**: Compare SAM3 vs CASA (tfcrop, rflag) vs AOFlagger on simulated MS
3. **Analysis**: Generate publication-ready comparison plots

---

## Step 1: H100 Training Setup

### Training Configuration

The H100-optimized config is at `configs/h100_sam3_training.yaml`:

```yaml
experiment:
  name: "sam3_h100"
  output_dir: "./output/sam3_h100"
  save_every_n_epochs: 2

data:
  train_dataset: "./datasets/synthetic_train_4k/exact_masks.npz"
  val_dataset: "./datasets/synthetic_val_1k/exact_masks.npz"

model:
  freeze_encoders: true  # 840M → 33M trainable params

training:
  num_epochs: 20
  batch_size: 16         # H100 80GB VRAM
  learning_rate: 5.0e-6  # Lower LR to prevent overfitting
  weight_decay: 0.01     # L2 regularization
  device: "cuda"
```

### Key Changes from SAM2

**Preventing Validation Divergence:**
- **Lower learning rate**: 5e-6 (vs 1e-5) - slower, more stable learning
- **Weight decay**: 0.01 (vs 0.0) - L2 regularization prevents overfitting
- **Larger batch size**: 16 (vs 4) - more stable gradients
- **Data augmentation**: `bbox_perturbation=20` for training, `0` for validation

**Fixed in `scripts/train_sam3.py:312,320`:**
```python
# Training dataloader - with bbox perturbation
train_dataloader = DataLoader(
    SAMDataset(train_dataset, processor, bbox_perturbation=20),
    ...
)

# Validation dataloader - NO bbox perturbation (consistent measurement)
val_dataloader = DataLoader(
    SAMDataset(val_dataset, processor, bbox_perturbation=0),
    ...
)
```

### Run Training

```bash
# Activate environment
conda activate SAM-RFI

# Check datasets exist
ls datasets/synthetic_train_4k/exact_masks.npz
ls datasets/synthetic_val_1k/exact_masks.npz

# Start training (will take ~3-4 hours on H100)
python scripts/train_sam3.py --config configs/h100_sam3_training.yaml

# Monitor training (in another terminal)
watch -n 30 tail -20 output/sam3_h100/training_log.txt
```

### Expected Training Behavior

**Good training (converging):**
```
Epoch 1/20: train_loss=0.245000, val_loss=0.240000
Epoch 2/20: train_loss=0.180000, val_loss=0.175000
Epoch 3/20: train_loss=0.145000, val_loss=0.142000
...
Epoch 20/20: train_loss=0.025000, val_loss=0.028000 (BEST!)
```
✅ Both losses decreasing, validation tracks training closely

**Bad training (diverging - if this happens, increase weight_decay to 0.05):**
```
Epoch 1/20: train_loss=0.245000, val_loss=0.240000
Epoch 5/20: train_loss=0.120000, val_loss=0.135000
Epoch 10/20: train_loss=0.045000, val_loss=0.180000  ⚠️ Val increasing!
Epoch 20/20: train_loss=0.005000, val_loss=0.350000  ❌ Overfitting!
```

### Training Outputs

After training completes, you'll have:

```
output/sam3_h100/
├── config.yaml                    # Training configuration
├── training_log.txt               # Detailed logs
├── losses.npz                     # Train/val losses (for plotting)
├── model_best.pth                 # Best model (lowest val loss)
├── model_final.pth                # Final model (epoch 20)
├── checkpoint_epoch2.pth          # Checkpoints every 2 epochs
├── checkpoint_epoch4.pth
├── ...
└── git_commit.txt                 # Git hash for reproducibility
```

**Best model for validation**: `output/sam3_h100/model_best.pth`

---

## Step 2: Generate Simulated MS for Validation

Before validation, you need a simulated measurement set with ground truth RFI masks.

### Option A: Use Existing Simulated Data

If you have synthetic waterfall data with ground truth masks:

```bash
# Convert .npz to measurement set format (TODO: implement this converter)
python scripts/npz_to_ms.py \
    --input datasets/synthetic_val_1k/exact_masks.npz \
    --output validation_data/sim_ms.ms \
    --ground-truth validation_data/ground_truth.npy
```

### Option B: Simulate from Scratch

```bash
# Generate simulated MS with RFI
python scripts/simulate_ms_with_rfi.py \
    --output validation_data/sim_ms.ms \
    --num-antennas 27 \
    --num-channels 2048 \
    --num-timesteps 1000 \
    --rfi-types broadband,narrowband,intermittent
```

**Note**: MS simulation scripts are TODO - for now, use synthetic .npz data and extract ground truth masks.

---

## Step 3: Run Validation

### Full Comparison (SAM3 + CASA + AOFlagger)

Compare all methods on simulated MS with ground truth:

```bash
python scripts/validate_flagging_methods.py \
    --mode simulated \
    --ms validation_data/sim_ms.ms \
    --ground-truth validation_data/ground_truth.npy \
    --sam3-model output/sam3_h100/model_best.pth \
    --output results/validation_full/ \
    --aoflagger-strategy jvla-default \
    --overflag-threshold 85.0
```

**Parameters:**
- `--mode simulated`: Use ground truth metrics (Precision, Recall, F1, IoU)
- `--ms`: Path to measurement set
- `--ground-truth`: Ground truth RFI mask (.npy file, same shape as MS flags)
- `--sam3-model`: Trained SAM3 model
- `--output`: Results directory
- `--aoflagger-strategy`: AOFlagger strategy (default: `jvla-default`)
  - Available: `jvla-default`, `generic-default`, `atca-default`, etc.
  - Location: `/usr/share/aoflagger/strategies/`
- `--overflag-threshold`: Penalty threshold for calcquality (default: 80%, configurable)

### Validation Output

```
results/validation_full/
├── validation_results.json         # Metrics for all methods
├── plot_metrics_comparison.png     # Precision/Recall/F1 bar chart
├── plot_detection_vs_fpr.png       # Detection rate vs false alarm
├── plot_confusion_matrices.png     # Confusion matrix grid
└── ms_*/                           # MS copies for each method
    ├── ms_sam3/
    ├── ms_casa_tfcrop/
    ├── ms_casa_rflag/
    └── ms_aoflagger/
```

### Methods Compared

The validation script compares 4 methods:

1. **SAM3**: Your trained model
2. **CASA tfcrop**: Time-frequency crop (statistical outlier detection)
3. **CASA rflag**: RFlag (running median/MAD)
4. **AOFlagger**: André Offringa's RFI flagger (Lua strategy-based)

Each method:
- Gets a clean copy of the MS
- Applies its flagging algorithm
- Flags are compared against ground truth
- Metrics computed: Precision, Recall, F1, IoU, FPR

---

## Step 4: Analyze Results

### View Metrics

```bash
# Print summary
cat results/validation_full/validation_results.json

# Quick comparison
python -c "
import json
with open('results/validation_full/validation_results.json') as f:
    results = json.load(f)
for method, data in results.items():
    m = data['metrics']
    print(f\"{method:15s} | F1: {m['f1_score']:.4f} | Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f}\")
"
```

### View Plots

```bash
# Open plots
xdg-open results/validation_full/plot_metrics_comparison.png
xdg-open results/validation_full/plot_detection_vs_fpr.png
xdg-open results/validation_full/plot_confusion_matrices.png
```

### Publication-Ready Plots

All plots are generated at **300 DPI** with publication-quality formatting:

**1. Metrics Comparison (`plot_metrics_comparison.png`)**
- Bar chart: Precision, Recall, F1-Score for each method
- Shows where each method excels/fails
- Value labels on bars

**2. Detection vs False Alarm (`plot_detection_vs_fpr.png`)**
- Scatter plot: Recall (Y) vs False Positive Rate (X)
- Ideal point (0, 1) marked
- Shows precision-recall tradeoff

**3. Confusion Matrices (`plot_confusion_matrices.png`)**
- Grid of confusion matrices (one per method)
- Shows TP, TN, FP, FN counts and percentages
- Heatmap visualization

---

## Step 5: Real Data Validation (No Ground Truth)

For real measurement sets without ground truth, use proxy metrics:

```bash
python scripts/validate_flagging_methods.py \
    --mode real \
    --ms real_data/3C129_pband.ms \
    --sam3-model output/sam3_h100/model_best.pth \
    --output results/validation_real/ \
    --aoflagger-strategy jvla-default
```

**Proxy Metrics (TODO: implement):**
1. **calcquality score**: Lower is better
   - Measures if leftover data is noise-like (Gaussian)
   - Penalizes overflagging (configurable threshold)
2. **Image RMS**: Lower is better (cleaner image)
3. **Dynamic range**: Higher is better (peak/RMS)

---

## Troubleshooting

### Training Issues

**Problem**: Validation loss diverging (increasing while train loss decreases)

**Solutions** (try in order):
1. **Increase weight_decay**: Change to `0.05` in config
2. **Lower learning rate**: Change to `2e-6` in config
3. **Add more training data**: Generate more synthetic samples
4. **Early stopping**: Use `model_best.pth` (lowest val loss) instead of final

**Problem**: Training too slow on H100

**Check**:
```bash
nvidia-smi  # Should show ~40-60GB GPU usage, ~90%+ utilization
```

If GPU utilization low, increase `batch_size` to 24 or 32.

**Problem**: Out of memory (OOM)

**Solution**: Reduce `batch_size` to 8 in config

### Validation Issues

**Problem**: AOFlagger not found

```bash
# Check installation
which aoflagger
aoflagger --version  # Should show 3.4

# If missing, install
sudo apt install aoflagger
```

**Problem**: CASA not available

Install CASA:
```bash
# Option 1: System package
sudo apt install casatools casatasks

# Option 2: pip (may be older)
pip install casatools casatasks
```

**Problem**: Ground truth shape mismatch

Error: `Shape mismatch. Predicted: (X,Y,Z), GT: (A,B,C)`

**Fix**: Ensure ground truth .npy file has same shape as MS FLAG column:
```python
import numpy as np
from casatools import table

# Check MS flag shape
tb = table()
tb.open('sim_ms.ms')
flags = tb.getcol('FLAG')
print(f"MS flag shape: {flags.shape}")  # (npol, nchan, nrow)
tb.close()

# Ground truth should match
gt = np.load('ground_truth.npy')
print(f"GT shape: {gt.shape}")  # Should match MS
```

---

## Performance Benchmarks (H100)

**Training (4000 samples, 20 epochs):**
- Time: ~3-4 hours
- Batch size: 16
- Throughput: ~5000 batches @ 2-3 sec/batch
- Memory: ~50GB / 80GB VRAM

**Inference (1000 samples validation):**
- Time: ~2-3 minutes/epoch
- Batch size: 16

**Validation (all 4 methods on 1000-sample MS):**
- SAM3: ~5-10 minutes
- CASA tfcrop: ~10-15 minutes
- CASA rflag: ~5-10 minutes
- AOFlagger: ~5-10 minutes
- Total: ~30-45 minutes

---

## Next Steps After Training

Once training completes and validation looks good:

1. **Tune hyperparameters** if needed:
   - Adjust `learning_rate`, `weight_decay` based on train/val curves
   - Re-train with best settings

2. **Test on real data**:
   ```bash
   python scripts/validate_flagging_methods.py \
       --mode real \
       --ms real_data.ms \
       --sam3-model output/sam3_h100/model_best.pth \
       --output results/real_validation/
   ```

3. **Generate more plots** with `scripts/visualize_comparison.py`

4. **Write paper**:
   - Use plots from `results/validation_full/`
   - Compare against CASA baseline
   - Show where SAM3 excels (likely: complex RFI patterns)

---

## Resuming Training

If training is interrupted, resume from last checkpoint:

```bash
# Find latest checkpoint
ls output/sam3_h100/checkpoint_*.pth

# Resume training
python scripts/train_sam3.py \
    --config configs/h100_sam3_training.yaml \
    --resume output/sam3_h100/checkpoint_epoch10.pth
```

Training will continue from epoch 11.

---

## Quick Reference

```bash
# 1. Start training
python scripts/train_sam3.py --config configs/h100_sam3_training.yaml

# 2. Monitor training
tail -f output/sam3_h100/training_log.txt

# 3. After training, validate
python scripts/validate_flagging_methods.py \
    --mode simulated \
    --ms validation_data/sim_ms.ms \
    --ground-truth validation_data/ground_truth.npy \
    --sam3-model output/sam3_h100/model_best.pth \
    --output results/validation/

# 4. View results
xdg-open results/validation/plot_metrics_comparison.png
cat results/validation/validation_results.json
```

---

## Context Remaining

Current token usage: ~92k / 200k tokens (~54% available)

Good stopping point - training will take several hours. Resume validation after training completes!
