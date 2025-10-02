# SAM-RFI Training Scripts

Comprehensive training and evaluation workflow for SAM2 RFI detection experiments.

## Quick Start

```bash
# 1. Generate synthetic training/validation data
samrfi generate-data --source synthetic --config configs/synthetic_train_4k.yaml --output datasets/synthetic_train_4k
samrfi generate-data --source synthetic --config configs/synthetic_val_1k.yaml --output datasets/synthetic_val_1k

# 2. Run Experiment 1 (pure synthetic)
python scripts/train_sam2.py --config configs/experiments/exp1_synthetic.yaml

# 3. Plot results
python scripts/plot_training_results.py --experiment output/exp1_synthetic --summary
```

## Training Workflow

### 1. Data Preparation

Generate datasets using the CLI:

```bash
# Synthetic data (exact ground truth)
samrfi generate-data --source synthetic \
    --config configs/synthetic_train_4k.yaml \
    --output datasets/synthetic_train_4k

# Real data from MS (threshold flags)
samrfi generate-data --source ms \
    --config configs/real_ms_data.yaml \
    --output datasets/real_train
```

**Output format:**
- `exact_masks.npz` - Perfect ground truth (synthetic only)
- `mad_masks.npz` - MAD threshold flags
- `metadata.json` - Dataset generation parameters

### 2. Training

Use the standalone training script for full experiment tracking:

```bash
python scripts/train_sam2.py --config configs/experiments/exp1_synthetic.yaml
```

**Key features:**
- ✓ Train/validation loss tracking
- ✓ Best model checkpointing (lowest val loss)
- ✓ Experiment config archiving
- ✓ Git commit tracking
- ✓ Resume from checkpoint
- ✓ Structured logging

**Output structure:**
```
output/exp1_synthetic/
├── config.yaml              # Archived config
├── git_commit.txt           # Git hash for reproducibility
├── training_log.txt         # Full training log
├── losses.npz               # Train/val losses per epoch
├── checkpoint_epoch5.pth    # Periodic checkpoints
├── checkpoint_epoch10.pth
├── model_final.pth          # Final model
└── model_best.pth           # Best validation loss model
```

### 3. Resume Training

Continue from a checkpoint:

```bash
python scripts/train_sam2.py \
    --config configs/experiments/exp1_synthetic.yaml \
    --resume output/exp1_synthetic/checkpoint_epoch10.pth
```

### 4. Plotting Results

**Single experiment:**
```bash
# Display plot
python scripts/plot_training_results.py --experiment output/exp1_synthetic

# Save to file
python scripts/plot_training_results.py \
    --experiment output/exp1_synthetic \
    --save figures/exp1_results.png

# Print summary statistics
python scripts/plot_training_results.py \
    --experiment output/exp1_synthetic \
    --summary
```

**Compare multiple experiments:**
```bash
python scripts/plot_training_results.py \
    --compare output/exp1_synthetic output/exp2_synthetic_real output/exp3_real_threshold \
    --save figures/experiment_comparison.png
```

## Experiment Scenarios

### Experiment 1: Pure Synthetic
**Config:** `configs/experiments/exp1_synthetic.yaml`

**Goal:** Establish baseline performance on clean synthetic data

**Data:**
- Training: 4K synthetic samples with exact ground truth
- Validation: 1K synthetic samples with exact ground truth

**Expected outcome:**
- Very low train/val loss (near-perfect segmentation)
- Baseline for comparing real data experiments
- May not generalize to real data

**Use case:**
- Proof of concept
- Upper bound on performance
- Debug training pipeline

---

### Experiment 2: Synthetic + Real (Mixed)
**Config:** `configs/experiments/exp2_synthetic_real.yaml`

**Goal:** Improve generalization by mixing synthetic and real data

**Data:**
- Training: Synthetic (exact) + real (threshold flags)
- Validation: Real data with threshold flags

**Expected outcome:**
- Better generalization to real data than Exp1
- Moderate loss (harder task than pure synthetic)
- Bridge domain gap between synthetic and real

**Use case:**
- Transfer learning from synthetic to real
- Limited real data availability
- Production model for diverse observations

---

### Experiment 3: Real Data with Threshold Flags
**Config:** `configs/experiments/exp3_real_threshold.yaml`

**Goal:** Train on real observations using automated flagging (MAD, SumThreshold)

**Data:**
- Training: Real MS with automated threshold flags
- Validation: Real MS with automated threshold flags

**Expected outcome:**
- Higher loss than synthetic (noisy labels)
- Model learns to refine threshold flags
- Performance limited by flag quality

**Use case:**
- No manual annotation available
- Large-scale datasets
- Baseline for human-annotated comparison

---

### Experiment 4: Real Data with Human Flags
**Config:** `configs/experiments/exp4_real_human.yaml`

**Goal:** Train on high-quality human-curated flags (gold standard)

**Data:**
- Training: Real MS with expert human annotations
- Validation: Real MS with expert human annotations

**Expected outcome:**
- Lower loss than Exp3 (better labels)
- Best real-world performance ceiling
- Gold standard for production

**Use case:**
- Critical observations requiring high accuracy
- Benchmark for evaluating other experiments
- Production model when annotation budget allows

## Loss Metrics

**DiceCE Loss** (Dice + Cross Entropy):
- **Range:** 0.0 (perfect) to ~1.5 (very poor)
- **Good performance:** < 0.1
- **Acceptable:** 0.1 - 0.3
- **Poor:** > 0.3

**Interpreting results:**
- **Train << Val:** Overfitting (increase regularization, add data)
- **Train ≈ Val:** Good generalization
- **Train >> Val:** Underfitting (increase capacity, train longer)

## Advanced Usage

### Custom Experiments

Create a new config file:

```yaml
experiment:
  name: "my_experiment"
  description: "Description of what you're testing"
  output_dir: "./output/my_experiment"
  save_every_n_epochs: 5

data:
  train_dataset: "./datasets/my_train/exact_masks.npz"
  val_dataset: "./datasets/my_val/exact_masks.npz"

model:
  checkpoint: "large"  # tiny, small, base_plus, large
  freeze_encoders: true

training:
  num_epochs: 20
  batch_size: 4
  learning_rate: 1.0e-5
  weight_decay: 0.0
  device: "cuda"
```

### Transfer Learning

Start from a pretrained model:

```yaml
model:
  checkpoint: "large"
  pretrained_weights: "./output/exp1_synthetic/model_best.pth"
```

### Hyperparameter Tuning

Key parameters to adjust:

1. **Learning rate:**
   - Default: `1e-5`
   - Fine-tuning: `5e-6`
   - From scratch: `1e-4`

2. **Batch size:**
   - Default: 4
   - Larger GPU: 8-16
   - Smaller GPU: 2

3. **Model size:**
   - Fastest: `tiny` (38M params)
   - Balanced: `small` (44M params)
   - Best: `large` (224M params)

## Troubleshooting

**Out of memory:**
```yaml
training:
  batch_size: 2  # Reduce batch size
model:
  checkpoint: "small"  # Use smaller model
```

**Poor validation loss:**
- Check data quality (visualize samples)
- Increase training data
- Try transfer learning from synthetic
- Verify labels are correct

**Training not converging:**
- Reduce learning rate
- Check for data preprocessing issues
- Verify normalization/stretch settings match data generation

## Integration with Existing CLI

The standalone training script complements the existing CLI:

**CLI (`samrfi train`):**
- Quick prototyping
- Simple train/val splits
- Legacy compatibility

**Standalone (`scripts/train_sam2.py`):**
- Full experiment tracking
- Structured outputs
- Reproducibility
- Comparison across experiments

Both methods work with `.npz` and HF Dataset formats.
