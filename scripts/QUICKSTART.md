# Quick Start: SAM-RFI Training Pipeline

## One-Command Test

```bash
# Generate test data and train (assumes you have configs set up)
samrfi generate-data --source synthetic --config configs/synthetic_val_100.yaml --output datasets/test && \
python scripts/train_sam2.py --config configs/experiments/exp1_synthetic.yaml
```

## Complete Workflow (4 Experiments)

### Step 1: Data Generation

```bash
# Synthetic training data (4K samples)
samrfi generate-data --source synthetic \
    --config configs/synthetic_train_4k.yaml \
    --output datasets/synthetic_train_4k

# Synthetic validation data (1K samples)
samrfi generate-data --source synthetic \
    --config configs/synthetic_val_1k.yaml \
    --output datasets/synthetic_val_1k

# Real data with threshold flags (TODO: create config)
# samrfi generate-data --source ms \
#     --config configs/real_ms_data.yaml \
#     --output datasets/real_train

# Real data with human flags (TODO: manually create)
# Requires manual annotation workflow
```

### Step 2: Run Experiments

```bash
# Exp 1: Pure synthetic (baseline)
python scripts/train_sam2.py --config configs/experiments/exp1_synthetic.yaml

# Exp 2: Mixed synthetic + real (generalization)
# python scripts/train_sam2.py --config configs/experiments/exp2_synthetic_real.yaml

# Exp 3: Real threshold flags (automated labels)
# python scripts/train_sam2.py --config configs/experiments/exp3_real_threshold.yaml

# Exp 4: Real human flags (gold standard)
# python scripts/train_sam2.py --config configs/experiments/exp4_real_human.yaml
```

### Step 3: Compare Results

```bash
# Plot single experiment
python scripts/plot_training_results.py --experiment output/exp1_synthetic --summary

# Compare all experiments
python scripts/plot_training_results.py \
    --compare output/exp1_synthetic output/exp2_synthetic_real \
              output/exp3_real_threshold output/exp4_real_human \
    --save figures/all_experiments.png
```

## Expected Timeline

| Experiment | Data Gen | Training (20 epochs) | Total |
|------------|----------|---------------------|-------|
| Exp1 (Synthetic) | ~30 min | ~4 hours (A100) | ~4.5 hrs |
| Exp2 (Mixed) | +30 min (real) | ~6 hours | ~6.5 hrs |
| Exp3 (Threshold) | ~45 min (MS load) | ~6 hours | ~7 hrs |
| Exp4 (Human) | Manual | ~6 hours | TBD |

**Total for all 4:** ~24-30 hours compute time

## Outputs to Expect

After training, you'll have:

```
output/
├── exp1_synthetic/
│   ├── losses.npz              # Train/val losses (plot this!)
│   ├── model_best.pth          # Best model (use for inference)
│   ├── training_log.txt        # Full log
│   └── config.yaml             # Experiment config
│
├── exp2_synthetic_real/
│   └── ...
│
├── exp3_real_threshold/
│   └── ...
│
└── exp4_real_human/
    └── ...
```

## Success Criteria

### Experiment 1 (Synthetic Baseline)
- ✓ Train loss < 0.05 (near-perfect on synthetic)
- ✓ Val loss < 0.10 (good generalization within synthetic domain)
- ✓ Train/val gap < 0.05 (no overfitting)

### Experiment 2 (Generalization Test)
- ✓ Val loss on real data < 0.30 (acceptable real-world performance)
- ✓ Better than Exp3 (proves synthetic pre-training helps)

### Experiment 3 (Automated Labels)
- ✓ Val loss < Exp2 (model refines threshold flags)
- ✓ Comparison baseline for Exp4

### Experiment 4 (Gold Standard)
- ✓ Lowest val loss on real data (best real-world model)
- ✓ Production-ready performance

## Troubleshooting

**Data generation too slow?**
- Reduce `num_samples` in config
- Increase `num_workers` in config

**Training OOM (out of memory)?**
- Reduce `batch_size` to 2 or 1
- Use smaller model (`tiny` or `small`)
- Reduce `patch_size` in data generation

**Poor validation loss?**
- Check data quality: visualize samples
- Verify normalization/stretch settings
- Try transfer learning from synthetic

**Can't create real data configs yet?**
- Start with Exp1 only (pure synthetic)
- Validate pipeline works end-to-end
- Prepare real data later
