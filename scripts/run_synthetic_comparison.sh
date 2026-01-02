#!/bin/bash
#
# Complete synthetic RFI comparison pipeline:
# 1. Inject synthetic RFI (30-40%) into template MS
# 2. Run TFCROP (1-pass, 2-pass)
# 3. Run RFLAG (1-pass)
# 4. Run SAM-RFI (all 4 models from HuggingFace)
# 5. Compare all methods vs ground truth
# 6. Generate metrics and plots
#
# Usage:
#   ./scripts/run_synthetic_comparison.sh

set -e  # Exit on error

# Configuration
TEMPLATE_MS="/mnt/Data/Data/SAM-RFI/flagging_comparison/synthetic_data/template_1024x1024.ms"
CONFIG="configs/validation.yaml"
OUTPUT_BASE="/mnt/Data/Data/SAM-RFI/flagging_comparison"
INJECTION_DIR="${OUTPUT_BASE}/injection_results"
COMPARISON_DIR="${OUTPUT_BASE}/comparison_results"

echo "========================================================================"
echo "Synthetic RFI Comparison Pipeline"
echo "========================================================================"
echo "Template MS: ${TEMPLATE_MS}"
echo "Output: ${COMPARISON_DIR}"
echo ""

# Step 1: Inject synthetic RFI
echo "========================================================================"
echo "[1/6] Injecting Synthetic RFI (30-40% target)"
echo "========================================================================"

python scripts/inject_synthetic_rfi.py \
    --input-ms "${TEMPLATE_MS}" \
    --config "${CONFIG}" \
    --output-dir "${INJECTION_DIR}"

echo ""
echo "✓ Injection complete"
echo ""

# Step 2: Run TFCROP 1-pass comparison
echo "========================================================================"
echo "[2/6] TFCROP 1-Pass Comparison"
echo "========================================================================"

mkdir -p "${COMPARISON_DIR}/tfcrop_1pass"

for MODEL in tiny small base_plus large; do
    echo ""
    echo "--- Running with SAM model: ${MODEL} ---"

    python scripts/compare_tfcrop_rflag_sam.py \
        --injection-dir "${INJECTION_DIR}" \
        --sam-model "polarimetic/sam-rfi/${MODEL}" \
        --sam-checkpoint "${MODEL}" \
        --config "${CONFIG}" \
        --tfcrop-passes 1 \
        --output-dir "${COMPARISON_DIR}/tfcrop_1pass/${MODEL}" \
        --device cuda
done

echo ""
echo "✓ TFCROP 1-pass complete"
echo ""

# Step 3: Run TFCROP 2-pass comparison
echo "========================================================================"
echo "[3/6] TFCROP 2-Pass Comparison"
echo "========================================================================"

mkdir -p "${COMPARISON_DIR}/tfcrop_2pass"

for MODEL in tiny small base_plus large; do
    echo ""
    echo "--- Running with SAM model: ${MODEL} ---"

    python scripts/compare_tfcrop_rflag_sam.py \
        --injection-dir "${INJECTION_DIR}" \
        --sam-model "polarimetic/sam-rfi/${MODEL}" \
        --sam-checkpoint "${MODEL}" \
        --config "${CONFIG}" \
        --tfcrop-passes 2 \
        --output-dir "${COMPARISON_DIR}/tfcrop_2pass/${MODEL}" \
        --device cuda
done

echo ""
echo "✓ TFCROP 2-pass complete"
echo ""

# Step 4: Aggregate results
echo "========================================================================"
echo "[4/6] Aggregating Results"
echo "========================================================================"

# Combine results from all runs
python - <<'PYEOF'
import json
from pathlib import Path
import numpy as np

base_dir = Path("/mnt/Data/Data/SAM-RFI/flagging_comparison/comparison_results")
output_dir = base_dir / "aggregate"
output_dir.mkdir(exist_ok=True)

models = ['tiny', 'small', 'base_plus', 'large']
tfcrop_variants = ['tfcrop_1pass', 'tfcrop_2pass']

aggregate = {
    'models': models,
    'tfcrop_variants': tfcrop_variants,
    'results': {}
}

for variant in tfcrop_variants:
    aggregate['results'][variant] = {}

    for model in models:
        results_file = base_dir / variant / model / 'results.json'

        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
                aggregate['results'][variant][model] = data
            print(f"✓ Loaded: {variant}/{model}")
        else:
            print(f"✗ Missing: {variant}/{model}")

# Save aggregate
output_file = output_dir / 'all_results.json'
with open(output_file, 'w') as f:
    json.dump(aggregate, f, indent=2)

print(f"\n✓ Saved aggregate results: {output_file}")

# Print quick summary
print("\n" + "="*70)
print("QUICK SUMMARY (Mean F1 Scores)")
print("="*70)
print(f"{'Variant':<20} | {'tiny':<10} | {'small':<10} | {'base_plus':<10} | {'large':<10}")
print("-"*70)

for variant in tfcrop_variants:
    row = f"{variant:<20}"
    for model in models:
        if model in aggregate['results'][variant]:
            try:
                sam_f1 = aggregate['results'][variant][model]['summary']['SAM-RFI']['seg_f1']
                mean_f1 = sam_f1[0]
                row += f" | {mean_f1:.4f}"
            except:
                row += f" | {'N/A':<10}"
        else:
            row += f" | {'N/A':<10}"
    print(row)

print("="*70)
PYEOF

echo ""
echo "✓ Aggregation complete"
echo ""

# Step 5: Generate summary plots
echo "========================================================================"
echo "[5/6] Generating Summary Plots"
echo "========================================================================"

python - <<'PYEOF'
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

base_dir = Path("/mnt/Data/Data/SAM-RFI/flagging_comparison/comparison_results")
aggregate_dir = base_dir / "aggregate"

# Load aggregate results
with open(aggregate_dir / 'all_results.json') as f:
    data = json.load(f)

models = data['models']
variants = data['tfcrop_variants']
metrics = ['iou', 'precision', 'recall', 'f1']

# Plot: SAM model comparison across variants
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

colors = {'tiny': 'tab:blue', 'small': 'tab:orange', 'base_plus': 'tab:green', 'large': 'tab:red'}

for idx, metric in enumerate(metrics):
    ax = axes[idx]

    x = np.arange(len(variants))
    width = 0.2

    for i, model in enumerate(models):
        values = []
        for variant in variants:
            try:
                val = data['results'][variant][model]['summary']['SAM-RFI'][f'seg_{metric}'][0]
                values.append(val)
            except:
                values.append(0)

        ax.bar(x + i*width, values, width, label=model, color=colors[model], alpha=0.8)

    ax.set_xlabel('TFCROP Configuration')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'SAM {metric.upper()} by Model Size', fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(['1-pass', '2-pass'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('SAM-RFI Model Comparison: TFCROP 1-pass vs 2-pass', fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = aggregate_dir / 'sam_model_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# Plot: Method comparison (best SAM vs CASA methods)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]

    # Get best SAM (large model, 2-pass)
    variant = 'tfcrop_2pass'
    methods = ['tfcrop', 'rflag', 'SAM-RFI']
    method_colors = {'tfcrop': 'tab:blue', 'rflag': 'tab:orange', 'SAM-RFI': 'tab:green'}

    values = []
    errors = []

    for method in methods:
        try:
            mean_val = data['results'][variant]['large']['summary'][method][f'seg_{metric}'][0]
            std_val = data['results'][variant]['large']['summary'][method][f'seg_{metric}'][1]
            values.append(mean_val)
            errors.append(std_val)
        except:
            values.append(0)
            errors.append(0)

    x = np.arange(len(methods))
    bars = ax.bar(x, values, yerr=errors, capsize=5,
                  color=[method_colors[m] for m in methods], alpha=0.8)

    ax.set_xlabel('Method')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'{metric.upper()} Comparison (TFCROP 2-pass)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['TFCROP\n(2-pass)', 'RFLAG', 'SAM-RFI\n(large)'])
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Method Comparison: SAM-RFI vs CASA Flaggers', fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = aggregate_dir / 'method_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

print("\n✓ All plots generated")
PYEOF

echo ""
echo "✓ Plotting complete"
echo ""

# Step 6: Final summary
echo "========================================================================"
echo "[6/6] Final Summary"
echo "========================================================================"

echo ""
echo "All results saved to: ${COMPARISON_DIR}"
echo ""
echo "Individual results:"
echo "  - tfcrop_1pass/{tiny,small,base_plus,large}/"
echo "  - tfcrop_2pass/{tiny,small,base_plus,large}/"
echo ""
echo "Aggregate results:"
echo "  - aggregate/all_results.json"
echo "  - aggregate/sam_model_comparison.png"
echo "  - aggregate/method_comparison.png"
echo ""
echo "========================================================================"
echo "✓ Pipeline Complete"
echo "========================================================================"
