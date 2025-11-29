#!/usr/bin/env python3
"""
Comprehensive Flagging Method Validation

Compares SAM3, CASA (tfcrop, rflag), and AOFlagger on simulated MS with ground truth.

Stage 1: Simulated MS (with ground truth)
  - Metrics: Precision, Recall, F1, IoU
  - Methods: SAM3, tfcrop, rflag, combined, AOFlagger

Stage 2: Real MS (without ground truth)
  - Metrics: calcquality, image RMS, dynamic range

Usage:
    # Stage 1: Simulated MS with ground truth
    python scripts/validate_flagging_methods.py \\
        --mode simulated \\
        --ms sim_data.ms \\
        --ground-truth ground_truth.npy \\
        --sam3-model output/sam3_h100/model_best.pth \\
        --output results/validation/

    # Stage 2: Real MS (proxy metrics)
    python scripts/validate_flagging_methods.py \\
        --mode real \\
        --ms real_data.ms \\
        --sam3-model output/sam3_h100/model_best.pth \\
        --output results/real_validation/
"""

import argparse
import sys
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# CASA imports
try:
    from casatasks import flagdata, tclean, imstat
    from casatools import table
    CASA_AVAILABLE = True
except ImportError:
    print("WARNING: CASA not available. Some methods will be skipped.")
    CASA_AVAILABLE = False

# SAM-RFI imports
try:
    from samrfi.inference import RFIPredictor
    SAMRFI_AVAILABLE = True
except ImportError:
    print("ERROR: SAM-RFI not available. Install with: pip install -e .")
    sys.exit(1)


def calcquality(data, flags, overflag_threshold=80.0, overflag_penalty_start=70.0):
    """
    Calculate quality metric for flagging (lower is better).

    From legacy SAM-RFI utilities.py with configurable overflagging penalty.

    Measures:
    1. Gaussian residuals: leftover data should be ~3σ (noise-like)
    2. Mean separation: flagged > unflagged (flagged RFI, not signal)
    3. Std separation: flagged != unflagged (RFI is variable)
    4. Overflagging penalty: configurable threshold

    Args:
        data: Waterfall data (channels, time)
        flags: Boolean flags (True = flagged)
        overflag_threshold: Percentage above which strong penalty applies
        overflag_penalty_start: Percentage where penalty starts

    Returns:
        Quality score (lower is better)
    """
    def printstats(arr):
        if len(arr) == 0:
            return 0, 0, 1
        return np.max(arr), np.mean(arr), np.std(arr)

    leftover = []
    flagged = []

    for chan in range(data.shape[0]):
        for tm in range(data.shape[1]):
            val = np.abs(data[chan, tm])
            if not flags[chan, tm]:
                leftover.append(val)
            else:
                flagged.append(val)

    # Statistics
    dmax, dmean, dstd = printstats(np.abs(data))
    rmax, rmean, rstd = printstats(leftover)
    fmax, fmean, fstd = printstats(flagged)

    maxdev = (rmax - rmean) / rstd if rstd > 0 else 0
    fdiff = fmean - rmean
    sdiff = fstd - rstd

    # Component scores
    aa = np.abs(np.abs(maxdev) - 3.0)  # Gaussian residuals
    bb = 1.0 / max(0.01, (np.abs(fdiff) - rstd) / rstd)  # Mean separation
    cc = 1.0 / max(0.01, np.abs(sdiff) / rstd)  # Std separation

    # Overflagging penalty (configurable)
    dd = 0.0
    pflag = (len(flagged) / (data.size)) * 100.0
    if pflag > overflag_penalty_start:
        dd = (pflag - overflag_penalty_start) / 10.0

    res = np.sqrt(aa**2 + bb**2 + cc**2 + dd**2)

    # Penalty for flagging clean data (negative fdiff)
    if fdiff < 0.0:
        res = res + res + 10.0

    return res


class FlaggingValidator:
    """Comprehensive flagging method validator"""

    def __init__(self, ms_path, output_dir, sam3_model_path=None, mode='simulated',
                 aoflagger_strategy='jvla-default'):
        """
        Initialize validator

        Args:
            ms_path: Path to measurement set
            output_dir: Output directory for results
            sam3_model_path: Path to trained SAM3 model
            mode: 'simulated' (with ground truth) or 'real' (proxy metrics)
            aoflagger_strategy: AOFlagger strategy name (default: jvla-default)
        """
        self.ms_path = Path(ms_path)
        self.output_dir = Path(output_dir)
        self.sam3_model_path = Path(sam3_model_path) if sam3_model_path else None
        self.mode = mode
        self.aoflagger_strategy = aoflagger_strategy

        if not self.ms_path.exists():
            raise FileNotFoundError(f"MS not found: {self.ms_path}")

        if sam3_model_path and not self.sam3_model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.sam3_model_path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results = {}
        self.ground_truth = None

        print("=" * 70)
        print("FLAGGING METHOD VALIDATOR")
        print("=" * 70)
        print(f"Mode:        {self.mode}")
        print(f"Input MS:    {self.ms_path}")
        print(f"SAM3 Model:  {self.sam3_model_path}")
        print(f"Output Dir:  {self.output_dir}")
        print("=" * 70)

    def load_ground_truth(self, ground_truth_path):
        """Load ground truth flags from .npy file"""
        gt_path = Path(ground_truth_path)
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth not found: {gt_path}")

        print(f"\nLoading ground truth: {gt_path}")
        self.ground_truth = np.load(gt_path)
        print(f"  Shape: {self.ground_truth.shape}")
        print(f"  RFI percentage: {100.0 * np.sum(self.ground_truth) / self.ground_truth.size:.2f}%")

    def create_ms_copy(self, method_name):
        """Create MS copy for a flagging method"""
        ms_copy = self.output_dir / f"ms_{method_name}"

        if ms_copy.exists():
            shutil.rmtree(ms_copy)

        shutil.copytree(self.ms_path, ms_copy)
        return ms_copy

    def get_flags_from_ms(self, ms_path):
        """Extract flags from MS"""
        tb = table()
        tb.open(str(ms_path))
        flags = tb.getcol('FLAG')
        tb.close()
        return flags

    def apply_sam3_flagging(self, ms_copy):
        """Apply SAM3 flagging"""
        print("\n[SAM3] Running SAM3 flagging...")

        if not self.sam3_model_path:
            print("  Skipped: No model path provided")
            return None

        predictor = RFIPredictor(
            model_path=str(self.sam3_model_path),
            sam_checkpoint='unified',  # SAM3 unified model
            device='cuda',
            batch_size=8
        )

        flags = predictor.predict_ms(
            ms_path=str(ms_copy),
            num_antennas=None,
            patch_size=1024,
            stretch='SQRT',
            apply_existing_flags=False,
            save_flags=True
        )

        return self.get_flags_from_ms(ms_copy)

    def apply_casa_tfcrop(self, ms_copy):
        """Apply CASA tfcrop"""
        print("\n[CASA tfcrop] Running tfcrop...")

        if not CASA_AVAILABLE:
            print("  Skipped: CASA not available")
            return None

        # Cross-hands
        flagdata(
            vis=str(ms_copy),
            mode='tfcrop',
            datacolumn='data',
            timecutoff=4.0,
            freqcutoff=3.0,
            maxnpieces=5,
            action='apply',
            correlation='ABS_XY,ABS_YX'
        )

        # Parallel-hands
        flagdata(
            vis=str(ms_copy),
            mode='tfcrop',
            datacolumn='data',
            timecutoff=3.0,
            freqcutoff=3.0,
            maxnpieces=2,
            action='apply',
            correlation='ABS_XX,ABS_YY'
        )

        # Extend
        flagdata(vis=str(ms_copy), mode='extend')

        return self.get_flags_from_ms(ms_copy)

    def apply_casa_rflag(self, ms_copy):
        """Apply CASA rflag"""
        print("\n[CASA rflag] Running rflag...")

        if not CASA_AVAILABLE:
            print("  Skipped: CASA not available")
            return None

        flagdata(
            vis=str(ms_copy),
            mode='rflag',
            datacolumn='data',
            timedevscale=4.0,
            freqdevscale=3.0,
            action='apply'
        )

        return self.get_flags_from_ms(ms_copy)

    def apply_aoflagger(self, ms_copy, strategy='jvla-default'):
        """
        Apply AOFlagger using CLI

        Args:
            ms_copy: Path to MS copy
            strategy: Strategy name (e.g., 'jvla-default', 'generic-default')
        """
        print("\n[AOFlagger] Running AOFlagger...")

        # Check if aoflagger is available
        try:
            import subprocess
            result = subprocess.run(['which', 'aoflagger'], capture_output=True, text=True)
            if result.returncode != 0:
                print("  Skipped: aoflagger command not found")
                return None
        except Exception as e:
            print(f"  Skipped: {e}")
            return None

        # Find strategy file
        strategy_path = f"/usr/share/aoflagger/strategies/{strategy}.lua"
        if not Path(strategy_path).exists():
            print(f"  WARNING: Strategy not found: {strategy_path}")
            print(f"  Using AOFlagger default strategy")
            strategy_path = None

        # Build command
        cmd = ['aoflagger']
        if strategy_path:
            cmd.extend(['-strategy', strategy_path])
        cmd.append(str(ms_copy))

        print(f"  Command: {' '.join(cmd)}")

        # Run AOFlagger
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                print(f"  ERROR: AOFlagger failed with return code {result.returncode}")
                print(f"  stderr: {result.stderr}")
                return None

            print(f"  AOFlagger completed successfully")

            # Read flags from MS
            return self.get_flags_from_ms(ms_copy)

        except subprocess.TimeoutExpired:
            print("  ERROR: AOFlagger timed out (>10 minutes)")
            return None
        except Exception as e:
            print(f"  ERROR: {e}")
            return None

    def compute_metrics_with_ground_truth(self, predicted_flags, method_name):
        """Compute precision, recall, F1, IoU against ground truth"""
        if self.ground_truth is None:
            raise ValueError("Ground truth not loaded")

        # Ensure shapes match
        if predicted_flags.shape != self.ground_truth.shape:
            print(f"  WARNING: Shape mismatch. Predicted: {predicted_flags.shape}, GT: {self.ground_truth.shape}")
            return None

        # Flatten for metrics
        pred_flat = predicted_flags.flatten()
        gt_flat = self.ground_truth.flatten()

        # Compute confusion matrix
        TP = np.sum((pred_flat == 1) & (gt_flat == 1))
        TN = np.sum((pred_flat == 0) & (gt_flat == 0))
        FP = np.sum((pred_flat == 1) & (gt_flat == 0))
        FN = np.sum((pred_flat == 0) & (gt_flat == 1))

        # Compute metrics
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

        flag_pct = 100.0 * np.sum(pred_flat) / pred_flat.size

        metrics = {
            'TP': int(TP),
            'TN': int(TN),
            'FP': int(FP),
            'FN': int(FN),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'iou': float(iou),
            'fpr': float(fpr),
            'flag_percentage': float(flag_pct)
        }

        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  IoU:       {iou:.4f}")
        print(f"  Flagged:   {flag_pct:.2f}%")

        return metrics

    def compute_calcquality_metrics(self, ms_copy, flags, method_name):
        """Compute calcquality score (for real data)"""
        # Load data from MS
        # This is a placeholder - need to implement MS data loading
        print(f"  calcquality score: NOT IMPLEMENTED YET")
        return None

    def run_validation(self):
        """Run complete validation pipeline"""

        methods = [
            ('sam3', self.apply_sam3_flagging, {}),
            ('casa_tfcrop', self.apply_casa_tfcrop, {}),
            ('casa_rflag', self.apply_casa_rflag, {}),
            ('aoflagger', self.apply_aoflagger, {'strategy': self.aoflagger_strategy}),
        ]

        for method_name, apply_func, kwargs in methods:
            print(f"\n{'='*70}")
            print(f"Method: {method_name.upper()}")
            print(f"{'='*70}")

            # Create MS copy
            ms_copy = self.create_ms_copy(method_name)

            # Apply flagging
            flags = apply_func(ms_copy, **kwargs)

            if flags is None:
                print(f"  Skipped: {method_name}")
                continue

            # Compute metrics
            if self.mode == 'simulated' and self.ground_truth is not None:
                metrics = self.compute_metrics_with_ground_truth(flags, method_name)
            else:
                metrics = self.compute_calcquality_metrics(ms_copy, flags, method_name)

            self.results[method_name] = {
                'metrics': metrics,
                'flags': flags,
                'ms_copy': str(ms_copy)
            }

        # Save results
        self.save_results()

        # Create plots
        if self.mode == 'simulated' and self.ground_truth is not None:
            self.create_comparison_plots()

    def save_results(self):
        """Save results to JSON"""
        results_path = self.output_dir / 'validation_results.json'

        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for method, data in self.results.items():
            json_results[method] = {
                'metrics': data['metrics'],
                'ms_copy': data['ms_copy']
            }

        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"\n✓ Results saved: {results_path}")

    def create_comparison_plots(self):
        """Create publication-ready comparison plots"""
        print("\n" + "=" * 70)
        print("GENERATING PLOTS")
        print("=" * 70)

        # Filter methods with valid metrics
        valid_methods = {k: v for k, v in self.results.items() if v['metrics'] is not None}

        if not valid_methods:
            print("No valid results to plot")
            return

        # Plot 1: Metrics comparison (Precision, Recall, F1)
        self._plot_metrics_comparison(valid_methods)

        # Plot 2: ROC-style (Detection Rate vs False Alarm Rate)
        self._plot_detection_vs_fpr(valid_methods)

        # Plot 3: Confusion matrix heatmap
        self._plot_confusion_matrices(valid_methods)

        print("\n✓ Plots created")

    def _plot_metrics_comparison(self, methods_dict):
        """Bar chart: Precision, Recall, F1"""
        fig, ax = plt.subplots(figsize=(12, 6))

        methods = list(methods_dict.keys())
        metrics_names = ['precision', 'recall', 'f1_score']
        metric_labels = ['Precision', 'Recall', 'F1-Score']

        x = np.arange(len(methods))
        width = 0.25
        colors = ['#3498db', '#e74c3c', '#2ecc71']

        for i, (metric, label) in enumerate(zip(metrics_names, metric_labels)):
            values = [methods_dict[m]['metrics'][metric] for m in methods]
            offset = width * (i - 1)
            bars = ax.bar(x + offset, values, width, label=label, color=colors[i], alpha=0.8)

            # Add value labels
            for j, (bar, val) in enumerate(zip(bars, values)):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('RFI Flagging Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').upper() for m in methods], fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        plot_path = self.output_dir / 'plot_metrics_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {plot_path.name}")

    def _plot_detection_vs_fpr(self, methods_dict):
        """Scatter plot: Recall vs FPR"""
        fig, ax = plt.subplots(figsize=(10, 8))

        methods = list(methods_dict.keys())
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

        for i, method in enumerate(methods):
            metrics = methods_dict[method]['metrics']
            recall = metrics['recall']
            fpr = metrics['fpr']

            ax.scatter(fpr, recall, s=300, c=colors[i % len(colors)],
                      label=method.replace('_', ' ').upper(),
                      alpha=0.8, edgecolors='black', linewidth=2)

            ax.text(fpr + 0.01, recall - 0.02, method.replace('_', ' ').upper(),
                   fontsize=10, fontweight='bold')

        # Ideal point
        ax.plot([0], [1], 'g*', markersize=20, label='Ideal', zorder=10)

        ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        ax.set_ylabel('Recall (Detection Rate)', fontsize=13, fontweight='bold')
        ax.set_title('Detection Rate vs False Alarm Rate', fontsize=15, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=11, loc='lower right')
        ax.set_xlim(-0.02, max(0.3, max(methods_dict[m]['metrics']['fpr'] for m in methods) + 0.05))
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        plot_path = self.output_dir / 'plot_detection_vs_fpr.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {plot_path.name}")

    def _plot_confusion_matrices(self, methods_dict):
        """Heatmap grid: Confusion matrices"""
        n_methods = len(methods_dict)
        ncols = min(3, n_methods)
        nrows = (n_methods + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        if n_methods == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, (method, data) in enumerate(methods_dict.items()):
            ax = axes[i]
            metrics = data['metrics']

            # Confusion matrix
            cm = np.array([[metrics['TN'], metrics['FP']],
                          [metrics['FN'], metrics['TP']]])

            # Normalize
            cm_norm = cm.astype('float') / cm.sum()

            # Plot
            im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

            # Labels
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Clean', 'RFI'], fontsize=11)
            ax.set_yticklabels(['Clean', 'RFI'], fontsize=11)
            ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
            ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
            ax.set_title(method.replace('_', ' ').upper(), fontsize=13, fontweight='bold')

            # Annotate cells
            for r in range(2):
                for c in range(2):
                    text = ax.text(c, r, f'{cm[r, c]}\n({cm_norm[r, c]:.2%})',
                                 ha='center', va='center', fontsize=11,
                                 color='white' if cm_norm[r, c] > 0.5 else 'black')

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plot_path = self.output_dir / 'plot_confusion_matrices.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {plot_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate RFI flagging methods',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--mode', choices=['simulated', 'real'], required=True,
                       help='Validation mode: simulated (with ground truth) or real (proxy metrics)')
    parser.add_argument('--ms', required=True, help='Path to measurement set')
    parser.add_argument('--ground-truth', help='Path to ground truth .npy file (required for simulated mode)')
    parser.add_argument('--sam3-model', help='Path to trained SAM3 model')
    parser.add_argument('--output', required=True, help='Output directory for results')
    parser.add_argument('--overflag-threshold', type=float, default=80.0,
                       help='Overflagging penalty threshold for calcquality (default: 80.0%%)')
    parser.add_argument('--aoflagger-strategy', default='jvla-default',
                       help='AOFlagger strategy file name (default: jvla-default). '
                            'Available: jvla-default, generic-default, atca-default, etc.')

    args = parser.parse_args()

    # Validate arguments
    if args.mode == 'simulated' and not args.ground_truth:
        parser.error("--ground-truth is required for simulated mode")

    # Create validator
    validator = FlaggingValidator(
        ms_path=args.ms,
        output_dir=args.output,
        sam3_model_path=args.sam3_model,
        mode=args.mode,
        aoflagger_strategy=args.aoflagger_strategy
    )

    # Load ground truth if provided
    if args.ground_truth:
        validator.load_ground_truth(args.ground_truth)

    # Run validation
    validator.run_validation()

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {args.output}")
    print("=" * 70)


if __name__ == '__main__':
    main()
