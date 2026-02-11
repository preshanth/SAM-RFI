#!/usr/bin/env python3
"""
Plot benchmark comparison results from SAM-RFI vs CASA methods

Usage:
    # Plot single benchmark results
    python scripts/plot_benchmark_results.py --results benchmark_results/results.json

    # Compare multiple benchmarks
    python scripts/plot_benchmark_results.py --compare \
        benchmark_v1/results.json \
        benchmark_v2/results.json \
        benchmark_v3/results.json

    # Save plot instead of displaying
    python scripts/plot_benchmark_results.py --results benchmark_results/results.json --save comparison.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_path):
    """Load results from JSON file"""
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path) as f:
        return json.load(f)


def plot_single_benchmark(results_path, save_path=None):
    """Plot comparison for a single benchmark result"""
    results = load_results(results_path)

    # Extract metrics
    sam = results["sam_rfi"]
    tfcrop = results["tfcrop_rflag"]

    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    # 1. Segmentation Metrics (vs Ground Truth)
    seg_metrics = ["iou", "precision", "recall", "f1", "dice"]
    sam_seg = [sam["segmentation"][m] for m in seg_metrics]
    tfcrop_seg = [tfcrop["segmentation"][m] for m in seg_metrics]

    x = np.arange(len(seg_metrics))
    width = 0.35

    ax1.bar(x - width / 2, sam_seg, width, label="SAM-RFI", color="#2E86AB", alpha=0.8)
    ax1.bar(x + width / 2, tfcrop_seg, width, label="tfcrop+rflag", color="#A23B72", alpha=0.8)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("Segmentation Metrics (vs Ground Truth)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [m.upper() if m == "iou" or m == "f1" else m.capitalize() for m in seg_metrics]
    )
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for i, (s, t) in enumerate(zip(sam_seg, tfcrop_seg, strict=False)):
        ax1.text(i - width / 2, s + 0.02, f"{s:.3f}", ha="center", va="bottom", fontsize=8)
        ax1.text(i + width / 2, t + 0.02, f"{t:.3f}", ha="center", va="bottom", fontsize=8)

    # 2. Statistical Metrics
    stat_metrics = ["mean", "median", "std", "mad"]
    sam_stats = [sam["statistics"][m] for m in stat_metrics]
    tfcrop_stats = [tfcrop["statistics"][m] for m in stat_metrics]

    # Normalize to log scale for visibility
    sam_stats_log = np.log10(np.array(sam_stats) + 1e-10)
    tfcrop_stats_log = np.log10(np.array(tfcrop_stats) + 1e-10)

    x2 = np.arange(len(stat_metrics))
    ax2.bar(x2 - width / 2, sam_stats_log, width, label="SAM-RFI", color="#2E86AB", alpha=0.8)
    ax2.bar(
        x2 + width / 2, tfcrop_stats_log, width, label="tfcrop+rflag", color="#A23B72", alpha=0.8
    )
    ax2.set_ylabel("log₁₀(Value)", fontsize=11)
    ax2.set_title("Statistical Metrics (Unflagged Data)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(
        [m.upper() if m == "mad" or m == "std" else m.capitalize() for m in stat_metrics]
    )
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    # 3. Flagging Quality Metrics
    ffi_metrics = ["ffi", "mad_reduction", "std_reduction"]
    sam_ffi = [sam["ffi"][m] for m in ffi_metrics]
    tfcrop_ffi = [tfcrop["ffi"][m] for m in ffi_metrics]

    x3 = np.arange(len(ffi_metrics))
    ax3.bar(x3 - width / 2, sam_ffi, width, label="SAM-RFI", color="#2E86AB", alpha=0.8)
    ax3.bar(x3 + width / 2, tfcrop_ffi, width, label="tfcrop+rflag", color="#A23B72", alpha=0.8)
    ax3.set_ylabel("Score", fontsize=11)
    ax3.set_title("Flagging Fidelity Index", fontsize=12, fontweight="bold")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(["FFI", "MAD Reduction", "STD Reduction"])
    ax3.legend(fontsize=10)
    ax3.set_ylim([-0.5, 1.0])
    ax3.grid(axis="y", alpha=0.3, linestyle="--")
    ax3.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

    # Add value labels
    for i, (s, t) in enumerate(zip(sam_ffi, tfcrop_ffi, strict=False)):
        ax3.text(i - width / 2, s + 0.03, f"{s:.3f}", ha="center", va="bottom", fontsize=8)
        ax3.text(i + width / 2, t + 0.03, f"{t:.3f}", ha="center", va="bottom", fontsize=8)

    # 4. Calcquality Components
    cq_metrics = ["calcquality", "sensitivity", "mean_shift", "std_shift", "overflagging_penalty"]
    sam_cq = [sam["calcquality"][m] for m in cq_metrics]
    tfcrop_cq = [tfcrop["calcquality"][m] for m in cq_metrics]

    x4 = np.arange(len(cq_metrics))
    ax4.bar(x4 - width / 2, sam_cq, width, label="SAM-RFI", color="#2E86AB", alpha=0.8)
    ax4.bar(x4 + width / 2, tfcrop_cq, width, label="tfcrop+rflag", color="#A23B72", alpha=0.8)
    ax4.set_ylabel("Score (lower is better)", fontsize=11)
    ax4.set_title("Calcquality Metric", fontsize=12, fontweight="bold")
    ax4.set_xticks(x4)
    ax4.set_xticklabels(
        ["Calcquality", "Sensitivity", "Mean Shift", "STD Shift", "Overflag"],
        rotation=15,
        ha="right",
    )
    ax4.legend(fontsize=10)
    ax4.grid(axis="y", alpha=0.3, linestyle="--")

    # Overall title with RFI percentage
    rfi_pct = results["ground_truth_rfi_percent"]
    fig.suptitle(
        f"SAM-RFI vs tfcrop+rflag Benchmark Comparison\nGround Truth RFI: {rfi_pct:.1f}%",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def plot_comparison(results_paths, save_path=None):
    """Compare multiple benchmark results"""
    all_results = [load_results(p) for p in results_paths]
    labels = [Path(p).parent.name for p in results_paths]

    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=150)

    colors_sam = ["#2E86AB", "#1F5F7A", "#144A5E"]
    colors_tfcrop = ["#A23B72", "#7A2D56", "#5C2241"]

    # 1. F1 Score Comparison
    f1_sam = [r["sam_rfi"]["segmentation"]["f1"] for r in all_results]
    f1_tfcrop = [r["tfcrop_rflag"]["segmentation"]["f1"] for r in all_results]

    x = np.arange(len(labels))
    width = 0.35

    ax1.bar(x - width / 2, f1_sam, width, label="SAM-RFI", color=colors_sam[0], alpha=0.8)
    ax1.bar(
        x + width / 2, f1_tfcrop, width, label="tfcrop+rflag", color=colors_tfcrop[0], alpha=0.8
    )
    ax1.set_ylabel("F1 Score", fontsize=12)
    ax1.set_title("F1 Score (Segmentation Accuracy)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.legend(fontsize=11)
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # 2. IoU Comparison
    iou_sam = [r["sam_rfi"]["segmentation"]["iou"] for r in all_results]
    iou_tfcrop = [r["tfcrop_rflag"]["segmentation"]["iou"] for r in all_results]

    ax2.bar(x - width / 2, iou_sam, width, label="SAM-RFI", color=colors_sam[0], alpha=0.8)
    ax2.bar(
        x + width / 2, iou_tfcrop, width, label="tfcrop+rflag", color=colors_tfcrop[0], alpha=0.8
    )
    ax2.set_ylabel("IoU Score", fontsize=12)
    ax2.set_title("Intersection over Union", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.legend(fontsize=11)
    ax2.set_ylim([0, 1.0])
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    # 3. FFI Comparison
    ffi_sam = [r["sam_rfi"]["ffi"]["ffi"] for r in all_results]
    ffi_tfcrop = [r["tfcrop_rflag"]["ffi"]["ffi"] for r in all_results]

    ax3.bar(x - width / 2, ffi_sam, width, label="SAM-RFI", color=colors_sam[0], alpha=0.8)
    ax3.bar(
        x + width / 2, ffi_tfcrop, width, label="tfcrop+rflag", color=colors_tfcrop[0], alpha=0.8
    )
    ax3.set_ylabel("FFI Score", fontsize=12)
    ax3.set_title("Flagging Fidelity Index (higher is better)", fontsize=13, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=15, ha="right")
    ax3.legend(fontsize=11)
    ax3.grid(axis="y", alpha=0.3, linestyle="--")

    # 4. Calcquality Comparison
    cq_sam = [r["sam_rfi"]["calcquality"]["calcquality"] for r in all_results]
    cq_tfcrop = [r["tfcrop_rflag"]["calcquality"]["calcquality"] for r in all_results]

    ax4.bar(x - width / 2, cq_sam, width, label="SAM-RFI", color=colors_sam[0], alpha=0.8)
    ax4.bar(
        x + width / 2, cq_tfcrop, width, label="tfcrop+rflag", color=colors_tfcrop[0], alpha=0.8
    )
    ax4.set_ylabel("Calcquality (lower is better)", fontsize=12)
    ax4.set_title("Calcquality Metric", fontsize=13, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=15, ha="right")
    ax4.legend(fontsize=11)
    ax4.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(
        "Multi-Benchmark Comparison: SAM-RFI vs tfcrop+rflag",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison plot saved to: {save_path}")
    else:
        plt.show()


def print_summary(results_path):
    """Print summary statistics for benchmark results"""
    results = load_results(results_path)
    exp_name = Path(results_path).parent.name

    print(f"\n{'='*70}")
    print(f"Benchmark: {exp_name}")
    print(f"{'='*70}")
    print(f"MS: {results['ms_path']}")
    print(f"Config: {results['config']}")
    print(
        f"Dimensions: {results['dimensions']['baselines']} baselines, "
        f"{results['dimensions']['channels']} channels, "
        f"{results['dimensions']['times']} times"
    )
    print(f"Ground Truth RFI: {results['ground_truth_rfi_percent']:.2f}%")

    sam = results["sam_rfi"]
    tfcrop = results["tfcrop_rflag"]

    print(f"\n{'SAM-RFI':<20} {'tfcrop+rflag':<20} {'Metric'}")
    print(f"{'-'*70}")

    # Segmentation metrics
    print("\nSegmentation (vs Ground Truth):")
    for metric in ["iou", "precision", "recall", "f1", "dice"]:
        s = sam["segmentation"][metric]
        t = tfcrop["segmentation"][metric]
        winner = "✓" if s > t else " "
        print(
            f"{s:>6.4f} {winner:<2}         {t:>6.4f}             {metric.upper() if metric in ['iou', 'f1'] else metric.capitalize()}"
        )

    # FFI metrics
    print("\nFlagging Fidelity:")
    for metric in ["ffi", "mad_reduction", "std_reduction"]:
        s = sam["ffi"][metric]
        t = tfcrop["ffi"][metric]
        winner = "✓" if s > t else " "
        label = "FFI" if metric == "ffi" else metric.replace("_", " ").title()
        print(f"{s:>6.4f} {winner:<2}         {t:>6.4f}             {label}")

    # Calcquality (lower is better)
    print("\nCalcquality (lower is better):")
    s_cq = sam["calcquality"]["calcquality"]
    t_cq = tfcrop["calcquality"]["calcquality"]
    winner = "✓" if s_cq < t_cq else " "
    print(f"{s_cq:>6.4f} {winner:<2}         {t_cq:>6.4f}             Calcquality")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark comparison results")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results", help="Plot single benchmark results (results.json)")
    group.add_argument("--compare", nargs="+", help="Compare multiple benchmark results")

    parser.add_argument("--save", help="Save plot to file instead of displaying")
    parser.add_argument("--summary", action="store_true", help="Print summary statistics")

    args = parser.parse_args()

    if args.results:
        if args.summary:
            print_summary(args.results)
        plot_single_benchmark(args.results, save_path=args.save)

    elif args.compare:
        if args.summary:
            for results_path in args.compare:
                print_summary(results_path)
        plot_comparison(args.compare, save_path=args.save)


if __name__ == "__main__":
    main()
