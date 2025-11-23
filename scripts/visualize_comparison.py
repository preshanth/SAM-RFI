#!/usr/bin/env python3
"""
Comparison Results Visualization Tool

Generates publication-quality plots and visualizations from comparison results.

Creates:
1. Performance metrics bar chart (Precision, Recall, F1)
2. Detection quality comparison table
3. Side-by-side RFI detection visualizations
4. Summary statistics

Usage:
    python scripts/visualize_comparison.py --results results/zeroshot_cpu/comparison_results.json
    python scripts/visualize_comparison.py --results results/comparison/results.json --output plots/
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(results_path):
    """Load comparison results from JSON file"""
    with open(results_path, 'r') as f:
        return json.load(f)


def create_metrics_barplot(results, output_dir):
    """Create bar chart comparing metrics across methods"""
    methods = list(results.keys())
    metrics = ['precision', 'recall', 'f1_score']
    metric_labels = ['Precision', 'Recall', 'F1-Score']

    # Extract data
    data = {metric: [] for metric in metrics}
    for method in methods:
        for metric in metrics:
            data[metric].append(results[method]['avg_metrics'][metric])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(methods))
    width = 0.25

    # Plot bars
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        offset = width * (i - 1)
        ax.bar(x + offset, data[metric], width, label=label, color=colors[i], alpha=0.8)

    # Formatting
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('RFI Detection Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)

    # Add value labels on bars
    for i, metric in enumerate(metrics):
        offset = width * (i - 1)
        for j, value in enumerate(data[metric]):
            ax.text(j + offset, value + 0.02, f'{value:.3f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / 'metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics plot: {output_path}")
    plt.close()


def create_detection_rate_plot(results, output_dir):
    """Create plot showing detection rates vs false alarm rates"""
    fig, ax = plt.subplots(figsize=(8, 6))

    methods = list(results.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for i, method in enumerate(methods):
        metrics = results[method]['avg_metrics']
        recall = metrics['recall']
        precision = metrics['precision']

        # False alarm rate = 1 - precision
        far = 1 - precision

        ax.scatter(far, recall, s=200, c=colors[i], label=method, alpha=0.8, edgecolors='black')
        ax.text(far + 0.01, recall + 0.01, method, fontsize=10)

    ax.set_xlabel('False Alarm Rate (1 - Precision)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Detection Rate (Recall)', fontsize=12, fontweight='bold')
    ax.set_title('Detection Rate vs False Alarm Rate', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.05, max(0.5, max(1 - results[m]['avg_metrics']['precision'] for m in methods) + 0.1))
    ax.set_ylim(0, 1.05)

    # Add ideal point annotation
    ax.plot([0], [1], 'g*', markersize=15, label='Ideal (100% detection, 0% false alarm)')
    ax.legend(fontsize=10)

    plt.tight_layout()
    output_path = output_dir / 'detection_vs_false_alarm.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved detection rate plot: {output_path}")
    plt.close()


def create_summary_table(results, output_dir):
    """Create summary statistics table"""
    # Prepare data
    rows = []
    for method in results.keys():
        metrics = results[method]['avg_metrics']
        rows.append([
            method,
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1_score']:.4f}",
            f"{metrics['iou']:.4f}"
        ])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, len(rows) * 0.6 + 1))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=['Method', 'Precision', 'Recall', 'F1-Score', 'IoU'],
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')

    plt.title('RFI Detection Performance Summary', fontsize=14, fontweight='bold', pad=20)

    output_path = output_dir / 'summary_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary table: {output_path}")
    plt.close()


def create_example_visualizations(results, output_dir, max_examples=3):
    """Create side-by-side visualizations of example detections

    Note: This requires access to the actual waterfall data and predictions.
    For now, we just create a placeholder showing this would be nice to have.
    """
    # This would require:
    # 1. Loading the actual waterfall images
    # 2. Loading the predicted masks for each method
    # 3. Creating side-by-side visualizations

    # For now, just save a note
    note_path = output_dir / 'example_visualizations_TODO.txt'
    note_path.write_text(
        "Example visualizations would show:\n"
        "1. Original waterfall image\n"
        "2. Ground truth RFI mask\n"
        "3. SAM3 predictions\n"
        "4. CASA tfcrop predictions\n"
        "5. CASA rflag predictions\n"
        "6. CASA combined predictions\n\n"
        "To implement: Load actual data and masks, create matplotlib grid.\n"
    )
    print(f"Note saved: {note_path}")


def generate_text_report(results, output_dir):
    """Generate a detailed text report"""
    report_path = output_dir / 'comparison_report.txt'

    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("RFI DETECTION COMPARISON REPORT\n")
        f.write("="*70 + "\n\n")

        # Overall summary
        f.write("SUMMARY\n")
        f.write("-"*70 + "\n")

        methods = list(results.keys())
        best_f1_method = max(methods, key=lambda m: results[m]['avg_metrics']['f1_score'])
        best_precision_method = max(methods, key=lambda m: results[m]['avg_metrics']['precision'])
        best_recall_method = max(methods, key=lambda m: results[m]['avg_metrics']['recall'])

        f.write(f"Best F1-Score:  {best_f1_method} ({results[best_f1_method]['avg_metrics']['f1_score']:.4f})\n")
        f.write(f"Best Precision: {best_precision_method} ({results[best_precision_method]['avg_metrics']['precision']:.4f})\n")
        f.write(f"Best Recall:    {best_recall_method} ({results[best_recall_method]['avg_metrics']['recall']:.4f})\n")
        f.write("\n")

        # Detailed metrics for each method
        f.write("DETAILED METRICS\n")
        f.write("-"*70 + "\n\n")

        for method in methods:
            f.write(f"{method.upper()}\n")
            metrics = results[method]['avg_metrics']
            f.write(f"  Precision:  {metrics['precision']:.6f}\n")
            f.write(f"  Recall:     {metrics['recall']:.6f}\n")
            f.write(f"  F1-Score:   {metrics['f1_score']:.6f}\n")
            f.write(f"  IoU:        {metrics['iou']:.6f}\n")

            if 'total_flagged' in metrics:
                f.write(f"  Total Flagged: {metrics['total_flagged']:.1f}%\n")

            f.write("\n")

        # Comparative analysis
        f.write("COMPARATIVE ANALYSIS\n")
        f.write("-"*70 + "\n")

        if 'sam3_zeroshot' in results and 'casa_combined' in results:
            sam3_f1 = results['sam3_zeroshot']['avg_metrics']['f1_score']
            casa_f1 = results['casa_combined']['avg_metrics']['f1_score']
            improvement = ((sam3_f1 - casa_f1) / casa_f1) * 100

            f.write(f"SAM3 vs CASA Combined:\n")
            f.write(f"  SAM3 F1:     {sam3_f1:.4f}\n")
            f.write(f"  CASA F1:     {casa_f1:.4f}\n")
            f.write(f"  Improvement: {improvement:+.2f}%\n")
            f.write("\n")

        f.write("="*70 + "\n")

    print(f"Saved text report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize RFI detection comparison results")
    parser.add_argument("--results", required=True, help="Path to comparison_results.json")
    parser.add_argument("--output", help="Output directory for plots (default: same as results)")
    args = parser.parse_args()

    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)

    print(f"Loading results from: {results_path}")
    results = load_results(results_path)

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = results_path.parent / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving visualizations to: {output_dir}\n")

    # Generate visualizations
    print("Generating visualizations...")
    create_metrics_barplot(results, output_dir)
    create_detection_rate_plot(results, output_dir)
    create_summary_table(results, output_dir)
    create_example_visualizations(results, output_dir)
    generate_text_report(results, output_dir)

    print(f"\n{'='*70}")
    print("Visualization complete!")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - metrics_comparison.png")
    print(f"  - detection_vs_false_alarm.png")
    print(f"  - summary_table.png")
    print(f"  - comparison_report.txt")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
