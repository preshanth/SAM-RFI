#!/usr/bin/env python3
"""
Evaluate SAM-RFI on du Toit et al. (2024) HERA and LOFAR datasets.

Usage:
    python scripts/evaluate_dutoit_datasets.py \
        --hera-path /mnt/Data/Data/SAM-RFI/HERA_28-03-2023_all.pkl \
        --hera-aof-path /mnt/Data/Data/SAM-RFI/HERA_AOF_20-07-2023_all.pkl \
        --lofar-path /mnt/Data/Data/SAM-RFI/LOFAR_Full_RFI_dataset.pkl \
        --output-dir ./dutoit_evaluation
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rfi_toolbox.evaluation import evaluate_segmentation
from tqdm import tqdm

from samrfi.inference import RFIPredictor


def load_dutoit_dataset(pkl_path):
    """Load du Toit dataset from pickle."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Format: [train_images, train_masks, test_images, test_masks]
    return {
        "train_images": data[0],
        "train_masks": data[1],
        "test_images": data[2],
        "test_masks": data[3],
    }


def evaluate_model_on_dataset(predictor, images, ground_truth, dataset_name, model_name):
    """Evaluate single model on dataset - one baseline at a time to avoid memory issues."""
    all_metrics = []

    print(f"  Evaluating {model_name} on {dataset_name} ({len(images)} samples)...")

    # Process one baseline at a time to avoid memory overflow
    for idx in tqdm(range(len(images)), desc=f"  {model_name}"):
        img = images[idx]  # Shape: (512, 512, 1 or 2)

        # Handle different formats
        if img.shape[-1] == 2:
            # HERA format: (mag, phase) -> convert to complex
            magnitude = img[..., 0]
            phase = img[..., 1]
            img_complex = magnitude * np.exp(1j * phase)
        else:
            # LOFAR format: single channel magnitude
            img_complex = img[..., 0].astype(np.complex64)

        # Shape: (1, 1, 512, 512) for predict_array
        img_4d = img_complex[np.newaxis, np.newaxis, :, :]

        # Predict on single baseline
        pred = predictor.predict_array(img_4d, patch_size=1024, threshold=None)
        pred = pred[0, 0, :, :]  # Extract (512, 512)

        gt = ground_truth[idx][..., 0]  # Remove channel dim

        # Compute metrics
        metrics = evaluate_segmentation(pred, gt)
        all_metrics.append(metrics)

    # Aggregate
    aggregated = {
        "iou": [m["iou"] for m in all_metrics],
        "precision": [m["precision"] for m in all_metrics],
        "recall": [m["recall"] for m in all_metrics],
        "f1": [m["f1"] for m in all_metrics],
        "dice": [m["dice"] for m in all_metrics],
    }

    return aggregated


def plot_metrics(results, output_dir):
    """Generate comparison plots."""
    output_dir = Path(output_dir)

    datasets = list(results.keys())
    models = ["tiny", "small", "base_plus", "large"]
    metrics = ["iou", "precision", "recall", "f1"]

    colors = {
        "tiny": "tab:blue",
        "small": "tab:orange",
        "base_plus": "tab:green",
        "large": "tab:red",
    }

    for dataset in datasets:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            for model in models:
                if model in results[dataset]:
                    values = results[dataset][model][metric]
                    # mean_val = np.mean(values)
                    # std_val = np.std(values)

                    # Box plot
                    positions = [models.index(model)]
                    bp = ax.boxplot(
                        [values], positions=positions, widths=0.6, patch_artist=True, showmeans=True
                    )
                    bp["boxes"][0].set_facecolor(colors[model])
                    bp["boxes"][0].set_alpha(0.6)

            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models)
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{metric.upper()} Distribution", fontweight="bold")
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"{dataset} - SAM Model Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()

        output_path = output_dir / f"{dataset}_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {output_path}")
        plt.close()


def generate_summary_table(results, output_dir):
    """Generate summary statistics table."""
    output_dir = Path(output_dir)

    datasets = list(results.keys())
    models = ["tiny", "small", "base_plus", "large"]
    metrics = ["iou", "precision", "recall", "f1"]

    table = []
    table.append("=" * 100)
    table.append("SAM-RFI Evaluation on du Toit et al. (2024) Datasets")
    table.append("=" * 100)

    for dataset in datasets:
        table.append(f"\n{dataset.upper()}")
        table.append("-" * 100)
        table.append(
            f"{'Metric':<12} | {'tiny':<18} | {'small':<18} | {'base_plus':<18} | {'large':<18}"
        )
        table.append("-" * 100)

        for metric in metrics:
            row = f"{metric.upper():<12}"
            for model in models:
                if model in results[dataset]:
                    values = results[dataset][model][metric]
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    row += f" | {mean_val:.4f} ± {std_val:.4f}"
                else:
                    row += f" | {'N/A':<18}"
            table.append(row)

    table.append("=" * 100)

    table_text = "\n".join(table)
    print("\n" + table_text)

    # Save to file
    output_path = output_dir / "summary_table.txt"
    with open(output_path, "w") as f:
        f.write(table_text)
    print(f"\n✓ Saved summary table: {output_path}")

    return table_text


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SAM-RFI on du Toit datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--hera-path", required=True, help="HERA dataset pickle")
    parser.add_argument("--hera-aof-path", required=True, help="HERA AOFlagger dataset pickle")
    parser.add_argument("--lofar-path", required=True, help="LOFAR dataset pickle")
    parser.add_argument("--output-dir", default="./dutoit_evaluation", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--use-test-set", action="store_true", help="Use test set (default: train set)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print(f"\n{'='*70}")
    print("Loading du Toit Datasets")
    print(f"{'='*70}")

    print("Loading HERA dataset (3.1GB)...")
    hera = load_dutoit_dataset(args.hera_path)
    print("  ✓ Loaded HERA")

    print("Loading HERA_AOF dataset (3.1GB)...")
    hera_aof = load_dutoit_dataset(args.hera_aof_path)
    print("  ✓ Loaded HERA_AOF")

    print("Loading LOFAR dataset (9.3GB)...")
    lofar = load_dutoit_dataset(args.lofar_path)
    print("  ✓ Loaded LOFAR")

    split = "test" if args.use_test_set else "train"
    print(f"Using {split} set")
    print(f"  HERA: {len(hera[f'{split}_images'])} samples")
    print(f"  HERA_AOF: {len(hera_aof[f'{split}_images'])} samples")
    print(f"  LOFAR: {len(lofar[f'{split}_images'])} samples")

    datasets = {
        "HERA": (hera[f"{split}_images"], hera[f"{split}_masks"]),
        "HERA_AOF": (hera_aof[f"{split}_images"], hera_aof[f"{split}_masks"]),
        "LOFAR": (lofar[f"{split}_images"], lofar[f"{split}_masks"]),
    }

    # Evaluate all models
    models = ["tiny", "small", "base_plus", "large"]
    results = {dataset_name: {} for dataset_name in datasets.keys()}

    print(f"\n{'='*70}")
    print("Evaluating SAM Models")
    print(f"{'='*70}")

    for model_name in models:
        print(f"\n[{model_name.upper()}]")
        model_path = f"polarimetic/sam-rfi/{model_name}"

        try:
            predictor = RFIPredictor(
                model_path=model_path, sam_checkpoint=model_name, device=args.device
            )

            for dataset_name, (images, masks) in datasets.items():
                metrics = evaluate_model_on_dataset(
                    predictor, images, masks, dataset_name, model_name
                )
                results[dataset_name][model_name] = metrics

        except Exception as e:
            print(f"  ✗ Error with {model_name}: {e}")
            continue

    # Save results
    print(f"\n{'='*70}")
    print("Saving Results")
    print(f"{'='*70}")

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        # Convert to serializable format
        json_results = {}
        for dataset, models_data in results.items():
            json_results[dataset] = {}
            for model, metrics in models_data.items():
                json_results[dataset][model] = {
                    k: [float(v) for v in vals] for k, vals in metrics.items()
                }
        json.dump(json_results, f, indent=2)

    print(f"✓ Saved metrics: {results_path}")

    # Generate plots
    plot_metrics(results, output_dir)

    # Generate summary table
    generate_summary_table(results, output_dir)

    print(f"\n{'='*70}")
    print("✓ Evaluation Complete")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
