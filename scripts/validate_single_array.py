#!/usr/bin/env python3
"""
Single array validation for SAM-RFI.

Generates a single 1024x1024 synthetic array with ~40% RFI coverage,
runs it through the trained SAM model, and validates all processing steps.

Usage:
    python scripts/validate_single_array.py \
        --model /path/to/model.pth \
        --config configs/validation.yaml \
        --output ./single_array_validation
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from samrfi.config import ConfigLoader
from samrfi.data_generation import SyntheticDataGenerator
from samrfi.evaluation import evaluate_segmentation, print_statistics_comparison
from samrfi.inference import RFIPredictor


def generate_single_array(config):
    """Generate single 1024x1024 complex array with RFI."""
    print("\n" + "=" * 60)
    print("STEP 1: Generate Synthetic RFI Array")
    print("=" * 60)

    generator = SyntheticDataGenerator(config)
    synth_config = config.synthetic

    # Build gen_kwargs from config
    rfi_config = generator._parse_rfi_config(synth_config)
    gen_kwargs = {
        "num_channels": synth_config.num_channels,
        "num_times": synth_config.num_times,
        "noise_level": synth_config.noise_mjy,
        "rfi_power_min": synth_config.rfi_power_min,
        "rfi_power_max": synth_config.rfi_power_max,
        "rfi_config": rfi_config,
        "enable_bandpass": synth_config.get("enable_bandpass_rolloff", False),
        "bandpass_order": synth_config.get("bandpass_polynomial_order", 8),
        "num_polarizations": synth_config.num_polarizations,
        "pol_corr": synth_config.get("polarization_correlation", 0.8),
        "synth_config": synth_config,
    }

    # Generate single sample: returns (1, num_pols, channels, times)
    waterfall, exact_mask, _ = generator._generate_single_sample(**gen_kwargs)

    # Extract single polarization
    waterfall = waterfall[0, 0]  # (channels, times)
    ground_truth = exact_mask[0, 0]

    rfi_coverage = np.sum(ground_truth) / ground_truth.size * 100
    print(f"  Shape: {waterfall.shape}")
    print(f"  RFI coverage: {rfi_coverage:.2f}%")
    print(f"  Noise level: {synth_config.noise_mjy} mJy")
    print(f"  RFI power: {synth_config.rfi_power_min}-{synth_config.rfi_power_max} Jy")

    return waterfall, ground_truth


def run_inference(
    waterfall, config, model_path, sam_checkpoint, device, save_probabilities=None, threshold=None
):
    """Run SAM inference."""
    print("\n" + "=" * 60)
    print("STEP 2: SAM Inference")
    print("=" * 60)

    # Prepare data: (baselines, pols, channels, times)
    data = waterfall[np.newaxis, np.newaxis, :, :]

    print(f"  Loading model: {model_path}")
    print(f"  SAM checkpoint: {sam_checkpoint}")
    print(f"  Device: {device}")

    predictor = RFIPredictor(
        model_path=model_path,
        sam_checkpoint=sam_checkpoint,
        device=device,
        batch_size=1,
    )

    # Get probabilities first
    probabilities = predictor.predict_array(
        data=data,
        patch_size=config.processing.patch_size,
        stretch=config.processing.stretch,
        enable_augmentation=config.processing.get("enable_augmentation", False),
        normalize_before_stretch=config.processing.get("normalize_before_stretch", False),
        normalize_after_stretch=config.processing.get("normalize_after_stretch", False),
        return_probabilities=True,
        save_probabilities=save_probabilities,
    )

    # Extract single baseline, single pol
    probs = probabilities[0, 0]

    # Apply threshold
    thresh = probs.mean() if threshold is None else threshold
    print(f"  Threshold: {thresh:.4f}")
    predicted_mask = probs > thresh

    flagged_pct = np.sum(predicted_mask) / predicted_mask.size * 100
    print(f"  Predicted flags: {flagged_pct:.2f}%")

    return predicted_mask, probs


def evaluate_results(ground_truth, predicted_mask):
    """Calculate segmentation metrics and confusion matrix."""
    print("\n" + "=" * 60)
    print("STEP 3: Evaluation")
    print("=" * 60)

    # Standard metrics from module
    metrics = evaluate_segmentation(predicted_mask, ground_truth)

    # Calculate confusion matrix
    pred_bool = predicted_mask.astype(bool)
    true_bool = ground_truth.astype(bool)

    TP = np.logical_and(pred_bool, true_bool).sum()
    TN = np.logical_and(~pred_bool, ~true_bool).sum()
    FP = np.logical_and(pred_bool, ~true_bool).sum()
    FN = np.logical_and(~pred_bool, true_bool).sum()
    total = ground_truth.size

    metrics["TP"] = int(TP)
    metrics["TN"] = int(TN)
    metrics["FP"] = int(FP)
    metrics["FN"] = int(FN)
    metrics["total_pixels"] = int(total)

    print("\nConfusion Matrix:")
    print(f"  TP (True Positive):  {TP:7d} ({TP/total*100:5.2f}%)")
    print(f"  TN (True Negative):  {TN:7d} ({TN/total*100:5.2f}%)")
    print(f"  FP (False Positive): {FP:7d} ({FP/total*100:5.2f}%)")
    print(f"  FN (False Negative): {FN:7d} ({FN/total*100:5.2f}%)")
    print(f"  Total:               {total:7d}")

    print("\nSegmentation Metrics:")
    print(f"  IoU:       {metrics['iou']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Dice:      {metrics['dice']:.4f}")

    return metrics, predicted_mask


def create_plots(waterfall, ground_truth, predicted_mask, probabilities, metrics, output_dir):
    """Create comprehensive visualization with waterfall overlays."""
    print("\n" + "=" * 60)
    print("STEP 4: Visualization")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from matplotlib.colors import LogNorm

    from samrfi.evaluation import compute_calcquality, compute_ffi, compute_statistics

    magnitude = np.abs(waterfall)
    vmin, vmax = magnitude[magnitude > 0].min(), magnitude.max()

    if ground_truth is not None:
        # Synthetic data: 2x4 grid with GT + probabilities
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))

        # Row 1: Input stages with overlays
        # (0,0) Magnitude with ground truth overlay
        im0 = axes[0, 0].imshow(
            magnitude, aspect="auto", cmap="viridis", alpha=0.7, norm=LogNorm(vmin=vmin, vmax=vmax)
        )
        axes[0, 0].contour(ground_truth, levels=[0.5], colors="red", linewidths=1.5, alpha=0.8)
        axes[0, 0].set_title("Magnitude + GT (red contour)", fontsize=14, fontweight="bold")
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("Frequency")
        plt.colorbar(im0, ax=axes[0, 0], label="Amplitude (Jy, log scale)")

        # (0,1) Magnitude with prediction overlay
        im1 = axes[0, 1].imshow(
            magnitude, aspect="auto", cmap="viridis", alpha=0.7, norm=LogNorm(vmin=vmin, vmax=vmax)
        )
        axes[0, 1].contour(predicted_mask, levels=[0.5], colors="cyan", linewidths=1.5, alpha=0.8)
        axes[0, 1].set_title("Magnitude + Pred (cyan contour)", fontsize=14, fontweight="bold")
        axes[0, 1].set_xlabel("Time")
        axes[0, 1].set_ylabel("Frequency")
        plt.colorbar(im1, ax=axes[0, 1], label="Amplitude (Jy, log scale)")

        # (0,2) Ground truth vs Prediction overlay
        axes[0, 2].imshow(
            magnitude, aspect="auto", cmap="gray", alpha=0.5, norm=LogNorm(vmin=vmin, vmax=vmax)
        )
        axes[0, 2].contour(
            ground_truth, levels=[0.5], colors="red", linewidths=2, alpha=0.9, label="GT"
        )
        axes[0, 2].contour(
            predicted_mask,
            levels=[0.5],
            colors="cyan",
            linewidths=1.5,
            alpha=0.9,
            linestyles="--",
            label="Pred",
        )
        axes[0, 2].set_title("GT (red) vs Pred (cyan)", fontsize=14, fontweight="bold")
        axes[0, 2].set_xlabel("Time")
        axes[0, 2].set_ylabel("Frequency")
        axes[0, 2].legend(loc="upper right", fontsize=9)

        # Row 2: Masks and metrics
        # (1,0) Ground truth mask
        axes[1, 0].imshow(ground_truth, aspect="auto", cmap="Reds", vmin=0, vmax=1)
        axes[1, 0].set_title("Ground Truth Mask", fontsize=14, fontweight="bold")
        axes[1, 0].set_xlabel("Time")
        axes[1, 0].set_ylabel("Frequency")
        gt_pct = np.sum(ground_truth) / ground_truth.size * 100
        axes[1, 0].text(
            0.5,
            1.05,
            f"{gt_pct:.2f}% flagged",
            ha="center",
            transform=axes[1, 0].transAxes,
            fontsize=11,
        )

        # (1,1) Prediction mask
        axes[1, 1].imshow(predicted_mask, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        axes[1, 1].set_title("SAM Prediction Mask", fontsize=14, fontweight="bold")
        axes[1, 1].set_xlabel("Time")
        axes[1, 1].set_ylabel("Frequency")
        pred_pct = np.sum(predicted_mask) / predicted_mask.size * 100
        axes[1, 1].text(
            0.5,
            1.05,
            f"{pred_pct:.2f}% flagged",
            ha="center",
            transform=axes[1, 1].transAxes,
            fontsize=11,
        )

        # (0,3) Probability heatmap
        im_prob = axes[0, 3].imshow(probabilities, aspect="auto", cmap="hot", vmin=0, vmax=1)
        axes[0, 3].set_title("SAM Probabilities", fontsize=14, fontweight="bold")
        axes[0, 3].set_xlabel("Time")
        axes[0, 3].set_ylabel("Frequency")
        plt.colorbar(im_prob, ax=axes[0, 3], label="Probability [0,1]")

        # (1,2) Metrics table
        axes[1, 2].axis("off")
        total = metrics["total_pixels"]
        metrics_text = (
            f"Confusion Matrix\n"
            f"{'='*30}\n"
            f"TP: {metrics['TP']:7d} ({metrics['TP']/total*100:5.2f}%)\n"
            f"TN: {metrics['TN']:7d} ({metrics['TN']/total*100:5.2f}%)\n"
            f"FP: {metrics['FP']:7d} ({metrics['FP']/total*100:5.2f}%)\n"
            f"FN: {metrics['FN']:7d} ({metrics['FN']/total*100:5.2f}%)\n\n"
            f"Metrics\n"
            f"{'='*30}\n"
            f"IoU:       {metrics['iou']:.4f}\n"
            f"Precision: {metrics['precision']:.4f}\n"
            f"Recall:    {metrics['recall']:.4f}\n"
            f"F1:        {metrics['f1']:.4f}\n"
            f"Dice:      {metrics['dice']:.4f}\n"
        )
        axes[1, 2].text(
            0.1, 0.5, metrics_text, fontsize=11, family="monospace", verticalalignment="center"
        )

        # (1,3) Probability histogram
        axes[1, 3].hist(
            probabilities.ravel(), bins=100, color="orange", alpha=0.7, edgecolor="black"
        )
        axes[1, 3].set_xlabel("Probability", fontsize=12)
        axes[1, 3].set_ylabel("Frequency", fontsize=12)
        axes[1, 3].set_title("Probability Distribution", fontsize=14, fontweight="bold")
        axes[1, 3].axvline(
            probabilities.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean={probabilities.mean():.3f}",
        )
        axes[1, 3].legend(fontsize=10)
    else:
        # Real data: 2x3 grid without GT + probabilities
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # (0,0) Magnitude with prediction overlay
        im0 = axes[0, 0].imshow(
            magnitude, aspect="auto", cmap="viridis", alpha=0.7, norm=LogNorm(vmin=vmin, vmax=vmax)
        )
        axes[0, 0].contour(predicted_mask, levels=[0.5], colors="cyan", linewidths=1.5, alpha=0.8)
        axes[0, 0].set_title("Magnitude + SAM Flags (cyan)", fontsize=14, fontweight="bold")
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("Frequency")
        plt.colorbar(im0, ax=axes[0, 0], label="Amplitude (Jy, log scale)")

        # (0,1) Prediction mask only
        im1 = axes[0, 1].imshow(predicted_mask, aspect="auto", cmap="Reds", vmin=0, vmax=1)
        axes[0, 1].set_title("SAM Prediction Mask", fontsize=14, fontweight="bold")
        axes[0, 1].set_xlabel("Time")
        axes[0, 1].set_ylabel("Frequency")
        pred_pct = np.sum(predicted_mask) / predicted_mask.size * 100
        axes[0, 1].text(
            0.5,
            1.05,
            f"{pred_pct:.2f}% flagged",
            ha="center",
            transform=axes[0, 1].transAxes,
            fontsize=11,
        )
        plt.colorbar(im1, ax=axes[0, 1], label="Flag")

        # (0,2) Probability heatmap
        im_prob = axes[0, 2].imshow(probabilities, aspect="auto", cmap="hot", vmin=0, vmax=1)
        axes[0, 2].set_title("SAM Probabilities", fontsize=14, fontweight="bold")
        axes[0, 2].set_xlabel("Time")
        axes[0, 2].set_ylabel("Frequency")
        plt.colorbar(im_prob, ax=axes[0, 2], label="Probability [0,1]")

        # (1,0) Statistics table
        axes[1, 0].axis("off")
        stats_before = compute_statistics(waterfall, flags=None)
        stats_after = compute_statistics(waterfall, flags=predicted_mask)
        ffi_metrics = compute_ffi(waterfall, predicted_mask)
        cq_metrics = compute_calcquality(waterfall, predicted_mask)

        stats_text = (
            f"Statistics\n"
            f"{'='*35}\n"
            f"Before: Mean={stats_before['mean']:.4e}\n"
            f"        Std={stats_before['std']:.4e}\n"
            f"After:  Mean={stats_after['mean']:.4e}\n"
            f"        Std={stats_after['std']:.4e}\n\n"
            f"FFI (Simple)\n"
            f"{'='*35}\n"
            f"Score:         {ffi_metrics['ffi']:.4f}\n"
            f"MAD Reduction: {ffi_metrics['mad_reduction']:.4f}\n"
            f"STD Reduction: {ffi_metrics['std_reduction']:.4f}\n\n"
            f"calcquality (Paper)\n"
            f"{'='*35}\n"
            f"Score:       {cq_metrics['calcquality']:.4f} ↓\n"
            f"Sensitivity: {cq_metrics['sensitivity']:.4f}\n"
            f"Mean Shift:  {cq_metrics['mean_shift']:.4f}\n"
            f"Std Shift:   {cq_metrics['std_shift']:.4f}\n"
            f"Overflag:    {cq_metrics['overflagging_penalty']:.4f}\n"
        )
        axes[1, 0].text(
            0.05, 0.5, stats_text, fontsize=10, family="monospace", verticalalignment="center"
        )

        # (1,1) Amplitude Histogram
        axes[1, 1].hist(magnitude.ravel(), bins=100, alpha=0.7, label="All data", log=True)
        axes[1, 1].hist(
            magnitude[~predicted_mask].ravel(), bins=100, alpha=0.7, label="Unflagged", log=True
        )
        axes[1, 1].set_xlabel("Amplitude (Jy)")
        axes[1, 1].set_ylabel("Count (log scale)")
        axes[1, 1].set_title("Amplitude Distribution", fontsize=14, fontweight="bold")
        axes[1, 1].legend()
        axes[1, 1].set_xscale("log")

        # (1,2) Probability histogram
        axes[1, 2].hist(
            probabilities.ravel(), bins=100, color="orange", alpha=0.7, edgecolor="black"
        )
        axes[1, 2].set_xlabel("Probability", fontsize=12)
        axes[1, 2].set_ylabel("Frequency", fontsize=12)
        axes[1, 2].set_title("Probability Distribution", fontsize=14, fontweight="bold")
        axes[1, 2].axvline(
            probabilities.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean={probabilities.mean():.3f}",
        )
        axes[1, 2].legend(fontsize=10)

    plt.tight_layout()

    output_path = output_dir / "single_array_validation.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Validate SAM on single array (synthetic or real)")
    parser.add_argument("--model", required=True, help="Path to trained model (.pth)")
    parser.add_argument("--config", default="configs/validation.yaml", help="Config file")
    parser.add_argument(
        "--sam-checkpoint", default="base_plus", help="SAM variant (tiny/small/base_plus/large)"
    )
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output", default="./single_array_validation", help="Output directory")
    parser.add_argument(
        "--input-npy", help="Load real data from .npy file (skips synthetic generation)"
    )
    parser.add_argument("--save-probabilities", help="Save probability maps to .npy file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="RFI probability threshold (default: None=use mean)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("SAM-RFI Single Array Validation")
    print("=" * 60)
    print(f"Model:  {args.model}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")

    # Load config
    config = ConfigLoader.load_data(args.config)

    # Load data (synthetic or real)
    if args.input_npy:
        print(f"Input:  {args.input_npy} (real data)")
        waterfall = np.load(args.input_npy)
        ground_truth = None
        print(f"  Loaded shape: {waterfall.shape}")
    else:
        print("Input:  Synthetic generation")
        waterfall, ground_truth = generate_single_array(config)

    # Run SAM inference
    predicted_mask, probabilities = run_inference(
        waterfall,
        config,
        args.model,
        args.sam_checkpoint,
        args.device,
        args.save_probabilities,
        args.threshold,
    )

    # Evaluate
    if ground_truth is not None:
        # Synthetic: compute metrics
        metrics, predicted_mask = evaluate_results(ground_truth, predicted_mask)
    else:
        # Real: compute statistics before/after flagging
        print_statistics_comparison(waterfall, predicted_mask)
        metrics = None

    # Visualize
    create_plots(waterfall, ground_truth, predicted_mask, probabilities, metrics, args.output)

    print("\n" + "=" * 60)
    print("Validation Complete")
    print("=" * 60)
    print(f"Results saved to: {args.output}")
    if metrics is not None:
        print(f"F1 Score: {metrics['f1']:.4f}")

    return metrics


if __name__ == "__main__":
    main()
