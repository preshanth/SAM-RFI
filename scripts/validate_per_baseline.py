#!/usr/bin/env python
"""
Per-Baseline Validation Script

Validates SAM-RFI model by:
1. Injecting synthetic RFI into each baseline of template MS
2. Running SAM-RFI prediction
3. Comparing per-baseline to ground truth
4. Plotting metrics per baseline

Usage:
    python scripts/validate_per_baseline.py \
        --model model.pth \
        --template-ms template.ms \
        --config configs/validation.yaml \
        --num-baselines 10
"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from samrfi.config import ConfigLoader
from samrfi.data_generation import SyntheticDataGenerator
from samrfi.evaluation import evaluate_segmentation, inject_synthetic_data
from samrfi.inference import RFIPredictor


def main():
    parser = argparse.ArgumentParser(description="Per-baseline SAM-RFI validation")
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--template-ms", required=True, help="Path to template MS")
    parser.add_argument("--config", required=True, help="Path to validation config")
    parser.add_argument(
        "--num-baselines",
        type=int,
        default=10,
        help="Number of baselines to validate (default: 10)",
    )
    parser.add_argument(
        "--output", default="./validation_results", help="Output directory for results"
    )
    parser.add_argument(
        "--sam-checkpoint",
        choices=["tiny", "small", "base_plus", "large"],
        default="large",
        help="SAM2 variant (default: large)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    print(f"\n{'='*60}")
    print("Loading Configuration")
    print(f"{'='*60}")
    config = ConfigLoader.load_data(args.config)
    print(f"Config: {args.config}")

    # Extract params
    synth_config = config.synthetic
    proc_config = config.processing
    val_config = getattr(config, "validation", {})

    patch_size = proc_config.get("patch_size", 1024)
    stretch = proc_config.get("stretch", None)
    enable_aug = proc_config.get("enable_augmentation", False)
    aug_rot = proc_config.get("augmentation_rotations", 1)
    norm_before = proc_config.get("normalize_before_stretch", False)
    norm_after = proc_config.get("normalize_after_stretch", False)

    # Use sam_checkpoint from config if not explicitly passed via CLI
    if args.sam_checkpoint == "large":  # Default value from argparse
        sam_checkpoint = val_config.get("sam_checkpoint", "large")
    else:
        sam_checkpoint = args.sam_checkpoint

    print(f"  Patch size: {patch_size}")
    print(f"  Stretch: {stretch}")
    print(f"  Augmentation: {enable_aug} (rotations={aug_rot})")
    print(f"  Normalize before/after: {norm_before}/{norm_after}")
    print(f"  SAM checkpoint: {sam_checkpoint}")

    # Initialize generator
    print(f"\n{'='*60}")
    print("Initializing Synthetic Generator")
    print(f"{'='*60}")
    generator = SyntheticDataGenerator(config)

    # Build generation kwargs
    num_channels = synth_config.get("num_channels", 1024)
    num_times = synth_config.get("num_times", 1024)
    noise_level = synth_config.get("noise_mjy", 1.0)
    rfi_power_min = synth_config.get("rfi_power_min", 1000.0)
    rfi_power_max = synth_config.get("rfi_power_max", 10000.0)
    rfi_config = generator._parse_rfi_config(synth_config)
    enable_bandpass = synth_config.get("enable_bandpass_rolloff", False)
    bandpass_order = synth_config.get("bandpass_polynomial_order", 8)
    num_polarizations = synth_config.get("num_polarizations", 4)
    pol_corr = synth_config.get("polarization_correlation", 0.8)

    gen_kwargs = {
        "num_channels": num_channels,
        "num_times": num_times,
        "noise_level": noise_level,
        "rfi_power_min": rfi_power_min,
        "rfi_power_max": rfi_power_max,
        "rfi_config": rfi_config,
        "enable_bandpass": enable_bandpass,
        "bandpass_order": bandpass_order,
        "num_polarizations": num_polarizations,
        "pol_corr": pol_corr,
        "synth_config": synth_config,
    }

    # Initialize predictor
    print(f"\n{'='*60}")
    print("Loading Model")
    print(f"{'='*60}")
    predictor = RFIPredictor(model_path=args.model, sam_checkpoint=sam_checkpoint, device="cuda")

    # Create working MS (copy of template)
    template_ms = Path(args.template_ms)
    work_ms = output_dir / "work.ms"

    print(f"\n{'='*60}")
    print("Setting Up Working MS")
    print(f"{'='*60}")
    if work_ms.exists():
        shutil.rmtree(work_ms)
    shutil.copytree(template_ms, work_ms)
    print(f"  Created: {work_ms}")

    # Get MS metadata
    from casatools import table

    tb_ant = table()
    tb_ant.open(str(template_ms / "ANTENNA"))
    num_antennas = tb_ant.nrows()
    tb_ant.close()

    # Get number of polarizations from MS
    tb_main = table()
    tb_main.open(str(template_ms))
    # Query first row to get pol dimension
    subtable = tb_main.query("ANTENNA1==0 && ANTENNA2==1")
    data_sample = subtable.getcol("DATA")
    num_pols = data_sample.shape[0]  # First dimension is pols
    subtable.close()
    tb_main.close()

    # Build baseline list in same order as MSLoader.load()
    baseline_list = []
    for i in range(num_antennas):
        for j in range(i + 1, num_antennas):
            baseline_list.append((i, j))

    num_baselines_total = len(baseline_list)
    num_baselines_validate = min(args.num_baselines, num_baselines_total)

    print(
        f"  MS structure: {num_antennas} antennas, {num_baselines_total} baselines, {num_pols} pols"
    )

    print(f"\n{'='*60}")
    print(f"Validating {num_baselines_validate}/{num_baselines_total} Baselines")
    print(f"{'='*60}")

    # Generate synthetic data for ALL baselines at once
    print(f"\n{'='*60}")
    print("Generating Synthetic Data")
    print(f"{'='*60}")
    print(f"  Generating {num_baselines_total} baselines with {num_pols} pols...")

    # Override num_polarizations to match MS
    gen_kwargs["num_polarizations"] = num_pols

    all_waterfalls = []
    all_ground_truth = []

    for baseline_idx in tqdm(range(num_baselines_total), desc="Generating"):
        waterfall, ground_truth, rfi_params = generator._generate_single_sample(**gen_kwargs)
        all_waterfalls.append(waterfall[0])  # Remove extra baseline dimension
        all_ground_truth.append(ground_truth[0])

    # Stack into full arrays
    full_waterfall = np.stack(all_waterfalls)  # (num_baselines, pols, channels, times)
    full_ground_truth = np.stack(all_ground_truth)  # (num_baselines, pols, channels, times)

    print(f"  Generated shape: {full_waterfall.shape}")

    # Inject all baselines into MS
    print(f"\n{'='*60}")
    print("Injecting Synthetic Data")
    print(f"{'='*60}")
    inject_synthetic_data(
        template_ms_path=work_ms,
        synthetic_data=full_waterfall,
        output_ms_path=work_ms,
        baseline_map=baseline_list,
    )

    # Run prediction ONCE on full MS
    print(f"\n{'='*60}")
    print("Running Prediction")
    print(f"{'='*60}")
    predicted_flags = predictor.predict_ms(
        ms_path=work_ms,
        patch_size=patch_size,
        stretch=stretch,
        save_flags=False,
        enable_augmentation=enable_aug,
        normalize_before_stretch=norm_before,
        normalize_after_stretch=norm_after,
    )

    # Extract and compare metrics per baseline
    print(f"\n{'='*60}")
    print("Computing Per-Baseline Metrics")
    print(f"{'='*60}")

    all_metrics = []
    baseline_labels = []

    for baseline_idx in tqdm(range(num_baselines_validate), desc="Metrics"):
        ant1, ant2 = baseline_list[baseline_idx]
        baseline_labels.append(f"{ant1}-{ant2}")

        pred_baseline = predicted_flags[baseline_idx]  # (pols, channels, times)
        gt_baseline = full_ground_truth[baseline_idx]  # (pols, channels, times)

        # Flatten across pols for comparison
        pred_flat = pred_baseline.max(axis=0)  # (channels, times)
        gt_flat = gt_baseline.max(axis=0)  # (channels, times)

        # Compute metrics
        metrics = evaluate_segmentation(pred_flat, gt_flat)
        all_metrics.append(metrics)

        # Save individual plot
        save_baseline_plot(
            ground_truth=gt_flat,
            prediction=pred_flat,
            metrics=metrics,
            baseline_label=f"{ant1}-{ant2}",
            output_path=output_dir / f"baseline_{ant1:02d}_{ant2:02d}.png",
        )

    # Plot metrics across baselines
    plot_metrics_across_baselines(
        all_metrics, baseline_labels, output_path=output_dir / "metrics_per_baseline.png"
    )

    # Save metrics to JSON
    results = {
        "baselines": baseline_labels,
        "metrics": all_metrics,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("Validation Complete")
    print(f"{'='*60}")
    print(f"  Baselines validated: {num_baselines_validate}")
    print(f"  Results saved to: {output_dir}")

    # Print aggregate statistics
    metrics_keys = all_metrics[0].keys()
    print("\nAggregate Statistics:")
    for key in metrics_keys:
        values = [m[key] for m in all_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {key:12s}: {mean_val:.3f} ± {std_val:.3f}")


def save_baseline_plot(ground_truth, prediction, metrics, baseline_label, output_path):
    """Save comparison plot for single baseline"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ground truth
    axes[0].imshow(ground_truth.T, aspect="auto", cmap="Reds", interpolation="nearest")
    axes[0].set_title(f"Ground Truth - Baseline {baseline_label}")
    axes[0].set_xlabel("Channel")
    axes[0].set_ylabel("Time")

    # Prediction
    axes[1].imshow(prediction.T, aspect="auto", cmap="Blues", interpolation="nearest")
    axes[1].set_title(f"Prediction - Baseline {baseline_label}")
    axes[1].set_xlabel("Channel")
    axes[1].set_ylabel("Time")

    # Difference
    diff = np.zeros((*ground_truth.shape, 3))
    diff[ground_truth & prediction] = [0, 1, 0]  # TP = green
    diff[prediction & ~ground_truth] = [1, 0, 0]  # FP = red
    diff[ground_truth & ~prediction] = [1, 1, 0]  # FN = yellow

    axes[2].imshow(diff.transpose(1, 0, 2), aspect="auto", interpolation="nearest")
    axes[2].set_title("Difference (TP=green, FP=red, FN=yellow)")
    axes[2].set_xlabel("Channel")
    axes[2].set_ylabel("Time")

    # Add metrics text
    metrics_text = "\n".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
    fig.text(0.5, 0.02, metrics_text, ha="center", fontsize=10, family="monospace")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_across_baselines(all_metrics, baseline_labels, output_path):
    """Plot metrics across all baselines"""
    metrics_keys = list(all_metrics[0].keys())
    num_baselines = len(all_metrics)

    fig, axes = plt.subplots(len(metrics_keys), 1, figsize=(12, 3 * len(metrics_keys)))

    if len(metrics_keys) == 1:
        axes = [axes]

    for idx, key in enumerate(metrics_keys):
        values = [m[key] for m in all_metrics]

        axes[idx].bar(range(num_baselines), values, color="steelblue", alpha=0.7)
        axes[idx].axhline(
            np.mean(values), color="red", linestyle="--", label=f"Mean: {np.mean(values):.3f}"
        )
        axes[idx].set_ylabel(key)
        axes[idx].set_ylim([0, 1])
        axes[idx].set_xticks(range(num_baselines))
        axes[idx].set_xticklabels(baseline_labels, rotation=45, ha="right")
        axes[idx].legend()
        axes[idx].grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Baseline")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
