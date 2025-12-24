#!/usr/bin/env python
"""
Model Validation Script

Generates synthetic RFI data, runs trained model, compares predictions vs ground truth,
and produces comparison plots.

Usage:
    python scripts/validate_model.py --model model.pth --template-ms data.ms --output validation_results/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from samrfi.data_generation import SyntheticDataGenerator
from samrfi.evaluation import evaluate_segmentation, inject_synthetic_data
from samrfi.inference import RFIPredictor


def generate_validation_samples(num_samples, config):
    """
    Generate synthetic RFI samples with ground truth.

    Args:
        num_samples: Number of samples to generate
        config: Configuration object for synthetic generator (DataConfig or dict-like)

    Returns:
        List of (waterfall_data, ground_truth_mask) tuples
    """
    print(f"\n{'='*60}")
    print("Generating Validation Data")
    print(f"{'='*60}")
    print(f"Samples: {num_samples}")

    generator = SyntheticDataGenerator(config)
    samples = []

    # Extract synthetic config from provided object (support DataConfig or dict)
    if hasattr(config, "synthetic"):
        synth_config = config.synthetic
        proc_config = getattr(config, "processing", {})
    elif isinstance(config, dict):
        synth_config = config.get("synthetic", {})
        proc_config = config.get("processing", {})
    else:
        # Fallback: assume loading via ConfigLoader.load_data was not used
        raise ValueError(
            'Config for validation must include a "synthetic" section (use ConfigLoader.load_data)'
        )

    # Build generation kwargs consistent with SyntheticDataGenerator.generate()
    num_channels = synth_config.get("num_channels", 1024)
    num_times = synth_config.get("num_times", 1024)
    noise_level = synth_config.get("noise_mjy", 1.0)
    rfi_power_min = synth_config.get("rfi_power_min", 1000.0)
    rfi_power_max = synth_config.get("rfi_power_max", 10000.0)

    rfi_config = generator._parse_rfi_config(synth_config)

    enable_bandpass = synth_config.get("enable_bandpass_rolloff", False)
    bandpass_order = synth_config.get("bandpass_polynomial_order", 8)
    num_polarizations = synth_config.get("num_polarizations", 1)
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

    for i in tqdm(range(num_samples), desc="Generating samples"):
        waterfall, exact_mask, rfi_params = generator._generate_single_sample(**gen_kwargs)
        samples.append((waterfall, exact_mask))

    print(f"✓ Generated {len(samples)} validation samples")
    return samples


def run_validation(
    model_path,
    template_ms,
    output_dir,
    num_samples=10,
    config=None,
    sam_checkpoint="large",
    allow_partial_load=False,
    auto_select_sam=False,
):
    """
    Run full validation: generate data, predict, compare, plot.

    Args:
        model_path: Path to trained model checkpoint
        template_ms: Path to template MS for structure
        output_dir: Directory to save results
        num_samples: Number of samples to validate
        config: Synthetic data configuration (uses default if None)
        sam_checkpoint: SAM variant to instantiate ('tiny','small','base_plus','large')
        allow_partial_load: If True, allow partial loading of checkpoint when shapes mismatch
        auto_select_sam: If True, automatically select the SAM variant that best matches the checkpoint

    Returns:
        Dictionary with aggregated metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data if config provided
    if config is not None:
        samples = generate_validation_samples(num_samples, config)
    else:
        print("No config provided - expecting pre-generated data")
        # TODO: Load pre-generated samples
        raise NotImplementedError("Pre-generated data loading not yet implemented")

    # Initialize predictor
    print(f"\n{'='*60}")
    print("Loading Model")
    print(f"{'='*60}")
    predictor = RFIPredictor(
        model_path=model_path,
        sam_checkpoint=sam_checkpoint,
        device="cuda",
        allow_partial_load=allow_partial_load,
        auto_select_sam=auto_select_sam,
    )

    # Process each sample
    all_metrics = []
    predictions_list = []
    ground_truth_list = []

    print(f"\n{'='*60}")
    print("Running Predictions")
    print(f"{'='*60}")

    for sample_idx, (waterfall, ground_truth) in enumerate(tqdm(samples, desc="Samples")):
        # Inject into MS
        synthetic_ms = output_dir / f"sample_{sample_idx:03d}.ms"
        inject_synthetic_data(
            template_ms_path=template_ms,
            synthetic_data=waterfall,
            output_ms_path=synthetic_ms,
        )

        # Run prediction
        predicted_flags = predictor.predict_ms(
            ms_path=synthetic_ms, save_flags=False  # Don't overwrite, just return
        )

        # Flatten for comparison (average over baselines and pols)
        # Ground truth shape: (baselines, pols, channels, times)
        # Predicted shape: (baselines, pols, channels, times)
        gt_flat = ground_truth.max(axis=(0, 1))  # (channels, times) - max over baselines/pols
        pred_flat = predicted_flags.max(axis=(0, 1))  # (channels, times)

        # Compute metrics
        metrics = evaluate_segmentation(pred_flat, gt_flat)
        all_metrics.append(metrics)

        # Store for plotting
        predictions_list.append(pred_flat)
        ground_truth_list.append(gt_flat)

        # Save individual sample visualization
        save_sample_comparison(
            ground_truth=gt_flat,
            prediction=pred_flat,
            metrics=metrics,
            output_path=output_dir / f"sample_{sample_idx:03d}.png",
        )

    # Aggregate metrics
    aggregated = aggregate_metrics(all_metrics)

    # Save summary plots
    plot_metrics_summary(all_metrics, output_path=output_dir / "metrics_summary.png")

    plot_example_grid(
        predictions=predictions_list[:6],  # First 6 samples
        ground_truth=ground_truth_list[:6],
        output_path=output_dir / "examples_grid.png",
    )

    # Print summary
    print(f"\n{'='*60}")
    print("Validation Results")
    print(f"{'='*60}")
    print(f"Samples: {len(all_metrics)}")
    print("\nAggregate Metrics:")
    for metric, value in aggregated.items():
        print(f"  {metric:12s}: {value['mean']:.3f} ± {value['std']:.3f}")

    print(f"\nResults saved to: {output_dir}")

    return aggregated


def aggregate_metrics(metrics_list):
    """Compute mean and std of metrics across samples"""
    keys = metrics_list[0].keys()
    aggregated = {}

    for key in keys:
        values = [m[key] for m in metrics_list]
        aggregated[key] = {"mean": np.mean(values), "std": np.std(values), "all": values}

    return aggregated


def save_sample_comparison(ground_truth, prediction, metrics, output_path):
    """Save side-by-side comparison of ground truth vs prediction"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ground truth
    axes[0].imshow(ground_truth.T, aspect="auto", cmap="Reds", interpolation="nearest")
    axes[0].set_title("Ground Truth")
    axes[0].set_xlabel("Channel")
    axes[0].set_ylabel("Time")

    # Prediction
    axes[1].imshow(prediction.T, aspect="auto", cmap="Blues", interpolation="nearest")
    axes[1].set_title("Prediction")
    axes[1].set_xlabel("Channel")
    axes[1].set_ylabel("Time")

    # Difference (TP=green, FP=red, FN=yellow)
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


def plot_metrics_summary(metrics_list, output_path):
    """Plot box plots of all metrics"""
    keys = list(metrics_list[0].keys())
    data = {k: [m[k] for m in metrics_list] for k in keys}

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = range(len(keys))
    ax.boxplot([data[k] for k in keys], positions=positions, labels=keys)
    ax.set_ylabel("Score")
    ax.set_title("Metrics Distribution Across Validation Samples")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_example_grid(predictions, ground_truth, output_path):
    """Plot grid of example comparisons"""
    n_samples = len(predictions)
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 3 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
        # Ground truth
        axes[i, 0].imshow(gt.T, aspect="auto", cmap="Reds", interpolation="nearest")
        axes[i, 0].set_ylabel(f"Sample {i+1}")
        if i == 0:
            axes[i, 0].set_title("Ground Truth")
        if i == n_samples - 1:
            axes[i, 0].set_xlabel("Channel")

        # Prediction
        axes[i, 1].imshow(pred.T, aspect="auto", cmap="Blues", interpolation="nearest")
        if i == 0:
            axes[i, 1].set_title("Prediction")
        if i == n_samples - 1:
            axes[i, 1].set_xlabel("Channel")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Validate SAM-RFI model on synthetic data")
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--template-ms", required=True, help="Path to template MS")
    parser.add_argument(
        "--output", default="./validation_results", help="Output directory for results"
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Number of validation samples")
    parser.add_argument(
        "--config", help="Path to synthetic data config (uses default if not provided)"
    )
    parser.add_argument(
        "--sam-checkpoint",
        choices=["tiny", "small", "base_plus", "large"],
        default="large",
        help="SAM2 base checkpoint variant to instantiate (default: large)",
    )
    parser.add_argument(
        "--allow-partial-load",
        action="store_true",
        help="Allow partial loading of checkpoint when shapes mismatch (not recommended)",
    )
    parser.add_argument(
        "--auto-select-sam",
        action="store_true",
        help="Auto-select the SAM2 variant that best matches the checkpoint (may download multiple models; potentially slow)",
    )
    parser.add_argument(
        "--print-checkpoint-keys",
        action="store_true",
        help="Print checkpoint top-level keys and model parameter shapes (if present) and exit",
    )

    args = parser.parse_args()

    # Load config if provided (data-generation style config expected)
    config = None
    if args.config:
        from samrfi.config import ConfigLoader

        # Use load_data to preserve the nested 'synthetic' and 'processing' sections
        config = ConfigLoader.load_data(args.config)

    # Optionally print checkpoint info and exit
    if args.print_checkpoint_keys:
        ck = torch.load(args.model, map_location="cpu")
        print("Checkpoint type:", type(ck))
        if isinstance(ck, dict):
            print("Top-level keys:", list(ck.keys()))
            # If model_state_dict present, show a subset of its keys and shapes
            candidate = None
            if "model_state_dict" in ck:
                candidate = ck["model_state_dict"]
            elif "state_dict" in ck:
                candidate = ck["state_dict"]
            elif all(isinstance(v, torch.Tensor) for v in ck.values()):
                candidate = ck

            if candidate is not None:
                print("\nModel parameter sample (first 40 items):")
                for i, (k, v) in enumerate(candidate.items()):
                    if i >= 40:
                        break
                    print(f"  {k}: {tuple(v.shape) if hasattr(v, 'shape') else type(v)}")
            else:
                print("No obvious model_state_dict found in checkpoint.")
        else:
            print("Checkpoint is not a dict; likely a plain state_dict mapping")
        print("\nExiting (print-checkpoint-keys requested).")
        return

    # Run validation
    results = run_validation(
        model_path=args.model,
        template_ms=args.template_ms,
        output_dir=args.output,
        num_samples=args.num_samples,
        config=config,
        sam_checkpoint=args.sam_checkpoint,
        allow_partial_load=args.allow_partial_load,
        auto_select_sam=args.auto_select_sam,
    )

    # Save results to JSON
    import json

    results_serializable = {
        k: {"mean": float(v["mean"]), "std": float(v["std"])} for k, v in results.items()
    }

    with open(Path(args.output) / "results.json", "w") as f:
        json.dump(results_serializable, f, indent=2)

    print("\n✓ Validation complete!")


if __name__ == "__main__":
    main()
