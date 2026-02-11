#!/usr/bin/env python
"""
Benchmark SAM-RFI vs tfcrop+rflag on synthetic data

1. Copy template MS
2. Generate synthetic noise + RFI (from validation.yaml config)
3. Inject into MS
4. Run SAM-RFI
5. Run tfcrop + rflag
6. Compute metrics vs ground truth
7. Save to CSV and JSON

Usage:
    python benchmark_synthetic.py template.ms --config configs/validation.yaml --output ./benchmark_results
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from casatasks import flagdata, flagmanager
from rfi_toolbox.data_generation import SyntheticDataGenerator
from rfi_toolbox.evaluation import (
    compute_calcquality,
    compute_ffi,
    compute_statistics,
    evaluate_segmentation,
)
from rfi_toolbox.io import MSLoader, inject_synthetic_data
from tqdm import tqdm

from samrfi.config import ConfigLoader
from samrfi.inference import RFIPredictor


def generate_and_inject_synthetic_data(config, ms_path, baseline_map, num_channels, num_times):
    """
    Generate synthetic RFI and inject into MS baseline-by-baseline.

    Returns synthetic data and ground truth masks.
    """
    synth_config = config.synthetic
    generator = SyntheticDataGenerator(config)

    # Generator kwargs matching MS dimensions
    num_pols = 4  # Standard 4 polarizations
    gen_kwargs = {
        "num_channels": num_channels,
        "num_times": num_times,
        "noise_level": synth_config.get("noise_mjy", 1.0),
        "rfi_power_min": synth_config.get("rfi_power_min", 1000.0),
        "rfi_power_max": synth_config.get("rfi_power_max", 10000.0),
        "rfi_config": generator._parse_rfi_config(synth_config),
        "enable_bandpass": synth_config.get("enable_bandpass_rolloff", False),
        "bandpass_order": synth_config.get("bandpass_polynomial_order", 0),
        "num_polarizations": num_pols,
        "pol_corr": synth_config.get("polarization_correlation", 0.8),
        "synth_config": synth_config,
    }

    all_data = []
    all_ground_truth = []

    print("  Generating and injecting synthetic data...")
    for ant1, ant2 in tqdm(baseline_map, desc="  Baselines"):
        # Generate synthetic data for this baseline
        waterfall, ground_truth, _ = generator._generate_single_sample(**gen_kwargs)

        # Save for later
        all_data.append(waterfall[0])  # Remove batch dim
        all_ground_truth.append(ground_truth[0])  # Remove batch dim

        # Inject this baseline into MS
        inject_synthetic_data(
            template_ms_path=ms_path,
            synthetic_data=waterfall[0][np.newaxis, :, :, :],  # Add baseline dim
            output_ms_path=ms_path,
            baseline_map=[(ant1, ant2)],
        )

    # Stack all baselines
    synthetic_data = np.stack(all_data)  # (baselines, pols, channels, times)
    ground_truth = np.stack(all_ground_truth)  # (baselines, pols, channels, times)

    return synthetic_data, ground_truth


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def plot_comparison(synthetic_data, ground_truth, sam_flags, tfcrop_flags, output_dir):
    """Create side-by-side waterfall plot comparison with overlaid flags."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    # Use first baseline, average over polarizations
    baseline_idx = 0
    data = np.abs(synthetic_data[baseline_idx].mean(axis=0))  # (channels, times)
    sam = sam_flags[baseline_idx].max(axis=0)
    tfcrop = tfcrop_flags[baseline_idx].max(axis=0)

    # Set color scale to show noise (avoid RFI saturation)
    vmin = np.percentile(data, 5)  # 5th percentile (noise floor)
    vmax = np.percentile(data, 50)  # 95th percentile

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: SAM-RFI
    im0 = axes[0].imshow(data, aspect="auto", cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("SAM-RFI Flags (white = flagged)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Channel")
    plt.colorbar(im0, ax=axes[0], label="Amplitude (Jy)")

    # Overlay SAM flags in white
    masked_sam = np.ma.masked_where(sam == 0, sam)
    axes[0].imshow(
        masked_sam,
        aspect="auto",
        cmap=mcolors.ListedColormap(["white"]),
        origin="lower",
        alpha=0.7,
        vmin=0,
        vmax=1,
    )

    # Right: tfcrop+rflag
    im1 = axes[1].imshow(data, aspect="auto", cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title("tfcrop+rflag Flags (white = flagged)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Channel")
    plt.colorbar(im1, ax=axes[1], label="Amplitude (Jy)")

    # Overlay tfcrop flags in white
    masked_tfcrop = np.ma.masked_where(tfcrop == 0, tfcrop)
    axes[1].imshow(
        masked_tfcrop,
        aspect="auto",
        cmap=mcolors.ListedColormap(["white"]),
        origin="lower",
        alpha=0.7,
        vmin=0,
        vmax=1,
    )

    plt.tight_layout()

    plot_path = output_dir / "comparison_waterfall.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n✓ Saved comparison plot: {plot_path}")

    return str(plot_path)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SAM-RFI vs tfcrop+rflag on synthetic data"
    )
    parser.add_argument("ms_path", help="Existing MS to inject synthetic data into")
    parser.add_argument("--config", required=True, help="validation.yaml config file")
    parser.add_argument("--output", default="./benchmark_results", help="Output directory")
    parser.add_argument("--model", default="polarimetic/sam-rfi/large", help="SAM-RFI model")
    parser.add_argument("--num-antennas", type=int, default=None, help="Number of antennas")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="SAM-RFI probability threshold (default: mean)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SAM-RFI vs tfcrop+rflag Benchmark (Synthetic)")
    print("=" * 60)
    print(f"MS: {args.ms_path}")
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")

    # Load config
    print("\n[1/7] Loading config...")
    config = ConfigLoader.load_data(args.config)

    # Get MS dimensions
    print("\n[2/7] Reading MS dimensions...")
    loader = MSLoader(args.ms_path)
    metadata = loader.get_metadata(num_antennas=args.num_antennas)
    loader.close()

    num_baselines = metadata["num_baselines"]
    num_channels = metadata["total_channels"]  # Use total across all SPWs
    num_times = metadata["num_times"]
    baseline_map = metadata["baseline_map"]
    num_spws = metadata["num_spws"]

    print(f"  Baselines: {num_baselines}")
    print(f"  SPWs: {num_spws}")
    print(f"  Channels: {num_channels} total ({metadata['num_channels']} per SPW)")
    print(f"  Times: {num_times}")

    # Generate and inject synthetic data
    print("\n[3/7] Generating and injecting synthetic data...")
    synthetic_data, ground_truth = generate_and_inject_synthetic_data(
        config, args.ms_path, baseline_map, num_channels, num_times
    )
    print(f"  Generated: {synthetic_data.shape}")
    print(f"  RFI: {ground_truth.sum() / ground_truth.size * 100:.2f}%")

    # Save ground truth
    ground_truth_path = output_dir / "ground_truth.npy"
    np.save(ground_truth_path, ground_truth)
    print(f"  Saved ground truth: {ground_truth_path}")

    # Backup original (unflagged)
    print("\n[4/7] Saving original flags...")
    flagmanager(vis=str(args.ms_path), mode="save", versionname="original")

    # Run SAM-RFI
    print("\n[5/7] Running SAM-RFI...")
    if args.threshold is not None:
        print(f"  Using threshold: {args.threshold}")
    else:
        print("  Using threshold: mean (default)")
    predictor = RFIPredictor(model_path=args.model, device="cuda")
    predictor.predict_ms_per_baseline(
        ms_path=str(args.ms_path),
        patch_size=1024,
        stretch=None,
        save_flags=True,
        threshold=args.threshold,
    )
    flagmanager(vis=str(args.ms_path), mode="save", versionname="samrfi")

    # Run tfcrop + rflag (each independently, then combined)
    print("\n[6/7] Running tfcrop + rflag...")

    # Restore to original
    flagmanager(vis=str(args.ms_path), mode="restore", versionname="original")

    # Run tfcrop on original
    flagdata(vis=str(args.ms_path), mode="tfcrop", datacolumn="data", action="apply")
    flagmanager(vis=str(args.ms_path), mode="save", versionname="tfcrop_only")

    # Restore to original
    flagmanager(vis=str(args.ms_path), mode="restore", versionname="original")

    # Run rflag on original
    flagdata(vis=str(args.ms_path), mode="rflag", datacolumn="data", action="apply")
    flagmanager(vis=str(args.ms_path), mode="save", versionname="rflag_only")

    # Restore to original
    flagmanager(vis=str(args.ms_path), mode="restore", versionname="original")

    # Run tfcrop+rflag sequentially (combined)
    flagdata(vis=str(args.ms_path), mode="tfcrop", datacolumn="data", action="apply")
    flagdata(vis=str(args.ms_path), mode="rflag", datacolumn="data", action="apply")
    flagmanager(vis=str(args.ms_path), mode="save", versionname="tfcrop")

    # Load predictions
    print("\n[7/8] Loading predictions...")

    # SAM-RFI flags
    flagmanager(vis=str(args.ms_path), mode="restore", versionname="samrfi")
    loader = MSLoader(str(args.ms_path))
    loader.load(mode="DATA")
    sam_flags = loader.load_flags()
    loader.close()

    # tfcrop+rflag flags
    flagmanager(vis=str(args.ms_path), mode="restore", versionname="tfcrop")
    loader = MSLoader(str(args.ms_path))
    loader.load(mode="DATA")
    tfcrop_flags = loader.load_flags()
    loader.close()

    # Create comparison plot
    print("\n[7/8] Creating comparison plot...")
    plot_comparison(synthetic_data, ground_truth, sam_flags, tfcrop_flags, output_dir)

    # Compute metrics (flatten across baselines and pols)
    print("\n[8/8] Computing metrics...")
    sam_flat = sam_flags.reshape(-1)
    tfcrop_flat = tfcrop_flags.reshape(-1)
    gt_flat = ground_truth.reshape(-1)

    # Segmentation metrics (vs ground truth)
    sam_seg_metrics = evaluate_segmentation(sam_flat, gt_flat)
    tfcrop_seg_metrics = evaluate_segmentation(tfcrop_flat, gt_flat)

    # Statistical metrics (on actual data)
    synthetic_data_flat = synthetic_data.reshape(-1)
    sam_stats = compute_statistics(synthetic_data_flat, sam_flat)
    tfcrop_stats = compute_statistics(synthetic_data_flat, tfcrop_flat)

    # Flagging Fidelity Index
    sam_ffi = compute_ffi(synthetic_data_flat, sam_flat)
    tfcrop_ffi = compute_ffi(synthetic_data_flat, tfcrop_flat)

    # Calcquality metric
    sam_calcq = compute_calcquality(synthetic_data_flat, sam_flat)
    tfcrop_calcq = compute_calcquality(synthetic_data_flat, tfcrop_flat)

    # Print
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    print("\nSAM-RFI:")
    print("  Segmentation Metrics (vs Ground Truth):")
    for k, v in sam_seg_metrics.items():
        print(f"    {k}: {v:.4f}")
    print("  Statistical Metrics:")
    for k, v in sam_stats.items():
        if k != "count":
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    print("  Flagging Fidelity Index:")
    for k, v in sam_ffi.items():
        print(f"    {k}: {v:.4f}")
    print("  Calcquality:")
    print(f"    calcquality: {sam_calcq['calcquality']:.4f}")
    print(f"    sensitivity: {sam_calcq['sensitivity']:.4f}")
    print(f"    mean_shift: {sam_calcq['mean_shift']:.4f}")
    print(f"    std_shift: {sam_calcq['std_shift']:.4f}")
    print(f"    overflagging_penalty: {sam_calcq['overflagging_penalty']:.4f}")

    print("\ntfcrop + rflag:")
    print("  Segmentation Metrics (vs Ground Truth):")
    for k, v in tfcrop_seg_metrics.items():
        print(f"    {k}: {v:.4f}")
    print("  Statistical Metrics:")
    for k, v in tfcrop_stats.items():
        if k != "count":
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    print("  Flagging Fidelity Index:")
    for k, v in tfcrop_ffi.items():
        print(f"    {k}: {v:.4f}")
    print("  Calcquality:")
    print(f"    calcquality: {tfcrop_calcq['calcquality']:.4f}")
    print(f"    sensitivity: {tfcrop_calcq['sensitivity']:.4f}")
    print(f"    mean_shift: {tfcrop_calcq['mean_shift']:.4f}")
    print(f"    std_shift: {tfcrop_calcq['std_shift']:.4f}")
    print(f"    overflagging_penalty: {tfcrop_calcq['overflagging_penalty']:.4f}")

    # Save to JSON
    results = {
        "config": args.config,
        "ms_path": str(args.ms_path),
        "ground_truth_path": str(ground_truth_path),
        "dimensions": {
            "baselines": num_baselines,
            "channels": num_channels,
            "times": num_times,
        },
        "ground_truth_rfi_percent": float(ground_truth.sum() / ground_truth.size * 100),
        "sam_rfi": {
            "segmentation": sam_seg_metrics,
            "statistics": sam_stats,
            "ffi": sam_ffi,
            "calcquality": sam_calcq,
        },
        "tfcrop_rflag": {
            "segmentation": tfcrop_seg_metrics,
            "statistics": tfcrop_stats,
            "ffi": tfcrop_ffi,
            "calcquality": tfcrop_calcq,
        },
    }

    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(convert_numpy_types(results), f, indent=2)
    print(f"\n✓ Saved JSON: {json_path}")

    # Save to CSV
    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "category", "metric", "value"])

        # SAM-RFI metrics
        for k, v in sam_seg_metrics.items():
            writer.writerow(["sam_rfi", "segmentation", k, v])
        for k, v in sam_stats.items():
            if k != "count":
                writer.writerow(["sam_rfi", "statistics", k, v])
        for k, v in sam_ffi.items():
            writer.writerow(["sam_rfi", "ffi", k, v])
        for k, v in sam_calcq.items():
            if k != "components":
                writer.writerow(["sam_rfi", "calcquality", k, v])

        # tfcrop+rflag metrics
        for k, v in tfcrop_seg_metrics.items():
            writer.writerow(["tfcrop_rflag", "segmentation", k, v])
        for k, v in tfcrop_stats.items():
            if k != "count":
                writer.writerow(["tfcrop_rflag", "statistics", k, v])
        for k, v in tfcrop_ffi.items():
            writer.writerow(["tfcrop_rflag", "ffi", k, v])
        for k, v in tfcrop_calcq.items():
            if k != "components":
                writer.writerow(["tfcrop_rflag", "calcquality", k, v])

    print(f"✓ Saved CSV: {csv_path}")

    print("\n" + "=" * 60)
    print("✓ Benchmark complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
