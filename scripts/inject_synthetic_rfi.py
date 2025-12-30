#!/usr/bin/env python3
"""
Inject Synthetic RFI into Real MS

Takes a real MS and injects synthetic RFI with known ground truth.
Designed for validation: inject 40% bandwidth with mixed RFI types.

Usage:
    python scripts/inject_synthetic_rfi.py \
        --input-ms one_antenna_3C219_sqrt.ms \
        --config configs/validation.yaml \
        --rfi-fraction 0.4 \
        --output-dir ./synthetic_injection
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from samrfi.config import ConfigLoader
from samrfi.data import MSLoader
from samrfi.data_generation import SyntheticDataGenerator
from samrfi.evaluation import inject_synthetic_data


def inject_rfi_into_ms(input_ms, config, output_dir="./synthetic_injection"):
    """
    Inject synthetic RFI into real MS for validation.

    Args:
        input_ms: Path to input MS
        config: Loaded config
        output_dir: Output directory

    Returns:
        Dict with paths and metadata
    """
    input_ms = Path(input_ms)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Synthetic RFI Injection for Validation")
    print(f"{'='*70}")
    print(f"Input MS: {input_ms}")

    # Load MS to get dimensions
    print(f"\n[1/4] Loading MS structure...")
    loader = MSLoader(input_ms)
    loader.load(mode="DATA")

    baselines, pols, channels, times = loader.data.shape
    print(f"  MS shape: {loader.data.shape}")
    print(f"  Baselines: {baselines}")
    print(f"  Pols: {pols}")
    print(f"  Channels: {channels}")
    print(f"  Times: {times}")

    baseline_map = loader.antenna_baseline_map
    loader.close()

    # Generate synthetic RFI
    print(f"\n[2/4] Generating synthetic RFI ({baselines} baselines)...")

    synth_config = config.synthetic
    generator = SyntheticDataGenerator(config)

    # Override config to match MS dimensions
    gen_kwargs = {
        "num_channels": channels,
        "num_times": times,
        "noise_level": synth_config.get("noise_mjy", 1.0),
        "rfi_power_min": synth_config.get("rfi_power_min", 1000.0),
        "rfi_power_max": synth_config.get("rfi_power_max", 10000.0),
        "rfi_config": generator._parse_rfi_config(synth_config),
        "enable_bandpass": synth_config.get("enable_bandpass_rolloff", False),
        "bandpass_order": synth_config.get("bandpass_polynomial_order", 8),
        "num_polarizations": pols,
        "pol_corr": synth_config.get("polarization_correlation", 0.8),
        "synth_config": synth_config,
    }

    all_waterfalls = []
    all_ground_truth = []

    for baseline_idx in tqdm(range(baselines), desc="Generating"):
        waterfall, ground_truth, _ = generator._generate_single_sample(**gen_kwargs)
        all_waterfalls.append(waterfall[0])
        all_ground_truth.append(ground_truth[0])

    full_waterfall = np.stack(all_waterfalls)
    full_ground_truth = np.stack(all_ground_truth)

    # Verify RFI fraction
    actual_rfi_fraction = np.sum(full_ground_truth) / full_ground_truth.size
    print(f"  Generated RFI: {actual_rfi_fraction*100:.1f}% of data")

    # Save ground truth
    gt_path = output_dir / "ground_truth.npy"
    np.save(gt_path, full_ground_truth)
    print(f"  ✓ Saved ground truth: {gt_path}")

    # Create working MS with injection
    print(f"\n[3/4] Creating working MS with synthetic RFI...")
    work_ms = output_dir / "synthetic_rfi.ms"

    if work_ms.exists():
        shutil.rmtree(work_ms)

    shutil.copytree(input_ms, work_ms)
    print(f"  Copied: {input_ms} → {work_ms}")

    # Inject synthetic data
    print(f"\n[4/4] Injecting synthetic RFI into MS...")
    inject_synthetic_data(
        template_ms_path=work_ms,
        synthetic_data=full_waterfall,
        output_ms_path=work_ms,
        baseline_map=baseline_map,
    )

    # Save metadata
    metadata = {
        "input_ms": str(input_ms),
        "work_ms": str(work_ms),
        "ground_truth": str(gt_path),
        "shape": list(full_waterfall.shape),
        "rfi_fraction_actual": float(actual_rfi_fraction),
        "baselines": baselines,
        "baseline_map": baseline_map,
    }

    metadata_path = output_dir / "injection_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print("✓ Synthetic RFI Injection Complete")
    print(f"{'='*70}")
    print(f"  Working MS: {work_ms}")
    print(f"  Ground truth: {gt_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"  RFI injected: {actual_rfi_fraction*100:.1f}% of data")
    print(f"{'='*70}\n")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Inject synthetic RFI into real MS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--input-ms", required=True, help="Path to input MS")
    parser.add_argument("--config", required=True, help="Path to validation config")
    parser.add_argument(
        "--output-dir", default="./synthetic_injection", help="Output directory"
    )

    args = parser.parse_args()

    # Load config
    config = ConfigLoader.load_data(args.config)

    # Inject RFI
    metadata = inject_rfi_into_ms(
        input_ms=args.input_ms,
        config=config,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
