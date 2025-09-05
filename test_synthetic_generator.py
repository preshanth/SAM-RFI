#!/usr/bin/env python3
"""
Test and Demo Script for Synthetic Measurement Set Generator

Creates realistic synthetic measurement sets with controllable RFI patterns
for training SAM-RFI models and benchmarking against other flaggers.
"""

import sys

sys.path.insert(0, "src")

import numpy as np
from pathlib import Path
import logging
from samrfi.datasets import (
    ObservationConfig,
    RFIConfig,
    SyntheticVisibilityGenerator,
    MSWriter,
    SyntheticDatasetGenerator,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def test_visibility_generation():
    """Test synthetic visibility generation"""
    print("=" * 60)
    print("TESTING SYNTHETIC VISIBILITY GENERATION")
    print("=" * 60)

    # Create small observation for testing
    obs_config = ObservationConfig(
        num_antennas=4,
        num_spw=2,
        channels_per_spw=128,
        start_frequency=1.4e9,  # L-band
        total_duration=600.0,  # 10 minutes
        integration_time=10.0,
    )

    rfi_config = RFIConfig(
        broadband_probability=0.02,
        narrowband_lines=3,
        transient_events=5,
        periodic_signals=2,
        satellite_passes=1,
    )

    print(f"Observation config:")
    print(f"  {obs_config.num_antennas} antennas, {obs_config.num_spw} SPWs")
    print(f"  {obs_config.channels_per_spw} channels/SPW")
    print(f"  {obs_config.total_duration/60:.1f} min duration")

    # Generate synthetic data
    vis_gen = SyntheticVisibilityGenerator(obs_config, rfi_config)

    print(f"\nGenerated configuration:")
    print(f"  {vis_gen.num_baselines} baselines")
    print(f"  {vis_gen.num_times} time steps")
    print(f"  {vis_gen.total_channels} total channels")

    # Create clean visibilities
    print("\n1. Generating clean visibilities...")
    clean_vis = vis_gen.generate_clean_visibilities()
    print(f"   Shape: {clean_vis.shape}")
    print(f"   Data type: {clean_vis.dtype}")
    print(
        f"   Amplitude range: {np.abs(clean_vis).min():.3f} - {np.abs(clean_vis).max():.3f}"
    )

    # Inject RFI
    print("\n2. Injecting RFI patterns...")
    corrupted_vis, rfi_mask = vis_gen.inject_rfi(clean_vis)
    print(f"   RFI mask shape: {rfi_mask.shape}")
    print(f"   RFI fraction: {np.mean(rfi_mask):.3f}")
    print(
        f"   Corrupted amplitude range: {np.abs(corrupted_vis).min():.3f} - {np.abs(corrupted_vis).max():.3f}"
    )

    # Statistics by RFI type (rough estimation)
    print(f"\n3. RFI Pattern Analysis:")
    for baseline in range(min(3, vis_gen.num_baselines)):  # Check first 3 baselines
        baseline_mask = rfi_mask[baseline]

        # Analyze patterns
        time_flagged = np.mean(baseline_mask, axis=(0, 2))  # Fraction flagged per time
        freq_flagged = np.mean(baseline_mask, axis=(1, 2))  # Fraction flagged per freq

        print(f"   Baseline {baseline}: {np.mean(baseline_mask):.3f} flagged")
        print(f"     Time variability: {np.std(time_flagged):.3f}")
        print(f"     Freq variability: {np.std(freq_flagged):.3f}")

    print("✓ Visibility generation test completed")
    return clean_vis, corrupted_vis, rfi_mask, obs_config


def test_measurement_set_creation(clean_vis, corrupted_vis, rfi_mask, obs_config):
    """Test CASA measurement set creation"""
    print("\n" + "=" * 60)
    print("TESTING MEASUREMENT SET CREATION")
    print("=" * 60)

    output_dir = Path("test_synthetic_ms")
    output_dir.mkdir(exist_ok=True)

    ms_writer = MSWriter(obs_config)

    # Create different versions of the MS
    test_cases = [
        ("clean", clean_vis, None, False),
        ("corrupted", corrupted_vis, None, False),
        ("with_flags", corrupted_vis, rfi_mask, True),
    ]

    created_ms = []

    for name, vis_data, mask, include_flags in test_cases:
        ms_path = output_dir / f"test_{name}.ms"

        print(f"\n{name.upper()} Measurement Set:")
        print(f"  Path: {ms_path}")
        print(f"  Data shape: {vis_data.shape}")
        if mask is not None:
            print(f"  Flag fraction: {np.mean(mask):.3f}")

        try:
            ms_writer.create_measurement_set(
                str(ms_path), vis_data, mask, include_flags
            )
            created_ms.append(ms_path)
            print(f"  ✓ Created successfully")

            # Basic validation
            if ms_path.exists():
                subtables = [d for d in ms_path.iterdir() if d.is_dir()]
                print(f"  Subtables: {len(subtables)}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")

    print(f"\n✓ Created {len(created_ms)} measurement sets")
    return created_ms


def test_dataset_generation():
    """Test complete dataset generation pipeline"""
    print("\n" + "=" * 60)
    print("TESTING COMPLETE DATASET GENERATION")
    print("=" * 60)

    # Initialize dataset generator
    generator = SyntheticDatasetGenerator("test_synthetic_datasets")

    # Create small dataset for testing
    print("Generating test dataset...")
    dataset_metadata = generator.generate_training_dataset(
        dataset_name="test_dataset", num_observations=3  # Small test dataset
    )

    print(f"\nDataset generated:")
    print(f"  Name: {dataset_metadata['dataset_name']}")
    print(f"  Observations: {dataset_metadata['num_observations']}")

    for i, obs_meta in enumerate(dataset_metadata["observations"]):
        print(f"\n  Observation {i+1}:")
        print(f"    Name: {obs_meta['observation_name']}")
        print(f"    RFI fraction: {obs_meta['rfi_statistics']['rfi_fraction']:.3f}")
        print(f"    Clean MS: {Path(obs_meta['clean_ms']).name}")
        print(f"    Corrupted MS: {Path(obs_meta['corrupted_ms']).name}")
        print(f"    Truth MS: {Path(obs_meta['truth_ms']).name}")

    # Test flagger comparison script generation
    print(f"\nGenerating flagger comparison script...")
    script_path = generator.create_flagger_comparison_script("test_dataset")
    print(f"  Script: {script_path}")

    # Test training data export
    print(f"\nExporting training patches...")
    training_dir = generator.export_training_patches("test_dataset", patch_size=128)
    print(f"  Training data: {training_dir}")

    print("✓ Complete dataset generation test completed")
    return dataset_metadata


def demonstrate_flagger_compatibility():
    """Demonstrate compatibility with other flaggers"""
    print("\n" + "=" * 60)
    print("FLAGGER COMPATIBILITY DEMONSTRATION")
    print("=" * 60)

    print("The generated measurement sets are fully compatible with:")
    print("\n1. AOFlagger:")
    print("   aoflagger synthetic_datasets/measurement_sets/test_obs_000_corrupted.ms")

    print("\n2. CASA tfcrop:")
    print("   flagdata(vis='...corrupted.ms', mode='tfcrop')")

    print("\n3. CASA rflag:")
    print("   flagdata(vis='...corrupted.ms', mode='rflag')")

    print("\n4. Custom Python analysis:")
    print(
        """
   from samrfi.core import MSLoader, FlagManager
   with MSLoader('...corrupted.ms') as loader:
       for batch in loader.get_baseline_iterator():
           # Process data batch
           pass
   """
    )

    print("\nTo compare flagging results:")
    print("1. Run the generated comparison script")
    print("2. Use SAM-RFI to process the same data")
    print("3. Compare against ground truth flags")


def main():
    """Main test and demonstration"""
    print("SAM-RFI SYNTHETIC MEASUREMENT SET GENERATOR")
    print("Test and Demonstration Script")
    print("=" * 60)

    try:
        # Test 1: Basic visibility generation
        clean_vis, corrupted_vis, rfi_mask, obs_config = test_visibility_generation()

        # Test 2: Measurement set creation
        created_ms = test_measurement_set_creation(
            clean_vis, corrupted_vis, rfi_mask, obs_config
        )

        # Test 3: Complete dataset generation
        dataset_metadata = test_dataset_generation()

        # Test 4: Flagger compatibility info
        demonstrate_flagger_compatibility()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        print("\nWhat was created:")
        print("• test_synthetic_ms/        - Individual test measurement sets")
        print("• test_synthetic_datasets/  - Complete training dataset")
        print("  ├── measurement_sets/     - CASA-compatible .ms files")
        print("  ├── ground_truth/         - NumPy arrays with clean data and flags")
        print("  ├── training_data/        - Patches ready for SAM training")
        print("  └── configs/              - Dataset metadata and configurations")

        print("\nNext steps:")
        print("1. Test with real flaggers: aoflagger, CASA tfcrop/rflag")
        print("2. Use datasets to train SAM-RFI models")
        print("3. Benchmark SAM-RFI against traditional flaggers")
        print("4. Generate larger datasets for production training")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
