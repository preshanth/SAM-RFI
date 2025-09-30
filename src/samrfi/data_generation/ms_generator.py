"""
MS Data Generator - Generate training data from measurement sets
"""

import os
import json
from pathlib import Path
from tqdm import tqdm

from samrfi.data import MSLoader, Preprocessor


class MSDataGenerator:
    """
    Generate SAM2 training datasets from CASA measurement sets

    Workflow:
        1. Load MS file → complex visibilities
        2. Extract magnitude → waterfall plots
        3. Patchify with 4-way rotation augmentation
        4. Normalize + stretch (SQRT/LOG10)
        5. Generate ground truth masks (MAD or custom flags)
        6. Save HuggingFace dataset to disk
    """

    def __init__(self, config):
        """
        Initialize MS data generator

        Args:
            config: Configuration object with MS and processing parameters
        """
        self.config = config

    def generate(self, output_path):
        """
        Generate dataset from measurement set

        Args:
            output_path: Directory to save generated dataset

        Returns:
            Path to saved dataset
        """
        print("=" * 60)
        print("MS Data Generation")
        print("=" * 60)

        # Validate MS path
        ms_path = self.config.ms.get("path")
        if not ms_path:
            raise ValueError("MS path not specified in config")

        if not Path(ms_path).exists():
            raise FileNotFoundError(f"Measurement set not found: {ms_path}")

        print(f"\nMeasurement Set: {ms_path}")
        print(f"Output Path: {output_path}")

        # Load MS data
        print("\n[1/4] Loading measurement set...")
        loader = MSLoader(ms_path)

        num_antennas = self.config.ms.get("num_antennas", None)
        data_mode = self.config.ms.get("data_mode", "DATA")

        loader.load(num_antennas=num_antennas, mode=data_mode)

        print(f"  Loaded shape: {loader.data.shape}")
        print(f"  (baselines, polarizations, channels, time)")

        # Load flags if using custom flags
        proc_config = self.config.processing
        use_custom_flags = proc_config.get("custom_flag", True)
        flags = None

        if use_custom_flags:
            print("\n[2/4] Loading MS flags...")
            flags = loader.load_flags()

        # Create dataset
        print("\n[3/4] Preprocessing data...")
        preprocessor = Preprocessor(loader.magnitude, flags=flags)

        dataset = preprocessor.create_dataset(
            patch_size=proc_config.get("patch_size", 128),
            stretch=proc_config.get("stretch", "SQRT"),
            flag_sigma=proc_config.get("flag_sigma", 5),
            use_custom_flags=use_custom_flags,
            num_patches=proc_config.get("num_patches", None),
            apply_stretching=proc_config.get("apply_stretching", True),
        )

        num_patches = len(dataset)
        print(f"  Generated {num_patches} patches")
        print(
            f"  Patch size: {proc_config.get('patch_size', 128)}x{proc_config.get('patch_size', 128)}"
        )
        print(f"  Stretch: {proc_config.get('stretch', 'SQRT')}")
        print(f"  Flag sigma: {proc_config.get('flag_sigma', 5)}")

        # Save dataset
        print("\n[4/4] Saving dataset...")
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save dataset using HuggingFace format
        dataset.save_to_disk(str(output_dir))

        # Save metadata
        metadata = {
            "source": "measurement_set",
            "ms_path": str(ms_path),
            "num_antennas": self.config.ms.get("num_antennas"),
            "data_mode": self.config.ms.get("data_mode", "DATA"),
            "num_patches": num_patches,
            "patch_size": proc_config.get("patch_size", 128),
            "stretch": proc_config.get("stretch", "SQRT"),
            "flag_sigma": proc_config.get("flag_sigma", 5),
            "custom_flag": proc_config.get("custom_flag", True),
            "augmentation": {"rotations": "four_way"},
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Dataset saved to: {output_dir}")
        print(f"  Metadata saved to: {metadata_path}")

        # Statistics
        print("\nDataset Statistics:")
        print(f"  Total patches: {num_patches}")
        print(
            f"  Image shape: {proc_config.get('patch_size', 128)}x{proc_config.get('patch_size', 128)}x3 (RGB)"
        )
        print(
            f"  Mask shape: {proc_config.get('patch_size', 128)}x{proc_config.get('patch_size', 128)} (binary)"
        )
        print(f"  Format: HuggingFace Dataset")

        print("\n" + "=" * 60)
        print("✓ Data generation complete!")
        print("=" * 60)

        return str(output_dir)
