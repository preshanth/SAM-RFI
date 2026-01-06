"""
MS data generator for SAM-RFI training datasets.

This module provides functionality to generate SAM2 training datasets from
CASA measurement sets (MS). It handles loading MS files, extracting complex
visibilities, preprocessing data through normalization and stretching, and
saving datasets in batched PyTorch format.

Classes
-------
MSDataGenerator
    Generate SAM2 training datasets from CASA measurement sets.

Examples
--------
>>> from samrfi.data_generation import MSDataGenerator
>>> from samrfi.config import ConfigLoader
>>>
>>> # Load configuration
>>> config = ConfigLoader.load_data('ms_config.yaml')
>>>
>>> # Generate dataset
>>> generator = MSDataGenerator(config)
>>> dataset_path = generator.generate('./output/ms_dataset')
>>> print(f"Dataset saved to: {dataset_path}")
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from samrfi.data import Preprocessor
from samrfi.data.ms_loader import MSLoader


class MSDataGenerator:
    """
    Generate SAM2 training datasets from CASA measurement sets.

    This class implements a complete pipeline for converting CASA measurement sets
    into training-ready datasets for SAM2 model training. The pipeline includes:

    1. Load MS file and extract complex visibilities
    2. Extract magnitude data as waterfall plots
    3. Patchify data with 4-way rotation augmentation
    4. Apply normalization and stretching (SQRT/LOG10)
    5. Generate ground truth masks (MAD or custom flags)
    6. Save BatchedDataset to disk (batch_*.pt files)

    Parameters
    ----------
    config : DataConfig
        Configuration object containing MS path, processing parameters,
        and output settings. Expected structure:

        - ms.path : str - Path to measurement set
        - ms.num_antennas : int, optional - Number of antennas to load
        - ms.data_mode : str - Data column ('DATA' or 'CORRECTED_DATA')
        - processing.patch_size : int - Patch size (128, 256, 512, 1024)
        - processing.stretch : str or None - Stretching method ('SQRT', 'LOG10', or None)
        - processing.flag_sigma : int - Sigma threshold for MAD flagging
        - processing.custom_flag : bool - Use MS flags as ground truth
        - processing.num_patches : int, optional - Maximum patches to generate
        - processing.num_workers : int - Number of parallel workers

    Attributes
    ----------
    config : DataConfig
        Stored configuration object.

    Examples
    --------
    >>> from samrfi.config import ConfigLoader
    >>> from samrfi.data_generation import MSDataGenerator
    >>>
    >>> # Load configuration from YAML
    >>> config = ConfigLoader.load_data('configs/ms_gen.yaml')
    >>>
    >>> # Create generator
    >>> generator = MSDataGenerator(config)
    >>>
    >>> # Generate dataset
    >>> output_path = generator.generate('./output/ms_dataset')
    >>> print(f"Dataset saved: {output_path}")
    Dataset saved: ./output/ms_dataset

    Notes
    -----
    The generator uses BatchWriter to save datasets in batched format
    (batch_*.pt files), which enables memory-efficient loading during training.
    Ground truth masks can come from either MS flags (custom_flag=True) or
    MAD-based automatic flagging (custom_flag=False).
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize MS data generator.

        Parameters
        ----------
        config : DataConfig
            Configuration object with MS and processing parameters.
        """
        self.config = config

    def generate(self, output_path: str) -> str:
        """
        Generate dataset from measurement set.

        This method performs the complete data generation pipeline:
        1. Validates MS path
        2. Loads MS data using MSLoader
        3. Optionally loads MS flags for ground truth
        4. Preprocesses data (patchify, normalize, stretch)
        5. Saves dataset in batched format
        6. Generates metadata JSON files

        Parameters
        ----------
        output_path : str
            Directory path where generated dataset will be saved.
            Will be created if it doesn't exist.

        Returns
        -------
        str
            Absolute path to the saved dataset directory.

        Raises
        ------
        ValueError
            If MS path is not specified in config.
        FileNotFoundError
            If measurement set doesn't exist at specified path.

        Examples
        --------
        >>> generator = MSDataGenerator(config)
        >>> dataset_path = generator.generate('./datasets/my_ms_data')
        ==========================================
        MS Data Generation
        ==========================================
        ...
        ✓ Data generation complete!

        Notes
        -----
        The output directory will contain:
        - batch_*.pt : PyTorch batched dataset files
        - metadata.json : Dataset metadata (source, parameters, statistics)

        The metadata includes MS path, number of antennas, patch size,
        stretching method, and augmentation details.
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
        print("  (baselines, polarizations, channels, time)")

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
            stretch=proc_config.get("stretch", None),
            flag_sigma=proc_config.get("flag_sigma", 5),
            use_custom_flags=use_custom_flags,
            num_patches=proc_config.get("num_patches", None),
            normalize_before_stretch=proc_config.get("normalize_before_stretch", True),
            normalize_after_stretch=proc_config.get("normalize_after_stretch", False),
            num_workers=proc_config.get("num_workers", 4),
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

        # Save dataset using BatchWriter (BatchedDataset format)
        from samrfi.data.torch_dataset import BatchWriter

        writer = BatchWriter(output_dir, samples_per_batch=100)
        writer.add_dataset(dataset)
        writer.finalize()

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
        print("  Format: BatchedDataset (batch_*.pt files)")

        print("\n" + "=" * 60)
        print("✓ Data generation complete!")
        print("=" * 60)

        return str(output_dir)
