"""
Synthetic RFI data generator for SAM-RFI training datasets.

This module provides functionality to generate SAM2 training datasets from
synthetic RFI simulations with exact ground truth masks. It supports physically
realistic RFI types including narrowband/broadband persistent signals, intermittent
periodic signals, random bursts, and frequency sweeps. The generator creates
datasets with 6 orders of magnitude dynamic range (1 mJy noise to 1000 Jy RFI).

Classes
-------
RawPatchDataset
    Simple container for raw complex patches without preprocessing.
SyntheticDataGenerator
    Generate SAM2 training datasets from synthetic RFI simulations.

Functions
---------
_init_worker
    Initialize worker process with generator instance for multiprocessing.
_worker_generate_and_preprocess
    Worker function to generate and optionally preprocess one sample.

Examples
--------
>>> from samrfi.data_generation import SyntheticDataGenerator
>>> from samrfi.config import ConfigLoader
>>>
>>> # Load configuration
>>> config = ConfigLoader.load_data('synthetic_config.yaml')
>>>
>>> # Generate synthetic dataset
>>> generator = SyntheticDataGenerator(config)
>>> dataset_path = generator.generate('./output/synthetic_dataset')
>>> print(f"Dataset with exact ground truth saved to: {dataset_path}")

Notes
-----
Physical parameters used for realistic RFI simulation:
- Noise level: ~1 mJy (milli-Jansky)
- RFI power: ~1000-10000 Jy (6 orders of magnitude above noise)
- Bandpass rolloff: 8th order polynomial edge effects (optional)
- Polarization correlation: Correlated RFI across polarizations (default 0.8)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from samrfi.data import Preprocessor

# Global generator instance for multiprocessing workers
_global_generator = None
_global_proc_config = None


class RawPatchDataset:
    """
    Simple container for raw complex patches without preprocessing.

    This class provides a minimal dataset interface compatible with BatchWriter
    for storing raw complex visibility data before channel extraction and
    augmentation. Used when save_raw=True in configuration.

    Parameters
    ----------
    complex_patches : torch.Tensor
        Complex patches tensor with shape (N, H, W) and dtype complex64,
        where N is number of patches, H is height, W is width.
    masks : torch.Tensor
        Binary masks tensor with shape (N, H, W) and dtype uint8,
        indicating RFI locations (1=RFI, 0=clean).

    Attributes
    ----------
    images : torch.Tensor
        Stored complex patches (named for BatchWriter compatibility).
    labels : torch.Tensor
        Stored binary masks (named for BatchWriter compatibility).

    Examples
    --------
    >>> import torch
    >>> patches = torch.randn(10, 128, 128, dtype=torch.complex64)
    >>> masks = torch.randint(0, 2, (10, 128, 128), dtype=torch.uint8)
    >>> dataset = RawPatchDataset(patches, masks)
    >>> len(dataset)
    10

    Notes
    -----
    The .images attribute contains raw complex data, not RGB images.
    Channel extraction (gradient, log_amp, phase) happens during training
    when using GPUDataset with on-the-fly transforms.
    """

    def __init__(
        self, complex_patches: torch.Tensor, masks: torch.Tensor
    ) -> None:
        """
        Initialize raw patch dataset.

        Parameters
        ----------
        complex_patches : torch.Tensor
            Complex patches with shape (N, H, W) and dtype complex64.
        masks : torch.Tensor
            Binary masks with shape (N, H, W) and dtype uint8.
        """
        # Use .images and .labels for BatchWriter compatibility
        self.images = complex_patches  # Raw complex data (not RGB images)
        self.labels = masks

    def __len__(self) -> int:
        """
        Get number of patches in dataset.

        Returns
        -------
        int
            Number of patches.
        """
        return len(self.images)


def _init_worker(config_dict: Dict[str, Any]) -> None:
    """
    Initialize worker process with generator instance for multiprocessing.

    This function is called once per worker process to create a global
    SyntheticDataGenerator instance. Required for multiprocessing Pool
    to avoid pickling the generator on every task.

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary (serializable for pickling).
        Will be converted to SimpleNamespace for attribute access.

    Notes
    -----
    Sets global variables:
    - _global_generator : SyntheticDataGenerator instance
    - _global_proc_config : Processing configuration dict

    This is an internal function used by multiprocessing.Pool.
    """
    global _global_generator, _global_proc_config

    from types import SimpleNamespace

    # Convert dict to namespace
    def dict_to_namespace(d: Any) -> Any:
        """Recursively convert dict to SimpleNamespace."""
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return d

    config = dict_to_namespace(config_dict)
    _global_generator = SyntheticDataGenerator(config)
    _global_proc_config = config_dict.get("processing", {})


def _worker_generate_and_preprocess(**gen_kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
    """
    Worker function to generate and optionally preprocess one sample.

    This function is executed by each worker process in the multiprocessing pool.
    It generates one synthetic waterfall sample, then either saves it as raw
    complex data or preprocesses it (patchify, augment, normalize, stretch).

    Parameters
    ----------
    **gen_kwargs : dict
        Keyword arguments for _generate_single_sample including:
        - num_channels : int - Number of frequency channels
        - num_times : int - Number of time samples
        - noise_level : float or tuple - Noise level in mJy
        - rfi_power_min : float or tuple - Minimum RFI power in Jy
        - rfi_power_max : float or tuple - Maximum RFI power in Jy
        - rfi_config : dict - RFI type configuration
        - enable_bandpass : bool - Enable bandpass rolloff
        - bandpass_order : int - Polynomial order for bandpass
        - num_polarizations : int - Number of polarizations
        - pol_corr : float - Polarization correlation coefficient
        - synth_config : object - Full synthetic configuration

    Returns
    -------
    dataset : RawPatchDataset or Dataset
        If save_raw=True, returns RawPatchDataset with complex patches.
        Otherwise, returns preprocessed Dataset with RGB images.
    rfi_params : dict
        Dictionary of RFI parameters for this sample (for metadata tracking).

    Notes
    -----
    Uses global variables set by _init_worker:
    - _global_generator : SyntheticDataGenerator instance
    - _global_proc_config : Processing configuration dict

    This is an internal function used by multiprocessing.Pool.
    """
    global _global_generator, _global_proc_config

    # Generate one sample
    waterfall, exact_mask, rfi_params = _global_generator._generate_single_sample(**gen_kwargs)

    # Check if we should save raw or preprocessed
    save_raw = _global_proc_config.get("save_raw", False)

    if save_raw:
        # Save raw complex patches (no preprocessing, no augmentation)
        # Waterfall shape: (1, num_pols, channels, times)
        # For patch_size = full size, just squeeze and convert to tensor

        # Average polarizations (taking magnitude for complex data)
        # Shape: (num_pols, channels, times) -> (channels, times)
        waterfall_squeezed = waterfall.squeeze(0)  # Remove baseline dimension

        # Take magnitude of complex values, then average across polarizations
        magnitude = np.abs(waterfall_squeezed)  # (num_pols, channels, times)
        averaged = magnitude.mean(axis=0).astype(np.float32)  # (channels, times)

        # Average masks across polarizations (use max to preserve RFI flags)
        mask_averaged = exact_mask.squeeze(0).max(axis=0).astype(np.uint8)  # (channels, times)

        # Convert to tensors (add batch dim)
        complex_patches = torch.from_numpy(averaged).unsqueeze(0)  # (1, H, W)
        masks = torch.from_numpy(mask_averaged).unsqueeze(0)  # (1, H, W)

        dataset = RawPatchDataset(complex_patches, masks)
    else:
        # Preprocess with augmentation (original behavior)
        preprocessor = Preprocessor(waterfall, flags=exact_mask)
        dataset = preprocessor.create_dataset(
            patch_size=_global_proc_config.get("patch_size", 128),
            stretch=_global_proc_config.get("stretch", None),
            flag_sigma=_global_proc_config.get("flag_sigma", 5),
            use_custom_flags=True,
            num_patches=_global_proc_config.get("num_patches", None),
            normalize_before_stretch=_global_proc_config.get("normalize_before_stretch", True),
            normalize_after_stretch=_global_proc_config.get("normalize_after_stretch", False),
            num_workers=0,  # No nested parallelism
            enable_augmentation=_global_proc_config.get("enable_augmentation", True),
            augmentation_rotations=_global_proc_config.get("augmentation_rotations", 4),
        )

    return dataset, rfi_params


class SyntheticDataGenerator:
    """
    Generate SAM2 training datasets from synthetic RFI simulations.

    This class provides a complete pipeline for generating synthetic radio
    interferometry data with physically realistic RFI signals and exact ground
    truth masks. The generator supports multiple RFI types with configurable
    parameters and outputs datasets ready for SAM2 training.

    Workflow
    --------
    1. Generate synthetic waterfall plots with realistic RFI types
    2. Add physically accurate RFI (6 orders of magnitude above noise)
    3. Generate EXACT ground truth masks (we know where RFI is!)
    4. Patchify with 4-way rotation augmentation
    5. Normalize and stretch (SQRT/LOG10)
    6. Save batched PyTorch dataset to disk

    RFI Types Supported
    -------------------
    - Narrowband persistent: GPS, cell towers, satellites
    - Broadband persistent: Lightning, power lines
    - Narrowband intermittent (periodic): Rotating radar
    - Narrowband bursty (random): Random pulsed transmitters
    - Broadband bursty (random): Lightning strikes
    - Frequency sweeps: Radar chirps, satellite drift (linear or quadratic)

    Physical Realism
    ----------------
    - Noise: ~1 mJy (milli-Jansky)
    - RFI: ~1000-10000 Jy (6 orders of magnitude higher)
    - Bandpass rolloff: 8th order polynomial edge effects (optional)
    - Polarization correlation: Correlated RFI across XX/YY (default 0.8)

    Parameters
    ----------
    config : DataConfig
        Configuration object with synthetic RFI parameters. Expected structure:

        - synthetic.num_samples : int - Number of waterfall samples to generate
        - synthetic.num_channels : int - Number of frequency channels (e.g., 2048)
        - synthetic.num_times : int - Number of time samples (e.g., 512)
        - synthetic.noise_mjy : float or tuple - Noise level in mJy
        - synthetic.rfi_power_min : float or tuple - Min RFI power in Jy
        - synthetic.rfi_power_max : float or tuple - Max RFI power in Jy
        - synthetic.rfi_types : list - RFI types to include
        - synthetic.rfi_type_counts : dict - Count per RFI type (int or [min, max])
        - synthetic.enable_bandpass_rolloff : bool - Enable bandpass
        - synthetic.bandpass_polynomial_order : int - Polynomial order (default 8)
        - synthetic.num_polarizations : int - Number of polarizations (default 1)
        - synthetic.polarization_correlation : float - Pol correlation (default 0.8)
        - synthetic.generation_batch_size : int - Samples per generation batch
        - synthetic.generation_workers : int - Parallel workers (default 1)
        - synthetic.generate_mad_masks : bool - Also generate MAD masks
        - processing.patch_size : int - Patch size (128, 256, 512, 1024)
        - processing.stretch : str or None - Stretching ('SQRT', 'LOG10', None)
        - processing.save_raw : bool - Save raw complex patches (default False)
        - processing.enable_augmentation : bool - Enable augmentation
        - processing.augmentation_rotations : int - Number of rotations (1, 2, 4)

    Attributes
    ----------
    config : DataConfig
        Stored configuration object.

    Examples
    --------
    >>> from samrfi.config import ConfigLoader
    >>> from samrfi.data_generation import SyntheticDataGenerator
    >>>
    >>> # Load configuration
    >>> config = ConfigLoader.load_data('configs/synthetic_gen.yaml')
    >>>
    >>> # Generate dataset
    >>> generator = SyntheticDataGenerator(config)
    >>> output_path = generator.generate('./datasets/synthetic_rfi')
    ==========================================
    Synthetic Data Generation with Physical Realism
    ==========================================
    ...
    ✓ Synthetic data generation complete!

    Notes
    -----
    The generator supports both preprocessed and raw output modes:

    - Preprocessed (save_raw=False): Generates RGB images with stretching,
      normalization, and 4-way augmentation. Ready for immediate training.

    - Raw (save_raw=True): Saves raw complex patches. Channel extraction
      and augmentation happen on-the-fly during training (GPU pipeline).

    Dynamic ranges can be randomized by specifying ranges instead of fixed values:
    - noise_mjy: [0.5, 1.5] - Random noise level per sample
    - rfi_power_min: [500, 1500] - Random minimum RFI power per sample
    - rfi_type_counts: {narrowband_persistent: [1, 5]} - Random count per sample
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize synthetic data generator.

        Parameters
        ----------
        config : DataConfig
            Configuration object with synthetic RFI parameters.
        """
        self.config = config

    def generate(self, output_path: str) -> str:
        """
        Generate synthetic dataset with exact ground truth masks.

        This method executes the complete synthetic data generation pipeline,
        including parallel generation (if workers > 1), batched processing,
        and saving in PyTorch format with comprehensive metadata.

        Parameters
        ----------
        output_path : str
            Directory path where generated dataset will be saved.
            Will be created if it doesn't exist.

        Returns
        -------
        str
            Absolute path to the saved dataset directory.

        Examples
        --------
        >>> generator = SyntheticDataGenerator(config)
        >>> dataset_path = generator.generate('./datasets/synthetic')
        ==========================================
        Synthetic Data Generation with Physical Realism
        ==========================================
        [1/5] Generating 100 synthetic samples...
        ...
        ✓ Synthetic data generation complete!

        Notes
        -----
        Output directory structure:
        - exact_masks/ : Dataset with exact ground truth
          - batch_*.pt : Batched PyTorch files
          - metadata.json : Batch metadata
        - mad_masks/ : Dataset with MAD-based masks (if generate_mad_masks=True)
        - generation_metadata.json : Generation parameters and statistics
        - rfi_parameters.json : Per-sample RFI parameters

        The generation process is memory-efficient, processing samples in
        batches and streaming to disk to avoid loading all samples in RAM.
        """
        print("=" * 60)
        print("Synthetic Data Generation with Physical Realism")
        print("=" * 60)

        # Extract config
        synth_config = self.config.synthetic
        proc_config = self.config.processing

        num_samples = synth_config.get("num_samples", 100)
        num_channels = synth_config.get("num_channels", 2048)
        num_times = synth_config.get("num_times", 512)

        # Physical scales (milli-Jansky)
        noise_level = synth_config.get("noise_mjy", 1.0)  # 1 mJy
        rfi_power_min = synth_config.get("rfi_power_min", 1000.0)  # 1000 Jy = 1e6 mJy
        rfi_power_max = synth_config.get("rfi_power_max", 10000.0)  # 10000 Jy

        print("\nPhysical Parameters:")
        print(f"  Noise level: {noise_level} mJy")
        print(f"  RFI power range: {rfi_power_min}-{rfi_power_max} Jy")

        # Compute dynamic range (handle both scalar and range values)
        is_range = isinstance(noise_level, (list | tuple)) or isinstance(
            rfi_power_max, (list | tuple)
        )
        if is_range:
            # Ranges provided - show min/max dynamic range
            noise_min = noise_level[0] if isinstance(noise_level, (list | tuple)) else noise_level
            noise_max = noise_level[1] if isinstance(noise_level, (list | tuple)) else noise_level
            rfi_min = (
                rfi_power_min[0] if isinstance(rfi_power_min, (list | tuple)) else rfi_power_min
            )
            rfi_max = (
                rfi_power_max[1] if isinstance(rfi_power_max, (list | tuple)) else rfi_power_max
            )
            dr_min = rfi_min * 1000 / noise_max  # Weakest RFI / highest noise
            dr_max = rfi_max * 1000 / noise_min  # Strongest RFI / lowest noise
            print(f"  Dynamic range: {dr_min:.1e} to {dr_max:.1e} (randomized per sample)")
        else:
            # Fixed values
            dr_value = rfi_power_max * 1000 / noise_level
            print(f"  Dynamic range: {dr_value:.1e}")

        print("\nConfiguration:")
        print(f"  Samples: {num_samples}")
        print(f"  Dimensions: {num_channels} channels × {num_times} times")
        print(f"  Output: {output_path}")

        # Get RFI configuration
        rfi_config = self._parse_rfi_config(synth_config)

        print("\nRFI Types Enabled:")
        for rfi_type, params in rfi_config.items():
            count = params["count"]
            # Handle both int and [min, max] list counts
            if isinstance(count, list | tuple):
                if count[1] > 0:  # Check max value
                    print(f"  {rfi_type}: {count[0]}-{count[1]} per sample (randomized)")
            elif count > 0:
                print(f"  {rfi_type}: {count} per sample")

        # Bandpass options
        enable_bandpass = synth_config.get("enable_bandpass_rolloff", False)
        if enable_bandpass:
            bandpass_order = synth_config.get("bandpass_polynomial_order", 8)
            print(f"\nBandpass: Enabled ({bandpass_order}th order polynomial rolloff)")

        # Polarization configuration
        num_polarizations = synth_config.get("num_polarizations", 1)
        pol_corr = synth_config.get("polarization_correlation", 0.8)
        print(f"Number of polarizations: {num_polarizations}")
        print(f"Polarization correlation: {pol_corr}")

        # Check augmentation settings for accurate reporting
        augmentation_rotations = proc_config.get("augmentation_rotations", 4)
        enable_augmentation = proc_config.get("enable_augmentation", True)
        effective_rotations = augmentation_rotations if enable_augmentation else 1
        expected_patches = num_samples * effective_rotations

        # Generate samples in batches to avoid memory exhaustion
        print(f"\n[1/5] Generating {num_samples} synthetic samples...")
        print(f"  With {effective_rotations}x augmentation → {expected_patches} patches expected")

        batch_size = synth_config.get("generation_batch_size", 50)  # Samples per generation batch
        num_batches = (num_samples + batch_size - 1) // batch_size
        print(f"  Generation batch size: {batch_size} raw samples/batch ({num_batches} batches)")

        # Check for parallel generation
        generation_workers = synth_config.get("generation_workers", 1)
        if generation_workers > 1:
            print(
                f"  Using {generation_workers} parallel workers (each does generation + augmentation)"
            )

        # Initialize BatchWriters for streaming to disk
        from samrfi.data.torch_dataset import BatchWriter

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if MAD generation is enabled
        generate_mad = synth_config.get("generate_mad_masks", False)

        exact_writer = BatchWriter(output_dir / "exact_masks", samples_per_batch=100)
        mad_writer = (
            BatchWriter(output_dir / "mad_masks", samples_per_batch=100) if generate_mad else None
        )

        # Prepare generation kwargs for workers
        gen_kwargs = {
            "num_channels": num_channels,
            "num_times": num_times,
            "noise_level": noise_level,
            "rfi_power_min": rfi_power_min,
            "rfi_power_max": rfi_power_max,
            "rfi_config": rfi_config,
            "enable_bandpass": enable_bandpass,
            "bandpass_order": synth_config.get("bandpass_polynomial_order", 8),
            "num_polarizations": num_polarizations,
            "pol_corr": pol_corr,
            "synth_config": synth_config,
        }

        all_rfi_parameters = []
        total_raw_samples = 0
        total_patches_written = 0

        # Create Pool ONCE outside loop (reuse for all batches)
        pool = None
        if generation_workers > 1:
            from functools import partial
            from multiprocessing import Pool

            # Convert config to dict for pickling
            def namespace_to_dict(obj):
                if hasattr(obj, "__dict__"):
                    return {k: namespace_to_dict(v) for k, v in obj.__dict__.items()}
                elif isinstance(obj, dict):
                    return {k: namespace_to_dict(v) for k, v in obj.items()}
                return obj

            config_dict = namespace_to_dict(self.config)

            # Create worker function with fixed kwargs
            worker_func = partial(_worker_generate_and_preprocess, **gen_kwargs)

            # Initialize pool ONCE (reuse across all batches)
            pool = Pool(generation_workers, initializer=_init_worker, initargs=(config_dict,))

        try:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                batch_samples = end_idx - start_idx

                expected_patches_batch = batch_samples * effective_rotations
                print(
                    f"\n  Batch {batch_idx + 1}/{num_batches}: {batch_samples} raw samples → {expected_patches_batch} patches expected"
                )

                if pool is not None:
                    # Parallel: submit tasks to existing pool
                    async_results = [pool.apply_async(worker_func) for _ in range(batch_samples)]

                    results = []
                    for ar in tqdm(async_results, total=batch_samples, desc="    Generating"):
                        results.append(ar.get())

                    # Collect all patches from all workers
                    for dataset, rfi_params in results:
                        exact_writer.add_batch(dataset)
                        all_rfi_parameters.append(rfi_params)
                        total_patches_written += len(dataset)

                else:
                    # Sequential: generate + preprocess one at a time
                    save_raw = proc_config.get("save_raw", False)

                    for _ in tqdm(range(batch_samples), desc="    Generating"):
                        waterfall, exact_mask, rfi_params = self._generate_single_sample(
                            **gen_kwargs
                        )

                        if save_raw:
                            # Save raw complex patches
                            # Take magnitude of complex values, then average across polarizations
                            waterfall_squeezed = waterfall.squeeze(0)  # Remove baseline dimension
                            magnitude = np.abs(waterfall_squeezed)  # (num_pols, channels, times)
                            averaged = magnitude.mean(axis=0).astype(
                                np.float32
                            )  # (channels, times)
                            mask_averaged = exact_mask.squeeze(0).max(axis=0).astype(np.uint8)

                            complex_patches = torch.from_numpy(averaged).unsqueeze(0)
                            masks = torch.from_numpy(mask_averaged).unsqueeze(0)

                            dataset = RawPatchDataset(complex_patches, masks)
                        else:
                            # Preprocess (original behavior)
                            preprocessor = Preprocessor(waterfall, flags=exact_mask)
                            dataset = preprocessor.create_dataset(
                                patch_size=proc_config.get("patch_size", 128),
                                stretch=proc_config.get("stretch", None),
                                flag_sigma=proc_config.get("flag_sigma", 5),
                                use_custom_flags=True,
                                num_patches=proc_config.get("num_patches", None),
                                normalize_before_stretch=proc_config.get(
                                    "normalize_before_stretch", True
                                ),
                                normalize_after_stretch=proc_config.get(
                                    "normalize_after_stretch", False
                                ),
                                num_workers=0,
                                enable_augmentation=proc_config.get("enable_augmentation", True),
                                augmentation_rotations=proc_config.get("augmentation_rotations", 4),
                            )

                        exact_writer.add_batch(dataset)
                        all_rfi_parameters.append(rfi_params)
                        total_patches_written += len(dataset)

                # Flush all accumulated patches to disk
                exact_writer._flush()
                total_raw_samples += batch_samples

                print(
                    f"    Wrote {total_patches_written} patches so far ({total_raw_samples}/{num_samples} raw samples processed)"
                )
        finally:
            # Clean up pool
            if pool is not None:
                pool.close()
                pool.join()

        # Finalize batch writing (flush remaining samples + write metadata)
        print("\n[2/5] Finalizing batch files...")
        exact_writer.finalize()
        if generate_mad:
            mad_writer.finalize()

        # Update metadata to indicate format (raw vs preprocessed)
        save_raw = proc_config.get("save_raw", False)
        metadata_path = output_dir / "exact_masks" / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                batch_metadata = json.load(f)
            batch_metadata["format"] = "raw" if save_raw else "preprocessed"
            with open(metadata_path, "w") as f:
                json.dump(batch_metadata, f, indent=2)

        # Summary
        print("\n[3/5] Generation summary:")
        print(f"  Raw samples generated: {total_raw_samples}")
        print(f"  Augmentation: {effective_rotations}x rotations")
        print(f"  Total patches written: {total_patches_written}")
        print(
            f"  Patch size: {proc_config.get('patch_size', 128)}×{proc_config.get('patch_size', 128)}"
        )
        print(f"  Stretch: {proc_config.get('stretch', None) or 'None'}")

        # Save generation metadata (separate from batch metadata)
        print("\n[4/5] Saving generation metadata...")
        # Compute dynamic range for metadata (use previously computed values)
        if is_range:
            dynamic_range_str = f"{dr_min:.1e} to {dr_max:.1e}"
        else:
            dynamic_range_str = f"{dr_value:.1e}"

        metadata = {
            "source": "synthetic",
            "physical_parameters": {
                "noise_mjy": noise_level,
                "rfi_power_min_jy": rfi_power_min,
                "rfi_power_max_jy": rfi_power_max,
                "dynamic_range": dynamic_range_str,
            },
            "num_raw_samples": total_raw_samples,
            "num_channels": num_channels,
            "num_times": num_times,
            "rfi_config": {
                k: v
                for k, v in rfi_config.items()
                if (v["count"][1] if isinstance(v["count"], list | tuple) else v["count"]) > 0
            },
            "bandpass": {
                "enabled": enable_bandpass,
                "polynomial_order": (
                    synth_config.get("bandpass_polynomial_order", 8) if enable_bandpass else None
                ),
            },
            "polarization_correlation": pol_corr,
            "augmentation": {
                "enabled": enable_augmentation,
                "rotations": effective_rotations,
            },
            "num_patches": total_patches_written,
            "patch_size": proc_config.get("patch_size", 128),
            "stretch": proc_config.get("stretch", None),
            "ground_truth": "exact",  # Not MAD-based!
            "batch_processing": {
                "generation_batch_size": batch_size,
                "num_batches": num_batches,
            },
        }

        metadata_path = output_dir / "generation_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save RFI parameters
        rfi_params_path = output_dir / "rfi_parameters.json"
        with open(rfi_params_path, "w") as f:
            json.dump(all_rfi_parameters, f, indent=2)

        print(
            f"  Exact masks dataset: {output_dir / 'exact_masks'} ({exact_writer.batch_file_idx} batch files)"
        )
        if generate_mad:
            print(
                f"  MAD masks dataset: {output_dir / 'mad_masks'} ({mad_writer.batch_file_idx} batch files)"
            )
        else:
            print("  MAD masks: Skipped (generate_mad_masks=False)")
        print(f"  Generation metadata: {metadata_path}")
        print(f"  RFI parameters: {rfi_params_path}")

        # Statistics
        print("\n[5/5] Dataset Statistics:")
        print(f"  Total waterfall samples: {num_samples}")
        print(f"  Total patches: {total_patches_written}")
        # print(f"  RFI coverage: {rfi_fraction:.2f}%")
        print(
            f"  Image shape: {proc_config.get('patch_size', 128)}×{proc_config.get('patch_size', 128)}×3 (RGB)"
        )
        print(
            f"  Mask shape: {proc_config.get('patch_size', 128)}×{proc_config.get('patch_size', 128)} (exact binary)"
        )
        print("  Format: Batched torch files (uncompressed .pt)")

        print("\n✓ Generation Complete!")
        if generate_mad:
            print("  ✓ TWO datasets generated: exact masks + MAD masks")
            print("  ✓ MAD masks for flagger comparison")
        else:
            print("  ✓ Dataset generated: exact masks only")
        print("  ✓ Exact ground truth for training")
        print("  ✓ Physical noise/RFI scales (~10^6 dynamic range)")
        print("  ✓ Batched format for low-memory training")
        print("  ✓ Realistic RFI types (sweeps, bursts, persistent)")
        print("  ✓ Frequency sweeps: linear & quadratic")
        if enable_bandpass:
            print(
                f"  ✓ Bandpass rolloff ({synth_config.get('bandpass_polynomial_order', 8)}th order)"
            )
        if pol_corr > 0:
            print(f"  ✓ Polarization correlation ({pol_corr})")

        print("\n" + "=" * 60)
        print("✓ Synthetic data generation complete!")
        print("=" * 60)

        return str(output_dir)

    def _generate_single_sample(
        self,
        num_channels: int,
        num_times: int,
        noise_level: float,
        rfi_power_min: float,
        rfi_power_max: float,
        rfi_config: Dict[str, Any],
        enable_bandpass: bool,
        bandpass_order: int,
        num_polarizations: int,
        pol_corr: float,
        synth_config: Any,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Generate a single synthetic waterfall sample with exact ground truth mask.

        This method creates one synthetic observation including Gaussian noise,
        optional bandpass rolloff, multiple RFI signals of various types,
        and correlated polarizations with complex visibilities.

        Parameters
        ----------
        num_channels : int
            Number of frequency channels.
        num_times : int
            Number of time samples.
        noise_level : float
            Noise level in mJy (milli-Jansky).
        rfi_power_min : float
            Minimum RFI power in Jy (Jansky).
        rfi_power_max : float
            Maximum RFI power in Jy (Jansky).
        rfi_config : dict
            RFI type configuration with counts per type.
        enable_bandpass : bool
            Whether to apply bandpass rolloff.
        bandpass_order : int
            Polynomial order for bandpass rolloff (e.g., 8).
        num_polarizations : int
            Number of polarizations to generate.
        pol_corr : float
            Polarization correlation coefficient (0.0 to 1.0).
        synth_config : object
            Full synthetic configuration object.

        Returns
        -------
        waterfall : np.ndarray
            Complex visibility waterfall with shape (1, num_polarizations, channels, times).
            Dtype is complex128 (complex real+imaginary components).
        exact_mask : np.ndarray
            Binary RFI mask with shape (1, num_polarizations, channels, times).
            Dtype is bool (True=RFI, False=clean).
        rfi_params : list of dict
            List of RFI parameter dictionaries for each RFI signal added.
            Each dict contains type, amplitude, and type-specific parameters.

        Notes
        -----
        The generation process:
        1. Creates Gaussian noise baseline (~1 mJy)
        2. Applies optional bandpass rolloff (polynomial edge attenuation)
        3. Adds RFI signals (6 orders of magnitude stronger: ~1000-10000 Jy)
        4. Creates polarizations with correlation
        5. Adds random phase to create complex visibilities

        Polarization handling:
        - Pol 0: Full RFI + noise
        - Pol 1: Correlated RFI (pol_corr fraction) + noise
        - Pol 2+: Noise only (no RFI)
        """
        # Sample noise level if range provided
        if isinstance(noise_level, (list | tuple)):
            noise_level = np.random.uniform(noise_level[0], noise_level[1])

        # Sample RFI power ranges if provided
        if isinstance(rfi_power_min, (list | tuple)):
            rfi_power_min = np.random.uniform(rfi_power_min[0], rfi_power_min[1])
        if isinstance(rfi_power_max, (list | tuple)):
            rfi_power_max = np.random.uniform(rfi_power_max[0], rfi_power_max[1])

        # Create base spectrograph (clean Gaussian noise at mJy scale)
        baseline = np.random.normal(noise_level, noise_level * 0.1, (num_channels, num_times))

        # Apply bandpass rolloff if enabled
        if enable_bandpass:
            bandpass = self._generate_bandpass(num_channels, bandpass_order)
            baseline = baseline * bandpass[:, np.newaxis]

        # Initialize RFI mask (exact locations)
        rfi_mask = np.zeros((num_channels, num_times), dtype=bool)
        rfi_signal = np.zeros((num_channels, num_times))

        rfi_params = []

        # Add each RFI type
        for rfi_type, params in rfi_config.items():
            count = params["count"]

            # Support random counts: if count is [min, max], sample randomly
            if isinstance(count, list | tuple) and len(count) == 2:
                count = np.random.randint(count[0], count[1] + 1)

            if count == 0:
                continue

            for _ in range(count):
                rfi_amplitude = np.random.uniform(rfi_power_min, rfi_power_max) * 1000  # Jy to mJy

                if rfi_type == "narrowband_persistent":
                    signal, mask, param = self._add_narrowband_persistent(
                        num_channels, num_times, rfi_amplitude, synth_config
                    )

                elif rfi_type == "broadband_persistent":
                    signal, mask, param = self._add_broadband_persistent(
                        num_channels, num_times, rfi_amplitude, synth_config
                    )

                elif rfi_type == "narrowband_intermittent":
                    signal, mask, param = self._add_narrowband_intermittent(
                        num_channels, num_times, rfi_amplitude, synth_config
                    )

                elif rfi_type == "narrowband_bursty":
                    signal, mask, param = self._add_narrowband_bursty(
                        num_channels, num_times, rfi_amplitude, synth_config
                    )

                elif rfi_type == "broadband_bursty":
                    signal, mask, param = self._add_broadband_bursty(
                        num_channels, num_times, rfi_amplitude, synth_config
                    )

                elif rfi_type == "frequency_sweep":
                    signal, mask, param = self._add_frequency_sweep(
                        num_channels, num_times, rfi_amplitude, synth_config
                    )

                else:
                    continue

                rfi_signal += signal
                rfi_mask = rfi_mask | mask
                rfi_params.append(
                    {**param, "type": rfi_type, "amplitude_mjy": float(rfi_amplitude)}
                )

        # Combine clean + RFI
        combined = baseline + rfi_signal

        # Create polarizations with correlation (COMPLEX for phase extraction)
        pols = []
        masks = []

        for pol_idx in range(num_polarizations):
            if pol_idx == 0:
                # Pol 1: Full RFI + noise
                pol_real = combined.copy()
                mask = rfi_mask.copy()
            elif pol_idx == 1:
                # Pol 2: Correlated RFI + noise
                pol_real = (
                    pol_corr * rfi_signal
                    + (1 - pol_corr) * np.random.normal(0, noise_level * 0.1, rfi_signal.shape)
                    + baseline
                )
                mask = rfi_mask.copy()
            else:
                # Pol 3+: Noise only (no RFI)
                pol_real = np.random.normal(
                    noise_level, noise_level * 0.1, (num_channels, num_times)
                )
                mask = np.zeros_like(rfi_mask)

            # Add random phase for complex visibilities
            pol_phase = np.random.uniform(0, 2 * np.pi, pol_real.shape)
            pol = pol_real * np.exp(1j * pol_phase)

            pols.append(pol)
            masks.append(mask)

        waterfall = np.stack(pols, axis=0)[np.newaxis, ...]  # (1, num_pols, channels, times)
        exact_mask = np.stack(masks, axis=0)[np.newaxis, ...]  # (1, num_pols, channels, times)

        return waterfall, exact_mask, rfi_params

    def _generate_bandpass(self, num_channels: int, order: int) -> np.ndarray:
        """
        Generate realistic bandpass response with polynomial rolloff at edges.

        Simulates realistic radio telescope bandpass with reduced sensitivity
        at band edges due to filter characteristics.

        Parameters
        ----------
        num_channels : int
            Number of frequency channels.
        order : int
            Polynomial order for rolloff curve (higher = sharper transition).
            Typical value: 8.

        Returns
        -------
        bandpass : np.ndarray
            Bandpass response array with shape (num_channels,).
            Values range from 0.0 (fully attenuated) to 1.0 (full sensitivity).

        Notes
        -----
        Applies polynomial rolloff to 10% of channels at each band edge.
        Central 80% of band has full sensitivity (response = 1.0).
        """
        bandpass = np.ones(num_channels)
        edge_fraction = 0.1  # Rolloff in 10% of channels at each edge
        edge_channels = int(num_channels * edge_fraction)

        # Polynomial rolloff at edges
        for i in range(edge_channels):
            # Low frequency edge
            t = i / edge_channels
            bandpass[i] = t**order

            # High frequency edge
            bandpass[-(i + 1)] = t**order

        return bandpass

    def _add_narrowband_persistent(
        self, nc: int, nt: int, amp: float, config: Any
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """
        Add persistent narrowband RFI signal.

        Simulates continuous narrowband interference sources like GPS satellites,
        cell towers, or broadcast transmitters that occupy a few channels
        continuously across all time samples.

        Parameters
        ----------
        nc : int
            Number of frequency channels.
        nt : int
            Number of time samples.
        amp : float
            RFI amplitude in mJy (milli-Jansky).
        config : object
            Configuration object (currently unused, for future extensibility).

        Returns
        -------
        signal : np.ndarray
            RFI signal array with shape (nc, nt).
        mask : np.ndarray
            Binary mask with shape (nc, nt), dtype bool.
        params : dict
            Parameters: center_freq (int), bandwidth (int).

        Notes
        -----
        - Center frequency: Random location in central 80% of band
        - Bandwidth: Random 1-10 channels
        - Persistence: Constant amplitude across all time samples
        """
        center_freq = np.random.randint(int(nc * 0.1), int(nc * 0.9))
        bandwidth = np.random.randint(1, 10)

        signal = np.zeros((nc, nt))
        mask = np.zeros((nc, nt), dtype=bool)

        # Ensure at least 1 channel is selected (bandwidth // 2 can be 0 for bandwidth=1)
        freq_slice = slice(
            max(0, center_freq - bandwidth // 2),
            min(nc, center_freq + (bandwidth // 2) + 1),
        )
        signal[freq_slice, :] = amp
        mask[freq_slice, :] = True

        params = {"center_freq": int(center_freq), "bandwidth": int(bandwidth)}
        return signal, mask, params

    def _add_broadband_persistent(
        self, nc: int, nt: int, amp: float, config: Any
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """
        Add persistent broadband RFI signal.

        Simulates broadband interference sources like power lines or continuous
        broadband emitters that affect all channels during a time window.

        Parameters
        ----------
        nc : int
            Number of frequency channels.
        nt : int
            Number of time samples.
        amp : float
            RFI amplitude in mJy (milli-Jansky).
        config : object
            Configuration object (currently unused).

        Returns
        -------
        signal : np.ndarray
            RFI signal array with shape (nc, nt).
        mask : np.ndarray
            Binary mask with shape (nc, nt), dtype bool.
        params : dict
            Parameters: center_time (int), time_width (int).

        Notes
        -----
        - Center time: Random location in central 80% of observation
        - Time width: Random 5-50 time samples
        - Broadband: Affects all frequency channels simultaneously
        """
        center_time = np.random.randint(int(nt * 0.1), int(nt * 0.9))
        time_width = np.random.randint(5, 50)

        signal = np.zeros((nc, nt))
        mask = np.zeros((nc, nt), dtype=bool)

        time_slice = slice(
            max(0, center_time - time_width // 2), min(nt, center_time + time_width // 2)
        )
        signal[:, time_slice] = amp
        mask[:, time_slice] = True

        params = {"center_time": int(center_time), "time_width": int(time_width)}
        return signal, mask, params

    def _add_narrowband_intermittent(
        self, nc: int, nt: int, amp: float, config: Any
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Add intermittent periodic narrowband RFI signal.

        Simulates periodic interference sources like rotating radar systems that
        emit narrowband signals at regular intervals with a duty cycle.

        Parameters
        ----------
        nc : int
            Number of frequency channels.
        nt : int
            Number of time samples.
        amp : float
            RFI amplitude in mJy (milli-Jansky).
        config : object
            Configuration object (currently unused).

        Returns
        -------
        signal : np.ndarray
            RFI signal array with shape (nc, nt).
        mask : np.ndarray
            Binary mask with shape (nc, nt), dtype bool.
        params : dict
            Parameters: center_freq (int), bandwidth (int), period (int), duty_cycle (float).

        Notes
        -----
        - Center frequency: Random location in central 80% of band
        - Bandwidth: Random 2-15 channels
        - Period: Random 20-200 time samples
        - Duty cycle: Random 0.1-0.5 (10-50% active time)
        """
        center_freq = np.random.randint(int(nc * 0.1), int(nc * 0.9))
        bandwidth = np.random.randint(2, 15)
        period = np.random.randint(20, 200)
        duty_cycle = np.random.uniform(0.1, 0.5)

        signal = np.zeros((nc, nt))
        mask = np.zeros((nc, nt), dtype=bool)

        freq_slice = slice(
            max(0, center_freq - bandwidth // 2), min(nc, center_freq + bandwidth // 2)
        )

        for t in range(0, nt, period):
            duration = int(period * duty_cycle)
            time_slice = slice(t, min(nt, t + duration))
            signal[freq_slice, time_slice] = amp
            mask[freq_slice, time_slice] = True

        params = {
            "center_freq": int(center_freq),
            "bandwidth": int(bandwidth),
            "period": int(period),
            "duty_cycle": float(duty_cycle),
        }
        return signal, mask, params

    def _add_narrowband_bursty(
        self, nc: int, nt: int, amp: float, config: Any
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """
        Add random bursty narrowband RFI signal.

        Simulates random pulsed narrowband emitters like intermittent transmitters
        with irregular burst patterns.

        Parameters
        ----------
        nc : int
            Number of frequency channels.
        nt : int
            Number of time samples.
        amp : float
            RFI amplitude in mJy (milli-Jansky).
        config : object
            Configuration object (currently unused).

        Returns
        -------
        signal : np.ndarray
            RFI signal array with shape (nc, nt).
        mask : np.ndarray
            Binary mask with shape (nc, nt), dtype bool.
        params : dict
            Parameters: center_freq (int), bandwidth (int), num_bursts (int).

        Notes
        -----
        - Center frequency: Random location in central 80% of band
        - Bandwidth: Random 2-20 channels
        - Number of bursts: Random 3-15 bursts
        - Burst widths: Random 2-20 time samples per burst
        - Burst times: Randomly distributed (no fixed period)
        """
        center_freq = np.random.randint(int(nc * 0.1), int(nc * 0.9))
        bandwidth = np.random.randint(2, 20)
        num_bursts = np.random.randint(3, 15)

        signal = np.zeros((nc, nt))
        mask = np.zeros((nc, nt), dtype=bool)

        freq_slice = slice(
            max(0, center_freq - bandwidth // 2), min(nc, center_freq + bandwidth // 2)
        )

        burst_times = np.random.choice(nt, num_bursts, replace=False)
        burst_widths = np.random.randint(2, 20, num_bursts)

        for t, width in zip(burst_times, burst_widths, strict=False):
            time_slice = slice(max(0, t - width // 2), min(nt, t + width // 2))
            signal[freq_slice, time_slice] = amp
            mask[freq_slice, time_slice] = True

        params = {
            "center_freq": int(center_freq),
            "bandwidth": int(bandwidth),
            "num_bursts": int(num_bursts),
        }
        return signal, mask, params

    def _add_broadband_bursty(
        self, nc: int, nt: int, amp: float, config: Any
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """
        Add random bursty broadband RFI signal.

        Simulates random broadband bursts like lightning strikes or other
        impulsive broadband interference affecting all channels.

        Parameters
        ----------
        nc : int
            Number of frequency channels.
        nt : int
            Number of time samples.
        amp : float
            RFI amplitude in mJy (milli-Jansky).
        config : object
            Configuration object (currently unused).

        Returns
        -------
        signal : np.ndarray
            RFI signal array with shape (nc, nt).
        mask : np.ndarray
            Binary mask with shape (nc, nt), dtype bool.
        params : dict
            Parameters: num_bursts (int).

        Notes
        -----
        - Number of bursts: Random 2-10 bursts
        - Burst widths: Random 1-5 time samples (very brief)
        - Burst times: Randomly distributed
        - Broadband: Affects all frequency channels
        """
        num_bursts = np.random.randint(2, 10)

        signal = np.zeros((nc, nt))
        mask = np.zeros((nc, nt), dtype=bool)

        burst_times = np.random.choice(nt, num_bursts, replace=False)
        burst_widths = np.random.randint(1, 5, num_bursts)

        for t, width in zip(burst_times, burst_widths, strict=False):
            time_slice = slice(max(0, t - width // 2), min(nt, t + width // 2))
            signal[:, time_slice] = amp
            mask[:, time_slice] = True

        params = {"num_bursts": int(num_bursts)}
        return signal, mask, params

    def _add_frequency_sweep(
        self, nc: int, nt: int, amp: float, config: Any
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """
        Add frequency sweep RFI signal.

        Simulates narrowband RFI that sweeps across frequency channels over time,
        like radar chirps or drifting satellite signals.

        Parameters
        ----------
        nc : int
            Number of frequency channels.
        nt : int
            Number of time samples.
        amp : float
            RFI amplitude in mJy (milli-Jansky).
        config : object
            Configuration object (currently unused).

        Returns
        -------
        signal : np.ndarray
            RFI signal array with shape (nc, nt).
        mask : np.ndarray
            Binary mask with shape (nc, nt), dtype bool.
        params : dict
            Parameters: start_freq (int), end_freq (int), bandwidth (int), sweep_order (int).

        Notes
        -----
        - Start frequency: Random location in lower half of band
        - End frequency: Random location in upper half of band
        - Bandwidth: Random 2-10 channels
        - Sweep order: 1 (linear) or 2 (quadratic/accelerating)
        - Creates diagonal patterns in time-frequency space
        """
        start_freq = np.random.randint(int(nc * 0.1), int(nc * 0.5))
        end_freq = np.random.randint(int(nc * 0.5), int(nc * 0.9))
        bandwidth = np.random.randint(2, 10)
        sweep_order = np.random.choice([1, 2])  # Linear or quadratic sweep

        signal = np.zeros((nc, nt))
        mask = np.zeros((nc, nt), dtype=bool)

        for t in range(nt):
            if sweep_order == 1:
                # Linear sweep
                progress = t / nt
            else:
                # Quadratic sweep (accelerating)
                progress = (t / nt) ** 2

            center = int(start_freq + (end_freq - start_freq) * progress)
            freq_slice = slice(max(0, center - bandwidth // 2), min(nc, center + bandwidth // 2))

            signal[freq_slice, t] = amp
            mask[freq_slice, t] = True

        params = {
            "start_freq": int(start_freq),
            "end_freq": int(end_freq),
            "bandwidth": int(bandwidth),
            "sweep_order": int(sweep_order),
        }
        return signal, mask, params

    def _parse_rfi_config(self, config: Any) -> Dict[str, Dict[str, Any]]:
        """
        Parse RFI configuration from configuration object.

        Extracts RFI types and counts from configuration, applying defaults
        for any unspecified types.

        Parameters
        ----------
        config : object
            Configuration object with optional attributes:
            - rfi_types : list - RFI types to include
            - rfi_type_counts : dict - Counts per RFI type (int or [min, max])

        Returns
        -------
        rfi_config : dict
            Dictionary mapping RFI type names to configuration dicts.
            Each config dict contains:
            - count : int or list - Number of instances (or [min, max] range)

        Examples
        --------
        >>> config = SimpleNamespace(
        ...     rfi_types=['narrowband_persistent', 'frequency_sweep'],
        ...     rfi_type_counts={'narrowband_persistent': [1, 3], 'frequency_sweep': 2}
        ... )
        >>> generator._parse_rfi_config(config)
        {'narrowband_persistent': {'count': [1, 3]}, 'frequency_sweep': {'count': 2}, ...}

        Notes
        -----
        Default RFI types if not specified:
        - narrowband_persistent: 1
        - broadband_persistent: 1
        - narrowband_bursty: 1
        - frequency_sweep: 1
        - narrowband_intermittent: 0 (disabled)
        - broadband_bursty: 0 (disabled)
        """
        rfi_types = config.get(
            "rfi_types", ["narrowband_persistent", "broadband_persistent", "frequency_sweep"]
        )

        default_counts = config.get("rfi_type_counts", {})

        rfi_config = {
            "narrowband_persistent": {"count": default_counts.get("narrowband_persistent", 1)},
            "broadband_persistent": {"count": default_counts.get("broadband_persistent", 1)},
            "narrowband_intermittent": {"count": default_counts.get("narrowband_intermittent", 0)},
            "narrowband_bursty": {"count": default_counts.get("narrowband_bursty", 1)},
            "broadband_bursty": {"count": default_counts.get("broadband_bursty", 0)},
            "frequency_sweep": {"count": default_counts.get("frequency_sweep", 1)},
        }

        # Override with rfi_types if provided
        if rfi_types:
            for rfi_type in rfi_config.keys():
                if rfi_type not in rfi_types and rfi_type not in default_counts:
                    rfi_config[rfi_type]["count"] = 0

        return rfi_config
